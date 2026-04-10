#!/usr/bin/env python3
"""
MapFormer: Full experiment pipeline.

Trains MapFormer-WM, MapFormer-EM, Transformer+RoPE, and LSTM on a
2D grid navigation task, then generates the key figures from
Rambaud et al. (2025):

1. Training curves (all models)
2. Length generalisation (train on T=64, test on 128, 256, 512, 1024)
3. Position state PCA (MapFormer only)
4. Grid cell autocorrelation (MapFormer only)

Usage:
    python3 -m mapformer.main [--epochs 50] [--device cpu] [--grid-size 10]
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent dir to path so we can run as `python3 -m mapformer.main`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.baselines import TransformerRoPE, LSTMBaseline
from mapformer.train import train
from mapformer.evaluate import (
    eval_length_generalisation,
    extract_position_states,
    compute_rate_map,
    plot_length_generalisation,
    plot_position_pca,
    plot_grid_cell_autocorrelation,
    plot_training_curves,
)
from mapformer.lie_groups import is_special_orthogonal
from mapformer.prefix_scan import parallel_prefix_product, sequential_prefix_product


def run_sanity_checks():
    """Mathematical sanity checks from Section 7.1 of the guide."""
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    from mapformer.lie_groups import (
        skew_symmetric_2d, exp_map_2d, build_block_diagonal_rotations_fast,
    )

    # 1. exp(A)^T @ exp(A) = I
    theta = torch.randn(16)
    R = exp_map_2d(theta)
    eye = torch.eye(2).expand(16, 2, 2)
    ortho_ok = torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-6)
    print(f"  [{'PASS' if ortho_ok else 'FAIL'}] exp(A)^T @ exp(A) = I (orthogonality)")

    # 2. det(exp(A)) = +1
    dets = torch.det(R)
    det_ok = torch.allclose(dets, torch.ones(16), atol=1e-6)
    print(f"  [{'PASS' if det_ok else 'FAIL'}] det(exp(A)) = 1 (special orthogonal)")

    # 3. Parallel prefix product matches sequential
    M_test = exp_map_2d(torch.randn(2 * 8)).reshape(2, 8, 2, 2)
    par = parallel_prefix_product(M_test)
    seq = sequential_prefix_product(M_test)
    scan_ok = torch.allclose(par, seq, atol=1e-5)
    print(f"  [{'PASS' if scan_ok else 'FAIL'}] Parallel prefix product matches sequential")

    # 4. Block-diagonal stays on SO(n)
    delta = torch.randn(4, 16, 3)  # 3 rotation blocks -> 6x6 matrices
    M_block = build_block_diagonal_rotations_fast(delta)
    P_block = parallel_prefix_product(M_block)
    # Check last position in each batch
    so_ok = is_special_orthogonal(P_block[:, -1], atol=1e-4)
    print(f"  [{'PASS' if so_ok else 'FAIL'}] Position state stays on SO(n) manifold")

    print()


def build_models(d_model, n_heads, n_rot_dims, action_vocab, obs_vocab, n_layers):
    """Instantiate all models."""
    models = {
        "MapFormer-WM": MapFormerWM(
            d_model=d_model, n_heads=n_heads, n_rot_dims=n_rot_dims,
            action_vocab=action_vocab, obs_vocab=obs_vocab, n_layers=n_layers,
        ),
        "MapFormer-EM": MapFormerEM(
            d_model=d_model, n_heads=n_heads, n_rot_dims=n_rot_dims,
            action_vocab=action_vocab, obs_vocab=obs_vocab, n_layers=n_layers,
        ),
        "Transformer+RoPE": TransformerRoPE(
            d_model=d_model, n_heads=n_heads,
            action_vocab=action_vocab, obs_vocab=obs_vocab, n_layers=n_layers,
        ),
        "LSTM": LSTMBaseline(
            d_model=d_model, action_vocab=action_vocab,
            obs_vocab=obs_vocab, n_layers=n_layers,
        ),
    }
    return models


def main():
    parser = argparse.ArgumentParser(description="MapFormer experiment pipeline")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu, cuda, mps)")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid world size")
    parser.add_argument("--n-obs-types", type=int, default=4,
                        help="Number of observation types (< grid_size^2)")
    parser.add_argument("--d-model", type=int, default=32, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Training sequence length")
    parser.add_argument("--n-batches", type=int, default=100, help="Batches per epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory for output figures")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Only train MapFormer models")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")
    print(f"Grid: {args.grid_size}x{args.grid_size}, {args.n_obs_types} obs types")
    print(f"Model: d={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    print(f"Training: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}, "
          f"seq_len={args.seq_len}")
    print()

    # Sanity checks
    run_sanity_checks()

    # Environment
    env = GridWorld(size=args.grid_size, n_obs_types=args.n_obs_types, seed=args.seed)

    # n_rot_dims must equal d_head // 2 for WM (RoPE-style rotation)
    d_head = args.d_model // args.n_heads
    n_rot_dims = d_head // 2

    # Build models
    action_vocab = GridWorld.N_ACTIONS
    obs_vocab = args.n_obs_types

    all_models = build_models(
        args.d_model, args.n_heads, n_rot_dims,
        action_vocab, obs_vocab, args.n_layers,
    )

    if args.skip_baselines:
        all_models = {k: v for k, v in all_models.items()
                      if k.startswith("MapFormer")}

    # ============================================================
    # Train all models
    # ============================================================
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    all_losses = {}
    trained_models = {}

    for name, model in all_models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n--- {name} ({n_params:,} params) ---")

        losses = train(
            model, env,
            n_epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, seq_len=args.seq_len,
            n_batches=args.n_batches, device=device,
        )

        all_losses[name] = losses
        trained_models[name] = model

    # ============================================================
    # Figure 1: Training curves
    # ============================================================
    print("\n" + "=" * 60)
    print("FIGURE 1: Training Curves")
    print("=" * 60)
    plot_training_curves(all_losses, save_path=str(output_dir / "fig1_training_curves.png"))

    # ============================================================
    # Figure 2: Length generalisation
    # ============================================================
    print("\n" + "=" * 60)
    print("FIGURE 2: Length Generalisation")
    print("=" * 60)

    test_lens = [args.seq_len, 2*args.seq_len, 4*args.seq_len,
                 8*args.seq_len, 16*args.seq_len]

    all_gen_results = {}
    for name, model in trained_models.items():
        print(f"\n  Evaluating {name}...")
        results = eval_length_generalisation(
            model, env, train_len=args.seq_len,
            test_lens=test_lens, n_trials=100, device=device,
        )
        all_gen_results[name] = results
        for T, acc in sorted(results.items()):
            print(f"    T={T:4d}: {acc:.3f}")

    plot_length_generalisation(
        all_gen_results,
        save_path=str(output_dir / "fig2_length_generalisation.png"),
    )

    # ============================================================
    # Figure 3: Position state PCA (MapFormer-WM only)
    # ============================================================
    print("\n" + "=" * 60)
    print("FIGURE 3: Position State PCA")
    print("=" * 60)

    if "MapFormer-WM" in trained_models:
        try:
            positions, states = extract_position_states(
                trained_models["MapFormer-WM"], env,
                n_samples=500, traj_len=32, device=device,
            )
            plot_position_pca(
                positions, states,
                save_path=str(output_dir / "fig3_position_pca.png"),
            )
        except Exception as e:
            print(f"  Skipped PCA (needs sklearn): {e}")

    # ============================================================
    # Figure 4: Grid cell autocorrelation (MapFormer-WM only)
    # ============================================================
    print("\n" + "=" * 60)
    print("FIGURE 4: Grid Cell Autocorrelation")
    print("=" * 60)

    if "MapFormer-WM" in trained_models:
        try:
            for cell_idx in [0, 1, 2]:
                rate_map = compute_rate_map(
                    trained_models["MapFormer-WM"], env,
                    cell_idx=cell_idx, n_trajectories=500,
                    traj_len=64, device=device,
                )
                plot_grid_cell_autocorrelation(
                    rate_map, cell_idx=cell_idx,
                    save_path=str(output_dir / f"fig4_autocorr_cell{cell_idx}.png"),
                )
        except Exception as e:
            print(f"  Skipped autocorrelation (needs scipy): {e}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nLength generalisation (train T={}, test T={}):".format(
        args.seq_len, test_lens[-1]))
    for name, results in all_gen_results.items():
        train_acc = results.get(args.seq_len, 0)
        test_acc = results.get(test_lens[-1], 0)
        print(f"  {name:20s}: train={train_acc:.3f}  16x={test_acc:.3f}")

    print(f"\nFigures saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
