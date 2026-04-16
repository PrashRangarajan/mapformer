#!/usr/bin/env python3
"""
MapFormer: Full experiment pipeline.

Trains MapFormer-WM and MapFormer-EM on a 2D torus grid navigation task,
then generates figures for length generalisation, PCA, and autocorrelation.

Usage:
    python3 -m mapformer.main [--device cuda] [--epochs 16]
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
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
    """Mathematical sanity checks."""
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    from mapformer.lie_groups import (
        skew_symmetric_2d, exp_map_2d, build_block_diagonal_rotations_fast,
    )

    theta = torch.randn(16)
    R = exp_map_2d(theta)
    eye = torch.eye(2).expand(16, 2, 2)
    ortho_ok = torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-6)
    print(f"  [{'PASS' if ortho_ok else 'FAIL'}] exp(A)^T @ exp(A) = I")

    dets = torch.det(R)
    det_ok = torch.allclose(dets, torch.ones(16), atol=1e-6)
    print(f"  [{'PASS' if det_ok else 'FAIL'}] det(exp(A)) = 1")

    M_test = exp_map_2d(torch.randn(2 * 8)).reshape(2, 8, 2, 2)
    par = parallel_prefix_product(M_test)
    seq = sequential_prefix_product(M_test)
    scan_ok = torch.allclose(par, seq, atol=1e-5)
    print(f"  [{'PASS' if scan_ok else 'FAIL'}] Parallel prefix product matches sequential")

    print()


def build_mapformer_models(vocab_size, d_model, n_heads, n_layers, grid_size):
    """Instantiate MapFormer models with unified vocab."""
    models = {
        "MapFormer-WM": MapFormerWM(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, grid_size=grid_size,
        ),
        "MapFormer-EM": MapFormerEM(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, grid_size=grid_size,
        ),
    }
    return models


def main():
    parser = argparse.ArgumentParser(description="MapFormer experiment pipeline")

    # Paper defaults (Rambaud et al., 2025, Appendix B)
    parser.add_argument("--epochs", type=int, default=16, help="Training epochs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--n-obs-types", type=int, default=16,
                        help="Number of non-blank observation types (K)")
    parser.add_argument("--p-empty", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=128,
                        help="Steps per trajectory (each step = 1 action + 1 observation)")
    parser.add_argument("--n-batches", type=int, default=98,
                        help="Batches per epoch (16 epochs * 98 * 128 ≈ 200K sequences)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="figures")
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

    # Sanity checks
    run_sanity_checks()

    # Environment
    env = GridWorld(
        size=args.grid_size, n_obs_types=args.n_obs_types,
        p_empty=args.p_empty, seed=args.seed,
    )

    d_head = args.d_model // args.n_heads
    n_rot_dims = d_head // 2
    total_seqs = args.epochs * args.n_batches * args.batch_size

    print(f"Device: {device}")
    print(f"Grid: {args.grid_size}x{args.grid_size} TORUS, "
          f"{args.n_obs_types} obs types + blank, p_empty={args.p_empty}")
    print(f"Unified vocab: {env.unified_vocab_size} "
          f"(4 actions + {args.n_obs_types} obs + 1 blank)")
    print(f"Model: d={args.d_model}, heads={args.n_heads}, layers={args.n_layers}, "
          f"d_head={d_head}, n_rot={n_rot_dims}")
    print(f"Training: {args.epochs} epochs × {args.n_batches} batches × "
          f"{args.batch_size} batch = {total_seqs:,} sequences")
    print(f"  n_steps={args.n_steps} (interleaved tokens: {2*args.n_steps})")
    print(f"  lr={args.lr}, AdamW, linear decay, wd=0.05")
    print()

    # Build models
    all_models = build_mapformer_models(
        env.unified_vocab_size, args.d_model, args.n_heads,
        args.n_layers, args.grid_size,
    )

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
            batch_size=args.batch_size, n_steps=args.n_steps,
            n_batches=args.n_batches, device=device,
        )

        all_losses[name] = losses
        trained_models[name] = model

        # Save checkpoint
        ckpt_path = output_dir / f"{name.replace('-', '_')}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "losses": losses,
            "config": {
                "vocab_size": env.unified_vocab_size,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "grid_size": args.grid_size,
                "n_obs_types": args.n_obs_types,
                "p_empty": args.p_empty,
            },
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

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

    test_lens = [args.n_steps, 2*args.n_steps, 4*args.n_steps,
                 8*args.n_steps, 16*args.n_steps]

    all_gen_results = {}
    for name, model in trained_models.items():
        print(f"\n  Evaluating {name}...")
        results = eval_length_generalisation(
            model, env, train_len=args.n_steps,
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
    # Figure 3: Position state PCA
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
            print(f"  Skipped PCA: {e}")

    # ============================================================
    # Figure 4: Grid cell autocorrelation
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
            print(f"  Skipped autocorrelation: {e}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nLength generalisation (train T={args.n_steps}, "
          f"test T={test_lens[-1]}):")
    for name, results in all_gen_results.items():
        train_acc = results.get(args.n_steps, 0)
        test_acc = results.get(test_lens[-1], 0)
        print(f"  {name:20s}: train={train_acc:.3f}  16x={test_acc:.3f}")

    print(f"\nFigures saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
