#!/usr/bin/env python3
"""Hexagonal-grid-cell test for Grid / Grid_Free / GridL15PC variants.

The standard hippocampal rate-map test plots ``cos(θ̂)`` per block. For
single-block-per-ω architectures (Vanilla, Level15) this is the right
quantity. For *multi-orientation* architectures (Grid family), each
block within a module produces a 1D stripe pattern; the **hexagonal
interference** appears only after summing the three blocks within a
module:

    h_module(x, y) = Σ_o cos(θ_{m,o,t})       summed over orientations o

This script computes per-module rate maps (one per (head, module) pair),
which is the right quantity to compute the Sargolini grid score on for
testing whether hex emerges.

Usage::

    python3 -m mapformer.grid_module_hex_test \\
        --variant Grid_Free --seed 0 --config clean \\
        --T 256 --n-trials 30 --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.hippocampal_analysis import VARIANT_CLS, build_model, grid_score


@torch.no_grad()
def compute_module_rate_maps(model, env, T=256, n_trials=30, device="cuda"):
    """Compute per-module rate maps by summing cos(θ̂) across orientations.

    Returns:
        rate_maps: (n_heads * n_modules, grid_size, grid_size) — one per module
        counts:    (grid_size, grid_size) — visit counts
    """
    grid = env.size
    H = model.n_heads
    M = model.n_modules
    O = model.n_orientations
    n_state = H * M

    sums = np.zeros((n_state, grid, grid), dtype=np.float64)
    counts = np.zeros((grid, grid), dtype=np.int64)

    for _ in range(n_trials):
        tokens, _, _ = env.generate_trajectory(T)
        visited = env.visited_locations
        tt = tokens.unsqueeze(0).to(device)
        try:
            _ = model(tt[:, :-1])
        except Exception:
            continue

        # Recover θ̂ either from the model's saved cache or by recomputing.
        if hasattr(model, "last_theta_hat") and model.last_theta_hat is not None:
            theta = model.last_theta_hat[0]  # (L, H, M*O)
        else:
            x = model.token_emb(tt[:, :-1])
            delta_2d = model.action_to_lie(x)  # (B, L, H, M, 2)
            dx = delta_2d[..., 0]; dy = delta_2d[..., 1]
            cos_o = torch.cos(model.path_integrator.orientation_angles)
            sin_o = torch.sin(model.path_integrator.orientation_angles)
            d_block = dx.unsqueeze(-1) * cos_o + dy.unsqueeze(-1) * sin_o
            cum = torch.cumsum(d_block, dim=1)
            omega = model.path_integrator.omega.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            theta = (cum * omega).reshape(1, -1, H, M * O)[0]   # (L, H, M*O)

        # Reshape (L, H, M*O) → (L, H, M, O), sum over O → (L, H, M)
        L = theta.shape[0]
        theta_per_block = theta.reshape(L, H, M, O)
        cos_per_block = torch.cos(theta_per_block)
        per_module = cos_per_block.sum(dim=-1).cpu().numpy()    # (L, H, M)

        for t in range(L):
            cell = visited[t // 2]
            x, y = cell
            sums[:, x, y] += per_module[t].reshape(-1)
            counts[x, y] += 1

    visited_mask = counts > 0
    rate = np.zeros_like(sums)
    rate[:, visited_mask] = sums[:, visited_mask] / counts[visited_mask]
    return rate, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   help="Grid_Free / GridL15PC_Free / etc. (Grid-family only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="clean")
    p.add_argument("--runs-dir", default="mapformer/runs")
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cls = VARIANT_CLS[args.variant]
    if not hasattr(cls, "__name__") or "Grid" not in cls.__name__:
        print(f"Warning: {args.variant} is not Grid-family — module test "
              "may not be meaningful.")

    ckpt_path = (Path(args.runs_dir) / f"{args.variant}_{args.config}"
                 / f"seed{args.seed}" / f"{args.variant}.pt")
    if not ckpt_path.exists():
        print(f"FATAL: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    model, cfg = build_model(args.variant, str(ckpt_path), device=args.device)
    print(f"Loaded {args.variant} from {ckpt_path}")
    print(f"  n_heads={model.n_heads}, n_modules={model.n_modules}, "
          f"n_orientations={model.n_orientations}")

    env = GridWorld(
        size=cfg.get("grid_size", 64),
        n_obs_types=cfg.get("n_obs_types", 16),
        p_empty=cfg.get("p_empty", 0.5),
        n_landmarks=cfg.get("n_landmarks", 0),
        seed=args.seed,
    )

    rate, counts = compute_module_rate_maps(
        model, env, T=args.T, n_trials=args.n_trials, device=args.device,
    )

    # Compute grid scores per module
    print("\nGrid scores (per-module hex test, Sargolini autocorrelation):")
    scores = []
    for j in range(rate.shape[0]):
        s = grid_score(rate[j])
        scores.append(s)
        head_id = j // model.n_modules
        mod_id = j % model.n_modules
        print(f"  head={head_id} module={mod_id:2d}: grid_score = {s:+.3f}")

    scores = np.array(scores)
    print(f"\nSummary for {args.variant} (config={args.config}, seed={args.seed}):")
    print(f"  mean grid score   = {scores.mean():+.3f}")
    print(f"  median grid score = {np.median(scores):+.3f}")
    print(f"  max grid score    = {scores.max():+.3f}")
    print(f"  fraction > 0.0    = {(scores > 0.0).mean():.2%}")
    print(f"  fraction > 0.3    = {(scores > 0.3).mean():.2%}  (Sargolini hex threshold)")


if __name__ == "__main__":
    main()
