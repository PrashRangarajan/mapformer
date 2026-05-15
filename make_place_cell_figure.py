"""Generate paper figure showing top place-cell-like units per variant.

For each variant + seed, load checkpoint, run trajectories, compute rate maps,
identify the top-3 place-cell-like units (highest peak ratio), plot as
64×64 heatmaps. Output: a multi-panel PNG.

Headline finding: place cells emerge in EVERY variant including RoPE.
Hex grid cells emerge in NONE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mapformer.environment import GridWorld
from mapformer.probe_hex_emergence import (
    VARIANT_CLS, build, collect_rate_maps, compute_unit_stats,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-png", default="paper_figures/place_cells_per_variant.png")
    ap.add_argument("--n-trajectories", type=int, default=100)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=200, seed=0)

    variants = ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop", "TEMFaithful"]
    per_variant = {}

    for v in variants:
        ckpt = Path(f"mapformer/runs/{v}_lm200/seed{args.seed}/{v}.pt")
        if not ckpt.exists():
            print(f"  skip {v}: no ckpt")
            continue
        print(f"\n=== {v} ===")
        model, cfg = build(v, ckpt)
        rate_maps, counts = collect_rate_maps(
            model, env, args.n_trajectories, args.T, v,
        )
        stats = compute_unit_stats(rate_maps, counts)
        # Pick top-k units by peak ratio
        peaks = np.array([s["peak"] for s in stats])
        top_idx = np.argsort(peaks)[-args.top_k:][::-1]
        per_variant[v] = {
            "rate_maps": rate_maps[top_idx],
            "peaks": peaks[top_idx],
            "grid_scores": np.array([stats[i]["grid"] for i in top_idx]),
            "counts": counts,
        }
        print(f"  top-{args.top_k} peak ratios: {peaks[top_idx]}")
        del model; torch.cuda.empty_cache()

    # Plot: rows = variants, cols = top-k units
    n_rows = len(per_variant)
    n_cols = args.top_k
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1: axes = [axes]
    for i, (variant, data) in enumerate(per_variant.items()):
        for j in range(min(n_cols, len(data["rate_maps"]))):
            ax = axes[i][j] if n_rows > 1 else axes[j]
            rm = data["rate_maps"][j]
            counts = data["counts"]
            # Mask unvisited cells (counts == 0)
            rm_disp = rm.copy()
            rm_disp[counts == 0] = np.nan
            im = ax.imshow(rm_disp, cmap="hot", interpolation="nearest")
            peak = data["peaks"][j]
            grid = data["grid_scores"][j]
            ax.set_title(f"{variant} unit, peak={peak:.1f}x, grid={grid:.2f}",
                          fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"Top-{args.top_k} place-cell-like units per variant "
                  f"(rate maps over 64×64 torus)", fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output_png}")


if __name__ == "__main__":
    main()
