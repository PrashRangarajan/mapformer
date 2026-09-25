"""Generate paper figure showing top place-cell-like units per variant.

v2 changes vs v1:
  - For TEMFaithful, capture the STRUCTURAL CODE g (not the readout x_hat).
    g is the spatial-state vector inside TEM; x_hat is the retrieved-content
    readout, which is content-typed and not place-cell-like.
  - For MapFormer variants, capture the post-LayerNorm last-layer hidden
    state (input to out_proj) — unchanged from v1, this is the right
    representation.
  - Mask cells with <min_visits visits before stats / display.
  - Gaussian-smooth rate maps (sigma=1.5 bins) before computing peak ratio
    and before display, matching standard place-cell analysis.

Headline finding (to be re-tested): place cells emerge in EVERY MapFormer
variant including RoPE. TEM's structural code g should show grid- /
place-like spatial tuning if anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from mapformer.environment import GridWorld
from mapformer.probe_hex_emergence import VARIANT_CLS, build, collect_rate_maps
from mapformer.probe_hex import spatial_autocorrelogram, grid_score


def collect_tem_g_rate_maps(model, env, n_trajectories, T):
    """Capture TEM's structural code g at obs positions, build rate map.

    Reimplements model.forward step-by-step (we need g per step, not just
    the final x_hat readout). g is updated by W_a only at action tokens; at
    obs token 2t+1 g represents the agent's post-action position positions[t].
    """
    gs = env.size
    d_g = model.d_g
    sums = np.zeros((d_g, gs, gs), dtype=np.float64)
    counts = np.zeros((gs, gs), dtype=np.float64)

    for _ in range(n_trajectories):
        tokens, _, _ = env.generate_trajectory(T)
        positions = list(env.visited_locations)
        tt = tokens.cuda()
        with torch.no_grad():
            W_a_all = model._orthogonal_W()                       # (n_act, d_g, d_g)
            g = model.g_init.unsqueeze(0).clone()                 # (1, d_g)
            t_obs = 0
            for t in range(tt.shape[0]):
                tok_id = int(tt[t].item())
                if tok_id < model.n_actions:
                    Wb = W_a_all[tok_id]                          # (d_g, d_g)
                    g = (Wb @ g.unsqueeze(-1)).squeeze(-1)        # (1, d_g)
                else:
                    if t_obs < len(positions):
                        x, y = positions[t_obs]
                        sums[:, x, y] += g[0].cpu().numpy()
                        counts[x, y] += 1
                    t_obs += 1

    rate_maps = np.zeros_like(sums)
    visited = counts > 0
    for u in range(d_g):
        rate_maps[u, visited] = sums[u, visited] / counts[visited]
    return rate_maps, counts


def smooth_and_score(rate_maps, counts, sigma=1.5, min_visits=5):
    """Gaussian-smooth rate maps (over visited cells only) and score units.

    Returns: (smoothed_maps, stats_list). stats has 'peak', 'grid' fields
    computed AFTER smoothing on cells with >= min_visits visits.
    """
    n_units, gs, _ = rate_maps.shape
    visit_mask = counts >= min_visits                             # (gs, gs)
    smoothed = np.zeros_like(rate_maps)
    stats = []
    for u in range(n_units):
        rm = rate_maps[u].copy()
        rm[counts == 0] = 0.0
        sm = gaussian_filter(rm, sigma=sigma, mode="wrap")
        smoothed[u] = sm
        rm_v = sm[visit_mask]
        if len(rm_v) < 10 or rm_v.std() < 1e-8:
            stats.append({"peak": 0.0, "grid": 0.0})
            continue
        peak = float(rm_v.max() / (abs(rm_v.mean()) + 1e-6))
        try:
            sac = spatial_autocorrelogram(sm)
            g_score = float(grid_score(sac))
        except Exception:
            g_score = 0.0
        stats.append({"peak": peak, "grid": g_score})
    return smoothed, stats, visit_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-png", default="paper_figures/place_cells_per_variant.png")
    ap.add_argument("--n-trajectories", type=int, default=500)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--smooth-sigma", type=float, default=1.5)
    ap.add_argument("--min-visits", type=int, default=5)
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
        if v == "TEMFaithful":
            rate_maps, counts = collect_tem_g_rate_maps(
                model, env, args.n_trajectories, args.T,
            )
            print(f"  TEM g representation: d_g={model.d_g}")
        else:
            rate_maps, counts = collect_rate_maps(
                model, env, args.n_trajectories, args.T, v,
            )

        smoothed, stats, visit_mask = smooth_and_score(
            rate_maps, counts,
            sigma=args.smooth_sigma, min_visits=args.min_visits,
        )
        peaks = np.array([s["peak"] for s in stats])
        top_idx = np.argsort(peaks)[-args.top_k:][::-1]
        per_variant[v] = {
            "smoothed": smoothed[top_idx],
            "peaks": peaks[top_idx],
            "grid_scores": np.array([stats[i]["grid"] for i in top_idx]),
            "counts": counts,
            "visit_mask": visit_mask,
        }
        cov = (counts > 0).mean()
        med_visits = float(np.median(counts[counts > 0])) if (counts > 0).any() else 0
        print(f"  coverage={cov:.2%}, median visits/visited cell={med_visits:.0f}")
        print(f"  top-{args.top_k} peak ratios (smoothed, min_visits>={args.min_visits}): {peaks[top_idx]}")
        del model; torch.cuda.empty_cache()

    n_rows = len(per_variant)
    n_cols = args.top_k
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1: axes = [axes]
    for i, (variant, data) in enumerate(per_variant.items()):
        for j in range(min(n_cols, len(data["smoothed"]))):
            ax = axes[i][j] if n_rows > 1 else axes[j]
            rm = data["smoothed"][j].copy()
            rm[~data["visit_mask"]] = np.nan
            im = ax.imshow(rm, cmap="hot", interpolation="nearest")
            peak = data["peaks"][j]
            grid = data["grid_scores"][j]
            ax.set_title(f"{variant} unit, peak={peak:.1f}x, grid={grid:.2f}",
                          fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(
        f"Top-{args.top_k} place-cell-like units per variant "
        f"(smoothed σ={args.smooth_sigma}, min_visits≥{args.min_visits}, "
        f"{args.n_trajectories}×T={args.T} trajs)",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output_png}")


if __name__ == "__main__":
    main()
