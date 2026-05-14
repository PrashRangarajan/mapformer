"""Probe for hex / place cell emergence in trained prediction-only models.

For each variant + seed, generate many trajectories on the training env,
record (agent_pos, internal_rep) at each obs position. For each unit in
internal_rep, compute a per-cell rate map and a Sargolini-style grid score.

If grid-cell-like hex patterns emerge from prediction training, units
should have grid scores > 0.3.

This is THE classical neuroscience test of cognitive-map representations:
do trained models show place cells (one peak per unit) and grid cells
(hex-periodic firing) like hippocampus and entorhinal cortex?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_gsf_nodrop import MapFormerWM_Level15GSF_NoDrop
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_tem_faithful import TEMFaithful
from mapformer.probe_hex import spatial_autocorrelogram, grid_score


VARIANT_CLS = {
    "RoPE":               MapFormerWM_RoPE,
    "Vanilla":            MapFormerWM,
    "Level15":            MapFormerWM_Level15InEKF,
    "Level15GSF_NoDrop":  MapFormerWM_Level15GSF_NoDrop,
    "TEMFaithful":        TEMFaithful,
}


class _MultiCallHiddenCapture:
    """Captures hidden state at every call of out_proj. For TEMFaithful (per-step
    forward loop) this gives all L hidden states. For batched MapFormer it gives
    one capture with shape (B, L, d)."""
    def __init__(self): self.hiddens = []
    def reset(self): self.hiddens = []
    def __call__(self, mod, inp, out):
        h = inp[0].detach()
        if h.dim() == 4: h = h[:, 0]  # GSF: take mode 0
        self.hiddens.append(h)


def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c["config"]; cls = VARIANT_CLS[variant]
    kw = dict(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
              n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
              grid_size=cfg["grid_size"])
    if variant == "Level15GSF_NoDrop": kw["n_modes"] = 8
    m = cls(**kw); m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval(), cfg


def collect_rate_maps(model, env, n_trajectories, T, variant, device="cuda"):
    """Build per-unit rate map by averaging hidden activations across cell visits."""
    cap = _MultiCallHiddenCapture()
    handle = model.out_proj.register_forward_hook(cap)

    # First pass: find the hidden dim
    cap.reset()
    tokens, _, _ = env.generate_trajectory(64)
    _ = model(tokens.unsqueeze(0).to(device))
    # For MapFormer family: hiddens is a list with 1 element of shape (1, L, d)
    # For TEM: hiddens is a list with L elements of shape (1, d) (per-step calls)
    if cap.hiddens[0].dim() == 3:
        # Single call, shape (B, L, d). MapFormer family.
        hidden_dim = cap.hiddens[0].shape[-1]
    else:
        # Many calls, each (B, d). TEMFaithful.
        hidden_dim = cap.hiddens[0].shape[-1]

    gs = env.size
    sums = np.zeros((hidden_dim, gs, gs), dtype=np.float64)
    counts = np.zeros((gs, gs), dtype=np.float64)

    for traj in range(n_trajectories):
        cap.reset()
        tokens, _, _ = env.generate_trajectory(T)
        positions = list(env.visited_locations)
        tt = tokens.unsqueeze(0).to(device)
        _ = model(tt)

        # Reassemble hidden states per step
        if len(cap.hiddens) == 1 and cap.hiddens[0].dim() == 3:
            # MapFormer: single (1, L, d) call. Take obs positions (odd indices).
            h_full = cap.hiddens[0][0]  # (L, d)
            for t, (x, y) in enumerate(positions):
                obs_idx = 2 * t + 1
                if obs_idx >= h_full.shape[0]: continue
                vec = h_full[obs_idx].cpu().numpy()
                sums[:, x, y] += vec
                counts[x, y] += 1
        else:
            # TEMFaithful: L per-step calls, each (1, d). Each call is at one token.
            # Token sequence is (a, o, a, o, ...). Obs positions = odd indices.
            for t, (x, y) in enumerate(positions):
                obs_idx = 2 * t + 1
                if obs_idx >= len(cap.hiddens): continue
                h = cap.hiddens[obs_idx]
                if h.dim() == 2: h = h[0]
                vec = h.cpu().numpy()
                sums[:, x, y] += vec
                counts[x, y] += 1

    handle.remove()
    # Build rate map: average per cell. Skip cells with zero visits.
    rate_maps = np.zeros_like(sums)
    visited = counts > 0
    for u in range(hidden_dim):
        rate_maps[u, visited] = sums[u, visited] / counts[visited]
    return rate_maps, counts


def compute_unit_stats(rate_maps, counts):
    """For each unit, compute (peak_score, sparsity, grid_score)."""
    n_units, gs, _ = rate_maps.shape
    stats = []
    for u in range(n_units):
        rm = rate_maps[u]
        # Restrict to visited cells
        if counts.sum() == 0: continue
        rm_v = rm[counts > 0]
        if len(rm_v) == 0:
            stats.append({"peak": 0.0, "sparsity": 0.0, "grid": 0.0}); continue
        # peak ratio: max activation / mean
        peak = float(rm_v.max() / (abs(rm_v.mean()) + 1e-6))
        # sparsity: kurtosis-like
        sparsity = float(((rm_v - rm_v.mean()) ** 4).mean() / ((rm_v - rm_v.mean()) ** 2).mean() ** 2 - 3)
        # grid score via SAC
        try:
            sac = spatial_autocorrelogram(rm)
            g_score = float(grid_score(sac))
        except Exception:
            g_score = 0.0
        stats.append({"peak": peak, "sparsity": sparsity, "grid": g_score})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-md", default="HEX_EMERGENCE_RESULTS.md")
    ap.add_argument("--n-trajectories", type=int, default=100)
    ap.add_argument("--T", type=int, default=256)
    args = ap.parse_args()

    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=200, seed=0)

    results = {}
    for variant in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop", "TEMFaithful"]:
        per_seed_stats = []
        for s in [0, 1, 2]:
            ckpt = Path(f"mapformer/runs/{variant}_lm200/seed{s}/{variant}.pt")
            if not ckpt.exists(): continue
            try:
                model, cfg = build(variant, ckpt)
            except Exception as e:
                print(f"  skip {variant} s{s}: {e}"); continue

            print(f"\n=== {variant} s{s} ===")
            print(f"  collecting rate maps from {args.n_trajectories} trajs of T={args.T}...")
            rate_maps, counts = collect_rate_maps(
                model, env, args.n_trajectories, args.T, variant,
            )
            print(f"  rate_maps shape: {rate_maps.shape}, visited cells: {(counts > 0).sum()}/{counts.size}")
            stats = compute_unit_stats(rate_maps, counts)
            scores = np.array([s["grid"] for s in stats])
            peaks = np.array([s["peak"] for s in stats])
            print(f"  grid scores: mean {scores.mean():.3f}, max {scores.max():.3f}, "
                  f"frac > 0.3: {(scores > 0.3).mean():.3f}")
            print(f"  peak ratios: mean {peaks.mean():.2f}, max {peaks.max():.2f}")
            per_seed_stats.append({
                "grid_scores": scores,
                "peak_ratios": peaks,
                "max_grid": float(scores.max()),
                "frac_grid": float((scores > 0.3).mean()),
                "frac_place": float((peaks > 5.0).mean()),
                "n_units": len(stats),
            })
            del model; torch.cuda.empty_cache()

        if per_seed_stats:
            results[variant] = {
                "max_grid_score": float(np.mean([s["max_grid"] for s in per_seed_stats])),
                "frac_grid_cells": float(np.mean([s["frac_grid"] for s in per_seed_stats])),
                "frac_place_cells": float(np.mean([s["frac_place"] for s in per_seed_stats])),
                "n_units": per_seed_stats[0]["n_units"],
                "n_seeds": len(per_seed_stats),
            }

    # Markdown
    lines = []
    lines.append("# Hex / place-cell emergence in prediction-trained models\n")
    lines.append("Classical neuroscience test: do trained cognitive-map models show grid-cell-like")
    lines.append("(hex-periodic firing) or place-cell-like (peaked at one cell) representations,")
    lines.append("analogous to entorhinal cortex and hippocampus?\n")
    lines.append("Procedure:\n")
    lines.append("1. Frozen prediction-trained lm200 checkpoint")
    lines.append("2. Run trajectories on training env, record (agent_pos, hidden_state) per step")
    lines.append("3. For each unit, build per-cell rate map across visits")
    lines.append("4. Compute Sargolini-style grid score (peak correlation at 60° & 120° minus 30° & 90° & 150°)")
    lines.append("5. Report frac of 'grid cells' (score > 0.3) and 'place cells' (peak ratio > 5x)\n")
    lines.append(f"Settings: {args.n_trajectories} trajectories × T={args.T} steps. Multi-seed (n=3 lm200 checkpoints).\n")

    lines.append("## Hex / place cell emergence per variant\n")
    lines.append("| Variant | Hidden dim | Max grid score | Frac 'grid cells' (>0.3) | Frac 'place cells' (peak>5x) | n_seeds |")
    lines.append("|---|---|---|---|---|---|")
    for v in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop", "TEMFaithful"]:
        if v not in results: print(f"| {v} | — | — | — | — | 0 |"); continue
        r = results[v]
        lines.append(f"| **{v}** | {r['n_units']} | {r['max_grid_score']:.3f} | {r['frac_grid_cells']:.3f} | {r['frac_place_cells']:.3f} | {r['n_seeds']} |")
    lines.append("")
    lines.append("Reference thresholds (Sargolini grid score):")
    lines.append("- 0.3 = canonical threshold for 'grid cell' (used in neuroscience papers)")
    lines.append("- 0.0 = chance / no hex structure")
    lines.append("- Negative = anti-grid (more 30°/90° periodicity than 60°/120°)\n")

    lines.append("## Interpretation\n")
    lines.append("- If any variant shows fraction > 0.05 of grid cells: hex emerges naturally from prediction training")
    lines.append("- If TEMFaithful shows higher fraction: TEM's per-action W_a + Hopfield structure favors hex (paper's claim)")
    lines.append("- If all near zero: hex doesn't emerge in our setup. The Sorscher 2019 conditions (non-negativity + DoG targets) likely needed.")
    lines.append("- Place-cell-like units (high peak ratio): more common, indicates the model learns localized spatial features\n")
    lines.append("*Auto-generated by probe_hex_emergence.py*\n")

    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-25:]))


if __name__ == "__main__":
    main()
