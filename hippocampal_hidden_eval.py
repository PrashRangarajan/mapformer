#!/usr/bin/env python3
"""Test A revisited — rate maps of HIDDEN-STATE activations.

The original `hippocampal_analysis.py::Test A` queried path-integrator
blocks individually and (correctly) found stripe-like 1D patterns
rather than hexagons. That test was wrong: the hexagonal-tiling
theorem requires three sinusoidal waves at the same frequency to
superpose at 60° angles, but MapFormer's path integrator has only
ONE block per ω. Hexagonal patterns must therefore emerge — if at
all — at the hidden-state level, where attention + FFN can mix
multiple path-integrator outputs.

This script extracts the hidden state (after `out_norm`) at each
agent position, accumulates per-(x, y) activations across many
trajectories, and computes:

  - Rate maps for each hidden dim (d_model dims total)
  - Sargolini grid score per hidden dim
  - Distribution of grid scores across hidden dims, per variant

If a sufficiently expressive model has discovered the hexagonal
solution, we should see SOME hidden dims with grid scores > 0.3.

Outputs:
  HIPPOCAMPAL_HIDDEN.md
  paper_figures/fig10_hidden_rate_maps.png   — top-grid-score hidden dims
  paper_figures/fig11_grid_score_dist.png    — distribution of grid scores per variant
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_baselines_extra import EXTRA_BASELINES
from mapformer.model_grid import MapFormerWM_Grid, MapFormerWM_Grid_Free
from mapformer.model_level15_pc import MapFormerWM_Level15PC
from mapformer.model_grid_l15_pc import (
    MapFormerWM_GridL15PC, MapFormerWM_GridL15PC_Free,
)

VARIANT_CLS = {
    "Vanilla":         MapFormerWM,
    "VanillaEM":       MapFormerEM,
    "Level1":          MapFormerWM_ParallelInEKF,
    "Level15":         MapFormerWM_Level15InEKF,
    "Level15EM":       MapFormerEM_Level15InEKF,
    "Level15PC":       MapFormerWM_Level15PC,
    "PC":              MapFormerWM_PredictiveCoding,
    "RoPE":            MapFormerWM_RoPE,
    "Grid":            MapFormerWM_Grid,
    "Grid_Free":       MapFormerWM_Grid_Free,
    "GridL15PC":       MapFormerWM_GridL15PC,
    "GridL15PC_Free":  MapFormerWM_GridL15PC_Free,
    **EXTRA_BASELINES,
}


def build_model(variant, ckpt_path, device="cuda"):
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(
        vocab_size=cfg.get("vocab_size"),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 2),
        n_layers=cfg.get("n_layers", 1),
        grid_size=cfg.get("grid_size", 64),
    )
    m.load_state_dict(c["model_state_dict"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def extract_hidden_rate_maps(model, env, T=512, n_trials=40, device="cuda"):
    """Hook out_norm output, accumulate by (x, y) cell."""
    grid_size = env.size
    if not hasattr(model, "out_norm"):
        return None, None
    # Determine d_model from out_norm.normalized_shape (not all baselines store it)
    d_model = model.out_norm.normalized_shape[0]

    sums = np.zeros((d_model, grid_size, grid_size), dtype=np.float64)
    counts = np.zeros((grid_size, grid_size), dtype=np.int64)

    captured = {"hidden": None}
    def hook(module, inp, out):
        captured["hidden"] = out.detach().cpu().numpy()  # (B, L, d)
    handle = model.out_norm.register_forward_hook(hook)

    try:
        for _ in range(n_trials):
            tokens, _, _ = env.generate_trajectory(T)
            visited = env.visited_locations
            tt = tokens.unsqueeze(0).to(device)
            try:
                _ = model(tt[:, :-1])
            except Exception:
                continue
            hidden = captured["hidden"][0]  # (L, d)
            L = hidden.shape[0]
            for t in range(L):
                cell = visited[t // 2]
                x, y = cell
                sums[:, x, y] += hidden[t]
                counts[x, y] += 1
    finally:
        handle.remove()

    visited_mask = counts > 0
    rate = np.zeros_like(sums)
    rate[:, visited_mask] = sums[:, visited_mask] / counts[visited_mask]
    return rate, counts


def grid_score(rate_map):
    """Sargolini hexagonal autocorrelation score (peaks at 60°,120° vs troughs at 30°,90°,150°)."""
    from scipy.signal import correlate2d
    from scipy.ndimage import rotate as nd_rotate

    rm = rate_map - rate_map.mean()
    if rm.std() < 1e-6:
        return -2.0
    ac = correlate2d(rm, rm, mode="same")
    ac = ac / (ac.max() + 1e-12)
    cx, cy = ac.shape[0] // 2, ac.shape[1] // 2
    Y, X = np.ogrid[:ac.shape[0], :ac.shape[1]]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    valid = (r >= 5) & (r <= min(ac.shape) // 3)

    def corr_at(angle):
        rot = nd_rotate(ac, angle, reshape=False, order=1)
        a, b = ac[valid], rot[valid]
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    peaks = (corr_at(60) + corr_at(120)) / 2
    troughs = (corr_at(30) + corr_at(90) + corr_at(150)) / 3
    return peaks - troughs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--variants", nargs="+",
                   default=["Vanilla", "VanillaEM", "Level1", "Level15",
                            "Level15EM", "MambaLike"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="clean")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--test-seed", type=int, default=12345)
    p.add_argument("--top-k", type=int, default=8,
                   help="show top-k hidden dims by grid score per variant")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-md", default="HIPPOCAMPAL_HIDDEN.md")
    p.add_argument("--output-figures", default="paper_figures")
    args = p.parse_args()

    runs = Path(args.runs_dir)
    figs = Path(args.output_figures); figs.mkdir(parents=True, exist_ok=True)

    grid_scores_by_variant = {}
    rate_maps_by_variant = {}

    for variant in args.variants:
        ckpt = runs / f"{variant}_{args.config}" / f"seed{args.seed}" / f"{variant}.pt"
        if not ckpt.exists():
            print(f"  [skip] {variant}: no ckpt", file=sys.stderr); continue
        try:
            model, cfg = build_model(variant, ckpt, args.device)
        except Exception as e:
            print(f"  [skip] {variant}: {e}", file=sys.stderr); continue
        env = GridWorld(
            size=cfg.get("grid_size", 64),
            n_obs_types=cfg.get("n_obs_types", 16),
            p_empty=cfg.get("p_empty", 0.5),
            n_landmarks=cfg.get("n_landmarks", 0),
            seed=args.test_seed,
        )
        rate, counts = extract_hidden_rate_maps(model, env, args.T,
                                                args.n_trials, args.device)
        if rate is None:
            print(f"  [skip] {variant}: no out_norm hook target", file=sys.stderr)
            continue
        # Compute grid score per dim
        d_model = rate.shape[0]
        scores = []
        for j in range(d_model):
            if rate[j].std() > 0.01:
                try:
                    scores.append((grid_score(rate[j]), j))
                except Exception:
                    pass
        scores.sort(reverse=True)
        grid_scores_by_variant[variant] = [s for s, _ in scores]
        rate_maps_by_variant[variant] = (rate, scores[:args.top_k])
        if scores:
            top_scores = [s for s, _ in scores[:5]]
            print(f"  {variant}: top-5 grid scores = "
                  f"{', '.join(f'{s:+.3f}' for s in top_scores)}", file=sys.stderr)

    # Figure 10: top-k hidden rate maps per variant
    n_var = len(rate_maps_by_variant)
    fig, axes = plt.subplots(n_var, args.top_k,
                             figsize=(args.top_k * 1.5, n_var * 1.6))
    if n_var == 1:
        axes = axes[None, :]
    for i, (variant, (rate, top)) in enumerate(rate_maps_by_variant.items()):
        for j in range(args.top_k):
            ax = axes[i, j]
            if j < len(top):
                score, dim = top[j]
                ax.imshow(rate[dim], cmap="RdBu_r")
                ax.set_title(f"d={dim}\ngs={score:+.2f}", fontsize=7)
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(variant, fontsize=10)
    fig.suptitle("Top-grid-score hidden-state rate maps (each panel: one hidden dim)\n"
                 "Sargolini grid score (gs) measures hexagonal autocorrelation; "
                 ">0.3 typically grid-like",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(figs / "fig10_hidden_rate_maps.png", dpi=150, bbox_inches="tight")
    print(f"  wrote {figs}/fig10_hidden_rate_maps.png", file=sys.stderr)
    plt.close(fig)

    # Figure 11: distribution of grid scores
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, scores in grid_scores_by_variant.items():
        if scores:
            ax.hist(scores, bins=25, alpha=0.5, label=f"{variant} (max={max(scores):.2f})")
    ax.axvline(0.3, color="black", linestyle=":", linewidth=1.0,
               label="grid-like threshold (Sargolini ≥ 0.3)")
    ax.set_xlabel("Sargolini grid score (per hidden dim)", fontsize=11)
    ax.set_ylabel("# hidden dims", fontsize=11)
    ax.set_title("Distribution of grid scores across hidden-state dimensions\n"
                 "(do any dims learn hexagonal firing fields?)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figs / "fig11_grid_score_dist.png", dpi=160, bbox_inches="tight")
    print(f"  wrote {figs}/fig11_grid_score_dist.png", file=sys.stderr)
    plt.close(fig)

    # Markdown
    md = ["# Hippocampal Test A Revisited — Hidden-State Rate Maps\n"]
    md.append("The original Test A queried individual path-integrator blocks "
              "and found stripe-like 1D patterns. That test was *correct in "
              "what it measured but wrong as a test of hexagonality*: each "
              "MapFormer block has ONE 1D phase angle per frequency, so by "
              "construction it cannot exhibit hexagonal interference (which "
              "requires 3 waves at 60° at the same frequency).\n")
    md.append("Hexagonality, if present, must emerge at the **hidden-state "
              "level**, where attention and FFN can mix multiple "
              "path-integrator blocks. We extract the hidden state (output "
              "of `out_norm`) at every visited cell, then compute spatial "
              "autocorrelation grid scores per hidden dim.\n")
    md.append("![Top hidden rate maps](paper_figures/fig10_hidden_rate_maps.png)\n")
    md.append("![Grid score distribution](paper_figures/fig11_grid_score_dist.png)\n")
    md.append("## Top-grid-score hidden dim per variant\n")
    md.append("| Variant | max grid score | # dims with score > 0.3 | # dims with score > 0 |")
    md.append("|---|---|---|---|")
    for variant, scores in grid_scores_by_variant.items():
        if not scores:
            md.append(f"| {variant} | — | — | — |"); continue
        max_score = max(scores)
        n_high = sum(1 for s in scores if s > 0.3)
        n_pos = sum(1 for s in scores if s > 0)
        md.append(f"| {variant} | {max_score:+.3f} | {n_high} | {n_pos} |")
    md.append("\n*Auto-generated by `hippocampal_hidden_eval.py`.*\n")
    Path(args.output_md).write_text("\n".join(md))
    print(f"\nwrote {args.output_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
