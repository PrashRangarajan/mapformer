#!/usr/bin/env python3
"""Hippocampal/grid-cell correspondence analysis.

Three tests, in order from simplest to most ambitious, that compare
trained models' representations to known properties of place / grid /
boundary cells in entorhinal cortex and hippocampus.

  Test A — Spatial rate maps + grid-cell autocorrelation
    For each path-integrator block (or hidden-state dim), plot
    activation as a function of agent (x, y). Real grid cells show
    hexagonal periodic firing fields; real place cells show single
    Gaussian-like peaks. We compare what each MapFormer variant
    produces.

  Test B — R_t as boundary/object cell activity
    Solstad et al. 2008 / Lever et al. 2009: entorhinal "boundary
    cells" and "object cells" fire selectively at distinctive features.
    Level 1.5's per-token R_t head should show the analogous pattern:
    small R (sharp posterior) at landmark tokens, large R elsewhere.

  Test C — ω frequency module structure
    Stensola et al. 2012 (Nature): grid-cell modules have spacings
    that follow a roughly √2 ratio. Plot the trained ω vector on log
    scale and compare to this prediction.

Outputs:
  HIPPOCAMPAL_ANALYSIS.md   — quantitative summary
  paper_figures/fig7_rate_maps.png         (Test A)
  paper_figures/fig8_R_landmark.png        (Test B)
  paper_figures/fig9_omega_modules.png     (Test C)

Usage:
  python3 -m mapformer.hippocampal_analysis \
      --runs-dir runs --output-md HIPPOCAMPAL_ANALYSIS.md \
      --output-figures paper_figures
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

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
from mapformer.model_grid_l15_pc import (
    MapFormerWM_GridL15PC, MapFormerWM_GridL15PC_Free,
)


VARIANT_CLS = {
    "Vanilla":         MapFormerWM,
    "VanillaEM":       MapFormerEM,
    "Level1":          MapFormerWM_ParallelInEKF,
    "Level15":         MapFormerWM_Level15InEKF,
    "Level15EM":       MapFormerEM_Level15InEKF,
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


# ----------------------------------------------------------------------
# Test A — Rate maps
# ----------------------------------------------------------------------

@torch.no_grad()
def compute_rate_maps(model, env, T=512, n_trials=40, device="cuda"):
    """For each path-integrator block, compute mean cos(θ) by (x, y).

    Returns:
      rate_maps:  (n_heads * n_blocks, grid_size, grid_size) ndarray
      visit_counts: (grid_size, grid_size) ndarray (number of times
        each cell was visited across all trials)
    """
    grid_size = env.size
    if not hasattr(model, "path_integrator"):
        return None, None

    n_heads = model.path_integrator.omega.shape[0]
    n_blocks = model.path_integrator.omega.shape[1]
    # For Grid models, the effective state dim is n_heads * n_modules * n_orientations
    # (omega has shape (H, M); orientations multiply the feature count).
    if hasattr(model, "n_modules") and hasattr(model, "n_orientations"):
        n_state = n_heads * model.n_modules * model.n_orientations
    else:
        n_state = n_heads * n_blocks

    sums = np.zeros((n_state, grid_size, grid_size), dtype=np.float64)
    counts = np.zeros((grid_size, grid_size), dtype=np.int64)

    for _ in range(n_trials):
        tokens, _, _ = env.generate_trajectory(T)
        visited = env.visited_locations
        tt = tokens.unsqueeze(0).to(device)
        try:
            _ = model(tt[:, :-1])
        except Exception:
            continue
        # Use the corrected θ̂ if available, else compute θ_path on the fly.
        if hasattr(model, "last_theta_hat") and model.last_theta_hat is not None:
            theta = model.last_theta_hat[0]   # (L, H, NB)
        elif hasattr(model, "n_modules") and hasattr(model, "n_orientations"):
            # Grid path integration: action_to_lie returns (B, L, H, M, 2);
            # need to project onto orientations and apply ω before cumsum-cum-prod.
            x = model.token_emb(tt[:, :-1])
            delta_2d = model.action_to_lie(x)             # (B, L, H, M, 2)
            dx = delta_2d[..., 0]
            dy = delta_2d[..., 1]
            cos_o = torch.cos(model.path_integrator.orientation_angles)
            sin_o = torch.sin(model.path_integrator.orientation_angles)
            d_block = dx.unsqueeze(-1) * cos_o + dy.unsqueeze(-1) * sin_o  # (B,L,H,M,O)
            cum = torch.cumsum(d_block, dim=1)
            omega = model.path_integrator.omega.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            angles = cum * omega                          # (B, L, H, M, O)
            theta = angles.reshape(angles.shape[0], angles.shape[1],
                                   angles.shape[2], -1)[0]   # (L, H, M*O)
        else:
            x = model.token_emb(tt[:, :-1])
            delta = model.action_to_lie(x)
            cum = torch.cumsum(delta, dim=1)
            theta = (cum * model.path_integrator.omega.unsqueeze(0).unsqueeze(0))[0]

        cos_theta = torch.cos(theta).cpu().numpy()  # (L, H, NB)
        # L = 2T - 1; trajectory has T (action,obs) pairs at indices 0..2T-1.
        # Position at token index t is visited[t//2] effectively — both action
        # and observation tokens correspond to the agent being AT that cell.
        L = cos_theta.shape[0]
        for t in range(L):
            cell = visited[t // 2]  # both action token (2k) and obs token (2k+1) are at cell visited[k]
            x, y = cell
            sums[:, x, y] += cos_theta[t].reshape(-1)
            counts[x, y] += 1

    visited_mask = counts > 0
    rate = np.zeros_like(sums)
    rate[:, visited_mask] = sums[:, visited_mask] / counts[visited_mask]
    return rate, counts


def plot_rate_maps(variant_rates, out_path, n_blocks_to_show=8):
    """Plot rate maps for several variants × several blocks."""
    variants = list(variant_rates.keys())
    n_var = len(variants)
    fig, axes = plt.subplots(n_var, n_blocks_to_show,
                             figsize=(n_blocks_to_show * 1.4, n_var * 1.6))
    if n_var == 1:
        axes = axes[None, :]

    for i, v in enumerate(variants):
        rate = variant_rates[v]
        if rate is None:
            for j in range(n_blocks_to_show):
                axes[i, j].axis("off")
            axes[i, 0].text(0.5, 0.5, f"{v}\n(no path int.)",
                            transform=axes[i, 0].transAxes, ha="center", va="center")
            continue
        # pick blocks that span the frequency spectrum (low, mid, high)
        n_state = rate.shape[0]
        idxs = np.linspace(0, n_state - 1, n_blocks_to_show, dtype=int)
        for j, idx in enumerate(idxs):
            ax = axes[i, j]
            ax.imshow(rate[idx], cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(v, fontsize=10)
            if i == 0:
                ax.set_title(f"block {idx}", fontsize=8)
    fig.suptitle("Spatial rate maps: cos(θ̂) by cell, per path-integrator block",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  wrote {out_path}", file=sys.stderr)
    plt.close(fig)


def grid_score(rate_map):
    """Spatial autocorrelation grid score (Sargolini et al. 2006).

    Computes the autocorrelation of the rate map, masks the central
    peak, and looks for hexagonal periodicity by comparing rotational
    symmetry at 60°/120° (peaks) vs 30°/90°/150° (troughs).

    Returns float in [-2, 2]; > 0.3 typically considered grid-like.
    """
    from scipy.signal import correlate2d
    from scipy.ndimage import rotate as nd_rotate

    rm = rate_map - rate_map.mean()
    ac = correlate2d(rm, rm, mode="same")
    # Normalize
    ac = ac / (ac.max() + 1e-12)

    # Mask central blob (small radius around center)
    cx, cy = ac.shape[0] // 2, ac.shape[1] // 2
    Y, X = np.ogrid[:ac.shape[0], :ac.shape[1]]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask_inner = r < 5
    mask_outer = r > min(ac.shape) // 3
    valid = ~mask_inner & ~mask_outer

    # Compute correlation under 60°, 120° rotations (should be high for grid)
    # vs 30°, 90°, 150° (should be lower)
    def corr_at(angle):
        rot = nd_rotate(ac, angle, reshape=False, order=1)
        a, b = ac[valid], rot[valid]
        return float(np.corrcoef(a, b)[0, 1])

    peaks = (corr_at(60) + corr_at(120)) / 2
    troughs = (corr_at(30) + corr_at(90) + corr_at(150)) / 3
    return peaks - troughs


# ----------------------------------------------------------------------
# Test B — R_t at landmark vs non-landmark tokens
# ----------------------------------------------------------------------

@torch.no_grad()
def collect_R_distribution(model, env, T=512, n_trials=40, device="cuda"):
    """For Level 1.5 variants only: collect log-R_t at landmark vs non-landmark
    observation positions.

    Returns (log_R_at_landmark, log_R_at_nonlandmark, log_R_at_blank) lists.
    """
    if not hasattr(model, "inekf"):
        return None, None, None

    log_R_lm = []
    log_R_obs = []
    log_R_blank = []

    for _ in range(n_trials):
        tokens, _, _ = env.generate_trajectory(T)
        visited = env.visited_locations
        tt = tokens.unsqueeze(0).to(device)
        try:
            _ = model(tt[:, :-1])
        except Exception:
            continue
        # last_R: (B, L, H, NB)
        if not hasattr(model, "last_R") or model.last_R is None:
            return None, None, None
        # average across heads × blocks for a per-token log-R scalar
        R = model.last_R[0]                       # (L, H, NB)
        log_R_per_tok = torch.log(R).mean(dim=(1, 2)).cpu().numpy()  # (L,)
        # Aligned with input tokens 0..L-1; classify each token by what it is.
        # Token index 2k = action at step k (cell visited[k]); index 2k+1 = obs.
        # Landmark detection: env.obs_map at cell tells us, but landmark IDs
        # are >= n_obs_types. We can also check the unified token directly.
        toks = tt[0, :len(log_R_per_tok)].cpu().numpy()
        for i, tok in enumerate(toks):
            # Determine whether this is an obs token (odd index) and what kind.
            # Vocab layout (see environment.py):
            #   actions: 0..N_ACTIONS-1 (=0..3)
            #   aliased obs: obs_offset + (0..n_obs_types-1)         (=4..19)
            #   blank: obs_offset + n_obs_types                      (=20)
            #   landmarks: obs_offset + (n_obs_types+1..n_landmarks) (=21..)
            if i % 2 == 0:
                continue  # action token, skip
            obs_id = tok - env.obs_offset
            if obs_id < 0:
                continue
            if obs_id == env.blank_token:
                log_R_blank.append(log_R_per_tok[i])
            elif obs_id < env.n_obs_types:
                log_R_obs.append(log_R_per_tok[i])
            else:
                # landmark (obs_id > n_obs_types)
                log_R_lm.append(log_R_per_tok[i])

    return np.asarray(log_R_lm), np.asarray(log_R_obs), np.asarray(log_R_blank)


def plot_R_landmark(by_variant, out_path):
    """Box/violin of log-R at landmark vs non-landmark obs vs blank."""
    variants = [v for v in by_variant if by_variant[v] is not None
                and by_variant[v][0] is not None]
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(variants)), 5))

    width = 0.25
    x = np.arange(len(variants))

    means_lm = [by_variant[v][0].mean() if len(by_variant[v][0]) > 0 else np.nan
                for v in variants]
    means_obs = [by_variant[v][1].mean() if len(by_variant[v][1]) > 0 else np.nan
                 for v in variants]
    means_blank = [by_variant[v][2].mean() if len(by_variant[v][2]) > 0 else np.nan
                   for v in variants]
    stds_lm = [by_variant[v][0].std() if len(by_variant[v][0]) > 0 else 0
               for v in variants]
    stds_obs = [by_variant[v][1].std() if len(by_variant[v][1]) > 0 else 0
                for v in variants]
    stds_blank = [by_variant[v][2].std() if len(by_variant[v][2]) > 0 else 0
                  for v in variants]

    ax.bar(x - width, means_lm, width, yerr=stds_lm, capsize=3,
           color="#43A047", label="Landmark tokens",
           edgecolor="black", linewidth=0.8)
    ax.bar(x, means_obs, width, yerr=stds_obs, capsize=3,
           color="#FFA000", label="Aliased obs (16 types)",
           edgecolor="black", linewidth=0.8)
    ax.bar(x + width, means_blank, width, yerr=stds_blank, capsize=3,
           color="#9E9E9E", label="Blank tokens",
           edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylabel(r"$\log R_t$ (mean ± std)", fontsize=11)
    ax.set_title("Per-token measurement noise $R_t$ by token type\n"
                 "(low $R_t$ = high informativeness = sharp posterior)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"  wrote {out_path}", file=sys.stderr)
    plt.close(fig)


# ----------------------------------------------------------------------
# Test C — ω frequency module structure (vs Stensola 2012 √2 ratio)
# ----------------------------------------------------------------------

def plot_omega_modules(by_variant, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ms = [(0.7, ".") , (0.95, "x"), (1.0, "o"), (1.0, "^"), (1.0, "s"), (1.0, "D")]
    for i, (variant, omega) in enumerate(by_variant.items()):
        if omega is None:
            continue
        flat = omega.flatten()
        flat_sorted = np.sort(flat)[::-1]   # high → low
        alpha, marker = ms[i % len(ms)]
        ax.semilogy(np.arange(len(flat_sorted)), flat_sorted,
                    "-" + marker, markersize=4, alpha=alpha,
                    label=variant, linewidth=1.2)
    ax.set_xlabel("Frequency block index (sorted high → low)", fontsize=11)
    ax.set_ylabel(r"$\omega$ (log scale)", fontsize=11)
    ax.set_title("Trained ω frequency spectrum (compare to grid-cell modules,\n"
                 "Stensola et al. 2012 — discrete √2-ratio spacings expected)",
                 fontsize=11)
    # Reference line: log-spaced from 2π to 2π/64, the geometric init
    omega_init = 2 * np.pi * (1/64) ** (np.linspace(0, 1, 32))
    ax.semilogy(np.arange(32)[::-1], omega_init, "--", color="black",
                alpha=0.4, label="Geometric init (untrained)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"  wrote {out_path}", file=sys.stderr)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--variants", nargs="+",
                   default=["Vanilla", "Level1", "Level15", "Level15EM",
                            "VanillaEM", "MambaLike"])
    p.add_argument("--seed", type=int, default=0,
                   help="model seed to load (only one needed for these analyses)")
    p.add_argument("--config-rate-maps", default="clean",
                   help="config to use for rate maps (Test A)")
    p.add_argument("--config-R", default="lm200",
                   help="config to use for R-at-landmark analysis (Test B)")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--test-seed", type=int, default=12345,
                   help="env seed for evaluation (fresh map)")
    p.add_argument("--output-md", default="HIPPOCAMPAL_ANALYSIS.md")
    p.add_argument("--output-figures", default="paper_figures")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    runs = Path(args.runs_dir)
    figs = Path(args.output_figures); figs.mkdir(parents=True, exist_ok=True)
    md = []

    md.append("# Hippocampal / Grid-Cell Correspondence Analysis\n")
    md.append("Three tests connecting trained MapFormer representations to "
              "known properties of place / grid / boundary cells in "
              "entorhinal cortex and hippocampus.\n")

    # ----- Test A: Rate maps -----
    print("[Test A] Rate maps — clean config", file=sys.stderr)
    md.append("## Test A — Spatial rate maps\n")
    rate_by_variant = {}
    grid_scores = {}
    for variant in args.variants:
        ckpt = runs / f"{variant}_{args.config_rate_maps}" / f"seed{args.seed}" / f"{variant}.pt"
        if not ckpt.exists():
            print(f"  [skip] {variant}: no ckpt", file=sys.stderr)
            rate_by_variant[variant] = None
            continue
        try:
            model, cfg = build_model(variant, ckpt, args.device)
        except Exception as e:
            print(f"  [skip] {variant}: {e}", file=sys.stderr)
            rate_by_variant[variant] = None
            continue
        env = GridWorld(
            size=cfg.get("grid_size", 64),
            n_obs_types=cfg.get("n_obs_types", 16),
            p_empty=cfg.get("p_empty", 0.5),
            n_landmarks=cfg.get("n_landmarks", 0),
            seed=args.test_seed,
        )
        rate, counts = compute_rate_maps(model, env, args.T, args.n_trials,
                                         args.device)
        rate_by_variant[variant] = rate
        if rate is not None:
            # Compute grid scores for several blocks; report best
            scores = []
            for j in range(rate.shape[0]):
                if rate[j].std() > 0.01:
                    try:
                        scores.append(grid_score(rate[j]))
                    except Exception:
                        pass
            if scores:
                grid_scores[variant] = (np.mean(scores), np.max(scores))
        print(f"  done {variant}", file=sys.stderr)

    plot_rate_maps(rate_by_variant, figs / "fig7_rate_maps.png")

    md.append("Each path-integrator block is shown as a heatmap of cos(θ̂) "
              "by (x, y), averaged over a fresh-environment trajectory. "
              "Real grid cells produce hexagonal periodic patterns; real "
              "place cells produce single-peak fields.\n")
    md.append("![Rate maps](paper_figures/fig7_rate_maps.png)\n")
    if grid_scores:
        md.append("**Grid score (Sargolini et al. 2006 — hexagonal "
                  "autocorrelation): higher = more grid-like.**\n")
        md.append("| Variant | mean across blocks | max across blocks |")
        md.append("|---|---|---|")
        for v, (m, mx) in grid_scores.items():
            md.append(f"| {v} | {m:+.3f} | {mx:+.3f} |")
        md.append("")

    # ----- Test B: R_t at landmark vs non-landmark -----
    print("\n[Test B] R_t at landmark tokens — lm200 config", file=sys.stderr)
    md.append("## Test B — Per-token measurement noise $R_t$ at landmarks\n")
    md.append("Hypothesis: Level 1.5's R_t head should learn to be **small** "
              "at landmark tokens (sharp posterior, high informativeness) "
              "and **large** at aliased / blank tokens (broad posterior, "
              "low informativeness). This mirrors the firing pattern of "
              "boundary cells (Solstad et al. 2008) and object cells "
              "(Lever et al. 2009) in entorhinal cortex.\n")

    R_by_variant = {}
    for variant in args.variants:
        if variant not in ("Level15", "Level15EM"):
            continue
        ckpt = runs / f"{variant}_{args.config_R}" / f"seed{args.seed}" / f"{variant}.pt"
        if not ckpt.exists():
            print(f"  [skip] {variant}: no ckpt", file=sys.stderr)
            continue
        try:
            model, cfg = build_model(variant, ckpt, args.device)
        except Exception as e:
            print(f"  [skip] {variant}: {e}", file=sys.stderr)
            continue
        env = GridWorld(
            size=cfg.get("grid_size", 64),
            n_obs_types=cfg.get("n_obs_types", 16),
            p_empty=cfg.get("p_empty", 0.5),
            n_landmarks=cfg.get("n_landmarks", 200),
            seed=args.test_seed,
        )
        log_R_lm, log_R_obs, log_R_blank = collect_R_distribution(
            model, env, args.T, args.n_trials, args.device
        )
        R_by_variant[variant] = (log_R_lm, log_R_obs, log_R_blank)
        if log_R_lm is not None and len(log_R_lm) > 0:
            print(f"  {variant}: log_R(landmark)={log_R_lm.mean():.3f}, "
                  f"log_R(obs)={log_R_obs.mean():.3f}, "
                  f"log_R(blank)={log_R_blank.mean():.3f}", file=sys.stderr)

    if R_by_variant:
        plot_R_landmark(R_by_variant, figs / "fig8_R_landmark.png")
        md.append("![R at landmarks](paper_figures/fig8_R_landmark.png)\n")
        md.append("| Variant | landmark $\\langle\\log R\\rangle$ | "
                  "aliased obs $\\langle\\log R\\rangle$ | "
                  "blank $\\langle\\log R\\rangle$ |")
        md.append("|---|---|---|---|")
        for v, (a, b, c) in R_by_variant.items():
            la = f"{a.mean():+.3f}" if a is not None and len(a) > 0 else "—"
            lb = f"{b.mean():+.3f}" if b is not None and len(b) > 0 else "—"
            lc = f"{c.mean():+.3f}" if c is not None and len(c) > 0 else "—"
            md.append(f"| {v} | {la} | {lb} | {lc} |")
        md.append("")
        md.append("**Predicted ordering:** "
                  "landmark < aliased obs < blank (smaller R = more informative).")
        md.append("If observed, this is direct quantitative correspondence to "
                  "boundary/object cell firing patterns.\n")

    # ----- Test C: ω frequency modules -----
    print("\n[Test C] ω frequency spectrum", file=sys.stderr)
    md.append("## Test C — ω frequency spectrum (grid-cell modules)\n")
    md.append("Stensola et al. (2012, *Nature*) showed grid-cell modules in "
              "entorhinal cortex have spacings following a roughly √2 "
              "geometric ratio. MapFormer's geometric initialisation gives "
              "ω with similar log-uniform structure; we plot the trained ω "
              "to see whether training preserves or breaks this.\n")

    omega_by_variant = {}
    for variant in args.variants:
        if variant in ("VanillaEM",):
            ckpt_path = runs / f"{variant}_{args.config_rate_maps}" / f"seed{args.seed}" / f"{variant}.pt"
        else:
            ckpt_path = runs / f"{variant}_{args.config_rate_maps}" / f"seed{args.seed}" / f"{variant}.pt"
        if not ckpt_path.exists():
            continue
        try:
            model, _ = build_model(variant, ckpt_path, args.device)
        except Exception:
            continue
        if hasattr(model, "path_integrator"):
            omega_by_variant[variant] = model.path_integrator.omega.detach().cpu().numpy()

    if omega_by_variant:
        plot_omega_modules(omega_by_variant, figs / "fig9_omega_modules.png")
        md.append("![ω modules](paper_figures/fig9_omega_modules.png)\n")
        md.append("Solid lines: trained ω per variant (high → low). "
                  "Dashed: untrained geometric init for reference.\n")

    md.append("\n---\n*Auto-generated by `hippocampal_analysis.py`.*\n")
    Path(args.output_md).write_text("\n".join(md))
    print(f"\nwrote {args.output_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
