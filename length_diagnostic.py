#!/usr/bin/env python3
"""Length-generalization diagnostic: why does Level15PC_NoBypass collapse at T=512?

Compares Level15, Level15PC, and Level15PC_NoBypass on a single OOD lm200
trajectory at T=128 and T=512. For each variant, extracts:

  - theta_path[t]:  cumulative path-integrated angle
  - theta_hat[t]:   InEKF-corrected angle
  - d_t:            correction d_t = theta_hat - theta_path
  - R_t, K_t:       per-token InEKF parameters

Looks for:
  - θ̂ drift: |θ̂[t] - θ̂_path[t]| growing over time?
  - K saturation: distribution of K — does it spike at certain positions?
  - The position where the model "loses track": when does θ̂ start diverging
    from a sensible value?

Output:
  - LENGTH_DIAGNOSTIC.md with per-variant statistics
  - paper_figures/fig_length_diag_{theta_drift,d_dist,K_dist}.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_level15_pc import MapFormerWM_Level15PC
from mapformer.model_level15_pc_v2 import MapFormerWM_Level15PC_NoBypass


VARIANT_CLS = {
    "Level15":           MapFormerWM_Level15InEKF,
    "Level15PC":         MapFormerWM_Level15PC,
    "Level15PC_NoBypass": MapFormerWM_Level15PC_NoBypass,
}


def build(variant, ckpt_path, device="cuda"):
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
def extract_trajectory_state(model, env, T, n_trials=5, device="cuda"):
    """For each trajectory, capture per-step InEKF tensors.

    Returns dict with keys:
        theta_path: (n_trials, L, H, NB)
        theta_hat:  (n_trials, L, H, NB)
        d:          (n_trials, L, H, NB) — d_t = theta_hat - theta_path
        R:          (n_trials, L, H, NB)
        K:          (n_trials, L, H, NB)
    """
    out = {k: [] for k in ["theta_path", "theta_hat", "d", "R", "K"]}
    for _ in range(n_trials):
        tokens, _, _ = env.generate_trajectory(T)
        tt = tokens.unsqueeze(0).to(device)
        try:
            _ = model(tt[:, :-1])
        except Exception:
            continue
        out["theta_path"].append(model.last_theta_path[0].cpu().numpy())
        out["theta_hat"].append(model.last_theta_hat[0].cpu().numpy())
        out["d"].append((model.last_theta_hat[0] - model.last_theta_path[0]).cpu().numpy())
        out["R"].append(model.last_R[0].cpu().numpy())
        out["K"].append(model.last_K[0].cpu().numpy())
    return {k: np.stack(out[k]) if out[k] else np.zeros((0,)) for k in out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--config", default="lm200")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-seed", type=int, default=12345)
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--output-md", default="LENGTH_DIAGNOSTIC.md")
    p.add_argument("--output-figs", default="paper_figures")
    args = p.parse_args()

    runs = Path(args.runs_dir)
    figs = Path(args.output_figs); figs.mkdir(parents=True, exist_ok=True)
    n_lm = 200 if args.config == "lm200" else 0

    # Collect for each variant at both T=128 and T=512
    data = {}
    for variant in ["Level15", "Level15PC", "Level15PC_NoBypass"]:
        ckpt = runs / f"{variant}_{args.config}" / f"seed{args.seed}" / f"{variant}.pt"
        if not ckpt.exists():
            print(f"[skip] {variant}: no ckpt", file=sys.stderr); continue
        m, cfg = build(variant, ckpt)
        env = GridWorld(
            size=cfg.get("grid_size", 64),
            n_obs_types=cfg.get("n_obs_types", 16),
            p_empty=cfg.get("p_empty", 0.5),
            n_landmarks=cfg.get("n_landmarks", n_lm),
            seed=args.test_seed,
        )
        np.random.seed(args.test_seed); torch.manual_seed(args.test_seed)
        data[variant] = {
            "T128": extract_trajectory_state(m, env, 128, args.n_trials),
            "T512": extract_trajectory_state(m, env, 512, args.n_trials),
        }
        del m; torch.cuda.empty_cache()
        print(f"  {variant}: collected", file=sys.stderr)

    # ----- Compute statistics -----
    md = ["# Length-generalization diagnostic\n"]
    md.append(f"Config: **{args.config}**, model seed {args.seed}, "
              f"OOD env seed {args.test_seed}, {args.n_trials} trajectories per length.\n")
    md.append("All quantities averaged across trajectories, heads, blocks. "
              "`d_t = theta_hat - theta_path` is the correction magnitude.\n")
    md.append("## Statistics by variant and length\n")
    md.append("| Variant | T | mean(|θ̂|) | std(|θ̂|) | mean(|d|) | mean(K) | "
              "mean(K, last 25%) | mean(R) | mean(R, last 25%) |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    for variant, vdata in data.items():
        for T_label, d in vdata.items():
            if d["theta_hat"].size == 0:
                continue
            theta_hat = d["theta_hat"]  # (n_trials, L, H, NB)
            d_corr = d["d"]
            K = d["K"]
            R = d["R"]
            L = theta_hat.shape[1]
            last_q = max(L * 3 // 4, 0)  # last 25% of positions

            mean_abs_theta = float(np.abs(theta_hat).mean())
            std_abs_theta  = float(np.abs(theta_hat).std())
            mean_abs_d     = float(np.abs(d_corr).mean())
            mean_K_all     = float(K.mean())
            mean_K_last    = float(K[:, last_q:].mean())
            mean_R_all     = float(R.mean())
            mean_R_last    = float(R[:, last_q:].mean())
            md.append(
                f"| {variant} | {T_label} | "
                f"{mean_abs_theta:.3f} | {std_abs_theta:.3f} | "
                f"{mean_abs_d:.3f} | "
                f"{mean_K_all:.3f} | {mean_K_last:.3f} | "
                f"{mean_R_all:.3f} | {mean_R_last:.3f} |"
            )

    md.append("\n## Per-position drift (does θ̂ wander?)\n")
    md.append("If a variant's θ̂ standard deviation grows monotonically over "
              "trajectory positions, it's drifting. Compare growth at T=512 "
              "vs T=128.\n")

    # Plot 1: |theta_hat| std over positions, for each variant at T=512
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, T_label in zip(axes, ["T128", "T512"]):
        for variant, vdata in data.items():
            if T_label not in vdata or vdata[T_label]["theta_hat"].size == 0:
                continue
            theta_hat = vdata[T_label]["theta_hat"]   # (n_trials, L, H, NB)
            # std across (heads × blocks × trials) per position
            std_per_pos = np.abs(theta_hat).std(axis=(0, 2, 3))
            ax.plot(std_per_pos, label=variant, linewidth=1.5)
        ax.set_title(f"std(|θ̂|) per position, {T_label}", fontsize=11)
        ax.set_xlabel("Token position", fontsize=10)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("std(|θ̂|)", fontsize=10)
    # Third panel: |d_t| per position at T=512
    ax = axes[2]
    for variant, vdata in data.items():
        if "T512" not in vdata or vdata["T512"]["d"].size == 0:
            continue
        d = vdata["T512"]["d"]
        mean_per_pos = np.abs(d).mean(axis=(0, 2, 3))
        ax.plot(mean_per_pos, label=variant, linewidth=1.5)
    ax.set_title("mean |correction d_t| per position, T=512", fontsize=11)
    ax.set_xlabel("Token position", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "fig_length_diag_drift.png", dpi=150, bbox_inches="tight")
    md.append(f"![Drift plots]({figs.name}/fig_length_diag_drift.png)\n")
    plt.close(fig)

    # Plot 2: K distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, T_label in zip(axes, ["T128", "T512"]):
        for variant, vdata in data.items():
            if T_label not in vdata or vdata[T_label]["K"].size == 0:
                continue
            K = vdata[T_label]["K"].flatten()
            ax.hist(K, bins=40, alpha=0.45, label=variant, density=True)
        ax.set_title(f"K distribution, {T_label}", fontsize=11)
        ax.set_xlabel("K_t (Kalman gain)", fontsize=10)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "fig_length_diag_K.png", dpi=150, bbox_inches="tight")
    md.append(f"![K histograms]({figs.name}/fig_length_diag_K.png)\n")
    plt.close(fig)

    # Plot 3: log_R distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, T_label in zip(axes, ["T128", "T512"]):
        for variant, vdata in data.items():
            if T_label not in vdata or vdata[T_label]["R"].size == 0:
                continue
            log_R = np.log(np.maximum(vdata[T_label]["R"].flatten(), 1e-9))
            ax.hist(log_R, bins=40, alpha=0.45, label=variant, density=True)
        ax.set_title(f"log_R distribution, {T_label}", fontsize=11)
        ax.set_xlabel("log R_t", fontsize=10)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "fig_length_diag_R.png", dpi=150, bbox_inches="tight")
    md.append(f"![log_R histograms]({figs.name}/fig_length_diag_R.png)\n")
    plt.close(fig)

    md.append("\n*Auto-generated by `length_diagnostic.py`.*\n")
    Path(args.output_md).write_text("\n".join(md))
    print(f"wrote {args.output_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
