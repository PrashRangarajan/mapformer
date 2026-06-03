#!/usr/bin/env python
"""
make_inekf_report_figure.py
---------------------------
A simplified length-generalization figure for the generals report: only the
uncorrected MapFormer and the InEKF-corrected MapFormer (plus one optional
baseline), with report-friendly names instead of the internal "Level15" etc.

Reuses the exact eval pipeline from make_paper_figures.py, so the numbers are
identical to fig2_length_gen.png -- just fewer curves and clearer labels.

Run from the mapformer repo root (same place fig2 is generated):

    python make_inekf_report_figure.py --runs_dir <RUNS_DIR> --out inekf_results.png

Then copy inekf_results.png into the report:
    cp inekf_results.png ../Hierarchical_Transformers/UW-Generals-Document/figures/

Edit SUBSET / LABELS below to change which models appear and how they are named.
"""
import argparse
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Reuse the faithful eval pipeline (same functions fig2 uses)
from make_paper_figures import build_model, eval_per_cell, GridWorld

# --- choose which internal variants to show, and their report names ----------
# Keep this small: the point is a clean 2-3 line comparison, not the full ablation.
SUBSET = ["Vanilla", "Level15"]            # add "RoPE" or "MambaLike" for a non-map baseline
LABELS = {
    "Vanilla":   "MapFormer (open-loop)",
    "Level15":   "MapFormer + InEKF",
    "RoPE":      "RoPE Transformer",
    "MambaLike": "Mamba-like SSM",
    "Level1":    "MapFormer + InEKF (basic)",
    "Level2":    "MapFormer + InEKF (Level 2)",
}
COLORS = {
    "Vanilla":   "#808080",
    "Level15":   "#1b7837",
    "RoPE":      "#606060",
    "MambaLike": "#FF7043",
    "Level1":    "#2196F3",
    "Level2":    "#FFA000",
}
CONFIG = "lm200"          # "lm200" (200 landmarks) or "clean"; lm200 is the cognitive-map setting
LENGTHS = [128, 256, 512, 1024, 2048]
SEEDS = [0, 1, 2]
# -----------------------------------------------------------------------------


def _audit_checkpoints(runs_dir: Path):
    """Print which (variant, seed) checkpoints exist before any eval runs."""
    print(f"Checkpoint audit for runs_dir = {runs_dir}, config = {CONFIG}, seeds = {SEEDS}")
    missing = []
    for variant in SUBSET:
        present = []
        for seed in SEEDS:
            ckpt = runs_dir / f"{variant}_{CONFIG}" / f"seed{seed}" / f"{variant}.pt"
            if ckpt.exists():
                present.append(seed)
            else:
                missing.append(str(ckpt))
        status = ",".join(map(str, present)) if present else "(none)"
        print(f"  {variant}_{CONFIG}: seeds present = {status}")
    if missing:
        print("Missing checkpoints (will be skipped in aggregation):")
        for p in missing:
            print(f"  {p}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out", default="inekf_results.png")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    _audit_checkpoints(runs_dir)

    n_lm = 200 if CONFIG == "lm200" else 0
    fig, ax = plt.subplots(figsize=(6, 4.2))

    for variant in SUBSET:
        means, stds, Ts = [], [], []
        for T in LENGTHS:
            vals = []
            for seed in SEEDS:
                ckpt = runs_dir / f"{variant}_{CONFIG}" / f"seed{seed}" / f"{variant}.pt"
                if not ckpt.exists():
                    continue
                env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                                n_landmarks=n_lm, seed=seed + 1000)
                m = build_model(variant, ckpt, env.unified_vocab_size)
                n_trials = 50 if T >= 2048 else 200
                per = eval_per_cell(m, env, T, n_trials, seed=seed + 2000)
                if per["overall"][0] is not None:
                    vals.append(per["overall"][0])
                del m
                torch.cuda.empty_cache()
            if vals:
                Ts.append(T)
                means.append(st.mean(vals))
                stds.append(st.pstdev(vals) if len(vals) > 1 else 0)
                print(f"  {variant} T={T}: mean={means[-1]:.3f} ± {stds[-1]:.3f}  (n={len(vals)})")
        if not Ts:
            print(f"  (no checkpoints found for {variant}_{CONFIG}; skipping)")
            continue
        ax.errorbar(Ts, means, yerr=stds, marker="o", capsize=3,
                    label=LABELS.get(variant, variant), color=COLORS.get(variant))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("Overall revisit accuracy")
    ax.set_title(f"Length generalization ({'200 landmarks' if CONFIG == 'lm200' else 'no landmarks'})")
    ax.axvline(x=128, linestyle=":", color="gray", alpha=0.5, label="train length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
