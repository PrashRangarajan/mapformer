#!/usr/bin/env python3
"""Simplified length-generalization figure for the generals report.

Single-panel curve plot showing overall revisit accuracy vs. sequence length T
on lm200 (200-landmark) checkpoints. Reuses build_model / eval_per_cell /
GridWorld from make_paper_figures.py so the numbers exactly match
fig2_length_gen.png (same seed+1000 env-seed and seed+2000 eval-seed).

Default subset: Vanilla, Level15, MambaLike.
Edit SUBSET below to swap MambaLike for RoPE / LSTM, or pass --subset.
"""

import argparse
import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.make_paper_figures import build_model, eval_per_cell

# ---- configuration --------------------------------------------------------

SUBSET_DEFAULT = ["Vanilla", "Level15", "MambaLike"]
LENGTHS = [128, 256, 512, 1024, 2048]
SEEDS = [0, 1, 2]
TRAIN_T = 128

# Report-friendly labels.
LABELS = {
    "Vanilla":   "MapFormer (open-loop)",
    "Level15":   "MapFormer + InEKF",
    "RoPE":      "RoPE Transformer",
    "MambaLike": "Mamba-like SSM",
    "LSTM":      "LSTM",
}

# Colors picked for report (print-friendly, colorblind-aware).
COLORS = {
    "Vanilla":   "#888888",
    "Level15":   "#1b7e3f",
    "RoPE":      "#444444",
    "MambaLike": "#e07b1c",
    "LSTM":      "#a0522d",
}

MARKERS = {
    "Vanilla":   "o",
    "Level15":   "s",
    "RoPE":      "^",
    "MambaLike": "D",
    "LSTM":      "v",
}

# ---- pipeline -------------------------------------------------------------

def audit_checkpoints(runs_dir, variants, seeds):
    """Print which seeds are present per variant; return (variant, seed)
    pairs that exist."""
    print("Checkpoint audit (lm200):")
    available = []
    for v in variants:
        present, missing = [], []
        for s in seeds:
            ckpt = Path(runs_dir) / f"{v}_lm200" / f"seed{s}" / f"{v}.pt"
            if ckpt.exists():
                present.append(s); available.append((v, s))
            else:
                missing.append(str(ckpt))
        present_str = ",".join(str(x) for x in present) or "NONE"
        print(f"  {v}_lm200: seeds present = {present_str}")
        for m in missing:
            print(f"    MISSING: {m}")
    print()
    return available


def run_one(runs_dir, variant, seed, T):
    ckpt = Path(runs_dir) / f"{variant}_lm200" / f"seed{seed}" / f"{variant}.pt"
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                    n_landmarks=200, seed=seed + 1000)
    m = build_model(variant, ckpt, env.unified_vocab_size)
    n_trials = 50 if T >= 2048 else 200
    per = eval_per_cell(m, env, T, n_trials, seed=seed + 2000)
    acc, _ = per["overall"]
    del m
    torch.cuda.empty_cache()
    return acc


def collect(runs_dir, variants, seeds, lengths):
    """Returns {variant: {T: [acc per seed]}}."""
    out = {v: {T: [] for T in lengths} for v in variants}
    available = set((v, s) for v in variants for s in seeds
                    if (Path(runs_dir) / f"{v}_lm200" / f"seed{s}" / f"{v}.pt").exists())
    for variant in variants:
        for T in lengths:
            vals = []
            for seed in seeds:
                if (variant, seed) not in available:
                    continue
                acc = run_one(runs_dir, variant, seed, T)
                if acc is not None:
                    vals.append(acc)
            out[variant][T] = vals
            if vals:
                m = st.mean(vals)
                sd = st.pstdev(vals) if len(vals) > 1 else 0.0
                print(f"  {variant} T={T}: mean={m:.3f} +/- {sd:.3f} (n={len(vals)})")
            else:
                print(f"  {variant} T={T}: NO DATA")
    return out


def plot(results, variants, lengths, output, train_T=TRAIN_T):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for variant in variants:
        means, stds, ts = [], [], []
        for T in lengths:
            vals = results[variant][T]
            if not vals: continue
            means.append(st.mean(vals))
            stds.append(st.pstdev(vals) if len(vals) > 1 else 0.0)
            ts.append(T)
        if not ts: continue
        ax.errorbar(ts, means, yerr=stds,
                    marker=MARKERS.get(variant, "o"),
                    markersize=6, linewidth=1.8, capsize=3,
                    color=COLORS.get(variant, None),
                    label=LABELS.get(variant, variant))
    ax.axvline(x=train_T, linestyle=":", color="#666666", alpha=0.8, linewidth=1)
    ax.text(train_T * 1.07, 0.99, "train length", color="#666666",
            fontsize=9, ha="left", va="top", alpha=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(T) for T in lengths])
    ax.set_xlabel("Sequence length $T$")
    ax.set_ylabel("Overall revisit accuracy")
    ax.set_title("Length generalization with InEKF correction (200 landmarks)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nSaved {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", "--runs-dir", default="runs",
                    help="Path to runs/ (same as make_paper_figures.py).")
    ap.add_argument("--out", default="inekf_results.png",
                    help="Output PNG path.")
    ap.add_argument("--subset", nargs="+", default=SUBSET_DEFAULT,
                    help="Variants to plot. Default: Vanilla Level15 MambaLike.")
    args = ap.parse_args()

    print(f"runs_dir: {args.runs_dir}")
    print(f"subset:   {args.subset}")
    print(f"lengths:  {LENGTHS}")
    print(f"seeds:    {SEEDS}\n")

    audit_checkpoints(args.runs_dir, args.subset, SEEDS)

    print("Running eval...")
    results = collect(args.runs_dir, args.subset, SEEDS, LENGTHS)

    plot(results, args.subset, LENGTHS, args.out)


if __name__ == "__main__":
    main()
