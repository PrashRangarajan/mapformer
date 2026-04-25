#!/usr/bin/env python3
"""Calibration analysis: ECE + reliability diagrams.

Expected Calibration Error (ECE) and reliability diagrams are standard
tools for quantifying "does the model know what it doesn't know?" For
our work, they make the NLL advantage of the Kalman/PC variants visible
in a way that's intuitive to readers.

Usage:
  python3 -m mapformer.calibration_analysis \
      --runs-dir runs/ --config clean \
      --output paper_figures/calibration.png
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF
from mapformer.model_inekf_level2 import MapFormerWM_Level2InEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_ablations import ABLATIONS
from mapformer.model_baselines_extra import EXTRA_BASELINES

VARIANT_CLS = {
    "Vanilla": MapFormerWM, "VanillaEM": MapFormerEM,
    "Level1": MapFormerWM_ParallelInEKF,
    "Level15": MapFormerWM_Level15InEKF,
    "Level15EM": MapFormerEM_Level15InEKF,
    "Level2": MapFormerWM_Level2InEKF,
    "PC": MapFormerWM_PredictiveCoding, "RoPE": MapFormerWM_RoPE,
    **ABLATIONS,
    **EXTRA_BASELINES,
}


def collect_predictions(model, env, T, n_trials, seed):
    """Return arrays of (confidence, correct) for each revisit prediction."""
    torch.manual_seed(seed); np.random.seed(seed)
    confs = []; corrects = []
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, om, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            logits = model(tt[:, :-1])
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            tgts = tt[0, 1:]; mask = rm[1:].cuda()
            if mask.sum() == 0: continue
            for i in mask.nonzero(as_tuple=True)[0].tolist():
                confs.append(conf[0, i].item())
                corrects.append(int(pred[0, i].item() == tgts[i].item()))
    return np.array(confs), np.array(corrects)


def compute_ece(confs, corrects, n_bins=15):
    """Expected Calibration Error: average weighted gap between confidence
    and accuracy per bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confs)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            bins.append((lo, hi, 0, 0, 0))
            continue
        bin_conf = confs[mask].mean()
        bin_acc = corrects[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(bin_conf - bin_acc)
        bins.append((lo, hi, bin_conf, bin_acc, mask.sum()))
    return ece, bins


def build_model(variant, ckpt_path, vocab_size):
    c = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config", default="clean")
    parser.add_argument("--variants", nargs="+",
                        default=["Vanilla", "VanillaEM", "RoPE", "Level1",
                                 "Level15", "Level15EM", "PC", "LSTM",
                                 "MambaLike"])
    parser.add_argument("--seed", type=int, default=0,
                        help="Which training seed to analyze (single seed, for the figure)")
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--output", default="paper_figures/calibration.png")
    args = parser.parse_args()

    n_lm = 200 if args.config == "lm200" else 0
    runs = Path(args.runs_dir)

    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                    n_landmarks=n_lm, seed=args.seed + 1000)  # fresh env

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    results = {}
    for variant in args.variants:
        ckpt = runs / f"{variant}_{args.config}" / f"seed{args.seed}" / f"{variant}.pt"
        if not ckpt.exists():
            print(f"skipping {variant}: no checkpoint")
            continue
        m = build_model(variant, ckpt, env.unified_vocab_size)
        confs, corrects = collect_predictions(m, env, args.T, args.n_trials, seed=args.seed + 2000)
        ece, bins = compute_ece(confs, corrects)
        results[variant] = {"ece": ece, "bins": bins, "confs": confs, "corrects": corrects}
        del m; torch.cuda.empty_cache()

    # Panel 1: Reliability diagram
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect calibration")
    for variant, r in results.items():
        bin_confs = [b[2] for b in r["bins"] if b[4] > 0]
        bin_accs = [b[3] for b in r["bins"] if b[4] > 0]
        ax[0].plot(bin_confs, bin_accs, "o-", label=f"{variant} (ECE={r['ece']:.3f})", markersize=5)
    ax[0].set_xlabel("Predicted confidence")
    ax[0].set_ylabel("Empirical accuracy")
    ax[0].set_title(f"Reliability diagram ({args.config}, T={args.T}, seed={args.seed})")
    ax[0].legend(loc="best", fontsize=8)
    ax[0].grid(True, alpha=0.3)

    # Panel 2: ECE bar chart
    variants = list(results.keys())
    eces = [results[v]["ece"] for v in variants]
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    bars = ax[1].bar(variants, eces, color=colors)
    ax[1].set_ylabel("Expected Calibration Error")
    ax[1].set_title("ECE across variants (lower is better)")
    for bar, v in zip(bars, eces):
        ax[1].text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax[1].grid(True, alpha=0.3, axis="y")
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # Also print summary
    print("\nCalibration summary:")
    for v, r in sorted(results.items(), key=lambda x: x[1]["ece"]):
        print(f"  {v:>12s}: ECE = {r['ece']:.4f}  (n={len(r['confs'])})")


if __name__ == "__main__":
    main()
