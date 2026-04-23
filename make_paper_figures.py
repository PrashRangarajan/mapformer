#!/usr/bin/env python3
"""Generate paper-ready figures from the multi-seed experiment data.

Produces:
  fig1_landmark_bar.png    — Landmark accuracy bar chart (headline figure)
  fig2_length_gen.png      — Length generalization curves (bounded-error)
  fig3_nll_comparison.png  — NLL heatmap/bars across configs
  fig4_ablation_level15.png — Level 1.5 ablation bar chart
  fig5_scan_ops.png        — Parallelism diagram (schematic)

Each figure is self-contained with caption text. Use in paper main text
or supplementary as appropriate.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import statistics as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_parallel import MapFormerWM_ParallelInEKF
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level2 import MapFormerWM_Level2InEKF
from mapformer.model_predictive_coding import MapFormerWM_PredictiveCoding
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_ablations import ABLATIONS

VARIANT_CLS = {
    "Vanilla": MapFormerWM, "Level1": MapFormerWM_ParallelInEKF,
    "Level15": MapFormerWM_Level15InEKF, "Level2": MapFormerWM_Level2InEKF,
    "PC": MapFormerWM_PredictiveCoding, "RoPE": MapFormerWM_RoPE,
    **ABLATIONS,
}

# Consistent colors across figures
VARIANT_COLORS = {
    "Vanilla": "#808080",
    "RoPE":    "#606060",
    "Level1":  "#2196F3",
    "Level2":  "#FFA000",
    "Level15": "#43A047",
    "PC":      "#9C27B0",
    "L15_ConstR": "#81C784",
    "L15_NoMeas": "#66BB6A",
    "L15_NoCorr": "#4CAF50",
    "L15_DARE":   "#388E3C",
}


def build_model(variant, ckpt_path, vocab_size):
    c = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()


def eval_per_cell(model, env, T, n_trials, seed):
    """Return dict: {landmark, regular, blank, overall} -> (acc, nll)."""
    import torch.nn.functional as F
    torch.manual_seed(seed); np.random.seed(seed)
    cats = ["landmark", "regular", "blank", "overall"]
    correct = {c: 0 for c in cats}
    total = {c: 0 for c in cats}
    nll_sum = {c: 0.0 for c in cats}
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, om, rm = env.generate_trajectory(T)
            tt = tokens.unsqueeze(0).cuda()
            logits = model(tt[:, :-1])
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]; tgts = tt[0, 1:]; mask = rm[1:].cuda()
            revisit_idx = mask.nonzero(as_tuple=True)[0].tolist()
            for idx in revisit_idx:
                tgt = tgts[idx].item()
                pred = preds[idx].item()
                nll = -lp[0, idx, tgt].item()
                if tgt == env.unified_blank: cat = "blank"
                elif tgt >= env.first_landmark_unified: cat = "landmark"
                else: cat = "regular"
                correct[cat] += int(pred == tgt); total[cat] += 1; nll_sum[cat] += nll
                correct["overall"] += int(pred == tgt); total["overall"] += 1
                nll_sum["overall"] += nll
    return {c: (correct[c] / total[c] if total[c] else None,
                nll_sum[c] / total[c] if total[c] else None) for c in cats}


def aggregate(runs_dir, variants, config, seeds, T, n_trials):
    """Run eval across (variants × seeds) and aggregate into mean/std per cell type."""
    n_lm = 200 if config == "lm200" else 0
    results = {v: {c: [] for c in ["landmark", "regular", "blank", "overall"]}
               for v in variants}
    for variant in variants:
        for seed in seeds:
            ckpt = Path(runs_dir) / f"{variant}_{config}" / f"seed{seed}" / f"{variant}.pt"
            if not ckpt.exists(): continue
            env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=seed + 1000)
            m = build_model(variant, ckpt, env.unified_vocab_size)
            per_cell = eval_per_cell(m, env, T, n_trials, seed=seed + 2000)
            for cat, (acc, _) in per_cell.items():
                if acc is not None:
                    results[variant][cat].append(acc)
            del m; torch.cuda.empty_cache()
    return results


def fig1_landmark_bar(runs_dir, output):
    """Bar chart: overall acc and landmark acc across variants at lm200 config."""
    variants = ["Vanilla", "RoPE", "Level1", "PC", "Level15"]
    res = aggregate(runs_dir, variants, "lm200", [0, 1, 2], 128, 200)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    for i, metric in enumerate(["overall", "landmark"]):
        means = [st.mean(res[v][metric]) if res[v][metric] else 0 for v in variants]
        stds = [st.pstdev(res[v][metric]) if len(res[v][metric]) > 1 else 0 for v in variants]
        colors = [VARIANT_COLORS[v] for v in variants]
        bars = ax[i].bar(variants, means, yerr=stds, capsize=5, color=colors,
                          edgecolor="black", linewidth=0.5)
        ax[i].set_ylabel(f"Revisit {metric} accuracy at T=128")
        ax[i].set_title(f"{metric.title()} accuracy — landmarks ({'all cells' if metric == 'overall' else 'landmark cells only'})")
        ax[i].set_ylim(0, 1.05)
        ax[i].grid(True, alpha=0.3, axis="y")
        for bar, v, mean in zip(bars, variants, means):
            ax[i].text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                        f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


def fig2_length_gen(runs_dir, output):
    """Length-generalization curves: accuracy vs T, for clean and lm200 configs."""
    variants = ["Vanilla", "Level1", "Level15"]
    lengths = [128, 256, 512, 1024, 2048]
    seeds = [0, 1, 2]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    for col, config in enumerate(["clean", "lm200"]):
        n_lm = 200 if config == "lm200" else 0
        for variant in variants:
            means, stds = [], []
            for T in lengths:
                vals = []
                for seed in seeds:
                    ckpt = Path(runs_dir) / f"{variant}_{config}" / f"seed{seed}" / f"{variant}.pt"
                    if not ckpt.exists(): continue
                    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                                    n_landmarks=n_lm, seed=seed + 1000)
                    m = build_model(variant, ckpt, env.unified_vocab_size)
                    n_trials = 50 if T >= 2048 else 200
                    per = eval_per_cell(m, env, T, n_trials, seed=seed + 2000)
                    if per["overall"][0] is not None:
                        vals.append(per["overall"][0])
                    del m; torch.cuda.empty_cache()
                if vals:
                    means.append(st.mean(vals))
                    stds.append(st.pstdev(vals) if len(vals) > 1 else 0)
                else:
                    means.append(None); stds.append(0)
            valid = [(T, m, s) for T, m, s in zip(lengths, means, stds) if m is not None]
            if not valid: continue
            Ts = [x[0] for x in valid]; ms = [x[1] for x in valid]; ss = [x[2] for x in valid]
            ax[col].errorbar(Ts, ms, yerr=ss, marker="o", label=variant, color=VARIANT_COLORS[variant], capsize=3)
        ax[col].set_xscale("log", base=2)
        ax[col].set_xlabel("Sequence length T")
        ax[col].set_ylabel("Overall revisit accuracy")
        ax[col].set_title(f"Length generalization ({config})")
        ax[col].axvline(x=128, linestyle=":", color="gray", alpha=0.5, label="train length")
        ax[col].legend()
        ax[col].grid(True, alpha=0.3)
        ax[col].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


def fig4_ablation_level15(runs_dir, output):
    """Level 1.5 ablation bars at lm200 config."""
    variants = ["Level15", "L15_ConstR", "L15_NoMeas", "L15_NoCorr", "L15_DARE"]
    labels = ["L1.5 (full)", "Const R", "No measure", "No correction", "DARE Π"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for col, config in enumerate(["clean", "lm200"]):
        accs = []
        n_lm = 200 if config == "lm200" else 0
        env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=1000)
        for variant in variants:
            ckpt = Path(runs_dir) / f"{variant}_{config}" / f"seed0" / f"{variant}.pt"
            if not ckpt.exists():
                accs.append(0); continue
            m = build_model(variant, ckpt, env.unified_vocab_size)
            per = eval_per_cell(m, env, 128, 200, seed=2000)
            accs.append(per["overall"][0] if per["overall"][0] else 0)
            del m; torch.cuda.empty_cache()
        colors = [VARIANT_COLORS[v] for v in variants]
        bars = ax[col].bar(labels, accs, color=colors, edgecolor="black", linewidth=0.5)
        ax[col].set_ylabel("Overall revisit accuracy at T=128")
        ax[col].set_title(f"Level 1.5 ablations ({config})")
        ax[col].set_ylim(0, 1.05)
        ax[col].grid(True, alpha=0.3, axis="y")
        plt.setp(ax[col].xaxis.get_majorticklabels(), rotation=20, ha="right")
        for bar, v in zip(bars, accs):
            ax[col].text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                          f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default="paper_figures")
    parser.add_argument("--figures", nargs="+",
                        default=["landmark_bar", "length_gen", "ablation"],
                        choices=["landmark_bar", "length_gen", "ablation"])
    args = parser.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if "landmark_bar" in args.figures:
        fig1_landmark_bar(args.runs_dir, out_dir / "fig1_landmark_bar.png")
    if "length_gen" in args.figures:
        fig2_length_gen(args.runs_dir, out_dir / "fig2_length_gen.png")
    if "ablation" in args.figures:
        fig4_ablation_level15(args.runs_dir, out_dir / "fig4_ablation_level15.png")


if __name__ == "__main__":
    main()
