#!/usr/bin/env python3
"""Generate per-visit-count figures from PER_VISIT_*.md tables.

Produces:
  fig5_per_visit_curves.png    — line plot, accuracy vs visit count k
  fig6_one_shot_bar.png        — bar chart of k=2 accuracy (the headline cell)

Uses the markdown tables produced by per_visit_eval.py. No model loading;
purely figure synthesis from the cached results.

Usage:
  python3 -m mapformer.make_per_visit_figure \
      --clean PER_VISIT_clean.md --lm200 PER_VISIT_lm200.md \
      --output-dir paper_figures
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Consistent colors with make_paper_figures.py
VARIANT_COLORS = {
    "Vanilla":    "#808080",
    "VanillaEM":  "#404040",
    "RoPE":       "#606060",
    "LSTM":       "#A0522D",
    "MambaLike":  "#FF7043",
    "Level1":     "#2196F3",
    "Level15":    "#43A047",
    "Level15EM":  "#2E7D32",
    "Level2":     "#FFA000",
    "PC":         "#9C27B0",
}

# Order in which to display variants (cognitive-map family first, then baselines)
DISPLAY_ORDER = [
    "Level15", "Level15EM", "VanillaEM", "Vanilla", "Level1", "PC",
    "LSTM", "MambaLike", "RoPE",
]

# Pretty labels
PRETTY = {
    "Vanilla":   "MapFormer-WM",
    "VanillaEM": "MapFormer-EM",
    "RoPE":      "RoPE",
    "Level1":    "Level 1 InEKF",
    "Level15":   "Level 1.5 InEKF",
    "Level15EM": "Level 1.5-EM",
    "Level2":    "Level 2",
    "PC":        "Predictive Coding",
    "LSTM":      "LSTM",
    "MambaLike": "MambaLike (SSM)",
}


def parse_per_visit_md(path: Path):
    """Parse PER_VISIT_*.md and return dict[variant] -> dict[k] -> (mean, std).

    k is an int 1..8 or "over_max". Values are floats.
    """
    text = path.read_text()
    # Find the markdown table — header line starts with "| Variant |"
    lines = text.splitlines()
    rows = []
    in_table = False
    headers = None
    for line in lines:
        if line.startswith("| Variant"):
            in_table = True
            headers = [h.strip() for h in line.strip().strip("|").split("|")]
            continue
        if in_table:
            if line.startswith("|---"):
                continue
            if not line.startswith("|"):
                break
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(cols)
    if headers is None:
        raise RuntimeError(f"No table found in {path}")

    # headers like ["Variant", "k=1", "k=2", ..., "k=over_max"]
    bin_keys = []
    for h in headers[1:]:
        m = re.match(r"k=(\w+)", h)
        if not m:
            continue
        v = m.group(1)
        bin_keys.append(int(v) if v.isdigit() else "over_max")

    out = {}
    for row in rows:
        variant = row[0]
        out[variant] = {}
        for k, cell in zip(bin_keys, row[1:]):
            if cell in ("—", "N/A", ""):
                out[variant][k] = (np.nan, np.nan)
                continue
            # cell like "0.910±0.044" or "0.910"
            if "±" in cell:
                m, s = cell.split("±")
                out[variant][k] = (float(m), float(s))
            else:
                out[variant][k] = (float(cell), 0.0)
    return out


def fig5_per_visit_curves(clean, lm200, out_path):
    """Two-panel line plot: accuracy vs visit count for clean + lm200."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, data, title in [(axes[0], clean, "Clean (no landmarks)"),
                            (axes[1], lm200, "200 landmarks")]:
        # x-axis bins: integer k=1..8 only (skip over_max for line plot)
        ks = sorted([k for k in next(iter(data.values())).keys() if isinstance(k, int)])

        # Plot in DISPLAY_ORDER so legend ordering is consistent
        for variant in DISPLAY_ORDER:
            if variant not in data:
                continue
            means = np.array([data[variant][k][0] for k in ks])
            stds = np.array([data[variant][k][1] for k in ks])
            color = VARIANT_COLORS.get(variant, "#000000")
            # Highlight Level 1.5 with thicker line
            lw = 3.0 if variant == "Level15" else 1.6
            alpha_main = 1.0 if variant in ("Level15", "Level15EM", "MambaLike", "RoPE") else 0.9
            ax.plot(ks, means, "-o", color=color, linewidth=lw, markersize=5,
                    label=PRETTY.get(variant, variant), alpha=alpha_main)
            # Shaded ±std
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.12)

        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(8.05, 0.51, "blank baseline (~0.5)",
                fontsize=8, color="black", alpha=0.6)

        ax.set_xlabel("Visit count $k$ (how many times this cell has been visited)",
                      fontsize=11)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Revisit prediction accuracy", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)

    fig.suptitle("Per-visit-count generalization curves "
                 "(TEM-style one-shot test)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"  wrote {out_path}", file=sys.stderr)
    plt.close(fig)


def fig6_one_shot_bar(clean, lm200, out_path):
    """Two-panel bar chart of k=2 (first revisit) accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)

    for ax, data, title in [(axes[0], clean, "Clean (no landmarks)"),
                            (axes[1], lm200, "200 landmarks")]:
        variants = [v for v in DISPLAY_ORDER if v in data]
        means = [data[v][2][0] for v in variants]
        stds = [data[v][2][1] for v in variants]
        colors = [VARIANT_COLORS.get(v, "#000000") for v in variants]
        labels = [PRETTY.get(v, v) for v in variants]

        x = np.arange(len(variants))
        bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.8,
                      yerr=stds, capsize=4, error_kw=dict(elinewidth=0.8, ecolor="black"))

        # Annotate each bar with mean
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{m:.2f}", ha="center", fontsize=8.5)

        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy at first revisit ($k=2$)", fontsize=11)

    fig.suptitle("One-shot generalization: accuracy on the FIRST revisit "
                 "to each cell\n"
                 "(higher = stronger cognitive-map structure; "
                 "no jump = memorization-style learning)",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"  wrote {out_path}", file=sys.stderr)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", default="PER_VISIT_clean.md")
    p.add_argument("--lm200", default="PER_VISIT_lm200.md")
    p.add_argument("--output-dir", default="paper_figures")
    args = p.parse_args()

    clean = parse_per_visit_md(Path(args.clean))
    lm200 = parse_per_visit_md(Path(args.lm200))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig5_per_visit_curves(clean, lm200, out / "fig5_per_visit_curves.png")
    fig6_one_shot_bar(clean, lm200, out / "fig6_one_shot_bar.png")

    print(f"  done → {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
