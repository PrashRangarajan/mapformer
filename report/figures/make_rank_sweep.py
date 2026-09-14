"""Figure: rank sweep on the torus (RANK_SWEEP.json, committed per-seed data).

Plots held-out revisit accuracy against evaluation length for each bottleneck rank r,
one marker per seed and a line through the per-arm mean. No statistic beyond the
per-arm mean (which RANK_SWEEP.md already reports) is computed.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "RANK_SWEEP.json")

d = json.load(open(SRC))
arms = [("Vanilla", "r=2 (paper)"), ("Vanilla_r4", "r=4"), ("Vanilla_r8", "r=8"),
        ("Vanilla_r16", "r=16"), ("Vanilla_r32", "r=32")]
lengths = [128, 512, 1024]
colors = ["#b2182b", "#2166ac", "#4393c3", "#92c5de", "#053061"]

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for i, ((arm, label), c) in enumerate(zip(arms, colors)):
    means = []
    for j, T in enumerate(lengths):
        vals = [row[1] for row in d[f"0.0|{arm}|{T}"]]
        means.append(sum(vals) / len(vals))
        xo = j + (i - 2) * 0.06
        ax.scatter([xo] * len(vals), vals, s=9, color=c, alpha=0.55, linewidths=0)
    ax.plot([j + (i - 2) * 0.06 for j in range(len(lengths))], means, "-o",
            color=c, ms=3.5, lw=1.4, label=label)
ax.set_xticks(range(len(lengths)))
ax.set_xticklabels([f"T={T}" for T in lengths])
ax.set_ylabel("held-out revisit accuracy\n(blank floor 0.506, off axis)")
ax.set_ylim(0.70, 1.01)
ax.set_title("Torus, MapWM, 8 seeds per rank, one batch", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "rank_sweep.pdf"))
print("wrote rank_sweep.pdf from", SRC)
