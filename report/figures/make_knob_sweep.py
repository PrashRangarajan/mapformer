"""Figure: torus knob sweep at n=8 and allocentric recoding (KNOB_SWEEP_n8.json).

One marker per seed for the path-integrated arm (Vanilla) and the index arm (RoPE)
in each condition, with each condition's measured floor. Means are the per-arm means
KNOB_SWEEP_n8.md reports; nothing else is computed.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "KNOB_SWEEP_n8.json")
d = json.load(open(SRC))

conds = [("baseline", "baseline\n(translate)"), ("rotate", "rotate\n(turn/forward)"),
         ("allocentric", "rotate,\nallocentric record"), ("allcombined", "all five\nknobs")]
fig, ax = plt.subplots(figsize=(5.4, 3.3))
for i, (c, label) in enumerate(conds):
    v, r, fl = d[c]["vanilla"], d[c]["rope"], d[c]["floor"]
    ax.scatter([i - 0.13] * len(v), v, s=12, color="#2166ac", alpha=0.7, linewidths=0,
               label="path-integrated (MapWM)" if i == 0 else None)
    ax.scatter([i + 0.13] * len(r), r, s=12, color="#b2182b", alpha=0.7, linewidths=0,
               label="index (RoPE)" if i == 0 else None)
    ax.plot([i - 0.22, i - 0.04], [sum(v) / len(v)] * 2, color="#2166ac", lw=2)
    ax.plot([i + 0.04, i + 0.22], [sum(r) / len(r)] * 2, color="#b2182b", lw=2)
    ax.plot([i - 0.3, i + 0.3], [fl, fl], color="grey", ls=":", lw=1,
            label="measured floor" if i == 0 else None)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([l for _, l in conds], fontsize=8)
ax.set_ylabel("held-out accuracy")
ax.set_ylim(0.45, 1.03)
ax.set_title("Torus knob sweep, 8 seeds per arm", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.2, 0.9))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "knob_sweep.pdf"))
print("wrote knob_sweep.pdf from", SRC)
