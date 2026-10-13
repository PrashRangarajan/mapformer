"""Figure 1 of the workshop outline: accuracy by revisit stratum, torus T = 1024 (FACTORIAL, registered strata secondary).
Reads FACTORIAL_{L1,L4,LOOP}_STRATA.json; writes fig_factorial_strata.pdf / .png next to this script."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [("L1", "Vanilla_r4", "path, 1 layer"), ("L1", "RoPE", "index, 1 layer"), ("L4", "RoPE", "index, 4 layers"),
         ("LOOP", "RoPELooped", "index, looped x4")]
STRATA = [("plain_lag<128", "revisit < 128 moves"), ("plain_lag>=128", "revisit >= 128 moves"), ("wrap", "wrap-only revisit")]
fig, ax = plt.subplots(figsize=(6.4, 2.8))
w = 0.2; x = np.arange(len(STRATA)); floor = []
for i, (f, arm, lab) in enumerate(CELLS):
    d = json.load(open(f"{REPO}/FACTORIAL_{f}_STRATA.json"))
    runs = [v for k, v in d.items() if k.split("|")[0] == arm and k.split("|")[2] == "1024"]
    m = [np.mean([r[s]["acc"] for r in runs]) for s, _ in STRATA]
    lo = [np.min([r[s]["acc"] for r in runs]) for s, _ in STRATA]; hi = [np.max([r[s]["acc"] for r in runs]) for s, _ in STRATA]
    floor += [r[s]["floor"] for r in runs for s, _ in STRATA]
    ax.bar(x + (i - 1.5) * w, m, w, yerr=[np.subtract(m, lo), np.subtract(hi, m)], capsize=2, label=f"{lab} (n={len(runs)})")
ax.axhline(np.mean(floor), color="k", lw=0.8, ls="--"); ax.text(2.45, np.mean(floor) + 0.01, "floor", fontsize=7, ha="right")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in STRATA], fontsize=8); ax.set_ylim(0.3, 1.02)
ax.set_ylabel("accuracy (held-out map)", fontsize=8); ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{HERE}/fig_factorial_strata.{ext}", dpi=200)
print("ok")
