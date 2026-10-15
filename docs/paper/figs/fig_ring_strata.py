"""Ring revisit accuracy by stratum (plan item 7; POST HOC, eval-only). Reads ring_strata.json (ring_strata.py). Lines: arm
means; dots: runs (jittered); hollow markers beside "wrap-only": plain revisits re-weighted to the wrap-only gap distribution.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_ring_strata.py"""
import json, os

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{HERE}/ring_strata.json")); RUNS = J["runs"]
ST = [("plain gap 2-7", "plain\ngap 2-7"), ("plain gap 8-31", "plain\n8-31"), ("plain gap 32-128", "plain\n32-127"),
      ("wrap-only", "wrap-only\n(gap 32-124)")]
GROUPS = [(("RING_DIAL|Signed_r4", "WRAP_EXISTENCE|SO", "MECHANISM|SO"), "signed (3 batches)", "C0", "o"),
          (("RING_DIAL|Abs_r4",), "|.| Abs_r4, RING_DIAL", "C1", "s"),
          (("WRAP_EXISTENCE|AO",), "|.| AO, WRAP_EXIST.", "C3", "^"),
          (("MECHANISM|AO",), "|.| AO, MECHANISM", "C4", "v"),
          (("MECHANISM|AO_high",), "|.| AO_high (start > pi)", "C2", "D")]
fig, ax = plt.subplots(figsize=(3.4, 2.45)); rng = np.random.default_rng(0); x = np.arange(len(ST))
for gi, (keys, lab, col, mk) in enumerate(GROUPS):
    rs = [v for k, v in RUNS.items() if "|".join(k.split("|")[:2]) in keys]
    off = (gi - 2) * 0.07
    A = np.array([[r[s] for s, _ in ST] for r in rs])
    for row in A:
        ax.scatter(x + off + rng.uniform(-0.02, 0.02, len(x)), row, s=3, color=col, alpha=0.35, lw=0)
    ax.plot(x + off, A.mean(0), color=col, marker=mk, ms=3.2, lw=0.9,
            label=f"{lab}  {sum(r['solved'] for r in rs)}/{len(rs)}")
    gm = np.mean([r["plain, wrap-gap-matched"] for r in rs])
    ax.scatter(x[-1] + off + 0.3, gm, s=12, facecolors="none", edgecolors=col, marker=mk, lw=0.8)
ax.text(x[-1] + 0.3, 0.705, "plain,\ngap-\nmatched", fontsize=5, ha="center", va="bottom", color="0.3")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in ST], fontsize=6)
ax.set_xlim(-0.35, x[-1] + 0.6); ax.set_ylim(0.70, 1.012); ax.tick_params(labelsize=6, length=2)
ax.set_ylabel("revisit accuracy (eval mode)", fontsize=6.5)
ax.text(0.01, 0.01, f"constant-predictor floor {J['floor']['all revisits']:.2f} (off axis)", transform=ax.transAxes,
        fontsize=5.3, color="0.3", va="bottom")
ax.legend(fontsize=5.3, loc="lower left", bbox_to_anchor=(0.0, 0.06), frameon=False, title="arm  SOLVED",
          title_fontsize=5.3, handlelength=1.8)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(f"{HERE}/fig_ring_strata.{ext}", dpi=300)
print("ok")
