"""Fig. K1: accuracy by revisit stratum, torus T = 1024 (FACTORIAL, registered batch; the strata are a registered
secondary). Reads FACTORIAL_{L1,L4,LOOP}_STRATA.json and torus_strata_floors_s500-503.json (this batch's own eval stream,
seeds 500-503; post hoc floors). Bars: means over n = 4 runs; dots: runs.
Floors (fixed 2026-10-16, NUMBERS_AUDIT item 1): drawn PER STRATUM, each the mean over the 4 seeds' streams of
  - the best constant predictor within the stratum (= always-blank here; the JSON 'floor'), and
  - the retrace rule (identical to the constant outside the short-gap stratum: retrace never reaches a gap >= 128 or a
    wrap-only revisit).
The old single line was the mean of the per-stratum constant floors over arms and runs (0.513), not the stated 0.507.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_factorial_strata.py"""
import json, os
import numpy as np
from figstyle import plt, BLUE, ORANGE, AQUA, YELLOW, INK2, MUTED, FLOOR_LS, save, write_out

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [("L1", "Vanilla_r4", "path, 1 layer", BLUE), ("L1", "RoPE", "index, 1 layer", ORANGE),
         ("L4", "RoPE", "index, 4 layers", AQUA), ("LOOP", "RoPELooped", "index, looped x4", YELLOW)]
STRATA = [("plain_lag<128", "revisit < 128 moves"), ("plain_lag>=128", "revisit >= 128 moves"), ("wrap", "wrap-only revisit")]
FL = json.load(open(f"{HERE}/torus_strata_floors_s500-503.json"))["per_seed"]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
w = 0.19; x = np.arange(len(STRATA)) * 1.35; out = ["FACTORIAL torus T = 1024, accuracy by stratum (mean, run range), n per cell"]
rng = np.random.default_rng(0)
for i, (f, arm, lab, col) in enumerate(CELLS):
    d = json.load(open(f"{REPO}/FACTORIAL_{f}_STRATA.json"))
    runs = [v for k, v in d.items() if k.split("|")[0] == arm and k.split("|")[2] == "1024"]
    A = np.array([[r[s]["acc"] for s, _ in STRATA] for r in runs]); m = A.mean(0)
    ax.bar(x + (i - 1.5) * w, m, w * 0.88, color=col, label=f"{lab} (n = {len(runs)})", zorder=2)
    for row in A:
        ax.scatter(x + (i - 1.5) * w + rng.uniform(-0.03, 0.03, len(x)), row, s=5, color="white", edgecolors=INK2, lw=0.5,
                   zorder=3)
    out.append(f"{lab:18s} n={len(runs)} " + "  ".join(f"{s}: {m[j]:.3f} ({A[:, j].min():.3f}-{A[:, j].max():.3f})"
                                                     for j, (s, _) in enumerate(STRATA)))
out.append("\nFloors on FACTORIAL's own eval stream (seeds 500-503), mean (range) over seeds:")
for j, (s, _) in enumerate(STRATA + [("all", "all")]):
    c = [FL[k][s]["const"] for k in FL]; r = [FL[k][s]["retrace"] for k in FL]; n = [FL[k][s]["n"] for k in FL]
    out.append(f"  {s:15s} n {sum(n)}  best constant {np.mean(c):.3f} ({min(c):.3f}-{max(c):.3f})  retrace {np.mean(r):.3f} "
               f"({min(r):.3f}-{max(r):.3f})")
    if s == "all":
        continue
    x0, x1 = x[j] - 2 * w, x[j] + 2 * w
    ax.plot([x0, x1], [np.mean(c)] * 2, color=INK2, lw=0.8, ls=FLOOR_LS, zorder=4)
    if np.mean(r) - np.mean(c) > 0.01:
        ax.plot([x0, x1], [np.mean(r)] * 2, color=INK2, lw=0.8, ls=(0, (1, 1.2)), zorder=4)
        ax.text(x1 + 0.02, np.mean(r), f"retrace {np.mean(r):.3f}", fontsize=5.6, va="center", color=INK2)
    ax.text(x1 + 0.02, np.mean(c), f"blank {np.mean(c):.3f}", fontsize=5.6, va="center", color=INK2)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in STRATA]); ax.set_ylim(0.3, 1.02); ax.set_xlim(-0.5, x[-1] + 0.85)
ax.grid(axis="x", visible=False)
ax.set_ylabel("accuracy (held-out map)")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
fig.tight_layout(pad=0.3)
save(fig, HERE, "fig_factorial_strata"); write_out(HERE, "fig_factorial_strata", out)
