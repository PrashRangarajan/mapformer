"""Plan Fig. 9 (docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md 3.6): the clock at the end of training, one dot per run.
ANALYSIS (post hoc, eval-only) of registered batches. ONE definition throughout (ANALYSIS_SCRIPT_AUDIT B3 / M-SG1):
expansion/scripts/sign/clock_table.py's run ratio -- per inverse pair (x, y) and rotary channel c, clock wrap(a_c(x) +
a_c(y)), symmetric map sqrt((wrap(a_c(x))^2 + wrap(a_c(y))^2) / 2), usage-weighted over the channels of both heads,
clock / map = sqrt(sum w clock^2 / sum w map^2). Read from clock_table_out.txt (the 'per run' lines); filled = SOLVED
(final-5% training loss < 0.05, stats_core.classify_run). Rank-2 runs marked 'collapsed' = unsolved, every head's ratio
< 0.15 (clock_table) AND labelled COLLAPSE by rank/sweep_final_out.txt (two map axes near-collinear): per-head r2 s5, s6,
block-diagonal s1, s7 (MAIN_APPENDIX2 G3). No side panel: the off-plane share of RANK's latent clock is on another
definition (audit M-RK1 / M-RK2), so it is not drawn here.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_clock_endstate.py"""
import os
import re

import numpy as np
from figstyle import plt, BLUE, ORANGE, AQUA, INK2, MUTED, save, write_out

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
CT = f"{REPO}/docs/paper/expansion/scripts/sign/clock_table_out.txt"
SW = f"{REPO}/docs/paper/expansion/scripts/rank/sweep_final_out.txt"

# ---- parse clock_table_out.txt ----
txt = open(CT).read().splitlines()
SETS = {}
i = 0
while i < len(txt):
    m = re.match(r"^(ring|torus) (\S+(?: \S+)?)\s{2,}(.+?)\s+SOLVED (\d+)/(\d+) \|", txt[i])
    if m:
        key = m.group(3).strip()
        runs = pis = None
        for j in range(i + 1, min(i + 8, len(txt))):
            if txt[j].strip().startswith("per run (run ratio"):
                runs = re.findall(r"([0-9.]+)\[([0-9./]+)\](\*?)", txt[j].split(":", 1)[1])
            if txt[j].strip().startswith("per run, usage share"):
                pis = re.findall(r"([0-9.]+)(\*?)", txt[j].split("rad):", 1)[1])
        SETS[key] = [dict(run=float(r), heads=[float(h) for h in hs.split("/")], solved=s == "*", pi=float(p))
                     for (r, hs, s), (p, _) in zip(runs, pis)]
        assert sum(r["solved"] for r in SETS[key]) == int(m.group(4)) and len(SETS[key]) == int(m.group(5)), key
    i += 1

# ---- collapsed class from sweep_final_out.txt ----
sw = open(SW).read().split("## ")
COLL = {}
for lab, hdr in (("shared r2", "torus G64 shared r2 (rank_mi)"), ("per-head r2", "torus G64 per-head r2 (rank_mi)"),
                 ("blockdiag r4 (2/head)", "torus G64 block-diag r4 (rank_sep)")):
    blk = next(b for b in sw if b.startswith(hdr))
    COLL[lab] = {int(s) for s, c in re.findall(r"^\s+s(\d+) \S+ (\S+)", blk, re.M) if c == "COLLAPSE"}

# rows: (label, [(set key, task, sign, seed filter)], colour)
RANK2 = ("shared r2", "per-head r2", "blockdiag r4 (2/head)")
ROWS = [
    ("signed, small start", [("SO", "ring"), ("SO_absmag", "ring"), ("SO_m", "ring"), ("SO4", "torus")], BLUE, None),
    ("rank 4 (shared, per-head)", [("shared r4", "torus"), ("per-head r4", "torus")], BLUE, None),
    ("rank 3 (per-head)", [("per-head r3", "torus")], BLUE, None),
    ("rank 2, solved runs", [(k, "torus") for k in RANK2], BLUE, True),
    ("rank 2, unsolved runs", [(k, "torus") for k in RANK2], BLUE, False),
    ("signed, move latents frozen", [("SO_absmag_fz", "ring")], BLUE, None),
    ("|.|, small start, ring", [("AO", "ring"), ("AO_absmag", "ring"), ("AO_m", "ring")], ORANGE, None),
    ("|.|, small start, torus", [("AO4", "torus")], ORANGE, None),
    ("|.|, start past pi, ring", [("AO_high", "ring")], ORANGE, None),
    ("signed, large start", [("SO_high", "ring"), ("SO4_high", "torus")], BLUE, None),
]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
rng = np.random.default_rng(0); out = ["clock / map at the end of training (clock_table.py run ratio; * = SOLVED)"]
for yi, (lab, keys, col, solved_sel) in enumerate(ROWS):
    y0 = len(ROWS) - 1 - yi; vals = []
    for key, task in keys:
        for s, r in enumerate(SETS[key]):
            if solved_sel is not None and r["solved"] != solved_sel:
                continue
            collapsed = key in COLL and not r["solved"] and s in COLL[key] and all(h < 0.15 for h in r["heads"])
            yy = y0 + rng.uniform(-0.22, 0.22); xv = max(r["run"], 1e-3)
            mk = "o" if task == "ring" else "s"
            if collapsed:
                ax.scatter(xv, yy, s=22, marker="X", color=AQUA, edgecolor="white", lw=0.3, zorder=4)
            else:
                ax.scatter(xv, yy, s=13, marker=mk, color=col if r["solved"] else "white", edgecolor=col, lw=0.8,
                           zorder=3, alpha=0.9)
            vals.append((key, s, r, collapsed, yy))
    v = np.array([r["run"] for _, _, r, _, _ in vals]); ns = sum(r["solved"] for _, _, r, _, _ in vals)
    out.append(f"{lab:30s} runs {len(v):2d} SOLVED {ns:2d}  mean {v.mean():.3f} median {np.median(v):.3f} "
               f"range {v.min():.3f}-{v.max():.3f}  sets: {', '.join(k for k, _ in keys)}")
    sv = v[[r["solved"] for _, _, r, _, _ in vals]]; uv = v[[not r["solved"] for _, _, r, _, _ in vals]]
    if 0 < len(sv) < len(v):
        out.append(f"{'':30s} solved {sv.min():.3f}-{sv.max():.3f}; unsolved {uv.min():.3f}-{uv.max():.3f}")
    for key, s, r, c, yy in vals:
        if c:
            out.append(f"{'':30s} collapsed: {key} s{s} run {r['run']:.3f} heads {r['heads']}")
    if lab == "signed, large start":
        pis = []
        for key, s, r, c, yy in vals:
            if key == "SO4_high" and r["solved"]:
                pis.append(r["pi"])
                out.append(f"{'':30s} torus SO4_high solved s{1200 + s}: ratio {r['run']:.3f}, usage share near pi {r['pi']:.2f}")
ax.axvline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(1.03, -0.45, "clock = map", fontsize=5.4, color=INK2, ha="left", va="center")
ax.set_xscale("log"); ax.set_xlim(7e-4, 3.0)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS][::-1]); ax.set_ylim(-0.6, len(ROWS) - 0.2)
ax.grid(axis="y", visible=False)
ax.set_xlabel("clock / map at the end of training (usage-weighted, symmetric map; log scale)")
H = [plt.Line2D([], [], ls="", marker="o", ms=3.6, color=BLUE, label="SOLVED (filled)"),
     plt.Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec=BLUE, label="not SOLVED (open)"),
     plt.Line2D([], [], ls="", marker="o", ms=3.6, mfc="white", mec=INK2, label="ring"),
     plt.Line2D([], [], ls="", marker="s", ms=3.6, mfc="white", mec=INK2, label="torus"),
     plt.Line2D([], [], ls="", marker="X", ms=4.6, color=AQUA, mec="white", label="rank 2, collapsed (clock-free, 1D map)")]
ax.legend(handles=H, loc="lower center", fontsize=5.4, handlelength=1.0, ncol=5, columnspacing=1.0, bbox_to_anchor=(0.45, 1.0))
out.append("signed / rank arms blue, |.| orange; values as printed by clock_table_out.txt (3 decimals: torus signed small start 0.001-0.002)")
fig.subplots_adjust(left=0.27, right=0.98, top=0.93, bottom=0.13)
save(fig, HERE, "fig_clock_endstate"); write_out(HERE, "fig_clock_endstate", out)
