"""Summary figure for the main paper (one image, three panels). All numbers are registered results, copied from the
results files named beside each value; the strata floors are analyses (docs/audits/2026-10-16, FIGS_NOTES.md).
(a) Torus T = 1024: one path layer vs index attention at 1 / 4 layers / looped x4 (FACTORIAL_RESULTS.md, n = 4),
    with this batch's map-free floors (blank 0.509, retrace 0.844).
(b) Failure by search: runs SOLVED (of 8) from scratch vs with an exact solution installed and then trained
    (rank 2: RANK_PROJ_RESULTS.md / RANK_MI; |.| ring: WRAP_EXISTENCE_RESULTS.md; |.| torus: GENERALITY_RESULTS.md).
(c) One failure signature: accuracy by revisit stratum at T = 1024 (MAIN_DRAFT1 Sec. 6.3; SIGN_MATCHED / RANK strata).
"""
import os
import numpy as np
from figstyle import plt, BLUE, ORANGE, AQUA, INK, INK2, MUTED, floor_line, save, write_out

HERE = os.path.dirname(os.path.abspath(__file__))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.0, 1.15, 1.1]})

# (a) depth cannot buy the map
ax = axs[0]
labels = ["index\n1 layer", "index\n4 layers", "index\nloop x4", "path\n1 layer"]
vals = [0.719, 0.860, 0.864, 1.000]
cols = [ORANGE, ORANGE, ORANGE, BLUE]
x = np.arange(len(vals))
ax.bar(x, [v - 0.4 for v in vals], bottom=0.4, width=0.62, color=cols, edgecolor="white", linewidth=1.0, zorder=2)
for xi, v in zip(x, vals):
    ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=INK)
floor_line(ax, 0.844, "retrace\n0.844", x=1.01, ha="left", va="center")
floor_line(ax, 0.509, "blank\n0.509", x=1.01, ha="left", va="center")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0.4, 1.08); ax.set_ylabel("held-out accuracy, torus T = 1024")
ax.set_title("(a) One path layer learns the map;\nindex depth does not", loc="left", color=INK)
ax.grid(axis="x", visible=False)

# (b) the solution exists and is kept, but is not found
ax = axs[1]
cases = ["rank 2\n(torus)", "|.| step\n(ring)", "|.| step\n(torus)"]
scratch = [0, 0, 0]
installed = [7, 8, 8]
x = np.arange(len(cases)); w = 0.34
ax.bar(x - w / 2 - 0.02, [s + 0.06 for s in scratch], width=w, color=ORANGE, edgecolor="white", linewidth=1.0,
       label="trained from scratch", zorder=2)
ax.bar(x + w / 2 + 0.02, installed, width=w, color=BLUE, edgecolor="white", linewidth=1.0,
       label="exact solution installed, then trained", zorder=2)
for xi, s, i in zip(x, scratch, installed):
    ax.text(xi - w / 2 - 0.02, s + 0.25, f"{s}/8", ha="center", va="bottom", fontsize=6, color=INK)
    ax.text(xi + w / 2 + 0.02, i + 0.15, f"{i}/8", ha="center", va="bottom", fontsize=6, color=INK)
ax.set_xticks(x); ax.set_xticklabels(cases)
ax.set_ylim(0, 12.6); ax.set_yticks([0, 2, 4, 6, 8]); ax.set_ylabel("runs that solve the map (of 8)")
ax.legend(loc="upper left", ncol=1, handlelength=1.0, borderaxespad=0.1)
ax.set_title("(b) The solution exists and training keeps it,\nbut training from scratch does not find it",
             loc="left", color=INK)
ax.grid(axis="x", visible=False)

# (c) one failure signature
ax = axs[2]
strata = ["gap\n< 128", "gap\n>= 128", "wrap-\nonly"]
series = [
    ("map learned\n(signed, rank 4)", [1.000, 0.995, 0.982], BLUE, "o", "-"),
    ("rank 2", [0.950, 0.671, 0.557], AQUA, "^", "-"),
    ("|.| step", [0.870, 0.630, 0.515], ORANGE, "s", "-"),
    ("index RoPE", [0.784, 0.462, 0.471], MUTED, "D", (0, (2, 1.5))),
]
xs = np.arange(3)
for name, ys, c, m, ls in series:
    ax.plot(xs, ys, color=c, marker=m, ms=4.2, lw=1.6, ls=ls, zorder=3,
            markeredgecolor="white", markeredgewidth=0.6)
    ax.text(xs[-1] + 0.08, ys[-1], name, ha="left", va="center", fontsize=5.8, color=INK2)
floor_line(ax, 0.507, "blank", x=0.02, ha="left")
ax.set_xticks(xs); ax.set_xticklabels(strata); ax.set_xlim(-0.2, 3.1)
ax.set_ylim(0.4, 1.04); ax.set_ylabel("revisit accuracy, torus T = 1024")
ax.set_title("(c) Every failure keeps short returns\nand loses long and wrap-around ones", loc="left", color=INK)
ax.grid(axis="x", visible=False)

fig.tight_layout(w_pad=2.2)
save(fig, HERE, "fig_summary")
write_out(HERE, "fig_summary", [
    "(a) FACTORIAL (n = 4): index 1L 0.719, 4L 0.860, loop x4 0.864, path 1L 1.000; floors blank 0.509, retrace 0.844",
    "(b) SOLVED of 8 from scratch / installed-then-trained: rank 2 0 / 7 (RANK_MI shared r2; RANK_PROJ trained); "
    "|.| ring 0 / 8 (WRAP_EXISTENCE AO / AO_bal); |.| torus 0 / 8 (GENERALITY AO4 / AO_bal, re-found)",
    "(c) strata short / long / wrap-only: signed 1.000 / 0.995 / 0.982; rank 2 (shared) 0.950 / 0.671 / 0.557; "
    "|.| 0.870 / 0.630 / 0.515; index RoPE 1L 0.784 / 0.462 / 0.471; blank ~0.507",
])
