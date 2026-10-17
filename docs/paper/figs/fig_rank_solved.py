"""F3 of the overall outline: SOLVED count by per-head rank of the content-to-angle map (registered batches; counts parsed
from the registered analysis outputs, no model). SOLVED = final-5% mean training loss < 0.05.
  (a) 2D torus, T = 1024, 900 epochs, n = 8 per arm, every arm built from our r = 2's initial weights at each seed:
      RANK_SEP_ANALYSIS.txt (A shared r2, B per-head r2, C_bd block-diagonal r4, C shared r4, D per-head r4) and
      RANK3_ANALYSIS.txt (E per-head r3; B and D there are the same runs as RANK_SEP's).
  (b) rank = D vs rank = D + 1 across the task's dimension D (each a different batch and recipe):
      1D ring 32 cells, T = 128, 300 epochs -- RANK_TESTS_ANALYSIS.txt R1 / R2 (rank 1 / rank 2);
      2D torus 32/side, T = 1024 -- RANK_ND_ANALYSIS.txt A2 / B2 (per-head rank 2 / 3);
      3D torus 10/side, T = 1024 -- RANK_ND A3 / B3 (per-head rank 3 / 4); and 3D 18/side -- RANK_WRAP_ANALYSIS.txt 3L
      (per-head rank 4; few wrap-only revisits).
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_rank_solved.py"""
import os, re
import numpy as np
from figstyle import plt, BLUE, ORANGE, BLUE_RAMP, INK2, MUTED, save, write_out

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))


def grab(f, pat):
    for line in open(f"{REPO}/{f}"):
        m = re.search(pat, line)
        if m:
            return m, line.strip()
    raise KeyError((f, pat))


out = []
# (a)
ARMS = [("A", "Vanilla", 2, "shared r 2", "RANK_SEP_ANALYSIS.txt"), ("B", "Vanilla_r2ph", 2, "per-head r 2", "RANK_SEP_ANALYSIS.txt"),
        ("C_bd", "Vanilla_r4mibd", 2, "block-diag. r 4", "RANK_SEP_ANALYSIS.txt"),
        ("E", "Vanilla_r3ph", 3, "per-head r 3", "RANK3_ANALYSIS.txt"),
        ("C", "Vanilla_r4mi", 4, "shared r 4", "RANK_SEP_ANALYSIS.txt"), ("D", "Vanilla_r4ph", 4, "per-head r 4", "RANK_SEP_ANALYSIS.txt")]
A = []
for code, v, rk, lab, f in ARMS:
    m, line = grab(f, rf"^\s*{re.escape(code)}\s+{re.escape(v)}\s+SOLVED (\d)/8\s+acc ([0-9.]+)")
    A.append((code, lab, rk, int(m.group(1)), float(m.group(2)))); out.append(f"(a) {f}: {line}")
# (b)
m1, l1 = grab("RANK_TESTS_ANALYSIS.txt", r"^\s*R1\s+SOLVED (\d)/8"); m2, l2 = grab("RANK_TESTS_ANALYSIS.txt", r"^\s*R2\s+SOLVED (\d)/8")
a2, la2 = grab("RANK_ND_ANALYSIS.txt", r"A2 D=2 \S+\s+SOLVED (\d)/8"); b2, lb2 = grab("RANK_ND_ANALYSIS.txt", r"B2 D=2 \S+\s+SOLVED (\d)/8")
a3, la3 = grab("RANK_ND_ANALYSIS.txt", r"A3 D=3 \S+\s+SOLVED (\d)/8"); b3, lb3 = grab("RANK_ND_ANALYSIS.txt", r"B3 D=3 \S+\s+SOLVED (\d)/8")
w3, lw3 = grab("RANK_WRAP_ANALYSIS.txt", r"3L D=3 N=18 \S+: SOLVED (\d)/8")
for l in (l1, l2, la2, lb2, la3, lb3, lw3):
    out.append(f"(b) {l}")
B = [("1D ring\n32 cells", int(m1.group(1)), int(m2.group(1)), None),
     ("2D torus\n32 / side", int(a2.group(1)), int(b2.group(1)), None),
     ("3D torus\n10 / side", int(a3.group(1)), int(b3.group(1)), int(w3.group(1)))]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1.25, 1]})
RC = {2: BLUE_RAMP[0], 3: BLUE_RAMP[2], 4: BLUE_RAMP[4]}
xs = np.array([0, 1, 2, 3.4, 4.8, 5.8])
for x, (code, lab, rk, ns, acc) in zip(xs, A):
    ax.bar(x, ns, 0.62, color=RC[rk], zorder=2)
    ax.text(x, ns + 0.15, f"{ns}/8", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.text(x, -0.35, lab.replace(" r ", "\nr "), ha="center", va="top", fontsize=5.6, color=INK2, linespacing=0.95)
for rk, xx in ((2, xs[:3]), (3, xs[3:4]), (4, xs[4:])):
    ax.text(xx.mean(), -2.0, f"per-head rank {rk}", ha="center", va="top", fontsize=6.3, color="#0b0b0b")
ax.set_xticks([]); ax.set_ylim(0, 10.2); ax.set_yticks([0, 2, 4, 6, 8]); ax.set_xlim(-0.6, 6.4)
ax.set_ylabel("runs SOLVED (of 8)"); ax.grid(axis="x", visible=False)
ax.set_title("(a) 2D torus, T 1024: bottleneck arms grouped by per-head rank", loc="left", fontsize=6.5)
w = 0.3; xb = np.arange(3)
for i, (lab, nd, nd1, extra) in enumerate(B):
    bx.bar(i - w / 2 - 0.02, nd, w, color=ORANGE, zorder=2, label="rank = D" if i == 0 else None)
    bx.bar(i + w / 2 + 0.02, nd1, w, color=BLUE, zorder=2, label="rank = D + 1" if i == 0 else None)
    bx.text(i - w / 2 - 0.02, nd + 0.15, f"{nd}/8", ha="center", va="bottom", fontsize=6, color=INK2)
    bx.text(i + w / 2 + 0.02, nd1 + 0.15, f"{nd1}/8", ha="center", va="bottom", fontsize=6, color=INK2)
    if extra is not None:
        bx.bar(i + 1.5 * w + 0.06, extra, w, color="white", edgecolor=BLUE, lw=1.0, hatch="////", zorder=2,
               label="rank = D + 1, 18 / side")
        bx.text(i + 1.5 * w + 0.06, extra + 0.15, f"{extra}/8", ha="center", va="bottom", fontsize=6, color=INK2)
bx.set_xticks(xb); bx.set_xticklabels([b[0] for b in B], fontsize=6); bx.set_ylim(0, 10.2); bx.set_yticks([0, 2, 4, 6, 8])
bx.set_xlim(-0.55, 2.8); bx.grid(axis="x", visible=False)
bx.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=5.6, ncol=3, handlelength=1.1, columnspacing=0.8)
bx.set_title("(b) rank = task dimension D vs D + 1", loc="left", fontsize=6.5)
fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.27, wspace=0.12)
save(fig, HERE, "fig_rank_solved"); write_out(HERE, "fig_rank_solved", out)
