"""F5 of the overall outline: existence before mechanism -- a solution in the constrained class EXISTS (installed, frozen),
is kept or re-found when trained from there, but is not found from scratch (registered batches; numbers parsed from the
registered analysis outputs, no model). Bars: runs SOLVED of 8 (final-5% training loss < 0.05).
  (a) rank 2 on the torus, T = 1024 (seeds 0-7): from scratch = RANK_SEP A (shared r 2); continued r 2 and the
      projected r 2 trained, and the r 4 continuation control = RANK_PROJ_ANALYSIS.txt TRAINABLE / REFERENCE / CONTROL
      (lr restart 1e-3); frozen projected r 2 = RANK_PROJ FROZEN mean. Rank-4 from scratch = RANK_SEP C.
  (b) |.| step on the ring, observation-step-free (WRAP_EXISTENCE, seeds 600-607): E3 (from scratch AO vs SO) and E2
      (installed balanced-gauge construction AO_bal vs SO_bal, trained), LOST (registered): max per-epoch training loss after the start > 0.3, or > 10x
      the signed control's max at that seed and > 0.05.
  (c) |.| step on the torus, observation-step-free (GENERALITY, seeds 700-707): E3 and E2 (lr 1e-4).
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_existence.py"""
import os, re
import numpy as np
from figstyle import plt, BLUE, ORANGE, INK2, MUTED, save, write_out

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
txt = lambda f: open(f"{REPO}/{f}").read()
out = []

rs = txt("RANK_SEP_ANALYSIS.txt"); rp = txt("RANK_PROJ_ANALYSIS.txt")
a_scr = int(re.search(r"^\s*A\s+Vanilla\s+SOLVED (\d)/8", rs, re.M).group(1))
c_scr = int(re.search(r"^\s*C\s+Vanilla_r4mi\s+SOLVED (\d)/8", rs, re.M).group(1))
cnt = lambda lab: int(re.search(rf"{lab}\s+\{{'SOLVED': (\d)", rp).group(1))
tr, ctl, ref = cnt("TRAINABLE r2-proj"), cnt("CONTROL r4-cont"), cnt("REFERENCE r2-cont")
frz = float(re.search(r"mean ([0-9.]+) -> ([0-9.]+); projected", rp).group(2))
out.append(f"(a) rank 2 torus: r2 from scratch {a_scr}/8; r2 continued {ref}/8; r2 installed (projected) + trained {tr}/8; "
           f"r4 from scratch {c_scr}/8; r4 continued {ctl}/8; frozen projected r2 acc {frz:.4f}")


def e23(f, e2tag):
    t = txt(f)
    m3 = re.search(r"E3: .*? -- AO4? ([0-9.]+) SOLVED (\d)/8 vs SO4? ([0-9.]+) SOLVED (\d)/8", t)
    m2 = re.search(rf"E2 \[{re.escape(e2tag)}.*?\]: (\S+(?: HELD)?) -- AO_bal ([0-9.]+) SOLVED (\d)/8 LOST (\d)/8 vs SO_bal ([0-9.]+) "
                   r"SOLVED (\d)/8 LOST (\d)/8", t)
    return m3, m2


panels = [("(a) rank 2, torus T 1024\n(RANK_SEP, RANK_PROJ)",
           [("r 2\nscratch", a_scr, ORANGE, False, None), ("r 2\ncontinued", ref, ORANGE, False, None),
            ("r 2 installed\n+ trained", tr, ORANGE, True, f"frozen\n{frz:.4f}"),
            ("r 4\nscratch", c_scr, BLUE, False, None), ("r 4\ncontinued", ctl, BLUE, True, None)])]
for f, tag, title in (("WRAP_EXISTENCE_ANALYSIS.txt", "AO_bal vs SO_bal, PRIMARY", "(b) |.| step, ring T 128\n(WRAP_EXISTENCE)"),
                      ("GENERALITY_ANALYSIS.txt", "AO_bal vs SO_bal, PRIMARY", "(c) |.| step, torus T 1024\n(GENERALITY)")):
    m3, m2 = e23(f, tag)
    out.append(f"{title.splitlines()[0]}: E3 AO acc {m3.group(1)} SOLVED {m3.group(2)}/8 vs SO {m3.group(3)} {m3.group(4)}/8; "
               f"E2 {m2.group(1)}: AO_bal {m2.group(2)} SOLVED {m2.group(3)}/8 LOST {m2.group(4)}/8 vs SO_bal {m2.group(5)} "
               f"SOLVED {m2.group(6)}/8 LOST {m2.group(7)}/8 (frozen = source, max |frozen - source| 0.0000)")
    panels.append((title, [("|.|\nscratch", int(m3.group(2)), ORANGE, False, f"acc\n{float(m3.group(1)):.3f}"),
                           ("|.| installed,\ntrained", int(m2.group(3)), ORANGE, True, f"lost {m2.group(4)}/8"),
                           ("signed\nscratch", int(m3.group(4)), BLUE, False, None),
                           ("signed inst.,\ntrained", int(m2.group(6)), BLUE, True, f"lost {m2.group(7)}/8")]))

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), sharey=True, gridspec_kw={"width_ratios": [5, 4, 4]})
for ax, (title, bars) in zip(axs, panels):
    for i, (lab, n, col, inst, note) in enumerate(bars):
        ax.bar(i, n, 0.62, color="white" if inst else col, edgecolor=col, hatch="////" if inst else None, lw=1.0, zorder=2)
        ax.text(i, n + 0.15, f"{n}/8", ha="center", va="bottom", fontsize=6, color=INK2)
        if note:
            ax.text(i, n + 1.05, note, ha="center", va="bottom", fontsize=5.2, color=MUTED, linespacing=0.9)
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=5.5, linespacing=0.95)
    ax.set_title(title, loc="left", fontsize=6.4); ax.grid(axis="x", visible=False); ax.set_xlim(-0.6, len(bars) - 0.4)
axs[0].set_ylim(0, 11); axs[0].set_yticks([0, 2, 4, 6, 8]); axs[0].set_ylabel("runs SOLVED (of 8)")
H = [plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="constrained class, from scratch"),
     plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=ORANGE, hatch="////", label="constrained, solution installed, then trained"),
     plt.Rectangle((0, 0), 1, 1, color=BLUE, label="control class")]
fig.legend(handles=H, loc="lower center", ncol=3, fontsize=5.8, bbox_to_anchor=(0.5, 0.0), handlelength=1.4)
fig.subplots_adjust(left=0.06, right=0.995, top=0.86, bottom=0.27, wspace=0.08)
save(fig, HERE, "fig_existence"); write_out(HERE, "fig_existence", out)
