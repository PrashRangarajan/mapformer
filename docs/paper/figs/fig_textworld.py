"""F6 of the overall outline: what the path model learns from navigation told in words (TEXTWORLD; registered batch, the
readouts are its DECLARED SECONDARIES, Amendment 1; no verdict). Reads TEXTWORLD_SECONDARY.json only (no model).
Per path seed (Vanilla_r4, 1 layer, T = 1024; seeds 0-1 were the pilot, CORRECTED block of TEXTWORLD_RESULTS.md):
  (a) opposition of the direction-word steps, mean over the two opposite pairs of |A + B| / mean(|A|, |B|), raw and after
      removing the common component c (the mean of the four direction steps); 0 = opposites cancel exactly;
  (b) |c| / |north|: the size of the common component (map seeds ~0.05; clock seeds 0.5-1.1, where c is parallel to the
      verbs' step: a per-step clock, 31-38 of 64 phase channels drift between visits -- tw_clock_probe.txt);
  (c) held-out accuracy (eval mode, no dropout re-scale) with all steps, with every non-direction word's step zeroed, and
      with the direction words' steps zeroed; floors on the same eval set: best constant 0.505, reversal-copy 0.597.
Seeds are labelled "clock" when |c| / |north| > 0.3 (the split is 0.04-0.06 vs 0.48-1.14; no seed in between).
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_textworld.py"""
import json, os
import numpy as np
from figstyle import plt, BLUE, ORANGE, INK2, MUTED, floor_line, save, write_out

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{REPO}/TEXTWORLD_SECONDARY.json")); S = [f"s{i}" for i in range(8)]
clock = {s: J[s]["raw"]["common_over_N"] > 0.3 for s in S}
col = {s: ORANGE if clock[s] else BLUE for s in S}
out = [f"floors: best constant {J['floors']['constant']:.4f}, reversal-copy {J['floors']['reversal_copy']:.4f} "
       f"(n {J['floors']['n']} targets)"]
for s in S:
    r, a = J[s]["raw"], J[s]["S3"]
    out.append(f"{s} {'clock' if clock[s] else 'map  '}: opposition raw {r['opposition_raw']:.3f} -> minus common "
               f"{r['opposition_minus_common']:.4f} [omega-scaled {J[s]['omega']['opposition_minus_common']:.4f}]; "
               f"|c|/|N| {r['common_over_N']:.3f}; cos(c, verb) {r['cos_common_verb']:+.3f}; synonym cos {r['synonym_cos']:.4f}; "
               f"acc full {a['full']:.3f}, non-direction zeroed {a['non_direction_zeroed']:.3f}, direction zeroed "
               f"{a['direction_zeroed']:.3f}")

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1, 0.8, 1.35]})
ax = axs[0]
for s in S:
    r = J[s]["raw"]
    ax.plot([0, 1], [r["opposition_raw"], r["opposition_minus_common"]], color=col[s], lw=0.9, marker="o", ms=3.5)
ax.set_yscale("log"); ax.set_xticks([0, 1]); ax.set_xticklabels(["raw steps", "common\ncomponent removed"])
ax.set_xlim(-0.3, 1.3); ax.minorticks_off(); ax.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3, 1, 2])
ax.set_yticklabels(["0.003", "0.01", "0.03", "0.1", "0.3", "1", "2"])
ax.set_ylabel("opposition (0 = cancel)"); ax.grid(axis="x", visible=False)
ax.set_title("(a) opposite directions cancel", loc="left")
ax = axs[1]
v = [J[s]["raw"]["common_over_N"] for s in S]
ax.bar(range(8), v, 0.62, color=[col[s] for s in S], zorder=2)
ax.set_xticks(range(8)); ax.set_xticklabels([s for s in S], fontsize=6); ax.grid(axis="x", visible=False)
ax.set_ylabel("|common| / |north step|"); ax.set_title("(b) a shared per-step clock", loc="left")
ax = axs[2]
K = [("full", "all steps", 0.0), ("non_direction_zeroed", "other words'\nsteps zeroed", 1.0),
     ("direction_zeroed", "direction\nsteps zeroed", 2.0)]
for s in S:
    y = [J[s]["S3"][k] for k, _, _ in K]
    ax.plot([x for _, _, x in K], y, color=col[s], lw=0.9, marker="o", ms=3.2, alpha=0.9)
floor_line(ax, J["floors"]["reversal_copy"], f"reversal-copy {J['floors']['reversal_copy']:.3f}", x=0.02, ha="left")
floor_line(ax, J["floors"]["constant"], f"constant {J['floors']['constant']:.3f}", x=0.02, ha="left", va="top")
ax.set_xticks([x for _, _, x in K]); ax.set_xticklabels([l for _, l, _ in K], fontsize=6); ax.set_xlim(-0.3, 2.45)
ax.set_ylim(0.15, 1.03); ax.set_ylabel("held-out accuracy, T 1024"); ax.grid(axis="x", visible=False)
ax.set_title("(c) the direction words carry position", loc="left")
H = [plt.Line2D([], [], color=BLUE, marker="o", ms=3.5, lw=0.9, label=f"map seeds ({sum(not clock[s] for s in S)})"),
     plt.Line2D([], [], color=ORANGE, marker="o", ms=3.5, lw=0.9, label=f"map + clock seeds ({sum(clock.values())})")]
fig.legend(handles=H, loc="upper right", ncol=2, fontsize=6, bbox_to_anchor=(0.995, 1.0))
for a in axs:
    a.title.set_fontsize(6.6)
fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.2, wspace=0.42)
save(fig, HERE, "fig_textworld"); write_out(HERE, "fig_textworld", out)
