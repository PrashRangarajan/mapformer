"""F2 of the overall outline: accuracy by revisit stratum for each failure mode on the torus (T = 1024, trained and tested
at T = 1024, 900 epochs; eval_rank_strata on the held-out map, eval mode; stored *_STRATA.json, no model):
  (a) index attention: FACTORIAL (registered, n = 4, seeds 500-503; --fast-attn): path 1L, index 1L / 4L / looped x4
  (b) rank of the content-to-angle map: RANK_MI (registered, n = 8, seeds 0-7: shared r 2, per-head r 2, shared r 4) and
      RANK3 (registered, per-head r 3, the same seeds and initial weights)
  (c) sign of the step: SIGN_MATCHED (registered, n = 8, seeds 0-7): signed, |.|, softplus; index RoPE 1L as reference
Strata (eval_rank_strata): plain revisit with gap < 128 moves, plain with gap >= 128, wrap-only (the unwrapped position is
new). Floors per stratum, on each batch's own stream (torus_strata_floors.py; post hoc): best constant (= blank) and
retrace (short-gap stratum only; it equals the constant elsewhere). Lines: arm means; dots: runs.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_strata_modes.py"""
import json, os
import numpy as np
from figstyle import plt, BLUE, ORANGE, AQUA, YELLOW, BLUE_RAMP, INK2, MUTED, FLOOR_LS, save, write_out

REPO = "/home/prashr/mapformer"; HERE = os.path.dirname(os.path.abspath(__file__))
ST = [("plain_lag<128", "gap < 128"), ("plain_lag>=128", "gap >= 128"), ("wrap", "wrap-only")]
PANELS = [
    ("(a) index attention (FACTORIAL, n 4)", "torus_strata_floors_s500-503.json", [
        ("FACTORIAL_L1_STRATA.json", "Vanilla_r4", "path, 1 layer", BLUE, "o"),
        ("FACTORIAL_L1_STRATA.json", "RoPE", "index, 1 layer", ORANGE, "s"),
        ("FACTORIAL_L4_STRATA.json", "RoPE", "index, 4 layers", AQUA, "^"),
        ("FACTORIAL_LOOP_STRATA.json", "RoPELooped", "index, looped x4", YELLOW, "D")]),
    ("(b) per-head rank (RANK_MI, RANK3, n 8)", "torus_strata_floors_s0-7.json", [
        ("RANK_MI_STRATA.json", "Vanilla_r4mi", "rank 4 (shared r 4)", BLUE, "o"),
        ("RANK3_STRATA.json", "Vanilla_r3ph", "rank 3 (per-head r 3)", ORANGE, "s"),
        ("RANK_MI_STRATA.json", "Vanilla_r2ph", "rank 2 (per-head r 2)", AQUA, "^"),
        ("RANK_MI_STRATA.json", "Vanilla", "rank 2 (shared r 2)", YELLOW, "D")]),
    ("(c) sign of the step (SIGN_MATCHED, n 8)", "torus_strata_floors_s0-7.json", [
        ("SIGN_MATCHED_STRATA.json", "Signed_r4", "signed", BLUE, "o"),
        ("SIGN_MATCHED_STRATA.json", "Abs_r4", "|.|", ORANGE, "s"),
        ("SIGN_MATCHED_STRATA.json", "Pos_r4", "softplus", AQUA, "^"),
        ("SIGN_MATCHED_STRATA.json", "RoPE", "index RoPE, 1 layer", YELLOW, "D")]),
]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
rng = np.random.default_rng(0); x = np.arange(3); out = []
for ax, (title, ffile, arms) in zip(axs, PANELS):
    out.append(title)
    for gi, (f, arm, lab, col, mk) in enumerate(arms):
        d = json.load(open(f"{REPO}/{f}"))
        runs = [v for k, v in d.items() if k.split("|")[0] == arm and k.split("|")[2] == "1024"]
        A = np.array([[r[s]["acc"] for s, _ in ST] for r in runs]); off = (gi - 1.5) * 0.06
        for row in A:
            ax.scatter(x + off + rng.uniform(-0.015, 0.015, 3), row, s=4, color=col, alpha=0.45, lw=0, zorder=2)
        ax.plot(x + off, A.mean(0), color=col, lw=1.5, marker=mk, ms=3.6, zorder=3, label=lab)
        out.append(f"  {lab:24s} n={len(runs)} " + "  ".join(f"{s}: {A[:, j].mean():.3f} ({A[:, j].min():.3f}-{A[:, j].max():.3f})"
                                                           for j, (s, _) in enumerate(ST)))
    FL = json.load(open(f"{HERE}/{ffile}"))["per_seed"]
    for j, (s, _) in enumerate(ST):
        c = np.mean([FL[k][s]["const"] for k in FL]); r = np.mean([FL[k][s]["retrace"] for k in FL])
        ax.plot([j - 0.3, j + 0.3], [c, c], color=INK2, lw=0.8, ls=FLOOR_LS, zorder=1)
        if r - c > 0.01:
            ax.plot([j - 0.3, j + 0.3], [r, r], color=INK2, lw=0.8, ls=(0, (1, 1.2)), zorder=1)
        out.append(f"  floor {s:15s} best constant {c:.3f}  retrace {r:.3f}  ({ffile})")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in ST]); ax.set_xlim(-0.4, 2.4); ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontsize=6.5)
    ax.legend(loc="lower left", fontsize=5.5, handlelength=1.6, bbox_to_anchor=(0.0, 0.0))
axs[0].set_ylim(0.3, 1.02); axs[0].set_ylabel("revisit accuracy (held-out map)")
axs[0].text(0.3, 0.925, "retrace", fontsize=5.4, color=INK2, va="bottom", ha="left")
axs[0].text(2.3, 0.522, "blank", fontsize=5.4, color=INK2, va="bottom", ha="right")
fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.1, wspace=0.07)
save(fig, HERE, "fig_strata_modes"); write_out(HERE, "fig_strata_modes", out)
