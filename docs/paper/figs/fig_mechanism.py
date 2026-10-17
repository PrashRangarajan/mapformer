"""Fig. 2 of the workshop draft (shading fixed 2026-10-16: only the mixed-sign quadrants are closed to |.|; omega's sign is
free, so the all-negative quadrant is open) (DRAFT4_workshop.md Sec. 4.3; MECHANISM batch, ring 32, p_plus 0.5, T = 128, 300 epochs,
seeds 800-807). Panel titles use the draft's terms (2026-10-15 relabel; arm codes only in the output text).
Per rotary channel (64 per run, 8 runs per arm), the pair of move angles (a(+1), a(-1)) = omega * step, at the start
(eval.json init_angles) and at the end of training (action_angles of the checkpoint).

View: each angle has its whole turns removed toward zero, rep(a) = sign(a) (|a| mod 2 pi). Adding 2 pi to one move's angle
leaves the model's function unchanged (theta = cumsum, only cos / sin enter), so this is a valid representative, and it keeps
the raw sign, which is what the non-negative constraint acts on (rule 8: the cancelling share is wrap-invariant; the
through-zero / by-wrap split is a property of the parameters, not of the function). In this view the cancelling pairs
(a(+1) + a(-1) = 0 mod 2 pi) lie on two lines: a(-1) = -a(+1) through the origin (needs opposite signs; closed to |.|) and
a(+1) + a(-1) = 2 pi (or -2 pi). Raw angles of the signed arms are all within (-2 pi, 2 pi), so rep is the identity for them;
AO_high's raw angles reach 17.7 rad (up to 3 turns).
Cancelling = |wrap(a(+1) + a(-1))| < 0.1 rad among channels with |wrap(a(+1))| > 0.05 (WRAP_EXISTENCE Amendment 1, the
registered wrap share); split by |raw sum| < pi (through zero) vs > pi (by wrapping), as mech_escape.py.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_mechanism.py"""
import os

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mapformer.analyze_wrap_mech import load
from mapformer.model_wrap_exist import action_angles

R = "/home/prashr/mapformer/runs/wrap_mech/p0"; HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = range(800, 808); TP = 2 * np.pi
ARMS = [("SO_absmag", "(a) signed step, same-direction start"), ("AO_absmag", "(b) |.| step, the same start"),
        ("AO_mid", "(c) |.| step, large start, sums < pi"), ("AO_high", "(d) |.| step, start past pi")]
BOX = {"AO_mid": (0.9, 1.5), "AO_high": (1.7, 4.5)}   # iid U(lo, hi) start of both move angles
ARROW_SEED = 800                                    # init -> final segments drawn for one seed (64 channels)


def rep(a):
    return np.sign(a) * np.mod(np.abs(a), TP)


def wrap(a):
    return np.angle(np.exp(1j * a))


data, lines = {}, []
for arm, _ in ARMS:
    I, F, solved = [], [], 0
    for s in SEEDS:
        r = load(R, arm, s)
        I.append(np.array(r["ev"]["init_angles"])); F.append(action_angles(r["model"]).double().numpy()); solved += r["solved"]
    I = np.stack(I); F = np.stack(F)                                   # (seed, 2, C)
    sel = np.abs(wrap(F[:, 0])) > 0.05; canc = np.abs(wrap(F[:, 0] + F[:, 1])) < 0.1
    opp = np.sign(F[:, 0]) != np.sign(F[:, 1])                          # through zero needs opposite raw signs
    via0 = sel & canc & opp & (np.abs(F[:, 0] + F[:, 1]) < np.pi); viaw = sel & canc & (np.abs(F[:, 0] + F[:, 1]) > np.pi)
    sel0 = np.abs(wrap(I[:, 0])) > 0.05; canc0 = np.abs(wrap(I[:, 0] + I[:, 1])) < 0.1
    data[arm] = dict(I=I, F=F, sel=sel, via0=via0, viaw=viaw, solved=solved, share=(sel & canc).sum() / sel.sum())
    lines.append(f"{arm:10s} SOLVED {solved}/8; cancelling share at start {(canc0 & sel0).sum() / sel0.sum():.3f}, at end "
                 f"{(sel & canc).sum() / sel.sum():.3f} (through zero, opposite signs {via0.sum() / sel.sum():.3f}, by wrap {viaw.sum() / sel.sum():.3f}, "
                 f"same-sign near the origin {(sel & canc & ~via0 & ~viaw).sum() / sel.sum():.3f}); "
                 f"channels with |wrap(a(+1))| <= 0.05 at end {1 - sel.mean():.3f}; raw final range {F.min():.2f}..{F.max():.2f}; "
                 f"init sum range {(I[:, 0] + I[:, 1]).min():.2f}..{(I[:, 0] + I[:, 1]).max():.2f}")

fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.8), sharex=True, sharey=True)
lo, hi = -1.5 * np.pi, TP
for ax, (arm, desc) in zip(axs.flat, ARMS):
    d = data[arm]; I = rep(d["I"]); F = rep(d["F"])
    # |.| class: omega * |s| with omega free in sign, so both move angles share omega's sign per channel -- quadrants I and
    # III are open to it; only the two MIXED-sign quadrants (II, IV) are closed (MOCK_REVIEW_draft5 W9; appendix E P3)
    ax.add_patch(plt.Rectangle((lo, 0), -lo, hi, color="0.92", lw=0, zorder=0))      # quadrant II: a(+1) < 0 < a(-1)
    ax.add_patch(plt.Rectangle((0, lo), hi, -lo, color="0.92", lw=0, zorder=0))      # quadrant IV: a(-1) < 0 < a(+1)
    x = np.linspace(lo, hi, 2)
    ax.plot(x, -x, color="C0", lw=0.9, alpha=0.45, zorder=1)            # through zero
    ax.plot(x, TP - x, color="C3", lw=0.9, alpha=0.45, zorder=1)        # by wrapping
    ax.plot(x, -TP - x, color="C3", lw=0.9, alpha=0.45, zorder=1)
    ax.plot(x, np.pi - x, color="0.5", lw=0.5, ls=":", zorder=1)        # sum = pi
    si = ARROW_SEED - SEEDS[0]
    if arm in BOX:
        b0, b1 = BOX[arm]
        ax.add_patch(plt.Rectangle((b0, b0), b1 - b0, b1 - b0, fill=False, ec="0.35", lw=0.6, ls="--", zorder=6))
    for c in range(I.shape[2]):
        if np.abs(d["F"][si, :, c]).max() >= TP:                     # a turn was removed: the segment would not be the path
            continue
        ax.plot([I[si, 0, c], F[si, 0, c]], [I[si, 1, c], F[si, 1, c]], color="0.6", lw=0.3, alpha=0.6, zorder=2)
    ax.scatter(I[:, 0].ravel(), I[:, 1].ravel(), s=1.5, color="0.55", marker=".", lw=0, zorder=3)
    other = ~(d["via0"] | d["viaw"])
    ax.scatter(F[:, 0][other], F[:, 1][other], s=2.5, color="k", lw=0, zorder=4)
    ax.scatter(F[:, 0][d["via0"]], F[:, 1][d["via0"]], s=3.5, color="C0", lw=0, zorder=5)
    ax.scatter(F[:, 0][d["viaw"]], F[:, 1][d["viaw"]], s=3.5, color="C3", lw=0, zorder=5)
    share = d["share"]                                                  # registered wrap share
    ax.set_title(f"{desc}\n{d['solved']}/8 SOLVED", fontsize=6.3, pad=2)
    ax.text(0.97, 0.03, f"cancel {share:.2f}", transform=ax.transAxes, fontsize=5.8, ha="right", va="bottom")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ticks = [-np.pi, 0, np.pi, TP]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(["-pi", "0", "pi", "2pi"], fontsize=6); ax.set_yticklabels(["-pi", "0", "pi", "2pi"], fontsize=6)
    ax.tick_params(length=2, pad=1)
for ax in axs[1]:
    ax.set_xlabel("a(+1)  [rad]", fontsize=6.5, labelpad=1)
for ax in axs[:, 0]:
    ax.set_ylabel("a(-1)  [rad]", fontsize=6.5, labelpad=1)
H = [plt.Line2D([], [], color="C0", lw=0.9, alpha=0.45, label="a(-1) = -a(+1) (through zero)"),
     plt.Line2D([], [], color="C3", lw=0.9, alpha=0.45, label="a(+1) + a(-1) = +-2pi (wrap)"),
     plt.Rectangle((0, 0), 1, 1, color="0.92", lw=0, label="mixed signs: closed to |.|"),
     plt.Line2D([], [], color="0.5", lw=0.5, ls=":", label="sum = pi"),
     plt.Line2D([], [], color="0.55", marker=".", ls="", ms=3, label="start (- - start box)"),
     plt.Line2D([], [], color="k", marker=".", ls="", ms=3, label="end, other (incl. collapsed)"),
     plt.Line2D([], [], color="C0", marker=".", ls="", ms=3.5, label="end, cancels through zero"),
     plt.Line2D([], [], color="C3", marker=".", ls="", ms=3.5, label="end, cancels by wrapping")]
fig.legend(handles=H, loc="lower center", ncol=2, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handlelength=1.6, columnspacing=0.8, handletextpad=0.4)
fig.subplots_adjust(left=0.14, right=0.98, top=0.925, bottom=0.205, hspace=0.36, wspace=0.1)
for ext in ("pdf", "png"):
    fig.savefig(f"{HERE}/fig_mechanism.{ext}", dpi=300)
txt = "\n".join(lines); print(txt); open(f"{HERE}/fig_mechanism_out.txt", "w").write(txt + "\n")
