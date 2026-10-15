"""THEORY REVIEW (2026-10-14; CPU; post hoc on committed MECHANISM checkpoints, not registered).
Per-channel tests of the cut-circle account (THEORY_REVIEW.md, claim 3):
 (a) |.| models: a channel can cancel only by wrapping (raw sum -> 2 pi); the account predicts that the channels which end
     cancelled are the ones whose initial sum a(+) + a(-) was largest (closest to / past the divide near pi).
 (b) signed models from same-direction starts: the account predicts that the angle that crosses zero is the initially
     smaller of the two (the gradient at small same-direction angles raises the larger and lowers the smaller).
 (c) every arm: final opposite-sign share by initial sign agreement.
Weighted by nothing: every channel counts once; selection |wrap(a(+))| > 0.05 at the end as mech_escape.py.
"""
import sys
sys.path.insert(0, "/home/prashr")
import numpy as np
from mapformer.analyze_wrap_mech import load
from mapformer.model_wrap_exist import action_angles

R = "/home/prashr/mapformer/runs/wrap_mech/p0"
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
for arm in ("AO", "AO_absmag", "AO_mid", "AO_high", "SO", "SO_absmag", "SO_high"):
    rows = []
    for s in range(800, 808):
        r = load(R, arm, s)
        A0 = np.array(r["ev"]["init_angles"]); A1 = action_angles(r["model"]).double().numpy()
        sel = np.abs(wrap(A1[0])) > 0.05
        canc = sel & (np.abs(wrap(A1.sum(0))) < 0.1)
        S0 = A0.sum(0)
        q = np.quantile(S0, 0.75)
        rows.append(dict(
            solved=r["solved"], canc=canc.mean(),
            S0_canc=np.median(S0[canc]) if canc.any() else np.nan, S0_non=np.median(S0[sel & ~canc]) if (sel & ~canc).any() else np.nan,
            canc_top=canc[S0 >= q].mean(), canc_bot=canc[S0 < q].mean(),
            S0_gt_pi=(S0 > np.pi).mean(),
            # (b) which angle changed sign: initially smaller-magnitude one?
            flip=((np.sign(A1) != np.sign(A0)).any(0)).mean(),
            flip_smaller=np.mean([(np.argmin(np.abs(A0[:, c])) == np.where(np.sign(A1[:, c]) != np.sign(A0[:, c]))[0][0])
                                  for c in range(A0.shape[1]) if (np.sign(A1[:, c]) != np.sign(A0[:, c])).sum() == 1] or [np.nan]),
            same0=(np.sign(A0[0]) == np.sign(A0[1])).mean(), opp1=(np.sign(A1[0]) != np.sign(A1[1])).mean(),
            omega_ratio=np.median(np.abs(A1).sum(0)[canc] / np.maximum(np.abs(A0).sum(0)[canc], 1e-9)) if canc.any() else np.nan))
    g = lambda k: np.nanmean([x[k] for x in rows])
    print(f"{arm:10s} solved {sum(x['solved'] for x in rows)}/8 | cancelling share {g('canc'):.2f} | median initial sum: cancelling "
          f"{g('S0_canc'):.2f} vs not {g('S0_non'):.2f} | cancel share in top-quartile initial sums {g('canc_top'):.2f} vs rest "
          f"{g('canc_bot'):.2f} | init sums > pi {g('S0_gt_pi'):.2f} | channels with a sign flip {g('flip'):.2f}, of single "
          f"flips the initially smaller angle {g('flip_smaller'):.2f} | same-sign at init {g('same0'):.2f} -> opposite at end "
          f"{g('opp1'):.2f} | |a+|+|a-| growth of cancelling channels x{g('omega_ratio'):.2f}")
