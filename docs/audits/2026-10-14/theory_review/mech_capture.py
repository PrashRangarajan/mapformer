"""THEORY REVIEW post hoc: final cancel share vs distance of the initial sum from 2 pi (AO_high, SO_high, MECHANISM), beside
the one-channel linear-response flow's capture (potential.py's V, plain gradient) for the same distances."""
import sys
sys.path.insert(0, "/home/prashr"); sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/theory_review")
import numpy as np
from mapformer.analyze_wrap_mech import load
from mapformer.model_wrap_exist import action_angles
import potential as P
R = "/home/prashr/mapformer/runs/wrap_mech/p0"
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
edges = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
for arm in ("AO_high", "SO_high"):
    D, C = [], []
    for s in range(800, 808):
        r = load(R, arm, s); A0 = np.array(r["ev"]["init_angles"]); A1 = action_angles(r["model"]).double().numpy()
        sel = np.abs(wrap(A1[0])) > 0.05
        D.append(np.abs(A0.sum(0) - 2 * np.pi)[sel]); C.append((np.abs(wrap(A1.sum(0))) < 0.1)[sel])
    D = np.concatenate(D); C = np.concatenate(C)
    print(arm, " | ".join(f"|S0-2pi| in [{a},{b}): n={((D>=a)&(D<b)).sum()} canc {C[(D>=a)&(D<b)].mean():.2f}" for a, b in zip(edges[:-1], edges[1:]) if ((D>=a)&(D<b)).sum() >= 5))
Ws, Wa, _ = P.pair_hist(n=600)
W = Ws - Wa
rng = np.random.RandomState(5); A0 = rng.uniform(1.7, 4.5, (120, 2))
ends = np.array([P.flow(W, a, "abs", steps=3000) for a in A0])
D = np.abs(A0.sum(1) - 2 * np.pi); C = np.abs(wrap(ends.sum(1))) < 0.1
print("1-channel linear flow (abs, plain GD)", " | ".join(f"[{a},{b}): n={((D>=a)&(D<b)).sum()} canc {C[(D>=a)&(D<b)].mean():.2f}" for a, b in zip(edges[:-1], edges[1:]) if ((D>=a)&(D<b)).sum() >= 3))
