"""THEORY REVIEW post hoc: final cancel share by initial per-channel sum bin (MECHANISM checkpoints). Divide account: the
basin of the wrap (sum 2 pi) is roughly (pi, 3 pi); of the through-zero cancel (signed) roughly (-pi, pi)."""
import sys
sys.path.insert(0, "/home/prashr")
import numpy as np
from mapformer.analyze_wrap_mech import load
from mapformer.model_wrap_exist import action_angles
R = "/home/prashr/mapformer/runs/wrap_mech/p0"
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
bins = [0, 0.5, 1.0, 2.0, np.pi, 1.5 * np.pi, 2 * np.pi, 2.5 * np.pi, 3 * np.pi]
for arm in ("AO", "AO_mid", "AO_high", "SO_absmag", "SO_high"):
    S0s, cs, sels = [], [], []
    for s in range(800, 808):
        r = load(R, arm, s); A0 = np.array(r["ev"]["init_angles"]); A1 = action_angles(r["model"]).double().numpy()
        sel = np.abs(wrap(A1[0])) > 0.05; canc = np.abs(wrap(A1.sum(0))) < 0.1
        S0s.append(np.abs(A0).sum(0)); cs.append(canc); sels.append(sel)
    S0 = np.concatenate(S0s); c = np.concatenate(cs); sel = np.concatenate(sels)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (S0 >= lo) & (S0 < hi) & sel
        if m.sum() >= 5:
            out.append(f"[{lo:.2f},{hi:.2f}) n={m.sum():4d} canc {c[m].mean():.2f}")
    print(f"{arm:10s} " + " | ".join(out))
