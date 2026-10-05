"""Refined content-phase lesion for MapWM (neuro_rank2_mech.py's strip_psi also changed action-key scores, i.e. the
action/observation type gating): strip psi only on observation keys, keep every action-key score intact. Post hoc, CPU."""
import sys
import numpy as np
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-05")
import neuro_rank2_mech as NR
toks, revs, pos = NR.walks()
inp, tgt, rv = toks[:, :-1], toks[:, 1:], revs[:, 1:]
for arm, seeds in [("Vanilla", range(10, 26)), ("Vanilla_r4", range(10, 18))]:
    d = []
    for s in seeds:
        m = NR.load(arm, s)
        a0 = (NR.forward(m, inp)[0].argmax(-1) == tgt)[rv].float().mean().item()
        a1 = (NR.forward(m, inp, "strip_obs")[0].argmax(-1) == tgt)[rv].float().mean().item()
        d.append(a1 - a0); print(f"{arm:11s} s{s} intact {a0:.4f} strip_psi_on_obs_keys {a1:.4f} diff {a1 - a0:+.4f}", flush=True)
    print(f"== {arm}: mean diff {np.mean(d):+.4f}; up {sum(x > 1e-4 for x in d)} down {sum(x < -1e-4 for x in d)}")
