"""Calibration for TW_STATECHANGE_PREREG.md on STORED text-world checkpoints (eval-only, CPU; before any run of the batch).
The state-clause readouts have an exact analogue for ASIDE sentences ("she thought about a cat ."), which the stored
text-world models already carry: per-clause net displacement (R, field shift in moves, distortion), aside drift at
revisits, and the functional net-cancellation readout L_aside (= what L_sc measures for state clauses). Their per-seed
values set the thresholds and the power scenarios of the registered readouts. Also per-seed accuracy (eval mode and
re-scored) for the power of the accuracy contrasts. Stream: the plain text world (StateChangeWorld with p_take = p_drop
= 0 and TextWorld's vocabulary: byte-identical to TextWorld) on the batch's eval stream (held-out map 10000, np seed
10**6, 200 walks; drift on the first 40).
Runs: runs/textworld/p0 (Vanilla_r4 = MapWM, RoPE 1L; seeds 0-7) and runs/tw_normstep/p0 (MapWM, NormStep, NormStepNB,
DirOnly; seeds 10-17). Output: tw_statechange_calib.json / _out.txt."""
import json
import sys

import numpy as np
import torch

from mapformer import tw_statechange_readouts as R
from mapformer.train_tw_statechange import make_env

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
D = "/home/prashr/mapformer/docs/audits/2026-10-08"
RUNS = [(f"/home/prashr/mapformer/runs/textworld/p0/Vanilla_r4_L1_s{s}/Vanilla_r4.pt", "TW", s) for s in range(8)] + \
       [(f"/home/prashr/mapformer/runs/textworld/p0/RoPE_L1_s{s}/RoPE.pt", "TW", s) for s in range(8)] + \
       [(f"/home/prashr/mapformer/runs/tw_normstep/p0/{a}_s{s}/{a}.pt", "TWNS", s)
        for s in range(10, 18) for a in ("MapWM", "NormStep", "NormStepNB", "DirOnly")]
env = make_env(R.HELDOUT, p_take=0.0, p_drop=0.0, state_vocab=False)
W = R.walks(env)
out = {}
for ck, batch, s in RUNS:
    r = R.readouts(ck, W=W)
    key = f"{batch}:{r['arm']}_s{s}"
    out[key] = r
    a = r["acc"]
    line = f"{key:22s} acc {a['intact']['all']:.4f} rs {a['intact_rs']['all']:.4f}"
    if "geom" in r:
        g = r["geom"]
        line += (f"  R_aside {g['R_aside']:.3f} shift {g['shift_aside']:.3f} moves dist {g['dist_aside']:.3f}"
                 f"  drift_aside {r['drift_aside']:.3f} rad  L_aside {r['L_aside']:+.4f} (n {r['n']['T1a']})"
                 f"  reliance {r['reliance']:.3f}  clock ch {r['clock_channels']}")
    print(line, flush=True)
json.dump(out, open(f"{D}/tw_statechange_calib.json", "w"), indent=1, default=float)
print("\n== summary by arm (median [min, max]) ==")
for arm in ("TW:MapWM", "TWNS:MapWM", "TWNS:NormStep", "TWNS:NormStepNB", "TWNS:DirOnly", "TW:RoPE"):
    rs = [v for k, v in out.items() if k.startswith(arm + "_")]
    def q(f):
        x = np.array([f(v) for v in rs], float)
        return f"{np.median(x):.4f} [{x.min():.4f}, {x.max():.4f}]"
    print(f"{arm:16s} acc {q(lambda v: v['acc']['intact']['all'])}  rs {q(lambda v: v['acc']['intact_rs']['all'])}")
    if "geom" in rs[0]:
        print(f"{'':16s} R_aside {q(lambda v: v['geom']['R_aside'])}  shift_aside {q(lambda v: v['geom']['shift_aside'])}"
              f"  L_aside {q(lambda v: v['L_aside'])}  drift_aside {q(lambda v: v['drift_aside'])}")
