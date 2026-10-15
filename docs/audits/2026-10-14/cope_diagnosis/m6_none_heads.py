"""M6 (post hoc): heads of runs whose query type has NO period-P count head (regime NONE in m2): sigma, kappa, obs-gate
spread tau, and the period-2 / period-3 share of the effective table. Question: near-miss of the period lock (sigma close
to 1 or 1.5 but outside 1/(2N)), or a different (content / recency) solution?"""
import json, sys
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis")
import numpy as np, torch
import lib_cope as lc
D = "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/"
src = open(D + "m2_per_query.py").read().split("trajs = lc.trajectories")[0]
ns = {"__name__": "m2lib"}; exec(src, ns); periodP = ns["periodP"]
m2 = json.load(open(D + "m2_out.json"))
none = {(o["batch"], o["arm"], o["seed"], o["a"]) for o in m2 if o["regime"] == "NONE"}
paths = {(b, a, s): p for b, a, s, p in lc.run_dirs()}
lock = []
print("run a h | sigma kappa tau | p2 share p3 share | z sd | |2 sigma - P| for the nearest P in {2,3}")
for (b, arm, s, a) in sorted(none):
    m, ck = lc.load(paths[(b, arm, s)])
    L, G, Z = lc.tables(m)
    inc = lc.increments(G)
    for h in range(2):
        x = inc[(h, a)]
        p2 = periodP(Z[h, a], 2)[0]; p3 = periodP(Z[h, a], 3)[0]
        P = 2 if abs(2 * x["sigma"] - 2) < abs(2 * x["sigma"] - 3) else 3
        err = abs(2 * x["sigma"] - P); lock.append((err, x["tau"], max(p2, p3)))
        print(f"{b[:3]} {arm:9s} s{s} a{a} h{h} | {x['sigma']:.3f} {x['kappa']:+.3f} {x['tau']:.3f} | {p2:.2f} {p3:.2f} | "
              f"{Z[h, a, :255].std():6.2f} | {err:.3f}")
e = np.array(lock)
print(f"\nNONE heads n {len(e)}: |2 sigma - P| median {np.median(e[:,0]):.3f} (min {e[:,0].min():.3f}); tau median {np.median(e[:,1]):.3f}; "
      f"heads with |2 sigma - P| < 0.01: {(e[:,0] < 0.01).sum()}, with tau > 0.1: {(e[:,1] > 0.1).sum()}; max period share median {np.median(e[:,2]):.2f}")
