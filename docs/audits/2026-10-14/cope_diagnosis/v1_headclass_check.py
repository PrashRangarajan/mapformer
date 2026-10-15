"""Rule-9 check of m2's head classifier on constructed inputs: P=2 construction (k 17 -> LATTICE), the same table with
kappa 2 (k 18 -> ALIASED), kappa 0.3 (OFF), a P=3 construction (sigma 1.5, kappa -7.5 -> k 11, period-3 table -> LATTICE),
and a flat table (noTable)."""
import sys, types
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis")
import numpy as np
src = open("/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/m2_per_query.py").read().split("trajs = lc.trajectories")[0]
ns = {}; exec(src, ns)
tri2 = np.array([200.0 if p % 2 == 0 else -200.0 for p in range(2048)])
tri3 = np.array([200.0 if p % 3 == 0 else -100.0 for p in range(2048)])
for name, inc, z in (("P2 k17", dict(sigma=1.0, kappa=1.0), tri2), ("P2 kappa 2", dict(sigma=1.0, kappa=2.0), tri2),
                     ("P2 kappa 0.3", dict(sigma=1.0, kappa=0.3), tri2), ("P3 kappa -7.5", dict(sigma=1.5, kappa=-7.5), tri3),
                     ("flat", dict(sigma=1.0, kappa=1.0), np.zeros(2048)), ("P2 anti-phase", dict(sigma=1.0, kappa=1.0), -tri2)):
    print(name, ns["head_class"](inc, z), ns["periodP"](z, 3 if "P3" in name else 2))
