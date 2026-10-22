"""Per-pass same-cell attention with the head chosen per seed (POST HOC, CPU, seconds).

pass_strata.py's tables average each attention column by head INDEX across seeds; head labels are a symmetry (rule 8),
and at L pass 2 the retrieval head is head 1 on seeds 4, 6, 7. Here each seed contributes its best head (max over the two
heads of that seed's value), then the 8 seeds are averaged (docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md M-RC2).
Reads out/pass_strata_{A,L,L4,C}.json (unchanged). Columns: obs_mass = attention mass on earlier same-cell observation
keys; argmax = share of queries whose top-attended key is a same-cell observation; act_mass = mass on same-cell action keys.
  python3 pass_strata_besthead.py > out/PASS_STRATA_BESTHEAD.txt
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import OUT  # noqa

NP = {"A": 1, "L": 4, "L4": 4, "C": 1}
print("| arm | pass | mass all | argmax all | mass long | mass wrap | argmax wrap | act mass all (max single seed-head) |")
print("|---|---|---|---|---|---|---|---|")
for a in ("A", "L", "L4", "C"):
    d = json.load(open(f"{OUT}/pass_strata_{a}.json"))
    for p in range(NP[a]):
        def col(stratum, key, best=True):
            v = np.array([np.array(d[str(s)][stratum][key])[p] for s in range(8)])      # (8 seeds, 2 heads)
            return v.max(1).mean() if best else v.max()
        print(f"| {a} | {p + 1} | {col('all', 'obs_mass'):.2f} | {col('all', 'argmax_same_cell'):.2f} | "
              f"{col('long', 'obs_mass'):.2f} | {col('wrap', 'obs_mass'):.2f} | {col('wrap', 'argmax_same_cell'):.2f} | "
              f"{col('all', 'act_mass'):.2f} ({col('all', 'act_mass', best=False):.2f}) |")
