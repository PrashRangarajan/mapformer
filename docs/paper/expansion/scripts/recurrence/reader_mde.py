"""Minimum detectable effect of the one-pass reader's code-quality contrasts (POST HOC, CPU, seconds);
docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md M-RC1. Reads out/phase_reader_soft_{A,L,L4,C}.json (soft reader accuracy
per seed and stratum). For each contrast b - a (8 vs 8 seeds, unpaired): difference, permutation p (stats_core.perm2_p),
and the exact-t two-sample MDE at alpha 0.05, power 0.80: (t_{0.975, 14} + t_{0.80, 14}) * sqrt(s_a^2 / 8 + s_b^2 / 8).
  python3 reader_mde.py > out/READER_MDE.txt"""
import json
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import OUT  # noqa
sys.path.insert(0, "/home/prashr")
from mapformer.stats_core import perm2_p  # noqa: E402

sr = {a: json.load(open(f"{OUT}/phase_reader_soft_{a}.json")) for a in ("A", "L", "L4", "C")}
k = stats.t.ppf(0.975, 14) + stats.t.ppf(0.80, 14)
print("| stratum | contrast | difference | perm p | exact-t MDE | verdict |")
print("|---|---|---|---|---|---|")
for st in ("all", "long", "wrap"):
    for lo, hi in (("A", "L"), ("A", "L4"), ("L", "C"), ("L4", "C")):
        a = np.array([sr[lo][str(s)][st] for s in range(8)]); b = np.array([sr[hi][str(s)][st] for s in range(8)])
        d = b.mean() - a.mean(); m = k * np.sqrt(a.var(ddof=1) / 8 + b.var(ddof=1) / 8)
        print(f"| {st} | {hi} - {lo} | {d:+.3f} | {perm2_p(a, b)['p']:.4f} | {m:.3f} | {'detectable' if abs(d) > m else 'UNMEASURED'} |")
