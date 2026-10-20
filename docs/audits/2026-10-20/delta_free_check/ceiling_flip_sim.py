"""Audit 2026-10-20, MAJOR 1: how often does a sub-noise path difference flip the registered primary to
'NOT SUPPORTED (helps both)'? PP per seed resampled from PAPER2X2's MapPoPE-Flat T=128 values; FP either 1.000 on every
seed (the free path arm saturates) or resampled like PP; dI ~ N(m, 0.06), PI ~ 0.68 + N(0, 0.05)."""
import numpy as np
from collections import Counter
import mapformer.analyze_delta_free as A
emp = np.array([1, 1, 1, 1, 0.9983, 0.9961, 1, 0.9993])
rng = np.random.default_rng(1); S = list(range(8))
for scen in ("FP=1.000 every seed", "FP resampled like PP"):
    for m in (0.08, 0.12):
        c = Counter(); n = 1500
        for _ in range(n):
            pi = 0.68 + rng.normal(0, 0.05, 8); fi = pi + rng.normal(m, 0.06, 8)
            pp = rng.choice(emp, 8); fp = np.ones(8) if scen.startswith("FP=1") else rng.choice(emp, 8)
            acc = {A.PI: dict(zip(S, pi)), A.FI: dict(zip(S, fi)), A.PP: dict(zip(S, pp)), A.FP: dict(zip(S, fp))}
            sol = {A.PP: {s: True for s in S}, A.FP: {s: True for s in S}}
            c[A.primary_label(acc, sol, S, 0.9381)[0]] += 1
        print(scen, "true dI", m, {k: round(v / n, 3) for k, v in c.most_common()})
