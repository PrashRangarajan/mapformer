"""Power of MAPPOPE_PAIR's rank-2 primary contrasts, by resampling paper2x2's per-seed T=128 accuracies (MapWM r2 and
MapPoPE-Flat r2, 8 seeds each): P(fires positive) = P(perm p < .05 and d >= 0.01) at n per arm, for the full effect
(TOTAL, or SCORE/COUNT if one change carries everything) and for a 50/50 split (the middle arm a per-seed mix)."""
import json, numpy as np
from mapformer.stats_core import perm2_p
J = json.load(open("/home/prashr/mapformer/_PAPER2X2_RAW.json"))
V = np.array([x[1] for x in J["0.0|Vanilla|128"]]); F = np.array([x[1] for x in J["0.0|MapPoPE-Flat|128"]])
rng = np.random.default_rng(0)
def fire(x, y): return perm2_p(x, y, n_mc=4000)["p"] < 0.05 and y.mean() - x.mean() >= 0.01
for n in (8, 12, 16):
    full = half = 0; K = 300
    for _ in range(K):
        x = rng.choice(V, n); y = rng.choice(F, n)
        mid = np.where(rng.random(n) < 0.5, rng.choice(V, n), rng.choice(F, n))   # half-effect arm: a 50/50 mixture
        full += fire(x, y); half += fire(x, mid)
    print(f"n={n:2d} per arm: P(full effect fires) {full / K:.2f};  P(half effect fires) {half / K:.2f}")
