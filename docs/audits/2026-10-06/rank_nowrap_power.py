"""Power for RANK_NOWRAP_PREREG.md + Amendments 1-3: probability of each registered branch (analyze_rank_nowrap.decide,
imported, so the simulation runs the registered logic) under five scenarios, by seeds per cell.

WRAP-AWARE PER-STRATUM TRANSPORT (Amendment 3; ND 32-torus pools only -- the paper-torus runs' hard sets are only ~24%
wrap-only): each stored RANK_ND D2 run (rank_nowrap_hard_pool.json, set ND32: 8 rank-2 and 8 rank-3 runs, T=1024, 900
epochs) is a vector of held-out accuracies on copy / blank_out / hard_wrap / hard_plain. A simulated run draws one stored
run of the cell's type (with replacement) and keeps its per-stratum competence; its raw accuracy is reweighted by the
cell's grid (32: with the wrap-only hard stratum; 256: none), its registered p is its hard_plain accuracy (the same on
both grids by the transport assumption), HIT = p >= HIT_P. GENERAL therefore means "rank 2's per-stratum competence is
unchanged on the 256-torus". HALF draws a HIT rank-3 run or a non-HIT rank-2 run with probability 1/2 each. A32 ~ rank 2, B32 ~ rank 3, BL ~ rank 3 always.
Scenarios (AL, M32): PERIODIC (rank 3, rank 2); FIXED-MAP (rank 3, rank 3); GENERAL (rank 2, rank 2);
FIXED-MAP-AT-32 (rank 2, rank 3); HALF (half, rank 2). Permutation tests: 2000 Monte-Carlo relabellings here (the
registered analysis: exact at n=10). Fisher is exact.
Amendment 4: run for TWO pools -- ND32 (the 32-torus runs only: rank 3 never fails there, so these are UPPER BOUNDS)
and ALL (ND32 + the paper-torus runs RANK_MI / RANK3, whose rank-3 runs fail 2/8) -- and report, per truth, the
"wrong pole" rate (a pole verdict -- PERIODIC, FIXED MAP WAS THE LIMIT, RANK LIMIT IS GENERAL, FIXED MAP AT 32 -- other
than the true one) and, under HALF, the rate of any pole verdict. The registered n is N_SEEDS (10; the option 12).
Run from /home/prashr: python3 mapformer/docs/audits/2026-10-06/rank_nowrap_power.py [n_sim]
"""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer import analyze_rank_nowrap as A          # noqa: E402
from mapformer.stats_core import perm2_p                # noqa: E402

ST4 = ("copy", "blank_out", "hard_wrap", "hard_plain")


def main():
    J = json.load(open("/home/prashr/mapformer/docs/audits/2026-10-06/rank_nowrap_hard_pool.json"))
    share = {int(k): v for k, v in J["share"].items()}
    vec = lambda p: np.array([p["acc"][k] for k in ST4])
    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    for pool_name, sets in (("ND32", ("ND32",)), ("ALL", ("ND32", "T64"))):
        P2 = np.array([vec(p) for p in J["pool"] if p["rank"] == 2 and p["set"] in sets])
        P3 = np.array([vec(p) for p in J["pool"] if p["rank"] == 3 and p["set"] in sets])
        simulate(pool_name, P2, P3, share, n_sim)


POLES = ("PERIODIC CODE IS THE LIMIT", "FIXED MAP WAS THE LIMIT", "RANK LIMIT IS GENERAL", "FIXED MAP AT 32, LARGE TORUS FAILS")


def simulate(pool_name, P2, P3, share, n_sim):
    H = A.HIT_P
    print(f"\n#### pool {pool_name}: rank 2 {len(P2)} runs, HIT {int((P2[:, 3] >= H).sum())}; rank 3 {len(P3)} runs, HIT {int((P3[:, 3] >= H).sum())}")
    half = (P3[P3[:, 3] >= H], P2[P2[:, 3] < H])
    w = {N: np.array([share[N][k] for k in ST4]) for N in (32, 256)}
    for N in (32, 256):
        print(f"stratum weights {N}: " + "  ".join(f"{k} {share[N][k]:.3f}" for k in ST4) + f" (sum {w[N].sum():.3f})")
    rng = np.random.default_rng(0)
    pf = lambda a, b: perm2_p(a, b, n_mc=2000, max_exact=0)

    def draw(kind, n, N):
        if kind == "r2":
            V = P2[rng.integers(0, len(P2), n)]
        elif kind == "r3":
            V = P3[rng.integers(0, len(P3), n)]
        else:
            pick = rng.random(n) < 0.5
            V = np.where(pick[:, None], half[0][rng.integers(0, len(half[0]), n)], half[1][rng.integers(0, len(half[1]), n)])
        h = V[:, 3]; raw = V @ w[N]
        return int((h >= H).sum()), h, raw

    scen = (("PERIODIC", "r3", "r2", "PERIODIC CODE IS THE LIMIT"), ("FIXED-MAP", "r3", "r3", "FIXED MAP WAS THE LIMIT"),
            ("GENERAL", "r2", "r2", "RANK LIMIT IS GENERAL"), ("FIXED-MAP-AT-32", "r2", "r3", "FIXED MAP AT 32, LARGE TORUS FAILS"),
            ("HALF", "half", "r2", None))
    for n in (8, 10, 12):
        for name, kal, km, want in scen:
            cnt = Counter()
            for _ in range(n_sim):
                hit, h, raw = {}, {}, {}
                for k, kind, N in (("A32", "r2", 32), ("B32", "r3", 32), ("M32", km, 32), ("AL", kal, 256), ("BL", "r3", 256)):
                    hit[k], h[k], raw[k] = draw(kind, n, N)
                cnt[A.decide(hit, n, h, raw, pfun=pf)[0]] += 1
            wrong = sum(cnt[b_] for b_ in POLES if b_ != want) / n_sim
            tgt = (f"P(target) {cnt[want] / n_sim:.2f}, wrong pole {wrong:.2f} | " if want
                   else f"any pole verdict {wrong:.2f} | ")
            print(f"n={n:2d} {name:16s} {tgt}" + "  ".join(f"{b}: {v / n_sim:.2f}" for b, v in cnt.most_common()), flush=True)


if __name__ == "__main__":
    main()
