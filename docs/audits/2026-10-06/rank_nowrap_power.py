"""Power for RANK_NOWRAP_PREREG.md + Amendments 1-2: probability of each registered branch (analyze_rank_nowrap.decide,
imported, so the simulation runs the registered logic) under five scenarios, by seeds per cell.

PER-STRATUM TRANSPORT (Amendment 2; replaces Amendment 1's assumption that floor-relative accuracy is grid-invariant):
each stored per-head run (rank_nowrap_hard_pool.json, built by rank_nowrap_hard_validate.py: RANK_ND D2 + RANK_MI /
RANK3, 16 runs per rank, T=1024, 900 epochs) is a vector of held-out accuracies on the strata copy / blank_out /
retrace_miss. A simulated run in a cell draws one stored run of the cell's type (with replacement) and keeps its
per-stratum accuracies; its raw accuracy is reweighted by the cell's grid (stratum shares of the 32- or 256-torus
held-out stream), its h is its retrace_miss accuracy, HIT = h >= HIT_H. HALF draws a HIT rank-3 run or a non-HIT rank-2
run with probability 1/2 each. A32 ~ rank 2, B32 ~ rank 3, BL ~ rank 3 always.
Scenarios (AL, M32): PERIODIC (rank 3, rank 2); FIXED-MAP (rank 3, rank 3); GENERAL (rank 2, rank 2);
FIXED-MAP-AT-32 (rank 2, rank 3); HALF (half, rank 2). Permutation tests: 2000 Monte-Carlo relabellings here (the
registered analysis: exact at n=10). Fisher is exact.
Run from /home/prashr: python3 mapformer/docs/audits/2026-10-06/rank_nowrap_power.py [n_sim]
"""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer import analyze_rank_nowrap as A          # noqa: E402
from mapformer.stats_core import perm2_p                # noqa: E402

ST3 = ("copy", "blank_out", "retrace_miss")


def main():
    J = json.load(open("/home/prashr/mapformer/docs/audits/2026-10-06/rank_nowrap_hard_pool.json"))
    share = {int(k): v for k, v in J["share"].items()}
    vec = lambda p: np.array([p["acc"][k] for k in ST3])
    P2 = np.array([vec(p) for p in J["pool"] if p["rank"] == 2]); P3 = np.array([vec(p) for p in J["pool"] if p["rank"] == 3])
    H = A.HIT_H
    print(f"pools: rank 2 {len(P2)} runs, HIT {int((P2[:, 2] >= H).sum())}; rank 3 {len(P3)} runs, HIT {int((P3[:, 2] >= H).sum())}")
    half = (P3[P3[:, 2] >= H], P2[P2[:, 2] < H])
    w = {N: np.array([share[N][k] for k in ST3]) for N in (32, 256)}
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
        h = V[:, 2]; raw = V @ w[N]
        return int((h >= H).sum()), h, raw

    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 400
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
            tgt = f"P(target) {cnt[want] / n_sim:.2f} | " if want else ""
            print(f"n={n:2d} {name:16s} {tgt}" + "  ".join(f"{b}: {v / n_sim:.2f}" for b, v in cnt.most_common()), flush=True)


if __name__ == "__main__":
    main()
