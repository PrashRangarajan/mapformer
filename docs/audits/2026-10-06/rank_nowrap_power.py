"""Power for RANK_NOWRAP_PREREG.md + Amendment 1: probability of each registered branch (analyze_rank_nowrap.decide,
imported, so the simulation runs the registered logic) under five scenarios, by seeds per cell.

Per-seed pools of stored runs (T=1024, 900 epochs, per-head rank, one layer, 2 heads), as floor-relative held-out
accuracy rel = (acc - f) / (1 - f) with the source stream's retrace-or-blank floor (0.750 on the 32-torus ND stream,
0.844 on the 64-torus):
  rank 2: RANK_ND D2 Vanilla_r2ph (seeds 0-7) + RANK_MI Vanilla_r2ph
  rank 3: RANK_ND D2 Vanilla_r3ph + RANK3 Vanilla_r3ph
A simulated run of a "rank-2-like" cell draws a stored rank-2 run's rel at random (with replacement); "rank-3-like"
likewise; HALF draws a HIT rank-3 run or a non-HIT rank-2 run with probability 1/2 each. raw = f + rel (1 - f) with the
target cell's floor (0.750 at 32, 0.872 at 256), HIT = rel >= 0.90. A32 ~ rank 2, B32 ~ rank 3, BL ~ rank 3 always.
Scenarios (AL, M32): PERIODIC (rank 3, rank 2); MEMORISATION (rank 3, rank 3); GENERAL (rank 2, rank 2);
MEMO-AT-32 (rank 2, rank 3); HALF (half, rank 2). The hard-target qualifier is not simulated (hm=None). Permutation tests
use 2000 Monte-Carlo relabellings here (the registered analysis: exact at n=10). Fisher is exact.
Run from /home/prashr: python3 mapformer/docs/audits/2026-10-06/rank_nowrap_power.py [n_sim]
"""
import json
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
from mapformer import analyze_rank_nowrap as A          # noqa: E402
from mapformer.stats_core import classify_run, perm2_p  # noqa: E402

RP = "/home/prashr/mapformer"


def pool(ck_fmt, accs, floor):
    out = []
    for s, acc in enumerate(accs):
        sv = classify_run(torch.load(ck_fmt.format(s=s), map_location="cpu", weights_only=False)["losses"])["registered"] == "SOLVED"
        out.append((sv, (acc - floor) / (1 - floor)))
    return out


def main():
    nd = json.load(open(f"{RP}/RANK_ND.json"))["D2"]["acc"]
    mi = {s: a for s, a, _ in json.load(open(f"{RP}/RANK_MI.json"))["0.0|Vanilla_r2ph|1024"]}
    r3 = {s: a for s, a, _ in json.load(open(f"{RP}/RANK3.json"))["0.0|Vanilla_r3ph|1024"]}
    P2 = (pool(f"{RP}/runs/rank_nd/D2/Vanilla_r2ph_s{{s}}/Vanilla_r2ph.pt", [nd["Vanilla_r2ph"]["1024"][str(s)] for s in range(8)], 0.750)
          + pool(f"{RP}/runs/rank_mi/p0/Vanilla_r2ph_s{{s}}/Vanilla_r2ph.pt", [mi[s] for s in range(8)], 0.844))
    P3 = (pool(f"{RP}/runs/rank_nd/D2/Vanilla_r3ph_s{{s}}/Vanilla_r3ph.pt", [nd["Vanilla_r3ph"]["1024"][str(s)] for s in range(8)], 0.750)
          + pool(f"{RP}/runs/rank3/p0/Vanilla_r3ph_s{{s}}/Vanilla_r3ph.pt", [r3[s] for s in range(8)], 0.844))
    H = A.HIT_REL
    for nm, P in (("rank 2", P2), ("rank 3", P3)):
        print(f"pool {nm}: loss-SOLVED {sum(x[0] for x in P)}/{len(P)}, HIT (rel >= {H}) {sum(x[1] >= H for x in P)}/{len(P)}, "
              f"agreement {sum((x[1] >= H) == x[0] for x in P)}/{len(P)}; rel " + " ".join(f"{x[1]:+.2f}{'S' if x[0] else ''}" for x in P))
    r2 = np.array([x[1] for x in P2]); r3v = np.array([x[1] for x in P3])
    half_pool = (r3v[r3v >= H], r2[r2 < H])
    rng = np.random.default_rng(0)
    pf = lambda a, b: perm2_p(a, b, n_mc=2000, max_exact=0)

    def draw(kind, n, floor):
        if kind == "r2":
            rel = rng.choice(r2, n)
        elif kind == "r3":
            rel = rng.choice(r3v, n)
        else:
            rel = np.where(rng.random(n) < 0.5, rng.choice(half_pool[0], n), rng.choice(half_pool[1], n))
        raw = np.minimum(floor + rel * (1 - floor), 1.0); rel = (raw - floor) / (1 - floor)
        return int((rel >= H).sum()), raw, rel

    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    scen = (("PERIODIC", "r3", "r2", "PERIODIC CODE IS THE LIMIT"), ("MEMORISATION", "r3", "r3", "MAP MEMORISATION WAS THE LIMIT"),
            ("GENERAL", "r2", "r2", "RANK LIMIT IS GENERAL"), ("MEMO-AT-32", "r2", "r3", "MEMORISATION AT 32, LARGE TORUS FAILS"),
            ("HALF", "half", "r2", None))
    for n in (8, 10, 12):
        for name, kal, km, want in scen:
            cnt = Counter()
            for _ in range(n_sim):
                hit, raw, rel = {}, {}, {}
                for k, kind, fl in (("A32", "r2", 0.750), ("B32", "r3", 0.750), ("M32", km, 0.750), ("AL", kal, 0.872), ("BL", "r3", 0.872)):
                    hit[k], raw[k], rel[k] = draw(kind, n, fl)
                cnt[A.decide(hit, n, raw, rel, None, pfun=pf)[0]] += 1
            tgt = f"P(target) {cnt[want] / n_sim:.2f} | " if want else ""
            print(f"n={n:2d} {name:12s} {tgt}" + "  ".join(f"{b}: {v / n_sim:.2f}" for b, v in cnt.most_common()), flush=True)


if __name__ == "__main__":
    main()
