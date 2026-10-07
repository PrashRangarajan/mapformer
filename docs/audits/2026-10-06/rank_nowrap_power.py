"""Power for RANK_NOWRAP_PREREG.md: probability of each registered branch (analyze_rank_nowrap.decide, imported, so the
simulation runs the registered logic) under three scenarios, by seeds per cell.

Per-seed pools (stored batches, T=1024, 900 epochs, per-head rank, one layer, 2 heads):
  rank 2: RANK_ND D2 Vanilla_r2ph (32-torus, seeds 0-7, 1/8 SOLVED) + RANK_MI Vanilla_r2ph (64-torus paper env, 2/8)
  rank 3: RANK_ND D2 Vanilla_r3ph (8/8) + RANK3 Vanilla_r3ph (64-torus, 6/8)
A simulated run draws SOLVED ~ Bernoulli(p) and an accuracy from the pool's runs of the same status, converted to
floor-relative units with its source floor (retrace 0.750 on the 32-torus ND stream, 0.844 on the 64-torus) and back
with the target cell's floor (0.750 at 32, 0.872 at 256: rank_nowrap_gate_out.txt). Scenarios for AL (rank 2 at 256):
  PERIODIC  AL behaves like rank 3     (p = p_r3)
  GENERAL   AL behaves like rank 2     (p = p_r2)
  HALF      AL solves half the time    (p = 0.5, accuracies mixed accordingly)
A32 ~ rank 2, B32 ~ rank 3, BL ~ rank 3 throughout. The permutation test uses 2000 Monte-Carlo relabellings here (the
registered analysis uses stats_core's exact / 200k MC); Fisher is exact.
Run from /home/prashr: python3 mapformer/docs/audits/2026-10-06/rank_nowrap_power.py
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
    p2 = np.mean([x[0] for x in P2]); p3 = np.mean([x[0] for x in P3])
    print(f"pools: rank 2 SOLVED {sum(x[0] for x in P2)}/{len(P2)} (p {p2:.3f}); rank 3 {sum(x[0] for x in P3)}/{len(P3)} (p {p3:.3f})")
    solved_pool = [r for sv, r in P2 + P3 if sv]
    fail2 = [r for sv, r in P2 if not sv]; fail3 = [r for sv, r in P3 if not sv]
    rng = np.random.default_rng(0)
    pf = lambda a, b: perm2_p(a, b, n_mc=2000, max_exact=0)

    def draw(p, n, floor, fail_pool):
        """SOLVED ~ Bernoulli(p); a solved run's accuracy from every stored solved run, a failed run's from the pool of
        its type (rank-2 failures for rank-2-like cells, rank-3 failures for rank-3-like cells), floor-relative."""
        sv = rng.random(n) < p
        rel = np.array([rng.choice(solved_pool) if s else rng.choice(fail_pool) for s in sv])
        raw = np.minimum(floor + rel * (1 - floor), 1.0)
        return int(sv.sum()), raw, (raw - floor) / (1 - floor)

    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    for n in (8, 10, 12):
        for name, pAL in (("PERIODIC", p3), ("GENERAL", p2), ("HALF", 0.5)):
            cnt = Counter()
            for _ in range(n_sim):
                sol, raw, rel = {}, {}, {}
                for k, p, fl, fp in (("A32", p2, 0.750, fail2), ("B32", p3, 0.750, fail3),
                                     ("AL", pAL, 0.872, fail3 if name == "PERIODIC" else fail2), ("BL", p3, 0.872, fail3)):
                    sol[k], raw[k], rel[k] = draw(p, n, fl, fp)
                cnt[A.decide(sol, n, raw, rel, pfun=pf)[0]] += 1
            print(f"n={n:2d} {name:9s} " + "  ".join(f"{b}: {v / n_sim:.2f}" for b, v in cnt.most_common()), flush=True)


if __name__ == "__main__":
    main()
