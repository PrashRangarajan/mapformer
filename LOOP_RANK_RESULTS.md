# H1: does a search aid recover rank 2? -- results (2026-09-27)

Pre-registration `LOOP_RANK_PREREG.md`; runs `runs/loop_rank` (+ `runs/loop_rank_repro`); full output
`LOOP_RANK_ANALYSIS.txt` (`python3 -m mapformer.analyze_loop_rank`). Torus, trained and tested at
T=1024, 900 epochs, 8 seeds, one batch. L is bit-identical to A at every seed (same 204,373 params)
and differs only in applying the block 4x.

## Registered verdict: UNMEASURED

H1 required L to fire against A **and** solve >= 4/8. It fired on accuracy and solved 2/8.

| arm | params | passes | SOLVED (<0.05) | T=1024 acc | vs A: Fisher / permutation |
|---|---|---|---|---|---|
| A r=2, 1 layer | 204,373 | 1 | 0/8 | 0.894 | -- |
| L r=2 + loop x4 | 204,373 | 4 | **2/8** | 0.973 | 0.47 / **0.0034** (+0.079, CI [+0.026, +0.131]) |
| L4 r=2, 4 real layers | 799,189 | 4 | **5/8** | 0.990 | **0.0256** / **0.0012** (+0.096) |
| C r=4, 1 layer | 204,757 | 1 | 8/8 | 0.998 | reference |

L4 - L: +0.017 accuracy (perm p 0.0009), SOLVED 5/8 vs 2/8 (Fisher 0.31). Holm-adjusted p: 0.0034 /
0.0028 / 0.0028. **Reproduction of A seed 0 was bit-exact** (max per-epoch loss diff 0.0 over 900).

## What the batch actually shows

**Three regimes, not two.** The final losses separate cleanly and the accuracy ordering follows:

| arm | tail losses, sorted |
|---|---|
| A | 0.051 0.177 0.308 0.321 0.404 0.587 0.605 0.675 |
| L | 0.045 0.045 0.053 0.066 0.066 0.070 0.114 0.127 |
| L4 | 0.011 0.031 0.042 0.047 0.047 0.054 0.064 0.070 |
| C | 0.002 0.003 0.003 0.003 0.004 0.004 0.004 0.010 |

Extra iterations move rank 2 out of A's regime, but **not into r=4's**: C sits an order of magnitude
lower on every seed. So more search gets rank 2 close to the solution and does not find the basin
r=4 finds.

**Threshold sensitivity, reported because the registered cutoff lands inside the new arms' spread**
(rule 5). The 0.05 cutoff was calibrated on the 300-epoch batch, where nothing sat between 0.05 and
0.15; here six of sixteen L and L4 runs do.

| cutoff | A | L | L4 | C | L - A Fisher |
|---|---|---|---|---|---|
| 0.03 | 0 | 0 | 1 | 8 | 1.00 |
| **0.05 (registered)** | **0** | **2** | **5** | **8** | **0.47** |
| 0.08 | 1 | 6 | 8 | 8 | 0.041 |
| 0.15 | 1 | 8 | 8 | 8 | 0.0014 |

At any cutoff from 0.08 up, H1's solved-count condition would have been met. **The registered verdict
stands as UNMEASURED**; this is why the verdict is uncertain, not a replacement for it.

## Reading

- **Partial support for the search account.** Both search aids move rank 2 a long way (accuracy 0.894
  -> 0.973 -> 0.990 against r=4's 0.998) at a bottleneck that provably can represent and hold the
  solution (`RANK_PROJ_RESULTS.md`). Rank 2 is not stuck for lack of capacity.
- **But it is not the clean "search at constant capacity" result H1 predicted.** The matched-parameter
  loop did not reach the registered bar, and **4 real layers beat it** on accuracy (p 0.0009) and
  solved more (5/8 vs 2/8). The registered compute reading therefore fires as "4x compute is
  sufficient; the loop is not special" -- and here it is parameters plus compute that do best.
- **Rank 4 remains in a class of its own.** Nothing at rank 2 reached its loss regime on any seed.
  Spending rank is still the only intervention that takes every seed to the solution.

Scope: torus, T=1024, n_heads 2, d 128, one recipe, 900-epoch budget, one loop count (4). Untested:
more loops, a longer budget (several L and L4 runs are still descending), the loop at rank 4.
