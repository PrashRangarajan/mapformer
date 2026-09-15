# COUNTER batch results (`COUNTER_BATCH.md`)

5 arms x 4 seeds, one batch, recency recipe (k_max 64, trained at T=1024). Chance 0.0625. Accuracy is mean
+/- sd over seeds. n=4, so everything here is exploratory by the project's convention.

| arm | acc T=1024 | acc T=2048 | final loss | first epoch with loss < 0.5 (per seed) | acc by k: 1-16 / 17-32 / 33-48 / 49-64 |
|---|---|---|---|---|---|
| `WM_Counter` (counter installed; offset via content phase) | **1.000 +/- 0.000** | **1.000 +/- 0.000** | 0.005 | 14, 13, 14, 15 | 1.00 / 1.00 / 1.00 / 1.00 |
| `EM_Counter` (counter installed; rewind via rank-4 bottleneck) | 0.740 +/- 0.081 | 0.583 +/- 0.077 | 0.951 | never | 0.82 / 0.67 / 0.68 / 0.76 |
| `TEM_Counter` (counter installed; rewind via full transform) | 0.337 +/- 0.026 | 0.228 +/- 0.029 | 2.471 | never | 0.82 / 0.20 / 0.14 / 0.16 |
| `Vanilla_r4` (MapWM from scratch) | 1.000 +/- 0.000 | 0.962 +/- 0.028 | 0.015 | 47, 49, 66, 86 | 1.00 / 1.00 / 1.00 / 1.00 |
| `VanillaEM_P0_r4` (MapEM from scratch) | 0.627 +/- 0.116 | 0.540 +/- 0.141 | 1.265 | never | 0.66 / 0.61 / 0.64 / 0.61 |

Paired by seed, T=1024:

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| WM_Counter - EM_Counter | +0.260 | 0.081 | 0.113 | 4/4 | detectable |
| WM_Counter - TEM_Counter | +0.663 | 0.026 | 0.037 | 4/4 | detectable |
| EM_Counter - TEM_Counter | +0.403 | 0.070 | 0.098 | 4/4 | detectable |
| WM_Counter - WM from scratch | +0.000 | 0.000 | 0.000 | 0/4 | (both at ceiling) |
| EM_Counter - EM from scratch | +0.113 | 0.197 | 0.275 | 3/4 | unmeasured |
| WM from scratch - EM from scratch | +0.373 | 0.116 | 0.162 | 4/4 | detectable |

Long-offset accuracy (k = 33-64) per seed:
- `WM_Counter`: 1.00, 1.00, 1.00, 1.00
- `EM_Counter`: 0.71, 0.85, 0.58, 0.75
- `TEM_Counter`: 0.18, 0.12, 0.14, 0.17

## Reading rules, as written before training

- **R1: MET.** With the counter given, MapWM reaches every offset on every seed (k 33-64: 1.00, 4/4), and the
  loss falls below 0.5 by epoch 13-15, against 47-86 from scratch.
- **R2: NOT MET.** The rule needed both position-side arms below 0.5 at k 33-64. TEM is (0.12-0.18), but
  MapEM through its bottleneck is not (0.58-0.85). Position-side rewinds do not uniformly fail at long
  offsets.
- **R3.** WM_Counter - EM_Counter = +0.260 (MDE 0.113, 4/4).
- **R4.** Installing the counter costs MapWM nothing in accuracy and saves training time. For MapEM the gain
  over from scratch is unmeasured (+0.113, MDE 0.275).

## What this says (exploratory, n=4)

- **Learning the counter is not EM's problem.** Given a perfect counter, MapEM still trails MapWM by 0.260 on
  every seed and never gets its loss below 0.5. MapWM with the same counter solves the task outright.
- **The rank-4 bottleneck is not the explanation either.** The arm with an unconstrained per-query transform
  (TEM) is WORSE than MapEM's bottlenecked rewind, by +0.403 on every seed. This drops the "bottleneck vs free
  per-block dial" account I proposed for WM's advantage.
- **What survives: where the offset is expressed.**
  - With the counter held identical, reaching the k-th item by content phase inside the query-key comparison
    (MapWM) is learned quickly and completely.
  - Reaching it by moving the query's position (MapEM's rewind, TEM's query transform) is learned partially
    or poorly.
- **MapWM - MapEM from scratch replicates in this batch** (+0.373, 4/4, against -0.375 in `RECENCY_EM_RESULTS.md`).

## Confounds and caveats

- **The three layers differ in more than where the offset lives.** MapWM scores content and position
  jointly. MapEM multiplies a content score by a position score. TEM retrieves by structure alone, with
  no content score at all, which may be why it trails MapEM.
- **TEM has about 1.7x the trainable parameters.**
- **The counter-installed arms use fixed frequencies,** while the from-scratch arms learn theirs.
- **n=4** per arm, one recipe, one task.
