# Rank at matched length -- results, 300-epoch batch (2026-09-23)

Pre-registration: `RANK_MATCHED_PREREG.md` (Amendment 2 governs how this is read). Runs
`runs/rank_matched` (2 arms x 8 seeds, one batch). Numbers: `RANK_MATCHED.json`,
`RANK_MATCHED_STRATA.json`, `RANK_SWEEP_STRATA.json` (old T=128 checkpoints, re-scored),
`RANK_MATCHED_GEOMETRY.md`. Every figure is printed by `python3 -m mapformer.analyze_rank_matched`.

> **CORRECTED the same day, after an end-to-end review.** The first version of this file
> concluded "the r=4 gap is entirely a training-speed difference" from a loss-matched
> residual of +0.002. **That is withdrawn.** Train and test are the same task at matched
> length, so held-out accuracy is close to a function of training loss for ANY solution,
> and a near-zero loss-matched residual is predicted whether r=4 trains faster, is more
> capable, or reaches a better solution. The regression was also weak: within-arm slopes
> differ 3.7x and the only r=4 runs inside the overlapping loss range are two failed
> seeds. The first version also judged convergence by a "flat" criterion that turned out
> inverted (it called the four STUCK r=2 runs flat and failed a solved r=4 run); it is
> replaced by Amendment 2's solved / stalled / descending classes.

## Verdict: UNREADABLE (budget-limited)

Under Amendment 2 (SOLVED = mean loss over the final 5% of epochs < 0.05; STALLED = not
solved and the final 10% within 5% of the 10% before; DESCENDING = neither):

| arm | SOLVED | STALLED | DESCENDING | final loss range |
|---|---|---|---|---|
| r=2 | 0 | 4 (loss 1.06-1.25) | 4 | 0.209 - 1.252 |
| r=4 | 3 | 0 | 5 | 0.0015 - 0.879 |

More than two runs per arm were still descending when the budget ran out, so no branch is
read. At T=128 the same recipe drove every run to <= 0.09 (r=4 <= 0.0006); at T=1024, with
the same tokens per step and 8x fewer independent walks, it does not.

## What the batch does show

**1. r=2 can REPRESENT the T=1024 solution, so the open question is learnability, not
capacity.** Projecting trained r=4 checkpoints onto their top two latent directions and
loading them into an r=2 model (all other weights unchanged) scores 0.9995 (seed 6, r=4
1.000) and 0.990 (seed 2, r=4 1.000) at T=1024; seeds 0 and 1 are unchanged and seed 5
drops to 0.954. Whatever this design eventually shows, it cannot be a capacity claim.

**2. r=4 trained further in the same budget.** Lower final loss on 7/8 seeds; 3/8 SOLVED
against 0/8 (Fisher p 0.20). Both batches agree r=4 reaches lower loss. Whether r=2 would
get there with more training, or stays stalled, is what the 900-epoch pilot measures.

**3. The arms' solutions differ in SHAPE, not just level (exploratory, not registered).**
Each run's accuracy on the two long-range strata minus its own accuracy on short-gap
revisits: r=2 loses -0.092 (gap >= 128) and -0.203 (wrap), r=4 stays flat (-0.002 /
-0.017), on every seed including r=4's unsolved ones (differences +0.090, MDE 0.083 and
+0.185, MDE 0.149, 8/8 each). This could be the order in which r=2 learns, but it is not
simply "less trained".

Raw contrasts, for the record (the registered primary test is the exact permutation test;
it is reported but no branch is read, because the batch is unreadable):

| readout | r=2 | r=4 | r4 - r2 | paired MDE | seeds r4 > r2 | permutation p |
|---|---|---|---|---|---|---|
| **acc T=1024 [primary]** | 0.741 | 0.898 | +0.157 | 0.203 | 7/8 | 0.039 |
| acc T=512 | 0.759 | 0.899 | +0.141 | 0.211 | 7/8 | 0.063 |
| acc T=2048 | 0.700 | 0.875 | +0.176 | 0.184 | 7/8 | 0.019 |
| NLL T=1024 (lower is better) | 0.758 | 0.272 | -0.486 | 0.602 | 7/8 | 0.032 |

Per stratum at T=1024 (floor = best constant prediction, 8-seed mean):

| stratum | floor | acc r=2 | acc r=4 | acc diff (MDE) | NLL diff (MDE) | runs below floor r2 / r4 |
|---|---|---|---|---|---|---|
| gap < 128 | 0.506 | 0.765 | 0.899 | +0.134 (0.211) | -0.377 (0.622) | 0 / 0 |
| gap >= 128 | 0.500 | 0.674 | 0.897 | +0.224 (0.219) | -0.797 (0.730) | 1 / 0 |
| wrap-only | 0.507 | 0.563 | 0.882 | +0.319 (0.169) | -1.274 (0.551) | 3 / 0 |

**4. The old number, re-scored by kind of revisit** (T=128-trained, `RANK_SWEEP_STRATA.json`).
The r=4 advantage sits in short-gap revisits (+0.096, MDE 0.078, 8/8). Where the revisit
needs something T=128 training never showed it is UNMEASURED, not absent: gap >= 128
+0.062 (MDE 0.143), wrap -0.003 (MDE 0.100, both arms below the floor).

**5. Wrap-only revisits become learnable once trained on** -- for r=4 (0.882, no run below
the floor). For r=2 only on average (0.563): 3/8 runs are below the floor.

## Exploratory geometry prediction: not supported

Predicted: 1024-step walks give r=2 a cleaner action code. Mean opposition went 0.495 ->
0.667 (worse); r=2 is still bimodal (seeds 1/2/3 cancel at 0.17-0.20 but put both axes on
one line, |cos| ~0.99; the others do not cancel). r=4 stays clean (0.087 / 0.091). Half
the r=2 runs are stalled, so this is confounded with training.

## Next

The 900-epoch pilot (seeds 0-1, Amendment 1) is running and is read by Amendment 2's
mechanical rule. The review also proposed two cheaper designs worth considering if the
pilot shows r=2 still descending or stalled: warm-start both arms from their solved T=128
checkpoints and fine-tune at T=1024 (skips the from-scratch plateau), and warm-start r=2
from the rank-2 projection of a solved r=4 model (if it holds, r=2's deficit is pure search).
