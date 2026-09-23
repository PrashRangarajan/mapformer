# Rank at matched length -- results (2026-09-23)

Pre-registration: `RANK_MATCHED_PREREG.md`. Runs `runs/rank_matched` (16 = 2 arms x 8
seeds, one batch). Numbers: `RANK_MATCHED.json`, `RANK_MATCHED_STRATA.json`,
`RANK_SWEEP_STRATA.json` (the old T=128 checkpoints, re-scored), `RANK_MATCHED_GEOMETRY.md`.
Every figure below is printed by `python3 -m mapformer.analyze_rank_matched`.

## Verdict: BUDGET-LIMITED -- no branch fires

The registered convergence check fails. A run is flat if its loss over epochs 271-300 is
within 5% of epochs 241-270, and at least 6/8 per arm had to be flat:

| arm | flat | final training loss (8 seeds) | last-30 / previous-30 |
|---|---|---|---|
| r=2 | 4/8 | 0.209 - 1.252 | 0.92 - 0.99 |
| r=4 | **0/8** | 0.0015 - 0.879 | 0.31 - 0.94 |

At T=128 the same recipe drove both arms to <= 0.09 (r=4 <= 0.0006). At T=1024, with the
same tokens per step and 8x fewer independent walks, neither arm gets close, and r=4 is
still falling steeply on every seed. So this batch cannot say whether r=2 has a capability
deficit at long sequences. R1, R2 and R3 are all unread.

## What the batch does show

**1. The r=4 advantage here is entirely a training-speed difference.** The losses overlap
this time (r=2 0.209-1.252, r=4 0.0015-0.879), so rule 9 can be applied properly:

- r(log final loss, T=1024 accuracy) = **-0.985 within r=2**, -0.836 within r=4, -0.845 pooled.
  Accuracy is the training loss.
- **Loss-matched r4 - r2 = +0.002** (MDE 0.115). At equal training loss the two ranks are
  indistinguishable.

Raw, r=4 is ahead everywhere but mostly inside the MDE, because some seeds of each arm
failed to train (r=2 0.591-0.937, r=4 0.657-1.000):

| readout (trained at T=1024) | r=2 | r=4 | r4 - r2 | MDE | seeds + |
|---|---|---|---|---|---|
| **acc T=1024 [primary]** | 0.741 | 0.898 | +0.157 | 0.203 | 7/8, unmeasured |
| acc T=512 | 0.759 | 0.899 | +0.141 | 0.211 | 7/8 |
| acc T=2048 | 0.700 | 0.875 | +0.176 | 0.184 | 7/8 |
| NLL T=1024 | 0.758 | 0.272 | -0.486 | 0.602 | 7/8 r4 better |
| T=1024 gap < 128 (floor 0.506) | 0.765 | 0.899 | +0.134 | 0.211 | 7/8 |
| T=1024 gap >= 128 (floor 0.500) | 0.674 | 0.897 | +0.224 | 0.219 | 7/8, detectable |
| T=1024 wrap-only (floor 0.507) | 0.563 | 0.882 | +0.319 | 0.169 | 8/8, detectable |

The two detectable strata inherit the same loss confound as the overall number. They are
not a capability result.

**2. r=4 trains faster -- the one thing both batches agree on.** At T=128 the losses did
not overlap (r=4 lower on 8/8). At T=1024 r=4 reaches lower loss on 7/8 seeds in the
same budget. That fits the "optimisation, not dimensionality" account from the D x r
batch. It is NOT evidence that r=2 cannot represent the task.

**3. The old number, re-scored by kind of revisit** (T=128-trained, from
`RANK_SWEEP_STRATA.json`). The r=4 advantage is in the plain gap<128 stratum (+0.096,
8/8, detectable) and absent where the revisit needs something T=128 training never showed
(gap >= 128: +0.062 unmeasured; wrap: -0.003). This confirms the audit, now from a
committed script.

**4. Wrap-only revisits are learnable once trained on.** Trained at T=128 both arms sit
below the floor (0.416 / 0.413). Trained at T=1024, r=4 reaches 0.882 and r=2 0.563, both
above the 0.507 floor. The difference between the arms is the loss confound again.

## Exploratory geometry prediction: NOT supported

The prediction was that 1024-step walks would give r=2 a cleaner code. Mean opposition went
0.495 -> 0.667 (worse). Per seed it is still bimodal: seeds 1/2/3 cancel (0.17-0.20) but
put both axes on one line (|cos| ~0.99); seeds 0/4/5/6/7 do not cancel (0.58-1.16). r=4
stays clean (opposition 0.087, |cos| 0.091). Four of eight r=2 runs are unconverged, so
this is confounded with training too.

## Next

The question is still open, and the budget has to change before it can be read. Run a
2-seeds-per-arm pilot at 3x epochs (900) to find a budget where both arms are flat,
then the full 8 seeds at that budget: about 1 h for the pilot, 5-6 h for the batch. A
larger batch at the same step count is the other lever, but it changes tokens per step
away from the T=128 match.
