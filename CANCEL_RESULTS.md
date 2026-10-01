# H3, the cancellation knob -- results (2026-09-28)

> **CORRECTED 2026-09-30 (audit).** (1) **Seeds 0 and 1 are the pilot**: `runs/cancel_pilot` L1 cells are
> byte-identical to the batch's s0/s1 (4 of 16 cells, including the registered primary's index 1-layer cells
> at p_plus 0.5 and 1.0). On the six fresh seeds alone (`docs/audits/2026-09-27/fresh_seeds.txt`): a1 =
> 0.710 / 0.677 / 0.824 / 1.000, every pair still differs (perm p 0.0022-0.0108); the non-monotone dip at 0.75
> holds. (2) **k = 3 at p_plus 0.9 is knife-edge**: the 2-layer gap there is 0.0115 (fresh seeds 0.0133)
> against the 0.01 threshold, with 7/8 runs STALLED; read "3 layers wherever steps can cancel" as budget- and
> threshold-scoped, and "however rarely" as overstated. (3) The eval stream (np seed 0) is the data stream of
> training batch 0 for seed-0 runs, and this task redraws the map per sequence, so 128 of the 200 eval
> sequences of seed-0 runs were training batch 0 (1 of 29,400 batches); the docstring's "held-out map" is wrong
> for this task.
Pre-registration `CANCEL_PREREG.md`; runs `runs/cancel` (128 runs, one batch, 8 seeds); full output
`CANCEL_ANALYSIS.txt` (`python3 -m mapformer.analyze_cancel`); task `environment_cancel.py`, trainer
`train_cancel.py`, gate `docs/audits/2026-09-27/gate_cancel.py`. A 32-cell ring, map redrawn per
trajectory, directed walk whose action is +1 with probability `p_plus`; revisit accuracy at the
training length T=128, 300 epochs.

## What held: one layer of path integration = three layers of index attention wherever steps cancel

T=128 accuracy, mean +/- sd over 8 seeds (SOLVED / STALLED / DESCENDING):

| p_plus | floor | path 1 layer | index 1 layer | index 2 layers | index 3 layers |
|---|---|---|---|---|---|
| 0.5 | 0.554 | **1.000 +/- 0.000** (8/0/0) | 0.717 +/- 0.022 (0/8/0) | 0.954 +/- 0.038 (0/8/0) | 0.998 +/- 0.001 (8/0/0) |
| 0.75 | 0.544 | **1.000 +/- 0.000** (8/0/0) | 0.676 +/- 0.008 (0/8/0) | 0.969 +/- 0.034 (0/8/0) | 0.998 +/- 0.001 (8/0/0) |
| 0.9 | 0.520 | **1.000 +/- 0.000** (8/0/0) | 0.825 +/- 0.006 (0/8/0) | 0.988 +/- 0.009 (1/7/0) | 0.999 +/- 0.000 (8/0/0) |
| 1.0 | 0.498 | 1.000 +/- 0.000 (8/0/0) | 1.000 +/- 0.000 (8/0/0) | 1.000 +/- 0.000 (8/0/0) | 1.000 +/- 0.000 (8/0/0) |

Secondary (registered, no verdict): the exchange rate k(p), the fewest index layers within 0.01 of
path's 1-layer accuracy, is **3 at p_plus 0.5, 0.75 and 0.9, and 1 at p_plus 1.0**. Path integration
solves every cell with one layer, 32/32 SOLVED. Index attention needs three layers as soon as any
step can be undone (at p_plus 0.9, one segment in ten runs backwards, the 2-layer gap is 0.0115 against the 0.01 threshold: knife-edge), and one layer
when none can. This reproduces Dyck's matched-depth exchange rate (`DYCK_MDEPTH_RESULTS.md`: one
layer of path integration ~ three of attention) on a second task, inside one task family, at the
training length. The 1-layer gap G1 = path - index: +0.283 / +0.324 / +0.175 / 0.000.

## Registered primary: no registered branch -- reported as it falls

Index 1-layer accuracy a1(p) is **not monotone**: 0.717 at 0.5, **0.676 at 0.75** (below 0.5: -0.041,
perm p 0.0011), 0.825 at 0.9, 1.000 at 1.0. Every pair differs (all p <= 0.0011).
- CONTINUUM required a1(0.5) < a1(0.75) < a1(0.9) < a1(1.0): fails at the 0.75 dip.
- DICHOTOMY required a1(0.75) and a1(0.9) not above a1(0.5): fails because a1(0.9) is +0.108 above.
By the pre-registration, no mechanism sentence is attached to the 0.75 dip.

## Caveats
- **Budget-scoped.** Every index 1-layer run at p_plus < 1 and 23/24 index 2-layer runs are
  STALLED at 300 epochs; the pilot already showed index 1-layer creeping down (loss 1.056 -> 1.027).
  The 1-layer accuracies are plateau values, not converged capacities.
- **Accuracy is loss** (r(final loss, acc) = -0.997 over 128 runs); at matched length that is expected.
- Path is at ceiling in every cell, so "one layer" is an upper bound on what path needs, and the
  exchange rate is read against a ceiling.
- The gap closes at p_plus = 1 where both are clocks; the p=1 path arm's phase is a LEARNED multiple
  of the index (pre-registered caveat), so equality there is not evidence about maps.
- T=512 (4x the training length, printed in the analysis, no verdict): path 1.000 / 0.994 / 0.962 /
  0.946, index 3-layer 0.578 / 0.607 / 0.676 / 1.000 -- the extrapolation gap is large, but it is
  extrapolation.
- Scope: 1D ring of 32 cells, T=128, d 128, 2 heads, one recipe, 300 epochs, n=8 per cell.
