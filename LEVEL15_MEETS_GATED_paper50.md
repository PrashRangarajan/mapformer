# Paper-task held-out revisit ACCURACY

Paper config: 1 layer, 2 heads, d=128, T=128, 200K sequences (16 epochs x 98 batches x 128).
Paper Table 2 (2D columns), IID: MapWM **0.99**, MapEM-os **1.0**. (An earlier version of this file cited 0.955 / 0.999; those numbers appear in no table of the paper and were retracted in CLAUDE.md on 2026-08-09.)

`same-map` = new trajectories on the trained obs_map; `fresh-map` = unseen obs_map (in-context map learning).

| variant | same-map acc | fresh-map acc |
|---|---|---|
| Vanilla | 0.993 ± 0.009 | 0.993 ± 0.009 |
| Vanilla_ExtraHead | 0.986 ± 0.022 | 0.985 ± 0.023 |
| Level15 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## The 16-epoch reading was a budget artifact, and this is why rule 5 exists

Same three arms, same seeds, same batch, only the budget differs:

| variant | 16 epochs (paper config) | 50 epochs | final train loss @50 |
|---|---|---|---|
| Vanilla | 0.989 ± 0.010 | 0.993 ± 0.009 | 0.1126 |
| Vanilla_ExtraHead | 0.972 ± 0.039 | 0.985 ± 0.023 | 0.1927 |
| **Level15** | **0.938 ± 0.080** | **1.000 ± 0.000** | **0.0068** |

At the paper's own 16-epoch budget Level15 looks like the WORST arm. At 50 epochs
it is the only one that reaches 1.000 on 3/3 seeds, with a training loss 16x
lower than Vanilla's. Reporting the 16-epoch number alone would have been a
false negative -- exactly the failure that voided `MAP_QUERY_RESULTS.md`, where
a budget copied from another task stopped before the metric moved.

The reason is mechanical: the InEKF's `R_t` head has to learn a token-type gating
function (high R on aliased observations, low where informative) before the
correction contributes anything. Until it does, the filter is noise added to a
working path integrator.

## It is not capacity

`Vanilla_ExtraHead` has MORE parameters than Level15 (270,934 vs 254,230 at this
1-layer config) and reaches 0.985, below both. So the Level15 result is not
bought with parameters. This is the same control that overturned the Hopfield
claim in `EXTRAHEAD_CONTROL.md` ("CAPACITY, not structure"); here it does not.

## Read the size of the win honestly

1.000 vs 0.993 is +0.007 accuracy against a measured 0.506 blank floor -- a
ceiling effect, and the accuracy metric has almost no room left to separate these
models. The training-loss gap is the informative one (0.0068 vs 0.1126, ~16x),
and it is consistent with the calibration advantage CLAUDE.md reports for
Level 1.5 across regimes. **The claim supported here is "Level 1.5 matches or
slightly exceeds Vanilla on the clean paper task at 50 epochs, with much better
likelihood", not "Level 1.5 is 0.7pp better".**

Note this does NOT conflict with `LEVEL15_MEETS_GATED_matchq.md`, where Level15
showed no advantage. That task withholds observations, so the R_t gating has
nothing to gate on. Here observations are revealed at every step.
