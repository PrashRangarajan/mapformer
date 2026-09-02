# Do tokens want different recursion depths here?

`Looped` on Match-Query 128^2, n=8 existing checkpoints, inference only.
T_explore=512, T_query=256, chance = 0.0625. Strata are equal slices of position WITHIN the blind query phase (Q1 earliest).

| loops k | overall | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|
| 1 | 0.384 | 0.412 | 0.379 | 0.371 | 0.352 | 0.336 |
| 2 | 0.835 | 0.888 | 0.793 | 0.824 | 0.779 | 0.800 |
| 3 | 0.866 | 0.907 | 0.829 | 0.866 | 0.842 | 0.790 |
| 4 | 0.865 | 0.908 | 0.822 | 0.875 | 0.850 | 0.769 |
| 5 | 0.865 | 0.910 | 0.829 | 0.870 | 0.842 | 0.776 |
| 6 | 0.863 | 0.909 | 0.830 | 0.863 | 0.835 | 0.772 |
| 8 | 0.856 | 0.903 | 0.833 | 0.853 | 0.836 | 0.758 |

## Best loop count per stratum

| stratum | Q1 | Q2 | Q3 | Q4 | Q5 | overall |
|---|---|---|---|---|---|---|
| argmax k | 5 | 8 | 4 | 4 | 2 | 3 |

## Verdict

argmax k spans 2..8 across strata -- but read the ORACLE, not the argmax.

A per-stratum oracle router, allowed to know the best depth for every stratum,
scores **0.854** against **0.847** for the single best global count. That is an
upper bound of **+0.007** on what ANY router could buy along this axis. Seed sd of
overall accuracy is **0.152**, so the bound is **22x smaller than the run-to-run
noise**.

**No router can pay here.** Above k=3 the entire curve is flat to within 0.010
(0.866 / 0.865 / 0.865 / 0.863 / 0.856 for k=3,4,5,6,8). The differing argmaxes
are wander on a flat curve. All the depth signal lives BELOW k=3 -- k=1 collapses
to 0.384 and k=2 recovers to 0.835 -- and that is a floor effect every token
shares, not a per-token preference. MoR's routing would reduce to picking one
global count, which LoopedSampled already does for free.

A note on how this was nearly misread: the first version of this script branched
on `spread != 0` and duly announced "strata want DIFFERENT depths" off an argmax
that wanders across a flat curve. Same error as the grid-16 pre-registration that
fired "graded" on +0.015. Branch against the noise floor, never against zero.

## Scope

Per-POSITION heterogeneity only, which is a subset of per-token. A router on the hidden state could find an axis this misses; position is simply where this repo's own evidence points (the same torus weights peak at 4 passes at T=128 and 2 at T=512). A null here is evidence against, not proof against. Single task, single width, one T_explore/T_query pair.
