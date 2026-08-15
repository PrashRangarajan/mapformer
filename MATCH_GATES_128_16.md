# Match-Query task -- pre-flight gates (CPU, no training)

Blind-continuation task: explore with observations revealed, then continue with them withheld and predict the observation at each cell.
Scored only at cells visited during explore AND non-blank. size=128, n_obs=16, so chance = 1/16 = 0.0625.

| T_explore | T_query | chance | marginal | n-gram o1 | o3 | o5 | never-moved | answerable rate | n |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 256 | 0.0625 | 0.0708 | 0.0640 | 0.0620 | 0.0731 | 0.0490 | 0.025 | 3813 |

**Reading it.** Every baseline column should sit at `chance` (0.0625). `n-gram` is the high-risk gate: a repeated cell inside the query phase makes the answer repeat. `answerable rate` is the fraction of query steps that are scoreable, which sets the effective sample size per episode.
