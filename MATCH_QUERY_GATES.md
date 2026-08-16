> **GATE CORRECTED (2026-08-09): the never-moved floor is 0.0893, not 0.0516.**
> The gate predicted the observation at the end-of-explore cell even when that
> cell is BLANK -- but answers are non-blank by construction and the model's
> logit slice excludes blank, so a blank guess is not expressible. Scoring it as
> a miss made ~44% of trials auto-fail, and the gate read 0.0516 ("below chance,
> PASS"). Scored only where the strategy is expressible it is **0.0893 = 1.43x
> chance**, i.e. a real shortcut that was never flagged.
>
> **Consequence for the headline: read Match-Query against 0.0893, not 0.0625.**
> The path-integration result is unaffected (0.730-0.888 >> 0.0893), but the
> INDEX-position models at 0.154 are 1.7x this floor rather than 2.5x chance --
> a smaller margin than previously presented.

# Match-Query task -- pre-flight gates (CPU, no training)

Blind-continuation task: explore with observations revealed, then continue with them withheld and predict the observation at each cell.
Scored only at cells visited during explore AND non-blank, so chance is 1/16 = 0.0625.

| T_explore | T_query | chance | marginal | n-gram o1 | o3 | o5 | never-moved | answerable rate | n |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 256 | 0.0625 | 0.0677 | 0.0625 | 0.0689 | 0.0664 | 0.0516 | 0.041 | 6246 |

**Reading it.** Every baseline column should sit at `chance` (0.0625). `n-gram` is the high-risk gate: a repeated cell inside the query phase makes the answer repeat. `answerable rate` is the fraction of query steps that are scoreable, which sets the effective sample size per episode.
