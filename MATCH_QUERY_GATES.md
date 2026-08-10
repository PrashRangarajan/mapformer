# Match-Query task -- pre-flight gates (CPU, no training)

Blind-continuation task: explore with observations revealed, then continue with them withheld and predict the observation at each cell.
Scored only at cells visited during explore AND non-blank, so chance is 1/16 = 0.0625.

| T_explore | T_query | chance | marginal | n-gram o1 | o3 | o5 | never-moved | answerable rate | n |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 128 | 0.0625 | 0.0722 | 0.0669 | 0.0613 | 0.0672 | 0.0534 | 0.032 | 2452 |
| 256 | 256 | 0.0625 | 0.0689 | 0.0745 | 0.0690 | 0.0645 | 0.0550 | 0.023 | 3543 |
| 512 | 128 | 0.0625 | 0.0736 | 0.0678 | 0.0661 | 0.0732 | 0.0444 | 0.045 | 3424 |
| 512 | 256 | 0.0625 | 0.0677 | 0.0625 | 0.0689 | 0.0664 | 0.0516 | 0.041 | 6246 |
| 1024 | 128 | 0.0625 | 0.0671 | 0.0651 | 0.0635 | 0.0669 | 0.0396 | 0.078 | 5962 |
| 1024 | 256 | 0.0625 | 0.0663 | 0.0576 | 0.0599 | 0.0642 | 0.0425 | 0.067 | 10351 |

**Reading it.** Every baseline column should sit at `chance` (0.0625). `n-gram` is the high-risk gate: a repeated cell inside the query phase makes the answer repeat. `answerable rate` is the fraction of query steps that are scoreable, which sets the effective sample size per episode.

## Verdict: PASS at T_explore >= 256, with per-cell dedup

Two gates FAILED on the first version and were fixed before any training:

| gate | before dedup | after dedup | chance |
|---|---|---|---|
| n-gram order-1 | 0.114 - 0.170 | **0.058 - 0.075** | 0.0625 |
| never-moved | 0.213 - 0.325 | **0.040 - 0.055** | 0.0625 |

Cause: the query walk revisits cells (run-lengths 1..10 send it back and forth),
so the same answer recurs and an order-1 model picks it up for free -- the same
family of shortcut that invalidated hier-goal twice. Fix: score each cell at most
ONCE per episode. This is load-bearing, not cosmetic.

The never-moved baseline was also mis-implemented at first (it used the first
scored query cell rather than the true end-of-explore position, which trivially
matched itself). Now uses the actual endpoint.

Chosen operating point: **T_explore=512, T_query=256** -- all gates at chance,
~10 scored queries per episode, sequence length 1536 tokens.

`marginal` sits slightly above chance (0.066-0.074 vs 0.0625) because the drawn
observation distribution is not exactly uniform. That is the marginal itself, not
an exploitable shortcut, and it is the honest floor to compare against.
