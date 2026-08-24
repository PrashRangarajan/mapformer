# H=12 allocentric: position effect vs training budget

Asks whether the partial recovery at Habitat's 12 headings was a property of
finer quantisation or simply undertraining. **Neither, cleanly: the budget curve
is NOT monotone, and the arm-to-arm variation is convergence variance.**

Torus GridWorld, `action_mode="rotate"`, `n_headings=12`,
`action_record="allocentric"`, `score_moves_only=True`. Held-out env seed 10000,
T=128, 32 batches x 64. All three budgets trained in one batch per budget, 16
epochs, n=3 seeds. `Vanilla` = path-integrated position, `RoPE` = index
position; the difference between them is the "position effect".

## Aggregate

| batches | floor | Vanilla | RoPE | position effect | n |
|---|---|---|---|---|---|
| 980 | 0.508 | 0.772 ± 0.101 | 0.508 ± 0.006 | **+0.264** | 3 |
| 2000 | 0.508 | 0.891 ± 0.005 | 0.508 ± 0.006 | **+0.383** | 3 |
| 4000 | 0.508 | 0.837 ± 0.060 | 0.551 ± 0.007 | **+0.286** | 3 |

## Per seed — the table above hides the finding

| batches | variant | s0 | s1 | s2 | final loss s0/s1/s2 |
|---|---|---|---|---|---|
| 980 | Vanilla | 0.661 | 0.798 | 0.858 | 1.357 / 0.881 / 0.629 |
| 2000 | Vanilla | 0.885 | 0.893 | 0.894 | 0.552 / 0.527 / 0.507 |
| 4000 | Vanilla | 0.807 | 0.799 | 0.906 | 0.834 / 0.815 / 0.422 |
| 980 | RoPE | 0.502 | 0.514 | 0.507 | 1.733 / 1.774 / 1.726 |
| 2000 | RoPE | 0.502 | 0.514 | 0.507 | 1.691 / 1.703 / 1.663 |
| 4000 | RoPE | 0.542 | 0.555 | 0.555 | 1.590 / 1.613 / 1.578 |

Across all 18 runs, accuracy and final training loss correlate at
**r = -0.996** (Spearman -0.953). Accuracy here is very nearly a deterministic
readout of whether that seed converged, so every difference below is an
optimisation difference, not an evaluation one.

## What this does and does not establish

**Holds.** The position effect at 12 headings is large and present at every
budget. The weakest path-integrated seed anywhere (0.661) is 0.15 above the
strongest index seed anywhere (0.555), and 0.16 above the marginal floor
(0.508). Allocentric recoding works at Habitat's heading resolution; that
direction never depended on the budget point.

**Falsified.** "Recovers once the budget is adequate, +0.264 -> +0.383" is not a
trend — 4000 batches goes back down to +0.286. Two of three seeds converged
WORSE at 4000 (loss 0.834 / 0.815) than every seed did at 2000 (0.507-0.552),
while the third converged better than any other run in the table (0.422). That
is bimodal basin selection, of the same kind as the lm200 non-convergence
recorded in CLAUDE.md, not a dose-response curve. nb=2000 is the best point
measured, and there is no evidence more compute improves on it.

**Second-order.** The index arm is no longer at the floor at 4000 (0.542-0.555
vs floor 0.508, and its loss falls monotonically 1.73 -> 1.69 -> 1.59). With
enough budget an index model does begin to fit this task slightly, which shrinks
the measured effect independently of what the path-integrated arm does.

**Not established.** Where the bimodality comes from. Same LR schedule shape
(linear decay over `epochs * n_batches`), same peak LR, 2x fresh data — more
steps at high LR is the obvious suspect but was not tested. Resolving it needs
more seeds at nb=4000, which is the honest next step if this number goes in a
paper. n=3 cannot separate "two unlucky seeds" from "the 4000-batch recipe is
worse".

Reference (n=8): H=4 allocentric **+0.488**, H=4 commanded **+0.050**,
translate baseline **+0.438**.

Regenerate: `python3 -m mapformer.eval_h12_budget --device cuda:1` (aggregate),
`python3 -m mapformer.eval_h12_perseed --device cuda:1` (per seed).
