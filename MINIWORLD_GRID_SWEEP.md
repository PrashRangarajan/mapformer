# MiniWorld grid sweep -- attention substitutability (fresh-map, oracle recode)

Vanilla=path-int, RoPE=index. Prediction: path-int - index climbs from -0.53 (grid 8) toward positive as the grid grows and attention can no longer integrate position over the longer revisit distances. chance 0.0625.

## T=512

| grid | Vanilla (path-int) | RoPE (index) | effect (V-R) |
|---|---|---|---|
| 8 | 0.448 | 0.977 | **-0.529** (n=3) |
| 16 | 0.886 | 0.738 | **+0.148** (n=3) |
| 24 | 0.694 | 0.618 | **+0.076** (n=3) |
| 32 | 0.703 | 0.615 | **+0.087** (n=3) |

## T=1024

| grid | Vanilla (path-int) | RoPE (index) | effect (V-R) |
|---|---|---|---|
| 8 | 0.289 | 0.543 | **-0.253** (n=3) |
| 16 | 0.712 | 0.379 | **+0.333** (n=3) |
| 24 | 0.506 | 0.298 | **+0.208** (n=3) |
| 32 | 0.505 | 0.333 | **+0.172** (n=3) |


## MAJOR CAVEAT (audit, 2026-08-27): the accuracy crossover IS a CONVERGENCE crossover

Final training loss across the very same runs:

| | grid 8 | grid >=16 |
|---|---|---|
| RoPE (index) | converges **3/3** (0.17, 0.17, 0.23) | converges **0/9** (0.47-0.84) |
| Vanilla (path-int) | converges **0/3** (0.93, 1.14, 1.14) | converges 4/9 |

**RoPE never converges at any grid >= 16, yet it is the subtrahend in every position
effect reported above.** And Vanilla never converges at grid 8. So the reported sign
flip coincides exactly with a flip in WHICH ARM TRAINS. The ordering is real, but the
MECHANISM is not established: "attention cannot span the distance" (representational)
and "the index model does not optimise at this grid size" (optimisation) are both
consistent with these numbers, and this experiment cannot separate them.

This is the same confound that voided the hierarchy amplification claim, rendered
ConvDelta a null and left GateDelta inconclusive -- loss ranges 0.00-0.86 across seeds
of a SINGLE model. It now touches the headline.

Per-seed consistency is also weaker than the means suggest: g32's +0.087 rests on one
seed (+0.349) against two negatives (-0.015, -0.073); g16 also has a negative seed.
Only the g8 -> g16 flip is large enough to be safe from this.

Minor: "RoPE collapses to ~0.62 at g>=16" -- g16 is **0.738**; 0.62 applies to g24/g32.

**DECISIVE EXPERIMENT (not yet run):** train every arm to convergence (longer schedule,
warmup, or lower LR) and re-measure. If RoPE still loses at g>=16 having converged, the
representational claim holds. If it converges and matches, the crossover was optimisation.
Until then the honest statement is: *the ordering flips with map size, and we have not
separated representation from optimisation.*
