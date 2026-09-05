# Match-Query: does r=4 remove the same failure mode as the loop?

128^2, TQ=256, chance 0.0625. All four arms in one batch, LOOP_HEADROOM's recipe. The loop is free on both rows and rank costs exactly +384 on both, so the interaction is parameter-matched.

| arm | params | mean | sd | **min** | per-seed |
|---|---|---|---|---|---|
| r=2, no loop | 204,373 | **0.416** | 0.181 | **0.122** | 0.12 0.32 0.66 0.31 0.40 0.65 0.37 0.49 |
| r=4, no loop | 204,757 | **0.421** | 0.209 | **0.150** | 0.15 0.38 0.80 0.26 0.26 0.59 0.46 0.46 |
| r=2, loop x4 | 204,373 | **0.833** | 0.082 | **0.713** | 0.83 0.80 0.85 0.81 0.79 0.88 0.71 1.00 |
| r=4, loop x4 | 204,757 | **0.986** | 0.020 | **0.941** | 0.99 1.00 0.98 1.00 1.00 0.94 0.98 1.00 |

## Effects

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| rank main effect, no loop  (r4 - r2) | +0.005 | 0.092 | 0.091 | 4/8 | unmeasured |
| rank main effect, with loop (r4 - r2) | +0.154 | 0.085 | 0.084 | 8/8 | DETECTABLE |
| loop main effect, r=2 | +0.417 | 0.167 | 0.165 | 8/8 | DETECTABLE |
| loop main effect, r=4 | +0.565 | 0.221 | 0.219 | 8/8 | DETECTABLE |
| **interaction (rank x loop)** | +0.149 | 0.113 | 0.112 | 7/8 | DETECTABLE |

## The pre-registered reading

The prediction was that r=4 **compresses variance more than it lifts the mean** -- raising the floor, the way the loop does. Read the `min` and `sd` columns first; the mean is secondary here by prior commitment, not by hindsight.

- Negative interaction with both mains positive -> the two remove the **same** failure mode, and $384$ parameters is the cheaper route.
- Interaction near zero with both mains holding -> independent fixes, and `r=4, loop x4` should be the best arm measured on this task.
- No rank effect at all -> the skewed-basin account does not transfer off the torus, and the D x r optimisation half is torus-specific.
