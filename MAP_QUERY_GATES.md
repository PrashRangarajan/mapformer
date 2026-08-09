# Map-Query task -- pre-flight gates (CPU, no training)

A gate at chance is a PASS. Above chance means the task is solvable without the capability under test.

| T_explore | chance | oracle | n-gram o1 | o3 | o5 | goal-only | assume-start | explore-only | room: best-constant (chance) | mean dist from start (uniform) |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.501 | 1.000 | 0.496 | 0.494 | 0.493 | 0.546 | 0.623 | 0.505 | 0.049 (0.016) | 23.3 (32) |
| 128 | 0.503 | 1.000 | 0.501 | 0.503 | 0.496 | 0.497 | 0.527 | 0.496 | 0.026 (0.016) | 28.9 (32) |
| 256 | 0.506 | 1.000 | 0.485 | 0.498 | 0.496 | 0.497 | 0.499 | 0.496 | 0.023 (0.016) | 32.0 (32) |
| 512 | 0.490 | 1.000 | 0.507 | 0.506 | 0.501 | 0.510 | 0.487 | 0.501 | 0.025 (0.016) | 32.7 (32) |
| 1024 | 0.500 | 1.000 | 0.492 | 0.487 | 0.493 | 0.491 | 0.490 | 0.491 | 0.024 (0.016) | 32.4 (32) |

**Reading it.** `chance` is the random-policy rate and is the PASS level for every baseline column. `goal-only` above chance means the fixed start leaves position too concentrated -- raise T_explore. `oracle` must be 1.000.

## Verdict: PASS at T_explore >= 256

At T_explore >= 256 every baseline sits at the measured chance rate (~0.50) and
the oracle is 1.000:

- n-gram orders 1/3/5 over the answer stream alone: 0.485-0.507. This is the gate
  hier-goal failed at 0.969 (order 1, raw BFS) and 0.971 (order 3, interleaved).
- goal-only, explore-only, assume-start: all 0.487-0.510.
- mean torus distance from the fixed start reaches 32.0-32.7 against a uniform
  value of 32 -- the walk is fully mixed, so position genuinely must be
  integrated.

**T_explore = 64 FAILS** (assume-start 0.623, goal-only 0.546, mean distance 23.3
of 32): the agent has not mixed, so the goal alone partly determines the answer.
This is the third shortcut of the same family, and it was found before training
rather than after. **T_explore = 256 is the minimum valid operating point** --
chosen from the mixing measurement, not picked.

## Two query types, and why both

| query | chance | resolution |
|---|---|---|
| goal direction (first action) | ~0.50 | poor -- several actions are simultaneously optimal on a torus |
| **room identity ("which room are you in?")** | **0.016** | 64 classes; best constant guess is 0.023-0.026 |

The direction query is the goal-directed one but its 0.50 floor makes it weakly
discriminative. The room query has 30x the resolution and measures absolute
position directly. Report both: room identity as the cognitive-map metric,
direction as the goal-directed one, closed-loop success as the behavioural one.

## Calibration bugs found and fixed in the gates themselves

1. `assume-start` was scored by SET INTERSECTION while `chance` was a single
   action. Two random 2-of-4 sets overlap ~83% of the time, so it read 0.69-0.88
   and looked like a failure. Now scored as a single action, comparable to chance.
2. `H(pos)` is capped at log2(n_episodes), so at 400 episodes it saturated at
   ~8.4 bits and measured sample size, not the walk. Replaced with mean torus
   distance from the start, which is sample-size independent.

## Post-hoc: the shortcut-free regime and the learnable regime do not overlap

Training diagnostics (Vanilla, 3 layers, d=128) after the gates were run:

| T_explore | assume-start gate | held-out room acc | room loss (chance 4.16) | direction loss (chance 0.69) |
|---|---|---|---|---|
| 16 | **FAILS** (walk unmixed) | 0.994 | 0.17 | 0.09 |
| 256 | PASSES | **0.121** (7.6x chance) | 2.33 @ep200, plateaued | **0.705 -- never moves in 200 epochs** |

At T_explore=16 the task is essentially solved (room 0.994, direction 0.969), so
the task, scorer, gradient path and metrics are all correct. But T=16 fails the
assume-start gate: the walk has not mixed (mean distance 23.3 of uniform 32), so
the goal alone partly determines the answer.

At T_explore=256 the gates pass, and the task becomes very hard. Final 200-epoch
numbers, held out: **room 0.1211** (chance 0.016, so 7.6x chance -- real signal,
and usable as a discriminative metric) and **direction 0.4966** (chance 0.50 --
completely dead, with its loss flat at 0.70 across all 200 epochs).

So the two query types behave completely differently at length:
- ROOM survives, weakly. 0.121 vs 0.994 at T=16, but well above chance.
- DIRECTION does not survive at all and should be dropped.

**This is a property of the query type, not the budget.** Answering "which room
am I in" requires DECODING absolute position -- a modular sum of ~256 signed
steps -- which is a known-hard regime for transformers. MapFormer's position code
exists to make positions COMPARABLE (its revisit task only ever needs to MATCH
two positions), never to make one readable. The task as designed asks for a
capability the architecture is not built to provide, so it cannot discriminate
between variants.

**Proposed revision: matching-based queries.** Ask "what observation is at the
cell you would reach by doing [a_1..a_k] from here?" -- the model integrates the
hypothetical offset onto its current position and MATCHES against memory, exactly
the operation the architecture supports, with no absolute decode. This should
stay learnable at long T_explore, where the shortcut gates pass. Not yet built.
