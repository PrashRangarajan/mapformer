> **SUSPECT — planner-demonstration task (2026-08-09).** Scores next-action
> prediction on optimal-planner demonstrations. That family of task is
> self-predictable from the action stream alone: on hier-goal an n-gram scored
> 0.969 at order 1 and 0.971 at order 3, and every model there collapsed to
> ~0.02 closed-loop against a 0.010 random floor. This file has NOT been
> re-validated with an action-only n-gram control at orders 1-5, which is the
> check that would settle it.

# Modular-clock navigation — multi-seed (mean ± std)

Train T_explore=64; eval at listed T_explore (>64 = OOD). Held-out env (seed=10000). Chance=0.25, BFS ceiling=1.00.

Seeds found: MapWM-Flat=[0, 1, 2, 3, 4, 5, 6, 7], MapWM-Hier=[0, 1, 2, 3, 4, 5, 6, 7], MapWM-Hier-CoarsePI=[0, 1, 2, 3, 4, 5, 6, 7], PoPE-Flat=[0, 1, 2, 3, 4, 5, 6, 7], MapPoPE-Hier=[0, 1, 2, 3, 4, 5, 6, 7], MapPoPE-Hier-CoarseIdx=[0, 1, 2, 3, 4, 5, 6, 7], Plain-Flat=[0, 1, 2, 3, 4, 5, 6, 7]


## Held-out action accuracy

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.980 ± 0.003 | 0.790 ± 0.091 | 0.707 ± 0.114 | 0.634 ± 0.164 |
| MapWM-Hier | 0.984 ± 0.002 | 0.643 ± 0.023 | 0.638 ± 0.041 | 0.647 ± 0.028 |
| MapWM-Hier-CoarsePI | 0.984 ± 0.001 | 0.642 ± 0.045 | 0.642 ± 0.034 | 0.632 ± 0.040 |
| PoPE-Flat | 0.972 ± 0.004 | 0.729 ± 0.062 | 0.635 ± 0.061 | 0.628 ± 0.057 |
| MapPoPE-Hier | 0.979 ± 0.001 | 0.822 ± 0.054 | 0.667 ± 0.067 | 0.615 ± 0.034 |
| MapPoPE-Hier-CoarseIdx | 0.976 ± 0.002 | 0.804 ± 0.074 | 0.673 ± 0.078 | 0.644 ± 0.062 |
| Plain-Flat | 0.981 ± 0.001 | 0.603 ± 0.009 | 0.591 ± 0.035 | 0.575 ± 0.027 |

## Held-out NLL (lower better)

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.071 ± 0.012 | 0.977 ± 0.411 | 1.196 ± 0.449 | 1.624 ± 0.790 |
| MapWM-Hier | 0.045 ± 0.010 | 2.187 ± 0.362 | 2.210 ± 0.383 | 1.962 ± 0.248 |
| MapWM-Hier-CoarsePI | 0.047 ± 0.007 | 2.227 ± 0.463 | 2.028 ± 0.465 | 1.950 ± 0.252 |
| PoPE-Flat | 0.098 ± 0.011 | 1.417 ± 0.358 | 2.181 ± 0.439 | 2.526 ± 0.557 |
| MapPoPE-Hier | 0.078 ± 0.004 | 0.952 ± 0.238 | 2.194 ± 0.607 | 2.908 ± 0.602 |
| MapPoPE-Hier-CoarseIdx | 0.083 ± 0.004 | 1.069 ± 0.388 | 1.881 ± 0.779 | 2.202 ± 0.775 |
| Plain-Flat | 0.066 ± 0.006 | 2.620 ± 0.489 | 2.621 ± 0.556 | 2.697 ± 0.483 |
