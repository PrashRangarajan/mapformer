> **INVALIDATED — hier-goal task (2026-08-09).** This file reports the
> hierarchical goal-directed navigation task, which does not measure navigation.
> Randomising the goal AND the entire explore phase leaves accuracy unchanged
> (0.912 -> 0.913); closed-loop success is 0.013-0.037 against a random floor of
> 0.010 with the BFS oracle at 1.000. An n-gram on the action stream alone scores
> 0.969 (order 1, raw BFS) / 0.971 (order 3, interleaved). See
> `HIERGOAL_ABLATION.md` and `HIERGOAL_CLOSEDLOOP.md`.

# Hierarchical goal-directed navigation — multi-seed (mean ± std)

Train T_explore=64; eval at listed T_explore (>64 = OOD). Held-out env (seed=10000). Chance=0.25, BFS ceiling=1.00.

Seeds found: VanillaEM_Fixed=[0, 1, 2], MapWM-Flat=[0, 1, 2], MapWM-Hier=[0, 1, 2], Plain-Flat=[0, 1, 2]


## Held-out action accuracy

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| VanillaEM_Fixed | 0.954 ± 0.005 | 0.832 ± 0.046 | 0.713 ± 0.061 | 0.598 ± 0.045 |
| MapWM-Flat | 0.958 ± 0.005 | 0.656 ± 0.206 | 0.746 ± 0.179 | 0.727 ± 0.188 |
| MapWM-Hier | 0.963 ± 0.004 | 0.907 ± 0.026 | 0.849 ± 0.065 | 0.853 ± 0.059 |
| Plain-Flat | 0.966 ± 0.001 | 0.548 ± 0.084 | 0.669 ± 0.106 | 0.591 ± 0.117 |

## Held-out NLL (lower better)

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| VanillaEM_Fixed | 0.154 ± 0.022 | 0.492 ± 0.120 | 0.891 ± 0.260 | 1.161 ± 0.063 |
| MapWM-Flat | 0.151 ± 0.020 | 0.977 ± 0.553 | 0.885 ± 0.608 | 1.036 ± 0.757 |
| MapWM-Hier | 0.114 ± 0.027 | 0.356 ± 0.076 | 0.510 ± 0.145 | 0.511 ± 0.115 |
| Plain-Flat | 0.106 ± 0.015 | 1.606 ± 0.590 | 1.131 ± 0.327 | 1.374 ± 0.459 |
