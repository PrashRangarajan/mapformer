> **INVALIDATED (2026-08-09)** -- see `HIERGOAL_ABLATION.md`. The task is
> solvable from the navigate action prefix alone: randomising the goal AND
> the entire explore phase leaves accuracy unchanged. An order-3 Markov model
> on actions alone scores 0.971 (interleaved) / 0.969 (raw BFS). No number in
> this file measures navigation.

# Hierarchical goal-directed navigation — multi-seed (mean ± std)

Train T_explore=64; eval at listed T_explore (>64 = OOD). Held-out env (seed=10000). Chance=0.25, BFS ceiling=1.00.

Seeds found: MapWM-Flat=[0, 1, 2], MapWM-Hier=[0, 1, 2], Plain-Flat=[0, 1, 2], Plain-Hier=[0, 1, 2], PoPE-Flat=[0, 1, 2], MapPoPE-Hier=[0, 1, 2]


## Held-out action accuracy

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.947 ± 0.001 | 0.911 ± 0.007 | 0.907 ± 0.019 | 0.893 ± 0.026 |
| MapWM-Hier | 0.953 ± 0.004 | 0.826 ± 0.011 | 0.794 ± 0.017 | 0.780 ± 0.011 |
| Plain-Flat | 0.957 ± 0.001 | 0.910 ± 0.014 | 0.915 ± 0.012 | 0.901 ± 0.023 |
| Plain-Hier | 0.958 ± 0.001 | 0.679 ± 0.028 | 0.689 ± 0.039 | 0.673 ± 0.045 |
| PoPE-Flat | 0.938 ± 0.001 | 0.938 ± 0.001 | 0.940 ± 0.001 | 0.935 ± 0.001 |
| MapPoPE-Hier | 0.949 ± 0.005 | 0.929 ± 0.003 | 0.916 ± 0.009 | 0.890 ± 0.017 |

## Held-out NLL (lower better)

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.169 ± 0.002 | 0.406 ± 0.028 | 0.402 ± 0.057 | 0.452 ± 0.069 |
| MapWM-Hier | 0.151 ± 0.009 | 0.571 ± 0.052 | 0.579 ± 0.009 | 0.672 ± 0.056 |
| Plain-Flat | 0.144 ± 0.006 | 0.461 ± 0.016 | 0.442 ± 0.023 | 0.489 ± 0.011 |
| Plain-Hier | 0.132 ± 0.006 | 1.718 ± 0.106 | 1.687 ± 0.301 | 1.876 ± 0.471 |
| PoPE-Flat | 0.208 ± 0.001 | 0.217 ± 0.002 | 0.208 ± 0.000 | 0.222 ± 0.001 |
| MapPoPE-Hier | 0.165 ± 0.022 | 0.291 ± 0.025 | 0.362 ± 0.018 | 0.456 ± 0.022 |
