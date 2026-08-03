# Hierarchical goal-directed navigation — multi-seed (mean ± std)

Train T_explore=64; eval at listed T_explore (>64 = OOD). Held-out env (seed=10000). Chance=0.25, BFS ceiling=1.00.

Seeds found: MapWM-Flat=[0, 1, 2], MapWM-Hier=[0, 1, 2], Plain-Flat=[0, 1, 2], Plain-Hier=[0, 1, 2]


## Held-out action accuracy

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.967 ± 0.001 | 0.681 ± 0.149 | 0.767 ± 0.175 | 0.743 ± 0.185 |
| MapWM-Hier | 0.971 ± 0.003 | 0.861 ± 0.050 | 0.822 ± 0.065 | 0.774 ± 0.099 |
| Plain-Flat | 0.973 ± 0.002 | 0.586 ± 0.138 | 0.625 ± 0.167 | 0.611 ± 0.186 |
| Plain-Hier | 0.978 ± 0.003 | 0.871 ± 0.067 | 0.881 ± 0.073 | 0.901 ± 0.051 |

## Held-out NLL (lower better)

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.129 ± 0.014 | 1.036 ± 0.295 | 0.855 ± 0.466 | 1.037 ± 0.585 |
| MapWM-Hier | 0.075 ± 0.007 | 0.540 ± 0.196 | 0.661 ± 0.285 | 0.653 ± 0.169 |
| Plain-Flat | 0.070 ± 0.006 | 2.457 ± 1.037 | 2.267 ± 1.102 | 2.378 ± 1.176 |
| Plain-Hier | 0.052 ± 0.005 | 0.631 ± 0.171 | 0.559 ± 0.128 | 0.538 ± 0.068 |
