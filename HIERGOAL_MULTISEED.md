# Hierarchical goal-directed navigation — multi-seed (mean ± std)

Train T_explore=64; eval at listed T_explore (>64 = OOD). Held-out env (seed=10000). Chance=0.25, BFS ceiling=1.00.

Seeds found: MapWM-Flat=[0, 1, 2], MapWM-Hier=[0, 1, 2], MapWM-Hier-CoarseIdx=[0, 1, 2], MapWM-Hier-CoarsePI=[0, 1, 2], PoPE-Flat=[0, 1, 2], MapPoPE-Flat=[0, 1, 2], MapPoPE-Hier=[0, 1, 2], Plain-Flat=[0, 1, 2], Plain-Hier=[0, 1, 2]


## Held-out action accuracy

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.958 ± 0.005 | 0.656 ± 0.206 | 0.746 ± 0.179 | 0.727 ± 0.188 |
| MapWM-Hier | 0.963 ± 0.004 | 0.907 ± 0.026 | 0.849 ± 0.065 | 0.853 ± 0.059 |
| MapWM-Hier-CoarseIdx | 0.967 ± 0.002 | 0.846 ± 0.012 | 0.862 ± 0.012 | 0.845 ± 0.056 |
| MapWM-Hier-CoarsePI | 0.961 ± 0.004 | 0.915 ± 0.019 | 0.890 ± 0.012 | 0.894 ± 0.009 |
| PoPE-Flat | 0.952 ± 0.001 | 0.950 ± 0.001 | 0.950 ± 0.000 | 0.947 ± 0.001 |
| MapPoPE-Flat | 0.952 ± 0.000 | 0.950 ± 0.001 | 0.951 ± 0.000 | 0.948 ± 0.000 |
| MapPoPE-Hier | 0.951 ± 0.001 | 0.949 ± 0.000 | 0.950 ± 0.000 | 0.948 ± 0.000 |
| Plain-Flat | 0.966 ± 0.001 | 0.548 ± 0.084 | 0.669 ± 0.106 | 0.591 ± 0.117 |
| Plain-Hier | 0.968 ± 0.001 | 0.700 ± 0.138 | 0.682 ± 0.122 | 0.624 ± 0.104 |

## Held-out NLL (lower better)

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.151 ± 0.020 | 0.977 ± 0.553 | 0.885 ± 0.608 | 1.036 ± 0.757 |
| MapWM-Hier | 0.114 ± 0.027 | 0.356 ± 0.076 | 0.510 ± 0.145 | 0.511 ± 0.115 |
| MapWM-Hier-CoarseIdx | 0.094 ± 0.017 | 0.516 ± 0.032 | 0.480 ± 0.029 | 0.519 ± 0.088 |
| MapWM-Hier-CoarsePI | 0.129 ± 0.022 | 0.359 ± 0.102 | 0.421 ± 0.057 | 0.441 ± 0.068 |
| PoPE-Flat | 0.176 ± 0.001 | 0.181 ± 0.002 | 0.179 ± 0.001 | 0.188 ± 0.002 |
| MapPoPE-Flat | 0.176 ± 0.000 | 0.180 ± 0.001 | 0.181 ± 0.002 | 0.191 ± 0.002 |
| MapPoPE-Hier | 0.175 ± 0.000 | 0.180 ± 0.001 | 0.185 ± 0.006 | 0.194 ± 0.006 |
| Plain-Flat | 0.106 ± 0.015 | 1.606 ± 0.590 | 1.131 ± 0.327 | 1.374 ± 0.459 |
| Plain-Hier | 0.094 ± 0.001 | 0.857 ± 0.310 | 0.886 ± 0.285 | 0.975 ± 0.202 |
