# Modular-clock navigation — multi-seed (mean ± std)

Train T_explore=64; eval at listed T_explore (>64 = OOD). Held-out env (seed=10000). Chance=0.25, BFS ceiling=1.00.

Seeds found: MapWM-Flat=[0, 1, 2], MapWM-Hier=[0, 1, 2], MapWM-Hier-CoarsePI=[0, 1, 2], PoPE-Flat=[0, 1, 2], MapPoPE-Hier=[0, 1, 2], MapPoPE-Hier-CoarseIdx=[0, 1, 2], Plain-Flat=[0, 1, 2]


## Held-out action accuracy

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.981 ± 0.001 | 0.733 ± 0.106 | 0.619 ± 0.054 | 0.547 ± 0.147 |
| MapWM-Hier | 0.984 ± 0.001 | 0.629 ± 0.016 | 0.646 ± 0.034 | 0.644 ± 0.027 |
| MapWM-Hier-CoarsePI | 0.984 ± 0.001 | 0.618 ± 0.016 | 0.630 ± 0.035 | 0.631 ± 0.014 |
| PoPE-Flat | 0.975 ± 0.005 | 0.771 ± 0.064 | 0.656 ± 0.076 | 0.636 ± 0.065 |
| MapPoPE-Hier | 0.978 ± 0.001 | 0.831 ± 0.028 | 0.623 ± 0.021 | 0.596 ± 0.006 |
| MapPoPE-Hier-CoarseIdx | 0.978 ± 0.002 | 0.814 ± 0.053 | 0.660 ± 0.056 | 0.626 ± 0.027 |
| Plain-Flat | 0.981 ± 0.000 | 0.604 ± 0.009 | 0.607 ± 0.047 | 0.595 ± 0.016 |

## Held-out NLL (lower better)

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.062 ± 0.006 | 1.308 ± 0.483 | 1.612 ± 0.145 | 2.174 ± 0.362 |
| MapWM-Hier | 0.048 ± 0.006 | 2.332 ± 0.210 | 2.054 ± 0.261 | 1.816 ± 0.229 |
| MapWM-Hier-CoarsePI | 0.047 ± 0.004 | 2.618 ± 0.128 | 2.291 ± 0.523 | 1.970 ± 0.337 |
| PoPE-Flat | 0.090 ± 0.013 | 1.246 ± 0.331 | 2.215 ± 0.486 | 2.757 ± 0.481 |
| MapPoPE-Hier | 0.080 ± 0.002 | 0.856 ± 0.102 | 2.409 ± 0.353 | 3.234 ± 0.385 |
| MapPoPE-Hier-CoarseIdx | 0.082 ± 0.005 | 1.153 ± 0.357 | 2.258 ± 0.815 | 2.599 ± 0.638 |
| Plain-Flat | 0.069 ± 0.003 | 2.407 ± 0.241 | 2.343 ± 0.502 | 2.382 ± 0.487 |
