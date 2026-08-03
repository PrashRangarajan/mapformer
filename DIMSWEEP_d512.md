# Hierarchical goal-directed navigation — multi-seed (mean ± std)

Train T_explore=64; eval at listed T_explore (>64 = OOD). Held-out env (seed=10000). Chance=0.25, BFS ceiling=1.00.

Seeds found: MapWM-Flat=[0, 1, 2], MapWM-Hier=[0, 1, 2], Plain-Flat=[0, 1, 2], Plain-Hier=[0, 1, 2]


## Held-out action accuracy

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.969 ± 0.007 | 0.832 ± 0.112 | 0.778 ± 0.119 | 0.768 ± 0.160 |
| MapWM-Hier | 0.983 ± 0.006 | 0.857 ± 0.064 | 0.818 ± 0.085 | 0.848 ± 0.067 |
| Plain-Flat | 0.980 ± 0.002 | 0.623 ± 0.145 | 0.635 ± 0.146 | 0.605 ± 0.149 |
| Plain-Hier | 0.980 ± 0.005 | 0.767 ± 0.128 | 0.780 ± 0.154 | 0.751 ± 0.180 |

## Held-out NLL (lower better)

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.092 ± 0.040 | 0.663 ± 0.297 | 0.743 ± 0.266 | 0.818 ± 0.399 |
| MapWM-Hier | 0.044 ± 0.010 | 0.465 ± 0.242 | 0.709 ± 0.414 | 0.513 ± 0.207 |
| Plain-Flat | 0.050 ± 0.003 | 2.637 ± 0.851 | 2.523 ± 0.869 | 2.871 ± 1.003 |
| Plain-Hier | 0.050 ± 0.012 | 0.870 ± 0.237 | 0.991 ± 0.500 | 1.159 ± 0.687 |
