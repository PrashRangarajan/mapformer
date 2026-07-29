# Modular-clock navigation — multi-seed (mean ± std)

Train T_explore=64; eval at listed T_explore (>64 = OOD). Held-out env (seed=10000). Chance=0.25, BFS ceiling=1.00.

Seeds found: MapWM-Flat=[0], MapWM-Hier=[0], MapWM-Hier-CoarsePI=[0], PoPE-Flat=[0], MapPoPE-Hier=[0], MapPoPE-Hier-CoarseIdx=[0], Plain-Flat=[0]


## Held-out action accuracy

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.981 ± 0.000 | 0.702 ± 0.000 | 0.585 ± 0.000 | 0.566 ± 0.000 |
| MapWM-Hier | 0.984 ± 0.000 | 0.641 ± 0.000 | 0.675 ± 0.000 | 0.681 ± 0.000 |
| MapWM-Hier-CoarsePI | 0.984 ± 0.000 | 0.602 ± 0.000 | 0.605 ± 0.000 | 0.648 ± 0.000 |
| PoPE-Flat | 0.980 ± 0.000 | 0.771 ± 0.000 | 0.619 ± 0.000 | 0.589 ± 0.000 |
| MapPoPE-Hier | 0.977 ± 0.000 | 0.803 ± 0.000 | 0.607 ± 0.000 | 0.597 ± 0.000 |
| MapPoPE-Hier-CoarseIdx | 0.975 ± 0.000 | 0.830 ± 0.000 | 0.668 ± 0.000 | 0.626 ± 0.000 |
| Plain-Flat | 0.981 ± 0.000 | 0.617 ± 0.000 | 0.673 ± 0.000 | 0.617 ± 0.000 |

## Held-out NLL (lower better)

| variant | T_exp=64 | T_exp=128 | T_exp=192 | T_exp=256 |
|---|---|---|---|---|
| MapWM-Flat | 0.059 ± 0.000 | 1.378 ± 0.000 | 1.789 ± 0.000 | 2.135 ± 0.000 |
| MapWM-Hier | 0.044 ± 0.000 | 2.432 ± 0.000 | 1.788 ± 0.000 | 1.492 ± 0.000 |
| MapWM-Hier-CoarsePI | 0.046 ± 0.000 | 2.506 ± 0.000 | 2.425 ± 0.000 | 1.887 ± 0.000 |
| PoPE-Flat | 0.075 ± 0.000 | 1.198 ± 0.000 | 2.185 ± 0.000 | 2.877 ± 0.000 |
| MapPoPE-Hier | 0.083 ± 0.000 | 0.968 ± 0.000 | 2.908 ± 0.000 | 3.401 ± 0.000 |
| MapPoPE-Hier-CoarseIdx | 0.089 ± 0.000 | 0.960 ± 0.000 | 2.023 ± 0.000 | 2.466 ± 0.000 |
| Plain-Flat | 0.073 ± 0.000 | 2.299 ± 0.000 | 1.649 ± 0.000 | 1.736 ± 0.000 |
