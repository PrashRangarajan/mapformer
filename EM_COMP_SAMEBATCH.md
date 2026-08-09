# Compositional-motif results — multi-seed (mean ± std)

Seeds requested: [0, 1, 2]. Fresh held-out env (seed=10000). cross_nb_acc = non-blank cross-instance cells (compositional target); exact_acc = exact-revisit recall.

Seeds actually found per variant: MapWM-Flat=[0, 1, 2], MapEM-Flat=[0, 1, 2], VanillaEM_P0=[0, 1, 2]


## cross_nb_acc (compositional target)

| variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| MapWM-Flat | 0.270 ± 0.030 (n=3) | 0.164 ± 0.021 (n=3) | 0.081 ± 0.006 (n=3) | 0.048 ± 0.012 (n=3) |
| MapEM-Flat | 0.097 ± 0.013 (n=3) | 0.047 ± 0.012 (n=3) | 0.026 ± 0.011 (n=3) | 0.015 ± 0.010 (n=3) |
| VanillaEM_P0 | 0.264 ± 0.025 (n=3) | 0.152 ± 0.015 (n=3) | 0.076 ± 0.005 (n=3) | 0.041 ± 0.004 (n=3) |

## exact_acc (fine recall)

| variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| MapWM-Flat | 0.924 ± 0.020 (n=3) | 0.889 ± 0.021 (n=3) | 0.767 ± 0.033 (n=3) | 0.646 ± 0.046 (n=3) |
| MapEM-Flat | 0.788 ± 0.168 (n=3) | 0.696 ± 0.167 (n=3) | 0.588 ± 0.135 (n=3) | 0.519 ± 0.070 (n=3) |
| VanillaEM_P0 | 0.894 ± 0.124 (n=3) | 0.870 ± 0.115 (n=3) | 0.825 ± 0.104 (n=3) | 0.769 ± 0.083 (n=3) |

## cross_nll (lower better)

| variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| MapWM-Flat | 1.371 ± 0.088 (n=3) | 1.596 ± 0.042 (n=3) | 2.144 ± 0.311 (n=3) | 2.577 ± 0.544 (n=3) |
| MapEM-Flat | 1.716 ± 0.157 (n=3) | 1.908 ± 0.150 (n=3) | 2.046 ± 0.108 (n=3) | 2.087 ± 0.083 (n=3) |
| VanillaEM_P0 | 1.433 ± 0.112 (n=3) | 1.668 ± 0.117 (n=3) | 1.819 ± 0.106 (n=3) | 1.878 ± 0.093 (n=3) |
