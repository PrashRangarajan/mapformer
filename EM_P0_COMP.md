# Compositional-motif results — multi-seed (mean ± std)

Seeds requested: [0, 1, 2]. Fresh held-out env (seed=10000). cross_nb_acc = non-blank cross-instance cells (compositional target); exact_acc = exact-revisit recall.

Seeds actually found per variant: MapEM-Flat=[0, 1, 2], VanillaEM_P0=[], MapWM-Flat=[0, 1, 2], MapWM-Hier=[0, 1, 2], Plain-Flat=[0, 1, 2]


## cross_nb_acc (compositional target)

| variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| MapEM-Flat | 0.097 ± 0.013 (n=3) | 0.047 ± 0.012 (n=3) | 0.026 ± 0.011 (n=3) | 0.015 ± 0.010 (n=3) |
| VanillaEM_P0 | nan ± nan (n=0) | nan ± nan (n=0) | nan ± nan (n=0) | nan ± nan (n=0) |
| MapWM-Flat | 0.270 ± 0.030 (n=3) | 0.164 ± 0.021 (n=3) | 0.081 ± 0.006 (n=3) | 0.048 ± 0.012 (n=3) |
| MapWM-Hier | 0.423 ± 0.144 (n=3) | 0.314 ± 0.144 (n=3) | 0.209 ± 0.166 (n=3) | 0.166 ± 0.174 (n=3) |
| Plain-Flat | 0.213 ± 0.001 (n=3) | 0.100 ± 0.002 (n=3) | 0.038 ± 0.001 (n=3) | 0.018 ± 0.001 (n=3) |

## exact_acc (fine recall)

| variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| MapEM-Flat | 0.788 ± 0.168 (n=3) | 0.696 ± 0.167 (n=3) | 0.588 ± 0.135 (n=3) | 0.519 ± 0.070 (n=3) |
| VanillaEM_P0 | nan ± nan (n=0) | nan ± nan (n=0) | nan ± nan (n=0) | nan ± nan (n=0) |
| MapWM-Flat | 0.924 ± 0.020 (n=3) | 0.889 ± 0.021 (n=3) | 0.767 ± 0.033 (n=3) | 0.646 ± 0.046 (n=3) |
| MapWM-Hier | 0.952 ± 0.035 (n=3) | 0.935 ± 0.047 (n=3) | 0.864 ± 0.098 (n=3) | 0.756 ± 0.165 (n=3) |
| Plain-Flat | 0.905 ± 0.000 (n=3) | 0.794 ± 0.002 (n=3) | 0.632 ± 0.007 (n=3) | 0.540 ± 0.016 (n=3) |

## cross_nll (lower better)

| variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| MapEM-Flat | 1.716 ± 0.157 (n=3) | 1.908 ± 0.150 (n=3) | 2.046 ± 0.108 (n=3) | 2.087 ± 0.083 (n=3) |
| VanillaEM_P0 | nan ± nan (n=0) | nan ± nan (n=0) | nan ± nan (n=0) | nan ± nan (n=0) |
| MapWM-Flat | 1.371 ± 0.088 (n=3) | 1.596 ± 0.042 (n=3) | 2.144 ± 0.311 (n=3) | 2.577 ± 0.544 (n=3) |
| MapWM-Hier | 1.097 ± 0.458 (n=3) | 1.276 ± 0.480 (n=3) | 1.804 ± 0.802 (n=3) | 2.402 ± 1.242 (n=3) |
| Plain-Flat | 1.598 ± 0.005 (n=3) | 1.826 ± 0.010 (n=3) | 2.025 ± 0.035 (n=3) | 2.124 ± 0.012 (n=3) |
