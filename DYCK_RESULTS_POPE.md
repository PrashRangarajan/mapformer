# READING -- PoPE and MapPoPE on Dyck-2 (amendment 1 of DYCK_PREREG.md)

- **MapPoPE-1L is the best arm in the batch**: 0.988 at the training cell, **0.927 at L128 D12**
  (MapWM-1L 0.868, MapEM-1L 0.888; the paper's MapWM-1L 0.94). Its grid is the flattest measured:
  0.923-0.929 across every depth at L=128.
- Encoding on the path-integrated row: MapPoPE - MapWM **+0.058** (MDE 0.039, 8/8) DETECTABLE.
- Position under PoPE: MapPoPE-1L - PoPE-1L **+0.312** (8/8), MapPoPE-1L - PoPE-2L +0.454 (8/8).
- Encoding on the index row: PoPE-1L - RoPE-1L +0.049 (MDE 0.054, 6/8) and PoPE-2L - RoPE-2L -0.025,
  both unmeasured. PoPE alone behaves like RoPE: fits the training cell, collapses with length.
- No evidence of an interaction at 1L: (MapPoPE - MapWM) - (PoPE - RoPE) = +0.009 (MDE 0.074, 4/8);
  PoPE's gain is similar in size on both rows and is detectable only where path integration keeps
  the seed variance low.
- Floor: MapPoPE-1L is +0.042 over the best n-gram at L128 D12 (MDE 0.046, 6/8 seeds above) --
  formally at the floor, and the only arm pointing above it.
- Its W_in geometry is bimodal: same-type opposition -0.65 +/- 0.69, i.e. some seeds do not use
  opposite directions at all.

# Dyck-2 replication -- `/home/prashr/mapformer/runs/dyck_bs128`

Pre-registration: `DYCK_PREREG.md`. F1 = mean per-prefix valid-continuation F1 (Goodale et al.), 1024 sequences per cell. Floor = best n-gram (orders 1-6) from `DYCK_GATES.md`.

## Seeds, convergence (rule 10)

| arm | params | seeds | final CE (floor 0.800) | slope /1k steps, median seed | max slope |
|---|---|---|---|---|---|
| MapWM-1L_r2 | 50,981 | 8 | 0.8800 | -0.0017 | -0.0036 |
| MapEM-1L_r2 | 51,109 | 8 | 0.8955 | -0.0016 | -0.0037 |
| RoPE-1L | 50,757 | 8 | 1.0248 | -0.0003 | -0.0063 |
| RoPE-2L | 398,085 | 8 | 0.8853 | -0.0021 | -0.0074 |
| CoPE-1L | 59,013 | 8 | 1.0370 | -0.0009 | -0.0051 |
| CoPE-2L | 431,109 | 8 | 0.8976 | -0.0018 | -0.0097 |
| MapWM-1L_r4 | 51,173 | 8 | 0.8537 | -0.0031 | -0.0068 |
| MapEM-1L_r4 | 51,301 | 8 | 0.8836 | -0.0029 | -0.0067 |
| PoPE-1L | 50,821 | 8 | 1.0309 | -0.0002 | -0.0044 |
| PoPE-2L | 398,341 | 8 | 0.8915 | -0.0014 | -0.0056 |
| MapPoPE-1L_r2 | 51,141 | 8 | 0.8835 | -0.0022 | -0.0094 |

## Headline cells (mean +/- sd over seeds; paper value in brackets)

| arm | L32_D4 | L128_D4 | L32_D12 | L128_D12 |
|---|---|---|---|---|
| MapWM-1L_r2 | 0.985 +/- 0.012 [1.00] | 0.900 +/- 0.050 [0.97] | 0.942 +/- 0.036 [0.98] | 0.868 +/- 0.045 [0.94] |
| MapEM-1L_r2 | 0.985 +/- 0.010 [1.00] | 0.878 +/- 0.045 [0.99] | 0.949 +/- 0.031 [0.98] | 0.888 +/- 0.037 [0.95] |
| RoPE-1L | 0.910 +/- 0.005 [0.91] | 0.575 +/- 0.064 [0.78] | 0.822 +/- 0.031 [0.86] | 0.566 +/- 0.058 [0.77] |
| RoPE-2L | 0.947 +/- 0.003 [0.97] | 0.521 +/- 0.050 [0.59] | 0.691 +/- 0.017 [0.66] | 0.498 +/- 0.034 [0.50] |
| CoPE-1L | 0.900 +/- 0.015 [0.92] | 0.703 +/- 0.013 [0.84] | 0.854 +/- 0.021 [0.93] | 0.703 +/- 0.009 |
| CoPE-2L | 0.937 +/- 0.003 [0.96] | 0.616 +/- 0.094 [0.78] | 0.668 +/- 0.019 [0.82] | 0.542 +/- 0.042 |
| MapWM-1L_r4 | 0.965 +/- 0.014 | 0.681 +/- 0.140 | 0.863 +/- 0.062 | 0.688 +/- 0.117 |
| MapEM-1L_r4 | 0.979 +/- 0.013 | 0.867 +/- 0.068 | 0.946 +/- 0.029 | 0.870 +/- 0.063 |
| PoPE-1L | 0.911 +/- 0.002 | 0.647 +/- 0.021 | 0.882 +/- 0.005 | 0.615 +/- 0.019 |
| PoPE-2L | 0.948 +/- 0.001 | 0.504 +/- 0.047 | 0.680 +/- 0.023 | 0.473 +/- 0.044 |
| MapPoPE-1L_r2 | 0.988 +/- 0.009 | 0.923 +/- 0.053 | 0.976 +/- 0.025 | 0.927 +/- 0.047 |
| n-gram floor | 0.904 | 0.896 | 0.903 | 0.884 |
| CE-optimal predictor | 0.911 | 0.933 | 0.753 | 0.944 |

## Full grids (ours / paper, rows D, columns L)

**MapWM-1L_r2**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.985 / 1.00 | 0.962 / 0.99 | 0.926 / 0.98 | 0.900 / 0.97 |
| 6 | 0.969 / 0.99 | 0.940 / 0.97 | 0.905 / 0.96 | 0.880 / 0.96 |
| 8 | 0.960 / 0.98 | 0.929 / 0.96 | 0.895 / 0.95 | 0.872 / 0.95 |
| 12 | 0.942 / 0.98 | 0.915 / 0.94 | 0.889 / 0.94 | 0.868 / 0.94 |

max |ours - paper| = 0.080; mean (ours - paper) = -0.045; cells within 0.03: 6/16

**MapEM-1L_r2**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.985 / 1.00 | 0.949 / 1.00 | 0.906 / 0.99 | 0.878 / 0.99 |
| 6 | 0.970 / 1.00 | 0.939 / 0.98 | 0.902 / 0.98 | 0.878 / 0.97 |
| 8 | 0.963 / 0.99 | 0.938 / 0.97 | 0.904 / 0.96 | 0.880 / 0.96 |
| 12 | 0.949 / 0.98 | 0.935 / 0.95 | 0.907 / 0.95 | 0.888 / 0.95 |

max |ours - paper| = 0.112; mean (ours - paper) = -0.053; cells within 0.03: 3/16

**RoPE-1L**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.910 / 0.91 | 0.697 / 0.84 | 0.616 / 0.80 | 0.575 / 0.78 |
| 6 | 0.881 / 0.89 | 0.689 / 0.84 | 0.612 / 0.80 | 0.574 / 0.78 |
| 8 | 0.871 / 0.88 | 0.683 / 0.83 | 0.608 / 0.79 | 0.571 / 0.78 |
| 12 | 0.822 / 0.86 | 0.674 / 0.81 | 0.601 / 0.78 | 0.566 / 0.77 |

max |ours - paper| = 0.209; mean (ours - paper) = -0.137; cells within 0.03: 3/16

**RoPE-2L**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.947 / 0.97 | 0.674 / 0.70 | 0.573 / 0.63 | 0.521 / 0.59 |
| 6 | 0.877 / 0.89 | 0.647 / 0.66 | 0.554 / 0.60 | 0.508 / 0.56 |
| 8 | 0.822 / 0.79 | 0.634 / 0.61 | 0.545 / 0.56 | 0.500 / 0.53 |
| 12 | 0.691 / 0.66 | 0.626 / 0.54 | 0.541 / 0.52 | 0.498 / 0.50 |

max |ours - paper| = 0.086; mean (ours - paper) = -0.010; cells within 0.03: 8/16

**CoPE-1L**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.900 | 0.789 | 0.732 | 0.703 |
| 6 | 0.874 | 0.786 | 0.733 | 0.705 |
| 8 | 0.863 | 0.786 | 0.731 | 0.704 |
| 12 | 0.854 | 0.783 | 0.728 | 0.703 |

**CoPE-2L**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.937 | 0.784 | 0.682 | 0.616 |
| 6 | 0.859 | 0.731 | 0.634 | 0.577 |
| 8 | 0.779 | 0.696 | 0.605 | 0.550 |
| 12 | 0.668 | 0.667 | 0.594 | 0.542 |

**MapWM-1L_r4**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.965 | 0.792 | 0.716 | 0.681 |
| 6 | 0.927 | 0.780 | 0.710 | 0.676 |
| 8 | 0.907 | 0.780 | 0.715 | 0.680 |
| 12 | 0.863 | 0.784 | 0.721 | 0.688 |

**MapEM-1L_r4**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.979 | 0.937 | 0.893 | 0.867 |
| 6 | 0.970 | 0.933 | 0.896 | 0.872 |
| 8 | 0.962 | 0.930 | 0.896 | 0.871 |
| 12 | 0.946 | 0.923 | 0.891 | 0.870 |

**PoPE-1L**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.911 | 0.756 | 0.691 | 0.647 |
| 6 | 0.883 | 0.743 | 0.680 | 0.640 |
| 8 | 0.877 | 0.734 | 0.670 | 0.632 |
| 12 | 0.882 | 0.714 | 0.651 | 0.615 |

**PoPE-2L**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.948 | 0.666 | 0.565 | 0.504 |
| 6 | 0.873 | 0.639 | 0.543 | 0.490 |
| 8 | 0.796 | 0.621 | 0.530 | 0.479 |
| 12 | 0.680 | 0.610 | 0.521 | 0.473 |

**MapPoPE-1L_r2**

| D \ L | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| 4 | 0.988 | 0.973 | 0.946 | 0.923 |
| 6 | 0.981 | 0.967 | 0.947 | 0.929 |
| 8 | 0.976 | 0.962 | 0.942 | 0.927 |
| 12 | 0.976 | 0.958 | 0.939 | 0.927 |

## Registered verdicts

- **R1 MapWM-1L_r2** L32 D4 = 0.985 (needs >= 0.97) -> **REPLICATES**
- **R2 MapWM-1L_r2** L128 D12 = 0.868 (needs >= 0.91) -> **DOES NOT REPLICATE**
- **R1 MapEM-1L_r2** L32 D4 = 0.985 (needs >= 0.97) -> **REPLICATES**
- **R2 MapEM-1L_r2** L128 D12 = 0.888 (needs >= 0.92) -> **DOES NOT REPLICATE**
- **R3 RoPE-2L** L32 D4 0.947 (>= 0.94), L128 D4 0.521 and L128 D12 0.498 (both <= 0.70) -> **REPLICATES**
- **R4 RoPE-1L** L32 D4 0.910 (< 0.94) -> **REPLICATES**
- **R5 CoPE-1L** (exploratory) L32 D4 0.900 [0.92], L128 D4 0.703 [0.84] -> **OUTSIDE 0.03**
- **R5 CoPE-2L** (exploratory) L32 D4 0.937 [0.96], L128 D4 0.616 [0.78] -> **OUTSIDE 0.03**

**R6 contrasts at L128 D12 (paired by seed)**

| contrast | paper | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| MapWM-1L_r2 - RoPE-2L | +0.44 | +0.370 | 0.054 | 0.053 | 8/8 | DETECTABLE |
| MapWM-1L_r2 - RoPE-1L | +0.17 | +0.302 | 0.073 | 0.072 | 8/8 | DETECTABLE |
| MapEM-1L_r2 - RoPE-2L | +0.45 | +0.390 | 0.056 | 0.056 | 8/8 | DETECTABLE |
| MapEM-1L_r2 - MapWM-1L_r2 | +0.01 | +0.020 | 0.055 | 0.054 | 5/8 | unmeasured |
| MapPoPE-1L_r2 - MapWM-1L_r2 | -- | +0.058 | 0.040 | 0.039 | 8/8 | DETECTABLE |
| PoPE-1L - RoPE-1L | -- | +0.049 | 0.054 | 0.054 | 6/8 | unmeasured |
| PoPE-2L - RoPE-2L | -- | -0.025 | 0.056 | 0.056 | 3/8 | unmeasured |
| MapPoPE-1L_r2 - PoPE-1L | -- | +0.312 | 0.044 | 0.043 | 8/8 | DETECTABLE |
| MapPoPE-1L_r2 - PoPE-2L | -- | +0.454 | 0.082 | 0.081 | 8/8 | DETECTABLE |

**R7 floor reading at L128 D12** (best n-gram 0.884)

- MapWM-1L_r2: -0.016 over the floor (MDE 0.045, 2/8 seeds above) -> at the floor
- MapEM-1L_r2: +0.003 over the floor (MDE 0.036, 5/8 seeds above) -> at the floor
- RoPE-1L: -0.319 over the floor (MDE 0.058, 0/8 seeds above) -> BELOW
- RoPE-2L: -0.386 over the floor (MDE 0.034, 0/8 seeds above) -> BELOW
- CoPE-1L: -0.181 over the floor (MDE 0.009, 0/8 seeds above) -> BELOW
- CoPE-2L: -0.343 over the floor (MDE 0.042, 0/8 seeds above) -> BELOW
- MapWM-1L_r4: -0.196 over the floor (MDE 0.115, 1/8 seeds above) -> BELOW
- MapEM-1L_r4: -0.014 over the floor (MDE 0.063, 4/8 seeds above) -> at the floor
- PoPE-1L: -0.270 over the floor (MDE 0.019, 0/8 seeds above) -> BELOW
- PoPE-2L: -0.412 over the floor (MDE 0.044, 0/8 seeds above) -> BELOW
- MapPoPE-1L_r2: +0.042 over the floor (MDE 0.046, 6/8 seeds above) -> at the floor

## Rule 9: r(final loss, F1) across all runs

- L32_D4: r = -0.797
- L128_D4: r = -0.236
- L32_D12: r = +0.029
- L128_D12: r = -0.223

## Mechanism (Fig 3f): cosines of W_in on bracket embeddings

| arm | cos('(' , ')') | cos('[' , ']') | abs cos('(' , '[') | mean norm brackets | norm BOS |
|---|---|---|---|---|---|
| MapWM-1L_r2 | -0.875 +/- 0.245 | -0.901 +/- 0.279 | +0.480 +/- 0.334 | +1.321 +/- 0.304 | +0.812 +/- 0.470 |
| MapEM-1L_r2 | -1.000 +/- 0.001 | -1.000 +/- 0.000 | +0.396 +/- 0.187 | +1.876 +/- 0.274 | +0.888 +/- 0.416 |
| MapWM-1L_r4 | -0.774 +/- 0.284 | -0.887 +/- 0.115 | +0.226 +/- 0.153 | +1.412 +/- 0.142 | +1.068 +/- 0.290 |
| MapEM-1L_r4 | -0.979 +/- 0.046 | -0.946 +/- 0.091 | +0.489 +/- 0.130 | +2.282 +/- 0.399 | +0.954 +/- 0.453 |
| MapPoPE-1L_r2 | -0.648 +/- 0.685 | -0.676 +/- 0.688 | +0.371 +/- 0.356 | +1.449 +/- 0.216 | +1.099 +/- 0.592 |
