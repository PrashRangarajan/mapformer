## S2  gradient at initialisation (8 batches, eval mode)

| arm | seed | rate (slope of y_k) | t | cos(g, dJ) | pathway/content grad | rms A_P | rms A_X |
|---|---|---|---|---|---|---|---|
| VanillaEM_P0_r4 | 0 | +4.000e-11 | +0.51 | -0.027 | 3.886e-04 | 8.43e-04 | 3.38e-01 |
| VanillaEM_P0_r4 | 1 | +1.345e-11 | +0.37 | +0.023 | 2.648e-04 | 7.56e-04 | 3.30e-01 |
| VanillaEM_P0_r4 | 2 | +3.626e-11 | +0.79 | +0.016 | 2.622e-04 | 8.80e-04 | 3.20e-01 |
| VanillaEM_P0_r4 | 3 | -1.146e-11 | -0.24 | -0.042 | 2.592e-04 | 5.79e-04 | 3.37e-01 |
| VanillaEM_P0_r4 | 4 | -5.301e-12 | -0.19 | -0.020 | 1.837e-04 | 6.27e-04 | 3.30e-01 |
| VanillaEM_P0_r4 | 5 | +2.417e-11 | +2.38 | +0.011 | 1.745e-04 | 5.65e-04 | 3.43e-01 |
| VanillaEM_P0_r4 | 6 | -2.158e-11 | -0.45 | -0.016 | 4.256e-04 | 6.36e-04 | 3.11e-01 |
| VanillaEM_P0_r4 | 7 | -2.966e-11 | -1.31 | -0.031 | 3.080e-04 | 6.79e-04 | 3.44e-01 |
| EMDoF_alignfree | 0 | +4.000e-11 | +0.51 | -0.027 | 3.884e-04 | 8.43e-04 | 3.38e-01 |
| EMDoF_alignfree | 1 | +1.345e-11 | +0.37 | +0.023 | 2.640e-04 | 7.56e-04 | 3.30e-01 |
| EMDoF_alignfree | 2 | +3.626e-11 | +0.79 | +0.016 | 2.616e-04 | 8.80e-04 | 3.20e-01 |
| EMDoF_alignfree | 3 | -1.146e-11 | -0.24 | -0.042 | 2.592e-04 | 5.79e-04 | 3.37e-01 |
| EMDoF_alignfree | 4 | -5.301e-12 | -0.19 | -0.020 | 1.835e-04 | 6.27e-04 | 3.30e-01 |
| EMDoF_alignfree | 5 | +2.417e-11 | +2.38 | +0.011 | 1.741e-04 | 5.65e-04 | 3.43e-01 |
| EMDoF_alignfree | 6 | -2.158e-11 | -0.45 | -0.016 | 4.253e-04 | 6.36e-04 | 3.11e-01 |
| EMDoF_alignfree | 7 | -2.966e-11 | -1.31 | -0.031 | 3.077e-04 | 6.79e-04 | 3.44e-01 |
| EMDoF_magonly | 0 | +4.000e-11 | +0.51 | -0.027 | 3.877e-04 | 8.43e-04 | 3.38e-01 |
| EMDoF_magonly | 1 | +1.345e-11 | +0.37 | +0.023 | 2.629e-04 | 7.56e-04 | 3.30e-01 |
| EMDoF_magonly | 2 | +3.626e-11 | +0.79 | +0.016 | 2.595e-04 | 8.80e-04 | 3.20e-01 |
| EMDoF_magonly | 3 | -1.146e-11 | -0.24 | -0.042 | 2.583e-04 | 5.79e-04 | 3.37e-01 |
| EMDoF_magonly | 4 | -5.301e-12 | -0.19 | -0.020 | 1.816e-04 | 6.27e-04 | 3.30e-01 |
| EMDoF_magonly | 5 | +2.417e-11 | +2.38 | +0.011 | 1.730e-04 | 5.65e-04 | 3.43e-01 |
| EMDoF_magonly | 6 | -2.158e-11 | -0.45 | -0.016 | 4.248e-04 | 6.36e-04 | 3.11e-01 |
| EMDoF_magonly | 7 | -2.966e-11 | -1.31 | -0.031 | 3.068e-04 | 6.79e-04 | 3.44e-01 |

VanillaEM_P0_r4: |t| < 2 on 7/8 seeds (H-rugged (ii) needs >= 6/8 for P0); sign of rate negative on 4/8

EMDoF_alignfree: |t| < 2 on 7/8 seeds (H-rugged (ii) needs >= 6/8 for P0); sign of rate negative on 4/8

EMDoF_magonly: |t| < 2 on 7/8 seeds (H-rugged (ii) needs >= 6/8 for P0); sign of rate negative on 4/8

## Ruggedness: interior local maxima on the straight path to the linear rewind

Peaks with prominence >= 1% of kappa(0); the registered raw strict count in parentheses (it reads float noise on flat paths -- a recorded deviation).

| set | k<=4 median | k 8-16 median | k 32 median | k 60-64 median | f(1) > f(0) at k>=32 |
|---|---|---|---|---|---|
| init VanillaEM_P0_r4 | 0.0 (1.0) | 0.0 | 1.0 | 2.0 (3.0) | 0.727 |
| init EMDoF_alignfree | 0.0 (1.0) | 0.0 | 1.0 | 2.0 (3.0) | 0.727 |
| init EMDoF_magonly | 0.0 (1.0) | 0.0 | 1.0 | 2.0 (3.0) | 0.727 |
| trained P0 (s0-7) | 0.0 (15.5) | 2.0 | 8.0 | 15.5 (16.0) | 1.000 |

H-rugged (i) init P0: k<=4 median 0.0 (<= 1 needed), k 60-64 median 2.0 (>= 5 needed) -> **NOT MET**

H-rugged (i) trained P0: k<=4 median 0.0 (<= 1 needed), k 60-64 median 15.5 (>= 5 needed) -> **MET**
