# R3: collapsed steps and the wall (post hoc; PREDICTIONS.md)

## (a) Collapse fractions

| batch | arm | collapsed (move, channel) entries, |a| <= 0.05: mean (run range) | channels with |wrap(a(+1))| <= 0.05 |
|---|---|---|---|
| SIGN_MATCHED torus | Signed_r4 | 0.343 (0.28-0.41) | 0.348 |
| SIGN_MATCHED torus | Abs_r4 | 0.695 (0.62-0.88) | 0.688 |
| SIGN_MATCHED torus | Pos_r4 | 0.624 (0.54-0.71) | 0.625 |
| RING_DIAL p0.5 | Signed_r4 | 0.104 (0.06-0.16) | 0.107 |
| RING_DIAL p0.5 | Abs_r4 | 0.302 (0.18-0.56) | 0.293 |
| RING_DIAL p0.75 | Signed_r4 | 0.107 (0.07-0.12) | 0.105 |
| RING_DIAL p0.75 | Abs_r4 | 0.270 (0.20-0.44) | 0.240 |
| RING_DIAL p0.9 | Signed_r4 | 0.156 (0.09-0.22) | 0.160 |
| RING_DIAL p0.9 | Abs_r4 | 0.235 (0.15-0.34) | 0.180 |
| RING_BUDGET | Abs_r4_e900 | 0.366 (0.20-0.48) | 0.344 |
| WRAP_EXISTENCE ring | SO | 0.157 (0.06-0.24) | 0.156 |
| WRAP_EXISTENCE ring | AO | 0.337 (0.18-0.60) | 0.344 |
| MECHANISM ring | SO | 0.145 (0.09-0.19) | 0.150 |
| MECHANISM ring | SO_absmag | 0.171 (0.09-0.23) | 0.166 |
| MECHANISM ring | SO_high | 0.001 (0.00-0.01) | 0.016 |
| MECHANISM ring | AO | 0.368 (0.27-0.50) | 0.363 |
| MECHANISM ring | AO_absmag | 0.424 (0.25-0.67) | 0.441 |
| MECHANISM ring | AO_mid | 0.010 (0.00-0.03) | 0.002 |
| MECHANISM ring | AO_high | 0.001 (0.00-0.01) | 0.014 |
| GENERALITY torus | SO4 | 0.357 (0.27-0.40) | 0.354 |
| GENERALITY torus | AO4 | 0.660 (0.58-0.75) | 0.646 |
| TORUS_START torus | SO4 | 0.402 (0.29-0.47) | 0.410 |
| TORUS_START torus | AO4 | 0.601 (0.41-0.77) | 0.613 |
| TORUS_START torus | SO4_high | 0.062 (0.00-0.18) | 0.074 |
| TORUS_START torus | AO4_high | 0.061 (0.03-0.12) | 0.066 |
| TORUS_START torus | AO4_mid | 0.127 (0.06-0.21) | 0.129 |

## (b) Ring observation-step-free arms: start, usage, gradient at the end

| batch | arm | collapsed entries (of all) | started near 0 (init |a| <= 0.05) / median init |a|, collapsed vs not | median usage, collapsed / not | dL/dDelta > 0 among collapsed | median |dL/dDelta|, collapsed / not | reading |
|---|---|---|---|---|---|---|---|
| MECHANISM ring | AO | 377 of 1024 (0.368) | 0.42 / 0.06 vs 0.33 rad (base rate: 0.20 of all entries start near 0; of those, 0.77 end collapsed) | 4.59 / 4.18 (ratio 1.10) | 0.51 (of which |g| >= 0.1 x non-collapsed median: 0.31) | 7.91e-04 / 4.92e-03 | MIXED |
| MECHANISM ring | AO_absmag | 434 of 1024 (0.424) | 0.38 / 0.07 vs 0.37 rad (base rate: 0.20 of all entries start near 0; of those, 0.81 end collapsed) | 5.99 / 13 (ratio 0.46) | 0.57 (of which |g| >= 0.1 x non-collapsed median: 0.35) | 4.44e-03 / 2.07e-02 | MIXED |
| MECHANISM ring | AO_mid | 10 of 1024 (0.010) | 0.00 / 1.15 vs 1.19 rad (base rate: 0.00 of all entries start near 0; of those, nan end collapsed) | 0.51 / 0.813 (ratio 0.63) | 0.70 (of which |g| >= 0.1 x non-collapsed median: 0.70) | 1.20e-03 / 3.75e-04 | WALL |
| MECHANISM ring | AO_high | 1 of 1024 (0.001) | 0.00 / 1.87 vs 3.06 rad (base rate: 0.00 of all entries start near 0; of those, nan end collapsed) | 0.364 / 0.225 (ratio 1.62) | 1.00 (of which |g| >= 0.1 x non-collapsed median: 0.00) | 1.76e-10 / 2.25e-09 | MIXED |
| MECHANISM ring | SO | 148 of 1024 (0.145) | 0.41 / 0.07 vs 0.20 rad (base rate: 0.20 of all entries start near 0; of those, 0.29 end collapsed) | 0.0406 / 0.117 (ratio 0.35) | 0.52 (of which |g| >= 0.1 x non-collapsed median: 0.16) | 9.22e-12 / 2.45e-10 | MIXED |
| MECHANISM ring | SO_absmag | 175 of 1024 (0.171) | 0.41 / 0.07 vs 0.20 rad (base rate: 0.20 of all entries start near 0; of those, 0.35 end collapsed) | 0.0472 / 0.126 (ratio 0.37) | 0.38 (of which |g| >= 0.1 x non-collapsed median: 0.19) | 7.14e-11 / 4.54e-10 | MIXED |
| WRAP_EXISTENCE ring | AO | 345 of 1024 (0.337) | not stored | 3 / 7.66 (ratio 0.39) | 0.55 (of which |g| >= 0.1 x non-collapsed median: 0.27) | 6.77e-04 / 8.07e-03 | MIXED |
| WRAP_EXISTENCE ring | SO | 161 of 1024 (0.157) | not stored | 0.0385 / 0.141 (ratio 0.27) | 0.58 (of which |g| >= 0.1 x non-collapsed median: 0.20) | 5.91e-12 / 2.09e-10 | MIXED |
