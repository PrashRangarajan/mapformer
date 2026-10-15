# R1: SOLVED counts by criterion (post hoc; PREDICTIONS.md)

Check at 0.05 against the registered counts: ALL REPRODUCE

| batch | arm | n | loss < 0.02 | loss < 0.05 | loss < 0.1 | acc >= 0.99 | final-5% losses, sorted |
|---|---|---|---|---|---|---|---|
| SIGN_MATCHED torus | Signed_r4 | 8 | 8 | 8 (reg 8) | 8 | 7 | 0.003 0.003 0.003 0.004 0.004 0.004 0.005 0.008 |
| SIGN_MATCHED torus | Abs_r4 | 8 | 0 | 0 (reg 0) | 0 | 0 | 0.201 0.314 0.349 0.446 0.492 0.834 0.962 1.043 |
| SIGN_MATCHED torus | Pos_r4 | 8 | 0 | 1 (reg 1) | 1 | 1 | 0.049 0.158 0.295 0.345 1.009 1.084 1.250 1.510 |
| RING_DIAL p0.5 | Signed_r4 | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| RING_DIAL p0.5 | Abs_r4 | 8 | 2 | 2 (reg 2) | 4 | 2 | 0.003 0.003 0.055 0.099 0.102 0.128 0.154 0.217 |
| RING_DIAL p0.75 | Signed_r4 | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| RING_DIAL p0.75 | Abs_r4 | 8 | 1 | 1 (reg 1) | 2 | 2 | 0.019 0.098 0.100 0.100 0.132 0.149 0.172 0.180 |
| RING_DIAL p0.9 | Signed_r4 | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 |
| RING_DIAL p0.9 | Abs_r4 | 8 | 4 | 4 (reg 4) | 4 | 4 | 0.003 0.003 0.004 0.012 0.148 0.168 0.376 0.435 |
| RING_BUDGET | Abs_r4_e300 | 8 | 3 | 3 (reg 3) | 4 | 3 | 0.003 0.008 0.012 0.098 0.150 0.197 0.252 0.301 |
| RING_BUDGET | Abs_r4_e900 | 8 | 3 | 3 (reg 3) | 5 | 3 | 0.000 0.002 0.002 0.056 0.070 0.129 0.195 0.215 |
| WRAP_EXISTENCE ring | SO | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| WRAP_EXISTENCE ring | AO | 8 | 0 | 0 (reg 0) | 0 | 0 | 0.206 0.216 0.225 0.245 0.314 0.343 0.422 0.581 |
| WRAP_EXISTENCE ring | SO_bal | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| WRAP_EXISTENCE ring | AO_bal | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| WRAP_EXISTENCE ring | SO_cont | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| MECHANISM ring | SO | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| MECHANISM ring | SO_same | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| MECHANISM ring | SO_absmag | 8 | 7 | 7 (reg 7) | 7 | 7 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.292 |
| MECHANISM ring | SO_high | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.002 0.016 |
| MECHANISM ring | AO | 8 | 0 | 0 (reg 0) | 1 | 0 | 0.063 0.167 0.317 0.330 0.416 0.442 0.454 0.508 |
| MECHANISM ring | AO_absmag | 8 | 0 | 0 (reg 0) | 0 | 0 | 0.202 0.227 0.405 0.523 0.544 0.554 0.609 0.665 |
| MECHANISM ring | AO_mid | 8 | 1 | 2 (reg 2) | 3 | 1 | 0.004 0.043 0.078 0.105 0.235 0.250 0.266 0.367 |
| MECHANISM ring | AO_high | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.004 0.005 0.005 |
| ACCOUNTS T2 ring | SO_same | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| ACCOUNTS T2 ring | SO_same_fz | 8 | 7 | 7 (reg 7) | 7 | 7 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.170 |
| ACCOUNTS T2 ring | SO_absmag | 8 | 7 | 7 (reg 7) | 7 | 7 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.427 |
| ACCOUNTS T2 ring | SO_absmag_fz | 8 | 1 | 1 (reg 1) | 1 | 1 | 0.000 0.232 0.252 0.326 0.329 0.346 0.403 0.890 |
| GENERALITY torus | SO4 | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.001 0.001 0.001 0.001 0.003 0.004 0.004 0.004 |
| GENERALITY torus | AO4 | 8 | 0 | 0 (reg 0) | 0 | 0 | 0.143 0.224 0.425 0.433 0.434 0.443 0.532 0.848 |
| GENERALITY torus | SO_bal | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.001 0.001 0.001 |
| GENERALITY torus | AO_bal | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.001 0.001 0.003 0.003 0.003 |
| GENERALITY torus | AO_bal_hi | 8 | 0 | 6 (reg 6) | 6 | 6 | 0.023 0.025 0.026 0.036 0.039 0.050 0.210 0.213 |
| GENERALITY ring | Signed_r4 | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| GENERALITY ring | CoPEF | 8 | 0 | 0 (reg 0) | 1 | 0 | 0.094 0.156 0.208 0.421 0.618 0.639 0.745 0.898 |
| GENERALITY ring | CARoPEF | 8 | 0 | 0 (reg 0) | 0 | 0 | 0.880 0.970 0.985 1.004 1.014 1.081 1.124 1.142 |
| GENERALITY ring | RoPE | 8 | 0 | 0 (reg 0) | 0 | 0 | 1.011 1.021 1.025 1.030 1.040 1.059 1.114 1.132 |
| TORUS_START torus | SO4 | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.001 0.001 0.001 0.001 0.001 0.003 0.003 0.004 |
| TORUS_START torus | AO4 | 8 | 0 | 1 (reg 1) | 1 | 1 | 0.027 0.235 0.336 0.454 0.462 0.539 0.584 0.623 |
| TORUS_START torus | AO4_high | 8 | 0 | 0 (reg 0) | 0 | 0 | 0.149 0.252 0.279 0.320 0.404 0.533 0.666 0.758 |
| TORUS_START torus | SO4_high | 8 | 1 | 2 (reg 2) | 2 | 2 | 0.017 0.049 0.300 0.313 0.332 0.335 0.422 0.862 |
| TORUS_START torus | AO4_mid | 8 | 0 | 0 (reg 0) | 1 | 0 | 0.091 0.229 0.307 0.316 0.373 0.469 0.600 0.680 |
| WRAP_COPE ring | CoPEF | 8 | 0 | 1 (reg 1) | 1 | 1 | 0.048 0.213 0.257 0.295 0.603 1.059 1.076 1.422 |
| ACCOUNTS T4 | CoPEF_ring | 16 | 0 | 0 (reg 0) | 2 | 2 | 0.057 0.065 0.162 0.219 0.341 0.463 0.588 0.594 0.595 0.595 0.606 0.646 0.687 0.841 0.860 1.228 |
| ACCOUNTS T4 | CoPEF_line | 16 | 0 | 1 (reg 1) | 8 | 7 | 0.041 0.062 0.065 0.072 0.083 0.085 0.089 0.091 0.451 0.465 0.536 0.587 0.643 0.734 0.774 1.347 |
| ACCOUNTS T4 | Signed_r4_line | 8 | 8 | 8 (reg 8) | 8 | 8 | 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 |
| ACCOUNTS T4 | RoPE_line | 8 | 0 | 0 (reg 0) | 0 | 0 | 0.904 0.911 0.919 0.927 0.929 0.933 0.964 1.091 |

## Registered contrasts at each criterion: x k/n vs ref k/n (Fisher two-sided p)

| batch | x vs ref | loss < 0.02 | loss < 0.05 | loss < 0.1 | acc >= 0.99 |
|---|---|---|---|---|---|
| SIGN_MATCHED torus | Abs_r4 vs Signed_r4 | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 7/8 (p 0.0014) |
| SIGN_MATCHED torus | Pos_r4 vs Signed_r4 | 0/8 vs 8/8 (p 0.0002) | 1/8 vs 8/8 (p 0.0014) | 1/8 vs 8/8 (p 0.0014) | 1/8 vs 7/8 (p 0.0101) |
| RING_DIAL p0.5 | Abs_r4 vs Signed_r4 | 2/8 vs 8/8 (p 0.0070) | 2/8 vs 8/8 (p 0.0070) | 4/8 vs 8/8 (p 0.0769) | 2/8 vs 8/8 (p 0.0070) |
| RING_DIAL p0.75 | Abs_r4 vs Signed_r4 | 1/8 vs 8/8 (p 0.0014) | 1/8 vs 8/8 (p 0.0014) | 2/8 vs 8/8 (p 0.0070) | 2/8 vs 8/8 (p 0.0070) |
| RING_DIAL p0.9 | Abs_r4 vs Signed_r4 | 4/8 vs 8/8 (p 0.0769) | 4/8 vs 8/8 (p 0.0769) | 4/8 vs 8/8 (p 0.0769) | 4/8 vs 8/8 (p 0.0769) |
| WRAP_EXISTENCE ring | AO vs SO | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) |
| MECHANISM ring | AO vs SO | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) | 1/8 vs 8/8 (p 0.0014) | 0/8 vs 8/8 (p 0.0002) |
| MECHANISM ring | AO_absmag vs SO_absmag | 0/8 vs 7/8 (p 0.0014) | 0/8 vs 7/8 (p 0.0014) | 0/8 vs 7/8 (p 0.0014) | 0/8 vs 7/8 (p 0.0014) |
| MECHANISM ring | AO_high vs AO_absmag | 8/8 vs 0/8 (p 0.0002) | 8/8 vs 0/8 (p 0.0002) | 8/8 vs 0/8 (p 0.0002) | 8/8 vs 0/8 (p 0.0002) |
| MECHANISM ring | AO_high vs AO_mid | 8/8 vs 1/8 (p 0.0014) | 8/8 vs 2/8 (p 0.0070) | 8/8 vs 3/8 (p 0.0256) | 8/8 vs 1/8 (p 0.0014) |
| MECHANISM ring | AO_high vs SO_high | 8/8 vs 8/8 (p 1.0000) | 8/8 vs 8/8 (p 1.0000) | 8/8 vs 8/8 (p 1.0000) | 8/8 vs 8/8 (p 1.0000) |
| ACCOUNTS T2 ring | SO_absmag_fz vs SO_absmag | 1/8 vs 7/8 (p 0.0101) | 1/8 vs 7/8 (p 0.0101) | 1/8 vs 7/8 (p 0.0101) | 1/8 vs 7/8 (p 0.0101) |
| ACCOUNTS T2 ring | SO_same_fz vs SO_same | 7/8 vs 8/8 (p 1.0000) | 7/8 vs 8/8 (p 1.0000) | 7/8 vs 8/8 (p 1.0000) | 7/8 vs 8/8 (p 1.0000) |
| GENERALITY torus | AO4 vs SO4 | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) |
| GENERALITY ring | CoPEF vs Signed_r4 | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (p 0.0002) | 1/8 vs 8/8 (p 0.0014) | 0/8 vs 8/8 (p 0.0002) |
| TORUS_START torus | AO4 vs SO4 | 0/8 vs 8/8 (p 0.0002) | 1/8 vs 8/8 (p 0.0014) | 1/8 vs 8/8 (p 0.0014) | 1/8 vs 8/8 (p 0.0014) |
| TORUS_START torus | AO4_high vs AO4 | 0/8 vs 0/8 (p 1.0000) | 0/8 vs 1/8 (p 1.0000) | 0/8 vs 1/8 (p 1.0000) | 0/8 vs 1/8 (p 1.0000) |
| ACCOUNTS T4 | CoPEF_line vs Signed_r4_line | 0/16 vs 8/8 (p 0.0000) | 1/16 vs 8/8 (p 0.0000) | 8/16 vs 8/8 (p 0.0222) | 7/16 vs 8/8 (p 0.0095) |
| ACCOUNTS T4 | CoPEF_line vs CoPEF_ring | 0/16 vs 0/16 (p 1.0000) | 1/16 vs 0/16 (p 1.0000) | 8/16 vs 2/16 (p 0.0538) | 7/16 vs 2/16 (p 0.1134) |
