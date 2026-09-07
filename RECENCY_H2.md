# H2 -- does the unconstrained arm learn a different accumulator per task?

`Signed_r4` is unconstrained and CONTAINS the monotone solution, so it can adopt whichever code a task rewards. The clock/map dichotomy predicts it adopts a cancelling code on the torus (a map, alpha ~ 0.5) and a one-signed counter on recency (a clock, alpha ~ 1.0). If the same accumulator appears on both, alpha is descriptive only.

| arm | task | n | alpha | negative fraction of Delta |
|---|---|---|---|---|
| `Signed_r4` | torus | 12 | 0.591 +/- 0.028 | 0.498 +/- 0.033 |
| `Signed_r4` | recency | 8 | 0.967 +/- 0.009 | 0.478 +/- 0.045 |
| `Abs_r4` | torus | 12 | 1.010 +/- 0.022 | 0.000 +/- 0.000 |
| `Abs_r4` | recency | 8 | 0.976 +/- 0.007 | 0.000 +/- 0.000 |
| `Pos_r4` | torus | 12 | 1.005 +/- 0.006 | 0.000 +/- 0.000 |
| `Pos_r4` | recency | 8 | 1.005 +/- 0.004 | 0.000 +/- 0.000 |
| `CARoPE_r4` | torus | 12 | 1.003 +/- 0.002 | 0.000 +/- 0.000 |
| `CARoPE_r4` | recency | 8 | 0.992 +/- 0.003 | 0.000 +/- 0.000 |

**`Signed_r4`: alpha 0.591 (torus) -> 0.967 (recency), delta +0.376, se 0.009; negative fraction 0.498 -> 0.478.**

**`Abs_r4`: alpha 1.010 (torus) -> 0.976 (recency), delta -0.033, se 0.007; negative fraction 0.000 -> 0.000.**

**`Pos_r4`: alpha 1.005 (torus) -> 1.005 (recency), delta +0.000, se 0.002; negative fraction 0.000 -> 0.000.**

**`CARoPE_r4`: alpha 1.003 (torus) -> 0.992 (recency), delta -0.011, se 0.001; negative fraction 0.000 -> 0.000.**
