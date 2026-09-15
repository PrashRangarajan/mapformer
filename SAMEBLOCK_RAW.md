# SAMEBLOCK raw report (SAMEBLOCK_PREREG.md; mechanical)

## Every run

| arm | format | seed | code | final loss | exact 30 / 60 / 100 / 150 | per-digit 60 / 100 / 150 |
|---|---|---|---|---|---|---|
| ChoPos_abs | role | 0 | seed-0 code | 0.4202 | 0.000 / 0.000 / 0.000 / 0.000 | 0.096 / 0.098 / 0.099 |
| ChoPos_abs | role | 1 | fast | 0.4387 | 0.000 / 0.000 / 0.000 / 0.000 | 0.106 / 0.097 / 0.098 |
| ChoPos_coupled | role | 0 | seed-0 code | 0.0000 | 1.000 / 0.998 / 0.451 / 0.000 | 1.000 / 0.993 / 0.962 |
| ChoPos_coupled | role | 1 | fast | 0.0000 | 1.000 / 1.000 / 0.801 / 0.043 | 1.000 / 0.998 / 0.977 |
| ChoPos_coupled | role | 2 | fast | 0.0000 | 1.000 / 0.982 / 0.838 / 0.057 | 1.000 / 0.998 / 0.986 |
| ChoPos_coupled | shared | 0 | seed-0 code | 0.0000 | 1.000 / 1.000 / 0.988 / 0.590 | 1.000 / 1.000 / 0.997 |
| ChoPos_nope | role | 0 | seed-0 code | 1.9350 | 0.000 / 0.000 / 0.000 / 0.000 | 0.125 / 0.112 / 0.105 |
| ChoPos_rope | role | 0 | seed-0 code | 1.2389 | 0.000 / 0.000 / 0.000 / 0.000 | 0.096 / 0.066 / 0.050 |
| ChoPos_rope | role | 1 | fast | 1.2433 | 0.000 / 0.000 / 0.000 / 0.000 | 0.100 / 0.094 / 0.067 |
| ChoPos_signed | role | 0 | seed-0 code | 0.0000 | 1.000 / 0.000 / 0.000 / 0.000 | 0.163 / 0.101 / 0.102 |
| ChoPos_signed | role | 1 | fast | 0.0000 | 1.000 / 0.000 / 0.000 / 0.000 | 0.354 / 0.116 / 0.101 |
| ChoPos_signed | role | 2 | fast | 0.0000 | 1.000 / 0.000 / 0.000 / 0.000 | 0.689 / 0.103 / 0.099 |

## Gates

- **G1** coupled (role) mean exact at 100 digits = 0.697 over 3 seeds -> **FAIL**
- **G2** ChoPos_signed (role): s0 pass (1.000), s1 pass (1.000), s2 pass (1.000)
- **G2** ChoPos_abs (role): s0 FAIL (0.000), s1 FAIL (0.000)
- **G2** ChoPos_rope (role): s0 FAIL (0.000), s1 FAIL (0.000)
- **G2** ChoPos_coupled (role): s0 pass (1.000), s1 pass (1.000), s2 pass (1.000)
- **G2** ChoPos_nope (role): s0 FAIL (0.000)

## Contrasts at 100 digits, role format, paired by seed (arm-seeds failing G2 enter as their raw score and are flagged)

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| P1 signed - abs (seeds [0, 1]; G2 fails in seeds [0, 1]) | +0.000 | 0.000 | 0.000 | 0/2 | unmeasured |
| P2 signed - rope (seeds [0, 1]; G2 fails in seeds [0, 1]) | +0.000 | 0.000 | 0.000 | 0/2 | unmeasured |
| signed - coupled (no prediction) (seeds [0, 1, 2]; G2 fails in seeds []) | -0.697 | 0.213 | 0.345 | 0/3 | DETECTABLE |

**G1 failed: by the pre-registration no contrast above is read.**

## Mechanism (descriptive): per-head cosine of mean role increments (s vs a, s vs b, a vs b)

- ChoPos_abs role s0: h0 (+0.61, +0.60, +0.74); h1 (+0.56, +0.60, +0.61); h2 (+0.62, +0.51, +0.58); h3 (+0.65, +0.42, +0.45)
- ChoPos_abs role s1: h0 (+0.66, +0.53, +0.57); h1 (+0.64, +0.53, +0.65); h2 (+0.54, +0.57, +0.66); h3 (+0.66, +0.44, +0.72)
- ChoPos_signed role s0: h0 (-0.76, -0.38, -0.30); h1 (-0.83, -0.51, -0.06); h2 (-0.89, -0.47, +0.03); h3 (-0.80, -0.72, +0.17)
- ChoPos_signed role s1: h0 (-0.74, -0.73, +0.08); h1 (-0.48, -0.60, -0.41); h2 (-0.63, -0.70, -0.11); h3 (-0.72, -0.71, +0.03)
- ChoPos_signed role s2: h0 (-0.85, -0.73, +0.25); h1 (-0.88, -0.79, +0.40); h2 (-0.92, -0.61, +0.25); h3 (-0.92, -0.59, +0.23)
