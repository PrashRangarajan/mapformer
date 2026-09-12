# SPREAD2 report (SPREAD2_PREREG.md)

## Per-arm

| arm | queries/token | primary (k in {4,16,64}) | whole trained set | final loss | at ceiling? |
|---|---|---|---|---|---|
| m4_e60 | 80,600 | **0.913 +/- 0.110** | 0.933 | 0.306 | no |
| m16_e240 | 80,600 | **0.957 +/- 0.098** | 0.964 | 0.295 | no |
| m16_e60 | 20,200 | **0.590 +/- 0.190** | 0.669 | 1.161 | no |
| m64_e60 | 5,000 | **0.248 +/- 0.175** | 0.187 | 2.502 | no |

## Design check (read FIRST)

m4_e60 must land in 0.60-0.95: measured 0.913 -> **PASS -- the exposure contrast is interpretable**

## Contrasts (paired by seed, n=8)

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| m16_e240 - m4_e60 (exposure-matched) | +0.045 | 0.168 | 0.166 | 5/8 | unmeasured |
| m4_e60 - m16_e60 (fixed budget) | +0.323 | 0.244 | 0.241 | 7/8 | DETECTABLE |
| m16_e60 - m64_e60 (fixed budget) | +0.342 | 0.252 | 0.249 | 7/8 | DETECTABLE |
| m4_e60 - m64_e60 (fixed budget) | +0.665 | 0.215 | 0.213 | 8/8 | DETECTABLE |

## Rule 9

rule 9: r(final loss, acc) = -0.936 over 32 runs; acc = 1.014 -0.316*loss, resid sd 0.114

## Registered verdicts

- **S5-P1 (exposure is the currency)**: +0.045 (MDE 0.166) -> **CONFIRMED**
- **S5-P2 (token count costs beyond exposure)**: -> **not confirmed**
- **S5-P3 (gradient replicates at a lower budget)**: ordering 0.913 > 0.590 > 0.248: holds; m4 - m64 +0.665 (DETECTABLE) -> **MET**
