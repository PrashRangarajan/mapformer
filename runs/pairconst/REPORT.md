# PAIRCONST report (PAIRCONST_PREREG.md)

## Per-arm

| arm | acc T=1024 | acc T=2048 | final loss |
|---|---|---|---|
| EMPairConst_r4 | **0.782 +/- 0.159** | 0.659 | 0.755 |
| EMPair_r4 | **0.880 +/- 0.136** | 0.784 | 0.422 |
| VanillaEM_P0_r4 | **0.600 +/- 0.126** | 0.510 | 1.340 |

## Contrasts (paired by seed, n=8)

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| EMPair - EMPairConst (C1: freedom) | +0.098 | 0.211 | 0.209 | 7/8 | unmeasured |
| EMPairConst - P0 (C2/C3: parameters) | +0.182 | 0.190 | 0.188 | 6/8 | unmeasured |
| EMPair - P0 (the effect under test) | +0.280 | 0.218 | 0.216 | 7/8 | DETECTABLE |

## Rule 9

rule 9: r(final loss, acc) = -0.977 over 24 runs; acc = 1.051 -0.354*loss, resid sd 0.038

## Registered verdicts

- **C1 (per-pair freedom does the work)**: +0.098 (MDE 0.209; needs >= +0.20 AND detectable) -> **NOT CONFIRMED**
- **C2 (the parameters buy nothing)**: EMPairConst - P0 = +0.182 (MDE 0.188) -> **MET**
- **C3 (it was capacity)**: -> **not confirmed**
- C1 and C3 both fail: unresolved; report both contrasts with their MDEs.

## Manipulation checks

```
compare  A=/home/prashr/mapformer/runs/pairconst/EMPair_r4_s0/EMPair_r4_recency.pt
         B=/home/prashr/mapformer/runs/pairorigin/EMPair_r4_s0/EMPair_r4_recency.pt
  tensors compared 29, differing 0, only-in-A 0, only-in-B 0
  loss curves bitwise equal
  => DETERMINISM: bitwise identical. This is the same computation run twice. It licenses pairing with / reusing the stored arm; it is NOT a replication and says nothing about effect-size stability (rule 27).
origin spread across tokens: EMPairConst 0.000e+00 (must be 0), EMPair 1.166e-01 (must be > 0) -> PASS
```
