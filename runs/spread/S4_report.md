# SPREAD report (SPREAD_PREREG.md)

## Per-arm

| arm | queries per token | primary (k in {4,16,64}) | k=4 | k=16 | k=64 | whole trained set | rewind fraction | final loss |
|---|---|---|---|---|---|---|---|---|
| m4_e300 | 403,000 | **1.000 +/- 0.000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.562 | 0.048 |
| m16_e300 | 101,000 | **0.996 +/- 0.010** | 1.000 | 0.989 | 0.999 | 0.978 | 0.898 | 0.245 |
| m16_e1200 | 403,000 | **1.000 +/- 0.000** | 1.000 | 1.000 | 1.000 | 1.000 | 0.898 | 0.071 |
| m64_e1200 | 101,000 | **0.928 +/- 0.131** | 1.000 | 0.890 | 0.893 | 0.930 | 0.857 | 0.359 |
| m64_e300 | 25,000 | **0.578 +/- 0.123** | 0.887 | 0.507 | 0.340 | 0.609 | 0.547 | 1.340 |

## Contrasts (primary readout, paired by seed, n=8)

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| m16_e1200 - m4_e300 (exposure-matched) | +0.000 | 0.000 | 0.000 | 0/8 | unmeasured |
| m64_e1200 - m16_e300 (exposure-matched) | -0.068 | 0.133 | 0.132 | 1/8 | unmeasured |
| m4_e300 - m16_e300 (fixed budget) | +0.004 | 0.010 | 0.010 | 2/8 | unmeasured |
| m16_e300 - m64_e300 (fixed budget) | +0.418 | 0.125 | 0.124 | 8/8 | DETECTABLE |
| m4_e300 - m64_e300 (fixed budget) | +0.422 | 0.123 | 0.122 | 8/8 | DETECTABLE |

## Registered verdicts

- **S4-P1 (exposure)**: matched pairs +0.000 (MDE 0.000) and -0.068 (MDE 0.132) -> **CONFIRMED**
- **S4-P2 (spread gradient)**: ordering m4 1.000 > m16 0.996 > m64 0.578: holds; m4 - m64 +0.422 (DETECTABLE) -> **MET**
- **S4-P3 (mechanism)**: r(rewind fraction, primary) = +0.573 over 40 arm-seed cells (>= 0.70 predicted) -> **NOT MET**

## Determinism re-check (licenses reuse of stored m64_e300)

```
compare  A=/home/prashr/mapformer/runs/spread/repro_s0/VanillaEM_P0_r4_recency.pt
         B=/home/prashr/mapformer/runs/dof/recency/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt
  tensors compared 25, differing 0, only-in-A 0, only-in-B 0
  loss curves bitwise equal
  => DETERMINISM: bitwise identical. This is the same computation run twice. It licenses pairing with / reusing the stored arm; it is NOT a replication and says nothing about effect-size stability (rule 27).
```
