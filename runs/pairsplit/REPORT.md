# PAIRSPLIT report (PAIRSPLIT_PREREG.md)

## Per-arm

| arm | n | acc T=1024 | final loss |
|---|---|---|---|
| EMPair_r4 | 48 | **0.897 +/- 0.107** | 0.358 |
| EMPairConst_r4 | 48 | **0.807 +/- 0.131** | 0.765 |
| VanillaEM_P0_r4 | 24 | **0.687 +/- 0.127** | 1.143 |

## Contrasts

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| EMPair - EMPairConst (C1: freedom) | +0.091 | 0.164 | 0.066 | 34/48 | DETECTABLE |
| EMPairConst - P0 (C2: pathway), n=24 | +0.100 | 0.197 | 0.113 | 14/24 | unmeasured |
| EMPair - P0, n=24 | +0.191 | 0.175 | 0.100 | 20/24 | DETECTABLE |

## Fresh-seed split on C1 (the registered guard)

replication guard: EMPair - EMPairConst (C1: freedom)
| seeds | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| first 8 | +0.098 | 0.211 | 0.209 | 7/8 | unmeasured |
| fresh (40) | +0.089 | 0.156 | 0.069 | 27/40 | DETECTABLE |
| pooled (48) | +0.091 | 0.164 | 0.066 | 34/48 | DETECTABLE |
  replicates: same sign on fresh seeds and detectable on them alone.

## Rule 9

rule 9: r(final loss, acc) = -0.954 over 96 runs; acc = 1.036 -0.328*loss, resid sd 0.038

## Registered verdicts

- **S1 (freedom is real but small: detectable, 0.05-0.15)**: +0.091 (MDE 0.066) -> **CONFIRMED**
- **S2 (freedom buys no accuracy)**: -> **not confirmed**
- **S3 (the n=8 estimate was low: > +0.15)**: -> **not confirmed**
- **C2 at n=24**: +0.100 (MDE 0.113) -> **unmeasured**

## Determinism re-check

```
compare  A=/home/prashr/mapformer/runs/pairsplit/EMPair_r4_s0/EMPair_r4_recency.pt
         B=/home/prashr/mapformer/runs/pairorigin/EMPair_r4_s0/EMPair_r4_recency.pt
  tensors compared 29, differing 0, only-in-A 0, only-in-B 0
  loss curves bitwise equal
  => DETERMINISM: bitwise identical. This is the same computation run twice. It licenses pairing with / reusing the stored arm; it is NOT a replication and says nothing about effect-size stability (rule 27).
```
