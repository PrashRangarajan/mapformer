# Indirect Indexing (PoPE paper sec 5.1) with path integration -- `/home/prashr/mapformer/runs/indirect`

Pre-registration: `INDIRECT_PREREG.md`. Final-token accuracy on a 10k test split.

## Arms

| arm | position | encoding | test accuracy | paper | seeds | final loss | slope /1k |
|---|---|---|---|---|---|---|---|
| RoPE | index | RoPE | 0.066 +/- 0.021 | 0.112 +/- 0.025 | 3 | 3.188 | -0.0020 |
| PoPE | index | PoPE | 0.344 +/- 0.398 | 0.948 +/- 0.029 | 3 | 1.918 | -0.0141 |
| MapWM_r2 | path integration | RoPE | 0.091 +/- 0.001 | -- | 3 | 3.016 | -0.0023 |
| MapPoPE_r2 | path integration | PoPE | 0.408 +/- 0.498 | -- | 3 | 1.857 | -0.0020 |

## Floors (strategies needing no pointer arithmetic)

- uniform over letters: 0.019
- copy the source character: 0.045
- a random letter of the string: 0.035
- a neighbour of the source character: 0.044

## Registered contrasts (paired by seed, MDE = 2.8 sd / sqrt(n))

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| R2 path integration on the RoPE row: MapWM_r2 - RoPE | +0.025 | 0.021 | 0.034 | 3/3 | unmeasured |
| R3 path integration on the PoPE row: MapPoPE_r2 - PoPE | +0.063 | 0.750 | 1.213 | 1/3 | unmeasured |
| the paper's own contrast: PoPE - RoPE | +0.278 | 0.377 | 0.610 | 3/3 | unmeasured |
| encoding, on the path-integrated row: MapPoPE_r2 - MapWM_r2 | +0.317 | 0.498 | 0.806 | 3/3 | unmeasured |
| R4 interaction | +0.039 | 0.738 | 1.193 | 1/3 | unmeasured |

## Registered verdicts

- **R1 replication**: RoPE 0.066 (needs < 0.30), PoPE 0.344 (needs > 0.80) -> **DOES NOT REPLICATE**

## Validation curves (accuracy every 5,000 steps, seed 0)

