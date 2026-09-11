# S3 report (SEARCH_PREREG.md)

## Per-arm

| arm | acc T=1024 mean | >= 0.9 | >= 0.95 | acc T=2048 | final loss | epochs to loss<0.5 (median, reached) |
|---|---|---|---|---|---|---|
| P0_fix64 | 0.985 +/- 0.043 | 7/8 | 7/8 | 0.946 | 0.065 | 54.0 (8/8) |
| P0_fix16 | 0.994 +/- 0.018 | 8/8 | 7/8 | 0.966 | 0.015 | 25.0 (8/8) |
| WM_fix64 | 0.947 +/- 0.120 | 7/8 | 6/8 | 0.755 | 0.119 | 99.0 (7/8) |
| P0_cur | 0.727 +/- 0.038 | 0/8 | 0/8 | 0.668 | 1.045 | 11.0 (8/8) |

## Fixed-k mechanism readouts (single query token)

Registered: sel, sel - sel0, linear ratio. Added after S1 (exploratory): trough selmin - selmin0, wrapped score, route.

| arm | seed | acc | sel_max | sel0_max | sel - sel0 | selmin - selmin0 | route | wrapped score | linear ratio (target) |
|---|---|---|---|---|---|---|---|---|---|
| P0_fix16 | 0 | 1.000 | 1.000 | 0.000 | +1.000 | +0.000 | peak-rewind | +0.898 | -2.77 (-15) |
| P0_fix16 | 1 | 1.000 | 1.000 | 0.000 | +1.000 | +0.000 | peak-rewind | +0.981 | -5.43 (-15) |
| P0_fix16 | 2 | 1.000 | 1.000 | 0.000 | +1.000 | +0.000 | peak-rewind | +0.972 | -7.25 (-15) |
| P0_fix16 | 3 | 1.000 | 1.000 | 0.000 | +1.000 | +0.000 | peak-rewind | +0.970 | -4.95 (-15) |
| P0_fix16 | 4 | 1.000 | 1.000 | 0.000 | +1.000 | +0.000 | peak-rewind | +0.982 | -9.69 (-15) |
| P0_fix16 | 5 | 1.000 | 1.000 | 0.000 | +1.000 | +0.000 | peak-rewind | +0.998 | -15.04 (-15) |
| P0_fix16 | 6 | 1.000 | 1.000 | 0.000 | +1.000 | +0.000 | peak-rewind | +0.871 | +0.97 (-15) |
| P0_fix16 | 7 | 0.950 | 1.000 | 0.000 | +1.000 | +0.000 | peak-rewind | +0.997 | -14.80 (-15) |
| P0_fix64 | 0 | 1.000 | 0.000 | 0.000 | +0.000 | +0.000 | none | -0.368 | +0.32 (-63) |
| P0_fix64 | 1 | 0.879 | 0.000 | 0.000 | +0.000 | +0.642 | trough-rewind | -0.841 | +0.84 (-63) |
| P0_fix64 | 2 | 1.000 | 1.000 | 0.000 | +1.000 | +0.691 | peak-rewind | -0.135 | +0.94 (-63) |
| P0_fix64 | 3 | 1.000 | 1.000 | 0.000 | +1.000 | +1.000 | peak-rewind | +0.356 | +1.25 (-63) |
| P0_fix64 | 4 | 1.000 | 1.000 | 0.000 | +1.000 | +0.999 | peak-rewind | +0.285 | +1.07 (-63) |
| P0_fix64 | 5 | 1.000 | 1.000 | 0.000 | +1.000 | +1.000 | peak-rewind | +0.224 | +1.42 (-63) |
| P0_fix64 | 6 | 1.000 | 1.000 | 0.000 | +1.000 | +0.000 | peak-rewind | +0.862 | +1.27 (-63) |
| P0_fix64 | 7 | 1.000 | 1.000 | 0.000 | +1.000 | +1.000 | peak-rewind | +0.410 | +1.16 (-63) |

## Registered verdicts

- **S3-P1** WM_fix64 >= 0.95 on 6/8 (>= 6 needed): **MET**
- **S3-P2** P0_fix16 >= 0.9 on 8/8 (>= 6), P0_fix64 >= 0.9 on 7/8 (<= 2): **REFUTED**

## Determinism re-check (licenses reuse of stored P0)

```
compare  A=/home/prashr/mapformer/runs/search/P0_repro_s0/VanillaEM_P0_r4_recency.pt
         B=/home/prashr/mapformer/runs/dof/recency/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt
  tensors compared 25, differing 0, only-in-A 0, only-in-B 0
  loss curves bitwise equal
  => DETERMINISM: bitwise identical. This is the same computation run twice. It licenses pairing with / reusing the stored arm; it is NOT a replication and says nothing about effect-size stability (rule 27).
```

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| P0_cur - P0 (stored) | +0.127 | 0.117 | 0.116 | 7/8 | DETECTABLE |
| solved cells k>=8 (of 57): P0_cur - P0 (exploratory) | +10.875 | 9.296 | 9.202 | 7/8 | DETECTABLE |

P0_cur linear slopes: [-0.006, 0.009, 0.003, 0.004, -0.009, 0.0, -0.015, 0.002]
- **S3-P3** (curriculum does NOT close the gap): **MET**
