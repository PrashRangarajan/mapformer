## S1-a  H-wrap: do solved large-k cells carry a per-token (wrapped) rewind?

diff = max_h sel_h - max_h sel0_h (registered primary); alt = max_h (sel_h - sel0_h)

| arm | cells k>=8 | solved (acc>=0.9) | solved with diff>=0.5 | failed (acc<=0.3) | failed with diff>=0.5 | r(acc, diff) | solved: mean att | solved: gain>0 |
|---|---|---|---|---|---|---|---|---|
| VanillaEM_P0_r4 | 1368 | 595 | 0.459 | 433 | 0.000 | +0.460 | 0.776 | 0.434 |
| EMDoF_alignlock | 1368 | 595 | 0.395 | 496 | 0.000 | +0.440 | 0.786 | 0.377 |
| EMDoF_magonly | 1368 | 600 | 0.425 | 456 | 0.000 | +0.455 | 0.773 | 0.403 |
| EMDoF_alignfree | 1368 | 1009 | 0.445 | 283 | 0.000 | +0.382 | 0.851 | 0.451 |
| VanillaEM_r4 | 1368 | 958 | 0.522 | 274 | 0.000 | +0.422 | 0.849 | 0.542 |

H-wrap (P0): solved 0.459 (>= 0.70 needed), failed 0.000 (<= 0.20 needed) -> **PARTIAL**
alt statistic, P0: solved 0.459, failed 0.000

## S1-b  H-phase: AlignFree - MagOnly by k bin (paired by seed, n=24)

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| k 1-16 | +0.124 | 0.147 | 0.084 | 21/24 | DETECTABLE |
| k 17-32 | +0.263 | 0.192 | 0.110 | 22/24 | DETECTABLE |
| k 33-48 | +0.077 | 0.232 | 0.133 | 15/24 | unmeasured |
| k 49-64 | +0.130 | 0.247 | 0.141 | 17/24 | unmeasured |

Rewind-route fraction of solved cells: AlignFree 0.445, MagOnly 0.425, difference +0.020 (<= -0.15 predicted) -> **REFUTED**

## Kernel-peak readout (Delta(q) = 0): argmax over n in 0..80 of the mean A_P profile, per head

| arm | heads | peak at n in {0,1} | peak at n >= 2 | peaks (n) |
|---|---|---|---|---|
| VanillaEM_P0_r4 | 48 | 1.000 | 0.000 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| EMDoF_alignlock | 48 | 1.000 | 0.000 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| EMDoF_magonly | 48 | 1.000 | 0.000 | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| EMDoF_alignfree | 48 | 0.000 | 1.000 | [2, 2, 2, 3, 3, 3, 3, 3, 5, 6, 7, 9, 10, 10, 10, 11, 11, 11, 11, 12, 13, 14, 16, 16, 16, 19, 19, 19, 20, 22, 22, 23, 25, 25, 26, 27, 30, 30, 33, 33, 39, 39, 42, 43, 46, 57, 57, 63] |
| VanillaEM_r4 | 48 | 0.000 | 1.000 | [4, 5, 6, 6, 9, 10, 11, 12, 13, 14, 14, 16, 19, 20, 22, 24, 25, 25, 26, 29, 30, 31, 31, 32, 32, 32, 32, 35, 37, 39, 44, 47, 48, 48, 49, 49, 49, 51, 52, 56, 57, 60, 60, 60, 62, 62, 73, 78] |

Manipulation check (rho=1 arms all peak at n <= 1): **PASS**
