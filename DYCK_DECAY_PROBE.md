# Dyck-2: metrics the always-valid opens cannot inflate

Eval-only on the checkpoints of `/home/prashr/mapformer/runs/dyck_decay`; 512 sequences per cell, mean over seeds. Closer accuracy: P(legal closer) > P(illegal closer) at positions with depth > 0, chance 0.500. strict-F1: the paper's F1 with a min over Val(s) in place of the mean. Distance = how far back the open bracket on top of the stack is.

## L = 32, depth = 4

Share of depth>0 positions by distance: 1-2 0.74, 3-8 0.17, 9-32 0.09, 33+ 0.00

| predictor | paper F1 | strict-F1 | closer acc | d 1-2 | d 3-8 | d 9-32 | d 33+ | acc d>=9 | after a close |
|---|---|---|---|---|---|---|---|
| n-gram k=1 | 0.807 | 0.683 | 0.797 | 0.898 | 0.510 | 0.489 | -- | 0.489 | 0.500 |
| n-gram k=3 | 0.850 | 0.773 | 0.874 | 1.000 | 0.511 | 0.510 | -- | 0.510 | 0.691 |
| MapPoPE_decay-1L_r2 | 0.994 | 0.987 | 0.999 | 1.000 | 0.999 | 0.997 | -- | 0.997 | 0.998 |
| PoPE_decay-1L | 0.866 | 0.822 | 0.905 | 0.993 | 0.727 | 0.509 | -- | 0.509 | 0.767 |

## L = 128, depth = 12

Share of depth>0 positions by distance: 1-2 0.66, 3-8 0.14, 9-32 0.13, 33+ 0.06

| predictor | paper F1 | strict-F1 | closer acc | d 1-2 | d 3-8 | d 9-32 | d 33+ | acc d>=9 | after a close |
|---|---|---|---|---|---|---|---|
| n-gram k=1 | 0.857 | 0.701 | 0.770 | 0.904 | 0.498 | 0.508 | 0.511 | 0.509 | 0.504 |
| n-gram k=3 | 0.886 | 0.772 | 0.832 | 1.000 | 0.495 | 0.502 | 0.500 | 0.501 | 0.638 |
| MapPoPE_decay-1L_r2 | 0.956 | 0.926 | 0.980 | 0.999 | 0.994 | 0.968 | 0.778 | 0.906 | 0.958 |
| PoPE_decay-1L | 0.879 | 0.797 | 0.853 | 0.987 | 0.700 | 0.499 | 0.511 | 0.503 | 0.683 |

