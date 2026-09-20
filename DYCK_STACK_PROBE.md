# Dyck-2: metrics the always-valid opens cannot inflate

Eval-only on the checkpoints of `/home/prashr/mapformer/runs/dyck_bs128`; 512 sequences per cell, mean over seeds. Closer accuracy: P(legal closer) > P(illegal closer) at positions with depth > 0, chance 0.500. strict-F1: the paper's F1 with a min over Val(s) in place of the mean. Distance = how far back the open bracket on top of the stack is.

## L = 32, depth = 4

Share of depth>0 positions by distance: 0-2 0.74, 3-8 0.17, 9-32 0.09, 33+ 0.00

| predictor | paper F1 | strict-F1 | closer acc | d 0-2 | d 3-8 | d 9-32 | d 33+ | acc d>=9 | after a close |
|---|---|---|---|---|---|---|---|
| n-gram k=1 | 0.807 | 0.683 | 0.797 | 0.898 | 0.510 | 0.489 | -- | 0.489 | 0.500 |
| n-gram k=3 | 0.850 | 0.773 | 0.874 | 1.000 | 0.511 | 0.510 | -- | 0.510 | 0.691 |
| MapPoPE-1L_r2 | 0.987 | 0.974 | 1.000 | 1.000 | 0.999 | 0.999 | -- | 0.999 | 0.999 |
| MapWM-1L_r2 | 0.985 | 0.970 | 0.999 | 1.000 | 0.999 | 0.998 | -- | 0.998 | 0.998 |
| MapEM-1L_r2 | 0.985 | 0.973 | 0.997 | 0.998 | 0.992 | 0.993 | -- | 0.993 | 0.992 |
| PoPE-1L | 0.912 | 0.869 | 0.940 | 0.978 | 0.860 | 0.779 | -- | 0.779 | 0.853 |
| RoPE-1L | 0.912 | 0.867 | 0.936 | 0.977 | 0.852 | 0.746 | -- | 0.746 | 0.842 |
| RoPE-2L | 0.947 | 0.917 | 0.979 | 1.000 | 0.915 | 0.928 | -- | 0.928 | 0.949 |

## L = 128, depth = 12

Share of depth>0 positions by distance: 0-2 0.66, 3-8 0.14, 9-32 0.13, 33+ 0.06

| predictor | paper F1 | strict-F1 | closer acc | d 0-2 | d 3-8 | d 9-32 | d 33+ | acc d>=9 | after a close |
|---|---|---|---|---|---|---|---|
| n-gram k=1 | 0.857 | 0.701 | 0.770 | 0.904 | 0.498 | 0.508 | 0.511 | 0.509 | 0.504 |
| n-gram k=3 | 0.886 | 0.772 | 0.832 | 1.000 | 0.495 | 0.502 | 0.500 | 0.501 | 0.638 |
| MapPoPE-1L_r2 | 0.926 | 0.871 | 0.978 | 0.999 | 0.996 | 0.971 | 0.730 | 0.892 | 0.953 |
| MapWM-1L_r2 | 0.868 | 0.808 | 0.928 | 0.988 | 0.906 | 0.799 | 0.611 | 0.738 | 0.846 |
| MapEM-1L_r2 | 0.888 | 0.855 | 0.938 | 0.991 | 0.933 | 0.851 | 0.576 | 0.761 | 0.867 |
| PoPE-1L | 0.616 | 0.414 | 0.864 | 0.961 | 0.740 | 0.645 | 0.569 | 0.620 | 0.707 |
| RoPE-1L | 0.567 | 0.352 | 0.822 | 0.937 | 0.641 | 0.570 | 0.539 | 0.560 | 0.622 |
| RoPE-2L | 0.497 | 0.206 | 0.856 | 0.967 | 0.683 | 0.602 | 0.592 | 0.599 | 0.690 |

