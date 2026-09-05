# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 60 trajectories, 5 seeds, one batch.

## D = 2, grid 32 (1024 cells, 4 actions, vocab 21)

| arm | final loss | T=128 | T=512 |
|---|---|---|---|
| `Vanilla` | 0.0125 | 0.998 ± 0.002 (n=5) | 0.854 ± 0.027 (n=5) |
| `Vanilla_r4` | 0.0001 | 1.000 ± 0.000 (n=5) | 0.971 ± 0.016 (n=5) |

Paired against `Vanilla`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r4` | 128 | +0.002 | 0.002 | 0.003 | 3/5 | unmeasured |
| `Vanilla_r4` | 512 | +0.117 | 0.019 | 0.024 | 5/5 | DETECTABLE |

- r(final loss, accuracy) at T=128: **-0.866** over 10 runs
- r(final loss, accuracy) at T=512: **-0.676** over 10 runs

## D = 3, grid 10 (1000 cells, 6 actions, vocab 23)

| arm | final loss | T=128 | T=512 |
|---|---|---|---|
| `Vanilla` | 0.2994 | 0.924 ± 0.014 (n=5) | 0.729 ± 0.074 (n=5) |
| `Vanilla_r3` | 0.1590 | 0.961 ± 0.015 (n=5) | 0.824 ± 0.053 (n=5) |
| `Vanilla_r5` | 0.0608 | 0.973 ± 0.027 (n=5) | 0.879 ± 0.057 (n=5) |

Paired against `Vanilla`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r3` | 128 | +0.037 | 0.028 | 0.035 | 5/5 | DETECTABLE |
| `Vanilla_r3` | 512 | +0.096 | 0.048 | 0.060 | 5/5 | DETECTABLE |
| `Vanilla_r5` | 128 | +0.049 | 0.030 | 0.038 | 5/5 | DETECTABLE |
| `Vanilla_r5` | 512 | +0.151 | 0.047 | 0.059 | 5/5 | DETECTABLE |

- r(final loss, accuracy) at T=128: **-0.880** over 15 runs
- r(final loss, accuracy) at T=512: **-0.824** over 15 runs

## D = 5, grid 4 (1024 cells, 10 actions, vocab 27)

| arm | final loss | T=128 | T=512 |
|---|---|---|---|
| `Vanilla` | 0.0941 | 0.977 ± 0.004 (n=5) | 0.872 ± 0.089 (n=5) |
| `Vanilla_r5` | 0.0486 | 0.987 ± 0.002 (n=5) | 0.868 ± 0.073 (n=5) |
| `Vanilla_r7` | 0.0443 | 0.989 ± 0.003 (n=5) | 0.938 ± 0.057 (n=5) |

Paired against `Vanilla`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r5` | 128 | +0.010 | 0.002 | 0.002 | 5/5 | DETECTABLE |
| `Vanilla_r5` | 512 | -0.004 | 0.117 | 0.146 | 2/5 | unmeasured |
| `Vanilla_r7` | 128 | +0.012 | 0.002 | 0.003 | 5/5 | DETECTABLE |
| `Vanilla_r7` | 512 | +0.066 | 0.076 | 0.096 | 4/5 | unmeasured |

- r(final loss, accuracy) at T=128: **-0.891** over 15 runs
- r(final loss, accuracy) at T=512: **-0.198** over 15 runs

