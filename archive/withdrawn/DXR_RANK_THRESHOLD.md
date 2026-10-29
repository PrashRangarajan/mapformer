# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 8 seeds, one batch.

## D = 2, grid 32 (1024 cells, 4 actions, vocab 21)

| arm | final loss | T=128 | T=512 |
|---|---|---|---|
| `Vanilla` | 0.0587 | 0.985 ± 0.034 (n=8) | 0.848 ± 0.056 (n=8) |
| `Vanilla_r4` | 0.0001 | 1.000 ± 0.000 (n=8) | 0.959 ± 0.036 (n=8) |

Paired against `Vanilla`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r4` | 128 | +0.015 | 0.034 | 0.033 | 6/8 | unmeasured |
| `Vanilla_r4` | 512 | +0.110 | 0.077 | 0.076 | 7/8 | DETECTABLE |

- r(final loss, accuracy) at T=128: **-1.000** over 16 runs  — |r| > 0.98, the held-out eval carries no information the loss does not
- r(final loss, accuracy) at T=512: **-0.697** over 16 runs

## D = 3, grid 10 (1000 cells, 6 actions, vocab 23)

| arm | final loss | T=128 | T=512 |
|---|---|---|---|
| `Vanilla` | 0.3398 | 0.907 ± 0.028 (n=8) | 0.707 ± 0.077 (n=8) |
| `Vanilla_r3` | 0.1644 | 0.953 ± 0.028 (n=8) | 0.793 ± 0.092 (n=8) |
| `Vanilla_r5` | 0.0940 | 0.967 ± 0.022 (n=8) | 0.860 ± 0.055 (n=8) |

Paired against `Vanilla`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r3` | 128 | +0.046 | 0.025 | 0.025 | 8/8 | DETECTABLE |
| `Vanilla_r3` | 512 | +0.086 | 0.089 | 0.088 | 7/8 | unmeasured |
| `Vanilla_r5` | 128 | +0.060 | 0.031 | 0.030 | 8/8 | DETECTABLE |
| `Vanilla_r5` | 512 | +0.153 | 0.049 | 0.048 | 8/8 | DETECTABLE |

- r(final loss, accuracy) at T=128: **-0.937** over 24 runs
- r(final loss, accuracy) at T=512: **-0.752** over 24 runs

## D = 5, grid 4 (1024 cells, 10 actions, vocab 27)

| arm | final loss | T=128 | T=512 |
|---|---|---|---|
| `Vanilla` | 0.0908 | 0.978 ± 0.004 (n=8) | 0.896 ± 0.076 (n=8) |
| `Vanilla_r5` | 0.0482 | 0.988 ± 0.002 (n=8) | 0.877 ± 0.072 (n=8) |
| `Vanilla_r7` | 0.0424 | 0.990 ± 0.002 (n=8) | 0.950 ± 0.045 (n=8) |

Paired against `Vanilla`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r5` | 128 | +0.009 | 0.002 | 0.002 | 8/8 | DETECTABLE |
| `Vanilla_r5` | 512 | -0.019 | 0.100 | 0.099 | 3/8 | unmeasured |
| `Vanilla_r7` | 128 | +0.011 | 0.002 | 0.002 | 8/8 | DETECTABLE |
| `Vanilla_r7` | 512 | +0.055 | 0.063 | 0.062 | 7/8 | unmeasured |

- r(final loss, accuracy) at T=128: **-0.951** over 24 runs
- r(final loss, accuracy) at T=512: **-0.182** over 24 runs

