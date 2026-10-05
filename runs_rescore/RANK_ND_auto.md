# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 8 seeds, one batch.

## D = 2, grid 32 (1024 cells, 4 actions, vocab 21)

| arm | final loss | T=1024 |
|---|---|---|
| `Vanilla_r2ph` | 0.3190 | 0.766 ± 0.122 (n=8) |
| `Vanilla_r3ph` | 0.0047 | 0.999 ± 0.001 (n=8) |

Paired against `Vanilla_r2ph`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r3ph` | 1024 | +0.233 | 0.121 | 0.120 | 7/8 | DETECTABLE |

- r(final loss, accuracy) at T=1024: **-0.965** over 16 runs

## D = 3, grid 10 (1000 cells, 6 actions, vocab 23)

| arm | final loss | T=1024 |
|---|---|---|
| `Vanilla_r3ph` | 0.3390 | 0.799 ± 0.092 (n=8) |
| `Vanilla_r4ph` | 0.2578 | 0.848 ± 0.175 (n=8) |

Paired against `Vanilla_r3ph`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r4ph` | 1024 | +0.049 | 0.247 | 0.244 | 5/8 | unmeasured |

- r(final loss, accuracy) at T=1024: **-0.958** over 16 runs

