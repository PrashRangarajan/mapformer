# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 2 seeds, one batch.

## D = 2, grid 32 (1024 cells, 4 actions, vocab 21)

| arm | final loss | T=1024 |
|---|---|---|
| `Vanilla_r2ph_om32` | 1.2963 | 0.632 ± 0.024 (n=2) |
| `Vanilla_r3ph_om32` | 1.2301 | 0.642 ± 0.069 (n=2) |

Paired against `Vanilla_r2ph_om32`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r3ph_om32` | 1024 | +0.010 | 0.045 | 0.089 | 1/2 | unmeasured |

- r(final loss, accuracy) at T=1024: **-0.979** over 4 runs

