# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 2 seeds, one batch.

## D = 2, grid 32 (1024 cells, 4 actions, vocab 21)

| arm | final loss | T=1024 |
|---|---|---|
| `Vanilla_r2ph_om32` | 1.2963 | 0.632 ± 0.024 (n=2) |
| `Vanilla_r3ph_om32` | 1.2301 | 0.642 ± 0.069 (n=2) |
| `Vanilla_r2ph_om32_redraw` | 2.1020 | 0.517 ± 0.002 (n=2) |

Paired against `Vanilla_r2ph_om32`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r3ph_om32` | 1024 | +0.010 | 0.045 | 0.089 | 1/2 | unmeasured |
| `Vanilla_r2ph_om32_redraw` | 1024 | -0.114 | 0.026 | 0.052 | 0/2 | DETECTABLE NEGATIVE |

- r(final loss, accuracy) at T=1024: **-0.982** over 6 runs  — |r| > 0.98, the held-out eval carries no information the loss does not

