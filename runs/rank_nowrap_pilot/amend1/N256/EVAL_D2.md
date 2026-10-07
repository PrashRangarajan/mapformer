# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 2 seeds, one batch.

## D = 2, grid 256 (65536 cells, 4 actions, vocab 21)

| arm | final loss | T=1024 | T=2048 |
|---|---|---|---|
| `Vanilla_r2ph_om32` | 1.2282 | 0.603 ± 0.016 (n=2) | 0.594 ± 0.016 (n=2) |
| `Vanilla_r3ph_om32` | 0.6590 | 0.752 ± 0.002 (n=2) | 0.743 ± 0.001 (n=2) |

Paired against `Vanilla_r2ph_om32`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r3ph_om32` | 1024 | +0.149 | 0.014 | 0.028 | 2/2 | DETECTABLE |
| `Vanilla_r3ph_om32` | 2048 | +0.149 | 0.017 | 0.033 | 2/2 | DETECTABLE |

- r(final loss, accuracy) at T=1024: **-0.999** over 4 runs  — |r| > 0.98, the held-out eval carries no information the loss does not
- r(final loss, accuracy) at T=2048: **-0.999** over 4 runs  — |r| > 0.98, the held-out eval carries no information the loss does not

