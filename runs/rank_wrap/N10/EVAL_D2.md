# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 8 seeds, one batch.

## D = 2, grid 10 (100 cells, 4 actions, vocab 21)

| arm | final loss | T=1024 | T=2048 |
|---|---|---|---|
| `Vanilla_r3ph` | 0.0680 | 0.273 ± 0.028 (n=8) | 0.282 ± 0.023 (n=8) |

Paired against `Vanilla_r3ph`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|

- r(final loss, accuracy) at T=1024: **+0.338** over 8 runs
- r(final loss, accuracy) at T=2048: **+0.308** over 8 runs

