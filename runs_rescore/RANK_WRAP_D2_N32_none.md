# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 8 seeds, one batch.

## D = 2, grid 32 (1024 cells, 4 actions, vocab 21)

| arm | final loss | T=1024 |
|---|---|---|
| `Vanilla_r3ph` | 0.0047 | 0.999 ± 0.001 (n=8) |

Paired against `Vanilla_r3ph`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|

- r(final loss, accuracy) at T=1024: **+0.180** over 8 runs

