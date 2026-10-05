# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 8 seeds, one batch.

## D = 3, grid 18 (5832 cells, 6 actions, vocab 23)

| arm | final loss | T=1024 |
|---|---|---|
| `Vanilla_r4ph` | 0.0058 | 1.000 ± 0.000 (n=8) |

Paired against `Vanilla_r4ph`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|

- r(final loss, accuracy) at T=1024: **+0.185** over 8 runs

