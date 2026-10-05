# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 8 seeds, one batch.

## D = 3, grid 10 (1000 cells, 6 actions, vocab 23)

| arm | final loss | T=1024 |
|---|---|---|
| `Vanilla_r4ph` | 0.2578 | 0.848 ± 0.175 (n=8) |

Paired against `Vanilla_r4ph`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|

- r(final loss, accuracy) at T=1024: **-0.997** over 8 runs  — |r| > 0.98, the held-out eval carries no information the loss does not

