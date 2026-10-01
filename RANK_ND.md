# The rank threshold across dimension

Held-out revisit accuracy on a fresh environment (env-seed 10000), 100 trajectories, 8 seeds, one batch.

## D = 2, grid 32 (1024 cells, 4 actions, vocab 21)

| arm | final loss | T=1024 | T=2048 |
|---|---|---|---|
| `Vanilla_r2ph` | 0.3190 | 0.769 ± 0.121 (n=8) | 0.689 ± 0.139 (n=8) |
| `Vanilla_r3ph` | 0.0047 | 0.999 ± 0.001 (n=8) | 0.976 ± 0.036 (n=8) |

Paired against `Vanilla_r2ph`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r3ph` | 1024 | +0.230 | 0.121 | 0.120 | 7/8 | DETECTABLE |
| `Vanilla_r3ph` | 2048 | +0.286 | 0.131 | 0.130 | 7/8 | DETECTABLE |

- r(final loss, accuracy) at T=1024: **-0.968** over 16 runs
- r(final loss, accuracy) at T=2048: **-0.974** over 16 runs

## D = 3, grid 10 (1000 cells, 6 actions, vocab 23)

| arm | final loss | T=1024 | T=2048 |
|---|---|---|---|
| `Vanilla_r3ph` | 0.3390 | 0.800 ± 0.092 (n=8) | 0.704 ± 0.128 (n=8) |
| `Vanilla_r4ph` | 0.2578 | 0.847 ± 0.176 (n=8) | 0.810 ± 0.205 (n=8) |

Paired against `Vanilla_r3ph`:

| arm | length | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `Vanilla_r4ph` | 1024 | +0.047 | 0.248 | 0.245 | 4/8 | unmeasured |
| `Vanilla_r4ph` | 2048 | +0.106 | 0.304 | 0.301 | 6/8 | unmeasured |

- r(final loss, accuracy) at T=1024: **-0.962** over 16 runs
- r(final loss, accuracy) at T=2048: **-0.929** over 16 runs

