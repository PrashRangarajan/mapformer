# Action geometry in DELTA space (the angle increments the path integrator cumsums)

`--space delta`: the same basis-invariant tests on Delta = W_out W_in x (per head,
Delta_h = W_out^h W_in^h x, for Vanilla_r2ph), all heads concatenated, omega not
applied. Unlike the latent, a head whose W_out^h is ~0 contributes ~0 here. The
2-plane energy is no longer 1.0 by construction at r=2 per head (4 latent dims).

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla | 2 | 1.1313 | 1.0000 | 0.6117 | 0.4061 |
| Vanilla_r4 | 4 | 0.0477 | 1.0000 | 0.0893 | 0.0238 |

## Verdict

None: delta space is descriptive only. The registered geometry readout is the latent one (`--space latent`, the default).

Inference only, 8 seed(s). `opposition` and `|cos|` are scale-free. With the SHARED bottleneck Delta = W_out z lies in an r-dim subspace, so `2-plane energy` is 1.0 by construction at shared r=2; per-head r=2 spans up to 4 dims. The last per-seed column is each head's share of the mean action |Delta_h| (a head near 0 moves nothing).

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm | action |Delta_h| share by head |
|---|---|---|---|---|---|
| Vanilla | 0 | 1.2599 | 0.8369 | 0.4854 | 0.552 / 0.448 |
| Vanilla | 1 | 0.0560 | 0.1295 | 0.1194 | 0.512 / 0.488 |
| Vanilla | 2 | 1.4238 | 0.2018 | 0.4434 | 0.459 / 0.541 |
| Vanilla | 3 | 1.5795 | 0.9568 | 0.4260 | 0.538 / 0.462 |
| Vanilla | 4 | 1.6158 | 0.6332 | 0.4043 | 0.597 / 0.403 |
| Vanilla | 5 | 1.0361 | 0.9384 | 0.3008 | 0.479 / 0.521 |
| Vanilla | 6 | 1.3510 | 0.3563 | 0.4443 | 0.526 / 0.474 |
| Vanilla | 7 | 0.7281 | 0.8406 | 0.6247 | 0.488 / 0.512 |
| Vanilla_r4 | 0 | 0.1287 | 0.1397 | 0.0634 | 0.387 / 0.613 |
| Vanilla_r4 | 1 | 0.0194 | 0.0154 | 0.0099 | 0.411 / 0.589 |
| Vanilla_r4 | 2 | 0.0375 | 0.0250 | 0.0186 | 0.565 / 0.435 |
| Vanilla_r4 | 3 | 0.0296 | 0.0591 | 0.0148 | 0.499 / 0.501 |
| Vanilla_r4 | 4 | 0.0183 | 0.1890 | 0.0086 | 0.400 / 0.600 |
| Vanilla_r4 | 5 | 0.0362 | 0.0695 | 0.0189 | 0.402 / 0.598 |
| Vanilla_r4 | 6 | 0.0319 | 0.0674 | 0.0160 | 0.523 / 0.477 |
| Vanilla_r4 | 7 | 0.0801 | 0.1494 | 0.0402 | 0.410 / 0.590 |
