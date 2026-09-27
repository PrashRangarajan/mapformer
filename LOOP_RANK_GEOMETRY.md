# Action geometry in DELTA space (the angle increments the path integrator cumsums)

`--space delta`: the same basis-invariant tests on Delta = W_out W_in x (per head,
Delta_h = W_out^h W_in^h x, for Vanilla_r2ph), all heads concatenated, omega not
applied. Unlike the latent, a head whose W_out^h is ~0 contributes ~0 here. The
2-plane energy is no longer 1.0 by construction at r=2 per head (4 latent dims).

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Looped | ? | 0.6279 | 1.0000 | 0.8835 | 0.1781 |
| Vanilla_L4 | ? | 0.5496 | 1.0000 | 0.7326 | 0.1729 |

## Verdict

None: delta space is descriptive only. The registered geometry readout is the latent one (`--space latent`, the default).

Inference only, 8 seed(s). `opposition` and `|cos|` are scale-free. With the SHARED bottleneck Delta = W_out z lies in an r-dim subspace, so `2-plane energy` is 1.0 by construction at shared r=2; per-head r=2 spans up to 4 dims. The last per-seed column is each head's share of the mean action |Delta_h| (a head near 0 moves nothing).

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm | action |Delta_h| share by head |
|---|---|---|---|---|---|
| Looped | 0 | 1.3459 | 0.7149 | 0.3801 | 0.541 / 0.459 |
| Looped | 1 | 0.0349 | 0.9997 | 0.0104 | 0.444 / 0.556 |
| Looped | 2 | 1.1299 | 0.9028 | 0.3120 | 0.617 / 0.383 |
| Looped | 3 | 0.0564 | 0.9999 | 0.0238 | 0.480 / 0.520 |
| Looped | 4 | 1.0735 | 0.5706 | 0.2613 | 0.394 / 0.606 |
| Looped | 5 | 1.2115 | 0.8800 | 0.3542 | 0.577 / 0.423 |
| Looped | 6 | 0.1055 | 1.0000 | 0.0510 | 0.631 / 0.369 |
| Looped | 7 | 0.0658 | 0.9999 | 0.0324 | 0.664 / 0.336 |
| Vanilla_L4 | 0 | 1.5485 | 0.1711 | 0.4228 | 0.509 / 0.491 |
| Vanilla_L4 | 1 | 0.0948 | 0.4237 | 0.1666 | 0.499 / 0.501 |
| Vanilla_L4 | 2 | 1.3043 | 0.8623 | 0.3399 | 0.420 / 0.580 |
| Vanilla_L4 | 3 | 0.0118 | 1.0000 | 0.0041 | 0.546 / 0.454 |
| Vanilla_L4 | 4 | 0.0249 | 0.9999 | 0.0104 | 0.684 / 0.316 |
| Vanilla_L4 | 5 | 0.0403 | 0.9998 | 0.0173 | 0.514 / 0.486 |
| Vanilla_L4 | 6 | 1.3380 | 0.4038 | 0.4080 | 0.570 / 0.430 |
| Vanilla_L4 | 7 | 0.0343 | 0.9999 | 0.0139 | 0.528 / 0.472 |
