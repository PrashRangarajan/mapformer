# Action geometry in DELTA space (the angle increments the path integrator cumsums)

`--space delta`: the same basis-invariant tests on Delta = W_out W_in x (per head,
Delta_h = W_out^h W_in^h x, for Vanilla_r2ph), all heads concatenated, omega not
applied. Unlike the latent, a head whose W_out^h is ~0 contributes ~0 here. The
2-plane energy is no longer 1.0 by construction at r=2 per head (4 latent dims).

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla_r3ph | ? | 0.4575 | 0.9967 | 0.3373 | 0.1199 |

## Verdict

None: delta space is descriptive only. The registered geometry readout is the latent one (`--space latent`, the default).

Inference only, 8 seed(s). `opposition` and `|cos|` are scale-free. With the SHARED bottleneck Delta = W_out z lies in an r-dim subspace, so `2-plane energy` is 1.0 by construction at shared r=2; per-head r=2 spans up to 4 dims. The last per-seed column is each head's share of the mean action |Delta_h| (a head near 0 moves nothing).

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm | action |Delta_h| share by head |
|---|---|---|---|---|---|
| Vanilla_r3ph | 0 | 1.8315 | 0.9946 | 0.4337 | 0.757 / 0.243 |
| Vanilla_r3ph | 1 | 0.0274 | 0.4516 | 0.0133 | 0.348 / 0.652 |
| Vanilla_r3ph | 2 | 0.0558 | 0.0270 | 0.0276 | 0.415 / 0.585 |
| Vanilla_r3ph | 3 | 1.5780 | 0.6488 | 0.4018 | 0.848 / 0.152 |
| Vanilla_r3ph | 4 | 0.0258 | 0.2305 | 0.0128 | 0.303 / 0.697 |
| Vanilla_r3ph | 5 | 0.0481 | 0.2257 | 0.0238 | 0.703 / 0.297 |
| Vanilla_r3ph | 6 | 0.0167 | 0.1155 | 0.0084 | 0.622 / 0.378 |
| Vanilla_r3ph | 7 | 0.0763 | 0.0049 | 0.0375 | 0.318 / 0.682 |
