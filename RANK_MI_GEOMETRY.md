# Action geometry in DELTA space (the angle increments the path integrator cumsums)

`--space delta`: the same basis-invariant tests on Delta = W_out W_in x (per head,
Delta_h = W_out^h W_in^h x, for Vanilla_r2ph), all heads concatenated, omega not
applied. Unlike the latent, a head whose W_out^h is ~0 contributes ~0 here. The
2-plane energy is no longer 1.0 by construction at r=2 per head (4 latent dims).

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla | 2 | 1.1313 | 1.0000 | 0.6117 | 0.4061 |
| Vanilla_r2ph | 2/head | 0.9267 | 0.9680 | 0.4380 | 0.3678 |
| Vanilla_r4mi | ? | 0.0534 | 0.9998 | 0.1009 | 0.0251 |

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
| Vanilla_r2ph | 0 | 1.4797 | 0.0852 | 0.3890 | 0.584 / 0.416 |
| Vanilla_r2ph | 1 | 0.7840 | 0.2600 | 0.1545 | 0.371 / 0.629 |
| Vanilla_r2ph | 2 | 0.0516 | 0.1875 | 0.0259 | 0.279 / 0.721 |
| Vanilla_r2ph | 3 | 1.6390 | 0.2027 | 0.7057 | 0.749 / 0.251 |
| Vanilla_r2ph | 4 | 1.5738 | 0.9089 | 0.7205 | 0.325 / 0.675 |
| Vanilla_r2ph | 5 | 0.1967 | 0.7264 | 0.0477 | 0.513 / 0.487 |
| Vanilla_r2ph | 6 | 0.0803 | 0.7831 | 0.0347 | 0.246 / 0.754 |
| Vanilla_r2ph | 7 | 1.6084 | 0.3502 | 0.8646 | 0.716 / 0.284 |
| Vanilla_r4mi | 0 | 0.1305 | 0.1449 | 0.0528 | 0.456 / 0.544 |
| Vanilla_r4mi | 1 | 0.0302 | 0.0584 | 0.0151 | 0.516 / 0.484 |
| Vanilla_r4mi | 2 | 0.0634 | 0.1315 | 0.0314 | 0.517 / 0.483 |
| Vanilla_r4mi | 3 | 0.0376 | 0.0793 | 0.0185 | 0.411 / 0.589 |
| Vanilla_r4mi | 4 | 0.0099 | 0.1357 | 0.0047 | 0.531 / 0.469 |
| Vanilla_r4mi | 5 | 0.0317 | 0.1347 | 0.0159 | 0.490 / 0.510 |
| Vanilla_r4mi | 6 | 0.0273 | 0.0203 | 0.0133 | 0.589 / 0.411 |
| Vanilla_r4mi | 7 | 0.0964 | 0.1022 | 0.0486 | 0.620 / 0.380 |
