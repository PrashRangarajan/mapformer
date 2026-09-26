# Action geometry in DELTA space (the angle increments the path integrator cumsums)

`--space delta`: the same basis-invariant tests on Delta = W_out W_in x (per head,
Delta_h = W_out^h W_in^h x, for Vanilla_r2ph), all heads concatenated, omega not
applied. Unlike the latent, a head whose W_out^h is ~0 contributes ~0 here. The
2-plane energy is no longer 1.0 by construction at r=2 per head (4 latent dims).

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla_r4mibd | ? | 0.9640 | 0.9719 | 0.6541 | 0.2459 |
| Vanilla_r4ph | ? | 0.0507 | 1.0000 | 0.1311 | 0.0252 |

## Verdict

None: delta space is descriptive only. The registered geometry readout is the latent one (`--space latent`, the default).

Inference only, 8 seed(s). `opposition` and `|cos|` are scale-free. With the SHARED bottleneck Delta = W_out z lies in an r-dim subspace, so `2-plane energy` is 1.0 by construction at shared r=2; per-head r=2 spans up to 4 dims. The last per-seed column is each head's share of the mean action |Delta_h| (a head near 0 moves nothing).

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm | action |Delta_h| share by head |
|---|---|---|---|---|---|
| Vanilla_r4mibd | 0 | 1.6196 | 0.9036 | 0.3944 | 0.405 / 0.595 |
| Vanilla_r4mibd | 1 | 0.3059 | 0.9732 | 0.1446 | 0.347 / 0.653 |
| Vanilla_r4mibd | 2 | 0.0528 | 0.2152 | 0.0230 | 0.666 / 0.334 |
| Vanilla_r4mibd | 3 | 1.5649 | 0.8416 | 0.3720 | 0.750 / 0.250 |
| Vanilla_r4mibd | 4 | 1.5086 | 0.3469 | 0.3767 | 0.743 / 0.257 |
| Vanilla_r4mibd | 5 | 1.2110 | 0.9559 | 0.2910 | 0.206 / 0.794 |
| Vanilla_r4mibd | 6 | 1.3807 | 0.1584 | 0.3338 | 0.347 / 0.653 |
| Vanilla_r4mibd | 7 | 0.0684 | 0.8381 | 0.0316 | 0.617 / 0.383 |
| Vanilla_r4ph | 0 | 0.1421 | 0.3070 | 0.0707 | 0.319 / 0.681 |
| Vanilla_r4ph | 1 | 0.0243 | 0.2161 | 0.0127 | 0.684 / 0.316 |
| Vanilla_r4ph | 2 | 0.0477 | 0.0598 | 0.0241 | 0.341 / 0.659 |
| Vanilla_r4ph | 3 | 0.0398 | 0.0763 | 0.0195 | 0.715 / 0.285 |
| Vanilla_r4ph | 4 | 0.0190 | 0.0770 | 0.0090 | 0.708 / 0.292 |
| Vanilla_r4ph | 5 | 0.0284 | 0.2391 | 0.0140 | 0.344 / 0.656 |
| Vanilla_r4ph | 6 | 0.0206 | 0.0560 | 0.0102 | 0.334 / 0.666 |
| Vanilla_r4ph | 7 | 0.0836 | 0.0178 | 0.0414 | 0.662 / 0.338 |
