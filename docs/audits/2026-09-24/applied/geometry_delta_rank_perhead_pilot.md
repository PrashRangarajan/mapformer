# Action geometry in DELTA space (the angle increments the path integrator cumsums)

`--space delta`: the same basis-invariant tests on Delta = W_out W_in x (per head,
Delta_h = W_out^h W_in^h x, for Vanilla_r2ph), all heads concatenated, omega not
applied. Unlike the latent, a head whose W_out^h is ~0 contributes ~0 here. The
2-plane energy is no longer 1.0 by construction at r=2 per head (4 latent dims).

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla_r2ph | 2/head | 1.1319 | 0.9259 | 0.1726 | 0.2717 |

## Verdict

None: delta space is descriptive only. The registered geometry readout is the latent one (`--space latent`, the default).

Inference only, 2 seed(s). `opposition` and `|cos|` are scale-free. With the SHARED bottleneck Delta = W_out z lies in an r-dim subspace, so `2-plane energy` is 1.0 by construction at shared r=2; per-head r=2 spans up to 4 dims. The last per-seed column is each head's share of the mean action |Delta_h| (a head near 0 moves nothing).

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm | action |Delta_h| share by head |
|---|---|---|---|---|---|
| Vanilla_r2ph | 0 | 1.4797 | 0.0852 | 0.3890 | 0.584 / 0.416 |
| Vanilla_r2ph | 1 | 0.7840 | 0.2600 | 0.1545 | 0.371 / 0.629 |
