# Does a wider bottleneck destroy the action geometry?

The paper's justification for r=2 is that `Delta_in` IS the 2D movement
vector, so the latent can be read directly. At r>2 that is not guaranteed.
Three basis-invariant structural tests on the trained latents, plus the
check that observations must not move the agent.

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla | 2 | 1.0952 | 1.0000 | 0.5817 | 0.4090 |
| Vanilla_r4 | 4 | 0.0479 | 1.0000 | 0.0480 | 0.0243 |

## Verdict

**The geometry survives.** At r=4 the four action latents still span a 2-plane (100.0% of their spectral energy), opposite actions still cancel (0.048 against r=2's 1.095), and the two axes are no less independent (0.048 vs 0.582). The extra rank is optimisation slack, not a different code: the displacement reading the paper relies on is preserved, and can be recovered exactly by projecting onto the top two singular directions.

Inference only, 8 seeds. `opposition` and `|cos|` are scale-free; `2-plane energy` is 1.0 by construction at r=2, so only r>2 rows are informative on that column.

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm |
|---|---|---|---|---|
| Vanilla | 0 | 1.3392 | 0.7338 | 0.5552 |
| Vanilla | 1 | 0.0497 | 0.0113 | 0.1082 |
| Vanilla | 2 | 1.3529 | 0.4665 | 0.4147 |
| Vanilla | 3 | 1.3884 | 0.9338 | 0.3759 |
| Vanilla | 4 | 1.5092 | 0.5680 | 0.3803 |
| Vanilla | 5 | 1.0985 | 0.8731 | 0.3512 |
| Vanilla | 6 | 1.2368 | 0.2378 | 0.4166 |
| Vanilla | 7 | 0.7868 | 0.8290 | 0.6698 |
| Vanilla_r4 | 0 | 0.1226 | 0.1391 | 0.0610 |
| Vanilla_r4 | 1 | 0.0199 | 0.0582 | 0.0106 |
| Vanilla_r4 | 2 | 0.0396 | 0.0281 | 0.0215 |
| Vanilla_r4 | 3 | 0.0286 | 0.0034 | 0.0148 |
| Vanilla_r4 | 4 | 0.0196 | 0.0053 | 0.0092 |
| Vanilla_r4 | 5 | 0.0377 | 0.0460 | 0.0192 |
| Vanilla_r4 | 6 | 0.0349 | 0.0115 | 0.0177 |
| Vanilla_r4 | 7 | 0.0803 | 0.0923 | 0.0402 |
