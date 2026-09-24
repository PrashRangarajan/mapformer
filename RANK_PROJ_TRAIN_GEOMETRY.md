# Does a wider bottleneck destroy the action geometry?

The paper's justification for r=2 is that `Delta_in` IS the 2D movement
vector, so the latent can be read directly. At r>2 that is not guaranteed.
Three basis-invariant structural tests on the trained latents, plus the
check that observations must not move the agent.

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla | 2 | 0.0085 | 1.0000 | 0.0596 | 0.0045 |

## Verdict

Inference only, 8 seeds. `opposition` and `|cos|` are scale-free; `2-plane energy` is 1.0 by construction at r=2, so only r>2 rows are informative on that column.

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm |
|---|---|---|---|---|
| Vanilla | 0 | 0.0087 | 0.0313 | 0.0040 |
| Vanilla | 1 | 0.0037 | 0.0215 | 0.0025 |
| Vanilla | 2 | 0.0038 | 0.0388 | 0.0017 |
| Vanilla | 3 | 0.0068 | 0.0260 | 0.0043 |
| Vanilla | 4 | 0.0175 | 0.1481 | 0.0090 |
| Vanilla | 5 | 0.0121 | 0.0663 | 0.0065 |
| Vanilla | 6 | 0.0041 | 0.1199 | 0.0025 |
| Vanilla | 7 | 0.0110 | 0.0248 | 0.0052 |
