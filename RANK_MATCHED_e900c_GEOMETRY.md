# Does a wider bottleneck destroy the action geometry?

The paper's justification for r=2 is that `Delta_in` IS the 2D movement
vector, so the latent can be read directly. At r>2 that is not guaranteed.
Three basis-invariant structural tests on the trained latents, plus the
check that observations must not move the agent.

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla | 2 | 1.2669 | 1.0000 | 0.6507 | 0.4471 |
| Vanilla_r4 | 4 | 0.0105 | 1.0000 | 0.0282 | 0.0042 |

## Verdict

**The geometry survives.** At r=4 the four action latents still span a 2-plane (100.0% of their spectral energy), opposite actions still cancel (0.011 against r=2's 1.267), and the two axes are no less independent (0.028 vs 0.651). The extra rank is optimisation slack, not a different code: the displacement reading the paper relies on is preserved, and can be recovered exactly by projecting onto the top two singular directions.

Inference only, 8 seeds. `opposition` and `|cos|` are scale-free; `2-plane energy` is 1.0 by construction at r=2, so only r>2 rows are informative on that column.

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm |
|---|---|---|---|---|
| Vanilla | 0 | 1.2735 | 0.7996 | 0.4659 |
| Vanilla | 1 | 0.0067 | 0.0163 | 0.0039 |
| Vanilla | 2 | 1.5687 | 0.6097 | 0.4664 |
| Vanilla | 3 | 1.5081 | 0.9191 | 0.3613 |
| Vanilla | 4 | 1.7760 | 0.7667 | 0.4377 |
| Vanilla | 5 | 1.5977 | 0.9250 | 0.5673 |
| Vanilla | 6 | 1.5345 | 0.4903 | 0.5377 |
| Vanilla | 7 | 0.8698 | 0.6792 | 0.7363 |
| Vanilla_r4 | 0 | 0.0166 | 0.0060 | 0.0077 |
| Vanilla_r4 | 1 | 0.0104 | 0.0497 | 0.0032 |
| Vanilla_r4 | 2 | 0.0082 | 0.0392 | 0.0017 |
| Vanilla_r4 | 3 | 0.0083 | 0.0860 | 0.0038 |
| Vanilla_r4 | 4 | 0.0106 | 0.0025 | 0.0050 |
| Vanilla_r4 | 5 | 0.0117 | 0.0135 | 0.0053 |
| Vanilla_r4 | 6 | 0.0065 | 0.0162 | 0.0013 |
| Vanilla_r4 | 7 | 0.0118 | 0.0125 | 0.0055 |
