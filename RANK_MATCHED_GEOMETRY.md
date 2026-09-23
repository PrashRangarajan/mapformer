# Does a wider bottleneck destroy the action geometry?

The paper's justification for r=2 is that `Delta_in` IS the 2D movement
vector, so the latent can be read directly. At r>2 that is not guaranteed.
Three basis-invariant structural tests on the trained latents, plus the
check that observations must not move the agent.

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla | 2 | 0.6670 | 1.0000 | 0.6441 | 0.1437 |
| Vanilla_r4 | 4 | 0.0874 | 1.0000 | 0.0915 | 0.0485 |

## Verdict

**The geometry survives.** At r=4 the four action latents still span a 2-plane (100.0% of their spectral energy), opposite actions still cancel (0.087 against r=2's 0.667), and the two axes are no less independent (0.091 vs 0.644). The extra rank is optimisation slack, not a different code: the displacement reading the paper relies on is preserved, and can be recovered exactly by projecting onto the top two singular directions.

Inference only, 8 seeds. `opposition` and `|cos|` are scale-free; `2-plane energy` is 1.0 by construction at r=2, so only r>2 rows are informative on that column.

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm |
|---|---|---|---|---|
| Vanilla | 0 | 0.8487 | 0.6183 | 0.1981 |
| Vanilla | 1 | 0.1760 | 0.9924 | 0.0444 |
| Vanilla | 2 | 0.1721 | 0.9920 | 0.0219 |
| Vanilla | 3 | 0.2033 | 0.9892 | 0.0363 |
| Vanilla | 4 | 1.0990 | 0.1870 | 0.1824 |
| Vanilla | 5 | 0.5796 | 0.4442 | 0.0933 |
| Vanilla | 6 | 1.1615 | 0.0908 | 0.2191 |
| Vanilla | 7 | 1.0955 | 0.8390 | 0.3537 |
| Vanilla_r4 | 0 | 0.1420 | 0.1941 | 0.0706 |
| Vanilla_r4 | 1 | 0.0822 | 0.0690 | 0.0411 |
| Vanilla_r4 | 2 | 0.2110 | 0.0272 | 0.1253 |
| Vanilla_r4 | 3 | 0.0590 | 0.0092 | 0.0392 |
| Vanilla_r4 | 4 | 0.0198 | 0.1106 | 0.0207 |
| Vanilla_r4 | 5 | 0.0768 | 0.1794 | 0.0370 |
| Vanilla_r4 | 6 | 0.0435 | 0.1256 | 0.0211 |
| Vanilla_r4 | 7 | 0.0650 | 0.0168 | 0.0327 |
