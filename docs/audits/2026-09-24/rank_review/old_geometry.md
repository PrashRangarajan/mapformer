# Does a wider bottleneck destroy the action geometry?

The paper's justification for r=2 is that `Delta_in` IS the 2D movement
vector, so the latent can be read directly. At r>2 that is not guaranteed.
Three basis-invariant structural tests on the trained latents, plus the
check that observations must not move the agent.

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla | 2 | 0.4950 | 1.0000 | 0.7833 | 0.1394 |
| Vanilla_r4 | 4 | 0.0922 | 1.0000 | 0.1754 | 0.0421 |

## Verdict

**The geometry survives.** At r=4 the four action latents still span a 2-plane (100.0% of their spectral energy), opposite actions still cancel (0.092 against r=2's 0.495), and the two axes are no less independent (0.175 vs 0.783). The extra rank is optimisation slack, not a different code: the displacement reading the paper relies on is preserved, and can be recovered exactly by projecting onto the top two singular directions.

Inference only, 8 seeds. `opposition` and `|cos|` are scale-free; `2-plane energy` is 1.0 by construction at r=2, so only r>2 rows are informative on that column.

## Per seed

| arm | seed | opposition | |cos(N,E)| | obs/action norm |
|---|---|---|---|---|
| Vanilla | 0 | 1.2155 | 0.3682 | 0.2857 |
| Vanilla | 1 | 0.1200 | 0.9946 | 0.0376 |
| Vanilla | 2 | 0.0914 | 0.9973 | 0.0235 |
| Vanilla | 3 | 0.1307 | 0.9944 | 0.0232 |
| Vanilla | 4 | 1.0855 | 0.3301 | 0.1825 |
| Vanilla | 5 | 0.5117 | 0.8987 | 0.0946 |
| Vanilla | 6 | 0.1399 | 0.9945 | 0.0188 |
| Vanilla | 7 | 0.6653 | 0.6884 | 0.4493 |
| Vanilla_r4 | 0 | 0.1479 | 0.3627 | 0.0729 |
| Vanilla_r4 | 1 | 0.0598 | 0.2547 | 0.0292 |
| Vanilla_r4 | 2 | 0.2045 | 0.1279 | 0.1000 |
| Vanilla_r4 | 3 | 0.0458 | 0.1736 | 0.0233 |
| Vanilla_r4 | 4 | 0.0438 | 0.0351 | 0.0274 |
| Vanilla_r4 | 5 | 0.0977 | 0.1976 | 0.0356 |
| Vanilla_r4 | 6 | 0.0719 | 0.1227 | 0.0171 |
| Vanilla_r4 | 7 | 0.0661 | 0.1285 | 0.0311 |
