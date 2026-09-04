# Does a wider bottleneck destroy the action geometry?

The paper's justification for r=2 is that `Delta_in` IS the 2D movement
vector, so the latent can be read directly. At r>2 that is not guaranteed.
Three basis-invariant structural tests on the trained latents, plus the
check that observations must not move the agent.

| arm | r | opposition <br><sub>|N+S|/scale, 0 is perfect</sub> | 2-plane energy <br><sub>1.0 = spans a plane</sub> | |cos(N,E)| <br><sub>0 = orthogonal</sub> | obs norm / action norm <br><sub>0 = no movement</sub> |
|---|---|---|---|---|---|
| Vanilla | 2 | 0.4950 | 1.0000 | 0.7833 | 0.1394 |
| Vanilla_r4 | 4 | 0.0922 | 1.0000 | 0.1754 | 0.0421 |
| Vanilla_r8 | 8 | 0.0972 | 1.0000 | 0.2030 | 0.0470 |
| Vanilla_r32 | 32 | 0.1109 | 0.9996 | 0.0855 | 0.0702 |

## Verdict

**The geometry survives.** At r=4 the four action latents still span a 2-plane (100.0% of their spectral energy), opposite actions still cancel (0.092 against r=2's 0.495), and the two axes are no less independent (0.175 vs 0.783). The extra rank is optimisation slack, not a different code: the displacement reading the paper relies on is preserved, and can be recovered exactly by projecting onto the top two singular directions.

Inference only, 8 seeds. `opposition` and `|cos|` are scale-free; `2-plane energy` is 1.0 by construction at r=2, so only r>2 rows are informative on that column.
