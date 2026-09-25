> **CORRECTED 2026-09-25.** The r=4 advantage here was measured after training at T=128
> (out of distribution only), and "r=2 loses because its code is skewed" is WITHDRAWN as the
> cause: within r=2 skew does not predict accuracy, and a rank-2 solution exists and is held
> under training (`RANK_PROJ_RESULTS.md`). At matched length AND matched initialisation
> (`RANK_MI_RESULTS.md`), within 900 epochs at T=1024: our shared r=2 0/8 solved, the paper's
> per-head r=2 2/8, shared r=4 8/8 -- the per-head rank decides whether training FINDS the
> solution. The skew is a symptom of the search failure. Our bottleneck is shared across heads;
> the paper's is per head.

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
