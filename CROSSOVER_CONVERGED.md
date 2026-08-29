# Is the crossover real? Grid 8 with BOTH arms converged

400 epochs, warmup + cosine, oracle recode, n=3 -- the same budget that settled grid 32. At 100 epochs grid 8 gave RoPE 0.977 vs Vanilla 0.448, but BOTH arms were unconverged there (Vanilla converged 0/3), so that -0.529 -- the anchor of the whole crossover claim -- was never a converged measurement.

| seed | Vanilla loss | flat? | Vanilla acc | RoPE loss | flat? | RoPE acc | delta |
|---|---|---|---|---|---|---|---|
| 0 | 0.0036 | Y | 0.992 | 0.0186 | Y | 1.000 | **-0.008** |
| 1 | 0.0089 | Y | 0.996 | 0.0172 | Y | 1.000 | **-0.003** |
| 2 | 0.0699 | Y | 0.982 | 0.0177 | Y | 1.000 | **-0.018** |

**grid 8 converged effect (path-int − index): -0.010** (sd 0.007, 6/6 arms flat)
**grid 32 converged effect, for comparison: +0.173**

**GRID 8 DOES NOT DISCRIMINATE.** The arms land within the noise floor of each other, so there is no crossover -- only a grid-32 position effect.

## FINAL: the crossover is withdrawn; the surviving claim is monotone in aliasing

Both MiniWorld points now have EVERY arm trained to a flat loss:

| environment | cells sharing an obs token | converged effect (path-int - index) |
|---|---|---|
| MiniWorld grid 8  | 2   | **-0.010** (n=3, 6/6 flat) -- no effect, both solve it |
| MiniWorld grid 32 | 32  | **+0.173** (n=3, 6/6 flat) -- above the 0.150 noise floor |
| Torus 64x64       | 128 | +0.461 (n=8, index arms at the chance floor) |

**WITHDRAWN:** the crossover. Index never "beat" path integration at grid 8 -- the
-0.529 was Vanilla failing to train (0.448 at 100ep/linear -> 0.990 at 400ep/cosine).
There is no regime where index is genuinely better; there is a regime where the
choice does not matter.

**WITHDRAWN:** the attention-horizon mechanism, already falsified by gate G6 (revisit
lags SHORTEN with grid size: 47/43/38/33, and the fraction inside the ~32-step
horizon RISES 0.43->0.50).

**SURVIVING CLAIM:** the position code matters in proportion to how ALIASED the
observations are. At 2 cells per token content nearly identifies location and an
integrated position buys nothing; at 32 it is ambiguous and path integration wins;
at 128 (torus) index sits at the chance floor. Monotone, no reversal, and consistent
with every measurement including the gate data that killed the previous mechanism.

**Scope, stated plainly:** two MiniWorld points plus one torus point from separate
work; aliasing co-varies with grid size here rather than being manipulated
independently. The clean test would hold grid size fixed and vary n_obs (16/8/4 at
grid 32), which would separate aliasing from map size outright. Not yet run.
