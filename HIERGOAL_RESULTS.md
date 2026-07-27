# Hierarchical goal-directed navigation — the MapFormer × hierarchy synergy

**The positive result that closes the hierarchy arc.** Everything prior showed
the time-hierarchy helping *generically* (works on a plain transformer too;
task-aware segmentation / collapse didn't help or hurt), and MapFormer barely
beating plain — the two helping orthogonal metrics with no interaction. This
task was designed to create the one demand those lacked: **absolute position at
multiple scales at once, sustained past the training horizon.** Here the
combination is genuinely super-additive.

## Task (see `environment_hier_goal.py`, `validate_hier_goal.py`)

Episode `[room_goal, local_goal, explore·T_e, navigate·T_n]`. Fixed start anchor
(0,0) → path integration yields *absolute* position; hierarchical goal (coarse
room id + fine local id) → target cell needs both scales; explore forces path
integration; navigate = BFS-optimal actions; loss = next-action CE (chance 0.25,
BFS ceiling 1.00). Trained at T_explore=64; evaluated at 64 / 128 / 192 / 256
(>64 = OOD explore length). obs redrawn per episode so localisation must come
from path integration, not memorised obs.

## Result (n=3, held-out env, `HIERGOAL_MULTISEED.md`)

Held-out action accuracy, mean ± std:

| variant | T=64 | T=128 | T=192 | T=256 |
|---|---|---|---|---|
| **MapWM-Hier** | 0.963 | **0.907 ± 0.026** | **0.849 ± 0.065** | **0.853 ± 0.059** |
| MapWM-Flat | 0.958 | 0.656 ± 0.206 | 0.746 ± 0.179 | 0.727 ± 0.188 |
| Plain-Hier | 0.968 | 0.700 ± 0.138 | 0.682 ± 0.122 | 0.624 ± 0.104 |
| Plain-Flat | 0.966 | 0.548 ± 0.084 | 0.669 ± 0.106 | 0.591 ± 0.117 |

**In-distribution (T=64) all four tie at ~0.96.** The task only discriminates OOD.

## The synergy is a genuine 2×2 interaction

interaction = (hierarchy's help on MapFormer) − (hierarchy's help on plain):

| eval length | Δ hier on MapWM | Δ hier on plain | **interaction** |
|---|---|---|---|
| T=128 | +0.251 | +0.152 | **+0.099** |
| T=192 | +0.103 | +0.013 | **+0.090** |
| T=256 | +0.126 | +0.033 | **+0.093** |

A consistent **+0.09–0.10 positive interaction at every OOD length** — the whole
exceeds the sum of parts. Equivalently, MapFormer's edge over plain is +0.08–0.14
flat but **+0.17–0.23 within the hierarchy**: the two amplify each other. This is
exactly what the compositional task lacked (there the hierarchy helped both
backbones equally → zero interaction).

`MapWM-Hier` is also uniquely **stable** OOD (±0.026–0.065 vs ±0.08–0.21 for the
rest) — the only variant that generalises reliably. NLL agrees emphatically
(0.36 vs 0.86–1.6 at T=128).

## Mechanism (why here and not before)

The objective needs absolute position at room scale *and* cell scale, sustained
over a horizon longer than training. MapFormer's path-integration code
extrapolates to OOD length (its known strength); the hierarchy gives it a stable
coarse channel to extrapolate *with* — region-scale path integration is even more
length-robust than cell-scale. Plain index-RoPE has neither the length-robust
position nor a position-bearing coarse level, so it degrades OOD with or without
the hierarchy.

## Honest caveats

- **n=3, single task family; the effect lives entirely at OOD length** — in
  distribution the four are tied.
- **Correction to the single-seed scan:** the scan showed Plain-Hier *collapsing*
  (0.48) and I read it as "hierarchy hurts plain." That was seed noise — at n=3
  hierarchy helps plain slightly too (+0.15/+0.01/+0.03). The real story is a
  super-additive interaction (hierarchy helps both, MapFormer far more), NOT a
  sign flip.
- MapWM-Flat is high-variance OOD (±0.18–0.21); part of MapWM-Hier's win is that
  the hierarchy *stabilises* MapFormer's OOD generalisation, not only raises its
  mean.

## Bottom line

On hierarchical goal-directed navigation at OOD explore length, MapFormer and the
time-hierarchy are load-bearing **together**: `MapWM-Hier` is best by a wide,
reliable margin with a consistent positive interaction. This is the "true
combination" the compositional task could not exhibit — and it required the
*task* to create the multi-scale-position demand, not a cleverer architecture.
