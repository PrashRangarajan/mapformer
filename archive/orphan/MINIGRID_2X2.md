# The torus 2x2 on MiniGrid-DoorKey-16x16 (n=3)

External, published benchmark; egocentric observation (the cell in front of the agent). All four arms trained in ONE batch (rule 3), 50 epochs, 25K cached trajectory buffer.

**Measured floor** (most common scored target) per length: T=128: **0.642**, T=512: **0.536**, T=1024: **0.495**. The original n=1 evaluation reported no floor at all.

| model | position | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| MapWM-Flat (RoPE + path int.) | path-integrated | 0.978 ± 0.007 | 0.800 ± 0.065 | 0.676 ± 0.067 |
| MapPoPE-Flat (PoPE + path int.) | path-integrated | 0.980 ± 0.001 | 0.944 ± 0.017 | 0.917 ± 0.024 |
| RoPE (index) | **index** | 0.947 ± 0.011 | 0.851 ± 0.015 | 0.746 ± 0.012 |
| PoPE-Flat (PoPE + index) | **index** | 0.944 ± 0.006 | 0.906 ± 0.011 | 0.882 ± 0.027 |

| *measured floor* | | *0.642* | *0.536* | *0.495* |

## The torus comparison

For reference, the identical 2x2 on the 64x64 torus paper task at n=8 (`INDEX_BASELINE_PAPER_TASK_n8.md`), floor 0.506:

| | index | path-integrated |
|---|---|---|
| RoPE encoding | 0.530 | **0.967** |
| PoPE encoding | 0.509 | **0.994** |

Both index arms sit ON the floor there. Whether that survives on an egocentric, rotation-actioned environment is what this file measures.

## The two factors swap dominance between environments

Averaging each factor over the other's levels, at matched horizon:

| | encoding effect (PoPE − RoPE) | position effect (path-int − index) |
|---|---|---|
| **torus paper task** (n=8) | **+0.011** | **+0.461** |
| MiniGrid DoorKey-16×16, T=512 | **+0.100** | **−0.007** |
| MiniGrid DoorKey-16×16, T=1024 | **+0.189** | **−0.017** |

On the torus, position decides everything and the encoding is irrelevant. On
MiniGrid it is the reverse: the encoding is worth +0.10 to +0.19 and path
integration is worth **nothing at all** (−0.01 to −0.02). Same 2x2, same four
model classes, opposite conclusion.

This is the strongest available answer to "is the torus result an artifact?" —
it is not an artifact, it is **environment-specific**, and the specificity is
measurable rather than a caveat.

## The n=1 inversion reproduces — but only for the standard model

`MINIGRID_DK16_RESULTS.md` (n=1) had RoPE beating Vanilla at T=512, 0.877 vs
0.754. At n=3: **RoPE 0.851 ± 0.015 vs MapWM-Flat 0.800 ± 0.065**, and at T=1024
0.746 vs 0.676. So the index model does beat the paper's own MapFormer-WM at long
horizon on this benchmark, and MapWM-Flat is also by far the least stable arm
here (±0.065-0.067 against ±0.01-0.03 for everything else).

## My pre-registered mechanism is PARTLY REFUTED

I predicted, before the run, that MiniGrid's rotation-based action space
(turn-left / turn-right / forward, so displacement depends on accumulated
heading) violates the path integrator's assumption that actions are
translations, and that path integration would therefore be actively misleading.

**MapPoPE-Flat is path-integrated and is the best model at every length**
(0.980 / 0.944 / 0.917). If rotations broke path integration as such, it should
have suffered too. It did not.

What is actually true is narrower: **RoPE-encoded path integration underperforms
index position here; PoPE-encoded path integration does not.** PoPE differs by
using d frequency bands rather than d/2, a learnable per-frequency phase bias,
and a content-dependent magnitude — a richer parameterisation of the same angle,
which appears able to absorb an angle that is partly mis-specified. That is a
description of the result, not a tested mechanism, and it should not be promoted
to one without an experiment that manipulates the angle directly.

The clean test remains available and is now more interesting than before: recode
the action stream egocentric → allocentric (the wrapper knows the heading, so it
can emit absolute N/S/E/W). If the angle is the problem, MapWM-Flat should
recover and close on MapPoPE-Flat. If it does not, the difference is about the
encoding's capacity, not about the angle at all.

## Index models are nowhere near the floor here

Measured floors are 0.642 / 0.536 / 0.495. Index arms score 0.947 / 0.851-0.906 /
0.746-0.882 — far above. On the torus the same arms sat ON the floor (0.530 and
0.509 against 0.506). So MiniGrid genuinely does not require path integration in
the way the torus does, which is consistent with it being small (256 cells) and
egocentric, and is why every arm is bunched at T=128.
