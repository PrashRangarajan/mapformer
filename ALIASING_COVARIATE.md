# Aliasing covariate for the dissociation sweep

`n_templates` moves two things at once: how many distinct motifs there are to compose (the intended axis) and how many grid cells look identical (an unintended one). This measures the second, so it can be reported beside the sweep instead of hand-waved.

World built by the environment's own `_draw_world` / `_obs_map_from` (rule 7). 64x64 grid, room size 8, n=3 worlds averaged.

| n_templates | cells sharing an observation | H(obs) bits | positions consistent with a 1-step run | 2-step | 4-step | 8-step |
|---|---|---|---|---|---|---|
| 2 | 1107 | 2.91 | 1107 | 304.4 | 50.63 | 20.43 |
| 4 | 1086 | 2.97 | 1086 | 298.9 | 33.64 | 7.40 |
| 8 | 1077 | 3.00 | 1077 | 293.9 | 28.24 | 3.35 |
| 16 | 1165 | 2.91 | 1165 | 332.0 | 30.76 | 2.14 |

## Result: this does NOT explain P2's failure, and rules out the explanation I gave

I predicted the confound would work like this: fewer templates means more cells
that look identical, so knowing precisely where you are pays off less, so path
integration's `exact_acc` advantage grows as n_templates rises. Two problems.

**1. Per-cell aliasing is flat.** ~1100 cells share any given observation at
every sweep point, and H(obs) sits at 2.91-3.00 bits throughout. The observation
marginal is set by `n_obs_types` and `p_empty`, which the sweep does not touch.
So "fewer templates means more identical-looking cells" is simply false.

**2. Sequence-level disambiguation moves a lot, but in the WRONG DIRECTION.**
`run_8` falls 20.4 -> 7.4 -> 3.4 -> 2.1, a 10x sharpening. Content localises far
better at high n_templates. Under my account that should make path integration
LESS necessary there. The sweep shows its advantage GROWING there
(+0.060 -> +0.158 -> +0.250). The covariate predicts the opposite of what
happened.

So P2's failure is not an aliasing artifact and cannot be dismissed as one.

## A better hypothesis, from the sweep's own numbers, NOT independently tested

Look at `exact_acc` @T=1024 by backbone as n_templates rises:

| | nt=2 | nt=4 | nt=8 |
|---|---|---|---|
| MapWM-Flat (path-integrated) | 0.609 | 0.764 | **0.871** |
| Plain-Flat (index) | 0.582 | 0.630 | **0.590** |

The path-integrated model improves by 0.26 across the sweep; the index model is
pinned within 0.05. The gap grows because **the position-aware ceiling rises with
n_templates while the index ceiling does not** — a richer, less repetitive world
gives a model that knows where it is more to exploit, and gives a model that does
not know where it is nothing extra.

That is a headroom account rather than an aliasing account, and it is consistent
with `run_8`: more distinguishable worlds are exactly the worlds where knowing
your position buys the most. It is read off the sweep's own table, so it is a
description of these data and not yet a tested claim. Testing it needs a
condition where world richness rises WITHOUT motif count rising.

## What this table can still be used for

Reporting, not correcting. It documents that the sweep axis moves sequence-level
disambiguation by 10x, which any reading of the sweep has to acknowledge — while
also showing that this particular confound cannot be the explanation for P2,
because its sign is wrong.

## What it cannot do

It measures a confound; it does not remove one. Separating "how much structure is
there to compose" from "how rich is the world" needs a design where the two move
independently — holding the number of distinct observation types per template
fixed while varying how many templates tile the grid. That is a different sweep,
not a reanalysis of this one.
