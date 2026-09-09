# EM vs WM on a realistic navigation benchmark, with rotation handled

## Why this batch

**EM has never been run on MiniGrid or MiniWorld at all.** Every EM number in this
project comes from the torus, Match-Query, compositional, family tree or one DoorKey
BC run. So the question "which fares better on realistic navigation" currently has
no evidence on either side.

And on MiniGrid as measured, **index codes win** (PoPE-Hier 0.955, PoPE-Flat 0.953
against every path-integrated arm), because rotation actions defeat a fixed
per-token increment. **Allocentric recoding** -- record absolute displacement rather
than the commanded turn -- took the position effect from `+0.050` to `+0.488` on the
knob sweep. It has never been combined with an EM arm, and every EM result in the
project is at `r=2`, the rank known to be badly conditioned.

So one batch answers both: does path integration beat an index code once rotation is
handled, and does EM's quadratic capacity pay when the position code is well
conditioned.

## Arms (8 seeds, ONE batch, DoorKey-16x16, allocentric, obj_color)

| arm | rank | params | role |
|---|---|---|---|
| `RoPE` | n/a | ~614K | index control -- the arm that currently wins here |
| `Vanilla` (WM) | 2 | 614,538 | the paper's default |
| `Vanilla_r4` (WM) | 4 | 614,922 | well-conditioned WM |
| `VanillaEM` | 2 | 614,794 | EM as every prior result had it |
| `VanillaEM_r4` | 4 | 615,178 | **the configuration nobody has built** |

Parameter spread across all five is under 0.11%. **`--fast-attn` is NOT used**: the
Hadamard `A_X (*) A_P` cannot be expressed as SDPA, and every arm in a batch must
share the setting.

## Predictions

**P1 (rotation).** With allocentric recoding, path-integrated arms beat `RoPE`.
*Refuted by* index still winning -- which would mean the MiniGrid deficit is not
about rotation actions and the torus allocentric result does not transfer.

**P2 (EM's capacity and noise filtering).** `EM >= WM` at matched rank. DoorKey is
egocentric, so `A_X` is noisy and EM's multiplicative AND-gate should filter it --
the one egocentric data point we have agrees (DoorKey BC match-acc 0.938 vs 0.875).
*Refuted by* EM < WM at matched rank, which would put the capacity account in
trouble on the only realistic benchmark.

**P3 (rank helps EM more).** `r=4` should help EM MORE than WM, i.e. a positive
interaction. EM's failure mode is a collapsing `A_P` multiplying the whole score,
and a better-conditioned phase attacks exactly that. *Refuted by* a null or negative
interaction.

**P4 (the shape of EM's failure, and the one I most expect).** EM's seed **variance**
should exceed WM's, and its failures should be **collapses rather than degradations**
-- a rank-one AND-gate has no additive fallback. `VOCAB_SWEEP_MULTISEED` already
shows this: at `n_obs=256`, EM_P0's per-seed scores are **0.910 / 0.502 / 0.906** --
two seeds beating every WM seed and one at the floor. **Report the per-seed minimum
and the spread, not only the mean**; a mean hides a bimodal arm, which is how that
result was previously read as "EM is worse".

## Checks before reading

Measured floor for this env (previous batches: 0.536 / 0.490), convergence slope,
r(final loss, accuracy) per length, MDE beside every contrast. Ceiling check: the
best arms here reach 0.955, so there is roughly 0.045 of headroom at T=1024 -- thin,
and any contrast must be read against it.
