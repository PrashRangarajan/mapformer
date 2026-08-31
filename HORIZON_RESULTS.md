# How far can attention path-integrate? The horizon is CAPACITY, not architecture

`REVISIT_DISTANCE.md` measured that an index-position model beats the blank floor
only at recurrence interval 1-2 and nowhere else, and I called that "attention
path-integrates over a horizon of about two steps". This grid asks whether that
horizon is an architectural bound or a property of the 1-layer paper config.

**It is the config.** RoPE (index position, architecture-matched to MapFormer-WM),
revisit accuracy by recurrence interval, n=3 seeds, floor ~0.50 throughout:

| config | params | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ | horizon |
|---|---|---|---|---|---|---|---|---|---|
| L1 d128, 16 ep | 204K | 0.563 | 0.521 | 0.503 | 0.504 | 0.507 | 0.485 | 0.488 | **~2** |
| L1 d128, 50 ep | 204K | 0.797 | 0.719 | 0.575 | 0.514 | 0.490 | 0.477 | 0.457 | **~8** |
| L2 d128, 16 ep | 402K | 0.712 | 0.698 | 0.716 | 0.610 | 0.473 | 0.457 | 0.454 | **~16** |
| L2 d256, 16 ep | 1.59M | 0.945 | 0.943 | 0.949 | 0.794 | 0.497 | 0.485 | 0.477 | **~16** |
| L4 d128, 16 ep | 799K | 0.997 | 0.995 | 0.993 | 0.947 | 0.579 | 0.475 | 0.469 | **~16-32** |
| L4 d256, 16 ep | 3.17M | 1.000 | 0.988 | 0.972 | 0.962 | 0.596 | 0.485 | 0.503 | **~32** |

(horizon = largest bucket clearing the floor by >0.10)

The horizon moves with **depth** (~2 → ~16 → ~32 for 1 → 2 → 4 layers), with
**width** (L2: 0.610 → 0.794 at interval 9-16 when d goes 128 → 256), and with
**training budget** (L1: ~2 → ~8 from 16 to 50 epochs). So the two-step figure was
an artifact of the paper's 1-layer config, and the strong "architectural bound"
version of the claim is **refuted**.

## But the wall does not go away — it moves and then stops

Every index config collapses to the floor beyond interval ~32, including 3.17M
parameters at 4 layers and 256 wide (0.596 at 17-32, then 0.485 and 0.503).
Meanwhile the path-integrated model handles every bucket at **1 layer and 204K
parameters**:

| | interval 33-64 | interval 65+ |
|---|---|---|
| Vanilla, L1 d128 (204K) | **0.935** | **0.880** |
| RoPE, L4 d256 (3.17M) | 0.485 | 0.503 |

**A 15x larger index model, four times as deep and twice as wide, still fails
past ~32 steps where a 1-layer path-integrated model succeeds at 65+.** That is
the quantitative form of what explicit path integration buys, and it survives the
capacity explanation rather than being dissolved by it.

## RETRACTED 2026-08-30: "scale HURTS the path-integrated model at long range"

> The section below does NOT reproduce under a fair schedule. Retrained at 300
> epochs with 5% warmup + cosine (n=3), the long-range means are **L1 0.948,
> L4 0.998, Looped x4 0.994** -- monotone, with the largest model best. The
> non-monotonicity was an artifact of the 16-epoch LinearLR budget, exactly as
> this section's own caveat allowed. See LOOPED_PILOT.md.
>
> **The rest of this file is budget-limited the same way.** Every number here was
> trained with LinearLR(1.0->0.0), which decays from step one and can trap a run
> on a plateau (standing rule 10). Retrained at 300 ep warmup+cosine, RoPE L1's
> horizon is **9-16**, not the ~2 reported at 16 epochs or the ~8 at 50. Treat the
> horizon values below as LOWER BOUNDS.
>
> **What survives, and is strengthened:** the wall. Under the fair budget every
> index configuration still collapses past interval ~32 (long-range means
> 0.498 / 0.515 / 0.497 for L1 / L4 / Looped x4) while a 204K one-layer
> path-integrated model holds 0.945 at 65+. Giving attention a proper budget
> raises its reach and still does not close the gap.

## (retracted) An unexpected finding: scale HURTS the path-integrated model at long range

Vanilla's long-interval accuracy is not monotone in capacity:

| Vanilla config | 33-64 | 65+ |
|---|---|---|
| L2 d256 (1.59M) | **0.989** | **0.976** |
| L4 d256 (3.17M) | 0.856 | 0.782 |
| L4 d128 (799K) | 0.935 | 0.841 |

The largest model is the second worst at 65+. Not investigated here; it could be
optimisation at a fixed 16-epoch budget rather than capacity per se, and the
budget was held fixed across configs by design. Flagged, not explained.

## What this does and does not license

**Does:** "explicit path integration extends the range over which position can be
recovered, and the gap is not closed by 15x the parameters" — with a measured
curve rather than an assertion.

**Does not:** the earlier framing that attention has a fixed ~2-step horizon. That
was one config's number reported as a property of attention, and it is withdrawn.

**Still open:** whether the wall at ~32 moves further with more scale than tested
here. Three depths and two widths do not establish saturation; they establish
that the horizon grows sublinearly in parameters (204K → 3.17M, a 15x increase,
moves it ~2 → ~32) while path integration clears it at the smallest size tested.

## Follow-up 2026-08-30: recursion buys the horizon that depth bought

`LOOPED_PILOT.md`. A weight-shared block applied 4x (param-parity exact with 1
layer: 207,457 vs L4's 802,273), torus, 300 ep warmup+cosine, n=3.

| index arm | params | 9-16 | 17-32 | 33-64 | 65+ | horizon |
|---|---|---|---|---|---|---|
| RoPE L1 | 204K | 0.615 | 0.492 | 0.496 | 0.499 | 9-16 |
| RoPE L4 | 802K | 0.995 | 0.878 | 0.523 | 0.508 | 17-32 |
| RoPE Looped x4 | 204K | 0.992 | 0.855 | 0.512 | 0.481 | 17-32 |

At interval 17-32 the loop is worth **+0.363 over L1 (sd 0.018, MDE 0.029, 3/3
seeds)** and is indistinguishable from four REAL layers (delta -0.023) at a
quarter of the parameters. So what depth was providing is effective ITERATION,
not per-layer specialisation.

On the path-integrated arm the loop shows no established gain (+0.046, sd 0.074,
MDE 0.120, one seed negative) -- but that arm is at 0.948 with 0.052 of headroom,
so the null is uninterpretable rather than informative. A headroom test on
Match-Query is the follow-up.

