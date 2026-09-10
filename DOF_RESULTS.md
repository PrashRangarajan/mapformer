# Phase degrees of freedom: real, reproducible, and an OPTIMISATION effect

Pre-registration: `DOF_PREREG.md`. 4 arms x 8 seeds, one batch. **Recency half
complete; torus half in flight** -- this file will be extended, not rewritten.

| arm | init `rho` | phase DOF | final loss | T=1024 | T=2048 |
|---|---|---|---|---|---|
| `AlignFree` | 1 | `n_b` | **0.613** | **0.868 +/- 0.079** | 0.730 +/- 0.098 |
| `sep` (random) | ~0 | `n_b` | 0.728 | 0.837 +/- 0.081 | 0.728 +/- 0.106 |
| `AlignLock` | 1 (always) | **0** | 1.048 | 0.720 +/- 0.083 | 0.600 +/- 0.079 |
| `P0` (single) | 1 (always) | **0** | 1.340 | 0.600 +/- 0.126 | 0.510 +/- 0.111 |

Chance 0.0625, most-recent floor 0.0771, chance loss 2.77.

## D4, the precondition, reproduced EXACTLY

`sep - P0 = +0.237` (sd 0.154, MDE 0.152, **8/8**) against the published **+0.237**
in `RECENCY_EM_RESULTS.md`. Same recipe, different batch, three decimal places.
Everything below is therefore about the same effect the project set out to explain.

## D1 CONFIRMED raw -- phase freedom, at matched initial coherence

| contrast | raw | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| **`AlignFree - AlignLock`** | **+0.148** | 0.090 | 0.089 | **8/8** | **DETECTABLE** |
| `AlignLock - P0` (D5) | +0.120 | 0.135 | 0.134 | 6/8 | unmeasured |
| `AlignFree - sep` | +0.031 | 0.127 | 0.126 | 5/8 | unmeasured |

Both D1 arms start at `rho = 1` and match to 64 parameters, so this is phase
freedom alone. The `+0.237` decomposes additively and exactly:

    sep - P0  =  +0.148 (phase freedom)  +0.120 (magnitude freedom)  -0.031 (coherence)

Only the first is detectable. **Do not call phase freedom "the" mechanism** --
magnitude freedom is nearly as large and merely underpowered. What is established
is that FREEDOM matters and initial coherence does not, which is what N5 already
implied from the other side.

Final losses order monotonically in total freedom: 0.613 < 0.728 < 1.048 < 1.340.

## But rule 9 fires, and it moves the claim out of the theory

**`r(final loss, accuracy) = -0.985` over the 32 runs** (`acc = 1.104 - 0.373*loss`).
Above 0.98 the held-out eval carries nothing the training loss does not, and the
only honest analysis is the loss-matched residual:

| contrast | raw | loss-matched | verdict |
|---|---|---|---|
| D1 `AlignFree - AlignLock` | +0.148 (8/8) | **-0.014** (MDE 0.031, 3/8) | **unmeasured** |
| D4 `sep - P0` | +0.237 (8/8) | +0.009 (MDE 0.037) | unmeasured |
| D5 `AlignLock - P0` | +0.120 | +0.011 (MDE 0.039) | unmeasured |

**Every contrast vanishes.** Conditional on reaching the same training loss, the
four arms generalise identically. What phase and magnitude freedom change is how
well the model FITS the task, not what it represents once fitted.

So the DOF account survives -- and lands OUTSIDE the kernel theory. `THEORY_KERNEL.md`
Sec 7 already lists optimisation as something the frame does not cover, and this is
now a measured instance rather than a caveat. The kernel axes describe a function
class; the origin vectors' freedom acts on the landscape.

This is the same shape as the loop result (`L15_LOOP_2X2.md`): raw
`Looped - Vanilla` +0.052 at 12/12, loss-matched +0.006. Two independent
mechanisms in this project that looked representational and measured as
optimisation.

**Caveat on the loss-matching itself.** At `r = -0.985` the regression absorbs
nearly all between-arm variance, so a genuine representational effect of the size
seen here could not survive it either -- the analysis cannot separate "no
representational effect" from "one too small to see beside a 2.2x loss spread".
The defensible statement is the conditional one: *given equal fit, arm identity
does not predict accuracy*. The unconditional differences are real, reproducible
at 8/8, and matter to anyone choosing a parameterisation.

## Still open

The torus half is running. It is the control: if freedom is an optimisation effect
on a task that must reshape the kernel, the map task -- where `phi = 0` is already
correct -- should show a much smaller loss spread across the same four arms.
D2 and D3 land with it.
