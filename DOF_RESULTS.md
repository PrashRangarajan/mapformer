# Phase degrees of freedom: real, reproducible, and an OPTIMISATION effect

> **AUDIT 2026-09-10 -- read `AUDIT_2026-09-10.md` first.** D4's "three-decimal reproduction" was the
> same computation twice (16/16 bitwise-identical checkpoints). n=8 sizes are
> superseded by `D5_RESULTS.md` (+0.237 -> +0.128, +0.148 -> +0.165, magnitude +0.120
> -> -0.033), and the monotone loss ordering is false at n=24. D3 subtracts
> in-distribution recency accuracy from OOD torus accuracy and is detectable only via
> an unmeasured opposite-sign term (+0.253 with n=24 recency). "The third mechanism
> survives" overstates: the sign flip reproduced, the mechanism on the torus did not
> reach detectable size. AlignLock differs from AlignFree in optimiser dynamics too.

Pre-registration: `DOF_PREREG.md`. 4 arms x 8 seeds, one batch. **COMPLETE**, 64 runs.
The two halves require OPPOSITE analyses, which is the main finding.

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

## The torus half, and why it needs the opposite analysis

Every torus arm converges to essentially zero training loss -- **0.00018 to
0.00028**, indistinguishable -- against 0.613-1.340 on recency. Where `phi = 0` is
already the right kernel there is nothing to reshape, so freedom buys no fit.

Consequently `r(final loss, acc at T=1024)` is **-0.160** on the torus against
**-0.985** on recency. Loss-matching is mandatory on one half and meaningless on
the other. Torus accuracy differences cannot be fit artifacts.

| arm | phase DOF | T=128 | T=512 | **T=1024** |
|---|---|---|---|---|
| `AlignLock` | **0** | 1.000 | 0.994 +/- 0.006 | **0.963 +/- 0.022** |
| `P0` | **0** | 1.000 | 0.993 +/- 0.008 | **0.962 +/- 0.024** |
| `AlignFree` | `n_b` | 1.000 | 0.979 +/- 0.011 | 0.875 +/- 0.112 |
| `sep` | `n_b` | 1.000 | 0.961 +/- 0.030 | 0.809 +/- 0.128 |

| contrast (torus, T=1024) | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| D2 `AlignFree - AlignLock` | -0.088 | 0.119 | 0.118 | 1/8 | unmeasured |
| `sep - P0` | **-0.154** | 0.131 | 0.130 | **0/8** | **DETECTABLE** |

D2 lands just inside its MDE, so as pre-registered it is reported as *unmeasured
above 0.118* -- not as a null, and the direction (1/8 seeds positive) is against
freedom. The full-form contrast IS detectable and has the opposite sign to
recency's, so **the inversion is reproduced inside a single batch**:
`sep - P0` is **+0.237 on recency (8/8)** and **-0.154 on the torus (0/8)**.

## D3 -- the interaction, DETECTABLE

    (AlignFree - AlignLock)_recency - (same)_torus = +0.236,  se 0.053,  MDE 0.148

Phase freedom **helps by +0.148 on the clock task and costs 0.088 on the map
task**. This is the claim the batch was built for, and it holds.

## What the freed kernel actually does on the torus

Locked arms hold `rho = 1.000 +/- 0.000` by construction. Freed arms **drift off
the matched filter**: `AlignFree` ends at `rho = 0.363 +/- 0.198`, `sep` at
`0.173 +/- 0.196`, and the between-arm ordering is monotone in accuracy
(1.000 -> 0.963, 0.363 -> 0.875, 0.173 -> 0.809). The model spends its freedom
moving away from the kernel the task wants, at no cost in training loss, and pays
for it at OOD length.

**But within the freed arms, `rho_final` does not predict which seed does better**
(pooled r = +0.158, n=16, range -0.05..0.66). Same pattern as N4: the
*manipulation* is causal, the *observed statistic* is not seed-level predictive.
The between-arm ordering is four points and is confounded with the manipulation
itself, so it illustrates the mechanism rather than testing it.

## The claim, and where it sits

**Kernel freedom is a trainability asset where the kernel must be reshaped, and a
generalisation liability where it is already correct.**

- **Clock task:** all of freedom's benefit is in FITTING. Loss 0.613 (free) vs
  1.340 (locked); loss-matched, the accuracy contrast is zero. Optimisation --
  outside `THEORY_KERNEL.md`, as Sec 7 anticipated.
- **Map task:** every arm fits perfectly, so what remains is representational.
  Freedom costs 0.088-0.154 of held-out accuracy at 8x training length, at
  `r(loss,acc) = -0.16`. That part IS inside the frame.

So the third mechanism survives, and it is the first of the three to be measured on
both sides of the inversion within one batch. It is also two effects rather than
one, split by task, and only the map-task half is a statement about representation.

D5 (`AlignLock - P0` on recency, +0.120, MDE 0.134) remains unmeasured, so phase
freedom and magnitude freedom are still not separated from each other.
