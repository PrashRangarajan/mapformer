# Pre-registration: D5 at n=24 -- is magnitude freedom worth anything?

Extends `runs/dof/recency` with seeds 8-23. Written before those exist.

## What D5 asks, and a verification that it asks it

`AlignLock` (`k_0i = s_i * q_0i`) differs from `P0` (`k_0 = q_0 = p_0`) in that the
per-block magnitude RATIO `s_i` is free. Neither has phase freedom. At n=8 the
contrast was **+0.120 accuracy (MDE 0.134, 6/8)** and **-0.292 final loss
(MDE 0.317, 2/8)** -- unmeasured on both, just inside each.

**A confound checked and cleared first.** `s_i` is unconstrained, so a negative
value would flip that block's phase to `pi` -- making `AlignLock` a *binary phase
freedom* arm rather than a pure magnitude arm, and D5 a diluted copy of D1. Measured
over all 512 block-scales on both tasks: **zero negative, on every seed.** The
scales move substantially (1.0 -> 0.794 on recency, 1.0 -> 0.500 on the torus), so
the freedom is exercised, just never into sign. D5 measures magnitude freedom alone.

## Design

4 arms x seeds 8-23 (16 new) on recency, one batch, recipe identical to
`run_dof.sh`. Pooled with the existing seeds 0-7 gives **n=24**. All four arms are
extended, not just the two D5 needs, so the additive decomposition of the +0.237
stays on a single `n` instead of becoming ragged.

Cross-batch pooling is licensed here by measurement, not assumption: recency
reproduced `sep - P0` at **+0.237 to three decimals** across two independent
batches (`DOF_RESULTS.md` D4).

Power at n=24: accuracy MDE **0.077** (resolves +0.120), loss MDE **0.183**
(resolves -0.292).

## Predictions

**E1 (primary).** `AlignLock - P0` accuracy at n=24 is **positive and clears
0.077**. *Falsified if* inside MDE or negative.

**E2.** The same contrast on final loss is **negative and clears 0.183** -- i.e.
magnitude freedom buys FIT. *Falsified if* inside MDE.

**E3 -- registered in advance so it is not read as a surprise.** The loss-matched
residual for D5 is **ZERO**. On this task `r(loss, acc) = -0.985`, and every
contrast measured so far vanishes under loss-matching (D1 +0.148 -> -0.014,
D4 +0.237 -> +0.009, D5 +0.120 -> +0.011). E1 and E2 together would therefore say
magnitude freedom improves the FIT and nothing else -- the same verdict D1 got, not
a representational claim. *Falsified if* the loss-matched residual clears its MDE,
which would make magnitude freedom the first origin-vector effect on this task to
survive rule 9.

**E4 -- independent replication.** Seeds 8-23 ALONE (n=16, MDE 0.095) reproduce the
sign and rough size of the n=8 estimate. This is the guard against a pooled result
being carried by the original seeds. *Falsified if* the new seeds alone give a
different sign, in which case the n=8 result was noise and the pooled number should
not be quoted.

**E5 -- D1 should survive the extension.** `AlignFree - AlignLock` stays detectable
at n=24 (n=8 gave +0.148, 8/8). A no-cost check that the extension did not change
the task.

## What this cannot settle

Even if E1 and E2 both land, phase freedom (+0.148) and magnitude freedom (+0.120)
are measured on the SAME axis of "how well can the kernel be reshaped" and are not
independent manipulations of one mechanism -- `AlignLock` sits between `P0` and
`AlignFree` on a single ladder. This batch can say both rungs are real; it cannot
attribute the effect to phase versus magnitude as separate causes.
