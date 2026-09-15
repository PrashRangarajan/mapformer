---
name: feedback-convergence-first
description: Verify convergence, noise floor and power BEFORE any architectural comparison — four retractions in one week all traced to this.
metadata:
  type: feedback
---

Run `python3 -m mapformer.experiment_audit --runs-dir <dir> --control <inert-twin>
--control-of <real-arm>` before interpreting ANY run directory. It checks all of the
below in ~30s and would have caught every retraction of 2026-08-26..28.

**Why:** ~200 training arms produced 5 architectural claims that week; 4 were
retracted. Every one had the same root cause, rediscovered separately days apart.

1. **Convergence + LR schedule.** `LinearLR(1.0->0.0)` decays from step one with no
   warmup — on a plateau-then-cliff landscape a run can never escape the plateau
   late, so the budget measures "did the transition fire early". Switching to 5%
   warmup + cosine-to-10% moved an arm from **0.448 to 0.990 on the same task** and
   INVERTED the headline's sign.
2. **Noise floor must be MEASURED**, via an arm provably function-identical to a real
   one (params identical, effect multiplied out, zero grad). Measured 0.150 on
   MiniWorld — larger than most effects being chased.
3. **Accuracy may just be the training loss.** r = −0.996 over 57 runs. Then the
   held-out eval adds nothing and only loss-matched residuals mean anything.
4. **"Null" requires power.** MDE = 2.8·sd/√n; at n=9, sd 0.18 that's 0.165. Say
   "unmeasured", not "null". And conditioning on convergence can select into a
   ceiling, showing ~0 by construction — report threshold sensitivity.
5. **Seed the comparison you're claiming.** Seeds on "A vs baseline" do not support a
   claim about "A vs its components".
6. **Cross-check your own gate data against your proposed mechanism.** The
   attention-horizon story was falsified by gate G6, collected before training and
   never looked at.

See [[project_miniworld_flip_negative]] for the substantive outcome, KNOWN_BUGS.md for
the silent bugs, CLAUDE.md standing rules 8–12.


## Loss-matching needs overlapping losses (bought 2026-09-13/14)

Twice in two days a loss-matched contrast removed the effect by construction.
- **MONOTONE Q1:** a two-arm pool with non-overlapping losses (0.0003 vs 0.31).
- **PAPER2X2 position:** index arms at 0.68-0.96, path-integrated arms at 0.00-0.38.

When the manipulation itself causes the loss gap, the acc~loss fit is identified by the arm difference and the
residual is uninformative, not null. **How to apply:** before registering a loss-matched verdict, check that loss
varies within arms and overlaps across them. Otherwise the RAW contrast is primary and loss-matched is reported
as "uninformative". A bigger pool with intermediate-loss arms changes the verdict, so state the pool.
