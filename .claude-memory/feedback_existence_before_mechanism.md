---
name: feedback-existence-before-mechanism
description: Method lessons of the EM/WM line: construct a solution before explaining a deficit, then test whether it holds, at the right scale; same-seed agreement is determinism; check for gauges.
metadata:
  type: feedback
---

The EM/WM line (2026-09-09..11) went through three wrong mechanisms before one survived.
Each rule below would have saved a batch. Numbers 27-33 match `CLAUDE.md`. Full account:
`EM_WM_STATE.md` Sec 7.

**Why:** a composition theorem (Thm 3) was built to explain EM's recency deficit and published
with a corollary. A 30-line construction killed it: a single-`p0` EM kernel solves recency
exactly once the query token rewinds the count. No training run could have shown that.

**How to apply:**

1. **Existence before mechanism (29).** Before explaining a deficit as a function-class limit,
   try to construct a solution inside that class.
2. **Existence, then stability (32).** Warm-start the constructed solution twice, FROZEN and
   TRAINABLE. Frozen tells you it is representable (1.000 here); trainable tells you whether
   training holds it.
3. **Install at the scale training would use (33).** A rewind stored at a 0.0156-per-step
   weight scale tested whether Adam can erode small weights. Installed 8x larger, the same
   Delta held 0.941 against 0.642.
4. **A parameterisation change is an optimiser change (31).** A scale initialised at 1.0
   moves ~50x slower under Adam than a 0.02 vector. Weight decay alone explained most of a
   "learned" magnitude. Build a control that shares the optimiser treatment (MagOnly).
5. **A same-seed rerun is determinism, not replication (27).** "+0.237 reproduced to three
   decimals" was 16/16 bitwise-identical checkpoints; n=24 gave +0.128. Always report the
   FRESH seeds alone beside a pooled estimate. The first eight seeds overestimated twice.
6. **Check for a sign or scale GAUGE before registering a contrast (28).** `rho = +1` vs `-1`
   had expectation zero, because the learned content branch absorbs the kernel's sign.
7. **Report the registered primary readout even when the verdict is obvious (30).**
8. **Loss-matching conditions on a mediator.** A zero loss-matched residual cannot tell
   optimisation from representation on its own; the existence argument does.
9. **An interaction must subtract like with like.** D3 subtracted OOD accuracy from
   in-distribution accuracy.
10. **Record the mechanism's state in every checkpoint, every epoch.** "Destroyed early" was
    inferred from a loss curve, and the recordings refuted it. An exhaustiveness claim needs
    every trainable route enumerated: a probe read one coordinate of a two-coordinate code.
11. **Say when a theory is a retrodiction.** THEORY_KERNEL.md was written after the batch it
    explains, and that batch's pre-registration had predicted the opposite sign.

Related: [[feedback-convergence-first]], [[feedback-probe-verification]], [[em-vs-wm-mechanism]].
