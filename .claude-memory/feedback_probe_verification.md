---
name: feedback-probe-verification
description: Five ways an analysis probe returned a confident wrong answer this week, and the check that catches each.
metadata:
  type: feedback
---

Every one of these produced a clean-looking verdict that was wrong. **Why:** the
scripts are now more dangerous than the experiments -- a broken probe still prints
a table. **How to apply:** before trusting any probe output, run the matching check.

1. **Read the CODE, not its COMMENT.** The math note transcribed
   `model_baseline_rope.py`'s comment (canonical RoPE) while the line beneath
   computed `base^(-c/(n_b-1))`. Two agent checks were needed to catch it.
2. **Verify WHAT a probe measures.** The first learned-rank probe read the
   weight-norm magnitude `original0`, a (64,1) column, and reported "100% of energy
   in the top 2 singular values" -- true of any rank-1 object. Reconstruct
   parameterised weights explicitly (`original0 * original1/||original1||`).
3. **A replacement anchored "from X to end of file" eats the file.** A docstring
   edit deleted every class in `model_rank.py`. Re-import after editing a module.
4. **Post-hoc truncation is not a sufficiency test.** An unconstrained map could not
   be projected below rank ~16 (-0.576 at rank 2) while r=4 TRAINS fine.
   Trained-at-rank and truncated-to-rank are different objects.
5. **Check whether a "reproduction failure" is the paper's own reported result.** I
   presented Fig. 4's non-orthogonality as evidence r=2 was defective; the paper
   documents it and proposes a fix.

**Also: a verdict threshold that a result clears by three points is not a test.** A
context-destruction check used "fails if it keeps >50% of control accuracy"; the
leak kept 46-47% and it printed PASS. Threshold against the MEASURED FLOOR instead.

See [[feedback-convergence-first]] and [[feedback-scheduler-and-measurement-traps]].

**2026-09-05/06 additions.** Three more, all mine: a probe that varied two
coordinates at once so nothing could read as structured; a Toeplitz test, which is
the wrong invariant once increments are content-dependent (the right one is
insensitivity to tokens outside the interval); and a perturbation that was purely
REAL, so a test of the PHASE never fired and both arms looked identical. Also a
loss-matched analysis that regressed accuracy on its own eval NLL -- circular, and
it appeared to null an established effect. See [[verify_before_relaying]].
