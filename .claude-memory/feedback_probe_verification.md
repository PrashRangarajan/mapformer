---
name: feedback-probe-verification
description: Probes, agents, web summarisers and my own scripts return confident wrong answers; verify what each measures before relaying or acting (CLAUDE.md rule 9, parts of 5 and 7).
metadata:
  type: feedback
---

**Why:** the output is always well-formed and plausible, so nothing signals the error; a broken probe
still prints a table. **How to apply:** before relaying a claim that would change what you do, name
the one-command check that would falsify it and run it. For a paper, read the text; for a probe, run
it on a case whose answer you know; for a correlation, check the variables are not one quantity twice.

## Probes and scripts

1. **Read the CODE, not its COMMENT.** The math note transcribed `model_baseline_rope.py`'s comment
   (canonical RoPE) while the line beneath computed `base^(-c/(n_b-1))`. Two agent checks to catch.
2. **Verify WHAT a probe measures.** The first learned-rank probe read the weight-norm magnitude
   `original0`, a (64,1) column, and reported "100% of energy in the top 2 SVs" -- true of any rank-1
   object. Reconstruct parameterised weights explicitly (`original0 * original1/||original1||`).
3. **A replacement anchored "from X to end of file" eats the file** (a docstring edit deleted every
   class in `model_rank.py`). Re-import after editing a module.
4. **Post-hoc truncation is not a sufficiency test.** An unconstrained map could not be projected
   below rank ~16 (-0.576 at rank 2) while r=4 TRAINS fine.
5. **Check whether a "reproduction failure" is the paper's own result** (Fig. 4's non-orthogonality
   is in its caption, with a proposed fix).
6. **Wrong invariants and dead perturbations**: a probe that varied two coordinates at once; a
   Toeplitz test, wrong once increments are content-dependent (the right invariant is insensitivity to
   tokens outside the interval); a purely REAL perturbation, so a PHASE test never fired.
7. **Circular loss-matching**: accuracy regressed on its own eval NLL (both read the same softmax)
   appeared to null an established effect. Training loss lives in the checkpoint's `losses`, not the
   eval JSON's third field.
8. **Case-sensitive greps**: `-i rope` matches `p-ROPE-rty` (I reported "RoPE 17 times" in a review
   where it is zero); `grep -ic Undefined` misses LaTeX's `undefined`.
9. **My own verification script printed the sign table BACKWARDS** (inconsistent branches) and a
   transpose gave 5e-01 where the truth was 1.5e-16. Caught only by running it against a claim I
   already believed and blaming the script. Write it, then check it against something you know.
10. **Thresholds and verdict cells**: a context-destruction check "fails if >50% of control" passed a
   leak at 46-47%; threshold against the measured floor. A registered verdict rule can be
   unmeasurable (sign ablation asked for a deficit where the baseline is 1.000 +/- 0.000): check the
   cell could have gone the other way.
11. **Commit the script for every number** (2026-09-20): four Dyck/PoPE figures had none and two did
   not reproduce; the surviving CROSS claim had no analysis script until `analyze_cross.py`. Report a
   REGISTERED readout even when the verdict looks obvious (a floor check, a learned-rate readout and a
   control contrast were each skipped, and each mattered).

## Intermediaries

- **A WebFetch summariser** called GRAPE "purely index-driven, no content-dependence"; the PDF has
  content-gated path-integral forms and a cumulative phase `Phi_t = sum omega_l`. pdftotext the saved
  PDF and grep it (WebFetch names the local path); the corpus is in `papers/txt/`.
- **Review agents** were largely right but mis-stated details; one cited a MapFormer v4 r=1 table
  absent from the locally held v3. Relay only what you verified; mark the rest unverified.

Related: [[feedback-convergence-first]], [[feedback-scheduler-and-measurement-traps]].
