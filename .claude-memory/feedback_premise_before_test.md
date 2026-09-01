---
name: feedback-premise-before-test
description: Check a mechanism's premise applies, prefer runtime knobs to retraining, and split hypotheses before testing them.
metadata:
  type: feedback
---

Three process errors from the loop/correction work, each of which cost real runs.

**1. Check the PREMISE applies before testing a mechanism.** I tested a
theta-refinement (Kalman-style) loop on Match-Query — where actions are CLEAN (no
drift to correct) and the query phase is BLIND (no observations to correct WITH).
Neither half of the premise held, and the repo ALREADY recorded that for the
sequence-axis version ("Match-Query (blind), nothing to correct with"). 16 runs to
replicate a known negative. **How to apply:** before running a mechanism, name the
condition it needs and verify the task supplies it.

**2. Check whether the knob is a RUNTIME argument before proposing a training
sweep.** I specified 12 training runs to test whether loop count drives OOD
degradation. Loop count is a forward-pass argument — the eval-only sweep took 90
seconds on existing checkpoints and is what actually found the mechanism (T=512
peaks at 2 passes, T=128 at 4). **How to apply:** ask "can I vary this at eval?"
first.

**3. Split a hypothesis before testing it.** I claimed "iteration compounds the
damage VIA residual-scale growth" as one statement, measured the residual (flat
across length → refuted), and retracted the WHOLE claim — then the iteration half
turned out right. **How to apply:** state separable parts separately so a failed
test kills only what it hits.

Also: **`train_hourglass_enwik8` saves metrics but NO checkpoints**, so no post-hoc
diagnostic on a trained language model is possible without retraining.

Related: [[feedback_convergence_first]], [[feedback_scheduler_and_measurement_traps]].
