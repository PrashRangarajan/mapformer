---
name: Check the training recipe before believing a task's ceiling
description: A task stuck at 0.415 for months was undertrained, not capability-limited. The recipe was worth more than any architecture effect ever measured on it.
metadata:
  type: feedback
---

The compositional task sat at **cross_nb 0.415** against a 0.072 floor and a 1.0
ceiling — the largest unclaimed headroom in the project, and long treated as the
task being hard.

Every published number on it was trained with `LinearLR(1.0 -> 0.0)` **from step
one** at lr 3e-4 for 50 epochs. Fixing that:

| arm | recipe | cross_nb |
|---|---|---|
| A | published: linear, 3e-4, 50ep | 0.354 |
| C | cosine, 1e-3, 150ep | **0.514** |

**+0.160, detectable, 7/8 seeds — larger than any architectural ingredient ever
measured on that task, including hierarchy's own +0.13.**

## Why this generalises

Standing rule 10 exists because LinearLR-from-step-one decays with no warmup and
cannot escape a plateau late; it once moved an arm **0.448 -> 0.990 on the same
task** and inverted a headline. The compositional trainer simply never got the fix
— it had **no `--schedule` flag at all**.

**Before attributing a low ceiling to capability, check the schedule, the LR and the
budget.** A trainer that predates a recipe fix is the first place to look, and
`grep -l LinearLR train_*.py` finds them in one command.

## One prediction of mine that failed here

I pre-registered that the better recipe would **compress seed variance** (it cut sd
3.5x on the torus), and wrote that a mean gain with unchanged variance would mean
something other than optimisation. **Variance roughly doubled** (0.070 -> 0.151).
So the compositional recipe effect is not the same phenomenon as the torus one, and
by my own criterion the mechanism is unidentified.

## The consequence to check every time

Every comparison previously made on that task was between arms **all** trained
badly. The hierarchy claim was re-measured because of this — see
[[project-hierarchy-negative]]. Ask: is the effect I am citing larger than the
recipe effect on the same task?
