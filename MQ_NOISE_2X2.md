# Does the filter pay where its premise holds?

Match-Query 128^2, T_explore=512, T_query=256, chance 0.0625.
4 arms x 2 noise levels x 8 seeds, one batch. Stochastic transitions apply
to the EXPLORE phase only, and evaluation uses the SAME noise as training.

Gated before training: every shortcut gate stays at its clean-task
level while drift rises 0 -> 13.05 cells at p=0.10.

## Accuracy (mean +/- sd)

| arm | p=0.0 | p=0.10 | drop |
|---|---|---|---|
| Vanilla | 0.505 +/- 0.263 | 0.215 +/- 0.063 | -0.290 |
| Level15 | 0.540 +/- 0.214 | 0.253 +/- 0.045 | -0.286 |
| Looped | 0.878 +/- 0.092 | 0.337 +/- 0.036 | -0.542 |
| Level15Looped | 0.737 +/- 0.169 | 0.312 +/- 0.044 | -0.425 |

## Convergence (re-measured here; recipe transfer across tasks is an
assumption, not a result)

| arm | p=0.0 final loss | p=0.10 final loss |
|---|---|---|
| Vanilla | 2.248  (0.615 – 3.635) | 3.823  (3.605 – 3.979) |
| Level15 | 1.673  (0.540 – 2.480) | 3.443  (3.298 – 3.669) |
| Looped | 0.534  (0.007 – 0.897) | 2.976  (2.727 – 3.137) |
| Level15Looped | 0.992  (0.209 – 1.593) | 3.101  (2.979 – 3.344) |

## Contrasts

| contrast | p=0.0 | p=0.10 | verdict at p=0.10 |
|---|---|---|---|
| Level15 - Vanilla <br><sub>THE PRIMARY: filter, no loop</sub> | +0.035 (sd 0.267, t +0.37, MDE 0.264, 5/8) | +0.038 (sd 0.069, t +1.56, MDE 0.068, 5/8) | UNMEASURED |
| Level15Looped - Looped <br><sub>filter, inside the loop</sub> | -0.141 (sd 0.156, t -2.56, MDE 0.155, 1/8) | -0.025 (sd 0.057, t -1.22, MDE 0.057, 4/8) | UNMEASURED |
| Looped - Vanilla <br><sub>loop, no filter</sub> | +0.373 (sd 0.253, t +4.17, MDE 0.251, 7/8) | +0.121 (sd 0.066, t +5.22, MDE 0.065, 8/8) | DETECTABLE POSITIVE |
| Level15Looped - Level15 <br><sub>loop, inside the filter</sub> | +0.197 (sd 0.159, t +3.50, MDE 0.158, 7/8) | +0.059 (sd 0.067, t +2.48, MDE 0.066, 6/8) | UNMEASURED |
| **INTERACTION** <br><sub>(L15Loop-Loop)-(L15-Vanilla)</sub> | **-0.176 (sd 0.343, t -1.45, MDE 0.340, 2/8)** | **-0.063 (sd 0.107, t -1.65, MDE 0.106, 3/8)** | **UNMEASURED** |

## Verdict against the pre-registration

**Primary.** Level15 - Vanilla is +0.035 at p=0 and +0.038 at p=0.10; the effect changes by **+0.003** as drift goes from 0 to 13 cells. At p=0.10 it is UNMEASURED (MDE 0.068, 5/8 seeds).

**The filter does not pay even here.** Drift is present and measurable, observations carry the correction signal, the task has headroom, and the contrast is still inside its noise floor. That is the sharpest test the InEKF has been given, and the 'stabilisation, not inference' reading survives it. Per rule 11 this is UNMEASURED rather than zero -- report the MDE beside it.

**Secondary.** Interaction at p=0.10: -0.063 (sd 0.107, t -1.65, MDE 0.106, 3/8) -> UNMEASURED. Last night's clean-torus 2x2 found no complementarity on a task with no premise; this re-asks it on one that has a premise.

## Scope

One task, one map size, two noise levels, n=8, one loop count. Two noise levels are two points -- a dose-response claim needs a third. Query transitions stay clean by design: scoring is keyed on the TRUE cell, so noisy query transitions would make the answer unknowable rather than the task harder.
