# Does the filter pay where its premise holds?

Match-Query 128^2, T_explore=512, T_query=256, chance 0.0625.
4 arms x 2 noise levels x 8 seeds, one batch. Stochastic transitions apply
to the EXPLORE phase only, and evaluation uses the SAME noise as training.

Gated before training: every shortcut gate stays at its clean-task
level while drift rises 0 -> 13.05 cells at p=0.10.

## Accuracy (mean +/- sd)

| arm | p=0.0 | p=0.10 | drop |
|---|---|---|---|
| Vanilla | 0.644 +/- 0.261 | 0.308 +/- 0.044 | -0.336 |
| Level15 | 0.790 +/- 0.173 | 0.313 +/- 0.072 | -0.477 |
| Looped | 0.789 +/- 0.095 | 0.365 +/- 0.040 | -0.425 |
| Level15Looped | 0.855 +/- 0.161 | 0.312 +/- 0.053 | -0.543 |

## Convergence (re-measured here; recipe transfer across tasks is an
assumption, not a result)

| arm | p=0.0 final loss | p=0.10 final loss |
|---|---|---|
| Vanilla | 1.492  (0.005 – 2.800) | 3.284  (3.130 – 3.442) |
| Level15 | 0.786  (0.367 – 1.834) | 3.107  (2.978 – 3.346) |
| Looped | 0.668  (0.439 – 1.091) | 2.784  (2.664 – 2.849) |
| Level15Looped | 0.505  (0.030 – 1.587) | 2.900  (2.543 – 3.127) |

## Contrasts

| contrast | p=0.0 | p=0.10 | verdict at p=0.10 |
|---|---|---|---|
| Level15 - Vanilla <br><sub>THE PRIMARY: filter, no loop</sub> | +0.146 (sd 0.329, t +1.26, MDE 0.326, 5/8) | +0.005 (sd 0.065, t +0.23, MDE 0.065, 5/8) | UNMEASURED |
| Level15Looped - Looped <br><sub>filter, inside the loop</sub> | +0.066 (sd 0.116, t +1.61, MDE 0.114, 4/8) | -0.053 (sd 0.058, t -2.57, MDE 0.058, 2/8) | UNMEASURED |
| Looped - Vanilla <br><sub>loop, no filter</sub> | +0.146 (sd 0.312, t +1.32, MDE 0.309, 6/8) | +0.057 (sd 0.026, t +6.31, MDE 0.025, 8/8) | DETECTABLE POSITIVE |
| Level15Looped - Level15 <br><sub>loop, inside the filter</sub> | +0.065 (sd 0.252, t +0.73, MDE 0.250, 5/8) | -0.001 (sd 0.077, t -0.04, MDE 0.076, 4/8) | UNMEASURED |
| **INTERACTION** <br><sub>(L15Loop-Loop)-(L15-Vanilla)</sub> | **-0.081 (sd 0.307, t -0.74, MDE 0.304, 4/8)** | **-0.058 (sd 0.080, t -2.05, MDE 0.079, 2/8)** | **UNMEASURED** |

## Verdict against the pre-registration

**Primary.** Level15 - Vanilla is +0.146 at p=0 and +0.005 at p=0.10; the effect changes by **-0.141** as drift goes from 0 to 13 cells. At p=0.10 it is UNMEASURED (MDE 0.065, 5/8 seeds).

**The filter does not pay even here.** Drift is present and measurable, observations carry the correction signal, the task has headroom, and the contrast is still inside its noise floor. That is the sharpest test the InEKF has been given, and the 'stabilisation, not inference' reading survives it. Per rule 11 this is UNMEASURED rather than zero -- report the MDE beside it.

**Secondary.** Interaction at p=0.10: -0.058 (sd 0.080, t -2.05, MDE 0.079, 2/8) -> UNMEASURED. Last night's clean-torus 2x2 found no complementarity on a task with no premise; this re-asks it on one that has a premise.

## Scope

One task, one map size, two noise levels, n=8, one loop count. Two noise levels are two points -- a dose-response claim needs a third. Query transitions stay clean by design: scoring is keyed on the TRUE cell, so noisy query transitions would make the answer unknowable rather than the task harder.
