# Sign-ablation probe: constraint integrity and action geometry

`nonneg` is a wiring check -- the constraint is enforced by construction, so
False means the wrong checkpoint was loaded. `oppose` is the ACTION_GEOMETRY
opposition score `||Delta(+x) + Delta(-x)|| / mean||Delta||`: 0 means opposite
actions cancel exactly, 2 means they are identical. A monotone code CANNOT
reach 0. `frac_neg` is the fraction of Delta entries below zero on a real
512-step stream.

| arm | nonneg | frac_neg | Delta range | oppose_x | oppose_y | \|cos(x,y)\| |
|---|---|---|---|---|---|---|
| `Signed_r4` | False | 0.500 | -3.322 .. +3.249 | 0.062 | 0.056 | 0.125 |

**Reading it.** If the signed arm's opposition scores are near 0 and the
constrained arms' are near 2, the mechanism is confirmed at the level of the
learned code, independently of accuracy. If the SIGNED arm's scores are also
far from 0, it never used the sign either, and any null in the accuracy table
says nothing about whether sign matters in principle.
