# Sign-ablation probe: constraint integrity and action geometry

`nonneg` is a wiring check -- the constraint is enforced by construction, so
False means the wrong checkpoint was loaded. `oppose` is the ACTION_GEOMETRY
opposition score `||Delta(+x) + Delta(-x)|| / mean||Delta||`: 0 means opposite
actions cancel exactly, 2 means they are identical. A monotone code CANNOT
reach 0. `frac_neg` is the fraction of Delta entries below zero on a real
512-step stream.

| arm | nonneg | frac_neg | Delta range | oppose_x | oppose_y | \|cos(x,y)\| |
|---|---|---|---|---|---|---|
| `Signed_r4` | False | 0.503 | -2.468 .. +2.368 | 0.125 | 0.106 | 0.218 |
| `Abs_r4` | True | 0.000 | +0.000 .. +3.742 | 1.849 | 1.855 | 0.587 |
| `Pos_r4` | True | 0.000 | +0.000 .. +4.010 | 1.905 | 1.885 | 0.723 |
| `CARoPE_r4` | True | 0.000 | +0.028 .. +1.000 | 1.981 | 1.977 | 0.930 |
| `Vanilla_r4` | False | 0.512 | -2.967 .. +2.823 | 0.128 | 0.130 | 0.133 |

**Reading it.** If the signed arm's opposition scores are near 0 and the
constrained arms' are near 2, the mechanism is confirmed at the level of the
learned code, independently of accuracy. If the SIGNED arm's scores are also
far from 0, it never used the sign either, and any null in the accuracy table
says nothing about whether sign matters in principle.
