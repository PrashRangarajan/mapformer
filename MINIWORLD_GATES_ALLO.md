# MiniWorld cross-cell-revisit task -- pre-flight gates (CPU, no training)

Env `MiniWorld-OneRoom-v0`, allocentric=False (raw 3-action macros), n_obs=16 (chance 1/16 = 0.0625), p_empty=0.5. 40 episodes/config at T=512.

Scored target = obs token at CROSS-CELL revisits (cell seen before and != previous cell).

## grid_size=8  (T=512, 2530 scored positions)

| gate | verdict | measured |
|---|---|---|
| G1 chance (ref) | PASS | 0.0625 = 1/16 |
| G2 marginal | PASS | all=0.419 (blank 0.42), non-blank=0.150 vs chance 0.062 |
| G3 copy-last-obs | PASS | 0.175 |
| G4 action n-gram | PASS | o1..5 0.427/0.398/0.388/0.374/0.363 (best 0.427 vs marg 0.419, non-blank best 0.140 vs 0.150) |
| G5 label mass | PASS | 0.124 of steps scored; 63.2/traj [35..85] |
| G6 revisit lag | PASS | median=49 p90=264 max=486; within8/16/32=0.02/0.21/0.40 |
| G7 oracle | PASS | 1.0000 over 2530 scored |

