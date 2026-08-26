# MiniWorld cross-cell-revisit task -- pre-flight gates (CPU, no training)

Env `MiniWorld-OneRoom-v0`, allocentric=False (raw 3-action macros), n_obs=16 (chance 1/16 = 0.0625), p_empty=0.5. 40 episodes/config at T=512.

Scored target = obs token at CROSS-CELL revisits (cell seen before and != previous cell).

## grid_size=8  (T=512, 2491 scored positions)

| gate | verdict | measured |
|---|---|---|
| G1 chance (ref) | PASS | 0.0625 = 1/16 |
| G2 marginal | PASS | all=0.507 (blank 0.51), non-blank=0.077 vs chance 0.062 |
| G3 copy-last-obs | PASS | 0.269 |
| G4 action n-gram | PASS | o1..5 0.512/0.508/0.507/0.490/0.470 (best 0.512 vs marg 0.507, non-blank best 0.049 vs 0.077) |
| G5 label mass | PASS | 0.122 of steps scored; 62.3/traj [31..88] |
| G6 revisit lag | PASS | median=47 p90=240 max=508; within8/16/32=0.01/0.22/0.43 |
| G7 oracle | PASS | 1.0000 over 2491 scored |

