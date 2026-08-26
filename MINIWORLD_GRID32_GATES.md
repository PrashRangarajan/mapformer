# MiniWorld cross-cell-revisit task -- pre-flight gates (CPU, no training)

Env `MiniWorld-OneRoom-v0`, allocentric=False (raw 3-action macros), n_obs=16 (chance 1/16 = 0.0625), p_empty=0.5. 40 episodes/config at T=512.

Scored target = obs token at CROSS-CELL revisits (cell seen before and != previous cell).

## grid_size=32  (T=512, 2015 scored positions)

| gate | verdict | measured |
|---|---|---|
| G1 chance (ref) | PASS | 0.0625 = 1/16 |
| G2 marginal | PASS | all=0.481 (blank 0.48), non-blank=0.077 vs chance 0.062 |
| G3 copy-last-obs | PASS | 0.256 |
| G4 action n-gram | PASS | o1..5 0.483/0.481/0.430/0.413/0.423 (best 0.483 vs marg 0.481, non-blank best 0.055 vs 0.077) |
| G5 label mass | PASS | 0.098 of steps scored; 50.4/traj [25..81] |
| G6 revisit lag | PASS | median=33 p90=244 max=491; within8/16/32=0.01/0.27/0.50 |
| G7 oracle | PASS | 1.0000 over 2015 scored |

