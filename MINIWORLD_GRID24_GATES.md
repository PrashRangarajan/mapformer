# MiniWorld cross-cell-revisit task -- pre-flight gates (CPU, no training)

Env `MiniWorld-OneRoom-v0`, allocentric=False (raw 3-action macros), n_obs=16 (chance 1/16 = 0.0625), p_empty=0.5. 40 episodes/config at T=512.

Scored target = obs token at CROSS-CELL revisits (cell seen before and != previous cell).

## grid_size=24  (T=512, 2269 scored positions)

| gate | verdict | measured |
|---|---|---|
| G1 chance (ref) | PASS | 0.0625 = 1/16 |
| G2 marginal | PASS | all=0.509 (blank 0.51), non-blank=0.079 vs chance 0.062 |
| G3 copy-last-obs | PASS | 0.278 |
| G4 action n-gram | PASS | o1..5 0.488/0.483/0.462/0.413/0.419 (best 0.488 vs marg 0.509, non-blank best 0.072 vs 0.079) |
| G5 label mass | PASS | 0.111 of steps scored; 56.7/traj [33..102] |
| G6 revisit lag | PASS | median=38 p90=251 max=510; within8/16/32=0.01/0.26/0.47 |
| G7 oracle | PASS | 1.0000 over 2269 scored |

