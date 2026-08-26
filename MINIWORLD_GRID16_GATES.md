# MiniWorld cross-cell-revisit task -- pre-flight gates (CPU, no training)

Env `MiniWorld-OneRoom-v0`, allocentric=False (raw 3-action macros), n_obs=16 (chance 1/16 = 0.0625), p_empty=0.5. 40 episodes/config at T=512.

Scored target = obs token at CROSS-CELL revisits (cell seen before and != previous cell).

## grid_size=16  (T=512, 2392 scored positions)

| gate | verdict | measured |
|---|---|---|
| G1 chance (ref) | PASS | 0.0625 = 1/16 |
| G2 marginal | PASS | all=0.496 (blank 0.50), non-blank=0.073 vs chance 0.062 |
| G3 copy-last-obs | PASS | 0.268 |
| G4 action n-gram | PASS | o1..5 0.495/0.485/0.446/0.428/0.424 (best 0.495 vs marg 0.496, non-blank best 0.065 vs 0.073) |
| G5 label mass | PASS | 0.117 of steps scored; 59.8/traj [36..90] |
| G6 revisit lag | PASS | median=43 p90=236 max=477; within8/16/32=0.01/0.24/0.44 |
| G7 oracle | PASS | 1.0000 over 2392 scored |

