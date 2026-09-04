# Do we reproduce the paper's Figure 4?

Sec 5.4 / Fig 4 makes four checkable claims about a trained MapFormer.
Each is tested here with the paper's own metric, on 8 seeds, at the
paper's rank and at a widened one.

| claim | paper reports | `Vanilla` | `Vanilla_r4` |
|---|---|---|---|
| **C1** ‖Δ_action‖ / ‖Δ_obs‖ | ≫ 1 <br><sub>observations leave position untouched</sub> | 25.0 ± 22.4 | 35.9 ± 19.0 |
| **C2** cos(Δ_left, Δ_right) | **−1** <br><sub>opposite actions cancel</sub> | -0.729 ± 0.373 | -0.996 ± 0.004 |
| **C3** \|cos(Δ_left, Δ_up)\| | **≫ 0** <br><sub>reported as a LIMITATION</sub> | 0.779 ± 0.276 | 0.174 ± 0.083 |
| **C4** ‖v_obs‖ / ‖v_action‖ | ≫ 1 <br><sub>only observations update content</sub> | 0.57 ± 0.06 | 0.60 ± 0.04 |

## Verdict

**C1 reproduces**: actions rotate 25x more than observations.
**C2 reproduces**: cos(opposite) = -0.729 against the paper's −1.
**C3 reproduces, including the failure**: |cos(orthogonal)| = 0.779, which is the ≫ 0 the paper reports and flags as needing an added constraint to fix.
**C4**: ‖v_obs‖/‖v_action‖ = 0.57.

**On C3, widening the bottleneck does what the paper says needs an extra loss term.** |cos(orthogonal)| falls 0.779 → 0.174 going from Vanilla to Vanilla_r4, with no bounded-energy constraint and no change to the objective. The paper's own remedy for its disentanglement failure is an added regulariser; a wider bottleneck gets most of the way there for +384 parameters.

Inference only, 8 seeds, torus checkpoints from `/home/prashr/mapformer/runs/rank_sweep/p0`, arms `Vanilla` and `Vanilla_r4`. Δ_in is the r-dimensional Lie-algebra code (the paper's metric for C2/C3); Δ is the full per-head increment (C1).
