# Does an unconstrained angle map rediscover the rank-2 bottleneck?

MapFormer's `W_Delta = W_out W_in` is rank 2 **by construction**, and the
paper's reason is structural: `r` is the dimensionality of the action
space, so on a 2D grid `Delta_in` is the displacement vector itself.
Selective RoPE's `W_omega` is full rank -- each of the H*n_b channels can
be an independent function of the token -- so on this task it would have
to learn the bias rather than be given it.

Effective rank of the learned map, torus checkpoints, 8 seeds:

| arm | matrix | top-2 energy | participation ratio | max possible |
|---|---|---|---|---|
| NoBottleneck | W_omega (64, 128) | 0.388 | 8.35 | 64 |
| SRoPEGen | W_omega (64, 128) | 0.432 | 7.49 | 64 |
| GateAngle | W_omega (64, 128) | 1.000 | 1.58 | 64 |
| ConvAngle | W_omega (64, 128) | 1.000 | 1.18 | 64 |
| Vanilla (r=2, by construction) | W_out W_in (64, 128) | 1.000 | 2.00 | 2 |

## Verdict

**The bias is NOT rediscovered.** Top-2 energy is only 43.2% and the participation ratio is 8.35, so the unconstrained map spreads across many directions. Either the extra directions carry something real -- which would explain the torus win and contradict the displacement-dimensionality story -- or they are unused capacity the optimiser never pruned. The two are distinguishable by whether zeroing the tail costs accuracy.

Inference only. Participation ratio is `(sum s^2)^2 / sum s^4`, a continuous effective rank: it equals k for k equal singular values and 1 for a rank-1 map.
