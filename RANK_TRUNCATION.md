# Are the extra directions load-bearing? (NoBottleneck, trained weights)

The unconstrained angle map does not collapse to rank 2 (effective rank
7.5-8.4, top-2 energy 39-43%). This truncates the TRAINED weight by SVD
and re-evaluates -- no retraining, so it isolates what the spectrum is
actually used for. MapFormer's own map is rank 2 by construction.

| rank kept | T=128 | T=512 |
|---|---|---|
| 1 | 0.375 ± 0.078 | 0.378 ± 0.072 |
| 2 | 0.418 ± 0.142 | 0.392 ± 0.124 |
| 3 | 0.485 ± 0.033 | 0.445 ± 0.031 |
| 4 | 0.594 ± 0.212 | 0.539 ± 0.184 |
| 8 | 0.877 ± 0.199 | 0.810 ± 0.191 |
| 16 | 0.992 ± 0.016 | 0.926 ± 0.028 |
| 64 (full) | 0.994 ± 0.015 | 0.966 ± 0.020 |

## Verdict

Truncating to rank 2 costs: +0.576 at T=128, +0.574 at T=512.

**The extra directions are load-bearing IN THIS MODEL -- which is not the same as
saying rank 2 is insufficient for the task, and the first draft of this verdict
made that slide.**

What the numbers show is that a map trained *without* the constraint distributes
its solution across roughly 16 directions and cannot be projected back to 2
afterwards. What they do not show is that 2 dimensions are inadequate --
`Vanilla`, which is rank 2 **by construction**, reaches **0.944 at T=512** on the
same task. So a rank-2 solution exists and is found routinely; the unconstrained
model simply does not land on one, and post-hoc truncation is not a test of
sufficiency. The analogy is pruning: that a dense network breaks when pruned to 2%
says little about whether a network trained sparse at 2% would work.

**So MapFormer's displacement-dimensionality justification survives this test.**
Rank 2 suffices, demonstrably.

What remains genuinely open is the +0.022 that the unconstrained arm gains over
rank 2 at T=512 (0.966 vs 0.944). Two candidates, which this probe cannot
separate because it only tests projection after the fact:

  (a) OPTIMISATION -- a full-rank map is better conditioned, so a good solution is
      easier to reach even though a rank-2 one exists;
  (b) the extra directions carry a little real signal that a rank-2 map cannot
      express, contra the sufficiency argument.

**The rank sweep now running separates them.** If `Vanilla_r16` matches
`NoBottleneck`, the axis is the rank available *during training* and (b) is live.
If `Vanilla_r16` matches `Vanilla_r2`, rank is not the axis at all and (a) is what
is left -- which is the same conclusion the gate probe pointed at.

Inference only, 8 seeds, held-out map. Truncation is exact SVD projection of the trained weight; the bias is left untouched.
