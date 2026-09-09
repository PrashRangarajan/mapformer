# Fixing EM's collapse: the 2x2 of both init pathologies

`MINIGRID_EM.md` measured EM's failure precisely. It is **not a capability deficit**:
EM matches WM on 7 of 8 seeds and collapses on one. Symmetrically trimming each arm's
worst seed takes `EM - WM` from `-0.019` to `-0.003` at r=4, and the
worst-to-second-worst gap is **0.144 for EM_r4 against 0.007 for WM_r4**.

Two init pathologies are candidates, and no previous batch could separate them:

| pathology | mechanism | fix | prior evidence |
|---|---|---|---|
| separate `q0/k0` | `A_P` is not peaked at zero displacement at init, so the AND-gate multiplies content by a **random position mask** and the gradient dies | shared `p0` (paper eq. 3) | **+0.358** on Match-Query, 3/3 |
| `r=2` | skewed action basis, \|cos(N,E)\| 0.78 vs 0.17 at r=4 | `r=4` | +0.085 torus, 8/8 |

`VanillaEM_P0` fixes only the first, `VanillaEM_r4` only the second.
**`VanillaEM_P0_r4` did not exist** and is built for this batch.

## Arms (8 seeds, ONE batch, same recipe as MINIGRID_EM)

`Vanilla_r4` (reference, the best arm measured there) + the full 2x2:
`VanillaEM` (neither fix) / `VanillaEM_r4` (rank only) / `VanillaEM_P0` (origin only)
/ `VanillaEM_P0_r4` (both). Parameter spread **0.08%**. No `--fast-attn`.

## Predictions

**P1 -- the fix works, and it is a FLOOR fix.** `VanillaEM_P0_r4` has **no collapsed
seed**: worst-to-second-worst gap below `0.03` (WM_r4's is 0.007; EM_r4's is 0.144),
and sd falling from `0.056` toward WM_r4's `0.015`. *Refuted by* a seed still
sitting 0.10+ below the pack, which would mean the collapse is not the origin vector.

**P2 -- the mean gain should be SMALL, and that is the point.** The trimmed means
were already equal (EM 0.823 vs WM 0.825), so removing the collapse should buy about
its own contribution, **~0.019**, and no more. *A larger mean gain would mean shared
`p0` does something beyond preventing collapse*, which nothing here predicts and
which I would have to explain rather than claim.

**P3 -- which fix carries it.** Shared `p0` should help MORE at r=2 than at r=4,
because r=2 carries both pathologies at once. Interaction predicted negative.

## Reading rule, fixed in advance

**Report the per-seed minimum and the worst-to-second-worst gap first, before any
mean.** This batch exists because a mean reported a bimodal arm as uniformly worse.
