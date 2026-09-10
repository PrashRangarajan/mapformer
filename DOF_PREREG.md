# Pre-registration: phase degrees of freedom in the position kernel

Written before any checkpoint in `runs/dof/` exists. Third mechanism proposed for
one observation, so the bar is higher than usual and the confound that killed the
first two is designed out rather than argued away.

## What is being explained, and what has already failed

Single `p_0` beats separate `q_0`/`k_0` on three map tasks (+0.089 / +0.167 /
+0.358, plus a removed collapse on MiniGrid) and **loses on the one clock task**
(-0.237, `RECENCY_EM_RESULTS.md`).

- **Explanation 1, "separate init is a pathology":** refuted -- it is task-dependent,
  not a defect.
- **Explanation 2, coherence `rho`:** refuted by N5. Coherence helps on BOTH tasks
  with the same sign (+0.292 torus 8/8, +0.180 recency), so it cannot produce an
  inversion. N5 also showed the axis is `|rho|`, since `kappa`'s sign is a gauge
  the content branch absorbs.
- **Explanation 3, this one:** the forms differ in how many PHASES they can move.
  `phi_i == 0` forever for single `p_0` (0 phase DOF); free for the separate form
  (`n_b` phase DOF). A map task wants a matched filter, which `phi = 0` already is,
  so freedom is worth nothing; a clock task must reshape `kappa`, so it is worth a
  lot.

## The confound that has to be designed out

Comparing single `p_0` with a randomly-initialised separate form varies coherence
AND freedom together -- exactly what N5 did with its frozen arms. Nothing can be
attributed from it. The discriminator below holds **initial coherence fixed at
`rho = 1`** and varies **only** phase freedom:

| arm | init `rho` | phase DOF | magnitudes | params |
|---|---|---|---|---|
| `VanillaEM_P0_r4` | 1 (always) | **0** | shared | 222,361 |
| `VanillaEM_r4` | ~0 | `n_b` | free | 222,489 |
| `EMDoF_alignfree` | **1** | **`n_b`** | free | 222,489 |
| `EMDoF_alignlock` | **1** (always) | **0** | free per block | 222,425 |

`AlignLock` sets `k_0i = s_i * q_0i` with `s_i` trainable, so it is NOT a freeze --
the spectral weights `a_i` can still be learned and only the phases cannot. N5's
freeze removed both and was catastrophic; this removes one.

`AlignFree - AlignLock` differs in phase freedom alone, at matched initial kernel
and within 64 parameters (0.03%; `AlignLock` carries `n_heads * n_blocks` scales in
place of a second vector). The alternative -- a free 2-vector whose angle is
discarded -- would count-match exactly but create dead parameters, which is worse.

**Verified before launch** (40 steps at lr 1e-2 on recency): `AlignLock` holds
`rho` at exactly 1.000000 while `AlignFree` drifts to **0.21**. So the constraint
binds, and on the clock task the optimiser actively moves phases AWAY from
alignment when allowed to. That drift is a stated prior for D1, not a later
discovery.

## Arms and batches

4 arms x 8 seeds x 2 tasks, each task ONE batch (rule 3) -- including fresh
`VanillaEM_P0_r4` / `VanillaEM_r4`, which are not read from `runs/recency_em`.
**Recency runs FIRST** this time: it carries the discriminator and is ~3x faster
per run than the torus.

Recipes unchanged: recency `k_max=64`, `T=1024`, eval {1024, 2048}, 300 ep cosine,
lr 1e-3, no `--fast-attn`; torus 300 ep, 98x128, `T=128`, eval {128, 512, 1024}.

## Predictions

**D1 -- the discriminator.** `AlignFree - AlignLock` on **recency** is positive and
clears its MDE. Phase freedom, at matched initial coherence, is what the clock task
needs. *Falsified if* inside MDE or negative.

**D2 -- the map control.** `AlignFree - AlignLock` on the **torus** is inside its
MDE. *Falsified if* detectable there.

**D3 -- the inversion, in DOF form.** `(AlignFree - AlignLock)_recency -
(same)_torus > 0`, clearing its MDE. This is the claim; D1 and D2 are its halves.

**D4 -- reproduce the original in-batch.** `VanillaEM_r4 - VanillaEM_P0_r4` on
recency is positive (published -0.237 the other way round, i.e. +0.237 here).
*Falsified if* it does not reproduce -- in which case nothing else in this batch
means anything.

**D5 -- the extra vector buys nothing without phase freedom.** `AlignLock -
VanillaEM_P0_r4` on recency is inside its MDE, despite `AlignLock` having two
vectors and free per-block magnitudes. *Falsified if* detectable.

## Power and ceilings, before the fact

Recency EM arms ran sd 0.081-0.126 -> **MDE 0.080-0.125** at n=8. The effect D1
must clear is of order the 0.237 it is meant to explain, so recency is adequately
powered. The interaction D3 adds variances: expect **MDE ~0.15**.

**Torus ceiling, stated because this trap has now cost two predictions.** Healthy
EM arms sit at 0.94-0.96 at T=1024 with sd ~0.025-0.030, leaving ~0.04 of headroom
and an **MDE near 0.030**. D2 is therefore a weak null: it can only rule out
effects larger than ~0.03. It is NOT evidence of exact equality, and will be
reported as "unmeasured above 0.03" rather than as a null. There IS real dynamic
range on the torus when the kernel is wrong (N5's `|rho|=0` arms scored 0.64-0.68),
so a large D2 effect would be visible if it existed. T=1024 is the readout; T=128
is at ceiling for every healthy arm.
