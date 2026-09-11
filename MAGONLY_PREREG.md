# Pre-registration: MagOnly -- is D1 phase freedom, or parameterisation?

Written before any `EMDoF_magonly` checkpoint exists. Tier 1, item 1 of the plan
following `AUDIT_2026-09-10.md`.

## Why this is the test that matters now

Phase freedom (`AlignFree - AlignLock` = **+0.165** at n=24, +0.173 on fresh seeds
alone) is the only result from the EM/WM line that survived the audit. The audit's
finding 8 is that the two arms also differ in **optimiser treatment**: `AlignLock`
stores magnitude as a scale `s` initialised at 1.0, which Adam moves ~50x slower in
relative terms than `AlignFree`'s raw `k0` entries (~0.02), and weight decay alone
predicts most of `s`'s drift (0.674 against a measured median ~0.70). D1 could be
parameterisation rather than phase freedom.

## The control

`EMDoF_magonly` (`model_em_magonly.py`) stores `k0` as a free vector `u` exactly as
`AlignFree` does, initialised equal to `q0`, and uses only each block's norm with the
phase pinned to `q0`: `k0_i = |u_i| q0_i / |q0_i|`. **Verified at construction:**
parameter count equal to `AlignFree` (222,489); every weight identical at init under
the same seed; logits agree to 2.4e-7, i.e. the **same function at init**; `rho = 1`;
both coordinates of every block receive gradient (no dead parameters).

So `AlignFree - MagOnly` differs in phase freedom ALONE -- same init function, same
parameter count, same optimiser scale.

## Reuse of existing arms, licensed by measurement

`AlignFree`, `AlignLock` and `VanillaEM_P0_r4` at seeds 0-23 already exist in
`runs/dof/recency`. The recency pipeline is bitwise deterministic (audit finding 10:
16/16 identical checkpoints across batches) and no file on those arms' code path has
changed since the DOF launch (`git log 457d885..HEAD`: empty for model.py,
model_em_dof.py, model_em_fixed.py, train_recency.py, environment_recency.py;
train_variant.py changed by registration lines only). **This is checked, not
assumed:** the batch re-trains `AlignFree` s0 and `VanillaEM_P0_r4` s0 into a
separate directory and compares them bitwise to the stored checkpoints. If either
differs, every comparator is re-trained in-batch before anything is read.

Only `EMDoF_magonly` seeds 0-23 are new: 24 runs plus 2 determinism controls.

## Predictions

**M1 (decisive).** `AlignFree - MagOnly` at n=24, accuracy at T=1024, is **positive
and clears its MDE** (expected ~0.09 from D1's sd 0.155). Phase freedom survives the
matched-optimiser control. *Falsified if* inside MDE -- then D1 was, at least in
part, parameterisation.

**M2 (complementarity, arithmetic).** `AlignFree - AlignLock` (+0.165) =
`(AlignFree - MagOnly)` + `(MagOnly - AlignLock)`. If M1 holds, MagOnly should sit
near AlignLock and the second term near zero; if M1 fails, the second term carries
the effect. Registered so the decomposition is read one way only.

**M3 (exploratory).** `MagOnly - P0`: magnitude freedom when the optimiser can
actually move it. D5 found nothing, but for a parameter that barely moved. No
registered direction.

**M4 (manipulation check).** MagOnly's k-side block magnitudes move by far more than
weight decay alone predicts. If they do not, the control failed as a control and M3
is uninformative.

**M5 (replication guard).** Seeds 8-23 alone reproduce M1's sign. *Falsified if*
the sign flips -- then M1 is carried by seeds 0-7 and is not quoted.

**M6 (registered so it is not read as news).** `r(final loss, acc)` will again be
~-0.98 and the loss-matched residuals ~0. After audit finding 7 that is not evidence
for "optimisation" on its own; the optimisation reading rests on the existence
argument, which item 2 of Tier 1 (the warm-start) tests directly.
