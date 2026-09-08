# MapPoPE at r=4, and the P4 repair

## Q-A: does r=4 help the best-measured arm?

`MapFormerWM_PoPE` defaults to `bottleneck_r=2`, so **every MapPoPE number in this
project is r=2** -- including the 0.994/0.970 that makes it the best arm measured on
the paper task. r=4 independently buys **+0.085 at T=1024 for 384 parameters** (8/8
seeds), and r=2's failure is a skewed basis rather than missing capacity
(opposition 0.495, |cos(N,E)| 0.783). The upgrade has never been applied.

**Prediction.** `MapPoPE_r4 - MapPoPE-Flat > 0` at T=512/1024, of order the rank
effect measured on the MapWM family. If it does NOT reproduce, the rank effect is
specific to the RoPE-style pairing and does not transfer to PoPE's one-frequency-
per-element layout -- which would itself be worth knowing, since the two differ in
`n_blocks` (64 vs 32).

Verified before launch, given that `_widen_to_d` silently reset the rank to 2 in a
neighbouring class until 2026-08-28: `MapPoPE_r4` has `w_in.out_features == 4`
(asserted at construction), `n_blocks == 64`, and 205,205 parameters against
`MapPoPE-Flat`'s 204,693 -- a difference of 512, which is what two extra rank rows
cost at this width. A rank that did not survive construction would have shown
identical counts, which is exactly how the earlier bug hid.

## P4 repair: does an explicit gate help r=2 more than r=4?

`GATED_PREREG.md` asked this and the batch could not answer it: `Vanilla_r2` was not
included, so the only available contrast was `Gated_r2 - Vanilla_r4`, which
confounds the gate with the rank. My error. This trains `Vanilla` (r=2) alongside
`Gated_r2` on both tasks.

**Prediction.** If the gate substitutes for the separator that the bottleneck
provides, then `Gated_r2 - Vanilla` should exceed `Gated_r4 - Vanilla_r4`, because
r=2 is the arm whose separator is impaired. The r=4 half comes from the gated batch;
each difference is within its own batch and the two batches share code and recipe,
but the comparison across them is stated as such rather than presented as a single
within-batch interaction.

**Prior, from what has already landed.** The gate buys nothing at r=4 on either
task, and its measured separation at r=2 is both weaker and far more variable
(2.55x, per-seed 0.92-5.30, against r=4's 4.16x, 3.33-5.44). So the honest prior is
that this fails too. It is run because the question was posed and left unanswerable,
not because the outcome is expected to be positive.
