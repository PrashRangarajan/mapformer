# T3GEN_PREREG -- is the per-token phase general, and is the freedom optional?

T3 (`T3_RESULTS.md`) removed MapPoPE's length-extrapolation collapse on Bach Chorales (4.616 ->
0.733 at 2-4x, inert twin flat) but its predicted cost to pure indexing never appeared, so the
trade-off corollary was withdrawn and replaced by a POST-HOC reading: the phase is additive and
zero-initialised, so a model that needs a pure positional kernel can decline it. Both the claim's
generality and that reading are untested. Three batches, all pre-registered here.

## G1 -- Dyck-2 (8 seeds, the recipe of `DYCK_PREREG.md`, 1 layer, 1 head)

Arms `MapPoPE_T3`, `MapPoPE_T3inert` against the existing `MapPoPE-1L_r2` (0.988 at the training
cell, 0.927 at L128 D12, already the best arm there).

**This is the case where the account predicts LITTLE**: Dyck's increments cancel, so the accumulator
is bounded and `S_t - S_s` never leaves its trained range -- there is nothing for a compensating
phase to absorb. Registered: T3 - MapPoPE at L128 D12 is SMALL (inside its MDE), much smaller than
the -3.883 seen on music. **A large gain here would mean the phase helps for some reason other than
absorbing an out-of-range accumulator, and the account is incomplete.**

## G2 -- torus paper task (8 seeds, 50 epochs cosine, the PAPERTASK recipe)

Arms `MapPoPE_T3`, `MapPoPE_T3inert`, `MapPoPE-Flat`, evaluated at the training length and at 4x and
8x. The torus accumulator is a MAP (opposite actions cancel; measured alpha ~0.5 in
`ACCUMULATOR.md`), so the account again predicts a small effect at training length. At 8x the
accumulator is bounded but the TASK is harder, so no direction is registered for the long evaluation
-- it is reported as measured.

## G3 -- forced phase (5 seeds, Bach, training context 512)

`--phase-init 0.02` and `0.1`: the phase heads start away from zero, so the model cannot keep the
what/where separation by leaving them there. Registered:

- If indexing-style purity is what the zero start protects, forcing the phase should COST something
  relative to T3 -- on Bach that shows as worse in-distribution NLL (0-512).
- If forced and optional phases behave the same, the post-hoc reading is wrong too, and the honest
  position becomes: a per-token additive phase is simply free on these tasks, with no mechanism
  identified for why PoPE's constant-delta design was ever preferable.
- Reported either way: the learned phase magnitude (mean |d^q|, |d^k|) in the zero-initialised T3
  runs, which says directly how much freedom the model chose to use. If it is near zero on Indirect
  Indexing and large on Bach, that supports "optional and used only where it pays" without needing
  a new run.
