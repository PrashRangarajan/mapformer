# Addition pilot 2: a faithful positive control

Written 2026-09-14, before training. Pilot, one seed per cell; not a result.

**Why.** In pilot 1 (`ADDITION_PILOT.md`) the coupled oracle (RoPE over coupled IDs) did not
length-generalise at all (0.00 at 24 digits), so there was no ceiling to compare MapFormer against.
Cho et al. reach 95.65% at 200 digits with LEARNED ABSOLUTE embeddings over coupled IDs, random
starting IDs, training on 1-30 digits, and about 50M problems.

**Changes from pilot 1:**
- A faithful oracle, `CoupledAPE` (`model_coupled_ape.py`): learned absolute embeddings over coupled
  IDs, random start in training, start at 1 in evaluation. It still uses this repo's WM layer (not
  their GEGLU / RMSNorm / d=512 model).
- Training up to 30 digits instead of 16.
- Budget 200 epochs x 100 batches x 512 = about 10M problems, about 4x pilot 1 and about 1/5 of Cho et al.
- Evaluation at 16, 30, 45, 60, 90 and 120 digits, 256 problems each.

**Arms, one seed, at 1 and 2 layers, d=256, 4 heads:**
- role-tagged digits: `CoupledAPE`, `CoupledRoPE`, `Vanilla_r4`, `Abs_r4`, `RoPE`, `NoPE`;
- shared digits: `CoupledAPE`, `Vanilla_r4`, `RoPE`.

**Pass condition for moving to a pre-registered batch, fixed now:** `CoupledAPE` reaches exact match
>= 0.5 at 60 digits (2x the training length) in at least one layer setting.
- If it does not, the control still does not work in this architecture or at this budget, and no
  comparison with it is interpretable.
- Whatever MapFormer does is recorded but not read until the control passes.
