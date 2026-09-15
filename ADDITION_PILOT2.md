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

## Results (one seed per cell)

Exact-match accuracy by operand length, training up to 30 digits:

| layers | format | arm | final loss | 16 | 30 | 45 | 60 | 90 | 120 | per-digit at 60 / 120 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | role | `CoupledAPE` (faithful oracle) | 0.0021 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.87 / 0.61 |
| 1 | role | `CoupledRoPE` | 0.0003 | 1.00 | 1.00 | 0.50 | 0.11 | 0.00 | 0.00 | 0.97 / 0.85 |
| 1 | role | `Vanilla_r4` (signed) | 0.0003 | 1.00 | 1.00 | 0.16 | 0.00 | 0.00 | 0.00 | 0.89 / 0.19 |
| 1 | role | `Abs_r4` (monotone) | 0.5808 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 / 0.09 |
| 1 | role | `RoPE` | 1.5368 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.11 / 0.10 |
| 1 | role | `NoPE` | 2.0442 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.11 / 0.11 |
| 1 | shared | `CoupledAPE` | 0.0035 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.92 / 0.78 |
| 1 | shared | `Vanilla_r4` | 1.5629 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 / 0.09 |
| 1 | shared | `RoPE` | 1.8321 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 / 0.10 |
| 2 | role | `CoupledAPE` | 0.8197 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 / 0.11 |
| 2 | role | `CoupledRoPE` | 0.0000 | 1.00 | 1.00 | 0.00 | 0.01 | 0.01 | 0.00 | 0.97 / 0.96 |
| 2 | role | `Vanilla_r4` | 0.0002 | 1.00 | 1.00 | **0.93** | **0.79** | 0.00 | 0.00 | 0.99 / 0.18 |
| 2 | role | `Abs_r4` | 0.0013 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 / 0.11 |
| 2 | role | `RoPE` | 0.0052 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.11 / 0.10 |
| 2 | role | `NoPE` | 0.9613 | 0.37 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 / 0.11 |
| 2 | shared | `CoupledAPE` | 0.0083 | 1.00 | 1.00 | 0.88 | 0.07 | 0.00 | 0.00 | 0.96 / 0.58 |
| 2 | shared | `Vanilla_r4` | 0.0303 | 0.90 | 0.74 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 / 0.10 |
| 2 | shared | `RoPE` | 0.0072 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 / 0.10 |

## Verdict on the pass condition: NOT MET

`CoupledAPE` does not reach 0.5 exact match at 60 digits in either layer setting (best 0.07). It
generalises to 1.5x the training length (1.00 and 0.88 at 45 digits, shared format) but not to 2x.
With 2 layers in the role format it did not train (loss 0.82). By the rule fixed before training, the
MapFormer comparison is recorded here and **not read**.

## Recorded, not read

- **Signed MapFormer, 2 layers, role format:** the best long-length exact match of any arm
  (0.93 at 45, 0.79 at 60 digits).
- **Monotone MapFormer, RoPE and NoPE** do not generalise past 30 digits in any setting, and fail
  even at training length with 1 layer.
- **Shared digit tokens:** no learned code generalises. Signed MapFormer does not even solve the
  training length with 1 layer.
- These fit the design's predictions, which is exactly why they are not read against a control that
  failed.

## Why the controls fall short of the literature, and the options

**Per-digit accuracy stays high where exact match collapses.** `CoupledRoPE`, 2 layers, is 0.97
per digit at 60 and 0.96 at 120, but exact match is about 0. One wrong digit in 121 fails the problem.
The oracles are close to right, not far off.

**This setup differs from Cho et al. in three ways:**
- about 1/5 of the problems (10M vs 50M);
- lr 1e-3 vs 1e-4;
- this repo's layer vs their d=512 GEGLU/RMSNorm model.

**Next options:**
1. Match Cho et al.'s recipe for the oracle (d=512, lr 1e-4, 50k x 1000) and check it reaches 200
   digits before running anything else. This is the faithful fix, and expensive.
2. Pre-register per-digit accuracy (or exact match up to a length where the control passes, e.g. 45
   digits) as the primary readout. This is cheaper but a weaker claim.
3. Raise the budget about 5x for all arms at the current architecture and re-check the control.
