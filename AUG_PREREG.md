# AUG_PREREG -- is the in-distribution comparison on Bach limited by overfitting, not by position?

## Why

In PoPE's own configuration (whole pieces, 2048 context, 3000 iterations) the encoding effect is
PoPE - RoPE = **-0.032 NLL**, and path integration on top of PoPE costs +0.011 (0/5 seeds). But the
same runs overfit hard: every arm's best validation step is 1000-1750 of 3000, final training loss is
0.117-0.236 against validation 0.565-0.694, and test NLL at the final step is 0.650-0.676 against
0.501-0.533 at best-valid. **That gap is about 0.15 NLL, five times the effect being compared.**

The PoPE paper augments MAESTRO with pitch transpositions sampled from -3..+3 and uses no
augmentation on JSB. Our token ids span 38..83 in a vocabulary of 90, so +/-3 semitones never clips.

## Arms

`PoPE` and `MapPoPE` with `--augment 3` (random transposition per training batch, training only;
validation and test untouched), 5 seeds, otherwise the recipe of `JSB_PREREG.md` verbatim -- full
2048 context, 3000 iterations. Baselines in hand from the same recipe: PoPE 0.5009, MapPoPE 0.5121,
MapWM 0.5286, RoPE 0.5331 (test at best valid).

## Registered verdicts

- **A1 (the point of the run)** augmented PoPE - unaugmented PoPE, paired by seed. **If the gain
  exceeds 0.032 -- the entire PoPE-over-RoPE effect -- then the positional comparison on this dataset
  has been running against an overfitting ceiling**, and encoding differences measured here are
  smaller than the headroom a standard augmentation recovers.
- **A2** the same contrast on MapPoPE. Registered: augmentation should help BOTH; a large difference
  between A1 and A2 would mean the two encodings differ in how much they overfit, which is a
  different claim from anything measured so far.
- **A3 (ordering)** does augmentation change the PoPE-vs-MapPoPE verdict? Unaugmented, PoPE wins by
  0.011 on 5/5 seeds. No direction registered: path integration overfits LESS in these runs (final
  test 0.576 against 0.650), so it has less to gain, which could go either way once the ceiling lifts.
- **A4** best-validation step, reported per arm. If augmentation pushes it from ~1000-1750 toward the
  budget end, the run was data-limited rather than capacity-limited, and the 3000-iteration budget
  itself becomes the next thing to question.
- Rule 9 check: r(final train loss, test NLL) across the augmented runs.

**Scope**: this changes the DATA, not the model, so it does not bear on any positional claim directly
-- it bounds how much of the in-distribution difference between encodings is worth interpreting.
