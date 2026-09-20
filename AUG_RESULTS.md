# Augmentation: the in-distribution comparison on Bach was running against an overfitting ceiling

Pre-registration: `AUG_PREREG.md`. Pitch transposition in [-3, +3] on training batches only, 5 seeds,
otherwise PoPE's recipe verbatim (full 2048 context, 3000 iterations). Test NLL at best validation.

| arm | test NLL | gain | best-validation step |
|---|---|---|---|
| PoPE | 0.5009 | -- | 1000 |
| **PoPE + augmentation** | **0.3936** | **-0.1073** | **3000 (the budget end)** |
| MapPoPE | 0.5121 | -- | 1500 |
| **MapPoPE + augmentation** | **0.4221** | **-0.0900** | **3000** |

- **A1 FIRES DECISIVELY.** Augmenting PoPE gains **0.107 NLL** (5/5 seeds, MDE 0.006) -- **3.3x the
  entire PoPE-over-RoPE effect** (0.032) that this dataset is used to demonstrate. By the registered
  criterion, the in-distribution encoding comparison here has been running against an overfitting
  ceiling, and differences measured under it are smaller than the headroom a standard augmentation
  recovers.
- **A2**: the same holds for MapPoPE, -0.090 (5/5). Both encodings were data-limited.
- **A3 (ordering)**: PoPE still wins, and by MORE. MapPoPE - PoPE goes from +0.011 (5/5) unaugmented
  to **+0.028 (5/5, MDE 0.008, detectable)** augmented. Lifting the ceiling did not rescue path
  integration on this dataset; it made its cost clearer.
- **A4**: the best-validation step moves from 1000/1500 to **3000 for both arms** -- the runs are no
  longer data-limited but BUDGET-limited, so the 3000-iteration recipe is now the binding constraint
  and any further comparison here should extend it first.

## What this means for the PoPE-paper line

- **Their published JSB number is beatable by a wide margin with their own augmentation.** PoPE's
  paper reports 0.4889 on this dataset and applies transposition to MAESTRO but not to JSB. Our
  augmented PoPE reaches **0.3936**. Different codebase, same data, splits and recipe otherwise, so
  this is indicative rather than a like-for-like claim -- but the direction is large and 5/5.
- **Every in-distribution conclusion in this project's Bach work stands in ordering and shrinks in
  importance.** PoPE beats RoPE; path integration costs on top of PoPE; both were true under a
  ceiling worth three times the effects being compared.
- It does not touch the length-extrapolation results, which are about a different regime -- but it
  does mean the in-distribution column of those tables is the least interesting one.

## Scope

This changes the DATA, not the model, so it bears on no positional claim directly. What it bounds is
how much of an in-distribution difference between encodings on this dataset is worth interpreting.
Rule 9 across the augmented runs: r(final train loss, test NLL) is not computed here because the
best-validation step is now at the budget end for every run, so final-step loss is no longer the
quantity the checkpoint was selected on.
