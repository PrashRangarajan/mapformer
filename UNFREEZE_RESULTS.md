# Freeze-then-unfreeze: the early window is refuted, and most of W4 was my install scale

Pre-registration: `UNFREEZE_PREREG.md`. Single-`p0` EM with the recency rewind installed
in its position pathway, released at different times or installed at a different weight
scale. 5 arms x seeds 0-7, one batch. The rewind's state was recorded every epoch into
the checkpoints, so "when" and "how" below are measured, not inferred.

**U1 passed** (checked before the batch finished): `EMUnf_0` s0 is bitwise identical to
the stored `EMWarm_train` s0 -- all 27 shared tensors and the full loss curve. The
recording was inert.

| arm | released | acc T=1024 | acc T=2048 | >= 0.95 | eff. slope | latent-code slope | leak |
|---|---|---|---|---|---|---|---|
| (`EMWarm_freeze`) | never | 1.000 +/- 0.000 | 1.000 | 8/8 | -1.000 | -1.000 | 0 |
| `EMUnf_0` | step 0 | 0.642 +/- 0.266 | 0.582 | 0/8 | -0.054 | -0.095 | 0.421 |
| `EMUnf_5` | epoch 5 | 0.609 +/- 0.142 | 0.575 | 0/8 | +0.013 | -0.229 | 0.393 |
| `EMUnf_30` | epoch 30 | 0.835 +/- 0.282 | 0.758 | 3/8 | -0.143 | -0.445 | 0.375 |
| `EMUnf_100` | epoch 100 | 0.605 +/- 0.370 | 0.531 | 1/8 | -0.125 | -0.203 | 0.308 |
| **`EMUnf_0_e8`** | **step 0, 8x scale** | **0.941 +/- 0.063** | **0.895** | 4/8 | -0.604 | **-0.989** | 0.726 |

Slopes: -1 is an exact rewind, 0 is none. Leak: ||Delta from content|| / ||Delta from the
latent code||, over symbol tokens.

## U2 -- early window: NOT SUPPORTED, and the trajectories refute its mechanism

Registered: `EMUnf_30` >= 0.95 on >= 7/8 seeds with final slope <= -0.8. Got **3/8** and
**-0.143**. The hard falsifier (mean <= 0.75) did not fire: the mean is 0.835, with seven
seeds at 0.88-1.00 and one collapsed at 0.144. Paired contrasts against `EMUnf_0` are all
unmeasured (MDE 0.33-0.44, seed variance dominates).

**The trajectories settle it where the accuracies cannot.** Epoch at which the
effective slope first rises above -0.5:

| arm | per seed | after release |
|---|---|---|
| `EMUnf_0` | 17, 9, 11, 13, 10, 10, 9, 10 | 9-17 epochs |
| `EMUnf_30` | 33, 34, 34, 33, 33, 33, 31, 36 | **1-6 epochs** |
| `EMUnf_100` | 101, 101, 119, 103, 101, 102, 103, 104 | **1-19 epochs** |

At epoch 30 the frozen twin's content branch is already trained (loss ~0.03). Releasing
the rewind then does not protect it -- it is gone within a handful of epochs on every
seed. **So the "random content gate dismantles the rewind" mechanism is wrong.** The
destruction does not need an untrained content branch.

## U3 -- install scale: the dominant cause

| contrast (paired) | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| **`EMUnf_0_e8 - EMUnf_0`** | **+0.298** | 0.272 | 0.270 | **7/8** | **DETECTABLE** |

The same Delta, installed at 8x the weight scale, raises trainable-from-step-0 accuracy
from 0.642 to **0.941** -- **84% of the gap to the frozen twin** -- and the latent rewind
code survives almost exactly (final slope **-0.989** against -0.095). By the registered
threshold this is not full survival (4/8 seeds >= 0.95, effective slope -0.604); the
residual comes from the second channel below. The stated confound -- at 8x the content
branch also sees a larger latent coordinate -- stands, but the mechanism data points
directly at Adam eroding a small code.

## U4 -- how the rewind breaks, and why these two channels are the only ones

The effective slope can leave -1 through exactly two routes. Changes to `w_out`, or to the
latent columns of `w_in`, act identically on `q_k` and on the symbols, so they preserve
the cancellation `Delta(q_k) = -(k-1) Delta(symbol)` by construction. What remains is
(i) the latent code itself changing and (ii) content leaking into Delta through `w_in`'s
content columns. Both were recorded.

- **At the 1/64 scale the latent code breaks first**: it leaves -0.9 at epochs 3-12 from
  step 0, before the effective slope crosses -0.5 (9-17), with leakage often later.
  **Erosion speed tracks the learning rate**: released at epoch 30, at the PEAK rate, the
  code leaves -0.9 within **1-4 epochs**, faster than the 3-12 epochs it took during
  warmup. That is what step-size erosion predicts, and the opposite of what the early-
  window account predicts.
- **At 8x scale the code holds; leakage does the damage**: leak exceeds 0.5 at epochs
  10-21 on every seed (final 0.73) and pulls the effective slope to -0.6.

## What this does to `WARM_RESULTS.md`

- **W4's "training dismantles the solution -- a landscape property" is WITHDRAWN in its
  strong form.** About 84% of the frozen-vs-trainable gap was the weight scale at which I
  stored the rewind: a code whose symbol step is 0.0156 in weight units is eroded by
  Adam's ~lr-per-coordinate steps. This is rule 31 -- "a parameterisation change is an
  optimiser change", written earlier the same day -- applied to my own warm-start design.
- **What remains of W4:** installed at a scale Adam does not erode, trainable EM mostly
  holds the rewind (0.941; 0.895 at 2x length), degraded by content leaking into Delta.
- **Unaffected:** W1 (frozen install scores 1.000, so the full model represents recency),
  and from scratch **0 of 40** EM runs find a rewind.

EM's recency deficit is therefore best described as a **search** problem: the solution
exists, can largely be held, and is never found from random initialisation.

## Open

1. **Why does from-scratch training never find the rewind?** This is now the question.
   One candidate worth measuring before theorising: the rewind needs `Delta(q_k)` spread
   over a 64:1 range along one direction, and nothing in a random init points there.
2. **The leakage channel.** `w_in`'s content columns start at zero and grow. Freezing only
   those columns in the 8x arm would show whether leakage is the entire residual.
3. The non-rewind mechanism behind `AlignFree`'s large-`k` advantage (`MAGONLY_RESULTS.md`).
