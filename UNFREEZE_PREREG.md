# Pre-registration: freeze-then-unfreeze

Written before any checkpoint in `runs/unfreeze/` exists. Follows `WARM_RESULTS.md`:
single-`p0` EM with the recency rewind installed and FROZEN scores 1.000 (8/8); the same
install left TRAINABLE falls to from-scratch level (0.642), sitting at chance for ~100
epochs while the frozen twin reaches loss 0.2-0.45 by epoch 10.

## Two explanations, and a third thing to measure

**Early window (the hypothesis this run was requested to test).** EM's score is
`A_X (*) A_P`, so `d score / d A_P = A_X`: while the content branch is random, the
position pathway receives gradients filtered through a random gate and the rewind is
dismantled before content can use it. Predicts SURVIVAL if the pathway is released
only after the content branch has learned.

**Install scale -- a hole in W4 found while designing this.** The rewind is stored in
embedding coordinates at `EPS = 1/64`, so one symbol step is 0.0156 in weight units,
and Adam moves every coordinate by roughly the learning rate per step regardless of its
size. The trainable twin may have lost the rewind because I installed it at a scale
Adam perturbs easily, not because the landscape rejects it -- rule 31, bought earlier
the same day. If so, `WARM_RESULTS.md`'s "landscape property" reading of W4 must be
withdrawn (W1 and the never-found result are unaffected).

**How it dies.** Intermediate checkpoints were not saved last time, so "destroyed
early" was an inference from the loss curve. This time three quantities are recorded
every epoch INTO THE CHECKPOINT: the effective rewind slope (-1 exact, 0 none), the
rewind slope of the latent embedding code alone, and the ratio of content-driven to
latent-driven Delta over symbol tokens (leakage through `w_in`).

## Arms (seeds 0-7, recency recipe unchanged, no `--fast-attn`)

| arm | position pathway released at | install scale | lr at release |
|---|---|---|---|
| `EMUnf_0` | step 0 | 1/64 | warmup start (~0) |
| `EMUnf_5` | epoch 5 (content still learning) | 1/64 | ~3.3e-4 (in warmup) |
| `EMUnf_30` | epoch 30 (frozen twin at loss ~0.03) | 1/64 | **~0.99e-3 (peak)** |
| `EMUnf_100` | epoch 100 | 1/64 | ~0.82e-3 |
| `EMUnf_0_e8` | step 0 | **1/8** | warmup start |

Warmup is 720 steps (epoch 15). Existing reference points, same seeds:
`EMWarm_freeze` = never released (1.000); `EMWarm_train` = released at 0 (0.642).

## Verified before launch

- `EMUnf_0` equals `EMWarm_train` at init on every shared tensor; it adds only the three
  recording buffers, and the recording draws no random numbers.
- The `eps = 1/8` install produces the SAME Delta for every token as `eps = 1/64`
  (`w_in` gain compensates), so it changes only the scale at which Adam sees the code.
- The release fires on schedule: in a smoke run with release at epoch 1, `w_out` first
  moves at step 48 exactly.

## Predictions

**U1 (determinism, precondition).** `EMUnf_0` s0 is bitwise identical in its shared
weights to the stored `EMWarm_train` s0. *If not*, the recording perturbed training and
nothing else is read until that is explained.

**U2 (early window).** Survival rises with release epoch. Registered as:
`EMUnf_30` scores **>= 0.95 at T=1024 on >= 7/8 seeds** with final rewind slope
**<= -0.8**. *Falsified if* `EMUnf_30` falls to from-scratch level (<= 0.75 mean). Because
`EMUnf_30` is released at the PEAK learning rate -- larger Adam steps than `EMUnf_0` saw
during warmup -- its survival would also count against pure step-size fragility.

**U3 (install scale).** No direction registered; both outcomes are informative.
`EMUnf_0_e8` surviving (>= 0.95, slope <= -0.8) means W4 was an installation artefact and
its landscape reading is withdrawn. `EMUnf_0_e8` collapsing like `EMUnf_0` means scale is
not the cause. Stated confound: at `eps = 1/8` the content branch also sees larger latent
coordinates (q_64 carries -7.9 in one embedding coordinate), so a difference could come
from the content side as well.

**U4 (mechanism, descriptive).** For every arm that loses the rewind, report which
recorded channel breaks first -- the latent code (`traj_lat` moving off -1) or content
leakage (`traj_leak` rising) -- and at which epoch. No direction registered.

**Not a rule-9 question**, for the same reason as `WARM_PREREG.md`: the claim is whether
a solution is held, and loss-matching would condition that away.
