# Pre-registration: the leakage test

Written before any checkpoint in `runs/noleak/` exists. Follows `UNFREEZE_RESULTS.md`.

## What is being tested

The installed rewind's effective slope can leave -1 through exactly two channels (U4):
the latent embedding code itself changing, and content leaking into Delta through
`w_in`'s content columns. At the 1/64 install scale the latent code eroded first; at 8x
it survived (-0.989) and leakage (final 0.73) was the residual, holding the trainable
twin at **0.941** against the frozen twin's **1.000**.

**Intervention:** hold `w_in`'s content columns at zero (they start at zero; gradient
masked; decoupled decay of zero is zero). `ActionToLieAlgebra` has no bias, so Delta then
depends on the latent code alone and the leak channel is closed completely. Everything
else trains. Built in a new module (`model_em_noleak.py`); the modules defining the
comparison arms are untouched.

## A 2x2 with two existing arms, same seeds 0-7

| | leak open | leak closed (new) |
|---|---|---|
| **install 1/64** | `EMUnf_0` 0.642 | `EMNoLeak_e64` |
| **install 1/8** | `EMUnf_0_e8` 0.941 | `EMNoLeak_e8` |

Plus one determinism re-check: `EMUnf_0_e8` s0 re-trained into a separate directory must
be bitwise identical to the stored run, which licenses pairing with the existing arms.

## Verified before launch

- Each new arm equals its leak-open counterpart at init on every shared tensor.
- Over two epoch boundaries of training with AdamW, weight decay and gradient clipping,
  the content columns stay exactly 0, the latent columns move, `traj_leak` stays 0, and
  the effective slope equals the latent-code slope -- the identity U4's argument implies.

## Predictions

**L1 (primary) -- is leakage the entire residual at 8x?** `EMNoLeak_e8` scores >= 0.95 at
T=1024 on >= 7/8 seeds, with final effective slope <= -0.95. *Falsified if*
`EMNoLeak_e8 - EMUnf_0_e8` is inside its MDE or negative: then leakage was not the
residual and something else limits the trainable install. Ceiling note: `EMUnf_0_e8` is
already at 0.941 with sd 0.063, so the contrast has at most ~0.06 of headroom and an MDE
near 0.06 -- the seed-count criterion is the readout that can resolve this, not the
paired delta.

**L2 -- are the channels separable?** `EMNoLeak_e64` still collapses: mean <= 0.75 with
final latent-code slope above -0.5, because at 1/64 the code itself erodes regardless of
leakage. *Falsified if* it survives, which would mean leakage also drove the small-scale
collapse and U4's ordering (latent code breaks first) was misleading.

**L3 (descriptive).** Scale effect with the leak closed vs open, and the interaction.

**Manipulation check.** `traj_leak` must be exactly 0 at every epoch in both new arms,
and effective slope must equal latent-code slope at every epoch. If either fails, the
intervention did not do what it claims and L1/L2 are not read.

## Revision BEFORE launch -- recorded here, not buried

A pre-launch diagnostic (endpoint of the existing unfreeze checkpoints) showed that
`UNFREEZE_RESULTS.md`'s U4 claim -- two recorded channels, exhaustive -- was false: the
latent code has two coordinates and trainable per-token rows, and the recorded
"latent-code slope" read coordinate 0 only. With the complete latent pathway, the
leak-open 8x arm ends at **-0.866** (coordinate 0: -0.989), and leakage takes it to -0.602.

Changes, all made before any `runs/noleak/` checkpoint exists:

- **Recording:** the new arms also record `traj_latpath`, the rewind slope from the
  latent-driven Delta alone (both coordinates). Verified: with the leak closed,
  `traj_slope == traj_latpath` exactly at every epoch, and `traj_leak == 0`. That identity
  is now the manipulation check; the original "effective = coordinate-0 slope" check was
  wrong and is dropped.
- **L1 is kept exactly as registered**, with a stated lower prior: closing the leak can
  produce full survival only if it ALSO stops the latent pathway degrading, which the
  leak-open arm did not.
- **L1b (added):** report `EMNoLeak_e8`'s final latent-pathway slope against the leak-open
  arm's -0.866. Does closing the leak protect the latent pathway as well, or only remove
  the leakage term?
- **L2 now reads the complete latent-pathway slope:** `EMNoLeak_e64` collapses if its mean
  accuracy is <= 0.75 with final `traj_latpath` above -0.5 (was: coordinate-0 slope).
