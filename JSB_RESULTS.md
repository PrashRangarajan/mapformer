# READING

> **CONTEXT ADDED 2026-09-20 (`AUG_RESULTS.md`)**: these runs are overfitting-limited. Pitch
> transposition -- the augmentation PoPE's own paper applies to MAESTRO but not to JSB -- gains 0.107
> NLL for PoPE and 0.090 for MapPoPE, 5/5 seeds, against the 0.032 that separates PoPE from RoPE here.
> The ordering below is unchanged (and the PoPE-over-MapPoPE gap widens to 0.028 under augmentation),
> but every effect on this page sits under a ceiling worth three times its size.

**The paper's result replicates, and path integration does not help.** First clean test of path
integration on a natural-sequence task with real content, and the registered prediction was
deliberately left open.

- **R1 REPLICATES**: PoPE - RoPE = -0.0322 NLL (paper -0.0192), 5/5 seeds, levels 0.5331 / 0.5009
  against the paper's 0.5081 / 0.4889. Both within 0.025 of the published values.
- **Path integration buys nothing at the best-validation checkpoint**: -0.0045 on the RoPE row
  (3/5 seeds, MDE 0.0126) and +0.0111 on the PoPE row (0/5 seeds better, MDE 0.0116) -- unmeasured
  in both directions, and the PoPE row's sign is against it on every seed.
- **PoPE's encoding survives on the path-integrated row**: MapPoPE - MapWM = -0.0165 (5/5,
  MDE 0.0111) DETECTABLE. So the encoding effect is robust to the position mechanism; the position
  mechanism adds nothing to it.
- **What path integration does do here is resist overfitting.** Every arm's best validation step is
  at 1000-1750 of 3000 iterations and every run degrades after it, but by the final step the
  path-integrated arms are far better on test (0.5755 / 0.5570) than the index arms (0.6764 /
  0.6503), with HIGHER training loss (0.23 vs 0.12). Not a pre-registered contrast; recorded as an
  observation, on 229 training pieces.
- Rule 9: r(final train loss, test NLL) = +0.064 over 20 runs, i.e. test NLL here is not the training
  loss in disguise -- the opposite of the usual pattern in this project, because this dataset is
  small enough to overfit.

