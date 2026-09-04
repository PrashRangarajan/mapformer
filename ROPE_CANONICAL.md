# Does the RoPE baseline's frequency schedule matter?

The repository computes `inv_freq_c = base^(-c/(n_b-1))`; canonical RoPE
is `base^(-c/n_b)`. They agree to within 1% over the high-frequency blocks
that resolve position at these lengths and differ by up to 25% at
frequencies whose wavelength (47k-63k tokens) is DC over anything run
here. Same task, same recipe, same batch, n=16.

| length | repo | canonical | canonical - repo | t | seeds | sign test |
|---|---|---|---|---|---|---|
| 16 | 0.676 | 0.633 | **-0.0425** (MDE 0.1480) | -0.80 | 7/16 | p=0.804 |
| 32 | 0.583 | 0.563 | **-0.0199** (MDE 0.0685) | -0.81 | 7/16 | p=0.804 |
| 64 | 0.540 | 0.531 | **-0.0083** (MDE 0.0339) | -0.68 | 7/16 | p=0.804 |
| 128 | 0.519 | 0.516 | **-0.0035** (MDE 0.0169) | -0.58 | 8/16 | p=1.000 |
| 256 | 0.510 | 0.508 | **-0.0023** (MDE 0.0086) | -0.77 | 7/16 | p=0.804 |

## Verdict

**The schedule does not matter.** Every contrast is inside its MDE and no sign count is lopsided enough to matter either.

**Action, per the pre-registration:**

1. Switch `model_baseline_rope.py` to the canonical `base**(-k_idx / self.n_blocks)`, so the line stops contradicting the comment above it.
2. Delete the canonical-vs-repo discussion from `mapformer_math.tex` and print the canonical formula plainly.
3. Record in `CLAUDE.md` that RoPE runs before this date used the `n_b-1` schedule. `inv_freq` is a registered buffer, so stored checkpoints keep their own values on load and remain valid; only future runs change.

## Scope

One task, one width, n=16. The schedules differ most at the lowest frequencies, whose effect should grow with sequence length — L=256 is the most informative row and the one to weight.
