# Context-destruction ablation on the paper's own task

T=128, fresh obs_map (env seed 10000), n=3 seeds, 20x128 sequences per cell. Scored at revisited observation positions -- the paper's own target.

**Floors, measured on these events, not assumed.** The paper's task scores every revisited observation including blanks, and p_empty=0.5 makes blank the majority class: measured blank rate **0.511**, so the always-predict-blank floor is that, not 1/16. Compare Match-Query, which restricts to non-blank answers and therefore has a 0.0625 chance and a 0.0893 never-moved floor.

| variant | intact | shuffle actions | shuffle obs |
|---|---|---|---|
| Vanilla | 0.9889 ± 0.0101 | 0.2315 ± 0.0102 | 0.2925 ± 0.0029 |
| VanillaEM_P0 | 0.9870 ± 0.0119 | 0.3822 ± 0.0217 | 0.2687 ± 0.0132 |

## The comparison this was run to make

| task | manipulation | intact | destroyed | drop | floor |
|---|---|---|---|---|---|
| paper task (Vanilla) | actions shuffled | 0.989 | 0.232 | **-0.757** | 0.511 (blank) |
| paper task (VanillaEM_P0) | actions shuffled | 0.987 | 0.382 | **-0.605** | 0.511 (blank) |
| Match-Query (MapWM-Flat) | query actions shuffled | 0.918 | 0.076 | **-0.842** | 0.089 (never-moved) |

Per-seed:

- `Vanilla` intact: 0.9996, 0.9875, 0.9795
- `Vanilla` shuffle_actions: 0.2262, 0.2251, 0.2433
- `Vanilla` shuffle_obs: 0.2957, 0.2918, 0.2902
- `VanillaEM_P0` intact: 0.9993, 0.9756, 0.9860
- `VanillaEM_P0` shuffle_actions: 0.4066, 0.3751, 0.3650
- `VanillaEM_P0` shuffle_obs: 0.2536, 0.2782, 0.2744

## Reading it: the prediction this was run to test FAILED

The hypothesis was that the paper's task leaves a content route to localisation
open, so destroying the action stream should cost it much less than it costs
Match-Query. It does not.

Both tasks fall **below their own floors** under the same manipulation:

| task | destroyed | its floor | below floor? |
|---|---|---|---|
| paper task (Vanilla) | 0.232 | 0.511 (blank) | yes, by 0.28 |
| Match-Query (MapWM-Flat) | 0.076 | 0.089 (never-moved) | yes, by 0.01 |

The paper's task does not degrade gracefully to a content-matching fallback; it
collapses past the trivial always-predict-blank baseline. **So this test gives no
support to the claim that Match-Query isolates path integration in a way the
paper's task does not, and that claim should not be made on this evidence.**

Two further things follow.

**The bit-budget argument was about what is POSSIBLE, not what is LEARNED.**
~4-5 revealed observations do carry enough information to pin a cell, but these
models evidently do not use that route even when it is available. Corroborating
evidence from elsewhere in the repo: the index-position model on the *clean*
task reaches 0.467 (`NOISE_CLEAN_REVALIDATION.md`) against a ~0.51 blank floor,
i.e. it fails the paper's task about as completely as it fails Match-Query. The
paper's task appears to require path integration in practice too.

**Falling below the floor limits what this ablation can conclude.** A model
genuinely falling back on content would still predict blank at roughly the blank
rate. Going under it indicates the shuffled sequence is out of distribution in a
way that scrambles the output, rather than a clean measurement of which route is
being used. The ablation shows both models USE the action stream; it cannot show
whether either COULD have solved the task without it. The clean version of that
question is a trained index-position baseline on the paper's task, not an
ablation.

## What Match-Query can still claim over the paper's task

Not "it closes the content route" -- unsupported. What survives is methodological:

- chance is 0.0625 by construction (non-blank answers only) instead of a 0.51
  blank majority, so the metric has ~8x the headroom
- the blind phase extends arbitrarily far past training length without
  retraining, giving an OOD axis the paper's task does not have
- its shortcuts are gated (n-gram orders 1-5, never-moved, marginal, oracle)

## Secondary observation, not over-read

EM retains more than WM under action shuffling (0.382 vs 0.232, 3/3 seeds). That
is consistent with EM's multiplicative gate leaving the content branch A_X intact
when A_P is scrambled. No mechanism is claimed: the A_P kernel-geometry account
was falsified on a pre-registered test (`AP_KERNEL_DIAGNOSTIC.md`), and this is a
single observation on one manipulation.
