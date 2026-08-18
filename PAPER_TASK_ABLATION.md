# Context-destruction ablation on the paper's own task

T=128, fresh obs_map (env seed 10000), n=3 seeds, 20x128 sequences per cell. Scored at revisited observation positions -- the paper's own target.

**Floors, measured on these events, not assumed.** The paper's task scores every revisited observation including blanks, and p_empty=0.5 makes blank the majority class: measured blank rate **0.506**, so the always-predict-blank floor is that, not 1/16. Compare Match-Query, which restricts to non-blank answers and therefore has a 0.0625 chance and a 0.0893 never-moved floor.

**shuffle vs resample.** `shuffle` permutes the slots, which also destroys the walk's run-length autocorrelation and puts the input off-manifold. `resample` substitutes the corresponding stream from an INDEPENDENT episode -- a perfectly valid walk that simply does not match the observations beside it. `resample` is the trustworthy column; `shuffle` is kept because it is what was reported first.

| variant | intact | shuffle actions | **resample actions** | shuffle obs | resample obs |
|---|---|---|---|---|---|
| Vanilla | 0.9889 ± 0.0102 | 0.2314 ± 0.0119 | 0.1783 ± 0.0137 | 0.2915 ± 0.0021 | 0.3092 ± 0.0020 |
| VanillaEM_P0 | 0.9873 ± 0.0118 | 0.3831 ± 0.0184 | 0.3617 ± 0.0449 | 0.2680 ± 0.0131 | 0.2877 ± 0.0122 |

## The comparison this was run to make

| task | manipulation | intact | destroyed | drop | floor |
|---|---|---|---|---|---|
| paper task (Vanilla) | actions shuffled | 0.989 | 0.231 | **-0.758** | 0.506 (blank) |
| paper task (Vanilla) | actions resampled (on-manifold) | 0.989 | 0.178 | **-0.811** | 0.506 (blank) |
| paper task (VanillaEM_P0) | actions shuffled | 0.987 | 0.383 | **-0.604** | 0.506 (blank) |
| paper task (VanillaEM_P0) | actions resampled (on-manifold) | 0.987 | 0.362 | **-0.626** | 0.506 (blank) |
| Match-Query (MapWM-Flat) | query actions shuffled | 0.918 | 0.076 | **-0.842** | 0.089 (never-moved) |

Per-seed:

- `Vanilla` intact: 0.9997, 0.9874, 0.9796
- `Vanilla` shuffle_actions: 0.2250, 0.2241, 0.2451
- `Vanilla` resample_actions: 0.1804, 0.1637, 0.1909
- `Vanilla` shuffle_obs: 0.2937, 0.2912, 0.2895
- `Vanilla` resample_obs: 0.3113, 0.3092, 0.3072
- `VanillaEM_P0` intact: 0.9995, 0.9759, 0.9864
- `VanillaEM_P0` shuffle_actions: 0.4038, 0.3770, 0.3686
- `VanillaEM_P0` resample_actions: 0.4072, 0.3605, 0.3174
- `VanillaEM_P0` shuffle_obs: 0.2531, 0.2778, 0.2732
- `VanillaEM_P0` resample_obs: 0.2738, 0.2967, 0.2925

## Reading it

### 1. The prediction this was run to test FAILED

The hypothesis was that the paper's task leaves a content route to localisation
open -- ~4-5 revealed observations carry enough bits to pin a cell on a
4096-cell torus -- so destroying the action stream should cost it much less than
the -0.842 it costs Match-Query. It does not. Both tasks fall BELOW their own
floors:

| task | destroyed | its floor | below floor by |
|---|---|---|---|
| paper task (Vanilla), actions resampled | 0.178 | 0.506 (blank) | 0.33 |
| Match-Query (MapWM-Flat), actions shuffled | 0.076 | 0.089 (never-moved) | 0.01 |

**So the "Match-Query closes a route the paper's task leaves open" framing has no
support and is withdrawn.** The bit-budget argument was about what is POSSIBLE,
not what is LEARNED. Corroborating from elsewhere in the repo: the
index-position model on the clean task reaches 0.467
(`NOISE_CLEAN_REVALIDATION.md`) against a ~0.51 blank floor -- it fails the
paper's task about as completely as it fails Match-Query. Path integration looks
necessary for both.

### 2. My OWN explanation for the below-floor result is also falsified

The first version of this file blamed the sub-floor collapse on the shuffle
putting the input off-manifold: permuting slots destroys the walk's run-length
autocorrelation, so the model sees a stream no valid trajectory could emit. That
was a plausible story and it is wrong. Substituting a real walk from an
independent episode -- correct statistics, on-manifold, merely uncorrelated with
the observations -- gives the SAME answer, in fact slightly worse:

| Vanilla | shuffle (off-manifold) | resample (on-manifold) |
|---|---|---|
| actions | 0.231 ± 0.012 | **0.178 ± 0.014** |
| observations | 0.292 ± 0.002 | 0.309 ± 0.002 |

So the collapse is real and not a distribution artifact. The original number
stands, and stands more firmly than when it was reported.

### 3. A plausible lie misleads more than an obvious one

Resampling is MORE destructive than shuffling for Vanilla (0.178 vs 0.231, 3/3
seeds; NLL 4.89 vs 4.63). A coherent-but-wrong trajectory is integrated
confidently to a wrong position; an incoherent one at least partly signals its
own corruption. This is a small effect and one manipulation, but it runs the
opposite way to the OOD story and is worth stating.

### 4. Below the floor means overconfidence, and that is measured

Vocabulary is 21 tokens, so uniform guessing costs ln(21) = 3.04 nats.

| condition | NLL |
|---|---|
| intact | 0.006 - 0.100 |
| actions resampled | 3.68 - 4.94 |
| observations resampled | 4.50 - 5.64 |

**Every destroyed condition is worse than uniform.** These models do not hedge
toward the 0.51 blank majority when their map is unusable; they commit to a
retrieval and the retrieval returns garbage. That is why accuracy lands under the
trivial floor, and it is a measurement rather than an inference.

### 5. What this still cannot say

It shows the trained models USE the action stream. It cannot show whether a model
COULD solve the paper's task without one -- a behavioural fact, not a capability
claim. The clean test of that is a trained index-position baseline on the paper's
task, which is a training run, not an ablation.

## What Match-Query can still claim over the paper's task

Not "it closes the content route" -- unsupported. What survives is methodological:

- chance is 0.0625 by construction (non-blank answers only) rather than a 0.51
  blank majority, so the metric has ~8x the headroom
- the blind phase extends arbitrarily far past training length with no
  retraining, an OOD axis the paper's task does not have
- its shortcuts are gated (n-gram orders 1-5, never-moved, marginal, oracle)

## Secondary observation, not over-read

EM retains far more than WM under both action manipulations (0.362 vs 0.178 on
resample, 3/3 seeds). Consistent with EM's multiplicative gate leaving the
content branch A_X intact when A_P is scrambled. No mechanism is claimed: the
A_P kernel-geometry account was falsified on a pre-registered test
(`AP_KERNEL_DIAGNOSTIC.md`).
