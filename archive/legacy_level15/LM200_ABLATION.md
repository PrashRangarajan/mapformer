# Context-destruction ablation on the paper's own task

T=128, fresh obs_map (env seed 10000), n=3 seeds, 12x64 sequences per cell. Scored at revisited observation positions -- the paper's own target.

**Floors, measured on these events, not assumed.** The paper's task scores every revisited observation including blanks, and p_empty=0.5 makes blank the majority class: measured blank rate **0.484**, so the always-predict-blank floor is that, not 1/16. Compare Match-Query, which restricts to non-blank answers and therefore has a 0.0625 chance and a 0.0893 never-moved floor.

**shuffle vs resample.** `shuffle` permutes the slots, which also destroys the walk's run-length autocorrelation and puts the input off-manifold. `resample` substitutes the corresponding stream from an INDEPENDENT episode -- a perfectly valid walk that simply does not match the observations beside it. `resample` is the trustworthy column; `shuffle` is kept because it is what was reported first.

| variant | intact | shuffle actions | **resample actions** | shuffle obs | resample obs |
|---|---|---|---|---|---|
| Vanilla | 0.8427 ± 0.1050 | 0.3546 ± 0.0157 | 0.3428 ± 0.0114 | 0.2733 ± 0.0064 | 0.2829 ± 0.0045 |
| Level15 | 0.9850 ± 0.0212 | 0.2184 ± 0.0944 | 0.1553 ± 0.1141 | 0.2607 ± 0.0102 | 0.2724 ± 0.0125 |
| Vanilla_ExtraHead | 0.9952 ± 0.0070 | 0.3137 ± 0.0835 | 0.2871 ± 0.0906 | 0.2477 ± 0.0094 | 0.2629 ± 0.0105 |

## The comparison this was run to make

| task | manipulation | intact | destroyed | drop | floor |
|---|---|---|---|---|---|
| paper task (Vanilla) | actions shuffled | 0.843 | 0.355 | **-0.488** | 0.484 (blank) |
| paper task (Vanilla) | actions resampled (on-manifold) | 0.843 | 0.343 | **-0.500** | 0.484 (blank) |
| paper task (Level15) | actions shuffled | 0.985 | 0.218 | **-0.767** | 0.484 (blank) |
| paper task (Level15) | actions resampled (on-manifold) | 0.985 | 0.155 | **-0.830** | 0.484 (blank) |
| paper task (Vanilla_ExtraHead) | actions shuffled | 0.995 | 0.314 | **-0.681** | 0.484 (blank) |
| paper task (Vanilla_ExtraHead) | actions resampled (on-manifold) | 0.995 | 0.287 | **-0.708** | 0.484 (blank) |
| Match-Query (MapWM-Flat) | query actions shuffled | 0.918 | 0.076 | **-0.842** | 0.089 (never-moved) |

Per-seed:

- `Vanilla` intact: 0.8935, 0.9126, 0.7219
- `Vanilla` shuffle_actions: 0.3727, 0.3460, 0.3450
- `Vanilla` resample_actions: 0.3524, 0.3301, 0.3459
- `Vanilla` shuffle_obs: 0.2806, 0.2706, 0.2686
- `Vanilla` resample_obs: 0.2881, 0.2800, 0.2806
- `Level15` intact: 0.9982, 0.9605, 0.9962
- `Level15` shuffle_actions: 0.1555, 0.3270, 0.1727
- `Level15` resample_actions: 0.0849, 0.2870, 0.0940
- `Level15` shuffle_obs: 0.2556, 0.2541, 0.2725
- `Level15` resample_obs: 0.2661, 0.2642, 0.2868
- `Vanilla_ExtraHead` intact: 0.9986, 0.9999, 0.9871
- `Vanilla_ExtraHead` shuffle_actions: 0.2406, 0.2960, 0.4047
- `Vanilla_ExtraHead` resample_actions: 0.2229, 0.2476, 0.3908
- `Vanilla_ExtraHead` shuffle_obs: 0.2553, 0.2372, 0.2505
- `Vanilla_ExtraHead` resample_obs: 0.2688, 0.2508, 0.2691

## 1. lm200 PASSES the gate

All three arms collapse far below the measured 0.484 blank floor when the action
stream is replaced by a valid walk from an independent episode:

| variant | intact | resample actions | drop |
|---|---|---|---|
| Level15 | 0.985 | **0.155** | **−0.830** |
| Vanilla_ExtraHead | 0.995 | 0.287 | −0.708 |
| Vanilla | 0.843 | 0.343 | −0.500 |

Level15's −0.830 is comparable to Match-Query's −0.842, the most thoroughly
validated task here. Destroying the observations is equally fatal (all three to
0.26-0.28). So the lm200 numbers are not a shortcut artifact, and rule 2 is now
satisfied on a result that had never been through it.

## 2. But the capacity control matches Level 1.5, which undercuts the mechanism

`Vanilla_ExtraHead` — a generic extra attention head, no filter, MORE parameters
than Level15 — was never in the published lm200 table. It should have been:

| variant | intact acc (n=3) | final train loss (mean / worst seed) |
|---|---|---|
| **Vanilla_ExtraHead** | **0.995 ± 0.007** | 0.018 / 0.046 |
| Level15 | 0.985 ± 0.021 | 0.068 / 0.188 |
| Vanilla | 0.843 ± 0.105 | 0.682 / **0.988** |

ExtraHead − Level15 = **+0.010, t=0.79** — a tie, with ExtraHead nominally ahead.
Level15 − Vanilla = +0.142, t=2.30.

So on lm200 the gap over plain Vanilla is real, but **it is not evidence for the
Kalman mechanism**: adding a generic head with no filter in it closes the same
gap. This is exactly the pattern `EXTRAHEAD_CONTROL.md` used to overturn the
Hopfield claim ("CAPACITY, not structure"), and it now applies to the
correction's own headline regime.

`LM200_CORRECTED_MULTISEED.md` reports Level15 0.990 vs Vanilla 0.742 at T=512
and has no capacity control. That +24.8pp should not be read as the filter
working until an ExtraHead arm is run at T=512 too.

## 3. Vanilla's lm200 non-convergence REPRODUCES

Vanilla seed 2 finished at training loss **0.988** — the same failure mode that
voided every April lm200 checkpoint (stuck near 1.0 instead of ~0.005). Its
intact accuracy is 0.722 against 0.894 and 0.913 for the other two seeds, and it
is what produces Vanilla's ±0.105 spread.

The regime is genuinely basin-sensitive and plain Vanilla is the arm that gets
stuck, in a fresh batch under current code. Two consequences:

- The retraction's diagnosis was right, and the sensitivity is a live property of
  lm200 rather than a historical bug that was fixed.
- **"Level15 beats Vanilla on lm200" is partly "Vanilla sometimes fails to
  converge on lm200."** Both Level15 (worst seed 0.188) and ExtraHead (worst
  0.046) train reliably where Vanilla does not. That is a real and useful
  property — it is just an optimisation property, not the sharp-measurement story
  the landmark regime was built to demonstrate.

## What this changes

lm200 keeps its gate but loses its interpretation. It shows that plain MapWM-Flat
is unstable in the landmark regime and that adding parameters — filter-shaped or
not — fixes that. It does not show that Kalman-style correction exploits landmark
measurements, which is the claim the regime was constructed to test.
