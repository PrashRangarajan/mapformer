# The gain-phase map on the new-object task -- results (2026-10-09)

Pre-registration `GAIN_PHASE_PREREG.md` (+ Amendments 1-3 from three independent audit rounds, all before launch). Runs
`runs/gain_phase/p0` (32 runs, one batch, seeds 8-15 fresh, n = 8 per arm, finished 2026-10-09 09:05). Registered output
`GAIN_PHASE_ANALYSIS.txt` (`analyze_gain_phase.py`), verdicts `GAIN_PHASE_VERDICTS.json`, per-run readouts
`GAIN_PHASE_EVAL.json`; declared secondaries `docs/audits/2026-10-06/gain_phase_secondary_out.txt`. Pilot (seeds 110-111)
disclosed in the prereg before launch.

A 2x2 of STEP (raw `W_out W_in e` vs NormStep `W_out W_in LN(e)`) x SCORE (MapWM's rotary content score vs the gain
score `mu_q mu_k sum_c A_c cos(dtheta_c)`, one non-negative gain per token per head on a content-free kernel). The
gain-phase map is NormStep + gain: moves set the phase, compared items set the gain. New-object task (32x32 torus,
fresh map and 16 fresh objects per sequence), T = 1024, rank 4, 1 layer, 900 epochs; readout = accuracy on UNSEEN
objects at object revisits.

| arm | step | score | unseen-object acc | min | SOLVED | leak L_ms (median) | S_id | leak-free loss nll_ms |
|---|---|---|---|---|---|---|---|---|
| MapWM | raw | rotary | 0.9895 +/- 0.0108 | 0.963 | 0/8 (8 descending) | +0.0069 | 0.0042 | 0.014 |
| NormStep | NormStep | rotary | 0.9940 +/- 0.0123 | 0.964 | 8/8 | +0.0002 | 0.0009 | 0.002 |
| GainRaw | raw | gain | 0.9906 +/- 0.0035 | 0.986 | 0/8 (8 descending) | +0.0100 | 0.0042 | 0.008 |
| **GainPhase** | **NormStep** | **gain** | **0.9994 +/- 0.0008** | **0.998** | **8/8** | **+0.0001** | **0.0008** | **0.0006** |

Floors (object identity at object revisits, test pool): retrace 0.518, last object 0.141, uniform over the sequence's
16 objects 0.0625. L_ms = accuracy lost to the leak (object steps mean-substituted); S_id = size of the object-identity
step relative to a move.

## Registered
- **D3 SEPARATE DEFECTS (predicted).** The step fix removes the leak under both scores (median L_ms +0.0069 -> +0.0002
  rotary, +0.0100 -> +0.0001 gain; S_id 0.0042 -> 0.0009 / 0.0008, perm p 0.0002 each). The score fix leaves it:
  GainRaw keeps MapWM's leak (L_ms +0.0100 >= 0.005; S_id 0.0042 vs 0.0042, p 0.91; field shift 0.0043 vs 0.0041 cells).
  The leak lives in theta, and no score rule removes it. GainRaw passed the leak-free convergence gate (nll_ms 0.0082
  <= 0.06; theta reliance 0.931).
- **D4 THE GAIN-PHASE MAP WORKS.** GainPhase vs MapWM +0.0099 (perm p 0.0002), SOLVED 8/8 vs 0/8 (Fisher p 0.0002);
  non-inferior to NormStep at margin 0.005 (one-sided p 0.0001), 8/8 vs 8/8.
- **D1 NORMSTEP HELPS UNDER BOTH SCORES.** Under the gain score on accuracy and SOLVED (+0.0088, p 0.0002; 8/8 vs 0/8).
  Under the rotary score (D1r, the fresh-seed replication of LEAK) **on SOLVED only** (8/8 vs 0/8, p 0.0002); the
  accuracy contrast +0.0045 is not significant (p 0.46). LEAK's +0.0107 accuracy effect did NOT replicate on accuracy:
  MapWM matched LEAK (0.9895 vs 0.9890), but NormStep fell from 0.9997 to 0.9940 because three seeds generalise less to
  unseen objects (s12 0.964, s15 0.993, s14 0.996) while still SOLVED on training loss (train-pool accuracy 0.9998).
- **D2 GAIN SCORE: NO DIFFERENCE** with either step. Raw step: +0.0011 (p 0.95; CI [-0.005, +0.009], MDE 0.012).
  NormStep step: +0.0054 (p 0.135; CI [-0.0001, +0.0139], MDE 0.013). The gain arms have 32k fewer parameters, so a
  null here is "no worse with fewer parameters".
- **D5 SPEED: NO DIFFERENCE.** GainPhase reaches training loss 0.05 at a geometric-mean epoch 643 vs NormStep 659
  (x1.03; perm p 0.021, below the 1.25 ratio the branch requires). GainRaw and MapWM never reached 0.05 (censored 8/8
  each). **GAIN_GRAIN's ~3x speed-up of the scalar gain does not carry over to this task** (rank 4, T = 1024).
- Holm over the five accuracy p's (secondary): D1 and D4 survive (0.0008); D2 and D1r do not.
- Dropout-scale re-score: every arm moves <= 0.0001; no contrast changes.

## Secondaries (no verdict)
- **GainPhase is the only arm with no weak seed.** Its sd is 0.0008 against NormStep's 0.0123: min 0.998 vs 0.964.
  NormStep's weak seeds are a train/test gap on unseen objects (train pool 0.9998, test 0.9940); GainPhase has none
  (0.9998 / 0.9994). Not registered and not tested; one reading is that the gain score's 2-wide content projections
  leave less room to memorise object codes. Unattributed.
- **Remap decomposition.** Both gain arms are pure gain with no field shift beyond the leak's (GainPhase 0.0008 cells).
  With the NormStep step the gain arm uses the finest band less (GainPhase 0.04-0.07 vs GainRaw 0.17-0.18) and the
  two middle bands more (0.35-0.42 vs 0.24-0.32).
- **Code-norm x2 / x4** (construction check): NormStep-step arms are exactly invariant (0.0000); raw-step arms lose
  0.03 / 0.12. Invariance is by construction, so it is not a robustness finding.
- r(final-5% loss, acc) over 32 runs -0.394. Accuracy is not just loss here: the raw-step arms' training loss is
  dominated by the leak (the reason the convergence gate reads nll_ms).
- Zeroing object steps (L_zero) costs NormStep-step arms ~0.19, while mean-substitution costs nothing: they carry a
  common per-token step (the gauge); only the identity-dependent part is the leak.

## What it means
- **Two defects, two fixes, one model.** The what-to-where leak is in the phase (theta) and is removed by the step fix;
  the score fix does not touch it. Combining NormStep with the gain score gives a model in which objects never move a
  field and compared items only set a gain. It is the best and most consistent arm here (0.9994, 8/8, min 0.998).
  It is as good as NormStep on average, not better (D2 NO DIFFERENCE).
- **The gain score is not a speed-up in general.** Its 3x speed on the paper torus (T = 128, rank 2) is absent at
  rank 4, T = 1024 on new objects.
- **LEAK's accuracy effect is seed-sensitive at n = 8**: the SOLVED contrast replicated exactly (8/8 vs 0/8), the
  accuracy one did not, because NormStep itself had weak seeds on unseen objects.

## Caveats
- One task, one length (T = 1024), rank 4, 1 layer, 2 heads, 900 epochs; n = 8.
- The gain arms differ from the rotary arms in content-projection width (32,444 fewer parameters): D2 and the
  GainPhase-vs-NormStep consistency cannot separate the gain form from the narrower projection.
- MapWM and GainRaw are DESCENDING at 900 epochs (budget-scoped; LEAK found the same for MapWM).
- The variance difference between GainPhase and NormStep is post hoc and untested.
