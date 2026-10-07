# The gain-phase map on the new-object task -- pre-registration (2026-10-06, before any run of the batch)

Written, with the arms, the analysis (`analyze_gain_phase.py`), its smoke test, the leak-readout validation, the
construction checks and the power table, BEFORE the pilot was launched; the pilot and cost sections are appended after
it (see "Pilot" for what was read).

## Question
Two fixes were each measured alone. STEP side: NormStep (the step reads LayerNorm(embedding)) removes the
what-to-where leak on the new-object task (`LEAK_RESULTS.md`, REG: +0.0107 unseen-object accuracy, 8/8 SOLVED vs MapWM
8/8 DESCENDING). SCORE side: one non-negative content gain per token per head on a fixed, content-free position kernel
(GainScalar: score = mu_q mu_k (2/sqrt d_h) sum_c A_c cos(dtheta_c)) is as good as MapPoPE on the paper torus and trains
~3x faster (`GAIN_GRAIN_RESULTS.md`, REG, with a NO HEADROOM qualifier). The gain-phase map
(`docs/theory/2026-10-05/neuro_design.md` 3, E4) combines them: moves set the phase (NormStep), compared items set the
gain (GainScalar), objects never shift a field. **Does the combination work where MapWM pays a cost, and do the two
fixes act on separate defects -- the step fix removes the leak, the score fix does not, because the leak is in theta?**

## Task and recipe (= `run_leak.sh` exactly; flags diffed in the driver dry run)
New-object task (`environment_newobj.py`): 2D torus 32x32 walk, a fresh map and 16 fresh objects per sequence, objects
= fixed random 64-d codes through a learned encoder / readout (`model_codes.use_object_codes`), train pool vs a disjoint
test pool. T = 1024 steps, batch 16, 900 epochs x 98 batches, lr 1e-3, 5% warmup + cosine, 1 layer, 2 heads, d 128,
**rank 4** (as LEAK), `--data-workers 3`, pool 1000, explicit attention path. Readout on the TEST pool (unseen
objects), in distribution (x1), LEAK's eval stream (200 sequences, eval seed 10^6, env seed 10000).

## Arms: a 2x2 of STEP x SCORE (`model_gain_phase.py`; entry point `train_gain_phase.py` = `train_newobj.main()` unchanged)
| arm | step | score | class | params |
|---|---|---|---|---|
| MapWM (W) | raw: W_out W_in e | rotary: content Q, K rotated by theta | `model_rank.MapFormerWM_r4` (unchanged; LEAK's) | 217,093 |
| NormStep (N) | W_out W_in LN(e) | rotary | `model_codes.MapWM_NormStep` (unchanged; LEAK's) | 217,349 |
| GainRaw (G) | raw | gain: mu_q mu_k (2/sqrt d_h) sum_c A_c cos(dtheta_c) | `GainWM_Raw` (new) | 184,649 |
| **GainPhase (GP)** | W_out W_in LN(e) | gain | `GainWM_Phase` (new): the gain-phase map | 184,905 |
The gain score is `model_em_pope.GainKernelLayer` with M = 1 (GAIN_GRAIN's GainScalar, unchanged): mu = softplus(
Linear(LN x)) >= 0, one per token per head; A_c = softplus(a_c) >= 0 learned per head per channel (init 1); delta = 0;
32 angles per head on MapWM's PathIntegrator. All four arms: rank 4 (`w_in` 4 outputs, shared across heads, as LEAK),
32 angles per head, identical initial omega (6.2832 .. 0.1963), causal.

### Design choices, each justified
- **2x2, no ActOnly, no MapPoPE-Pair.** ActOnly (oracle action-only step) was LEAK's leak-free reference and solved 8/8 at
  0.9997, the same as NormStep; its remaining role here -- a model whose leak is exactly 0 -- is served by the leak
  readout's validation on the committed ActOnly checkpoints (below), at no GPU cost. MapPoPE-Pair is a different
  question (per-frequency gain; GAIN_GRAIN found the scalar gain as good at rank 2); adding either arm costs +8 runs
  (+25%, ~3 h) for a question this batch does not ask.
- **Retrain MapWM and NormStep** (rule 12): they are the 2x2's rotary row and D1r is a fresh-seed replication of LEAK.
- **Initial weights shared across the 2x2** (`gain_phase_equiv_out.txt` 1, every batch and pilot seed): MapWM and NormStep
  are identical except `step_ln`; GainRaw and GainPhase are identical except `step_ln`; MapWM and GainRaw share all 24
  common tensors (embedding special rows, code encoder A, readout, W_in, W_out, omega, v / o projections, norms, FFN,
  output norm). Only the score's content projections differ: q_proj / k_proj (rotary) vs q_gain / k_gain (gain; drawn
  from a separate generator seeded by the run seed + 7,919,000 with nn.Linear's default distribution) and amp_raw
  (constant). The global CPU RNG state and initial seed after construction are identical in all four arms, so every
  arm at a seed sees the same data stream, and one train-mode forward consumes the same RNG in all four.
- **Consequence (stated now):** the gain arms have 32,444 fewer parameters (the score's content projections shrink from
  2 x 128 x 128 to 2 x 128 x 2), as in GAIN_GRAIN. A SCORE effect (D2) cannot separate "gain form" from "projection
  width"; a STEP effect within a score row (D1, D1r) is unaffected.
- **Fresh seeds 8-15** (n = 8 per arm). The new-object task has used seeds 0-7 (LEAK, read) and 100 (pilots) only; 8-15
  are fresh for every arm. The pilot uses 110-111.
- **n = 8**, from the power table below: every registered decision whose expected outcome is a firing has power >= 0.96
  at n = 8 under each scenario; n = 10 or 12 changes only the non-inferiority branch under minority failures, and not by
  enough to matter (a GP failing 1 run in 8 passes AS GOOD with 0.34 / 0.25 / 0.18 at n = 8 / 10 / 12 -- AS GOOD reads
  "no failure in n runs", whatever n). Cost: n = 10 would be +8 runs (~+25%).

## Readouts (`gain_phase_eval.py`, per run; run for the batch by `eval_gain_phase.py`)
- **acc**: object-identity accuracy on the test pool at x1 (argmax over the active pool's 1000 objects at revisit
  targets whose answer is an object): LEAK's registered readout, same stream and scoring rule.
- **L_ms, the mean-substitution leak (new, registered leak readout)**: every object token's step replaced at eval by the
  mean step over all 1000 test-pool codes; L_ms = acc(substituted) - acc. It removes only the object-IDENTITY-dependent
  part of the step and keeps the part shared by all objects, so it is invariant to the per-move gauge that made LEAK's
  zeroing readout wrong for NormStep (`docs/NORMSTEP_NOTES.md` 2: zeroing NormStep's object steps "costs" -0.197 because
  it removes the shared step the actions compensate).
- **S_id, the identity spread (new, registered)**: in phase units (omega x step), rms over the 1000 test codes of
  ||Delta(o) - mean_o Delta|| over the mean of ||(Delta(+a) - Delta(-a))/2|| for the two axes. Geometric,
  score-independent (it reads only the step map), gauge-invariant (a common offset cancels in both).
- Also per run (secondary): L_zero (LEAK's readout), the field shift in cells (projection of the identity part on the
  two axis displacements) and its distortion share, the shared and blank steps, x2 / x4 code norm, train-pool accuracy,
  and the gain arms' key/query gains and spectrum.
- **Floors** (`docs/audits/2026-10-06/gain_phase_floor_out.txt`, the same stream, 47,033 object targets): retrace
  predictor 0.518, last object 0.141, most frequent so far 0.099, uniform over the sequence's 16 objects 0.0625.

### Validation of the leak readouts on the 24 committed LEAK checkpoints (`docs/audits/2026-10-06/gain_phase_leak_validate.py` / `_out.txt`; eval-only, before any run)
- V0 wiring: acc and L_zero reproduce LEAK.json's registered values exactly (max diff 0.0, 24/24).
- V1 ActOnly: L_ms = 0.0 and S_id = 0.0 exactly on 8/8 (no object step by construction).
- V2 MapWM: L_ms +0.0106 mean (median +0.0098, min +0.0074) vs L_zero +0.0107; per seed L_ms - L_zero within
  -0.0006 .. +0.0001, r = 0.998. Where zeroing is valid, the new readout agrees with it.
- V3 NormStep: L_ms +0.0002 (0.0000 .. +0.0004) where L_zero read -0.197; S_id 0.0010 vs MapWM 0.0045 (4.7x lower,
  matching NORMSTEP_NOTES 3's ~5x on two seeds). MapWM's identity step is a pure field shift (distortion share
  0.001-0.004): each object moves its own key by ~0.0045 cells.
- V4 gauge: adding c to every observation step and -c to every action step (c = the model's own shared object step)
  leaves S_id identical to 6 digits and L_ms within 0.0001 on NormStep s0-s3 and MapWM s0-s1, while L_zero moves from
  -0.20 to -0.34 (NormStep). The readout is gauge-invariant; zeroing is not.
- V5 positive controls (NormStep s0-s3, which has no leak): injecting a fixed zero-mean identity-dependent step with
  MapWM's median S_id (0.0048) is read by S_id (0.0048-0.0049) and by L_ms: +0.0008 .. +0.0028 if isotropic over the
  64 channels, +0.0143 .. +0.0222 if a pure field shift (MapWM's structure). The readout detects a leak of MapWM's size
  in a model that has none; a shift costs more than isotropic noise of the same norm.

## Registered decisions (`analyze_gain_phase.py`; all at .05, two-sided unless stated; n = 8 per arm)
Shared rules: an accuracy contrast fires if the two-sided permutation p < .05 (`perm2_p`, exact, C(16, 8)) AND |d| >=
**MIN_D = 0.005** (half of LEAK's registered leak cost, +0.0107; at n = 8 the LEAK sd of MapWM, 0.003, gives an MDE ~0.0045).
SOLVED (`classify_run`: final-5% training loss < 0.05) fires on a two-sided Fisher p < .05. `contrast_state(x, y)`:
CONFLICT (accuracy and SOLVED fire in opposite directions) -> BETTER (either fires positive; which one is printed) ->
WORSE -> CEILING (both arms >= 0.999 on every seed and SOLVED does not fire: no headroom) -> NO DIFFERENCE (95%
permutation CI and two-sample exact-t MDE printed).

**D1 STEP.** D1r: N vs W (rotary score) -- a fresh-seed replication of LEAK and the batch's positive control. D1: GP vs G
(gain score). Headline (`step_verdict`, exhaustive 5 x 5 in the smoke output):
| D1r | D1 | REGISTERED D1 |
|---|---|---|
| BETTER | BETTER | **NORMSTEP HELPS UNDER BOTH SCORES** (the step fix does not depend on the score) |
| BETTER | CEILING | NORMSTEP HELPS UNDER THE ROTARY SCORE; UNDER THE GAIN SCORE NO HEADROOM (GainRaw at ceiling: the gain score alone avoids the leak's accuracy cost; D3 says why) |
| BETTER | NO DIFFERENCE | NORMSTEP HELPS UNDER THE ROTARY SCORE ONLY (no step effect detected under the gain score) |
| BETTER | WORSE | NORMSTEP HELPS UNDER THE ROTARY SCORE BUT HURTS UNDER THE GAIN SCORE (interference) |
| BETTER | CONFLICT | conflict under the gain score |
| not BETTER | any | **REPLICATION FAILED**: the batch cannot show whether the step effect carries over; D1 reported as it falls |

**D2 SCORE.** G vs W (raw step) and GP vs N (NormStep step), each `contrast_state`: GAIN SCORE BETTER / WORSE / CEILING /
NO DIFFERENCE / CONFLICT, reported per step type. WORSE / BETTER carry the projection-width confound above.

**D3 LEAK DISSOCIATION (the "separate defects" question).** Thresholds on the median L_ms over seeds, set from LEAK's
per-seed values: **ABSENT <= 0.002** (NormStep max +0.0004, ActOnly 0) and **PRESENT >= 0.005** (MapWM min +0.0074).
- Step fix at a score (`leak_side(raw, norm)`): NO LEAK TO REMOVE if median L_ms(raw) <= 0.002; REMOVES if median
  L_ms(norm) <= 0.002 AND S_id(norm) < S_id(raw) (two-sided permutation p < .05); else INCOMPLETE. Evaluated at the
  rotary score (W, N) and at the gain score (G, GP).
- The leak under the gain score with the raw step (`gain_leak`): PERSISTS if median L_ms(G) >= 0.005; ABSENT if <= 0.002,
  split by S_id: UNLEARNED if S_id(G) < S_id(W) (p < .05), else TOLERATED (the identity step is still in theta but costs
  nothing); PARTIAL in between.
Headline (`leak_verdict`, exhaustive 3 x 3 x 4 in the smoke output; "step fix OK" = rotary REMOVES and gain REMOVES or
NO LEAK TO REMOVE):
| condition | REGISTERED D3 |
|---|---|
| rotary NO LEAK TO REMOVE | NO LEAK TO DISSOCIATE (MapWM does not leak on these seeds; D3 void) |
| step fix OK, GainRaw PERSISTS | **SEPARATE DEFECTS**: the step fix removes the leak under both scores; the score fix leaves it (it is in theta) -- predicted |
| step fix OK, GainRaw ABSENT, TOLERATED | THE GAIN SCORE TOLERATES THE LEAK (in theta, costs nothing under the gain score) |
| step fix OK, GainRaw ABSENT, UNLEARNED | THE GAIN SCORE ALSO REMOVES THE LEAK (training unlearns the identity step): not separate defects |
| step fix OK, GainRaw PARTIAL | PARTIAL: the raw step leaks less under the gain score |
| otherwise | STEP FIX INCOMPLETE -- reported as it falls |
(Some combinations are unreachable, e.g. gain-side NO LEAK TO REMOVE with GainRaw PERSISTS; the table is exhaustive anyway.)

**D4 COMBINATION.** GP vs W two-sided (`contrast_state`) and GP vs N by non-inferiority (`ni_state`): CONFLICT / WORSE /
BETTER as two-sided; else **AS GOOD** if non-inferior on accuracy at **MARGIN 0.005** (one-sided permutation shift test,
p < .05) AND SOLVED(GP) >= SOLVED(N) (slack 0, GAIN_GRAIN Amendment 1: realised size ~0.03 for bimodal outcomes; AS GOOD
reads "no failure the reference did not also have"); else UNDETERMINED (why is printed).
| GP vs W | GP vs N | REGISTERED D4 |
|---|---|---|
| BETTER | AS GOOD | **THE GAIN-PHASE MAP WORKS**: better than MapWM and as good as NormStep ("(non-inferiority at ceiling)" appended when both are >= 0.999 on every seed) |
| BETTER | BETTER | THE GAIN-PHASE MAP WORKS AND BEATS NORMSTEP |
| BETTER | WORSE | BETTER THAN MapWM, BUT THE GAIN SCORE COSTS AGAINST NormStep |
| BETTER | UNDETERMINED | BETTER THAN MapWM; non-inferiority to NormStep undetermined |
| WORSE | any | THE GAIN-PHASE MAP FAILS |
| CEILING / NO DIFFERENCE | any | NO GAIN OVER MapWM DETECTED |
| CONFLICT in either | | CONFLICT |

**D5 SPEED (registered here; descriptive in GAIN_GRAIN).** Per run, the first epoch at which the 10-epoch running mean
training loss is < 0.05 (GAIN_GRAIN's definition), censored at 901 if never. GP vs N and G vs W: two-sided permutation
on log epochs; FASTER / SLOWER if p < .05 and the geometric-mean ratio is >= 1.25 / <= 0.8; NEITHER CONVERGED if every
run of both arms is censored (expected for G vs W if GainRaw behaves like MapWM, which never reached 0.05 in LEAK); else
NO DIFFERENCE. Why registered: with both NormStep-step arms expected at ceiling, speed is where a score effect can show,
and rule 2 asks for it beside accuracy.

**Predictions (stated, not branches):** D1r BETTER; D1 BETTER; D2 raw NO DIFFERENCE or BETTER (the gain score's faster
training may shorten MapWM's under-convergence), D2 NormStep CEILING; D3 SEPARATE DEFECTS; D4 WORKS; D5 FASTER for GP vs
N (GAIN_GRAIN: ~3x).

Multiplicity: five registered decisions answering different questions, each at its own .05; no family-wise claim. Holm
over the five two-sided accuracy p's is printed as a secondary.

## Power (`docs/audits/2026-10-06/gain_phase_power.py` / `_out.txt`; computed before the pilot)
`analyse` run unchanged on 200 resampled batches per scenario. W-like / N-like = a LEAK MapWM / NormStep run drawn with
replacement (accuracy, L_ms, S_id from the validation, the checkpoint's loss curve). GainRaw and GainPhase are unknown,
so scenarios span them. Probability of the label in brackets:
| scenario | n = 8 | n = 10 | n = 12 |
|---|---|---|---|
| A predicted (G W-like, GP N-like): D1 [BETTER], D3 [SEPARATE], D4 [WORKS] | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| B G tolerates (N-like accuracy, W-like S_id): D1 [CEILING], D3 [TOLERATES] | 1.00 / 0.96 | 1.00 / 0.97 | 1.00 / 0.98 |
| C G unlearns (N-like): D3 [ALSO REMOVES] | 1.00 | 1.00 | 1.00 |
| D G half-way (+0.005 acc, half leak): D3 PARTIAL / SEPARATE; D2 raw BETTER / NO DIFF | 0.51 / 0.49; 0.49 / 0.51 | 0.56 / 0.45; 0.48 / 0.52 | 0.54 / 0.46; 0.48 / 0.52 |
| E GP fails 1 in 8 (W-like run): D4 vs N AS GOOD / UNDET / WORSE | 0.34 / 0.66 / 0.01 | 0.25 / 0.73 / 0.01 | 0.18 / 0.81 / 0.01 |
| F GP fails 1 in 4: AS GOOD / UNDET / WORSE | 0.11 / 0.82 / 0.07 | 0.06 / 0.86 / 0.08 | 0.01 / 0.83 / 0.15 |
| I GP stalls hard 1 in 8 (acc 0.60): AS GOOD / UNDET / WORSE | 0.35 / 0.61 / 0.04 | 0.21 / 0.72 / 0.06 | 0.18 / 0.71 / 0.10 |
| G GP 3x faster: D5 [FASTER] | 1.00 | 1.00 | 1.00 |
| H null (GP = N): D5 any firing; D4 [AS GOOD] | 0.00; 1.00 | 0.00; 1.00 | 0.00; 1.00 |
Reading: every expected firing is caught at n = 8 and no null fires. The weak point is minority failures in GP: a
1-in-8 failure rate is called AS GOOD 34% of the time at n = 8 (when none of the 8 runs fails) and is never called
WORSE; no affordable n fixes that (0.18 at n = 12). The half-way scenario (D) splits by construction (its median L_ms
sits on the 0.005 line). Caveat: scenario variances are LEAK's; the gain arms' own variance is unknown.

## Declared secondaries (`docs/audits/2026-10-06/gain_phase_secondary.py`, run by the driver after the done marker; no verdict)
(a) leak decomposition per arm: L_zero, L_ms, S_id, field shift (cells), distortion share, shared and blank steps.
(b) remap decomposition for the gain arms (remap_probe's three levers, analytic here): GAIN = cv over the 1000 unseen
codes of the key gain mu_k, blank vs objects, action keys and queries; WIDTH = none by construction (the kernel is
content-free; its spectrum share per 8-channel band, fine -> coarse, is printed); SHIFT = the identity field shift (the
leak expressed as remapping: "objects never shift a field" is the design's claim, read here as shift ~ NormStep's).
(c) r(final-5% loss, acc) over the 32 runs and within arm (rule 2); run classes; train-pool accuracy.
(d) x2 / x4 code norm: construction checks for N and GP (must equal x1), a distribution shift for W and G (rule 10).
(e) dropout-scale re-score (`rescore_hook`, GainKernelLayer registered; MapWM / NormStep layers already known): per-arm
means and the registered accuracy contrasts on re-scored values; the hook's skipped-class list must be empty.
(f) the STEP x SCORE interaction (GP - G) - (N - W), descriptive; Holm over the registered accuracy p's.
Not run: the field-level remap probe on rotary arms (MapWM's content phase makes it a separate analysis; GAIN_GRAIN's
remap secondary covers the paper torus).

## Construction and equivalence checks (`docs/audits/2026-10-06/gain_phase_equiv.py` / `_out.txt`, ALL PASS)
1. Matched initialisation at seeds 8-15 and 110-111 (above), RNG state identical after construction.
2. Limits, 4 held-out walks (255 tokens), eval mode, step_ln moved off its init: NormStep with step_ln = identity equals
   MapWM (max |dlogit| 0.0) and GainPhase with step_ln = identity equals GainRaw (0.0); GainRaw equals
   `model_em_pope.MapFormerWM_GainScalar(bottleneck_r=4)` with the same weights (0.0). Positive controls differ
   (0.19-0.71).
3. Causality: changing token 300 leaves every earlier logit unchanged (0.0) in all four arms.
4. Rank 4, 32 angles per head, identical initial omega in all four arms.
5. The gain score reads the code embedding: the layer's input is the CodeEmbedding output (identical tensor); object
   gains differ across codes (range 0.27) and are invariant to x2 code norm (9.7e-06, LayerNorm); the code encoder A
   receives gradient through the gain score.
6. One train-mode forward consumes identical RNG in all four arms.
Readout floors: `gain_phase_floor.py` (above). Analysis smoke test (`gain_phase_smoke.py` / `_out.txt`, ALL PASS):
every branch of every state function on synthetic data, the exhaustive D1 / D3 / D4 tables, VOID on a missing run, and
the main path end to end (load_runs + analyse) on 32 synthetic checkpoints.

## Void
Any of the 32 checkpoints missing or not 900 epochs (analyse prints VOID); md5 guard trips (at launch, before eval,
before analysis); the pilot reproduction check failing.

## What it can and cannot show
- Can: on the new-object task (rank 4, T = 1024, 1 layer, 900 epochs), whether NormStep's leak fix holds under a
  content-gain score; whether the gain score alone removes, tolerates or keeps the leak; whether the combination beats
  MapWM and matches NormStep; which of them trains faster.
- Cannot: separate the gain score's form from its 32k fewer content-projection parameters (D2); generalise beyond one task,
  one length, one depth, one budget (MapWM was still descending at 900 epochs in LEAK, so W-row contrasts are
  budget-scoped, rule 4); test robustness to code norm (by construction for NormStep steps); detect a GP failure rate
  below ~1 in 4 (power above).

## Launch (not done; rule 29: independent code audit first, findings as Amendment 1 before launch)
    cd /home/prashr/mapformer && setsid nohup bash run_gain_phase.sh > /dev/null 2>&1 &
