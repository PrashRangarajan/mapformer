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
  -0.20 to -0.34 (NormStep). The readout is gauge-invariant; zeroing is not. (Amendment 1: the per-move gauge is exact
  only for observation-to-observation scores; an action query sees the offset c once, which is why accuracy moves by up to
  0.0001 here.)
- V5 positive controls (NormStep s0-s3, which has no leak): injecting a fixed zero-mean identity-dependent step with
  MapWM's median S_id (0.0048) is read by S_id (0.0048-0.0049) and by L_ms: +0.0008 .. +0.0028 if isotropic over the
  64 channels, +0.0143 .. +0.0222 if a pure field shift (MapWM's structure). The readout detects a leak of MapWM's size
  in a model that has none; a shift costs more than isotropic noise of the same norm.

## Registered decisions (`analyze_gain_phase.py`; all at .05, two-sided unless stated; n = 8 per arm)
Shared rules: an accuracy contrast fires if the two-sided permutation p < .05 (`perm2_p`, exact, C(16, 8)) AND |d| >=
**MIN_D = 0.005** (half of LEAK's registered leak cost, +0.0107; at n = 8 the LEAK sd of MapWM, 0.003, gives an MDE ~0.0045; Amendment 1: in the NormStep row the sd is ~0.0002, so the MDE there is ~0.0003 and MIN_D, not the
MDE, decides whether a NormStep-row contrast fires).
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
2. Limits, 4 held-out walks (511 tokens; corrected in Amendment 1 from "255"), eval mode, step_ln moved off its init: NormStep with step_ln = identity equals
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

## Pilot (`docs/audits/2026-10-06/gain_phase_pilot.sh`, `gain_phase_pilot_long.py`; runs `runs/gain_phase_pilot`; seeds 110-111)
Launched AFTER this pre-registration, the analysis, every branch, the power table and n were committed (d3b617e); nothing
in them was changed after it. Part 1 used the batch's flags with a 30-epoch schedule; part 2 (added after part 1 was
read, because of what it showed) ran the batch's real 900-epoch schedule for the two gain arms, planned to stop at epoch 150, stopped early at ~26 (below).
(a) **Reproduction** (`gain_phase_repro.py` / `_out.txt`, before the pilot; pass = bitwise-equal per-epoch losses):
through `train_gain_phase` (which imports `model_gain_phase`), MapWM s0 and NormStep s0 at the LEAK recipe reproduce the
stored `runs/leak/p0` runs **5/5 epochs bitwise each** (torch 2.10.0+cu128; Amendment 1: its user-site dist-info is dated 2026-02-24, so LEAK almost certainly ran on the same stack -- nothing changed between LEAK and this batch).
(b) **Timing** (part 1, 8 of our jobs = 4 per GPU, plus another user's 12 non-mapformer jobs on the same GPUs at ~98%
utilisation, which lib_driver does not count): MapWM 14.1-14.5 s/epoch, NormStep 12.1-12.2, GainRaw 15.4-16.7, GainPhase
13.7-14.5 (GPU placement differs by arm); mean 14.1. GPU memory ~21.5 of 24.5 GB per GPU at 4 jobs each. Eval of 8 runs:
26 s. LEAK's logs: 10.6-11 s/epoch at the same concurrency without the other user's load.
(c) **Pipeline** end to end on the 8 pilot checkpoints: `eval_gain_phase` (all readouts), `analyse` (every decision prints;
n = 2 cannot fire), `gain_phase_secondary.py` (all six parts; the re-score hooked all 8 models, skipped none; x2 / x4 of
the NormStep-step arms within 0.0001 of x1, as constructed).
(d) **Outcome, READ (disclosed):** part 1 (30-epoch schedule; LR warmup 1.5 epochs, decayed by epoch 30), final training
loss / unseen-object accuracy: MapWM 0.46, 0.72 / 0.960, 0.934; NormStep 0.80, 0.31 / 0.921, 0.999; **GainRaw 1.50, 3.26 /
0.736, 0.273; GainPhase 3.56, 3.60 / 0.160, 0.113.** On this short schedule both gain arms trained far slower than the
rotary arms -- the reverse of GAIN_GRAIN (T = 128, rank 2), where the scalar gain was fastest. Part 2 (the batch's real 900-epoch schedule,
GainRaw and GainPhase s110 only, planned 150 epochs) was **STOPPED EARLY at epoch ~26 of 150 on the user's instruction
(no GPU work while another user's jobs share the GPUs)**; its last logged training losses (epoch 25): GainRaw 2.13,
GainPhase 2.02 (epochs 5 / 10 / 15 / 20: 4.16 / 3.19 / 2.64 / 2.42 and 4.17 / 3.66 / 2.63 / 2.32). LEAK's logs at epoch 25,
same schedule, seeds 0-7: MapWM 1.98-3.97, NormStep 1.94-3.99. So under the real schedule the gain arms were NOT behind at
epoch 25; part 1's gap is plausibly the 30-epoch schedule (1.5-epoch warmup, LR decayed by epoch 30), but whether the gain
arms converge by 900 epochs at T = 1024 is unknown (GAIN_GRAIN's evidence is T = 128, rank 2). Part 2's orphaned data
workers were terminated by pid after the stop; nothing was saved.
No branch, threshold, n or readout was changed after reading either part.

## Cost (ESTIMATE from the partial pilot: part 1's 30 epochs at full concurrency; part 2 stopped at epoch ~26)
32 runs x 900 epochs at ~14 s/epoch (pilot part 1 mean; gain arms ~15-16) = ~3.5 h per run (gain arms ~3.9 h). With all
8 slots (4 per GPU): 4 waves -> **~15 h** of training, + launch spacing and queue tail ~0.3 h, eval ~2 min, analysis and
secondaries ~5 min. **The slots are shared** (lib_driver counts every mapformer.train_ job): RANK_NOWRAP (the other
batch being built now, 48 runs at T = 1024, 900 epochs, same picker) would take half the slots if launched together, giving
**~28-30 h** for this batch (8 waves on ~4 slots); both batches together occupy the 8 slots for ~35 h. The other user's
load on the same GPUs is outside lib_driver's count and may change the per-epoch time either way (part 2 ran at ~7.5
s/epoch with only 2 of our jobs on the GPUs). Per the user's instruction, the batch is not to be launched while another
user's jobs are on the GPUs; the ETA counts from a launch on otherwise free GPUs, where LEAK's 10.6-11 s/epoch gives
~2.7 h per run and ~11 h for 4 waves alone.

## Open for the independent audit (raised by the pilot, not acted on: no branch changed)
- D3 and convergence: if GainRaw does not converge by 900 epochs, a PERSISTS reading (L_ms >= 0.005, S_id high) may reflect
  an unconverged step map rather than "the leak is in theta" (in pilot part 1 every arm's S_id was 0.25-1.5, i.e.
  unconverged steps, 50-300x LEAK's converged values). LEAK's MapWM was itself DESCENDING at 900 epochs, so the same
  caveat already applies to the rotary row. A qualifier (e.g. D3 carries "confounded with convergence" when GainRaw is
  not SOLVED on a majority of seeds) is a candidate for Amendment 1.
- Whether the gain score converges at T = 1024 within 900 epochs is unmeasured (pilot part 2 stopped at epoch ~26).

## Launch (not done; rule 29: independent code audit first, findings as Amendment 1 before launch)
    cd /home/prashr/mapformer && setsid nohup bash run_gain_phase.sh > /dev/null 2>&1 &

## Amendment 1 (2026-10-06, after an independent code audit, BEFORE launch; CPU only, per the user's instruction)
The audit was blind to batch results (none exist) and saw only the pilot. Changes, each answering a finding; all made
before launch; the analysis, eval and driver are md5-guarded from launch.
1. **D3 could read a gain arm that never used theta (DESIGN, major).** Untrained gain models read L_ms 0.0000 with S_id
   ~1, so D3 could print "THE GAIN SCORE TOLERATES THE LEAK" or "ALSO REMOVES" for an arm that does not path-integrate.
   (a) New per-run readout **theta reliance** = acc - acc with every token's step replaced by its token-type mean (the
   four action steps by their mean, objects by the mean object step; blank unchanged), so theta carries no position
   (`gain_phase_eval`, mode 'typemean'). Validated on CPU (`gain_phase_leak_validate_out.txt` V6): LEAK MapWM 0.923-0.944,
   NormStep 0.941-0.972, ActOnly 0.940-0.970; **untrained models of all four arms (seeds 8, 110): |reliance| <= 0.0001**
   (accuracy <= 0.0013, L_ms |.| <= 0.0001, S_id 0.98-1.84). (b) **Convergence gate** (`converged_gate`): a gain arm's
   D3 labels are read only if median acc >= min acc(MapWM) - 0.01 (the house materiality floor below the reference's
   worst run), median theta reliance >= 0.2 (converged path models 0.92-0.97, untrained 0.0000, pilot GainPhase 0.04-0.08),
   and median S_id <= 3 x max S_id(MapWM) (LEAK MapWM max 0.0050 -> 0.015; untrained >= 0.98; a persisting MapWM-sized leak
   sits at <= 1x). (c) Otherwise the registered D3 headline is **"D3 UNMEASURED: <arm> not converged (median acc, gate
   values, k/n SOLVED, j/n DESCENDING)"**; the gain-side labels are still printed, marked "[label not read: gate]". The
   rotary-side label and the NO LEAK TO DISSOCIATE void are unchanged. On the real 30-epoch pilot checkpoints (CPU
   re-evaluation, `gain_phase_pilot_rescore_a1.py`, `runs/gain_phase_pilot/analysis_out_A1.txt`) both gain arms fail
   the gate (GainRaw on accuracy; GainPhase on accuracy and reliance 0.06) and D3 reads UNMEASURED.
2. **Budget scope on D1, D2, D4 (DESIGN).** `budget_flag` appends "(budget-scoped: <arm> k/n SOLVED, j/n DESCENDING at
   900 epochs; ...)" to a REGISTERED line when a gain arm is less trained than its rotary counterpart: GainPhase vs
   NormStep by DESCENDING count (NormStep was 8/8 SOLVED in LEAK); GainRaw vs MapWM by loss regime (median final-5%
   loss above MapWM's largest, with a DESCENDING run), because MapWM never reaches 0.05 and is itself 8/8 DESCENDING.
   D1 carries both arms' flags, D2 each row's, D4 GainPhase's. Labels unchanged.
3. **Negative L_ms (DESIGN).** ABSENT and NO LEAK TO REMOVE now need |median L_ms| <= 0.002; a median < -0.002 is a new
   label **NEGATIVE** (removing the identity step costs accuracy: the identity step is used), in `leak_side` and
   `gain_leak`; any NEGATIVE makes the D3 headline "NEGATIVE L_ms ... reported as it falls".
4. **Duplicate launch (driver, rule 21).** The driver also skips a run whose `--output-dir` appears in a running
   python3's argv (checked by comm, never pgrep -f), before its log is opened; tested with a dummy process in a scratch
   dry run (31 launches, the running one skipped).
5. **Eval device / memory.** `eval_gain_phase` defaults to `auto` = the CUDA device with the most free memory
   (`pick_device`; `GAIN_PHASE_DEVICE` overrides); the driver waits until some GPU has > 6 GB free before eval;
   DRV_MINFREE = 5500 MiB per launch (pilot: ~4.7 GB per job). The secondary uses the same picker.
6. **TOLERATED must separate a cheap isotropic spread from an ignored field shift.** `gain_leak` prints GainRaw's
   distortion share and field shift (cells) beside MapWM's; a TOLERATED headline carries "[mostly isotropic spread |
   mostly a field shift the gain score ignores: distortion share x, field shift y cells vs MapWM z]" (share >= 0.5 vs
   < 0.5; V5: an isotropic spread of MapWM's size costs 0.001-0.003, a field shift 0.014-0.022).
7. **Wording.** combo_verdict WORSE x AS GOOD now reads "WORSE THAN MapWM BUT AS GOOD AS NormStep: NormStep is itself
   below MapWM on these seeds (see D1r)". The REGISTERED D1 line states which test made D1r fire ("[D1r fires on accuracy
   and SOLVED]", or "neither test").
8. **Dropout-scale re-score on the registered lines.** The eval now also records `acc_rescored` (o_proj input x 1/(1-p)
   on this model only; equals `rescore_hook`'s re-score exactly on MapWM s0 and NormStep s0, V7). If any accuracy-bearing
   label (D1r, D1, D2 raw / NormStep, D4 vs MapWM / vs NormStep) differs when recomputed on re-scored accuracies, its
   REGISTERED line carries "[FLAG: under the dropout-scale re-score <decision> reads <label>; registered verdict
   unchanged]".
9. **Prereg text.** The equivalence walks are 511 tokens (position 300 is an action), not 255; the per-move gauge is
   exact only for observation-to-observation scores (an action query sees the offset once; V4 accuracy moves <= 0.0001);
   torch 2.10.0+cu128 is in the user site since 2026-02-24 (dist-info date; the audit's note said 2026-04-08), so LEAK
   almost certainly ran on it and nothing changed between LEAK and this batch; the NormStep-row MDE is ~0.0003, so there
   MIN_D, not the MDE, decides firing.
Re-run after the changes, all CPU (CUDA hidden): equivalence checks (`gain_phase_equiv_out.txt`, byte-identical to the
pre-amendment output, ALL PASS); leak-readout validation with V6 / V7 added (`gain_phase_leak_validate_out.txt`; V0 is
now CPU vs LEAK.json's GPU eval: max difference 8.5e-05, about 4 of 47,033 targets; every other number within 0.0001 of the GPU run); the
branch smoke test with every new branch, the exhaustive 4 x 4 x 5 D3 table and three end-to-end synthetic batches
(predicted; unconverged gain arms -> D3 UNMEASURED and budget flags on D2 / D4; a re-score flip -> FLAG on D4, verdict
unchanged) (`gain_phase_smoke_out.txt`, ALL PASS); power (`gain_phase_power_out.txt`, below).

### Power after Amendment 1 (n = 8; n = 10 / 12 in the file)
The gate costs nothing where the gain arms behave like a converged LEAK run: scenarios A-I give the same labels with the
same probabilities as before (A predicted: D1 BETTER, D3 SEPARATE DEFECTS, D4 WORKS, each 1.00; B: D3 TOLERATES 0.96;
C: ALSO REMOVES 1.00; E: AS GOOD 0.34). New scenarios: **J, both gain arms untrained-like** (acc 0.14, reliance 0, S_id 1.0,
L_ms 0, loss still descending): D3 **UNMEASURED 1.00** (before the amendment the same inputs gave a gain-side reading);
D4 FAILS 1.00 with the budget qualifier. **K, GainRaw alone untrained-like:** D3 UNMEASURED 1.00, D4 WORKS 1.00.
