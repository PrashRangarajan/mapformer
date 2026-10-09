# State-change clauses in the text world -- pre-registration (2026-10-08, before any run of the batch)

Written, with the task, the gate, the arms, the readouts, the analysis (`analyze_tw_statechange.py`), its smoke test, the
construction checks, the calibration of the readouts on stored runs and the power table, BEFORE any GPU run of this
task. No pilot has been run (the GPUs are occupied by GAIN_PHASE); the pilot is specified below and its results will be
appended as an amendment before launch. CPU only so far (`CUDA_VISIBLE_DEVICES=""` for every check).

## Question
In the text world (`TEXTWORLD_RESULTS.md`, REG: path 1L 0.969 vs RoPE 1L 0.505) every action is a move. Narratives also
contain actions that change STATE without changing location ("she took the lamp", "she dropped the key"). A cognitive
map should give such actions zero spatial step -- they are actions, but not moves -- while the state they change may
matter for prediction. MapFormer's step is a learned, context-free function of the token, `Delta(w) = W_out W_in e(w)`;
nothing tells it which words move. Questions: (A) does location recall still need path integration when state clauses
are mixed into the walk; (B) do learned steps keep state-change clauses OFF THE MAP (net displacement ~0 relative to a
move), as they did for movement-free asides (`TW_NORMSTEP_RESULTS.md`: a small, role-specific offset); (C) is the changed
state actually used, i.e. bound to the place where it happened; (D) does the form of the step map matter (NormStep;
an oracle with steps only on direction words)?

## Task (`environment_tw_statechange.StateChangeWorld`, gated `docs/audits/2026-10-08/gate_tw_statechange.py` / `_out.txt`)
The text world unchanged (64x64 torus, K = 16 objects, p_empty 0.5, the directed walk, 58 words, synonyms, adverbs,
fillers, asides with p 0.15), plus STATE CLAUSES told right after a step's core clause, at the agent's current cell:
- TAKE "she {took | grabbed | lifted} the <obj> ." -- eligible when the cell currently holds an object and the hands are
  empty; the cell becomes empty and the agent holds the object.
- DROP "she {dropped | placed | released} the <obj> ." -- eligible when the cell is currently empty and the agent holds
  an object; the cell now holds it and the hands are empty.
- p_take = p_drop = 0.4 when eligible; every cell changes state AT MOST ONCE per sequence (scope: the current content is
  the latest event's, so no recency rule among several changes is needed); the agent holds at most one object.
No state word is a direction word or a movement verb ("picked UP" / "put DOWN" are excluded on purpose: the step is
context-free, so a direction word in a non-movement sense would move the phase by construction -- a separate question).
Seven new words (vocabulary 65), appended after the text world's, so every shared word keeps its id.
**Targets** are the text world's: the object word of each step at a revisited cell, which now reports the cell's CURRENT
content. Strata of the scored slots: **T1** the cell never changed before this slot (pure location; 75.4% of revisit
targets); **T2take / T2drop** the FIRST return after the cell's change (10.3% / 9.3%; the answer differs from the last
'saw' at the cell on every target, so it needs the location AND the state clause); **T3** later returns (5.0%; the last
'saw' already shows the new state). T1s = T1 targets with >= 1 state clause between the cell's first visit and the slot.
With p_take = p_drop = 0 the stream is byte-identical to TextWorld's (checked), so stored text-world checkpoints can be
read with the same readout code (the calibration below).

### Gate (rule 11; calls the task code; held-out map 10000, the eval stream np seed 10^6, 200 walks, T = 1024)
- Stream: 118.1 moves per sequence, 8.67 words per move; state clauses on 18.5% of moves (take 0.094, drop 0.091),
  asides 0.148; tokens: state clauses 10.6%, asides 9.7%, all optional words 30.7%. Revisit targets 26.9 per sequence
  (0.228 of slots): T1 4052, T2take 555, T2drop 498, T3 270 on the eval stream; T1s 2809.
- Construction checks: no direction word or movement verb inside a state clause (0); no state word in a core position
  (0); every state clause is attached to the preceding slot's cell and event (0 violations); no T2 answer equals the last
  'saw' at its cell (0); the replay check -- every slot's word rebuilt from the TOKENS (parsed state clauses), the slot
  locations and the map alone -- reproduces 200/200 walks.
- Floors (accuracy of each rule on every revisit target, read per stratum):

| rule | all | T1 | T1s | T2take | T2drop | T3 |
|---|---|---|---|---|---|---|
| best constant ('nothing') | 0.514 | 0.512 | 0.530 | 1.000 | 0.000 | 0.493 |
| word n-gram, best of orders 1-5 (fit on a training map) | 0.514 | 0.512 | 0.530 | 1.000 | 0.000 | 0.493 |
| reversal-copy (copy the object seen two moves back on a reversal) | 0.566 | 0.607 | 0.565 | 0.760 | 0.000 | 0.582 |
| reversal-copy with state (same, applying that visit's state clause) | 0.610 | 0.607 | 0.565 | 1.000 | 0.211 | 0.582 |
| last dropped object (location-free state rule) | 0.106 | 0.045 | 0.034 | 0.031 | 0.649 | 0.189 |
| stale map: first 'saw' at the cell (location oracle, no state) | 0.754 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| last 'saw' at the cell (location oracle with recency, no state) | 0.804 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
No word n-gram beats the constant. T2take is decided by the constant ('nothing'), so it cannot show state use; **T2drop
is the clean state stratum**: every state-free rule scores 0 on it (the order-5 word n-gram 0.002; corrected in Amendment 1), location-free state rules at most 0.211 when applied
uniformly (reversal-copy with state, the best rule overall) and 0.649 for the last-dropped rule (which scores 0.106 on all
targets: no model can apply it only where it is right without knowing the location).
- p_take = p_drop: 0.2 gives state clauses on 9.9% of moves and only 133 T2drop targets per 100 walks; 0.6 gives 27.4% and
  shifts the stream further from the text world (words/move 9.15). 0.4 keeps T2drop at ~250 targets per 100 walks with
  three quarters of the targets pure location.

## Arms (`train_tw_statechange.py`; one batch, all retrained, seeds 50-57 fresh for this task; pilot seed 150)
| arm | step | why |
|---|---|---|
| MapWM | learned raw step `Vanilla_r4` (r = 4 shared) | the question's subject: does a learned step map keep state verbs off the map |
| NormStep | `W_out W_in LN(e)` (`model_codes.MapWM_NormStep`) | the prediction "NormStep may help"; the 5-word state clause would carry 5 x the LayerNorm-bias tick if beta's step is not cancelled (the TW_NORMSTEP risk) |
| DirOnly | steps only on the 12 direction words (`model_textstep.MapWM_DirOnly`) | the oracle "only moves move": every state clause exactly off the map by construction; tests whether exact zero (no role offset) is better or worse than learned steps |
| RoPE | index RoPE 1 layer (canonical base 10^4) | "an index model cannot do the location part" at matched depth |
MapWM, NormStep and DirOnly share every base weight at a seed (checked, seeds 50-57 and 150). Not included, with reasons:
- **RoPE 2 layers**: in the text world it stalled at 0.772 (0/8 SOLVED). Its role here would be "how much of T2 can an
  index model with an induction circuit get", which the gate's short-range rule (reversal-copy with state, 0.211 on
  T2drop) bounds at no GPU cost. +8 runs at 2 layers (~+35% of the batch) for a non-question.
- **GainScalar score** (the gain-phase map): the question is about the step. A content-free kernel cannot prefer keys by
  role offset, so a gain arm would test whether the SCORE can bind state, a separate question; and whether gain arms
  converge at T = 1024 within 900 epochs is unknown (GAIN_PHASE pilot: slow; that batch is running now). Candidate
  follow-up once GAIN_PHASE reports.
- **NormStepNB**: TW_NORMSTEP found the per-word drift NOT from beta; the bias question is settled for this budget.
- **2 layers for the path arms**: the registered recipe is the text world's (1 layer). A pure-state target ("she dropped
  the ___": what is carried) needs a previous-token circuit (2 layers) for every arm, so it is NOT a target; the state
  targets used (T2) are retrievals at a location, which a one-layer path model can in principle solve by a role offset
  on the state clause's words (as the aside offset separates aside nouns: `docs/theory/2026-10-04/04_language.md` 2.1).
  Whether it does within 900 epochs is the content of C (and the pilot).

## Recipe (= the text world's / TW_NORMSTEP's; `run_tw_statechange.sh`)
T = 1024 words, batch 16, 900 epochs x 98 batches, lr 1e-3, 5% warmup + cosine, 1 layer, d 128, 2 heads, r = 4 shared,
`--data-workers 3`. Training map = env seed = run seed. Eval: the trainer's (200 walks, held-out map env seed 10000, np
seed 10^6 -- disjoint from every training batch stream, checked) at T = 1024 (registered) and 2048 (no verdict). The
readouts recompute everything on CPU on the same stream (`tw_statechange_readouts.py`), stratified; the trainer's
eval.json overall accuracy is a cross-check (a difference > 0.002 is printed).

## Readouts (`tw_statechange_readouts.py`, per run; all in phase units P = omega x Delta)
- **acc[stratum]**: revisit accuracy per stratum, re-scored (x 1/(1-p) on the attention output, `rescaled`, equal to
  `rescore_hook --scale auto` exactly: check 6; registered) and in eval mode (the FLAG lines).
- **shift_sc (registered geometry)**: for each state-clause variant (3 verbs x 16 nouns, take and drop), the clause's NET
  phase displacement (sum of its five words' P) projected on the MAP PLANE spanned by the two axis half-steps
  h_NS = (P[north] - P[south]) / 2 and h_EW (synonyms averaged), L2 norm in units of one move (mean L2 norm of h_NS and
  h_EW); mean over variants, (take + drop) / 2. It is how far one state clause moves the agent's position estimate, in
  moves. Gauge: invariant to (Delta k, omega / k) and to a per-move shift between movement verbs and direction words
  (opposite steps cancel the common component; a clause contains neither): check 7, deviation <= 1.3e-15. The same for
  asides (shift_aside, 3 frames x 16 nouns) in the same model: the within-model reference. A projection, not
  least-squares coordinates: least squares gave 10.4 "cells" for a 0.08 net on one stored run whose two axes are
  collinear (NormStepNB s10, axis cosine 1.000); the projection is bounded by the net's norm.
- R_sc / R_aside (mean |wrap| over channels of the net / the direction half-step: any direction, incl. off the plane),
  dist (off-plane remainder), carry_cos (cosine of the mean take net and mean drop net: -1 would be an inventory channel,
  take +u / drop -u), the step table by word class: descriptive.
- **drift** at revisits (40 walks): the phase integrated over state-clause positions only, between the cell's first visit
  and the revisit, mean |wrap| over channels, over pairs straddling >= 1 state clause; the same for asides and for
  adverbs + fillers; the TW_NORMSTEP clock-channel count (all positions, > 1 rad).
- **L_sc (functional, secondary)**: T1s accuracy with every complete state clause's NET step removed at its final '.'
  (its partial sums -- the role offsets -- kept, check 8) minus intact; L_sc0 with every state-clause step zeroed; L_aside
  the aside analogue; theta **reliance** = accuracy minus accuracy with every direction word's step replaced by the mean
  direction step (theta then advances identically on every move, check 8).
- Floors (`floors`) on the same stream and the replay check, recomputed by the analysis (the gate's numbers above).

### Calibration on stored text-world runs (`docs/audits/2026-10-08/tw_statechange_calib.py` / `_out.txt` / `.json`)
Eval-only, CPU, before any run of this task: the readouts applied to the 40 stored text-world checkpoints on this
batch's eval stream with p_take = p_drop = 0 (byte-identical to TextWorld). Asides are the analogue of state clauses that
the stored models already carry (a movement-free optional sentence naming an object). Median [min, max] per pool:

| pool (n) | accuracy eval mode | re-scored | aside field shift (moves/clause) | aside R | L_aside (net cancelled) |
|---|---|---|---|---|---|
| text world MapWM s0-7 (8) | 0.9985 [0.881, 1.000] | 0.9987 [0.972, 1.000] | 0.036 [0.006, 0.106] | 0.27 [0.09, 0.74] | -0.0036 [-0.079, -0.002] |
| TW_NORMSTEP MapWM s10-17 (8) | 0.9961 [0.830, 1.000] | 0.9966 [0.991, 1.000] | 0.022 [0.008, 0.038] | 0.26 [0.11, 0.46] | -0.0113 [-0.070, -0.003] |
| NormStep (8) | 0.9999 [0.831, 1.000] | 0.9999 [0.980, 1.000] | 0.032 [0.013, 0.099] | 0.29 [0.13, 0.50] | -0.0044 [-0.054, -0.002] |
| NormStepNB (8) | 0.9941 [0.814, 1.000] | 0.9962 [0.975, 1.000] | 0.033 [0.011, 0.098] | 0.19 [0.08, 0.52] | -0.0083 [-0.070, -0.001] |
| DirOnly (8) | 0.9718 [0.971, 0.974] | 0.9727 [0.971, 0.974] | 0 (by construction) | 0 | 0 |
| text world RoPE 1L (8) | 0.5142 [0.511, 0.517] | 0.5142 [0.511, 0.518] | -- | -- | -- |

Readings used below: (1) asides move the map plane by 0.006-0.106 of a move per sentence (31 of 32 learned-step models
<= 0.10), while their total displacement R is 0.08-0.74 of a direction half-step: the aside offset lies mostly OFF the map
plane (distortion), as the "two-level address" reading predicts; so R would call asides "on the map" and the plane
projection does not. (2) Cancelling the aside's net step costs accuracy on EVERY stored model (-0.001 to -0.079; the large
values are the below-ceiling runs in eval mode): net cancellation is not a clean "what if exactly off" counterfactual (the
models rely on, or compensate for, the offsets), so the functional readout is a declared secondary, not a decision.
(3) Eval-mode accuracy under-reports the below-ceiling runs by up to 0.17 (the 1/(1-p) attention-dropout scale,
`TW_NORMSTEP_RESULTS.md` CORRECTED block); re-scored, every learned-step run is >= 0.972 and DirOnly sits at 0.971-0.974.

## Registered decisions (`analyze_tw_statechange.analyse`; n = 8 per arm; each at .05, two-sided)
**Registered accuracy is the RE-SCORED accuracy** (attention output x 1/(1-p) at eval, `rescore_hook`'s correction applied
to the one model; exact equality with `rescore_hook --scale auto` is check 6). Reason, from the calibration and power:
eval mode under-reports one-layer text-world runs below ceiling by up to 0.17 (the CORRECTED blocks of TEXTWORLD /
TW_NORMSTEP; all committed batches re-scored in `docs/audits/2026-10-04/DROPOUT_RESCORE.md`), and with it the
TW_NORMSTEP-like DirOnly - MapWM difference is detected with power 0.85 re-scored vs 0.18 in eval mode. Every
accuracy-bearing REGISTERED line carries "[FLAG: in eval mode without the dropout-scale correction <decision> reads
<label>; registered verdict unchanged]" when eval mode gives a different label, so the text-world-comparable reading is
always printed.
Shared accuracy rule (`contrast_state`, as TW_NORMSTEP Amendment 1): BETTER / WORSE if exact permutation p < .05 AND
|d| >= 0.02; a Fisher-only firing on SOLVED (final-5% training loss < 0.05) reads "SOLVED RATE HIGHER / LOWER
(convergence; accuracy unmeasured below max(MDE, 0.02))"; CONFLICT if both fire in opposite directions; CEILING if both
arms >= 0.999 on every seed and Fisher does not fire; else NO DIFFERENCE (unmeasured below max(MDE, 0.02)).

**A LOCATION.** MapWM vs RoPE on T1 accuracy. BETTER -> **PATH NEEDED FOR LOCATION**; WORSE -> INDEX ABOVE
PATH ON LOCATION; otherwise "LOCATION CONTRAST <state>". Also printed: RoPE seeds within 0.01 of the constant floor.

**C STATE BOUND TO PLACE** (computed before B; B carries its qualifier). Per path arm with learned steps (MapWM,
NormStep), per seed T2drop accuracy minus a floor, sign-flip test over seeds, margin 0.10:
- **STATE BOUND TO PLACE** -- beats F2 = the best location-free rule on T2drop alone (last dropped, 0.649) by >= 0.10, p < .05;
- **STATE USED, NOT SHOWN BEYOND THE LAST-DROP RULE** -- not the above, but beats F1 = the T2drop accuracy of the best
  location-free rule chosen on ALL targets (reversal-copy with state, 0.211) by >= 0.10, p < .05;
- **STATE NOT SHOWN USED** -- otherwise.

**B STATE VERBS OFF THE MAP.** Per learned-step arm (MapWM, NormStep): per seed shift_sc (moves per clause).
Geometry: **OFF** if shift_sc <= SHIFT_OFF = 0.10 on >= 6 of 8 seeds; **ON** if > 0.10 on >= 6 of 8; else MIXED (k/8 off).
Aside comparison (paired by seed, sign-flip on shift_sc - shift_aside): LARGER / SMALLER THAN ASIDES (p < .05) or NOT
DISTINGUISHED FROM ASIDES. REGISTERED B line: **STATE VERBS STAY OFF THE MAP** / **STATE VERBS MOVE THE MAP** /
**MIXED: STATE VERBS OFF THE MAP ON k/8 SEEDS**, followed by "(<aside comparison>)", followed by "[state not shown used by
this arm (C): the clauses are not shown task-relevant to it]" when C reads STATE NOT SHOWN USED (an off-map reading for
clauses the model ignores is the filler result, not the claim). Threshold: 0.10 of a move per clause, set from the calibration
(asides: 0.006-0.106, 31 of 32 stored learned-step models <= 0.10), so "OFF" means "no more than a movement-free aside
moves the map", the prediction's reference ("as asides got a small offset only"). An aside-like state clause reads OFF
with probability 0.98 at n = 8 (power).

**D STEP FORM.** Accuracy on all revisit targets: D1 NormStep vs MapWM; D2 DirOnly vs MapWM (`contrast_state`).

**SUMMARY** (computed from A, B MapWM, C MapWM; no new test): PREDICTION HOLDS iff A is PATH NEEDED, B MapWM starts with
STATE VERBS STAY OFF THE MAP and C MapWM is not STATE NOT SHOWN USED; else PREDICTION DOES NOT HOLD AS STATED.
Multiplicity: A, B x 2, C x 2, D x 2 answer different questions, each at .05; no family-wise claim.

**Predictions (stated, not branches):** A PATH NEEDED; C MapWM STATE BOUND TO PLACE (a role offset on the state clause's
words lets the revisit query prefer the state event at the cell); B MapWM and NormStep STAY OFF THE MAP, not
distinguished from asides; D1 NO DIFFERENCE; D2 DirOnly WORSE (its aside errors as in TW_NORMSTEP, and with no role
offset a take clause's noun sits exactly on the cell's old 'saw' noun).

## Declared secondaries (printed by the analysis; no verdict)
Eval-mode versions of every accuracy (FLAG lines on the registered ones) and of L_sc; per-stratum contrasts DirOnly - MapWM and NormStep - MapWM on T1,
T1clean (no aside ever told at the cell), T2take, T2drop, T3; the functional readout L_sc with a COSTS / USED / NO COST
label (sign-flip, |mean| >= 0.01) beside its aside analogue L_aside (calibration above: stored runs at >= 0.99 read
-0.001 to -0.012 for asides, below-ceiling ones down to -0.08 in eval mode, so net cancellation is NOT a clean "what if exactly off"
counterfactual -- hence secondary); L_sc0 (zeroed); R, dist, drift by class, clock channels, carry_cos, theta reliance,
the step table by word class; T = 2048; r(final-5% loss, accuracy) over 32 runs (rule 2); run classes.

## Power (`docs/audits/2026-10-08/tw_statechange_power.py` / `_out.txt`; the registered functions on resampled stored data)
Pools from the calibration (per-seed accuracy on this batch's eval stream, run class from each checkpoint's losses,
aside shift): MapWM-like 16 runs (15/16 SOLVED), NormStep-like 8, DirOnly-like 8, RoPE-like 8; 2000 draws per cell;
probability of each label (n = 6 / 8 / 10 in the file; n = 8 here):
| decision, scenario | n = 8 |
|---|---|
| A, MapWM-like vs RoPE-like | PATH NEEDED 1.00 (also 1.00 at n = 6) |
| B MapWM, state shift = an aside-like draw (the prediction) | OFF 0.98 (NOT DISTINGUISHED FROM ASIDES 0.94) |
| B MapWM, state shift = 2 x aside-like | OFF 0.81, MIXED 0.18; LARGER THAN ASIDES 0.27 |
| B MapWM, state shift = 3 x aside-like | MIXED 0.48, OFF 0.50, ON 0.02; LARGER THAN ASIDES 0.57 |
| B NormStep, 1 x / 2 x / 3 x aside-like | OFF 0.99 / 0.93 / 0.38; LARGER THAN ASIDES 0.02 / 0.37 / 0.77 |
| C, T2drop ~ the seed's accuracy (bound) | STATE BOUND TO PLACE 1.00 |
| C, T2drop uniform 0.65-0.85 (partial) | BOUND 0.53, USED NOT SHOWN BEYOND LAST-DROP 0.47 |
| C, bound on 5 of 8 seeds, stale on 3 | NOT SHOWN USED 0.63, USED NOT BEYOND LAST-DROP 0.37 |
| C, not used (T2drop uniform 0-0.3) | NOT SHOWN USED 1.00 |
| D1 NormStep-like vs MapWM-like, as stored | NO DIFFERENCE 1.00 (eval mode: NO DIFFERENCE 0.83, BETTER 0.17) |
| D1, NormStep-like - 0.03 | WORSE 0.99 |
| D2 DirOnly-like vs MapWM-like, as stored (TW_NORMSTEP's -0.024 re-scored) | WORSE 0.85 (eval mode: WORSE 0.18) |
| D2, DirOnly-like - 0.03 | WORSE 1.00 |
| null, MapWM-like vs MapWM-like | NO DIFFERENCE 1.00 (eval mode: false firing 0.05) |
Reading: A, C (when the state is learned or not learned on every seed) and an aside-like B are decided at n = 8. **B's
geometry is insensitive to moderate leaks**: a state clause moving the map 2-3x as much as an aside (0.04-0.3 of a move)
still reads OFF or MIXED; ON needs > 0.10 of a move per clause on 6 of 8 seeds (a large leak). The aside comparison
detects a 3x leak in 0.57-0.77 of batches. The simulation draws state and aside shifts from DIFFERENT seeds; within one
model they will correlate, so the paired test is conservative here. A minority of seeds that fail to bind state reads
NOT SHOWN USED or the intermediate label (C is a test that every seed binds state, by design of the sign-flip test).
n = 10 changes none of these by more than 0.05 except B at 3x (NormStep LARGER 0.85), for +8 runs (+25%). n = 6 is
nearly as powered on A, B and D2 (0.83), but its sign-flip tests (B's aside comparison, C) bottom out at p = 2/64 =
0.031, so one discordant seed of six blocks every firing, and the text world's clock/map split (4/8 seeds each) would be
sampled by 6. **n = 8** (minimum sign-flip p 0.0078), the text-world batches' n.

## Construction checks (`docs/audits/2026-10-08/tw_statechange_checks.py` / `_out.txt`, ALL PASS) and smoke test
Checks (CPU, all PASS): (1) p = 0 stream byte-identical to TextWorld on two maps with the same RNG state after; the
state vocabulary only appends 7 words; (2) shared initialisation of MapWM / NormStep / DirOnly (24 shared tensors, same
torch RNG state) at seeds 50-57 and 150; (3) `fwd_steps(m, x, step_of(m, x))` equals the model's forward exactly for the
three path arms, DirOnly with an all-ones mask equals MapWM exactly; (4) DirOnly's step is exactly 0 on every
non-direction token incl. the 7 state words; (5) causality in all four arms; (6) `rescaled` equals `rescore_hook.install
('auto')` bit for bit (MapWM, RoPE); (7) B's readouts (shift, R, aside shift) invariant to (Delta k, omega / k) (deviation
0) and to a verb <-> direction per-move shift (1.3e-15) while the raw direction step moves 0.05 rad; (8) net cancellation
leaves the phase before a clause untouched and makes the phase after each clause equal to the phase before it; reliance
makes every move's direction-phase increment identical; (9) the trainer at p = 0 with TextWorld's vocabulary reproduces
`train_tw_normstep` bit for bit on CPU (2 epochs x 2 batches; losses and final weights; MapWM, NormStep, DirOnly) and the
registered configuration runs; (10) the eval np seed 10^6 is no training batch's seed.
Smoke test (`tw_statechange_smoke.py` / `_out.txt`, 47 cases, ALL PASS): every branch of `contrast_state` (BETTER, WORSE,
CONFLICT, SOLVED RATE HIGHER / LOWER, CEILING, NO DIFFERENCE, a 1% gap that must not fire), `geom_label` (OFF at 8/8 and
6/8, MIXED at 5/8 and 3/8, ON, the inclusive threshold), `aside_label` (3), `b_verdict` (3 x 2 with and without C's
qualifier), `func_label` (3), `c_label` (3 + mixed signs), A's headlines; end-to-end batches: predicted (A PATH NEEDED, B
OFF, C BOUND, D2 WORSE, SUMMARY HOLDS), a leak (B MapWM MOVE / LARGER, NormStep OFF, SUMMARY does not hold), state unused
(C NOT SHOWN, B qualified), index equal to path, an eval-mode disagreement (FLAG on D2), VOID on a missing run, on a
short run and on a non-zero DirOnly state step; and `collect()` end to end (readouts, floors with n-grams, replay check)
on four untrained checkpoints. Driver dry run (`DRV_DRYRUN=1` in a scratch mirror of the repo, scratchpad): 31 launches
logged, the run whose exact `--output-dir` a dummy python3 held (NormStep_s51) skipped, a dummy on the prefix `..._s5`
skipping nothing; knobs after sourcing lib_driver: MAXPG 4, SPACING 15, MINFREE 5500.

## Reproduction plan
Check 9 shows the new trainer equals the TW_NORMSTEP trainer at p = 0 bit for bit on CPU (tiny schedule, losses and final
weights). CPU cannot reproduce the stored GPU runs, so the GPU reproduction is DEFERRED to the pilot:
`docs/audits/2026-10-08/tw_statechange_repro.py 5 cuda:0` runs `train_tw_statechange` at p = 0 with TextWorld's
vocabulary and the TW_NORMSTEP recipe (900-epoch schedule), records every batch loss, stops after 5 epochs, and compares
the re-accumulated epoch losses with `runs/tw_normstep/p0/{MapWM,NormStep,DirOnly}_s10`'s stored `losses` with == (GAIN_PHASE's
`gain_phase_repro.py` pattern; GAIN_PHASE got 5/5 bitwise on this torch 2.10.0+cu128 stack). Pass = 5/5 for each arm. A
failure voids the batch until explained. The script itself was run on CPU with K = 1 as a test of the script (expected
to differ from the GPU-trained reference; `tw_statechange_repro_cpu_out.txt`): it runs end to end for all three arms and
gives epoch-1 losses 4.04868 / 4.04857 / 4.04841 vs the stored GPU 4.04810 / 4.04798 / 4.04782 (MapWM / NormStep /
DirOnly; ~6e-4 apart: the CPU/GPU difference, 0/1 bitwise, as expected). The GPU check is the one that counts.

## Pilot (GPU; NOT run; results to be appended as an amendment before launch)
Seed 150 (outside the batch). (a) The reproduction check above. (b) Timing: the four arms at seed 150, 900 epochs, at the
batch's concurrency (s/epoch per arm; the cost below is an estimate until then). (c) Pipeline end to end: `python3 -m
mapformer.analyze_tw_statechange --readouts --runs-dir runs/tw_statechange_pilot/p0 --seeds 150 --out
runs/tw_statechange_pilot/TWSC_PILOT.json` (per-run readouts; no verdicts at n = 1). Outcomes will be disclosed in the
amendment; no branch, threshold, n or readout changes after reading them except through a dated amendment saying so.
Pilot command (from /home/prashr/mapformer, one line per arm, GPU picked by free memory):
`setsid nohup python3 -u -m mapformer.train_tw_statechange --arm <ARM> --seed 150 --device cuda:<G> --output-dir
runs/tw_statechange_pilot/p0/<ARM>_s150 > runs/tw_statechange_pilot/p0/<ARM>_s150.log 2>&1 &`

## Void
Any of the 32 runs missing or not 900 epochs (the analysis prints VOID); the md5 guard trips (at launch, before the
readouts, before the analysis); DirOnly with a non-zero state-clause shift or a cancellation that changes its accuracy
(asserted); the replay check failing on the eval stream (asserted); the pilot's reproduction check failing.

## What it can and cannot show
- Can: whether learned context-free steps keep state-change clauses off the map plane, relative to a move and relative to
  asides in the same model; whether a one-layer path model binds a state change to the place it happened (T2drop beyond
  every location-free rule); whether index RoPE at matched depth can do the location part with state clauses present;
  whether NormStep or the direction-words-only oracle changes accuracy.
- Cannot: separate "off the map because it is an optional sentence" from "off the map because it is a state change"
  except through the aside comparison (both are non-moving optional sentences; asides have no consequences); test cells
  that change more than once (recency among state events), carried-object targets (2 layers), negated or hypothetical
  state ("did not take"), state verbs that are also movement words, or natural text. One grammar, 1 layer, r = 4, T = 1024,
  900 epochs (budget-scoped, rule 4), n = 8 with bimodal accuracy (clock vs map seeds in the text world).

## Cost (ESTIMATE; to be measured in the pilot)
Data generation is the bottleneck: 0.080 s per batch of 16 sequences on the loaded CPU (TextWorld 0.049), i.e. ~7.8 s
of generation per epoch split over 3 workers (~2.6 s/epoch) against 0.8-2.1 s/epoch measured for TW_NORMSTEP at 6
concurrent jobs. Estimate 2.5-3 s/epoch at 8 concurrent jobs: ~40-45 min per run, 32 runs in 4 waves on 8 slots
~2.7-3 h, ~3.6-4 h at 6 slots; the slots are shared (lib_driver counts every mapformer.train_ job), so concurrent sibling
batches lengthen it. Readouts on CPU ~1-1.5 min per run under the current load (~45 min for 32) + floors ~3 min.

## Launch (not done)
    cd /home/prashr/mapformer && setsid nohup bash run_tw_statechange.sh > /dev/null 2>&1 &
Only after: GAIN_PHASE finishes (or the user frees slots), the pilot amendment is committed, and an independent code audit
(rule 29) has been written into an amendment.

## Open for the independent audit
- Whether a one-layer model can learn T2 at all (C) is unknown before the pilot; if not, B carries the "not shown used"
  qualifier on every learned-step arm and the batch answers only A, B-geometry and D.
- SHIFT_OFF is set from the aside calibration on stored runs (asides are the only movement-free optional sentence with a
  measured step); a different threshold rationale (e.g. a fraction of a move that would break exact return over the
  typical number of state clauses between visits) is a candidate amendment.
- The T2take stratum cannot show state use (the constant answers it); it is reported but carries no decision.

## Amendment 1 (2026-10-08, after an independent code audit of 87d57f3, BEFORE any run; CPU only, no pilot, no launch)
The audit (blind; no batch result exists) found no bug. Finding -> change; the text above is kept for the record and the
items below REPLACE it where they conflict. Code: `analyze_tw_statechange.py`, `run_tw_statechange.sh`, the smoke test.
1. **SUMMARY ignored the aside comparison.** A state shift ~4x the aside shift (e.g. 0.09 vs 0.02) read "STAY OFF THE MAP
   (larger than asides)" and "PREDICTION HOLDS"; SHIFT_OFF = 0.10 is the TOP of the calibrated aside range (pooled median
   ~0.03), so OFF means "up to ~3x a median aside". -> SUMMARY now also requires B MapWM not to be LARGER THAN ASIDES, and
   the OFF label reads "STATE VERBS STAY OFF THE MAP PLANE (within the aside range, <= 0.10 move per clause)".
2. **B is an unweighted in-plane projection, not "net displacement ~0"** (the Question's wording, line 14-15, overstated
   it). -> The labels say MAP PLANE ("STAY OFF THE MAP PLANE" / "MOVE THE MAP PLANE (> 0.10 move per clause, above the aside
   range)" / "MIXED: ... OFF THE MAP PLANE ON k/8 SEEDS"); per seed, R (total displacement, any direction) and the off-plane
   dist are printed beside the shift, with each seed's axis |cos|. Rule: a seed is UNDEFINED -- counted neither OFF nor ON
   -- if its shift is not finite (zero axis step, hbar = 0; it was counted ON before) or its two axis half-steps have
   |cos| > 0.99 (QR's second plane direction is then arbitrary; stored example NormStepNB s10, 1.000). OFF / ON still need
   6 of 8 seeds; if fewer than 6 seeds are defined, B reads "B UNDEFINED: NO MAP PLANE ON u/8 SEEDS"; otherwise undefined
   seeds are named in the line. Undefined pairs are dropped from the aside comparison.
3. **C had no positive control (rule 14).** No existence construction is built at this stage (a hand-set one-layer model
   needs the location map as well as the role offset; the pilot's trained seed is the first evidence either way). ->
   Registered branch: if MapWM, NormStep AND DirOnly are all <= F1 (0.211) on T2drop on every seed, C reads
   "STATE BINDING UNMEASURED AT 1 LAYER (no path arm above F1 on any seed; no existence construction)" for both arms, B
   carries NO "not shown used" qualifier on that basis, and SUMMARY reads "PREDICTION HOLDS FOR A AND B; C UNMEASURED AT 1
   LAYER" (or "PREDICTION DOES NOT HOLD AS STATED (see A, B); C UNMEASURED AT 1 LAYER"). The "STATE NOT SHOWN USED" label
   and B's qualifier remain for the case where some path arm does bind state on some seed. The rule is re-checked on
   eval-mode accuracy for the FLAG line.
4. **SOLVED now includes the state targets.** The training loss (and so SOLVED, final-5% loss < 0.05) is taken over every
   revisit target, T2 and T3 included; an arm that solves location but not state may never reach 0.05, so the Fisher route
   of `contrast_state` may go inert, and the power table's SOLVED rates (from text-world runs, where all targets were
   location) need not transfer. The accuracy route decides every registered accuracy label.
5. **C's test, stated exactly (code and text agree):** per seed d = acc(T2drop) - F; a label fires if mean(d) >= 0.10 (a
   materiality floor) AND the exact sign-flip test of mean(d) = 0 gives p < .05. It is not a test against 0.10.
6. **Gate table:** the order-5 word n-gram scores 0.002 on T2drop, not 0 (fixed in place above; the floor F is unchanged).
7. **Driver.** lib_driver's `drv_wait_slot` / `drv_wait_dir` loop forever and `drv_wait_dir` counts any python3 with the run
   dir in argv. -> The driver uses its own bounded waits: a free slot within 48 h (`TWSC_SLOT_TIMEOUT`), and after the last
   launch it waits only on this batch's TRAINERS (python3 with `mapformer.train_tw_statechange` AND the run dir in argv),
   at most 24 h (`TWSC_DIR_TIMEOUT`), each failing loudly (no done marker). The duplicate-launch guard matches the exact
   absolute `--output-dir`, the relative `runs/tw_statechange/p0/<run>`, any path ending in `/runs/tw_statechange/p0/<run>`,
   and trailing slashes. The driver's first analysis call (`--readouts`) now writes the readouts only; the verdicts file is
   written once, by the second call. Dry run (scratch mirror, `DRV_DRYRUN=1`): 30 launches; a dummy holding the RELATIVE
   path of NormStep_s51 and one holding MapWM_s52 with a trailing slash were both skipped; a dummy on the prefix `..._s5`
   and a non-trainer dummy carrying the run dir in argv (an orphaned-worker stand-in) blocked nothing; the driver then
   failed at the missing checkpoints, as it must in a dry run.
Re-run after the changes, all CPU: smoke (`tw_statechange_smoke_out.txt`, 64 cases, ALL PASS; new: UNDEFINED from a
NaN shift and from collinear axes, NaN not counted ON, undefined seeds named, the B x C table with the unmeasured C, the
C-unmeasured rule, end-to-end "OFF but larger than asides" -> SUMMARY does not hold, 3 collinear-axis seeds -> B
UNDEFINED, no path arm above F1 -> C UNMEASURED with B unqualified and SUMMARY "HOLDS FOR A AND B; C UNMEASURED"; R, dist
and axis |cos| printed); construction checks (`tw_statechange_checks_out.txt`, ALL PASS, unchanged); power
(`tw_statechange_power_out.txt`: labels unchanged in substance -- B's OFF / ON / MIXED counts and the aside comparison are
computed as before; scenario numbers in the Power section stand, identical
draws). New reading from the same table: B MapWM's SUMMARY condition (OFF and not LARGER THAN ASIDES) holds with probability
0.96 for an aside-like clause, but still 0.62 for a clause at 2x the aside shift and 0.29 at 3x: the paired aside test is
the only guard against a moderate leak and it is weak (conservative simulation, see Power).. The gate is untouched (not re-run).

## Amendment 2 (2026-10-09, pilot outcomes, BEFORE the batch; no branch, threshold, n or readout changed)
Pilot run by `docs/audits/2026-10-09/queue_tw_statechange_pilot.sh` after GAIN_PHASE finished (log
`runs/tw_statechange_pilot/queue.log`), seed 150 (outside the batch).
(a) **Reproduction PASS** (`docs/audits/2026-10-08/tw_statechange_repro_gpu_out.txt`): at p = 0, `train_tw_statechange`
reproduces the stored `runs/tw_normstep/p0/{MapWM,NormStep,DirOnly}_s10` runs 5/5 epochs bitwise each.
(b) **Timing**: four arms concurrently (2 per GPU), another user's jobs present: wall time per 900-epoch run MapWM ~21
min, DirOnly ~21, RoPE ~27, NormStep ~32. At 8 slots the batch estimate stays ~3 h (4 waves).
(c) **Pipeline** end to end: `analyze_tw_statechange --readouts` exit 0 on the 4 checkpoints (`TWSC_PILOT.json`).
(d) **Outcome, READ (disclosed; n = 1, no verdict)** -- held-out map, T = 1024:

| arm | acc | T1 | T2drop | class (training tail) | shift_sc | L_sc | theta reliance |
|---|---|---|---|---|---|---|---|
| MapWM | 0.936 | 0.979 | 0.673 | DESCENDING (0.158) | 0.0099 | +0.0021 | 0.571 |
| NormStep | 0.999 | 0.9995 | 0.998 | SOLVED (0.022) | 0.0245 | -0.0025 | 0.560 |
| DirOnly | 0.971 | 0.971 | 0.928 | SOLVED (0.046) | 0 (construction) | 0 | 0.507 |
| RoPE | 0.515 | 0.514 | 0.004 | STALLED (1.709) | -- | -- | -- |

Floors on T2drop: F1 (reversal-copy with state) 0.211, F2 (last dropped) 0.649. What this settles for the open items: a
one-layer path model CAN bind a state change to its place (NormStep 0.998 on T2drop, far above F2), so Amendment 1
item 3's "C UNMEASURED AT 1 LAYER" branch is unlikely to be reached; MapWM was still descending at 900 epochs with T2drop
just above F2 (one seed). Both learned-step arms read shift_sc below SHIFT_OFF (0.10) on this seed. Nothing above is used
to change the design; the batch runs as registered (seeds 50-57, 32 runs).
