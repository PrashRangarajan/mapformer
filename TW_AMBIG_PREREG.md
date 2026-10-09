# Direction words as actions and as observed content -- pre-registration (2026-10-08, before any run)

Status: built and checked on CPU only (a GAIN_PHASE batch holds every GPU slot). NOT piloted on GPU, NOT launched.
An independent blind audit of 3f304db found no bug and two major design issues: **Amendment 1 at the end REPLACES the
text it names** (task, arms, G / non-inferiority, budget, cost). Remaining before launch: the GPU pilot, an Amendment
with it, and the user's go.

## Question
In the text world (`TEXTWORLD_RESULTS.md`) a direction word always moves the walker, so MapFormer's context-free step
`Delta = W_out W_in emb(word)` suffices. Here the SAME 12 direction words also appear as observed or reported content
that does not move the walker ("she saw a sign pointing north .", "a cold east wind blew .", "she dreamt , after a long
pause , walked north ..."). A context-free step gives "north" one fixed step, so it must move the phase on every
non-movement use and the map drifts from there on. (1) How much does that cost a context-free path model, against an
oracle told each use's role? (2) Does a context-dependent step -- HSR, the hidden-state step of the context-step pilots
(`CTXSTEP_HSR_PILOT.md`), alpha initialised at 0 -- recover it, and (3) does it learn to IGNORE non-movement uses, at a
cue 1-3 tokens away and at one 6-12 tokens away?

What is decided by construction (said here so that no one counts it as a finding): a context-free step cannot cancel a
direction-specific step with a direction-independent cue (`CONTEXT_STEP_DESIGN.md`, "Which kinds of context can fix
it"), so the swap ratio of MapWM and CF2 is exactly 1.00 (a wiring check, void if not), and the oracle's advantage on
CONTAMINATED revisit gaps is expected. Not decided by construction: its size (a trained context-free model beat the
gate's contaminated-path estimate by +0.19 on ctx3 lead/far, gate section 6), whether HSR learns the context step at the
text-world budget (the HS line failed to learn any step in 7/14 runs; HSR 4/4 at 2x this budget), whether it ignores
far-cue uses, and how (gate-inside vs cancel-later).

## Task (`environment_tw_ambig.TextWorldAmbig`, p_nm 0.3; gate `docs/audits/2026-10-08/gate_tw_ambig.py` / `_out.txt`)
TextWorld's 64x64 torus walk, map, movement clause, asides and revisit targets, plus, per step:
- movement clauses in two extra FRAMES (LEAD: "she [PAD] <move cue> [PAD] <verb> [adv] <dir> ..."; TRAIL: "later , she
  <verb> [adv] <dir> [PAD] <move cue> [PAD] saw <obj> ."), each near/far with probability p_nm/6;
- with probability p_nm one NON-MOVEMENT sentence with a uniformly drawn direction synonym, class shares: nat 1/3
  (observation content, cue 1-2 tokens: "she saw a sign pointing <dir> .", "she heard a sound from the <dir> .",
  "a cold <dir> wind blew ."); lead_near / lead_far / trail_near / trail_far 1/6 each, the same frames with
  non-movement cues ({dreamt, imagined, recalled} leading; {said, read, showed} trailing "... the sign ."). In the far
  frames the cue sits 7-12 (lead) / 6-10 (trail) tokens from the direction word and the 4 tokens before and 3 after the
  word are drawn exactly as in the movement clause of the same frame; near and far use the same tokens in a different
  order (the ctx3 construction). A lead-form reported clause names the CURRENT cell's object (true; never scored).
At p_nm = 0 the stream, vocabulary included, is byte-identical to TextWorld's (checked). 89 words, T = 1024 words.

Gate (CPU, the registered eval stream: held-out map 10000, np seed 10**6, 200 walks; n-grams fit on a training-map
sample):
- 77.2 object slots / sequence (TextWorld ~132), revisit fraction 0.222 (0.231), 17.2 scored targets / sequence.
- Non-movement share 0.224 of direction words (22.4 per sequence); per synonym 0.212-0.239 (role is not in the word).
- Walk reproduced from movement-role direction words alone: 0 mismatches in 15449 steps.
- Role from the LOCAL window (4 before + 3 after; frequency tables and a smoothed naive-Bayes over the 7 positions):
  lead_far 0.501-0.519 and trail_far 0.478-0.520 vs base rates 0.509 / 0.522 (undecidable locally); lead_near 1.000
  (from before), trail_near 0.999 (from after), nat 1.000. Sentence-level cue rule 0.9997 (the misses are sentences cut
  by the sequence end before their trailing cue). Measured cue distances: nat 1-3, lead_near 2-3, lead_far 7-12,
  trail_near 1, trail_far 6-10.
- Floors: best constant 0.5112; reversal-copy 0.6132; word n-grams 1-5 0.466-0.511 (none above the constant).
- Path oracles: clean (true cell) 1.000; CONTAMINATED (every direction word integrated as a move) 0.600, i.e. 1.000 on
  the 17% of revisit gaps with no non-movement use in them and 0.518 on the 83% that have one. Calibration on a trained
  context-free model (ctx3 pilot CF s0, T=2048): lead/far oracle 0.607 vs model 0.809 (contaminated gaps 0.538 vs
  0.776); trail/far 0.579 vs 0.619. So the estimate is not a bound. Dose (no verdict): p_nm 0.15 -> 0.676, 0.45 -> 0.546.
- No leak: targets are object words only; tagged stream = untagged after untag, tags exactly at non-movement uses;
  every lead-form reported object is the current cell's.
The prediction in the request "accuracy drops by the share of non-movement uses (0.22)" is too optimistic for a phase that
accumulates: one wrong step corrupts every later comparison with an earlier visit, and 83% of revisit gaps contain one.

## Arms (`model_tw_ambig.py`, trainer `train_tw_ambig.py`), one batch, every run retrained
| arm | step | layers | role |
|---|---|---|---|
| MapWM | context-free (`MapFormerWM_r4` = VARIANT_MAP['Vanilla_r4'], the text-world path arm) | 1 | the question |
| RoleTag | context-free over (word, role): a non-movement use reads its own step-embedding row (12 rows initialised as copies, so RoleTag == MapWM at init, max logit diff 0.0); content sees the plain word | 1 | ORACLE (told the role, for the step only) |
| DirOnlyRole | steps only on movement-role direction words (TW_NORMSTEP's DirOnly told the role) | 1 | reference (the user's 'oracle steps'; capped by asides at ~0.97 in TW_NORMSTEP) |
| HSR | `HiddenStepResWM`: Delta = W(emb + alpha LN(h1)), alpha init 0, h1 an index-RoPE layer | 2 | the context step |
| CF2 | HSR with alpha FIXED at 0 (a buffer) | 2 | one-knob control for HSR (same architecture, context-free step; == HSR at init) |
| RoPE1, RoPE2 | index RoPE | 1, 2 | no-path floor; RoPE2 the depth-matched floor for HSR |
Shared init checked (`tw_ambig_checks_out.txt` C2): every base tensor equals the context-free arm's at the seed (1L:
MapWM's 24; 2L: Vanilla_r4 at 2 layers' 40). The oracle arms read a TAGGED stream with identical random draws.
Seeds 40-47 (fresh for this line: no text-world or context-step run has used them; GAIN_GRAIN used them on the
torus, a different task); pilot seed 140. Seed outer, arm inner.
Recipe = the text world's: T = 1024 words, batch 16, 900 epochs x 98 batches, lr 1e-3, 5% warmup + cosine, AdamW wd
0.05, d 128, 2 heads, r = 4 shared, dropout 0.1, `--data-workers 3`. HSR is 2 layers by design (justified: its step reads
an attention layer); CF2 and RoPE2 are its depth-matched controls.

## Readouts
Registered accuracy: revisit object accuracy, held-out map, 200 walks, T = 1024, EVAL mode (the trainer, as every
text-world batch). SOLVED = final-5% training loss < 0.05 (`stats_core.classify_run`). Mode check: the same 200 walks in
TRAIN mode (dropout on, mean of 3 dropout seeds; trainer). Per run (`tw_ambig_readouts.py`, after the batch): accuracy on
clean vs contaminated revisit gaps; swap test per class (north <-> south at a direction word, mean |dtheta| over the 64
channels 15 tokens later; ratio = non-movement class / movement; also at the word itself); drift (channels > 1 rad
between visits), non-movement drift (phase integrated over non-movement sentences only, rad); per-move displacement and
its sharp-channel share; theta reliance (type-mean steps); 1/(1-p) re-scored accuracy; HSR's alpha.

## Primary A (accuracy, eval mode). Contrast fires: exact permutation p < .05 AND |d| >= 0.02 (`stats_core.perm2_p`)
N = RoleTag - MapWM (the need); R = HSR - CF2 (recovery, one knob); G = HSR - RoleTag; non-inferiority at 0.03 = the
permutation test at shift -0.03 rejects with d > -0.03 (inverts `perm2_ci` at the margin).
Validity: RoleTag SOLVED on >= 6/8 seeds, else **UNINTERPRETABLE: THE ORACLE DID NOT LEARN**.
- N fires (RoleTag above):
  - R fires (HSR above): G fires below -> **CONTEXT STEP NEEDED; HSR RECOVERS PART OF IT**; G fires above -> **...;
    HSR RECOVERS IT AND EXCEEDS THE ORACLE**; else non-inferior -> **...; HSR RECOVERS IT (non-inferior to the oracle at
    0.03)**; else **...; HSR RECOVERS IT (gap to the oracle unmeasured at 0.03)**.
  - R fires (HSR below) -> **CONTEXT STEP NEEDED; HSR BELOW ITS CONTEXT-FREE TWIN**.
  - R does not fire -> **CONTEXT STEP NEEDED; THE CONTEXT STEP FAILS TOO**.
- N fires (MapWM above) -> **ORACLE BELOW CONTEXT-FREE** (a defect of the oracle or the task; no claim).
- N does not fire: MapWM non-inferior to RoleTag at 0.03 -> **CONTEXT-FREE COPES**; else **UNMEASURED**.
Qualifiers (appended, no new test): MAPWM AT THE INDEX FLOOR (MapWM - RoPE1 < 0.02); CEILING (RoleTag, HSR and MapWM
>= 0.99 on every seed). Mode check: A recomputed on train-mode accuracy; if the branch differs the verdict line reads
MODE-DEPENDENT and gives both.

## Primary B (mechanism: does HSR ignore non-movement uses, by cue distance)
Seeds where HSR learned a step = move change >= 0.01 (the pilots' rule). If fewer than 7/8: **HSR DID NOT RELIABLY
LEARN A STEP; B NOT READ**. Else per class c in {nat, lead_near, trail_near | lead_far, trail_far}: HSR IGNORES c if the
median swap ratio at +15 tokens over those seeds is <= 0.3.
- all five -> **HSR IGNORES NON-MOVEMENT USES AT EVERY CUE DISTANCE**
- the three near, neither far -> **HSR IGNORES NEAR-CUE USES ONLY**; both far, no near -> **... FAR-CUE USES ONLY**
- otherwise, some -> **HSR IGNORES SOME CLASSES: (named)**; none -> **HSR TREATS NON-MOVEMENT USES AS MOVES**.

## Void
Any of the 56 runs missing; the md5 guard trips; MapWM or CF2 swap ratio outside 1.00 +/- 0.01 on any class and seed;
DirOnlyRole's non-movement swap not exactly 0 (all asserted in `analyze_tw_ambig.void_checks`).

## Declared secondaries (no verdict)
HSR - MapWM; CF2 - MapWM (does an index layer let a context-free step cope?); RoleTag - DirOnlyRole (the aside cap
replicated with roles); DirOnlyRole - MapWM; RoPE2 - RoPE1; HSR - RoPE2. Clean vs contaminated gap accuracy per arm
(prediction: MapWM ~ RoleTag on clean gaps, the loss on contaminated ones); MapWM / CF2 contaminated-gap accuracy vs the
contaminated oracle's 0.518 (above = the context-free step copes in part; the ctx3 pilot says it may). Swap ratio at the
word vs +15 per class (gate-inside: both ~0; cancel-later: ~1 at the word, ~0 at +15 -- predicted for the trailing
classes); HSR far - near ratio paired by seed (sign-flip). Non-movement drift and drift channels per arm ("the map
drifts"). Compromise step: MapWM - RoleTag sharp-channel share and displacement (does MapWM move its map to coarse
channels that one wrong step barely shifts?). Theta reliance per arm. 1/(1-p) re-scored accuracy (meaningful for the
1-layer arms only; it compounds in 2-layer models, `DROPOUT_RESCORE.md`). HSR alpha. r(final loss, acc) over all runs.
T = 2048 accuracy. SOLVED counts with Fisher.

## Power (`docs/audits/2026-10-08/tw_ambig_power.py` / `_out.txt`; the registered decision function, simulated)
Pools: oracle = the 32 text-world path runs (TEXTWORLD path s0-7, TW_NORMSTEP MapWM / NormStep / NormStepNB s10-17),
eval mode mean 0.966, sd 0.063, SOLVED 29/32 (bimodal); train mode 0.994, sd 0.009 (24 runs). Context-free arms
N(0.62, 0.03) (CF-LOW, the gate's estimate) or N(0.80, 0.06) (CF-HIGH, the ctx3 pilot). HSR: each seed learns the
context step with probability p (then an oracle-pool draw), else a context-free draw. 150-300 simulations per cell.
| n | HSR scenario | P(N) | P(R fires) CF-LOW / CF-HIGH | P(registered 'HSR RECOVERS IT') LOW / HIGH | P('FAILS TOO') LOW / HIGH |
|---|---|---|---|---|---|
| 8 | p = 1.00 | 1.00 | 1.00 / 1.00 | 0.90 / 0.90 | 0 / 0 |
| 8 | p = 0.75 | 1.00 | 0.94 / 0.78 | 0.76 / 0.67 | 0.06 / 0.22 |
| 8 | p = 0.50 | 1.00 | 0.60 / 0.42 | 0.38 / 0.33 | 0.39 / 0.57 |
| 8 | p = 0 (HSR context-free) | 1.00 | 0.01 / 0.01 | 0 / 0 | 0.95 / 0.93 |
| 8 | partial (oracle - 0.05 on every seed) | 1.00 | 1.00 / 0.89 | 0.61 / 0.57 | 0 / 0.11 |
| 10 | p = 0.75 | 1.00 | 1.00 / 0.88 | 0.66 / 0.68 | 0 / 0.11 |
| 12 | p = 0.75 | 1.00 | 1.00 / 0.91 | 0.60 / 0.71 | 0 / 0.08 |
- N is ~certain at any n (by construction). R at n = 8 has power >= 0.78 if HSR learns on >= 3/4 of seeds; at p = 0.5
  every n is ambiguous between RECOVERS and FAILS TOO (that is the right reading: half the seeds fail).
- The non-inferiority qualifier is weak in eval mode (0.23 even at p = 1: the bimodal eval-mode accuracy) and strong in
  train mode (1.00 at p = 1, the registered mode check). "Gap to the oracle unmeasured" is the expected eval-mode wording
  when HSR works; the train-mode line is where non-inferiority can be shown.
- P(UNINTERPRETABLE) 0.02-0.08 (the oracle pool's 3/32 unsolved runs).
**n = 8 per arm (56 runs).** n = 10 buys R power at p = 0.75 (0.78 -> 0.88 CF-HIGH) for +25% cost; not taken.

## Pilot (GPU, seed 140, outside the batch; `TAG=tw_ambig_pilot SEEDS="140"`; all 7 arms, 900 epochs)
Needs a GPU, so not done. It must show, before the batch (Amendment): (P1) HSR learns a step (move change >= 0.01) and
whether it is SOLVED / still descending at 900 epochs -- if the pilot HSR or RoleTag run is not SOLVED at 900 epochs the
batch runs 1800 epochs for EVERY arm (decided before launch, cost x2); (P2) the oracle solves; (P3) the measured
s/epoch at the planned concurrency (cost); (P4) the GPU half of the reproduction check:
`python3 docs/audits/2026-10-08/tw_ambig_repro.py --device cuda:0 --epochs 2` against the stored TW_NORMSTEP MapWM s10
losses (bitwise if the stored run's kernels were deterministic; the difference is reported either way); (P5) the
readouts and the analysis run end to end on the pilot's runs (n=1 path: no verdicts).

## Checks done on CPU (`docs/audits/2026-10-08/tw_ambig_checks.py` / `_out.txt`)
ALL PASS. C1 MapWM / RoPE are VARIANT_MAP's 'Vanilla_r4' / 'RoPE'. C2 shared init on seeds 40, 41 (all 7 arms). C3 RoleTag
(tagged) == MapWM (plain) logits at init, max diff 0.0; CF2 == HSR at init, 0.0; DirOnlyRole steps only at movement-role
direction words; RoleTag with nm_emb = 0 has zero step at every non-movement use. C4 every arm causal (logit and step
diff 0.0); CF2's step context-free. C5 CF2's alpha is a buffer (0 after an AdamW step); HSR's alpha moves (+0.0100).
C6 rescore_hook hooks every layer of every arm (all `model.WMTransformerLayer`, a KNOWN class: no new class to
register); the readout's rescale equals rescore_hook's on stored MapWM s12 (0.9889 both; 0.8307 without). C7 swap
ratio 1.000 on every class for untrained MapWM and CF2 (by construction); exactly 0 for RoleTag with nm_emb = 0 and
for DirOnlyRole, whose non-movement-sentence drift is exactly 0. C8 positive control, stored TW_NORMSTEP MapWM s10
through this eval at p_nm = 0: 0.9997 = registered 0.9997; theta reliance 1.000 -> 0.493 (untrained 0.013 -> 0.014);
drift 4/64 channels, displacement 2.10 rad. C9 CPU bitwise: `train_tw_ambig --p-nm 0` == `train_tw_normstep` (losses,
weights, eval accuracy; MapWM, seed 40, small config); HSR on the p_nm 0.3 task is deterministic on CPU. C10 the trainer
path re-running the stored GPU run's first epoch on CPU under its 900-epoch schedule: 4.048683 vs 4.048096 (+5.9e-4,
not bitwise across devices; the GPU-to-GPU check is deferred to the pilot, P4). C11 the full analysis report runs on a
synthetic n=8 set with every readout key and on the n=1 pilot path. Analysis smoke test, every A and B branch, both
qualifiers and the void check reached on synthetic data (`docs/audits/2026-10-08/tw_ambig_smoke_out.txt`, SMOKE PASS).
Driver (`run_tw_ambig.sh`, sources `lib_driver.sh`; knobs set after sourcing): flock single instance, least-loaded slot
picker shared with every mapformer.train_* job (MAXPG 3), md5 guard over 24 files (every
imported module, the analysis, stats_core, lib_driver and the driver itself), re-checked before every launch and before the readouts/analysis, git HEAD
and status logged to the run dir, exact-argv-token duplicate guard (tested: MapWM_s40 running does not block
MapWM_s4), skip on eval.json, done marker only after TW_AMBIG.json and TW_AMBIG_ANALYSIS.txt exist. Dry run
(DRV_DRYRUN=1, 2 seeds) enumerated 14 launches in seed-outer order and refused the marker (0/14 eval.json).

## Cost
Measured on the committed batches (not yet on this task; the pilot re-measures): TW_NORMSTEP, 1 layer at 6 concurrent
(MAXPG 3 x 2 GPUs): 2.0 s/epoch, 32 runs in 3.0 h; the text world: 1 layer 1.5 s/epoch, 2 layers (RoPE L2) 2.5 s/epoch at
4 concurrent (x1.67). Estimate: 1-layer run ~30 min, 2-layer ~50-55 min at 6 concurrent. Per seed 4 x 0.5 + 3 x 0.9 =
4.7 run-h; 8 seeds 38 run-h / 6 slots = **~6.5 h wall** + readouts and analysis on a GPU ~0.5 h. Pilot (7 runs, seed 140):
~1-1.2 h. If the pilot forces 1800 epochs: ~13 h.

## What it can and cannot show
- One scripted grammar, single-word cues, one non-movement rate (0.3 per step), T = 1024, 1 layer for the context-free
  and oracle arms, 2 for HSR, 900 epochs, n = 8.
- N is close to guaranteed; its size and the clean/contaminated split are the information. The informative primaries
  are R / G (does the context step recover the oracle's accuracy) and B (does it ignore the uses, and at what distance).
- Near vs far differ only in token order within a frame; the natural class differs in construction as well.
- HSR is one context-step design (attention then step). A window-limited step (CG, SR) is not an arm: its window limit
  is close to guaranteed by construction (`CTXSTEP_PREREG.md`, stopped for that reason).
- The non-movement uses carry no scored target; a task where the observed direction is itself content to be recalled
  (a sign's direction at a cell) would test the what/where split on one token more directly. Not this batch.

---

## Amendment 1 (2026-10-08, after an independent blind code audit of 3f304db; before any GPU run or result)
The audit found no bug. Finding -> change. Where this section and the text above disagree, this section holds.

1. **MAJOR: lead-frame non-movement uses were identifiable without their cue.** A lead-form non-movement clause named
   the CURRENT cell's object, which always repeats the previous movement object (P(repeat | non-move) 1.000 vs 0.272 for
   a move); repetition alone gave the role at ~0.86 vs base 0.50, 3-6 tokens after the direction word, and the gate's
   word-identity tables cannot see equality. lead_far was therefore not a far-cue-only class (B's FAR/NEAR branches and
   the far-minus-near secondary confounded). -> `environment_tw_ambig.py`: the reported object is now the object of a
   uniformly drawn cell of the map (the map's marginal; reported content, never scored, not at the current cell). Gate
   re-run with an EQUALITY-aware check (repetition of the last movement object; each window token == that object; all
   28 within-window token equalities, in a naive-Bayes with the positional words): P(repeat | move / non-move) lead_far
   0.241 / 0.283, lead_near 0.253 / 0.264; equality-aware role prediction lead_far **0.500** (base 0.507), trail_far
   **0.497** (base 0.503); near classes 0.75-1.00 (decidable, as designed). B's far classes are kept, now gated.
2. **MAJOR: G and non-inferiority compared the 2-layer HSR with a 1-layer oracle in eval mode.** Eval-mode
   under-reporting is depth-dependent (1-layer text-world path arms up to ~0.16, 2-layer not), and HSR has an extra
   attention layer (RoPE 2L - 1L +0.27 in the text world); the auditor's re-simulation with HSR unbiased gave
   P(non-inferior) 0.95 vs the registered 0.23. -> new arm **RoleTag2** (`model_tw_ambig.RoleTag2`: CF2 -- HSR's
   architecture with alpha fixed at 0 -- whose step reads the tagged stream as RoleTag's does; == CF2 at init, max logit
   diff 0.0). **G = HSR - RoleTag2 and non-inferiority vs RoleTag2**; N stays RoleTag - MapWM (both 1 layer), R stays HSR
   - CF2 (both 2 layers). If RoleTag2 is SOLVED on < 6/8 seeds, the R-fires branch reads "HSR RECOVERS IT (gap to the
   depth-matched oracle NOT READABLE)". The train-mode mode check is kept. 64 runs (8 arms x 8 seeds).
3. The floor qualifier compared under-reported eval-mode MapWM with RoPE1 -> it is computed in eval mode AND on the
   1/(1-p) re-scored accuracy (both 1 layer); fires as "AT THE INDEX FLOOR (both)" or "... IN <mode> ONLY".
4. The 'every direction word is a move' oracle's reversal-copy fallback read the movement-role words (role-informed)
   -> role-free fallback (last two direction words of any role). Re-gated on the amended task: **0.598** overall,
   **0.459** on contaminated gaps (843 clean / 2430 contaminated targets); calibration on the ctx3 CF pilot: oracle 0.592
   vs model 0.809 (lead/far), 0.563 vs 0.619 (trail/far). The 0.518 reference is replaced by 0.459.
5. Sentence order carried role information (a non-movement sentence never followed another; 15% of framed words) ->
   non-movement sentences now come BEFORE the step's clause, K ~ Geometric with continuation q = p_nm / (1 + p_nm)
   (mean p_nm = 0.3 per step), so the next sentence is non-movement with probability q whatever came before. Gate:
   P(previous sentence is non-movement | move / non-move) 0.216-0.242 / 0.217-0.237 across the four frames.
6. VOID did not stop the verdicts -> `analyze_tw_ambig.report` prints "REGISTERED A: VOID / B: VOID", returns 3 and the
   script exits non-zero, so the driver sets no done marker (smoke-tested).
7. The epoch count was a pilot-dependent knob -> **1800 epochs registered now for every arm** (the HSR pilots at 1800
   epochs, T=2048, were not SOLVED; the 900-epoch rule would almost surely have fired). It is the driver's default
   (md5-covered); `collect()` asserts one epoch count across all runs (config and loss-curve length). The pilot rule is
   replaced: the pilot reports whether HSR learns a step and is SOLVED at 1800 epochs; if it learns no step, the batch
   is not launched without a new decision.
8. 'sharp' (the compromise-step secondary) now wraps the per-move phases to (-pi, pi] (rule 8).
9. Timeouts: each run `timeout 8h`, readouts + analysis `timeout 6h` (driver knobs RUN_TIMEOUT / ANA_TIMEOUT); a timeout
   leaves eval.json or the analysis missing, so no done marker.

Amended gate numbers (`gate_tw_ambig_out.txt`, re-run): 77.2 object slots / sequence, revisit fraction 0.211, 16.4 scored
targets / sequence; non-movement share 0.229 (per synonym 0.214-0.243); walk from movement words 0 mismatches / 15430;
local window (frequency tables, naive-Bayes) lead_far 0.489-0.501, trail_far 0.497-0.517 vs base 0.507 / 0.503;
floors: best constant 0.4956, reversal-copy 0.5991 (movement-role words; role-free 0.5759), word n-grams 0.444-0.496;
clean oracle 1.000; p_nm = 0 stream still byte-identical to TextWorld; tagged stream checks pass.

Amended power: re-simulated with the amended decision function (`tw_ambig_power_out.txt`), RoleTag2 drawn from the same
pool as a successful HSR seed. n = 8: R fires with power 1.00 / 0.94 / 0.53 (CF-LOW) and 1.00 / 0.80 / 0.35 (CF-HIGH)
at p = 1 / 0.75 / 0.5; P('HSR RECOVERS IT') 0.92 / 0.80 (LOW) and 0.91 / 0.73 (HIGH) at p = 1 / 0.75; HSR context-free
(p = 0) -> 'FAILS TOO' 0.91. Non-inferiority vs RoleTag2: eval mode 0.19 at p = 1, train mode (the mode check) **1.00**.
The eval-mode figure is pessimistic by construction: no 2-layer path text-world runs exist, so both 2-layer arms are
drawn from the 1-layer pool, whose bimodality is the eval-mode under-report (the auditor's re-simulation with 2-layer
arms unbiased gave 0.95). n = 8 kept; n = 12 moves p = 0.75 CF-HIGH from 0.80 to 0.95 on R for +50% cost.

Amended cost (estimates from TW_NORMSTEP's 2.0 s/epoch for 1 layer at 6 concurrent and x1.67 for 2 layers; the pilot
re-measures): at 1800 epochs a 1-layer run ~1.0 h, a 2-layer run ~1.8 h. Per seed 4 x 1.0 + 4 x 1.8 = 11.3 run-h;
8 seeds 90 run-h / 6 slots = **~15 h wall** + readouts and analysis ~0.7 h. Pilot (8 runs, seed 140): ~3 h. Dropping the
two secondary-only arms (DirOnlyRole, RoPE2) would save ~2.8 run-h per seed (~11 h wall); not done.

Checks re-run (`tw_ambig_checks_out.txt`): ALL PASS, now including RoleTag2: shared init with the 2-layer reference, RoleTag2 (tagged) == CF2 (plain) at
init (max diff 0.0), causal, 2 layers hooked by rescore_hook, non-movement swap exactly 0 with nm_emb = 0; C9 CPU bitwise
reproduction and HSR determinism on the amended stream; C10 unchanged (+5.9e-4). The trainer now returns nan instead of
dividing by zero when a (check-sized) eval set has no revisit target. Smoke (`tw_ambig_smoke_out.txt`): every A branch incl. the new
depth-matched ones (EXCEEDS / PART / NOT READABLE / unmeasured), both floor-qualifier forms, the ceiling qualifier,
every B branch, the void check and VOID stopping the verdicts with exit code 3: SMOKE PASS.
