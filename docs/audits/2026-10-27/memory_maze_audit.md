# Memory Maze batch: independent code check (rule 29)

Checked 2026-10-27 against commit 8ebaba5 (not launched). Spec: `MEMORY_MAZE_PREREG.md` (+ Amendment 1),
`MEMORY_MAZE_GATE_V3.md`, `docs/audits/2026-10-27/memory_maze_batch/`. Code: `mm_data.py`, `mm_encoder.py`,
`model_mm.py`, `train_mm.py`, `test_mm.py`, `analyze_mm.py`, `analyze_mm_pilot.py`, `mm_floors.py`, `mm_gate_v3.py`
(+ the v1/v2 helpers they import), `hyak/mm_runs.slurm`, `hyak/mm_sync.sh`, `hyak/mm_batch_tasks.txt`,
`run_mm_pilot*.sh`. No training was launched; nothing was run on a GPU; no tracked file was edited.

**Verdict: no BLOCKER. Launchable after an Amendment 2 records the three MAJOR items (all are interpretation /
readout items; none requires a code change to keep the registered primary valid). The MINOR fixes are cheap and
should go in before launch if code is touched at all (m1, m2, m4 especially).**

## What was run (CPU, `CUDA_VISIBLE_DEVICES=""`)

- `test_mm` on a 3-episode zip cut from train-part0 (members 0-2) plus a `mm_encoder.py prep --smoke` array dir, both
  in the session scratchpad: **ALL PASS** (scored steps = gate rows; E feature x fitted map = gate class-E dead
  reckoning; causality for all 7 arms; blind frames invisible; ActOnly; GRU 197,008 vs 198,272; GPS alignment;
  InfoNCE mask and 'both'; install frozen/trainable; scoring oracles; resume reproduces an uninterrupted run).
- `analyze_mm --selftest`: **SELFTEST ALL PASS** (all five primary labels, the three secondary label forms, NOT RUN).
- Full-split scored-set check from the batch arrays (float32 pos/dir) with `mm_data.scored_steps` and the gate's
  archive-index schedule: eval **5132** non-local re-entries (gate diag 5132), held 1600-2899 **6900** (floors_held
  6900); all-returns 5976 / 8001. The batch's scoring rows equal the gate's on both full splits, not only on the
  2-episode unit test.
- Pilot provenance: md5s and configs of all 17 pilot `results.json` files read.

## Builder's list, answered

1. **Amendment 1 loss.** Applied to every arm identically: `--loss-steps` defaults to `both` and `--epochs` to 600
   (`train_mm.py:257,269`); the task table has no extra arguments and `mm_runs.slurm` passes only `$EXTRA` (empty), so
   no arm can differ. Disclosed in the prereg (Task, Recipe, Amendment 1), but NOT in the trainer's docstrings
   (`train_mm.py:11-13`, `:92` still describe sighted-only as the default) nor in `analyze_mm`'s printed primary label
   (m3).
2. **Scoring fidelity.** Correct. The schedule is the gate's (`mm_gate_v2.sighted_mask` seeded by archive index;
   eval rows 0-999 = archive 0-999, held rows = archive 1600-2899, matching `mm_gate_v3` with `--score-zip` and
   `mm_floors.py`). Including f_t, future frames and blind frames among R1's candidates is right for this readout and
   is not a leak: keys are `Linear(LayerNorm(frozen embedding))`, a content-only function of the frame, never of the
   model state; the query at s_t (token 2t-1) cannot see f_t (token 2t) -- SDPA `is_causal=True` (`model.py:250`),
   tested by perturbing f_400.. and checking tokens < 800; blind frame tokens carry the blank and their content is
   invisible (tested). Same-pose blind frames inside the run count as correct answers; the model never saw them, so
   credit there means "the appearance of this pose", which is the intended 'where' readout. The gate's floors use the
   same candidate set (`mm_gate_v2.episode`, `D[u]` over all T), so floor and model readout are comparable in
   definition (not in feature space -- disclosed in the prereg).
3. **Arm equivalence.** P_odo and GPS both get d_t in step content (`model_mm.py:40,44`); S4 (P_odo vs GPS) is therefore
   matched on content and differs only in how position enters (relative rotary phase vs additive absolute sinusoid on
   every token, blind frame tokens included). GPS is "position code + odometry content", as the prereg table says;
   it is more than R1 has, which is the point of S4, not of the primary. GPS alignment: f_0 -> p_0, s_t and f_t ->
   p_t, with p = cumsum(d) = pos_t - pos_0 (tested). Index RoPE: token indices 0-2000, inv freq 10000^(-k/32) =
   canonical base^(-2c/d_head) at d_head 64. GRU hidden 184: core 197,008 vs 198,272 (0.994).
4. **Probe fidelity** (Pasukonis 2210.13383 Sec. 3.3, Table E.3, read first-hand for this check): targets, BCE/MSE,
   Walls = accuracy over tiles, Objects = MSE averaged over objects (squared distance per object; reproduces the
   constant 64.8 as 65.10), 4x1024 MLP with LayerNorm + ELU, input = representation concatenated with agent position
   and orientation, scored on the second half of eval episodes: all match. Deviations in m6.
5. **Destruction modes.** Correct as descriptives: `frames_swapped` rolls the input frames by one episode within the
   batch, keeps own keys, own blind mask and own steps; `steps_shuffled` permutes all step features (e, odo, raw) in
   time with a fixed generator (same permutation sequence for every arm). Note: for GPS and P_odo, shuffling d_t also
   scrambles the given position; for P_raw the content still carries heading (n2).
6. **Pilot code states.** Four, not three (m1): wave 1 ran on code without `--loss-steps` (configs lack the key) yet
   records train_mm md5 d8e8c3, because `results.json` hashes the files at the END of the run (`train_mm.py:384`);
   wave 2 and two wave-3 runs record d8e8c3, the other wave-3 runs 1909c1; the batch code is f47340 (none of them).
   model_mm / mm_data / model.py md5s are identical across all pilots and the batch (936c58 / 396fd0 / 0a379a).
7. **House-default deviations.** Dropout 0 (disclosed, reason given), SDPA (exact to ~1e-6 forward), fp32 -- except the
   GRU arm, which runs TF32 through cuDNN by default (m2), and the step features, stored fp16 (m3b). SDPA's fp32 GPU
   backward is not bitwise run-to-run deterministic without `--deterministic` (rule 19); the prereg's "no RNG
   difference" is true but does not imply bitwise reproducibility (n4).
8. **Split hygiene.** Verified: encoder and E map on train-part0 0-299 (inside the training range), training 0-1599,
   held 1600-2899 untouched by any fit or gate (floors fit on 300-599), pilots validated on 1500-1599, eval archive
   names from the 2022-09-20 run (`eval_info.json`). The eval split WAS read by gates v1/v2 and the v3 diagnostic
   (map-free statistics only; already disclosed). Trap: `mm_encoder.py:227` silently embeds train-part0 as "eval" if
   `eval.zip` is missing (n3).

## MAJOR

**M1. The readout, not the step map, sets every arm's level; the rule-14 install shows it, and the primary must be
read accordingly.** Pilot (s3693, validation): installed exact class-E map FROZEN 0.283, scratch P_E 0.311, installed
TRAINABLE 0.376; the gate's class-E integrator on eval is 0.659 (train split 0.696) and the odometry integrator 0.927,
while P_odo reaches 0.537. So a frozen phase that carries the class-E dead reckoning exactly is read out at less than
half of what that phase supports, and a FROZEN exact E map underperforms scratch training. Two consequences:
(a) the one-layer retrieval (attention + learned query/key heads under InfoNCE) is the bottleneck for every arm, so
P_E - R1 measures "how much position information one learned retrieval layer extracts from a path phase vs an index
phase", not integration accuracy; (b) the scratch P_E solution need not be class E (frozen E < scratch), and the
trainable install ending 0.065 above scratch is a search signal (one seed, SE ~0.02 per point) of the kind seen in the
rank line. Rule 14 is met only in the weak sense "the class holds a solution at scratch's level"; the install's
weight scale was not varied (rule 14 asks for the scale training would use).
*Failure scenario:* PATH WINS is reported as "path integration of given self-motion retrieves the view", when the
gain is partly the phase's usability to a weak readout. *Fix:* disclose in Amendment 2 (text below). Optional,
declared descriptive, not a registered contrast: P_E installed-frozen at seeds 3600-3607 (8 runs, ~4-6 A40-h); it has
exactly R1's trainable parameter count (304,128 vs 304,128), so P_E-frozen - R1 is the cleanest "fixed path phase vs
index phase at identical trainable parameters" reading available.

**M2. Claim scope after the pilots: R4 and GPS above P_E, and three registered predictions already contradicted.**
Pilot (one seed each): R4 0.444, GPS 0.365, P_E 0.311/0.320, R1 0.170/0.135. The prereg already says the R4
prediction is contradicted and that the registration does not change -- correct. What is missing:
(a) R4 has 2.95x the parameters (898,944 vs 305,120), so S1 is "depth AND parameters", not a depth control; (b) if
S1 reads R4 ABOVE P_E, the house claim "1 layer of path integration ~ 3 layers of attention" (Dyck, H3 in CLAUDE.md)
does not hold on this task, and that must be stated at the headline level of any report, not only among failures;
(c) "P_E near the class-E integrator (0.66)" and "P_odo near 0.93" are also contradicted by the pilots (0.31; 0.54);
(d) PATH WINS, if it fires, is a one-layer statement at E = 600, budget-scoped.
*Fix:* Amendment 2 note (below). No change to labels, thresholds or arms.

**M3. The batch has no convergence readout that can fail.** `analyze_mm.py:149` counts DESCENDING from the final-10%
training-loss slope, but the schedule is cosine to 0, so the last 10% runs at <2.5% of peak LR and is flat by
construction -- the rule-3 trap ("a flat final window read after LR decay is NOT convergence"). The batch has no
validation curve (`--val-every 0`; the only unused training-split indices are inside the held range, which the trainer
asserts against, `train_mm.py:308`). The pilot "plateau epoch" (`analyze_mm_pilot.py:21`) uses a 0.01 tolerance
below the per-point SE (~0.02), so it is not a convergence test either; in the pilots R4 (0.426 -> 0.444), GPS
(0.352 -> 0.365) and GRU (0.135 -> 0.150) were still rising over the last 100 epochs, and P_E jumps late (epochs
150-250).
*Failure scenario:* S1/S2/S4 read on runs still improving; a late-jumping or never-jumping P_E seed makes the
primary bimodal with no way to tell STALLED from slow. *Fix (pick one, before launch):* (i) read the training-loss
slope over 60-85% of training (before the LR collapses) and declare that window; or (ii) carve 2800-2899 out of the
held set (held becomes 1600-2799, floors recomputed on it) and validate every 50 epochs on it, descriptive only, no
selection; or (iii) state every secondary as budget-scoped at E = 600 and report the pilot curves beside it. (i) costs
nothing and needs only an analysis change.

## MINOR

**m1. Code provenance is recorded at the end of the run.** `train_mm.py:384` hashes `CODE_FILES` when writing
`results.json`, so a file edited during a run is recorded as the code that ran (wave-1 pilots show d8e8c3 with no
`loss_steps` in their config). The batch (f47340) matches no pilot code state, and the pilot states cannot be
reconstructed from git (the files first appear in 8ebaba5). *Fix:* hash at start into `cfg`, re-hash at the end and
record both (or refuse on mismatch); state in Amendment 2 that the pilots span four code states and that only the
loss option, the epoch default and a logging fix are claimed to differ (unverifiable from git).

**m2. The GRU arm is not fp32.** torch 2.10 here reports `torch.backends.cudnn.allow_tf32 = True` and
`torch.backends.cudnn.rnn.fp32_precision = 'tf32'`; `nn.GRU` on CUDA runs through cuDNN, so on A40/A100/L40 (and the
pilots' 4090) the GRU matmuls are TF32. The transformer arms are fp32 (`matmul.allow_tf32 = False`, no override in
`train_mm.py`/`model.py`). Rule 19 records TF32 breaking a trainer before. *Fix:* set
`torch.backends.cudnn.allow_tf32 = False` in `train_mm.main` (affects only the GRU; pilots are never pooled), or
disclose "GRU in TF32".

**m3. Disclosure gaps for the pilot-informed loss.** (a) `train_mm.py:11-13` and `:92` still describe sighted-only as
the training loss / default; (b) `analyze_mm.py:130` prints `REGISTERED PRIMARY: ...` with no pilot-informed flag,
and that line is what gets pasted into results files. *Fix:* docstrings; append "(loss pilot-informed, Amendment 1)"
to the printed primary and HELD labels. (b) also: step features are stored fp16 on device (`train_mm.py:62-64`);
one-hots are exact, the heading entries are rounded (~5e-4 relative). Negligible (random-walk error over 1000 steps
~3e-3 cell), but the prereg says fp32; disclose or keep them fp32 (+~0.5 GB).

**m4. `analyze_mm.load` pools whatever it finds.** `analyze_mm.py:46-53` accepts any `<ARM>_s<seed>` dir with DONE:
no check of the seed set (3600-3607), of the config (epochs 600, loss_steps both, install none, train_range
[0,1600], eval_n 1000, pilot False, val_every 0), of md5 uniformity across runs, or of `env.torch` (the prereg forbids
pooling workstation and cluster runs). `N_REQ` is a minimum, so a stray ninth seed is silently pooled. *Fix:* assert
all of these and require exactly the registered seeds.

**m5. Label semantics and selftest coverage.** 'UNMEASURED' also covers |d| < 0.05 with p < .05 -- the selftest's own
UNMEASURED case has p 0.016 and MDE 0.013, i.e. a measured difference below the bar. Keep the registered label but
print a sub-label (e.g. "below the 0.05 bar, MDE x") so it is not reported as "unmeasured". The selftest does not
exercise d >= bar with p >= .05, INCOMPLETE via P_odo, the HELD replicate, `best_floor` on the real floor JSONs, or
`load()`; its Holm check (`analyze_mm.py:217`) cannot fail (Holm never lowers p). *Fix:* add those cases.

**m6. Probe deviations from Pasukonis et al. to disclose.** (a) probe data: 1600 training episodes (1.6M states, 1600
layouts) vs the 30M-step dataset; a 4x1024 MLP over 1600 layouts may overfit Walls; (b) training: 10k steps of 1024
on frozen final states vs a probe trained alongside 1M model steps; (c) Table E.3 uses Adam eps 1e-5 and LayerNorm
eps 1e-3, the code uses PyTorch defaults (1e-8, 1e-5); (d) the GRU probe reads `LayerNorm(Linear(h))` (128-d,
`model_mm.py:187-191`), not the GRU hidden state (184-d) that gate v2 named; (e) steps 500-1000 inclusive are 501
frames vs "the second half (500 steps)". None affects the registered primary; together they make the probe a reading
beside the published rows, not a replication.

**m7. Cost.** The A40 figure is taken from a 4090 synthetic benchmark ("similar or a little slower"). An A40 has
~45% of a 4090's fp32 matmul throughput and ~70% of its bandwidth; pilots alone on a 4090 ran 1.25 s/epoch at one
layer. Expect ~1.5-2.5 s/epoch (1 layer) and ~4-7 s (R4, GRU) on an A40, plus scoring (4 eval passes + held) and ~10-20
min of probe: ~35-55 A40-h for 56 runs rather than 30-40. Every run still fits the 8-h limit with large margin.

**m8. The function-identical twin floor was dropped.** Gate v3's design lists "a function-identical twin noise floor";
the prereg has none (rule 1). The 8-seed sd covers seed noise, so this is a disclosure, not a fix.

## NOTE

- **n1.** Verified exact: scored sets on both full scoring splits (5132 / 6900), the per-run metric (pooled step-level
  R1 on non-local rows), floors read as the max over `s2_all` (eval 0.04716, held 0.04725), permutation p two-sided
  (`perm2_p(b, a)` = mean(a) - mean(b)), Holm step-down over the four secondaries, UNREADABLE checked before the
  contrast, INCOMPLETE on any of P_E / R1 / P_odo. MDE uses the one-sample multiplier at df 7 (3.26) x sd x sqrt(2/8),
  conservative against the two-sample df 14 (3.01), and matches the prereg text. With the pilot sds (P_E 0.045, R1
  0.029) the MDE is ~0.06, so the 0.05 bar sits near the MDE and the pilot gap (+0.14..+0.28) is well above both.
- **n2.** Arms: the path step map reads only step features (`model_mm.step_delta`; frame tokens 0; tested). E feature
  rank 4 shared latent contains class E exactly (168 -> 2 is rank 2); the 6 extra 'none' pairs only occur at t = 1
  (gate v3 said 144-d). Shared rank 4 is >= 3 per head, inside the regime the rank results call solvable, and wd
  0.05 is on (RANKWD). P_raw's step map is action-only but its CONTENT carries the E feature (heading), so P_raw - P_E
  isolates the step-map input, not the information available.
- **n3.** `mm_encoder.py:227`: `ez if os.path.exists(ez) else tz` -- a non-smoke prep with `eval.zip` missing would embed
  train-part0 as "eval". Harmless now (eval names are 20220920T...); guard it with `args.smoke`.
- **n4.** Determinism: the resume test runs on CPU; on GPU the fp32 SDPA backward is not bitwise deterministic, so a
  preempted-and-resumed run is a statistical, not bitwise, continuation. Prereg line 104 should say so.
- **n5.** Resume / DONE: per-epoch checkpoint replaced atomically; `results.json` written via tmp + `os.replace`, then
  DONE; a run preempted after epoch 600 resumes straight into scoring. The ckpt holds only tensors and Python
  primitives (loads under `weights_only=True`, torch >= 2.6). `peak_gib` after a resume covers only the resumed
  segment; `env.gpu` names device 0. The unexplained `peak_gib` 0.0 of several pilots (all on `cuda:1`) is not used by
  any readout.
- **n6.** Task table: 56 lines, 7 arms x seeds 3600-3607, seed outer, md5 adf8812985b59b8c6c2dfa0044f91ef1 (matches
  the prereg); `mm_runs.slurm` names run dirs `<ARM>_s<SEED>`, which is what `analyze_mm.load` globs.
- **n7.** Two `test_mm` checks are tautological (GPS alignment re-implements the expression instead of probing the
  model; "step independent of frames" calls `step_delta` twice). The properties hold structurally (read in the code).
- **n8.** Training data order and blind phases are seeded by (seed, epoch[, batch]) only, so every arm at a seed sees
  the same episodes and masks (paired data, unpaired test -- fine).

## Proposed "## Amendment 2" text

> ## Amendment 2 (2026-10-27, after the independent code check, before launch; no result read)
>
> Source: `docs/audits/2026-10-27/memory_maze_audit.md`. Nothing here changes a registered arm, metric, label or
> threshold.
>
> 1. **What the primary measures.** In the pilots the readout, not the step input, limits every arm: P_odo 0.54 against
>    the odometry integrator's 0.93; the installed class-E map frozen 0.28, scratch P_E 0.31, installed trainable 0.38,
>    against the class-E integrator's 0.66. A frozen exact E map is read out at less than half of what it supports.
>    P_E - R1 is therefore read as "how much one learned retrieval layer gains from a path phase over an index phase
>    with the same inputs", not as the accuracy of integration. Rule 14 is met only as "the class holds a solution at
>    scratch's level"; the install scale was not varied.
> 2. **Scope of a PATH WINS.** One layer, E = 600, budget-scoped. R4 has 2.95x the parameters of P_E, so S1 is a depth-
>    and-parameters control. If S1 reads 'R4 ABOVE P_E', the exchange rate "one path layer ~ three attention layers"
>    (Dyck, H3) does not hold on this task, and every summary states it beside the primary. Predictions contradicted
>    by the pilots before launch: R4 (above P_E, not "a little above R1"), P_E (0.31, not near 0.66), P_odo (0.54, not
>    near 0.93).
> 3. **Pilot-informed loss, in every citation.** The primary is read under Amendment 1's loss, chosen after it opened the
>    P_E - R1 gap on outside seeds and a validation split. Every report of the primary says "loss pilot-informed
>    (Amendment 1)"; the fresh seeds and the untouched eval and held splits make it a confirmation of the pilot
>    finding, not a test of the original registration (whose loss failed for every arm, wave 1).
> 4. **Convergence.** The final-10% loss slope is flat by construction under cosine-to-zero. DESCENDING is read on
>    the training-loss slope over 60-85% of training [or: on a validation slice, 2800-2899, carved out of the held
>    set, which becomes 1600-2799 with its floor recomputed]. Secondaries are budget-scoped at E = 600.
> 5. **Precision and provenance.** The GRU arm runs [fp32: `torch.backends.cudnn.allow_tf32 = False` is set] [or: in
>    TF32 through cuDNN, as in the pilots]; step features are fp16-rounded. Code md5s are recorded at start and end of
>    each run. The pilots span four code states (wave 1 without the loss option; d8e8c3; 1909c1; the batch is
>    f47340); their md5s were recorded at the end of each run.
> 6. **Probe.** Deviations from Pasukonis et al.: 1600 training episodes, 10k post hoc steps on frozen states, default
>    Adam / LayerNorm epsilons, the GRU probed at its projected 128-d output, 501 scored steps. The probe is a reading
>    beside the published rows, not a replication.
> 7. **Analysis guards.** `analyze_mm` requires exactly seeds 3600-3607 per arm, the registered config, one md5 set and
>    the cluster torch version; when |d| < 0.05 at p < .05 it prints "below the 0.05 bar" beside UNMEASURED.
> 8. **Cost.** On A40s, ~35-55 A40-h for the 56 runs (the 30-40 figure assumed 4090 speed).
> 9. Optional, declared descriptive if run: P_E installed-frozen at seeds 3600-3607 (8 runs, ~4-6 A40-h). It has R1's
>    exact trainable parameter count, so P_E-frozen - R1 reads a fixed path phase against the index phase at identical
>    trainable parameters.
