# IMAGE_TORUS -- independent code check (rule 29), 2026-10-25

Checker: independent agent, results-blind (no run exists), CPU only. No tracked file and no file under review edited;
nothing committed; nothing run on a GPU. Files read in full: `IMAGE_TORUS_PREREG.md`, `image_bank.py`, `model_image.py`,
`train_image.py`, `eval_image_torus.py`, `gate_image_torus.py`, `IMAGE_TORUS_GATE.md/.json`, `analyze_image_torus.py`,
`test_image_torus.py`, `run_image_torus.sh`, `run_image_torus_pilot.sh`; plus the parts they rely on: `train.py`
(loop, schedule, data seed), `train_variant.py` (seeding, env, FACTORIAL path), `model.py` (MapFormerWM forward,
WMTransformerLayer is pre-LN), `model_codes.py` (ActOnly / NormStep), `lib_driver.sh`, `stats_core.py`
(perm2_p / perm2_ci / classify_run / fisher_solved), `analyze_factorial_n8.floors_walk`, `run_factorial*.sh`.

**Verdict: launchable AFTER Amendment 1.** No BLOCKER. Three MAJOR (one driver fix, one design decision about the
encoder's output scale, one power / readout gap for P2), seven MINOR, ten NOTES.

## What was run
- `python3 -m mapformer.test_image_torus` (CUDA hidden): **11/11 passed**.
- `python3 -m mapformer.analyze_image_torus --selftest`: **PASSED** (all P1/P2 labels, NOT LEARNED, block B,
  DESCENDING tag, five voids, Holm).
- Driver dry run in a scratch `git clone` (scratchpad `imgcheck/mapformer`, files under review copied in, data dir
  symlinked; `DRV_DRYRUN=1`, `drv_launch` returns before launching): exit 0, **32 DRYRUN launches**, seed-major, flags
  identical to `run_factorial*.sh` (`--epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 --n-steps 1024 --schedule
  cosine --data-workers 3 --fast-attn`), eval + analysis logged, md5 guard over 110 files; `IMG_BLOCKS=AB` refused
  against "Declared blocks: A".
- Extra CPU checks: md5-guard import closure (99 modules); `torch.initial_seed()` after building an image arm (3100 =
  token twin's, so `train()`'s data `base_seed` and walk stream are the twin's); full cross-split duplicate scan (96,000
  x 16,000 images: 0 byte-identical, 0 duplicates within a split); init scale of the encoder vs token rows (M2);
  a power simulation for P2 under bimodality (M3).

## Verified correct (no action)
- **Splits.** `ImageObsEmbedding.forward` picks the bank by `split or ("train" if training else "test")`; an explicit
  "test" in training mode raises (`model_image.py:233-234`); draws counted host-side per split; trainer asserts
  `draws["test"] == 0` and saves the counts (`train_image.py:89-101`); evaluator records the draws of the registered
  length loop only (`eval_image_torus.py:303-323`), the train-images readout comes after `d1`; analysis voids on
  either (`analyze_image_torus.py:80-86`). `train()` never calls `model.eval()` (no epoch callback passed).
- **Token twin / pairing.** `MI.build` seeds exactly as `train_variant` (manual_seed(seed) right before the class);
  the encoder is built inside `fork_rng(devices=[])` from its own seed; global RNG state and `initial_seed` unchanged
  (tested + checked); ActOnly / NormStep share every base weight with image MapWM; same `train()` arguments as
  `train_variant` on the FACTORIAL path (wd 0.05, no noise, no aux, offset 0).
- **Training-map protocol.** `GridWorld(64, 16, 0.5, n_landmarks=0, seed=run seed)` fixed for the run; `train()` only
  draws walks (data workers pickle the same env); evaluation on env seed 10000 with np seed 1234 + s, and `own|1024` on
  env seed = run seed. The prereg's correction of the brief ("map redrawn per sequence") is right.
- **Step map input.** `MapFormerWM.forward` (`model.py:185-188`) feeds `action_to_lie` the output of `token_emb`, which
  in image arms is the CNN embedding at every odd slot: W_in reads the image embedding exactly where the token model
  reads the observation-token row. ActOnly masks by raw token id (`model_codes.py:134-135`), unchanged by images.
- **No class/instance leak.** Instances are drawn i.i.d. per slot per forward (`torch.rand` on a private generator),
  independent of the cell; the target observation is never in the input (causal; inputs are `tok[:, :-1]`).
  Revisits cannot be solved by recognition alone; S5 (shuffle) is the on-model check.
- **Leak hook.** `_ZeroObj` zeroes `action_to_lie`'s output at object positions only; for ActOnly it is an exact no-op
  given identical images (same reseed per chunk), so the 0.002 void is a real wiring check.
- **Analysis.** Holm step-down correct; `perm2_ci` is for mean(b) - mean(a), as printed; WITHIN uses the unadjusted
  95% CI (symmetric thresholds); missing runs void before anything else; exit 1 on VOID stops the done marker.
- **Driver.** flock; required env vars captured before `lib_driver.sh` defaults; markers match the drivers that write
  them (`.ssm_sign_a2_done` from `run_ssm_sign.sh` A2 mode, `.factorial_n8_done`); `drv_ntrain` matches
  `mapformer.train_` (data workers are `spawn`ed, not counted); md5 guard re-checked per launch, before eval, before the
  analysis; data files md5-only; eval split round-robin, `drv_wait_dir` sees the evaluators (their argv carries `$R/`).

## Findings

### MAJOR

**M1. The md5 guard does not cover `model_codes.py`, which defines I_ActOnly (and I_NormStep).**
`run_image_torus.sh:46-52` builds the guard list from the modules loaded by importing train_image / eval / analyze /
gate / data_parallel / image_bank / model_image. `model_image._classes()` (`model_image.py:282-286`) imports
`mapformer.model_codes` (and `train_variant`) lazily, inside the function. `train_variant` happens to be pulled in by
another import; `model_codes` is not: verified, the closure has 99 modules and `mapformer.model_codes` is the only
module that `MI._classes()` adds. Failure scenario: `model_codes.py` is edited and committed mid-batch (it is the LEAK /
GAIN_PHASE line's shared file); the start-time clean-tree check has already passed, the per-launch guard does not
see the file, and later I_ActOnly seeds train a different class than earlier ones without any refusal. Fix: add
`import mapformer.model_codes` to the guard's import line (or call `mapformer.model_image._classes()` there), and
assert `model_codes.py` is in `GUARD` (e.g. `printf '%s\n' "${GUARD[@]}" | grep -qx model_codes.py || abort`).

**M2. At initialisation the image observations are ~27x smaller than the action embeddings, and image MapWM's object
steps ~18x smaller than its action steps; in the token twin both are the same scale.**
Measured on CPU (seed 3100, 2,000 test images): encoder output norm 0.417 (blank/zero image 0.387), within-class /
per-instance deviation 0.082, vs action token rows 11.30 (nn.Embedding N(0,1), sqrt(128)); MapWM step norm action
1.713 vs object 0.097. The layers are pre-LN, so attention mostly sees directions, but the step map reads the RAW
embedding (`model.py:188`), so image MapWM starts close to ActOnly. Consequences:
(a) P2 asks whether a leak is LEARNED (the encoder growing its output in W_in's row space), not whether an initial one
is removed; the prereg's mechanism (lines 136-137, 141-142: "per-instance step noise until the encoder learns to keep
its output out of W_in's row space", "early per-instance step noise wrecks the phase code") is not what this init
does; (b) S1 (cost of pixels) and S2's `L(I_MapWM) - L(T_MapWM)` compare models that differ in observation scale as
well as modality (rule 15: a scale is an optimiser change). Not a code bug and P1 / P2 stay valid internal contrasts
(all image arms share the encoder), so not a BLOCKER, but it needs a decision before launch. Options:
(i) a fixed gain on the encoder output, identical in every image arm, set so the init mean norm is sqrt(d_model)
(about x27; e.g. initialise the final Linear's weight and bias x27, or multiply the output by a registered constant);
re-run the tests (the gate's classifier is unaffected); or (ii) keep the design and amend the predictions and the
S1 / S2 wording as below. (i) makes P2 test the stated mechanism; (ii) changes what P2 means. Recommend (i).

**M3. P2 cannot detect the predicted effect if it arrives as a minority of slow / stalled MapWM seeds; the Power
section says the permutation test handles that, and it does not.**
Simulation (`stats_core.perm2_p`, ActOnly 0.990 +/- 0.002 x8; MapWM 0.990 +/- 0.002 with k seeds at 0.90 +/- 0.02;
20 reps): k = 1 (d ~0.011) median p 0.50, 0% p < .05; k = 2 (d ~0.022) 0.27, 10%; k = 3 (d ~0.034) 0.11, 10%;
k = 4 (d ~0.045) 0.032, 70%; k = 5: 0.009, 100%. With a few outliers the exact permutation p is bounded by where the
outliers can fall (2 x C(16-k, 8-k) / C(16, 8): 0.20 at k = 3, 0.077 at k = 4). The predicted P2 range (d +0.01 to
+0.04, "more DESCENDING at 900 epochs") therefore reads UNMEASURED unless at least 4-5 of 8 MapWM seeds fall short.
(Holm is not the problem: with P1 near its minimum p, P2's adjusted p equals its raw p.) Fix: correct the Power text;
add a declared count readout: seeds within 0.01 of ceil_vote, I_ActOnly vs I_MapWM, two-sided Fisher exact
(`stats_core.fisher_solved`), reported beside P2 (secondary, or co-primary with Holm over three -- decide before launch).

### MINOR

**m1. cuDNN TF32 differs between training and evaluation.** `train_image.py:63-64` turns TF32 off for matmuls and
convs; `eval_image_torus.py` sets nothing, so evaluation convs run under PyTorch's default `cudnn.allow_tf32 = True`.
The encoder is evaluated with different numerics than it was trained with (same for every arm, no verdict bias).
Fix: set `torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False` in `eval_image_torus.main`
(and the pilot's stage 3 inherits it).

**m2. Directional labels have no floor qualifier.** The NOT LEARNED suffix (`analyze_image_torus.py:162-164`) applies
only to WITHIN. P1 can read PATH WINS IN PIXELS with I_MapWM below the retrace floor (e.g. 0.80 vs 0.72, retrace 0.843),
and the floor is a token floor (perfect recognition), so it is optimistic for pixels. Fix: append
"[winning arm at or below the retrace floor]" to any directional label where the winning arm's mean <= the mean retrace
floor; state in the prereg that the retrace floor assumes perfect recognition.

**m3. "IN PIXELS" without a token ActOnly.** P2's labels name pixels, but there is no T_ActOnly cell: a P2 effect may be
a generic ActOnly-vs-MapWM property on this torus (ActOnly also removes blank steps, prereg line 213), not a pixel
effect. T_MapWM is 1.000 on tokens, but ActOnly's token value here is unmeasured. Fix: add T_ActOnly to block B
(8 x ~60 min), or state that the pixel-specific question is S2 and P2's label words describe the condition only.

**m4. Selftest and alignment gaps.** No selftest case where Holm (not the raw p) changes a label (e.g. raw p 0.03 and
0.04 -> Holm 0.06 / 0.06: both UNMEASURED); the void "recorded no train draws" is untested; per-seed floors are aligned
by POSITION (`tags`, `S1 - ceil`: `ret[:n]`, `ceil[:len(D[c])]`), correct only because the gate's seeds equal `SEEDS`
and a missing run voids. Fix: add the Holm case and the no-draw void to `selftest()`; assert
`sorted(int(k) for k in gate["floors"]) == SEEDS` in `run()` (or index floors by seed).

**m5. Rule 11 ("the gate must CALL the task code") is met only in part.** `gate_image_torus.stream_check` (line 55) and
`test_revisit_mask_and_stream_unchanged` construct `GridWorld` with arguments copied from `train_image.py:69`, so a
change there would not be seen; `test_eval_stream_equals_eval_noise_refine` re-implements the stream instead of calling
`eval_noise_refine`. Fix: factor `make_env(seed, grid)` in `train_image.py` and call it from the trainer, the gate and the
test; compare against `eval_noise_refine`'s own walk loop.

**m6. "Declared blocks:" takes the first matching line** (`run_image_torus.sh:22`, `head -1`). An amendment that appends
"Declared blocks: AB" has no effect (line 7 still says A) and the driver refuses AB. Fails safe; say in the prereg that
choosing AB means editing line 7 (and noting it in the amendment).

**m7. SOLVED / STALLED mean something else for image arms.** The training loss of a converged image run has a floor set
by recognition errors on the train split (~1-2% object errors), which can sit near the 0.05 SOLVED cut; with LR at 10%
of peak a converged run can read STALLED. The prereg already makes SOLVED a tag; add that STALLED in an image arm is not
"stuck" and is read together with "within 0.01 of ceil_vote" and DESCENDING.

### NOTES

- **n1.** Token n-gram floors are fitted on 50 walks of ONE training map per seed (`gate_image_torus.py:67-81`), not
  16 maps as in `score_rank_floors.py`. Immaterial: retrace (0.843) dominates the n-gram (0.517); the action-only
  n-gram equals blank, as it should.
- **n2.** The ceiling uses a standalone classifier (10 epochs, ~1e6 presentations); the model's encoder sees ~7e8
  presentations (weaker, indirect supervision). ceil_vote can be exceeded; "image cell - ceil_vote" may be positive.
  The tag counts `>= ceil - 0.01`, so that is handled. The tag is meaningless for token cells (harmless).
- **n3.** Eval pairing across GPUs: round-robin over seed-major REQ sends I_MapWM / I_RoPE to GPU 0 and I_ActOnly /
  T_MapWM to GPU 1 at every seed. Identical images across arms then rely on identical Philox output on both devices;
  both are RTX 4090, so it holds. Reseeding is per chunk of 4 walks, not per walk (same chunking in every arm). The
  analysis is unpaired anyway.
- **n4.** The interleaving check runs only outside training (`model_image.py:236`); training relies on the GridWorld
  invariant. A one-time check of the first training batch in `train_image.py` would cost nothing. Optional.
- **n5.** Eval mode without the attention-dropout rescale (declared, as every registered torus batch). Per
  `DROPOUT_RESCORE.md` this under-reports one-layer runs below ceiling (up to +0.17), so it mostly affects I_RoPE and
  could only shrink P1 if rescaled; keep as declared.
- **n6.** Recognition alone cannot produce P1 (same encoder init and recognition task in both arms), but I_RoPE's
  encoder gets a weaker gradient (it cannot find the earlier visit), so P1 is a total effect (map + learned
  perception); S3's probe separates them.
- **n7.** Splits: 0 byte-identical images across splits on the full 96,000 x 16,000 scan (prereg line 36 is right);
  `test_bank_real` checks only 20,000 train images and allows 5 -- tighten to `== 0` on the full set (~10 s).
- **n8.** The markers are a refusal, not a wait. If either running batch ends VOID, no marker is written and the launch
  needs `DRV_SKIP_MARKERS=1` (logged). The driver checks only this user's trainers plus free memory >= 4000 MiB; check
  other users' GPU jobs by hand at launch (`feedback_gpu_sharing_and_cost.md`).
- **n9.** Pilot stage 4 runs a 150-epoch cosine (a compressed schedule): a smoke check, not a learning curve comparable
  to the 900-epoch batch. The memory baseline is read once at the start (another user's job would be misattributed).
- **n10.** P2's 0.01 threshold cites LEAK's +0.0107, from the new-object task. On this fixed torus T_MapWM is 1.000, so
  the token leak cost here is 0. The threshold sits well above eval sampling noise (binomial SE ~0.0006 at 0.99 on
  ~32,500 targets).

## Proposed "## Amendment 1" text (to paste into IMAGE_TORUS_PREREG.md before launch, after the fixes)

> ## Amendment 1 (2026-10-25, before any run; independent code check `docs/audits/2026-10-25/image_torus_audit.md`)
> Results-blind (no run exists). Changes:
> 1. **md5 guard** (M1): `mapformer.model_codes` (the I_ActOnly / I_NormStep classes, imported lazily by
>    `model_image._classes`) was missing from the guard's import closure. The driver now imports it and refuses if
>    `model_codes.py` is not in the guard list.
> 2. **Encoder output scale** (M2), measured at init (seed 3100): observation embedding norm 0.42 vs action rows 11.3;
>    MapWM object steps 0.097 vs action steps 1.71. [Option (i):] Every image arm multiplies the encoder output by a
>    fixed gain G = <value> (init mean norm = sqrt(128)); identical across arms; tests re-run. [Or option (ii):] Kept
>    as is. P2 then tests whether image MapWM LEARNS a costly observation step from a start close to ActOnly, and the
>    prediction about early per-instance step noise is withdrawn; S1 and S2's token-vs-image contrasts are confounded
>    with the observation scale and are descriptive only.
> 3. **P2 power** (M3): the permutation test does not rescue a bimodal I_MapWM; with 3 of 8 seeds short (d ~0.03) P2
>    reads UNMEASURED (simulated median p 0.11). Added, beside P2 [as a declared secondary / as co-primary with Holm
>    over three]: seeds within 0.01 of ceil_vote, I_ActOnly vs I_MapWM, two-sided Fisher exact
>    (`stats_core.fisher_solved`).
> 4. **Evaluation numerics** (m1): the evaluator turns TF32 off for matmuls and cuDNN, as in training.
> 5. **Floor qualifier** (m2): any directional label whose winning arm's mean is at or below the mean retrace floor gets
>    "[winning arm at or below the retrace floor]". The retrace floor assumes perfect recognition (optimistic for pixels).
> 6. **P2 wording** (m3): [T_ActOnly added to block B / the words "IN PIXELS" name the condition; the pixel-specific
>    question is S2 (L(I_MapWM) - L(T_MapWM)); ActOnly is not measured on tokens here].
> 7. **Analysis hygiene** (m4): `run()` asserts the gate's seeds equal the batch seeds; the selftest adds a case where
>    Holm changes a label and the "no train draws recorded" void.
> 8. **Gate** (m5): the environment construction is one function in `train_image.py`, called by the trainer, the gate
>    and the test. The gate is regenerated; its numbers are unchanged if the function is unchanged.
> 9. **Blocks** (m6): choosing AB means editing the "Declared blocks:" line itself (the driver reads the first one).
> 10. **Run classes** (m7): in image arms STALLED is not "stuck" (the loss has a recognition floor); read it with
>    "within 0.01 of ceil_vote".
> Notes recorded, no change: n-gram floors fitted on one training map (retrace dominates); eval pairing across the two
> GPUs relies on both being RTX 4090; eval mode without the dropout rescale; P1 is a total effect (map + perception).
