# IMAGE_TORUS Amendment 2 -- independent code check (rule 29), 2026-10-25

Checker: independent agent, results-blind for the batch (no run of seeds 3100-3107 exists; `runs/image_torus/` does not
exist). Pilot evidence read as given (`IMAGE_TORUS_PILOT.txt`, `IMAGE_TORUS_PILOT2.txt`, pilot dirs listed, not opened
beyond file names). CPU only, GPUs hidden (`CUDA_VISIBLE_DEVICES=""`) in every command. No file under review edited,
nothing committed, nothing launched. Scratch: `.../scratchpad/imgcheck2/{old,new}/mapformer` (two `git clone`s of HEAD
eeb1c97; `new` has the working-tree versions of the 8 changed files copied in; `data/image_torus` symlinked).

Files read in full: `IMAGE_TORUS_PREREG.md` (registration, Amendment 1, Pilots, Amendment 2), the previous audit,
`model_image.py`, `train_image.py`, `eval_image_torus.py` (load / score / evaluate_run header), `analyze_image_torus.py`,
`run_image_torus.sh`, `run_image_torus_pilot2.sh`, the diff of `test_image_torus.py`, `model_codes.py` (ActOnly /
NormStep / _StepOverride), `lib_driver.sh` (drv_launch dry-run path), `stats_core.perm2_ci`.

**Verdict: launchable AFTER a short Amendment 3 (analysis and prereg text only; no trainer, model or driver change).**
No BLOCKER. One MAJOR (P2's label names a mechanism the pilots contradict on one of two seeds; a per-seed end-state
readout is missing), six MINOR, nine NOTES. The training path is verified bitwise unchanged on the default gain.

## What was run
- `python3 -m mapformer.test_image_torus` (new clone, CPU): **13/13 passed** (incl. `test_encoder_gain_runtime_flag`).
- `python3 -m mapformer.analyze_image_torus --selftest`: **SELFTEST PASSED**, 21 checks: every P1 / P2 label with both
  suffixes, Holm changing both labels, six voids, the three G labels, gate-seed void, Fisher-count-is-secondary, Holm.
- **Default-path equivalence (old = 4955021 code at HEAD, new = working tree)**: `train_image.main` on CPU, seed 3190,
  2 epochs x 2 batches x B 2 x T 64, `--data-workers 0 --fast-attn`, for all five default block-A cells (I_ActOnly,
  I_MapWM, I_RoPE, T_MapWM, T_ActOnly): **state dicts bitwise identical, losses identical, same keys, same variant
  name, same config keys** (4955021 already wrote `config["enc_gain"]`). An explicit `--enc-gain 30.3729` is bitwise the
  default and names the cell `I_MapWM`.
- **`--enc-gain` end to end**: `--enc-gain 1.0` -> cell `I_MapWM_g1`, file `I_MapWM_g1.pt`, `config.enc_gain` 1.0,
  `args.enc_gain` 1.0, buffer `token_emb.enc_gain` 1.0, losses differ from the default (the flag acts); `--enc-gain
  2.5` -> `I_MapWM_g2.5`; `--obs token --enc-gain 1.0` refused (SystemExit). The evaluator's `load` restores 1.0 / 2.5 /
  30.37 / None (token) by strict load, also loads a 4955021-era checkpoint; a tampered checkpoint (config 30.37, buffer
  1.0) trips the new assert. Eval JSON `cell` = checkpoint `variant` = `I_MapWM_g1`, and `analyze.load` asserts
  `r["cell"] == c`, so the name chain driver cell -> trainer -> checkpoint -> eval JSON -> analysis cell is self-checking:
  a wrong gain in a cell produces a differently named checkpoint, `drv_require` fails, the batch stops.
- **Driver dry run** in the new clone (`DRV_DRYRUN=1 IMG_BLOCKS=A DRV_GPUS="0 1" DRV_MAXPG=4`, GPUs hidden;
  `drv_launch` returns before `setsid` when `DRV_DRYRUN=1`, `lib_driver.sh:88`; the dry-run branch never calls
  `drv_wait_slot`): exit 0, **48 DRYRUN launches**, seed-major, 6 cells per seed in the declared order, `--enc-gain 1.0`
  on exactly the 8 `I_MapWM_g1` launches and nowhere else, flags `--epochs 900 --lr 1e-3 --n-batches 98 --batch-size
  16 --n-steps 1024 --schedule cosine --data-workers 3 --fast-attn`; eval of 48 checkpoints and the analysis logged;
  **md5 guard over 111 files**, including `model_codes.py`, `model_rank.py`, `model_image.py`, `train_variant.py`,
  `stats_core.py`, `analyze_image_torus.py`, the prereg, the gate, the test and the five data files. `IMG_BLOCKS=AB`
  refused against the last "Declared blocks: A" line (three such lines exist, 7 / 319 / 414; the driver reads the last).
- **Seeds**: no directory under `runs/` named for 3100-3107; a context grep (`seed`, `_s`, `--seed` + 3100-3107) over
  every .md/.sh/.py/.txt/.json/.log outside `papers/` hits only the image-torus files (bare-number hits are p-values and
  counts, e.g. "0.3102", "[ 3101]"). Pilot seeds 3180-3183 only in the two pilot scripts, the pilot dirs and
  `test_image_torus.py` (3182, synthetic banks). koopman not checked.
- **Pilot code = batch code**: `runs/image_torus_pilot2/code_md5.txt` equals the current md5 of `model_image.py`,
  `train_image.py`, `eval_image_torus.py`, `image_bank.py`, `model_codes.py`. Pilot-1 stage 4 wrote no eval JSON
  (`runs/image_torus_pilot/learn/*/` hold only `.pt` + `train.log`), so the image RoPE arm's accuracy is still unseen.
- Two CPU simulations (below): P2 when some MapWM seeds escape; `perm2_ci` run time and shape on bimodal data
  (0.0-0.3 s per contrast; walk-out CI never None).

## Verified correct (no action)
- **(1) Plumbing.** `run_image_torus.sh:46-48, 76-77, 84` (cell gain field, `GA=()` when "-", `"${GA[@]}"` safe under
  `set -u`); `train_image.py:67-69, 95-98, 104, 120` (flag, token refusal, build, cell name, config); `model_image.py:65,
  71, 151-156, 182-193, 208-211` (buffer = `ENC_GAIN` when None, otherwise the value; persistent; `cell_name` adds
  `_g{gain:g}` only for image arms off `ENC_GAIN` by > 1e-6); `eval_image_torus.py:47-51, 202` (build with the saved
  gain, strict load, assert, gain in the JSON); `analyze_image_torus.py:62-65` (cell lists). Default path bitwise.
- **(2) P1 swap.** `MapWM_ActOnly.step` (`model_codes.py:134-135`) multiplies `action_to_lie(x)` by `(tokens < 4)`: the
  step reads only action positions, which are even slots, which `ImageObsEmbedding.forward` (`model_image.py:108, 125-127`)
  never replaces -- action rows are the model's own `nn.Embedding` rows. So I_ActOnly's step map is the same function of
  the same parameters as T_ActOnly's (same class, same mask by token id, same seed); the image enters only content
  (Q/K/V/FFN), and the step-to-encoder gradient is exactly 0. I_ActOnly vs I_RoPE share the encoder init (own seed in
  `fork_rng`), the gain, banks, instance stream, recipe, flags and evaluation walks/reseeds; they differ in the position
  code (and base weights from different classes; params 284,725 vs 283,893). Fair as "an action-driven path integrator
  with pixel content vs index rotary with pixel content".
- **(3) Holm / G.** `registered()` (`analyze_image_torus.py:187-188`) runs Holm over exactly `[P1.p, P2.p]`; G
  (`:315-321`) and the Fisher count use raw p and are outside Holm. G's labels follow the prereg rule; `perm2_ci` walks
  out from d until rejection, so `ci.hi` is never None (the `is not None` guards are dead but harmless).
- **(7) Guard / blocks.** As in the dry run. The md5 guard re-runs per launch, before evaluation and before analysis.
- **(8) Selftest** reaches every label and every void and compares exact strings, so each case can fail.
- Cost arithmetic of Amendment 2 reproduces (32 x 87 + 16 x 63 = 3,792 lane-min; 7.9 h on 8 lanes) up to n2 below.
  Memory: +1.5 GB per image job solo, +5.6 GB for four on one GPU; `DRV_MINFREE` 4000 MiB; 32 CPU cores, 125 GB RAM.
  Markers `.ssm_sign_a2_done` and `.factorial_n8_done` exist. Both GPUs idle at check time.

## Findings

### MAJOR

**M1. P2's label says "LEAK"; the pilots show two different failures, one of which is not a what-to-where leak, and the
analysis has no per-seed readout that tells them apart.**
`analyze_image_torus.py:170` ("THE LEAK COSTS IN PIXELS"), `:263-270` (S2 prints MEANS of object/action and
instance/action step ratios), `:289-295` (S3 prints the MEAN probe). Prereg Amendment 2 item 2 calls P2 "the
what-to-where leak". Pilot 2: P2d kept a huge per-instance step (instance/action 122, probe 0.87: noise in the "where",
a leak); P2c's encoder COLLAPSED to a constant (probe 0.13, per-instance step 0: no information in the "what" -- the
leak is gone, and so is perception). Both sit at the memoryless loss. Failure scenario: P2 fires at d ~ +0.48 on 8
fresh seeds, the report and later CLAUDE.md row read "the what-to-where leak costs 0.48 in pixels", while on (say) 5 of 8
seeds the encoder collapsed; the S2/S3 means (one seed at a step ratio of 572 dominates a mean of ratios) cannot show it.
This is the pattern of the 2026-10-11 withdrawals ("state changes" that were location + recency). Fix (analysis +
prereg, before launch, because the analysis is md5-guarded during the batch):
(a) P2's label words: "THE OBSERVATION STEP COSTS IN PIXELS" (or keep the label and print, on the same line,
"[mechanism: see the per-seed end states; 'leak' only for NOISY seeds]");
(b) a declared descriptive line beside P2 (no verdict): per I_MapWM seed, end state = COLLAPSED if probe test acc
< 0.5, NOISY if instance/action >= 1 and probe >= 0.5, LEARNED if acc >= retrace floor, OTHER otherwise -- thresholds
fixed now, results-blind; counts per state; and print S2/S3 per cell as median and range, not mean.

### MINOR

**m1. The "pilot-informed, not blind" disclosure is in the prereg and the analysis docstring only, not in the analysis
output or the verdicts JSON.** `analyze_image_torus.py:218` prints `P2 -> <label>` and `:224` writes the JSON without it;
`:323` prints G without it. A reader of `IMAGE_TORUS_ANALYSIS.txt` / `IMAGE_TORUS_VERDICTS.json` (what the driver
produces) sees P2 and G as blind registered contrasts. Fix: append "[PILOT-INFORMED, NOT BLIND: direction known from
pilots 1-2 (3 seeds at 150 epochs)]" to the P2 output line and "[PILOT-INFORMED: one pilot seed]" to G's, and add
`"pilot_informed": true` to both JSON entries; add both strings to a selftest assertion.

**m2. P1 is partly pilot-informed too, and that is not disclosed.** The new path arm's value is known from P2a
(I_ActOnly 0.9944 at 150 epochs, quoted in the P1 prediction); image RoPE's training loss is known (pilot 1: 1.20 at
epoch 150, far above ActOnly's 0.015); only image RoPE's accuracy is unseen (no stage-4 eval exists). Fix: say so in
Amendment 3; do not evaluate the pilot-1 I_RoPE checkpoint (`runs/image_torus_pilot/learn/I_RoPE/I_RoPE.pt`) before the
batch is read, so the index arm stays blind.

**m3. The original P1 is demoted to P1' without a label, although the registration's own "Stated risk" said this outcome
would be reported as a result.** Prereg lines 141-143 (stated risk: "I_MapWM may stall ... Then P1 reads UNMEASURED or
WITHIN [NOT LEARNED] and P2 fires large; that is a result ... not a failure of the test") vs the pilot stop rule (line
170-171: "stop, do not launch ... redesign"). Both were pre-declared; the redesign follows the stop rule, but the
originally registered question should still get its answer. `analyze_image_torus.py:329` prints P1' as d / p only.
Fix: print P1' with `label_p1` applied to its RAW p (marked "original P1 rule, outside Holm, superseded by Amendment 2"),
and record that the pre-pilot P1 prediction (I_MapWM 0.97-0.99, line 133) was contradicted by the pilots.

**m4. Power text for P2 is right only up to three escaping seeds.** Prereg line 398-399 ("a seed that escapes would
shrink d but not the label", "predicted shortfall is on all seeds"). The pilots ran a 150-epoch cosine; the batch keeps
LR near peak far longer (at epoch 150 of 900 the cosine is still ~93% of peak), so escapes are the live risk.
Simulation (ActOnly N(0.990, 0.002) x 8; k MapWM seeds escape to N(0.990, 0.002), the rest N(0.505, 0.003); P1 at its
minimum p; Holm over two; 100 reps):
| MapWM seeds that escape | median d | P2 = THE ... COSTS | median p_Holm |
|---|---|---|---|
| 0 / 1 / 2 / 3 | +0.485 / +0.425 / +0.364 / +0.303 | 1.00 / 1.00 / 1.00 / 1.00 | 0.0003 / 0.0008 / 0.004 / 0.014 |
| 4 | +0.243 | 0.71 | 0.036 |
| 5 / 6 / 7 | +0.182 / +0.121 / +0.061 | 0.19 / 0.13 / 0.07 | 0.11 / 0.20 / 0.48 |
| 8 | 0.000 | 0.00 | 0.56 |
Fix: replace the sentence with "P2 holds if at most 3 of 8 image MapWM seeds escape within 900 epochs; with 4 it is a
likely but not certain firing (0.71); with 5 or more it reads UNMEASURED although d stays >= ~0.18 (the exact
permutation p is bounded by where the escaped seeds fall)".

**m5. G's label names a mechanism and a trend that two gain values on the whole encoder output cannot carry.**
`analyze_image_torus.py:316-321`; prereg item 3. The gain multiplies the image embedding everywhere -- the step map's
input AND the residual stream / attention values -- so G is not specific to the step map (P2 is: ActOnly and MapWM
differ only in the step at gain 30.37, and ActOnly learns there, so the residual-stream scale alone is not fatal -- but
an interaction is not excluded). Two points are not "scaling" (rule 4). Also "GAIN DOES NOT HELP" fires when both arms
learn (gain 30.37 MapWM escapes) or when gain 1 is significantly WORSE -- both mislabelled by those words. Fix: labels
"GAIN 1 RESCUES MAPWM" (d >= 0.05, p < 0.05), "NO RESCUE >= 0.05" (CI below +0.05), "AS IT FALLS"; one sentence in the
prereg that the gain acts on the whole image embedding, so G is a total effect of the encoder-output scale.

**m6. The prereg reads inconsistently end to end (rule 28).** Line 3 "Status: DRAFT ... Not committed ... Nothing in it
has been run on a GPU" (committed at 4955021; two pilots ran); line 69 "Pilot seeds 3180-3181" (now 3180-3183); line
68-70 / 72-76 "Block A = 32 runs" and "ActOnly is the core remedy" framed for the old P2; Amendment 1 item 2's last
sentences ("So the prereg's mechanism wording stands: P2 asks whether an initial, per-instance observation step is
removed (MapWM) or never present (ActOnly)") are superseded by Amendment 2 item 6 without saying so; the main
Predictions section (lines 132-143) is superseded by Amendment 2's but not marked. Fix: a one-line "superseded by
Amendment 2" marker at each of those places (no rewriting of registered text), and update line 3.

### NOTES

- **n1. Readings that could fail (rule 7).** P1 PATH WINS fails if image RoPE comes within 0.05 of image ActOnly, or if
  ActOnly stalls on fresh seeds (its value is known on one pilot seed only). P2 fails (UNMEASURED) if >= 5 of 8 image
  MapWM seeds escape in 900 epochs (m4), WITHIN 0.01 if all do. G fails (AS IT FALLS / NO RESCUE) if gain-1 MapWM stalls
  on several fresh seeds (one pilot seed). The residual I_ActOnly - I_MapWM_g1 "within +/-0.01" fails if gain-1 MapWM
  converges slower (P2b used its observation steps: zeroobj -0.055). None is unfailable.
- **n2. Cost.** Image RoPE ran 6.9 s/epoch contended, not 5.8 (pilot stage 2): 8 x 104 min instead of 8 x 87 -> 3,928
  lane-min = 65.5 lane-h, **~8.2 h on both GPUs** (4 each), ~16.4 h on one; plus DRV_SPACING 30 s x 48 and the tail of the
  last wave. The pilot measured 4 jobs on one GPU; 8 jobs on two (24 data workers) was not measured, but FACTORIAL ran
  the same packing with the same flags.
- **n3. P2's cells are evaluated on different GPUs.** With 6 cells seed-major and round-robin over two GPUs, I_ActOnly /
  I_RoPE / T_MapWM go to GPU 0 and I_MapWM / I_MapWM_g1 / T_ActOnly to GPU 1 at every seed. Identical images across arms
  then rely on identical Philox output on two RTX 4090s (Amendment 1 n3); P2 is unpaired, S2b is paired.
- **n4. No explicit gain check in the analysis.** The name chain makes a gain mix-up fail at `drv_require` (verified),
  so this is belt-and-braces: a void if `r["enc_gain"]` differs from 30.3729 (+/-1e-4) in I_* cells other than
  `I_MapWM_g1`, or from 1.0 there.
- **n5. The new unit test is build-level.** `test_encoder_gain_runtime_flag` does not go through `train_image.main`'s
  argparse and checkpoint; this check did (above). A tiny CPU `train_image.main([... '--enc-gain', '1.0' ...])` in the
  test would take ~10 s and pin the end-to-end chain.
- **n6. The evaluator's gain assert** (`eval_image_torus.py:50-51`) compares the restored buffer with the config saved
  from the same buffer: near-tautological, but it does catch a tampered or hand-edited checkpoint (verified).
- **n7. Block B, if ever declared,** still uses the non-learning I_MapWM as the path arm in S7 (`analyze_image_torus.py:
  333`: I_MapWM - I_RoPE_L4, I_NormStep - I_MapWM). Under Amendment 2 the depth reference should read I_ActOnly -
  I_RoPE_L4. Not used by block A.
- **n8. Stale driver comment.** `run_image_torus.sh:44` ("deep first within a seed ...") precedes the new format comment
  on line 45; harmless.
- **n9. P1 is now close to "the path code is untouched by pixels by construction".** I_ActOnly's step never sees an
  image, so P1 asks whether content attention can match pixel instances across visits given a perfect action-driven
  position code, against index RoPE. That is a fair and useful question; the report should word it that way (and P1 is
  still a total effect: map + learned perception, Amendment 1 n6). The token reference for index RoPE at one layer is
  FACTORIAL's 0.72 (stored, descriptive only; T_RoPE is block B).

## Proposed "## Amendment 3" text (paste into IMAGE_TORUS_PREREG.md before launch, after the fixes)

> ## Amendment 3 (2026-10-25, before any batch run; independent re-check `docs/audits/2026-10-25/image_torus_amendment2_audit.md`)
> Results-blind for the batch (no run of seeds 3100-3107 exists). The re-check found 1 MAJOR, 6 MINOR, 9 NOTES, no
> blocker; the trainer, model and driver are unchanged (default gain verified bitwise against 4955021 on CPU; `--enc-gain`
> verified end to end; dry run 48 launches, md5 guard 111 files). Changes (analysis and text only):
> 1. **P2 wording and end states (M1).** P2's labels become THE OBSERVATION STEP COSTS IN PIXELS / ACTION-ONLY HURTS /
>    WITHIN 0.01 / UNMEASURED (rule, threshold, test and Holm unchanged). The pilots showed two failures at gain 30.37: a
>    per-instance step that scrambles the phase (a leak; P2d) and a collapsed encoder (probe 0.13; P2c), so "leak" is not
>    asserted by the label. Declared descriptive beside P2 (no verdict): per image-MapWM seed, end state COLLAPSED (probe
>    test acc < 0.5) / NOISY (instance/action step >= 1, probe >= 0.5) / LEARNED (acc above the retrace floor) / OTHER,
>    counted per state; S2 and S3 are printed per cell as median and range.
> 2. **Disclosure in the output (m1, m2).** The analysis prints, and writes to the verdicts JSON, "PILOT-INFORMED, NOT
>    BLIND" for P2 (pilots 1-2: three seeds at 150 epochs) and G (one seed). P1 is partly pilot-informed: the image
>    ActOnly value is known from one pilot seed (0.9944) and image RoPE's pilot training loss (1.20 at epoch 150); image
>    RoPE's accuracy has not been measured, and the pilot-1 I_RoPE checkpoint will not be evaluated before the batch is read.
> 3. **The original P1 (m3).** P1' (I_MapWM - I_RoPE) is printed with the original P1 label rule on its raw p, marked
>    "original P1 rule, outside Holm, superseded". The pre-pilot P1 prediction (I_MapWM 0.97-0.99) was contradicted by
>    the pilots; the registration's stated risk anticipated this outcome, and the pre-declared stop rule required the
>    redesign.
> 4. **P2 power (m4).** Replaces Amendment 2's Power sentence: P2 fires if at most 3 of 8 image MapWM seeds escape within
>    900 epochs; with 4, probably (simulated 0.71); with 5 or more it reads UNMEASURED although d stays >= ~0.18. The
>    pilots used a 150-epoch cosine, so escapes under the 900-epoch schedule are the live risk.
> 5. **G labels (m5).** GAIN 1 RESCUES MAPWM (d >= 0.05, perm p < 0.05) / NO RESCUE >= 0.05 (95% CI below +0.05) / AS
>    IT FALLS. The gain scales the whole image embedding (step-map input and residual stream), so G is a total effect of
>    the encoder-output scale at two values, not a dose-response and not specific to the step map; P2 is the
>    step-specific contrast at gain 30.37.
> 6. **Text (m6).** "Superseded by Amendment 2" markers on the Status line, the pilot-seed line (now 3180-3183), the
>    block-A count, the main Predictions section and the last sentences of Amendment 1 item 2.
> 7. **Notes, no change:** cost re-estimated with image RoPE at its measured 6.9 s/epoch: ~65.5 lane-hours, ~8.2 h on
>    two GPUs; P2's cells are evaluated on different GPUs (both RTX 4090); the analysis relies on the cell-name chain
>    for the gain (an explicit gain void [is / is not] added); block B's S7 would need I_ActOnly as its path arm.
>    Checks re-run after this amendment: `test_image_torus`, `analyze_image_torus --selftest` (adds the disclosure strings,
>    the end-state line, the P1' label and the renamed G labels), dry run.
