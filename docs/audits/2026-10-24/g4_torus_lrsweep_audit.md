# G-4 TORUS_LRSWEEP independent code check (rule 29), 2026-10-24, before launch

Checked at commit 5c6b76e (`TORUS_LRSWEEP_PREREG.md`, `analyze_torus_lrsweep.py`, `run_torus_lrsweep.sh`), blind to every
result except the disclosed timing pilot. CPU only (kalman), scratch seeds 2470; GENERALITY's committed checkpoints (seeds
700-707) were read for the loss-scale questions. No code was edited. Scripts: `g4_check/check1.py`, `g4_check/check2.py`
(run as `cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/audits/2026-10-24/g4_check/checkN.py`).

**Verdict: launchable after Amendment 1.** The registered machinery is LRSWEEP's, imported unchanged and applied correctly to
three rates; the command lines are GENERALITY's torus recipe except lr; the driver is sound. Two MAJOR points, both cheap:
the COST CLOSES branch attributes to lr what may be the coupled decay (RANKWD shows decay matters on this exact task), and the
UNSTABLE qualifier's fixed 2.08 threshold sits inside the per-seed spread of the context-free loss. Nine MINORs.

**Incident during the check (disclosed).** A driver test in a scratch `git clone --shared` (path under the scratchpad) was run
once WITHOUT `DRV_DRYRUN=1` (my error: it was meant to test that a modified tracked doc passes the clean-tree check, which it
does). It launched two real 900-epoch jobs (SO / AO, seed 2400, lr 1e-3) on kalman GPU 1 at 22:43; both were killed by PID
within ~2 min, with their data workers; no orphans; the clone and its `runs/` were deleted. Nothing was written under
`/home/prashr/mapformer/runs/` (no `runs/torus_lrsweep/`), the main tree is clean, the SIGNSEP jobs on GPU 0 were not touched.

## Verified (no action)

1. **Command lines.** `DRV_DRYRUN=1 DRV_GPUS=1 DRV_MAXPG=2` in a scratch clone: md5 guard, 48 launches, seed outer / rate /
   arm inner, every line identical after removing variant / seed / lr / output-dir:
   `--weight-decay 0.05 --n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0
   --schedule cosine --data-workers 3 --fast-attn --epochs 900 --device cuda:1`; fails closed (MISSING x 48, no marker).
   Against `run_generality.sh` `torus()` with `--epochs 900`: the same flag set plus `--weight-decay 0.05`. A checkpoint
   trained with the driver's line differs from GENERALITY SO4 s700's stored `args` only in the run-specific keys and in
   `weight_decay` / `snapshot_every`, which did not exist then (absent in GENERALITY's args; the new defaults reproduce the
   old path). `MODE=pilot` dry run: AO at 1e-2 / 3e-2, seeds 2490-2491, wd 0.05, separate root `runs/torus_lrsweep_pilot`,
   fails closed.
2. **Code since GENERALITY.** GENERALITY's `code_md5.txt` matches the current tree except `model_wrap_exist.py`,
   `train.py`, `train_variant.py` (and two files these arms do not use). The diffs (47a766b / 62334d9 -> HEAD) are additive:
   `epoch_callback=None` default (bitwise-tested), `--weight-decay` default 0.05 = train()'s old default, `--snapshot-every`
   off, `freeze_latent` read with `getattr(..., False)`, new eval-only functions. The 1e-3 cell runs GENERALITY's
   computation. (The prereg's "no code changes" is inaccurate in wording only: MINOR 8.)
3. **lr and decay reach the optimiser.** `train()`: `AdamW(lr=lr, weight_decay=wd)`, `LambdaLR` (5% linear warm-up, cosine
   to 0.1 x lr), so decoupled decay per step is lr_t x wd. `decay_exponent` reproduces 2414.5 x lr (2.41 / 7.24 / 24.15 /
   72.44) and LRSWEEP's ring 805 x lr. Grad clip 1.0.
4. **Shared init and stream.** SO and AO built through train_variant's path (`manual_seed(s)`, GridWorld, class) have
   identical state dicts at 2400, 2403, 2407. The stream seed is `torch.initial_seed()` (= the seed); nothing in `train()`
   draws RNG as a function of lr; both arms draw dropout identically.
5. **`--fast-attn` is valid for both classes.** Their layers are plain `WMTransformerLayer` (train_variant refuses
   otherwise). Explicit vs SDPA, eval mode, T = 512: float64 logits max |diff| 1.3e-15, gradient relative diff 2.2e-16;
   float32 logits 7.8e-07 / 6.6e-07. Training-time dropout RNG differs from the explicit path (rule 19), irrelevant here
   (every cell uses SDPA; the 1e-3 cell is a fresh-seed replication, not a bitwise one).
6. **Analysis machinery.** `label_from`, `select`, `b1_test`, `b1_decide`, `holm`, `sel_score`, `_profiles` imported from
   `analyze_lrsweep` (last changed 7ddb263, LRSWEEP Amendment 1; its md5 equals LRSWEEP's `runs/lrsweep/code_md5.txt`
   record). Three rates: C(16, 8) = 12,870 relabellings, exact;
   decision p = max(selection-aware, naive Fisher / permutation); `r_so_low` = lowest rate at SO's maximum count, used only
   in the reading (LRSWEEP Amendment 1 MAJOR 1). `--selftest` passes (2 s): every B1 label and reading reached, STAGE 2
   TRIGGERED + edge qualifier, every stage-2 reading, tie-breaks, one-rate null = Fisher, clock_table angles = model
   angles. `--power` (51 s) reproduces Sec. 5 exactly (deterministic: AO <= 3/8 fires at 0.0256, 4/8 0.077; simulated
   0.99 / 0.94 / 0.76 / 0.29 at p = 0.1 / 0.2 / 0.3 / 0.5; size 0/400). No label is guaranteed.
7. **Inputs of the analysis.** eval JSON keys `0.0|<V>|<T>` -> `[seed, acc, nll]` and strata keys `<V>|<s>|1024` ->
   `plain_lag<128` / `plain_lag>=128` / `wrap` -> `acc`, as read (checked on a CPU smoke checkpoint through the driver's
   eval command lines). A NaN-weight model evaluates without crashing (argmax of NaN logits) and is scored 0 by the
   analysis. The model-derived secondaries return NaN, not exceptions, on a zero step map (heavy decay) and on a 1e4-scaled
   one. The clock / map ratio is `clock_table.ratios` loaded from its file (symmetric map, usage pooled over heads, per-head
   values, near-pi share), torus pairs (0, 1), (2, 3) = N/S, W/E (`environment.ACTION_DELTAS`), |.| applied for AO.
8. **Driver.** `flock -n` lock shared by all modes; DRV_GPUS / DRV_MAXPG required and assigned after sourcing (rule 27);
   `CUDA_DEVICE_ORDER=PCI_BUS_ID`; clean-tree check before any wait. The md5 guard list (99 modules from `sys.modules` +
   clock_table + lib + driver) covers every `mapformer.*` module actually imported at run time by `train_variant`
   (`-X importtime`, 88 modules), `eval_noise_refine` and `eval_rank_strata`; all are tracked. Guard re-checked before the
   evals (and before the pilot analysis); artifacts-then-marker; resume skips finished checkpoints (written only at the end
   of training) and running ones (`running()` matches `--output-dir OUT ` exactly); `drv_wait_dir "$R/"` cannot match the
   pilot root (`torus_lrsweep_pilot/`). Stage 2: `AO_rate` 0.01 -> "1e-2" tag, wd `repr(5e-5/lr)`, own subroot, refuses
   without the trigger or the md5 record.
9. **Modified docs passing the clean-tree check is safe.** No guarded module reads a tracked non-code file (grep of every
   `open` / `json.load` / `torch.load` in the 99 modules: only run artefacts). The prereg itself is protected.
10. **Weight decay coupled to lr** is disclosed (Sec. 1), with the stage-2 control for the SLB branch (but see MAJOR 1).

## Findings

### MAJOR 1 -- COST CLOSES / NO COST / UNMEASURED at AO's best >= 1e-2 attribute to lr what may be the coupled decay
At 1e-2 the decay exponent is 24 vs 2.4 at the default. LRSWEEP's MAJOR 2 argued decay at high rates could only HURT |.|
(it shrinks the angle scale), so a control was needed only for SIGNED LEARNS BETTER. On this task the registered evidence
points the other way too: RANKWD (registered, T = 1024 torus, same recipe) found rank 4 needs AdamW decay (8/8 vs 1/8 at wd
0) and that decay reshapes omega. So "AO learns the map at 1e-2" may be "AO learns it with 10x the decay". The reading then
says "on the torus the barrier learns the map at its own larger rate ... a learning-rate-sensitivity (conditioning)
result", which the batch cannot separate. **Fix (preferred):** trigger stage 2 whenever AO's selected rate is >= 1e-2 and
B1 is SLB, NSC or NLB (any reading that rests on AO's high-rate cell); stage-2 readings for the closing branch: stage 2 NSC
or NLB -> "COST CLOSES AT HIGHER LR (holds at matched decay)"; SLB -> "COST CLOSES WITH COUPLED DECAY (lr alone does not
close it)"; UNMEASURED / UNREADABLE -> "COST CLOSES at higher lr with coupled decay; lr and decay not separated". Expected
cost small (16 runs, ~8 h, only in a branch the prereg itself puts at ~0.1). **Minimum fix (wording only):** COST CLOSES and
the UNMEASURED clause say "at a larger rate with its coupled decay (lr x wd 10x the default's; not separated)".

### MAJOR 2 -- the UNSTABLE qualifier's fixed 2.08 sits inside the per-seed context-free loss, and counts SOLVED runs
The training map is fixed per seed, so a context-free model's loss is that map's revisit-observation entropy, not the
analytic 2.079. Measured (`check1.py` map entropy; `check2.py` revisit targets, 6 x 16 walks of 1024):

| seed | 2400 | 2401 | 2402 | 2403 | 2404 | 2405 | 2406 | 2407 | 2490 (pilot) | 2491 |
|---|---|---|---|---|---|---|---|---|---|---|
| map entropy | 2.061 | 2.073 | 2.102 | 2.051 | 2.054 | 2.070 | 2.094 | 2.086 | -- | -- |
| revisit-target entropy | 2.064 | 2.068 | 2.115 | 2.064 | 2.048 | 2.065 | 2.082 | 2.090 | 2.056 | 2.090 |

So one post-warm-up epoch at the context-free level trips RUNAWAY on 2402 / 2406 / 2407 but not on the other five; a run
0.03 above its own context-free level escapes on 2404. (The timing pilot's 2.03-2.05 on seed 2490 is at its context-free
level 2.056, so "context-ignoring" is right.) GENERALITY AO4 at 1e-3 peaks at 1.64-1.89 right at the end of warm-up
(epochs 45-50), and the pilot shows |.| loss RISING toward the context-free level as the peak rate is reached; at 3e-3 a
transient approach is plausible, and the flag would then depend on the seed's map. `reading()` counts every flagged run at
AO's best rate, SOLVED or not, and two of them append "no learnability sentence rests on this rate" to the headline.
COLLAPSED at 1.9 is fine: 0.15-0.21 below every context-free level, and no known healthy run (GENERALITY AO4 / SO4 tails
0.0006-0.85) ends above it. The step clause (1e3) is ~100x GENERALITY's final step maxima (3.0-9.8). **Fix:** RUNAWAY's
loss clause = any post-warm-up epoch above H_map(seed) + 0.05 (H_map computed in the analysis from `GridWorld(64, 16, 0.5,
seed).obs_map`; the 0.05 margin covers the map-vs-revisit difference, <= 0.013 here, and epoch noise); the qualifier counts
UNSTABLE runs that are not SOLVED; print per flagged run its max post-warm-up loss minus H_map and its tail.

### MINOR 1 -- the pilot rule's tie-break is noise when both rates collapse
If AO collapses at both 1e-2 and 3e-2 (the timing pilot and the ring's 3e-2 make this the likely case), both have 0 SOLVED
and tails at their context-free levels; "lower mean selection score" then picks by map entropy, i.e. a coin flip that adds
16 runs (7-8 h). It cannot bias the batch (outside seeds, decided before batch data) but it can waste a third of a day.
**Fix:** EXTEND iff k(3e-2) > k(1e-2), or k equal, mean score at 3e-2 lower by >= 0.05, and no 3e-2 pilot run UNSTABLE.
Optional: use seeds 2492-2493 (unused) so no pilot run repeats the partly observed timing run (s2490 AO 1e-2).

### MINOR 2 -- the model-derived secondaries are computed before the verdict
`load()` builds the usage stream, the clock module and every model secondary inside `run_record`, before `analyse()`
writes the verdict JSON, so "verdict written before the secondaries" holds only for the printed ones. I found no input that
raises (zero and huge step maps give NaN), so this is robustness only. **Fix:** wrap the `want_model` block in try/except
(NaN + a logged warning), or compute it after the verdict is written.

### MINOR 3 -- code edits during a 20-25 h batch are caught only at the evaluation step
The guard is checked at start and before the evals; a `git pull` on koopman mid-batch makes later launches import new
code and the batch then fails at hour ~24. **Fix:** call `drv_md5_guard "$GROOT" "${GUARD[@]}" || drv_fail ...` in
`torus()` before each launch (cheap), and a runbook line: no pull / checkout on koopman while the batch runs.

### MINOR 4 -- stage 2 inherits the batch's md5 record, so its own check cannot edit any guarded file
The prereg promises stage 2 "its own short independent check"; any fix to the driver or the analysis would then make
`MODE=stage2` refuse. **Fix:** state now that stage-2 fixes are limited to text, or that stage 2 records its own md5 under
`$R/stage2` after checking that the training modules (`train_variant`, `train`, `model_wrap_exist`, `model`, `model_sign`,
`environment`, `data_parallel`) are unchanged against `$R/code_md5.txt`.

### MINOR 5 -- `pilot()` asserts only seed, lr, epochs and n_batches
It reads the checkpoints without the batch's recipe asserts (variant, wd, T, batch, fast_attn, data_workers, environment).
**Fix:** apply `RECIPE` and the `args` checks of `run_record` in `pilot()` (no evaluation needed).

### MINOR 6 -- n_trials and the eval map are not asserted
As LRSWEEP's MINOR 7: the eval JSON does not record n_trials (100), env seed (10000) or the walk seed; only the driver fixes
them. No action beyond a sentence (changing the evaluator would change a guarded module).

### MINOR 7 -- cross-references
The driver and analysis cite "prereg Sec. 6" (pilot; it is Sec. 8), "Sec. 4" (stage 2; it is Sec. 3) and "Sec. 3"
(UNSTABLE; it is Sec. 2). Text only; fix with the amendment.

### MINOR 8 -- "no code changes" (Sec. 1)
Replace with: "train.py, train_variant.py and model_wrap_exist.py changed since GENERALITY (47a766b / 62334d9); the changes
are additive and leave this path unchanged (audit 2026-10-24 item 2)".

### MINOR 9 -- jobs per GPU
`project_state.md` records that packing was measured slower on koopman, and RANKWD's 4.2 s/epoch for two vs 2.0 alone means
DRV_MAXPG=2 costs ~5% over serial (25.2 h vs 24.0 h without `--fast-attn`). Either is acceptable; MAXPG=1 is marginally
faster and keeps a failure to one run.

### Not findings (checked)
- The null is unpaired although SO / AO share init and stream at a seed: valid and conservative (LRSWEEP Amendment 1 MINOR 2).
- The 1e-3 cell is not a reproduction control (fresh seeds, another GPU), so the absence of a bitwise check is fine.
- SO is gated at its best rate (UNREADABLE if < 4/8 at every rate), as LRSWEEP.
- Selection never uses held-out accuracy; an UNSTABLE rate cannot be selected unless every rate fails.

## Cost and trimming
48 runs x 900 epochs: 24-25 h without `--fast-attn` gain (RANKWD's koopman timing), ~19-22 h if SDPA saves 10-20%; plus
~0.5 h evals. Pilot ~2 h; a 3e-2 extension +7-8 h; stage 2 +7-8 h when triggered. **No trimming keeps the primary at its
registered power.** The selection-aware null relabels whole rate profiles between arms, so every rate must exist for BOTH
arms on every seed (dropping SO's 1e-2 breaks exchangeability); dropping 1e-3 removes SO's known best rate and the in-batch
replication (GENERALITY cannot be pooled, rule 12); n = 6 makes 2/6 vs 6/6 non-significant (Fisher 0.06), losing the
ring-size cost. The only optional saving is skipping the pilot (~2 h), accepting a possible "not bracketed" qualifier; with
MINOR 1's rule the pilot is worth keeping.

## Proposed Amendment 1 (before launch)

> **Amendment 1 (independent check, `docs/audits/2026-10-24/g4_torus_lrsweep_audit.md`, before any batch run).**
> - **Decay in the closing branch (MAJOR 1).** Stage 2 (SO / AO at AO's selected rate, wd = 5e-5 / lr, seeds 2400-2407)
>   triggers whenever AO's selected rate is >= 1e-2 and B1 reads SIGNED LEARNS BETTER, NO SIGN COST or NON-NEGATIVE LEARNS
>   BETTER. Readings for COST CLOSES AT HIGHER LR: stage 2 NSC / NLB -> "holds at matched decay"; SLB -> "COST CLOSES WITH
>   COUPLED DECAY (lr alone does not close it)"; UNMEASURED / UNREADABLE -> "at higher lr with its coupled decay; not
>   separated". The SLB readings are unchanged. Until stage 2 reports, COST CLOSES and UNMEASURED's "may shrink with the
>   rate" read "with the rate and its coupled decay".
> - **UNSTABLE (MAJOR 2).** RUNAWAY's loss clause becomes: an epoch after the warm-up above H_map(seed) + 0.05, H_map = the
>   entropy of the seed's training map (blank + 16 types; 2.051-2.102 on seeds 2400-2407). The step clause (1e3) and
>   COLLAPSED (final-5% > 1.9) are unchanged. The qualifier counts UNSTABLE runs that are not SOLVED; each flagged run's
>   max post-warm-up loss minus H_map and its tail are printed.
> - **Pilot rule (MINOR 1).** EXTEND iff the pilot AO solves more at 3e-2 than at 1e-2, or solves as many with a mean
>   selection score lower by >= 0.05 and no 3e-2 pilot run UNSTABLE. [Optional: pilot seeds 2492-2493.]
> - **Code, no design content (MINOR 2, 3, 5).** Model-derived secondaries cannot block the verdict (try/except -> NaN);
>   the md5 guard is re-checked before every launch; `pilot()` applies the batch's recipe asserts.
> - **Stage 2 check (MINOR 4).** Stage-2 fixes are text-only, or stage 2 records its own md5 after verifying the training
>   modules against the batch's record.
> - **Text (MINOR 6, 7, 8).** n_trials / eval seeds fixed by the driver only; cross-references corrected; "no code changes"
>   replaced by the additive-diff statement.
> - **Launch (MINOR 9).** DRV_MAXPG 1 or 2 (cost ~equal); no git pull / checkout on koopman while the batch runs.
