# FACTORIAL_N8 -- independent code check (rule 29), results-blind

Checked 2026-10-24/25 at HEAD `92002e0` with the untracked `FACTORIAL_N8_PREREG.md`, `analyze_factorial_n8.py`,
`run_factorial_n8.sh`, `run_factorial_n8_pilot.sh`. No training launched, nothing committed, no file under review edited.

What was run:
- `python3 -m mapformer.analyze_factorial_n8 --selftest` (CPU): **SELFTEST ALL PASS** (FACTORIAL's `analyze_factorial.py`
  md5 equals its recorded one; FACTORIAL's own smoke passes; floors reproduce seed 10 n 32255 / blank 0.5049 / retrace
  0.8411 / wrap 0.081; s500 stratum counts and blank floor equal `FACTORIAL_L1_STRATA.json`).
- CPU script: `load_batch(REF)` on FACTORIAL's 16 real checkpoints and JSONs (every assertion passes; FACTORIAL's `args` are
  identical across its four seeds in each cell); the five pilot checkpoints' `args` vs FACTORIAL's of the same cell
  (minus `SKIP_ARGS`): the ONLY difference is `epochs` (10/15 vs 900); no config-key differences.
- Scratch clone (`git clone --shared`) with `setsid` / `nohup` / `python3 -m mapformer.*` stubs on PATH that refuse:
  (a) `DRV_DRYRUN=1 DRV_GPUS=1 DRV_MAXPG=4`: exit 0, 16 DRYRUN launches in FACTORIAL's order (seed outer; L4, LOOP, L1 RoPE,
  L1 Vanilla_r4) with the right `--n-layers`, 21 DRYRUN evals, DRYRUN analysis, md5 guard over 104 files;
  (b) no `DRV_GPUS`: REFUSED, exit 1; (c) real mode, guarded files untracked: "REFUSED: a guarded file is not committed",
  exit 1, before any md5 file is written.
- Every module in the guarded import closure (94) against `4041da9`: 86 byte-identical, 6 new
  (`analyze_factorial_n8`, `model_mono_baselines`, `model_pope_delta`, `model_rankclock`, `model_srope`, `model_wrap_exist`),
  2 changed (`train.py`, `train_variant.py`) -- the prereg's claim holds.

## BLOCKER
None.

## MAJOR

**M1 (operational). An 8 h, 104-file md5 guard in a working tree another builder is editing.**
`run_factorial_n8.sh:24, 58, 70, 87`. The guard covers the whole import closure of `train_variant` (`train_variant.py`,
`train.py`, `stats_core.py`, `data_parallel.py`, `environment.py`, ~85 `model_*.py`). G-3 is active in the same tree now
(three modified tracked files) and earlier batches added variants by editing `train_variant.py` (G-1, DELTA_FREE, SROPE).
Failure scenarios: (i) an edit to any guarded file during the ~6 h launch window makes the per-launch guard call
`drv_fail` -- the remaining launches never happen, the already running jobs finish orphaned, eval and analysis never run;
(ii) an edit after the last launch fails "code changed before eval"; (iii) relaunch then refuses on md5 (correctly), so the
batch is stranded unless someone deletes `code_md5.txt` (a rule-22 breach that hides a mid-batch code change). Also, at
the moment the clean-tree check (`:24`) refuses a real launch because G-3's `G3_SSM_PREREG.md`, `analyze_ssm_sign.py`,
`run_ssm_sign.sh` are modified. Nothing here is a correctness bug; the guard is doing its job.
Fix (amendment, no code change): declare a freeze on the 104 guarded files from launch to `.factorial_n8_done` and tell
the other sessions (G-3's own files -- `model_ssm_sign.py`, `train_ssm_sign.py`, ... -- are NOT in this closure, so G-3 can
proceed in them); recovery rule: if the guard aborts, restore the offending file to the launch commit
(`git checkout <launch-sha> -- <file>`) so the md5 matches again and rerun the driver (it skips finished and running
runs); never delete `runs/factorial_n8/code_md5.txt` once any checkpoint exists or any job is running.

## MINOR

**m1. The replication summary covers two of the five registered readings, and "PARTIAL" covers "nothing fired".**
`analyze_factorial_n8.py:132-140`; prereg `:69-70` vs the question (`:7` "Do FACTORIAL's registered readings hold") and
`:149` ("whether FACTORIAL's four readings recur"). REPLICATES needs only C1 and C2 DOES NOT CATCH UP; C0 and both C3 lines
(FACTORIAL: PATH WINS, HELPS, HELPS) are not summarised, so a batch where C3 fails would still read REPLICATES. And with C1
and C2 both UNMEASURED the summary reads "PARTIAL", although nothing replicated. It is a pure function of registered labels
(good) and REPLICATES / DOES NOT REPLICATE are sensible for the catch-up question. Fix: name it "REPLICATION SUMMARY (catch-up
readings C1, C2)"; PARTIAL only when exactly one of C1 / C2 is DOES NOT CATCH UP; "NOT SHOWN AT n = 4" when neither fires
and neither is WITHIN; add one descriptive line "same label as FACTORIAL: C0 yes/no, C3_L4 yes/no, C3_Loop yes/no". Match on
the label head (`R[k].startswith("INDEX (")` + the label words, or split at ':') rather than substring anywhere.

**m2. GPU 1 is G-3's default too; the "~8 h on GPU 1 alone" estimate assumes it is not shared.**
Prereg `:135`; `run_ssm_sign.sh:26` defaults `DRV_GPUS=1` at 4 per GPU; `lib_driver.sh:40-43` counts every one of the
user's `mapformer.train_*` processes on a device, so concurrent batches share four slots and both slow down (up to ~2x
over the overlap). No numerical effect. GPU 0 is idle now (21 MiB used). Fix: state the GPU plan in the amendment (n8 on
GPU 1 with G-3 elsewhere, or `DRV_GPUS="0 1"` -- device has no numeric effect and FACTORIAL itself ran on GPU 0) and the
wall estimate that goes with it.

**m3. A stale `code_md5.txt` blocks a re-run after the amendment.**
`run_factorial_n8.sh:48`, `lib_driver.sh:72-83`. Amendment 1 edits `FACTORIAL_N8_PREREG.md`, which is guarded: any
existing `runs/factorial_n8_dryrun/code_md5.txt` makes the next dry run ABORT, and the same holds for `runs/factorial_n8`
after an aborted start. Today neither directory exists (checked; only `factorial_n8_dryrun.log` from the builder's 23:00 dry
run remains). Fix: amendment text "before the post-amendment dry run delete `runs/factorial_n8_dryrun/`; before the launch
`runs/factorial_n8/` must not exist".

**m4. `experiment_audit` cannot read this layout.** Prereg `:139`; `experiment_audit.py:186-210`. It finds no
`(json, pt)` pairs, runs only check [1] (convergence on the checkpoints' loss curves) and exits 2 by design. Fix: prereg
says "exit 2 is expected on this layout; read check [1]; the registered run classes are `stats_core.classify_run` in the
analysis".

**m5. Selftest gaps** (`analyze_factorial_n8.py:266-352`; it can fail -- each `chk` prints BAD and exits 1 -- and it reaches
every registered label). Gaps: presence checks only (the "replicates" case does not assert the absence of UNMEASURED /
WITHIN); untested replication branches "C1 and C2 both UNMEASURED" and "C1 DOES NOT CATCH UP + C2 WITHIN"; no DOES NOT
CATCH UP through the permutation route alone (Fisher n.s., e.g. index 1/4 SOLVED with all accuracies below path); the
"within" pooled check (`:335-336`) is a disjunction of the two outcomes that case can produce; the flag-drift test changes a
value but never adds or removes a key. Fix: add those cases (a few lines each). Not blocking: the registered decision
functions are FACTORIAL's own, imported unchanged and md5-checked.

## NOTE

1. **COMMON flags** (`run_factorial_n8.sh:52-53, 60-61`) are token-for-token `run_factorial.sh:26-28`; OMP_NUM_THREADS 3,
   `PYTORCH_CUDA_ALLOC_CONF`, MAXPG 4, MINFREE 4000, SPACING 30 also match (`:19-21` vs `:9-11`). `load_batch`'s assertion
   (`analyze_factorial_n8.py:100-103`) compares with `ref_args[cell][0]`; sufficient, since FACTORIAL's args are identical
   across its seeds (checked). Arms: RoPE L1, RoPE `--n-layers 4`, RoPELooped `--n-layers 1`, Vanilla_r4 L1 -- as FACTORIAL.
2. **Fisher with AF.N = 4**: correct for the fresh batch, and `registered()` asserts n = 4 per cell (`:123`); the pooled
   line computes Fisher with its own n (`:146`) and scales SOLVED >= 3/4 to >= 0.75 n (6/8).
3. **Floor port**: `floors_walk` is `docs/audits/2026-10-05/score_rank_floors.py:44-55` verbatim plus the stratum from
   `eval_rank_strata.kinds_vec`; the selftest pins it to the original output and to FACTORIAL's stratum counts. RoPE and
   Vanilla_r4 have identical stratum counts per seed in `FACTORIAL_L1_STRATA.json` (eval re-seeds per variant and length,
   `eval_noise_refine.py:45`), so "same walks" holds for every cell. Wording: prereg `:87` "copies the observation 2j tokens
   back" is 2j steps (4j interleaved tokens).
4. **train.py / train_variant.py since 4041da9 are off the default path for these variants**: `epoch_callback` defaults
   None and is skipped; `weight_decay=0.05` is passed explicitly = `train()`'s default (`train.py:36, 63`); `_snapshotter`
   returns None at `--snapshot-every 0` before any side effect; the new module-level imports have no module-level
   `torch.backends` / seed / default-dtype statements, and seeding happens in `main()` (`train_variant.py:558`) after all
   imports. Pooling as DESCRIPTIVE is therefore conservative but correctly labelled in every pooled line (`:153, 205, 211,
   218`) and in the prereg.
5. **CUDA_DEVICE_ORDER=PCI_BUS_ID**: the host has two identical RTX 4090s (PCI 41:00.0 = index 0, 61:00.0 = index 1); the
   default FASTEST_FIRST order breaks ties by PCI bus order, so cuda:N == nvidia-smi N with or without it. It does not enter
   the checkpoint `args` and has no numeric effect.
6. **Seeds 2800-2803 are unused**: no run dir, log, driver or prereg names them outside the four files; the only other hits
   are decimals ("0.2800") in `LM200_ABLATION.md`. Pilot seeds 2890-2891 are outside the batch.
7. **`--fast-attn` / rule 19**: both batches `--fast-attn`, neither `--deterministic`; asserted per run
   (`analyze_factorial_n8.py:95-96`). A fresh-seed replication, never a determinism check -- stated in the prereg.
8. **Eval lengths and strata** identical to FACTORIAL (512 / 1024 / 2048, 100 trials, held-out env 10000, strata at 1024
   with `--check-json`, own map per seed).
9. **Branch rules** are FACTORIAL's prereg + Amendment 1 (D1 DESCENDING tag, D2 renamed margin label, D5 no correction),
   imported, not copied. The 0.02 margin rule sits below per-seed spread (RoPE_L4 sd ~0.03) -- inherited by design.
10. **Artifacts-then-marker**: `drv_done` requires the analysis text, verdicts JSON and the three eval JSONs; strata
    failures are fatal earlier; own-map failures are tolerated as secondary (as FACTORIAL). A refused real launch leaves
    empty `runs/factorial_n8/{L1,L4,LOOP}/p0` (the `mkdir` at `:32` precedes the refusal checks); harmless.
11. **Guard scope**: the analysis also reads FACTORIAL's 16 checkpoints, its `*_OWN_s*.json` and
    `runs/factorial/code_md5.txt` (selftest), none md5-guarded. Read-only and not edited by anyone; accept, or add them.
12. Prereg `:44` says 93 modules; the closure is 94 (it adds `analyze_factorial_n8`), 104 guarded files in all.
13. **Memory / time**: pilot 1.7 GB per deep job, 5.3 GB for four, on a 24 GB card with MINFREE 4000: fine. FACTORIAL's
    8 h 07 min on one 4090 at 4 jobs is the right reference if GPU 1 is not shared (m2).

## Proposed "## Amendment 1"

```
## Amendment 1 (2026-10-25, after the independent check `docs/audits/2026-10-25/factorial_n8_audit.md`; before launch)
The check found no blocker. It ran the selftest (ALL PASS), loaded FACTORIAL's 16 checkpoints through load_batch (all
assertions pass), confirmed the pilot checkpoints' args equal FACTORIAL's except epochs, confirmed 86 of the 94 guarded
modules byte-identical to 4041da9 (6 new, unused; train.py / train_variant.py changed off the default path), and dry-ran
the driver (16 launches in FACTORIAL's order, refusals on missing DRV_GPUS and on uncommitted guarded files). Changes:
- A1 (code freeze, M1). From launch to `.factorial_n8_done` the 104 md5-guarded files (the import closure of
  train_variant / eval / analysis, ckpt_guard.py, lib_driver.sh, the driver, this file, FACTORIAL's six JSONs) are frozen
  for every session. If the guard aborts, the offending file is restored to the launch commit and the driver re-run (it
  skips finished and running runs); `runs/factorial_n8/code_md5.txt` is never deleted once a checkpoint exists or a job runs.
- A2 (GPU plan, m2). <state: GPU 1 alone with G-3 on GPU 0 | DRV_GPUS="0 1">; wall estimate <~8 h | ...>. The device has
  no numeric effect.
- A3 (replication summary, m1). Renamed "REPLICATION SUMMARY (catch-up readings C1, C2)": REPLICATES iff both DOES NOT
  CATCH UP; DOES NOT REPLICATE if either is WITHIN 0.02 OF PATH; PARTIAL iff exactly one DOES NOT CATCH UP; otherwise NOT
  SHOWN AT n = 4. A descriptive line reports whether C0, C3 (RoPE_L4) and C3 (RoPE_Loop) carry FACTORIAL's label. Labels
  are matched on their head. Selftest cases added: both UNMEASURED, C1 DNCU + C2 WITHIN, DNCU by permutation alone,
  absence checks, a flag-drift case with an added key.
- A4 (procedure, m3 / m4). Before the post-amendment dry run `runs/factorial_n8_dryrun/` is deleted; at launch
  `runs/factorial_n8/` does not exist. `experiment_audit` exits 2 on this layout by design (no (json, pt) pairs): read its
  check [1]; the registered run classes are `stats_core.classify_run` in the analysis.
- Wording: the retrace rule copies the observation 2j STEPS back; the guarded closure is 94 modules (104 files).
Selftest re-run after the A3 edit: <ALL PASS>.
```

## Verdict
Launchable after the amendment (A1 and A2 are procedural; A3 is a small edit to `analyze_factorial_n8.py` plus a selftest
re-run, and must be committed before launch because the analysis is md5-guarded). Nothing found affects the arms, the
flags, the evaluation or the registered decision rules.
