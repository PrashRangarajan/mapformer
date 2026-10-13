# CLOCK_SPEED_REP independent code check (rule 29), 2026-10-12

Scope: `CLOCK_SPEED_REP_PREREG.md`, `analyze_clock_speed_rep.py`, `run_clock_speed_rep.sh` (+ `lib_driver.sh`), diffed against
the audited CLOCK_SPEED batch (`clock_speed_audit.md`). `runs/clock_speed_rep` does not exist and was not created in the repo.
`runs/clock_speed/p0` (finished) and `CLOCK_SPEED_VERDICTS.json` were read to test the analysis. Nothing in the repo modified
except this file; no GPU job (`nvidia-smi` queried only). Scratch: `.../scratchpad/audit_rep/` (`t1.py` end-to-end test,
`t2_power.py`, dry-run copy of the driver under `drv/`).

## Verified correct

- **Driver diff** (`diff run_clock_speed.sh run_clock_speed_rep.sh`): only header, log, lock (`.run_clock_speed_rep.lock`), run
  dir (`runs/clock_speed_rep/p0`, in the md5 guard, the wait regex and REQ), default `DRV_GPUS` 0 -> 1, seeds 230-245, arms
  Signed_r4 / Abs_r4 (CARoPE removed from both loops), GUARD adds `analyze_clock_speed_rep.py` and the new driver, artifact count
  64, analysis module and outputs, done marker `.clock_speed_rep_done` gated on `CLOCK_SPEED_REP_ANALYSIS.txt` and
  `CLOCK_SPEED_REP_VERDICTS.json`. Training flags byte-identical to CLOCK_SPEED's. Seed outer, arm inner. No stale name: the wait
  regex `runs\/clock_speed_rep\/p0` and CLOCK_SPEED's `runs\/clock_speed\/p0` cannot match each other's jobs.
- **Same code as batch 1**: every file in `runs/clock_speed/code_md5.txt` (train_recency, environment_recency, train_variant,
  model, model_sign, model_baseline_rope, train, analyze_clock_speed, stats_core, lib_driver) has the same md5 today. Working
  tree clean; the three new files are committed at 4352a4b. Both GPUs are RTX 4090, so the GPU change (0 -> 1) does not change
  hardware or TF32 numerics between batches.
- **Dry run** (scratch copy, `DRV_DRYRUN=1`, `DRV_GPUS` unset so the default applies, spacing 0): 32 unique (arm, seed)
  launches, seeds 230-245, **all 32 `--device cuda:1`**; md5 list 12 files; with no artifacts 64 MISSING and "done marker NOT
  set" (fails closed).
- **GPU 0 is unreachable** with the default or the documented launch line: `drv_pick` iterates `DRV_GPUS` only. Slot counting is
  per device (`drv_ntrain` counts `mapformer.train_*` python3 processes whose argv contains `--device cuda:<g>`); FACTORIAL
  (`run_factorial.sh`, `DRV_GPUS` default 0, its 4 live jobs all `--device cuda:0`, its eval step also via `drv_wait_slot` on
  GPU 0) and this batch therefore count disjoint sets and cannot block or overfill each other. Distinct lock files and done
  markers. GPU 1 at check time: 52 MiB used, no process. Only caveat (as before): an inherited `DRV_GPUS` in the launching shell
  overrides the default; the launch line sets `DRV_GPUS=1` explicitly. CPU: 32 cores, load ~4.7; 4 x OMP 2 is fine.
- **Seeds 230-245 are fresh**: no path under `runs/` contains `_s23[0-9]` or `_s24[0-5]` (any task); recency seeds in use are
  0-47, 100, 200-207, 210-225; no driver other than this one loops over them (`run_level15_meets_gated.sh` hit is a param count).
- **Analysis**: imports `branch` and `first` from `analyze_clock_speed` unchanged (md5 identical to batch 1). `load()` repeats
  batch 1's provenance asserts (variant, seed, k_max, min_gap, 300 epochs; T / k_max / min_gap from the log header). End to end
  on batch 1's runs (module `R`, `SEEDS` = 210-225 and `REPO` redirected to scratch): reproduces CLOCK_SPEED exactly (S1
  UNMEASURED -0.197, p 0.0533, CI [-0.390, -0.000], 12/16; arm lines identical; AUC lists identical to
  `CLOCK_SPEED_VERDICTS.json`). Runtime ~60 s with the pooled line. The pooled line is labelled "POST HOC" and uses only
  210-225 (not CLOCK_SIDE's hypothesis-generating 200-207); pairing by list index is correct (verdict rows are in seed order).

## BUG

None that blocks launch.

## DESIGN / reporting (state in an amendment before launch)

**R1. Power: UNMEASURED is about as likely as firing.** Bootstrap of batch 1's paired d (mean -0.197, sd 0.372), n = 16,
registered rule, 2000 reps:

| true effect | NON-NEGATIVE FASTER fires | SIGNED fires |
|---|---|---|
| -0.268 (CLOCK_SIDE, post hoc selected) | 0.75 | 0 |
| -0.197 (CLOCK_SPEED) | 0.52 | 0 |
| -0.15 | 0.34 | 0.001 |
| -0.10 | 0.19 | 0.002 |
| 0 | 0.04 | 0.015 |

The estimate already shrank -0.268 -> -0.197 between the selected and the first fresh batch (winner's curse); at a further
shrink power is ~1/3. NO DIFFERENCE needs the 95% CI inside +/-0.10; at sd 0.37 the n = 16 half-width is ~0.19, so it is
practically unreachable. State: expected power ~50% at batch 1's estimate; an UNMEASURED here is not evidence against the effect
and not a failed replication.

**R2. Wording if it fires.** Batch 1 did not fire, so there is nothing registered to "replicate". The correct statement is
"S1 fires on fresh seeds 230-245 (CLOCK_SPEED_REP); the first registered batch was UNMEASURED in the same direction (p 0.053)",
scoped as in batch 1: loss AUC within 300 epochs, one recipe (lr 1e-3, cosine, batch 16), 1 layer, one clock task, nothing
converged. If SIGNED fires, the two batches conflict and the clock side is unresolved.

**R3. Optional stopping and the pooled line.** This batch exists because batch 1 narrowly missed. Analysing it alone (as
registered) keeps its alpha; the pooled 32-seed reading does not (the pool includes a batch selected for being a near-miss).
Demonstration: pooling batch 1 with a copy of itself prints "NON-NEGATIVE LEARNS THE CLOCK FASTER, p 0.0049" -- the pooled line
can fire while the registered line is UNMEASURED. Pre-commit: (a) the pooled line can never upgrade the verdict or appear as a
headline; (b) this is the last batch on this contrast whatever the outcome (no third batch on a second near-miss).

**R4. Pooled-line mislabels (minor, cosmetic unless fixed before launch -- the md5 guard aborts the analysis if
`analyze_clock_speed_rep.py` changes after the batch starts).** (i) At n = 32 `stats_core.signflip_p` switches to Monte Carlo
(n > 20, 200k draws, seed 0), but `branch()` prints "exact sign-flip p" -- say "Monte Carlo" in the results file. (ii) No
check that `CLOCK_SPEED_VERDICTS.json` holds 16 seeds: truncated to 8, the line still says "pooled 32 seeds" while using 24
(tested). The file is not in the md5 GUARD. Suggest `assert len(V[arm]) == 16` and the rows' seeds == 210-225.

**R5. Carry-overs from `clock_speed_audit.md`, still unaddressed in the analysis.** m1: report the paired eval-NLL (the
`.json` has `nll` at T and 2T) or accuracy contrast beside the AUC verdict, since dropout inflation of training loss differs
by arm (0.146 Abs vs 0.171 Signed on CLOCK_SIDE: slightly anti-conservative for S1); the analysis prints only per-arm mean
accuracy. m5: provenance does not assert n_layers / d_model / n_heads or the `[fast-attn]` log line (flags verified identical in
the driver, so low risk). Neither changes the registered verdict; both can be computed post hoc from saved artifacts.

## Cost check
32 runs x 300 epochs = 9,600 run-epochs; batch 1 ran 12,000 in 1.6 h on one GPU (09:32-11:08), so ~1.3-1.5 h at the same
concurrency; the prereg's 2.5-3 h is conservative.
