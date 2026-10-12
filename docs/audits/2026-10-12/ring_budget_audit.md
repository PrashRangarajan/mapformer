# RING_BUDGET pre-launch audit (rule 29), 2026-10-12

Independent, read-only check of `RING_BUDGET_PREREG.md`, `analyze_ring_budget.py`, `run_ring_budget.sh` (+ `lib_driver.sh`)
at cd36648, before launch. Blind to results: no `runs/` output was read (`runs/ring_budget` does not exist). The trainer and
the Abs_r4 arm were audited in `ring_dial_audit.md`; this covers what is new.

Verified:
- Tree clean for every file on the path. The training code is byte-identical to RING_DIAL's batch commit 44efc5d
  (`git diff 44efc5d HEAD` is empty over the trainer, models, env, train.py, rescore_hook, stats_core, lib_driver).
- End to end on CPU: 8 fabricated Abs_r4 runs (900-epoch loss curves: early solve, late cliff, flat, slow descent, slow rise),
  read by a copy of the module with only `R` and the JSON path patched
  (`/tmp/claude-1002/.../scratchpad/audit_rb/`). It runs in 10 s. Provenance asserts pass. Classes are as
  expected. `epochs_to_solved` gives 167 / 609 for the two SOLVED runs and 901 for the rest.
  The re-score hook attaches to all 8 models (`STATS` hooked 8, fired 3200). The JSON serialises.
  `--smoke` ALL PASS. The decision boundaries were checked separately: k=6 SLOWER; k=5 and k=3 PARTIAL;
  k=2 with 4 STALLED PERSISTENT; k=2 with 3 STALLED STILL DESCENDING; k=0 with 4 STALLED PERSISTENT.
  `decide` matches the prereg text, and all four branches can be reached.
- The import is correct: `epochs_to_solved` is the SOLVED-only version from the B1 audit. Runs that are not SOLVED get budget + 1 = 901.
- The order matches RING_DIAL: `wrap_opposition` runs before `install`, and neither calls `.eval()`. The hook attaches at `evaluate`'s
  `model.eval()`. Abs_r4's layer is `model.WMTransformerLayer`, which is a KNOWN class. The re-score uses the same stream as the trainer's eval
  (te seed 10000, np seed 10**6, 400 trajectories).
- The hardcoded comparison "Abs 0.04, Signed 0.82" matches `RING_DIAL_ANALYSIS.txt` at p_plus 0.5.
- Driver: seeds 40-47, `--epochs 900`, `--p-plus 0.5`, `--arm Abs_r4`. `DRV_GPUS=0` is exported, so `drv_pick` loops over "0"
  only. The knobs are assigned after sourcing (rule 27). The `flock` lock, the `--output-dir` exact-match running check and the
  eval.json skip are all present. The md5 guard runs at start and again before the analysis. The analysis's exit code is checked. `drv_done` runs only
  after both artifacts exist, and the analysis runs from `/home/prashr` with CUDA hidden.

## BUG

None changes the registered verdict on the paths exercised. One label mismatch has low impact:

- **B1. RISING counts toward "STILL DESCENDING".** `classify_run(...)["registered"]` folds RISING (the final 10% sits >5% above
  the window before it) into DESCENDING. `decide` counts only STALLED, so a rising run pushes the verdict toward
  "STILL DESCENDING (budget-limited again)". A rising run is not descending. Under the 10%-floor cosine this is unlikely,
  but the label could be wrong. Fix: count `cls` and print RISING runs. Either count them as not-descending or name them in the verdict line.

## DESIGN

- **D1. The PERSISTENT / STILL DESCENDING split reads exactly what rule 3 says not to read.** Rule 3 says "a flat final window
  read after LR decay is NOT convergence (it passed stuck runs and runs still descending)". `classify_run`'s STALLED rule
  compares the means of epochs 811-900 and 721-810. Under the stretched cosine, the LR in those windows is 0.20 -> 0.12 and
  0.12 -> 0.10 of peak. At that LR, a run sitting on a plateau reads flat whether it is stuck or would escape with more
  LR or epochs. The plateau-then-cliff dynamics seen in the rank line are exactly this case.
  - Consequence: if k <= 2, PERSISTENT HANDICAP is close to the default outcome, and "persistent" overclaims.
  - Also: the windows are fractions of the run (90 epochs here, 30 at 300). A run descending at a constant 0.06-0.17%/epoch
    reads STALLED at 300 epochs and DESCENDING at 900. Classes are therefore not comparable across the two batches.
  - Recommended (amend before launch):
    - (a) Rename the branch to something budget-scoped, e.g. "NOT SOLVED AT 3x (flat at end of schedule)", and state that
      it cannot separate stuck from slow.
    - (b) Validate the rule on runs of known status (rule 3): classify RING_DIAL's 300-epoch Abs runs (6 unsolved at
      p_plus 0.5, 17 across the dial) from the committed checkpoints, CPU only. Then pre-state how to read the case where
      most of them were STALLED at 300 and this batch still solves more of them.
    - (c) Add a secondary that does not depend on the schedule. For each unsolved run, report the loss at epoch 300 and at
      epoch 900 (did the extra 600 epochs move it at all?) and the minimum loss. Print `tail` and `ratio` per run.
- **D2. The stretched cosine changes the budget AND the LR at every epoch.**
  - At epoch 300 the LR is 0.82 of peak here vs 0.10 in RING_DIAL. Integrated LR to epoch 300 is 261 vs 164
    peak-LR-epochs; the total is 493 = 3.0x.
  - It is a fair "3x budget, same recipe shape" test, and it matches how the project extended budget before
    (e.g. LOOP_RANK_E1800). The checkpoints hold no optimizer state, so an exact continuation was not available anyway.
  - The prereg should say so. "SLOWER, NOT STUCK" then means "budget-limited at 300 epochs with this schedule". It does not
    mean "the RING_DIAL trajectories continued".
  - An epochs-to-SOLVED below 300 cannot be read as "needed more than 300 epochs"; it is a schedule effect.
  - Cross-batch seeds: 2/8 on seeds 30-37 vs k/8 on seeds 40-47. The binomial sd of a count of 8 at p 0.25 is ~1.2 runs,
    so k = 3-4 is indistinguishable from 2/8.
  - Optional cleaner design: add a 300-epoch Abs_r4 twin on seeds 40-47 in the same batch. It costs +2,400 run-epochs, about
    +0.5 h. The twin has the same init and the same data stream (batches are seeded by index), so the comparison is a paired,
    within-batch SOLVED count at 300 vs 900 epochs, and the cross-batch confound goes away.
- **D3. SOLVED is a training-loss threshold, not "found the map".**
  - Abs's misses at 300 epochs are near-misses (acc 0.973). Three times the training can push a non-cancelling code under
    0.05 loss, so k >= 6 could fire without the wrap solution.
  - State in advance that the primary answers only "does the loss reach the threshold".
  - Declare in advance how the wrap secondary qualifies the label, e.g. "SOLVED with share cancelling < 0.2 = the threshold
    is reached with a non-cancelling code".
  - Name the accuracy at T = 512 (Abs 0.878 vs Signed 0.995 at 300 epochs) as the exact-map diagnostic.
  - Print per-run tail losses so the reader can see how close each run is to the threshold.
- **D4 (wording).**
  - PERSISTENT fires at k = 0 with 4 STALLED + 4 DESCENDING, i.e. when half the unsolved runs are still descending.
    Consider "a majority of unsolved runs STALLED" or keep the rule but say so.
  - k = 2 equals RING_DIAL's rate. Say explicitly that no contrast is tested and that the labels are descriptive.

## minor

- m1. Add `assert rescore_hook.STATS["hooked"] == 8` after re-scoring (the ring_dial audit asked for this per model; it
  works today, as verified above).
- m2. The provenance asserts drop `n_steps == 128` and `size == 32`, which RING_DIAL's analysis had. Add those, plus `lr`,
  `batch_size` and `n_batches`. Also bind eval.json to its checkpoint:
  `abs(ev_json["final_loss"] - mean(losses[-45:])) < 1e-9`.
- m3. md5 guard: `lie_groups.py` is imported by `model.py` and missing from both this guard and RING_DIAL's. It is unused in
  `model.py`, so a change could break the import but not change results. Dropping `model_baseline_rope.py` (not on Abs_r4's path) is fine.
- m4. Slots: `DRV_MAXPG=8` counts every one of the user's `mapformer.train_*` processes on cuda:0. At audit time GPU 0 was empty (22 MiB)
  and GPU 1 was at 99% (another user). If another of the user's drivers starts on GPU 0 meanwhile (lock files exist for
  tw_landmark_budget, tinystories and rank_tests_pilot), it waits until this batch falls below that driver's own cap, and
  this batch fills to 8 jobs total. That is not a correctness issue.
- m5. CPU: 8 jobs x (main + 3 data workers) = 32 processes on 32 cores, vs 6 jobs in RING_DIAL. Data generation is
  CPU-bound, so the ~1.5 h estimate (RING_DIAL estimated ~1 h and took 2.9 h) may be optimistic.
