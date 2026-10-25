# Does the torus depth result replicate on fresh seeds? FACTORIAL at n = 8 -- pre-registration (2026-10-24, before any run)

## Question
`FACTORIAL_RESULTS.md` (registered, n = 4, seeds 500-503) found on the 2D torus at T = 1024: path-integrated MapWM at 1
layer 1.000 (4/4 SOLVED) vs index RoPE at 1 layer 0.719, at 4 real layers 0.860, looped x4 0.864 (each 0/4 SOLVED, all
STALLED). Every registered firing was a complete separation at the n = 4 minimum p (0.029). Rule 6: n = 4 is thin, and a
headline needs its seeds on the comparison. **Do FACTORIAL's registered readings hold on four fresh seeds, analysed
alone?** This batch repeats FACTORIAL exactly (same cells, recipe, flags, evaluation, decision rules) on seeds 2800-2803.

## Cells and seeds (one batch, every cell retrained; rule 12)
| cell | variant (`train_variant`) | layers | params | seeds (fresh) | runs |
|---|---|---|---|---|---|
| RoPE_L1 | `RoPE` | 1 | ~204k | 2800, 2801, 2802, 2803 | 4 |
| RoPE_L4 | `RoPE` --n-layers 4 | 4 real layers | 798,741 | 2800, 2801, 2802, 2803 | 4 |
| RoPE_Loop | `RoPELooped` (one shared block x4; run with --n-layers 1, as FACTORIAL) | depth 4, 1 block | 203,925 | 2800-2803 | 4 |
| Path_L1 | `Vanilla_r4` (MapWM, shared rank 4) | 1 | ~204k | 2800-2803 | 4 |
16 runs. Seed-major launch order, deep cells first within a seed (FACTORIAL's order).

**Seeds.** 2800-2803 appear nowhere in the repo (no run dir `*_s28xx`, no driver or pre-registration names them;
checked 2026-10-24). FACTORIAL used 500-503; they are not reused. Pilot seeds 2890-2891 are outside the batch (rule 29).
The training map is GridWorld(seed = run seed); evaluation walks use np seed 1234 + run seed on the held-out map (env seed
10000), so the fresh batch is scored on walks disjoint from FACTORIAL's.

## Recipe and flags (FACTORIAL's, verified from its checkpoints)
`python3 -u -m mapformer.train_variant --variant V --seed S --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16
--n-steps 1024 --n-layers NL --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --data-workers 3 --fast-attn
--device cuda:G --output-dir OUT`, OMP_NUM_THREADS=3, 4 jobs per GPU (DRV_MAXPG=4), as `run_factorial.sh`.
- Verified: the `args` dict stored in FACTORIAL's checkpoints (e.g. `runs/factorial/L4/p0/RoPE_s500/RoPE.pt`) equals the
  pilot checkpoints' `args` on every key except seed, device, output dir, epochs (pilot only) and the two keys added since
  (`weight_decay` 0.05 = train()'s default, `snapshot_every` 0 = off). The analysis ASSERTS this per run against FACTORIAL's
  run of the same cell, so a flag drift stops the analysis.
- `--fast-attn` (SDPA, TF32 off) in every cell, as FACTORIAL; NO `--deterministic`, as FACTORIAL (`deterministic: False`
  in its configs). Per rule 19 / `FAST_ATTN_RANK.md`, --fast-attn training is not bit-identical to the explicit path and its
  backward is run-to-run deterministic only with --deterministic: neither batch is bitwise reproducible, and this batch is
  a fresh-seed replication, never a same-seed determinism check.
- Environment: torch 2.10.0+cu128 (user site), RTX 4090 (kalman GPU 1; FACTORIAL ran on kalman GPU 0, same model).
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set so cuda:1 is nvidia-smi's GPU 1 (FACTORIAL did not set it; it ran on cuda:0).

## Code version and what that means for pooling
FACTORIAL ran at `4041da9` (clean). Of FACTORIAL's 16 md5-guarded files, two changed since: `train.py` (+8 lines: an
`epoch_callback` argument, default None, called after the per-epoch loss is logged; bitwise-verified in
`test_epoch_callback.py`) and `train_variant.py` (+61 lines: new variant classes and imports, `--weight-decay` default
0.05 passed to train(), whose default is 0.05, and `--snapshot-every` default 0 = no callback). Over the whole import
closure of `train_variant` / `data_parallel` / `eval_noise_refine` / `eval_rank_strata` / `analyze_factorial` / `analyze_factorial_n8` (94 modules; 104 guarded files with the driver's extras),
every other module is byte-identical to 4041da9 or new (model_mono_baselines, model_pope_delta, model_rankclock,
model_srope, model_wrap_exist: not used by these cells). The edits are additive and off the default path, but a guarded
training module DID change, so **pooling with FACTORIAL is DESCRIPTIVE ONLY** (labelled cross-batch in every line).

## Evaluation (FACTORIAL's, unchanged)
`eval_noise_refine` (noise 0, held-out map env seed 10000, 100 walks per run, np seed 1234 + s; T = 1024 registered, 512 /
2048 secondary, out of distribution), `eval_rank_strata` at T = 1024 (asserts its 'all' column equals the eval JSON), and
each run on its own training map (`--env-seed <seed>`, T = 1024; FACTORIAL Amendment 1 D3). Outputs
`FACTORIAL_N8_{L1,L4,LOOP}.json/.md`, `FACTORIAL_N8_*_STRATA.json`, `FACTORIAL_N8_*_OWN_s*.json`.

## Registered primary: the fresh batch ALONE (`analyze_factorial_n8.py`; accuracy at T = 1024; n = 4 vs 4)
The decision functions are FACTORIAL's own (`analyze_factorial.c0 / catch_up / helps`), imported unchanged
(`analyze_factorial.py` md5 = the one in `runs/factorial/code_md5.txt`; the selftest checks it). Exact permutation test
(C(8,4) = 70 relabellings) and Fisher on SOLVED (`stats_core.classify_run`: final-5% training loss < 0.05).
- **C0 (sanity, one layer): Path_L1 - RoPE_L1.** PATH WINS AT ONE LAYER: d >= 0.05 and perm p < .05; otherwise AS IT FALLS.
- **C1 (depth): RoPE_L4 vs Path_L1.** INDEX WITHIN 0.02 OF PATH (margin rule, no equivalence test at n = 4): mean
  acc(RoPE_L4) >= mean acc(Path_L1) - 0.02 AND SOLVED(RoPE_L4) >= 3/4. INDEX DOES NOT CATCH UP: Path_L1 - RoPE_L4 >= 0.05
  with perm p < .05, OR Fisher(SOLVED) p < .05 with Path ahead. UNMEASURED otherwise. Every label carries "[within 900
  epochs; <cell> still DESCENDING k/4]" (FACTORIAL D1).
- **C2 (loop): RoPE_Loop vs Path_L1.** The same three labels.
- **C3 (does depth / the loop help index): RoPE_L4 - RoPE_L1 and RoPE_Loop - RoPE_L1.** HELPS: d >= 0.05 and perm p < .05;
  otherwise AS IT FALLS.
- C0-C3 are separate questions; no multiplicity correction is possible or claimed at n = 4 (FACTORIAL D5). With 1 and 4
  layers only, this asks whether 4 layers or a loop x4 suffice, not the exchange rate (D4).
- **Replication summary (a function of the registered labels, not a new test):** REPLICATES iff C1 and C2 both read DOES
  NOT CATCH UP (FACTORIAL's reading); DOES NOT REPLICATE if either reads INDEX WITHIN 0.02 OF PATH; PARTIAL otherwise.
  (Refined before any data by Amendment 1 A3: PARTIAL vs NOT SHOWN AT n = 4, plus a C0 / C3 same-label line.)

**Power (rule 5).** At 4 vs 4 the minimum two-sided p of both the exact permutation test (2/70) and Fisher (4/4 vs 0/4) is
0.029: a label fires ONLY on a complete separation, and a single overlapping seed makes it UNMEASURED / AS IT FALLS. An
UNMEASURED here is not evidence against FACTORIAL. Pooled n = 8 vs 8 would have minimum p 2/12870 = 0.00016, but it is
cross-batch and descriptive (above).

## Declared secondaries (no verdict)
1. **Pooled n = 8** with FACTORIAL's seeds 500-503, cross-batch, DESCRIPTIVE: per cell mean and SOLVED k/8; C0 and C3 d
   with perm p; C1 / C2 through FACTORIAL's branch rule at n = 8, printed as "would read ..." (not a verdict).
2. **Batch shift** per cell: fresh mean - FACTORIAL mean, perm p (4 vs 4). Runs have failed to reproduce across batches
   before (`REFINE_RESULTS.md`, per-seed drift 0.185).
3. **Revisit strata** (T = 1024; FACTORIAL's secondary): short-gap (plain revisit, gap < 128 moves), long-gap (plain, gap
   >= 128), wrap-only (reachable only round the torus); per cell mean over seeds, each beside the retrace floor on that
   stratum.
4. **Floors measured on this batch's own eval walks** (model-free; rule 1). FACTORIAL quoted only RANK_MI's blank 0.507 and
   n-gram 0.576. The retrace predictor (while a run of actions reverses the previous run, step j copies the observation 2j
   steps back; `docs/audits/2026-10-05/score_rank_floors.py`, ported into the analysis and checked against its output:
   seed 10 n 32255 / blank 0.5049 / retrace 0.8411) reaches 0.843 at T = 1024. Each cell's accuracy minus the retrace floor
   ON THE SAME WALKS (paired by seed), and k/4 seeds above it. Best n-gram is quoted (0.576), not re-measured.
5. Own (training) map vs held-out (memorisation; FACTORIAL D3). 6. Accuracy at 512 / 2048 (out of distribution, rule 10).
7. r(final-5% training loss, accuracy at T = 1024) over the 16 fresh runs (rule 2). 8. Run classes per cell.
Not declared: the dropout-scale re-score (FACTORIAL's pre-registration listed it for the one-layer cells but its analysis
did not run it; it would lower the deep and looped cells).

### Floors on the eval walks, measured now (before the batch; `python3 -m mapformer.analyze_factorial_n8 --floors-only`)
| seeds | blank | retrace (all) | retrace short-gap | retrace long-gap | retrace wrap-only | wrap-only share |
|---|---|---|---|---|---|---|
| fresh 2800 / 2801 / 2802 / 2803 | 0.512 / 0.514 / 0.507 / 0.509 | 0.851 / 0.841 / 0.847 / 0.845 (mean 0.846) | 0.909-0.914 | 0.507-0.524 | 0.497-0.527 | 0.068-0.081 |
| FACTORIAL 500 / 501 / 502 / 503 | 0.510 / 0.508 / 0.505 / 0.515 | 0.847 / 0.842 / 0.840 / 0.848 (mean 0.844) | 0.911-0.915 | 0.502-0.535 | 0.507-0.528 | 0.069-0.089 |
Retrace at long gaps and wrap-only is the blank level (it can only copy within a reversal), so ~0.51 there is the floor.

**Noticed while building this (post hoc on FACTORIAL; descriptive, reported here so the reader sees it before the fresh
numbers).** On their own walks, FACTORIAL's index cells sit close to the retrace floor: RoPE_L4 +0.015 overall (3/4 seeds
above; s500 -0.033), RoPE_Loop +0.020 (4/4), RoPE_L1 -0.125; by stratum, RoPE_L4 short-gap 0.943 vs retrace 0.912
(+0.031), long-gap 0.458 and wrap-only 0.457 vs ~0.515 (below the blank level); RoPE_Loop +0.026 / -0.008 / -0.007.
Path_L1 +0.156 (1.000 on every stratum). So "depth buys the short-range part of the map" is, in floor terms, "depth buys
about 0.03 over a map-free retrace rule on short gaps, and nothing at long range". This does not change any FACTORIAL
verdict (all are path vs index); it is why secondary 4 is declared.

## Predictions (stated before the batch)
- C0 PATH WINS AT ONE LAYER: Path_L1 1.000 (4/4 SOLVED) vs RoPE_L1 0.69-0.75.
- C1 and C2: INDEX DOES NOT CATCH UP, both routes (perm and Fisher, 4/4 vs 0/4); RoPE_L4 and RoPE_Loop 0.81-0.89, 0/4
  SOLVED, STALLED (none DESCENDING). Replication summary: REPLICATES.
- C3 HELPS for both (+0.10 to +0.17). Risk stated: FACTORIAL's RoPE_L4 minimum (0.814) cleared RoPE_L1's maximum (0.739)
  by 0.075; one low 4-layer seed or one high 1-layer seed would make C3 AS IT FALLS at n = 4.
- Secondaries: index short-gap 0.90-0.96 (within 0.05 of the retrace floor's 0.91); long-gap and wrap-only 0.37-0.53, at or
  below the blank level; RoPE_L4 and RoPE_Loop overall within +/-0.05 of the retrace floor; batch shift within +/-0.03 per
  cell; r(loss, acc) strongly negative (FACTORIAL: -0.94).

## Pilot (timing / memory only; seeds 2890-2891, outside the batch; `run_factorial_n8_pilot.sh`, `runs/factorial_n8_pilot/`)
Run 2026-10-24 22:54-22:58 on kalman GPU 1 at `f4ac860` (GPU 0 held by another builder's pilot).
- Solo, slowest cell (RoPE --n-layers 4, s2890, 10 epochs): 2.9 s/epoch, 1,724 MiB, 48 s wall including start-up.
- Contended, the four cells at once (s2891, 15 epochs; the batch's 4-per-GPU packing): RoPE_L4 and RoPE_Loop 6.2 s/epoch
  (9.1 s in the first 5 epochs while all four started), RoPE_L1 3.2, Vanilla_r4 3.0-4.0; memory 1,716-1,724 MiB per deep
  job, 914-994 MiB per one-layer job, 5,348 MiB total. Wall to the deep checkpoints 138 s. (The pilot script's bare `wait`
  also waited on its memory-monitor loop, which was killed by hand; its logged "contended wall 217 s" is inflated. Fixed in
  the script; the per-epoch times above are the measurement.)
- Flags: the pilot checkpoints' `args` equal FACTORIAL's (see Recipe).

## Cost
FACTORIAL's own wall times on this GPU type (kalman GPU 0, RTX 4090, 4 jobs at a time, `factorial.log` and checkpoint
mtimes): RoPE_L4 and RoPE_Loop 173-186 min each, RoPE_L1 49 min, Vanilla_r4 58-61 min; whole batch 20:38 -> 04:45 =
**8 h 07 min** including evaluation (~15 min). Its deep cells ran 12.6 s/epoch when four deep jobs shared the GPU vs 6.2
with two (pilot) and 2.9 alone: the GPU is saturated and packing mostly reorders work. Estimate for this batch: **~8 h on
GPU 1 alone** (16 runs; ~8 GPU-hours), plus the CPU analysis (~1 min). Nothing runs on GPU 0.

## Process
- Independent code check (rule 29) before launch, blind to results; its findings go into an amendment here before launch.
- Before reading any run: `python3 -m mapformer.experiment_audit --runs-dir runs/factorial_n8/L4` (and L1, LOOP); exit 2
  is expected on this layout (Amendment 1 A4).
- Driver `run_factorial_n8.sh` (sources `lib_driver.sh`): flock single instance; refuses on modified tracked files or an
  uncommitted guarded file; md5 guard over the import closure + `ckpt_guard.py`, `lib_driver.sh`, the driver, THIS FILE and
  FACTORIAL's six eval/strata JSONs, re-checked before every launch, before evaluation and before the analysis; DRV_GPUS and
  DRV_MAXPG required in the environment; artifacts before the `.factorial_n8_done` marker. Dry run:
  `DRV_DRYRUN=1 DRV_GPUS=1 DRV_MAXPG=4 bash run_factorial_n8.sh` (own run dir and log).
- Launch (after commit and the check's amendment):
  `cd /home/prashr/mapformer && DRV_GPUS=1 DRV_MAXPG=4 setsid nohup bash run_factorial_n8.sh > /dev/null 2>&1 &`

## What this can and cannot show
It can say whether FACTORIAL's four readings recur on four new seeds and new evaluation walks under the same code path;
pooled, it describes n = 8. It cannot: separate anything a single overlapping seed hides (n = 4, complete separations only);
test more than 4 layers or another budget (900 epochs; FACTORIAL's index runs were all STALLED, not descending); test
equivalence; speak to other tasks, widths or heads. Pooled numbers are cross-batch and code-version-adjacent, descriptive.

## Amendment 1 (2026-10-25, after the independent check `docs/audits/2026-10-25/factorial_n8_audit.md`; before launch, no data)
The check found no blocker (1 MAJOR operational, 5 MINOR, 13 NOTE). It ran the selftest (ALL PASS), loaded FACTORIAL's 16
checkpoints through `load_batch` (every assertion passes), confirmed the pilot checkpoints' args equal FACTORIAL's except
epochs, confirmed 86 of the 94 guarded modules byte-identical to 4041da9 (6 new, unused by these cells; train.py /
train_variant.py changed off the default path), and dry-ran the driver in a scratch clone (16 launches in FACTORIAL's
order, refusals on a missing DRV_GPUS and on uncommitted guarded files). Nothing affects the arms, flags, evaluation or the
registered decision rules. Changes:
- **A1 (code freeze, M1).** From launch to `.factorial_n8_done` the 104 md5-guarded files (the import closure of
  train_variant / data_parallel / eval_noise_refine / eval_rank_strata / analyze_factorial / analyze_factorial_n8,
  ckpt_guard.py, lib_driver.sh, run_factorial_n8.sh, this file, FACTORIAL's six eval / strata JSONs) are FROZEN for every
  session on kalman: no edit, move, checkout or pull that touches them. G-3's own files (model_ssm_sign.py,
  train_ssm_sign.py, analyze_ssm_sign.py, run_ssm_sign.sh, G3_SSM_PREREG.md) are not in this closure. If the guard aborts
  (per launch, before eval or before the analysis), the offending file is restored to the launch commit
  (`git checkout <launch-sha> -- <file>`) so the md5 matches again, and the driver is re-run (it skips finished and
  running runs). `runs/factorial_n8/code_md5.txt` is NEVER deleted once a checkpoint exists or a job is running. Both this
  batch's and G-3's files are committed together before either launches, so the tree is clean at launch (the driver
  refuses otherwise).
- **A2 (GPU plan, m2).** FACTORIAL_N8 runs on GPU 1 only (`DRV_GPUS=1 DRV_MAXPG=4`); G-3 A2 runs on GPU 0 only (its driver
  defaults to GPU 1, so it is launched with `DRV_GPUS=0`). `lib_driver.sh`'s slot counter counts the user's
  `mapformer.train_*` processes per device (by `--device cuda:N` in argv), so the two batches do not take each other's
  slots. Wall estimate unchanged: ~8 h (FACTORIAL's 8 h 07 min on one 4090 at 4 jobs). They share the 32 CPU cores
  (4 x (3 data workers + OMP 3) here); the device and the packing have no numerical effect.
- **A3 (replication summary, m1).** Renamed "REPLICATION SUMMARY (catch-up readings C1, C2)": REPLICATES iff C1 and C2 both
  read DOES NOT CATCH UP; DOES NOT REPLICATE if either reads INDEX WITHIN 0.02 OF PATH; PARTIAL iff exactly one reads DOES
  NOT CATCH UP (and neither WITHIN); otherwise NOT SHOWN AT n = 4. Added one descriptive line, "same label as FACTORIAL":
  whether C0 (PATH WINS AT ONE LAYER), C3 RoPE_L4 (HELPS) and C3 RoPE_Loop (HELPS) carry FACTORIAL's label. Both remain pure
  functions of the registered labels, matched on the label head (text before the first ':'), not on a substring anywhere.
- **A4 (procedure, m3 / m4).** No stale md5 record: before the post-amendment dry run `runs/factorial_n8_dryrun/` does not
  exist (it was removed after the builder's dry run and the post-amendment dry run ran in a scratch clone), and at launch
  `runs/factorial_n8/` does not exist (checked 2026-10-25). `experiment_audit` finds no (json, pt) pairs on this layout,
  runs only its check [1] (convergence on the checkpoints' loss curves) and exits 2 by design: read check [1]; the
  registered run classes are `stats_core.classify_run` in the analysis.
- **A5 (selftest, m5).** Added: absence checks (each case asserts the registered lines do NOT contain the other labels);
  the cases "C1 and C2 both UNMEASURED" (NOT SHOWN AT n = 4), "C1 DOES NOT CATCH UP + C2 WITHIN" (DOES NOT REPLICATE) and
  "DOES NOT CATCH UP by the permutation route alone" (index 1/4 SOLVED, Fisher p 0.14); a check that every registered label
  is reached across the cases; the pooled check in the "within" case is now one exact expected label; the pooled rule's
  UNMEASURED branch; flag drift by a changed value, an added key and a removed key.
- **Wording (notes 3, 12).** The retrace rule copies the observation 2j STEPS back (4j interleaved tokens); the guarded
  closure is 94 modules (104 files).
Selftest after A3 / A5: SELFTEST ALL PASS (87 checks; also from a scratch clone importing its own copy). Post-amendment
dry run in a scratch clone at 92002e0 + these four files (`DRV_DRYRUN=1 DRV_GPUS=1 DRV_MAXPG=4`): exit 0, md5 guard over
104 files, 16 DRYRUN launches in FACTORIAL's order on cuda:1, 18 DRYRUN evals (3 roots x (eval + strata + 4 own-map); the
audit's '21' counted differently), DRYRUN analysis; only warning 'a guarded file is not committed'. Without DRV_GPUS:
REFUSED, exit 1. test_guards 13/13. No `runs/factorial_n8*` directory or `code_md5.txt` exists in the real repo.
