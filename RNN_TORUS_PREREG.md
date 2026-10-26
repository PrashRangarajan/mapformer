# Does a recurrent network learn the long-walk torus map? LSTM / GRU baselines -- pre-registration (DRAFT, 2026-10-26, before any batch run)

Status: registered text + Amendment 1 (after the independent check, `docs/audits/2026-10-26/rnn_torus_audit.md`; at the end
of this file, superseding the body where they differ). Not committed, not launched. A pilot (seed 3390, outside the batch)
ran on kalman GPUs for timing, memory and the learning rate (section Pilot; finished 2026-10-26 03:01, rates picked by the
pre-written rule). Nothing from the batch seeds has been run.
Declared blocks: A (the only block).

## Question
The headline torus result: at T = 1024 one path-integrated attention layer (MapWM, shared rank 4) learns the 64 x 64 map
(FACTORIAL + FACTORIAL_N8, n = 8: 0.999, 8/8 SOLVED), while index RoPE does not at 1 layer (0.724), 4 layers (0.845) or
looped x4 (0.862), each 0/8, with 4 index layers at the map-free retrace floor. A reviewer's first attack
(`docs/audits/2026-10-26/CRITICAL_REVIEW.md` B-M9 and A-M2; `docs/paper/BIG_PICTURE_2026-10-24.md` A5): **there is no
recurrent baseline, and an RNN given the actions integrates position by construction.** Integration is the easy half; an
RNN must also store a freshly drawn map (on the held-out map, ~700 distinct cells per 1,024-step walk, about half of them
holding one of 16 objects) in a fixed-size state and read it back at revisits up to ~1,000 steps later. **Does an LSTM or
GRU trained on the same stream, with the same loss, recipe and budget, learn the map -- at matched parameters (~205k, one
and two layers) and at 4x parameters (~800k)?** If one does, "one attention layer" is the wrong unit and the claim becomes
"a map needs an integrator, and path attention is one". If none does, the one-layer result sharpens: the path model beats
the standard recurrent integrator as well as index attention.

## Task, data, loss (unchanged; FACTORIAL / FACTORIAL_N8)
`environment.GridWorld(64, 16 object types, p_empty 0.5, seed = run seed)` as `train_variant` builds it: one training map
per run, fresh walks every batch (directed walks of 1-10 repeated moves), interleaved `[a1, o1, a2, o2, ...]`, vocabulary 21
(4 actions, 16 objects, blank), loss on revisited observation positions only (`train.train`, unchanged). The RNN arms run
through `train_rnn`, which registers them into `train_variant.VARIANT_MAP` and calls `train_variant.main()`, so the
environment, the data stream (`--data-workers 3`, `ParallelBatchGenerator` seeded by the run seed), the loss, the optimiser,
the schedule and the checkpoint are the anchors' code path. At a seed every arm sees the same training map and the same
walks. Evaluation on the held-out map (env seed 10000), walks from np seed 1234 + run seed (FACTORIAL's protocol).

## Arms (one batch, every cell trained; rule 12)
| cell | variant | layers | hidden | params | state floats | s / epoch solo (4090) | how |
|---|---|---|---|---|---|---|---|
| Path_L1 (anchor) | `Vanilla_r4` (MapWM, shared rank 4) | 1 | d 128 | 204,757 | -- | 0.9 | `train_variant --fast-attn` |
| RoPE_L1 (anchor) | `RoPE` | 1 | d 128 | 203,925 | -- | 0.8 | `train_variant --fast-attn` |
| LSTM 1L small | `LSTM_L1_h157` | 1 | 157 | 205,377 (+0.3%) | 314 | 2.9 | `train_rnn` |
| GRU 1L small | `GRU_L1_h181` | 1 | 181 | 205,637 (+0.4%) | 181 | 2.8 | `train_rnn` |
| LSTM 2L small | `LSTM_L2_h111` | 2 | 111 | 203,817 (-0.5%) | 444 | 1.6 | `train_rnn` |
| GRU 2L small | `GRU_L2_h128` | 2 | 128 | 203,797 (-0.5%) | 256 | 0.7 | `train_rnn` |
| LSTM 1L large | `LSTM_L1_h313` | 1 | 313 | 800,049 (+0.2% of RoPE 4L's 798,741) | 626 | 4.0 | `train_rnn` |

- **Model** (`model_rnn.py`): token embedding (dim = hidden) -> `nn.LSTM` / `nn.GRU` (batch_first; inter-layer dropout 0.1
  at 2 layers, none at 1) -> LayerNorm -> linear to the vocabulary: the repo's existing `LSTM` variant
  (`model_baselines_extra.LSTMBaseline`, fixed h = 128, 137,749 params) with the hidden size set per arm. Hidden sizes are
  the integer whose closed-form count is nearest the target (`model_rnn.match_hidden`; tested against the module). LSTM
  forget-gate bias starts at 1 (b_ih forget slice 1, b_hh 0; Jozefowicz et al. 2015), the standard strengthening for long
  memory; GRU keeps PyTorch's default init. Causal by construction (tested, with a bidirectional negative control).
- **Why these arms.** 1 and 2 layers at matched parameters for both cell types (the reviewer's "LSTM/GRU"); one 4x arm at
  RoPE 4L's count, so capacity is matched to the deepest index model. The large arm is the 1-layer LSTM: the canonical
  form, the larger state per layer, and the faster of the two LSTM shapes (4.0 vs 4.7 s / epoch). The 4x GRU and the 4x
  2-layer arms (`GRU_L1_h361`, `LSTM_L2_h222`, `GRU_L2_h256`, all built and smoke-tested) are NOT run (cost, below).
- **Not in the batch, and why.** No other index cells (FACTORIAL_N8 has them; not needed for this question, and not pooled:
  rule 12). No "GPS-PE" control (index attention given cumsum(d_t) as an absolute code; review A-M2): a different question
  (is it the position code or the integration?), noted for later.

## Seeds
Batch: 3300-3307 (n = 8 per cell; 64 runs after Amendment 1 A3). No run dir `*_s33xx`, no driver and no pre-registration names them (grep of
runs/, *.md, *.sh, *.py on 2026-10-26: the only hits are unrelated numbers such as loss values; `test_rnn_torus.py`
re-checks runs/). Pilot: 3390 (outside the batch, rule 29). The concurrent exp/softplus torus batch (another builder) must
not use 3300-3399 -- to be confirmed by the checker.

## Recipe
Anchors, FACTORIAL_N8's flags exactly: `python3 -u -m mapformer.train_variant --variant V --seed S --epochs 900 --lr 1e-3
--n-batches 98 --batch-size 16 --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine
--data-workers 3 --fast-attn --device cuda:G --output-dir OUT` (AdamW wd 0.05, 5% warm-up + cosine to 10%, grad clip 1.0).
RNN arms: `python3 -u -m mapformer.train_rnn --variant V --seed S --lr LR <the same flags without --fast-attn> --n-layers NL`.
Differences, all forced in `train_rnn.py`: no `--fast-attn` (train_variant refuses it for a model without attention); no
`--deterministic` (as the anchors); cuDNN TF32 OFF (`torch.backends.cudnn.allow_tf32` defaults to True and would run the
recurrent matmuls in TF32; the anchors' `--fast-attn` turns both TF32 switches off, so every arm is fp32); `--init-from`
refused. `--d-model 128 --n-heads 2` are passed and IGNORED by the RNN classes (the hidden size is fixed by the variant
name; the checkpoint's config records d_model 128). The analysis asserts per run: variant, seed, 900 losses, layers, length
1024, grid 64, lr = the cell's, fast_attn = the cell's, wd 0.05, no init_from, and that every run's remaining CLI arguments
equal one shared recipe equal to FACTORIAL_N8's path run (`runs/factorial_n8/L1/p0/Vanilla_r4_s2800/Vanilla_r4.pt`).
Environment: torch 2.10.0+cu128 (user site), cuDNN 9.10.2, RTX 4090 x 2 (kalman). Neither the anchors (SDPA backward
without --deterministic) nor the RNNs are bitwise reproducible; this is a fresh-seed batch, never a determinism check.

**Learning rate.** Anchors 1e-3 (FACTORIAL's). RNN arms: each at the rate picked by the pre-written rule from the pilot
(section Pilot): **LSTM_L1_h313 1e-3, LSTM_L1_h157 3e-3, GRU_L1_h181 1e-3, LSTM_L2_h111 3e-3, GRU_L2_h128 3e-3**. Run
roots: `runs/rnn_torus/anchor`, `runs/rnn_torus/lr1e-03` (LSTM_L1_h313, GRU_L1_h181), `runs/rnn_torus/lr3e-03` (the other
three). Cells in the analysis are named `<variant>@<rate>` (e.g. `LSTM_L1_h157@0.003`).

## Floors on this batch's evaluation walks (measured now, model-free; `python3 -m mapformer.analyze_rnn_torus --floors-only`)
| seed | 3300 | 3301 | 3302 | 3303 | 3304 | 3305 | 3306 | 3307 | mean |
|---|---|---|---|---|---|---|---|---|---|
| blank | 0.512 | 0.502 | 0.503 | 0.508 | 0.508 | 0.507 | 0.515 | 0.512 | 0.508 |
| retrace (all) | 0.844 | 0.842 | 0.840 | 0.850 | 0.849 | 0.841 | 0.840 | 0.846 | 0.844 |
| retrace short-gap | 0.910 | 0.911 | 0.909 | 0.911 | 0.913 | 0.906 | 0.912 | 0.911 | 0.910 |
| retrace long-gap / wrap-only | 0.510 / 0.502 | 0.494 / 0.504 | 0.479 / 0.515 | 0.505 / 0.521 | 0.511 / 0.512 | 0.498 / 0.518 | 0.488 / 0.516 | 0.500 / 0.535 | 0.498 / 0.515 |
| wrap-only share | 0.075 | 0.086 | 0.074 | 0.075 | 0.076 | 0.074 | 0.097 | 0.081 | 0.080 |
Retrace (`analyze_factorial_n8.floors_walk`, FACTORIAL_N8's port of `score_rank_floors.py`): while a run of actions reverses
the previous run, step j copies the observation 2j steps back. At long gaps and wrap-only revisits it is the blank level.
Best action-stream n-gram 0.576 (quoted, RANK_MI; not re-measured). ~320 scored revisits per walk, ~32,000 per seed.

## Gates (rule 11)
- Task: FACTORIAL's, gated there (n-gram, chance, vocabulary). Token ids: the RNN embedding has exactly the 21-token
  vocabulary (`test_rnn_torus`: GridWorld's unified vocabulary = 21 = model_rnn.VOCAB).
- **Context destruction on every trained model (S8, run by the driver before the analysis).** On 50 of each run's
  evaluation walks the model reads the same actions with the observations of ANOTHER map (GridWorld seed 20000; same
  walk, same revisit mask, asserted) and is scored on the true map's targets. A model that reads its context drops to the
  agreement rate of two maps (~0.27); a context-free predictor stays at the blank level (~0.51). Label per cell: CONTEXT
  USED if destroyed accuracy <= blank floor - 0.10, else CONTEXT NOT SHOWN. Validated before the batch on two
  constructed models (`test_rnn_torus`): an in-context perfect-memory oracle 1.000 intact -> < 0.4 destroyed; an
  always-blank model unchanged (0.4-0.6 both). The intact walks are eval_noise_refine's walks (tested: same accuracy).
  Not a verdict on the primary; a cell reading CONTEXT NOT SHOWN is reported as such beside its accuracy.

## Registered primary (`analyze_rnn_torus.primary`; accuracy at T = 1024, held-out map; n = 8 vs 8)
Run classes `stats_core.classify_run` (SOLVED: final-5% training loss < 0.05; STALLED; DESCENDING). Contrasts use the
exact two-sample permutation test (C(16, 8) = 12,870 relabellings) on accuracy and Fisher's exact test on SOLVED.
beats(c) = (acc(path) - acc(c) >= 0.05 and perm p < .05) or (Fisher p < .05 with path ahead) -- FACTORIAL's "does not
catch up" rule at n = 8.
- **UNREADABLE** if Path_L1 SOLVED < 6/8 (it was 8/8 in FACTORIAL and FACTORIAL_N8; the anchor did not reproduce).
- **PATH WINS** if beats(c) for EVERY RNN cell (6/6 after Amendment 1 A3). An intersection-union test: level .05 with no correction, and
  selection-proof -- whichever RNN cell is best, path beats it. The binding cell is named.
- **RNN WINS** else if some RNN cell is ahead of path: (acc(c) - acc(path) >= 0.05 and Holm-adjusted perm p < .05) or
  (Holm-adjusted Fisher p < .05, c ahead); Holm over the RNN cells (6 after Amendment 1 A3), per route.
- **RNN MATCHES** else if the SELECTED best RNN cell has acc >= acc(path) - 0.02 and SOLVED >= 6/8. Selection uses
  training loss only, never held-out accuracy: most SOLVED, then lowest mean final-5% training loss, then fewer
  parameters, then name.
- **UNMEASURED** otherwise (names the cells path does not separate from).
Every label carries "[within 900 epochs; DESCENDING k/8 in <cell>]" for the selected cell and every cell path does not
beat (budget-scoped, rule 4).
**Boundaries against the noise floor.** Path's between-seed sd at this recipe is ~0.002 (FACTORIAL_N8 0.996-1.000); the
exact-t two-sample MDE at n = 8 (multiplier 3.01) is 0.021 / 0.032 / 0.053 for an RNN sd of 0.02 / 0.03 / 0.05, so the
0.05 margin sits at or above the MDE for RNN sd <= 0.047, and the 0.02 match margin is 10x path's sd. Fisher: path 8/8
separates from an RNN cell at k <= 3/8 (p 0.026; 4/8 p 0.077). The minimum permutation p is 0.00016. Given those, a
PATH WINS cannot come from noise in the path arm; an RNN arm that solves 4-5/8 with a mean within 0.05 reads UNMEASURED.

## Declared secondaries (no verdict)
- **S1, RNN vs index RoPE 1L** (per RNN cell; the selected cell flagged): d, perm p, Fisher. RNN ABOVE INDEX 1L if
  d >= 0.05 and perm p < .05; RNN BELOW INDEX 1L if d <= -0.05 and perm p < .05; AS IT FALLS otherwise. No correction.
- **S2, accuracy minus the retrace floor on the same walks**, paired by seed, k/8 above (the floor beside every number).
- **S3, revisit strata** (T = 1024; `eval_rank_strata`, which asserts its 'all' equals the eval JSON): short-gap (gap < 128
  moves), long-gap (>= 128), wrap-only (reachable only round the torus), each beside the retrace floor of the stratum.
  This is where an RNN's state limit should show: a recency-limited memory scores retrace-like short-gap returns and the
  blank level on long-gap and wrap-only revisits.
- **S4, T = 512 and 2048** (out of distribution, rule 10; an RNN has no position extrapolation problem, so a T = 2048 drop
  is a memory-length effect).
- **S5, own (training) map vs held-out** (does the RNN memorise its training map in its weights?).
- **S6, r(final-5% training loss, accuracy at T = 1024)** over all runs (64 after Amendment 1 A3; rule 2).
- **S7, run classes** per cell (SOLVED / STALLED / DESCENDING).
- **S8, context destruction** (gate above).
- **S9, capacity**: LSTM_L1_h313 (1e-3) - LSTM_L1_h157 (3e-3) (perm p, SOLVED); the two run at different pilot-picked
  rates, so this is capacity at each arm's chosen rate, not at one rate.
- **Matched-parameter reading (registered companion)**: the primary's rule over the four ~205k cells only (printed as
  "REGISTERED MATCHED-PARAMETER READING"); it carries the one-layer claim if the 4x arm alone closes the gap.
Not declared: per-layer probes, the dropout re-score (RNN dropout is between layers only, off at eval), any pooled
cross-batch number (FACTORIAL_N8's cells are quoted beside, labelled cross-batch, never pooled).

## Predictions (stated before the batch; PILOT-INFORMED -- see Pilot)
**PILOT-INFORMED, NOT BLIND.** Every RNN prediction below was written after seeing one pilot seed's training loss to epoch
300 (Pilot table). The anchors' predictions come from FACTORIAL / FACTORIAL_N8, not from the pilot. Training loss here is
on the run's own training map, so it can overstate held-out accuracy if an RNN memorises that map in its weights (S5).
- Anchors: Path_L1 0.999-1.000, 8/8 SOLVED; RoPE_L1 0.70-0.75, 0/8, STALLED (as FACTORIAL_N8).
- No RNN cell at ~205k parameters SOLVED within 900 epochs (0/8 each): their pilot losses were 0.30-0.33 at epoch 300
  and falling slowly (0.32-0.36 at epoch 200), far above the 0.05 SOLVED threshold. Held-out accuracy 0.84-0.92 for
  each: above RoPE_L1, at or above the retrace floor (0.844), below path by >= 0.05. Most runs STALLED or DESCENDING
  slowly at 900 epochs.
- LSTM_L1_h313 (4x): lowest RNN loss (0.158 at epoch 300 vs path's 0.06-0.12 and RoPE 4L's 0.57-0.61 at the same epoch
  in FACTORIAL_N8); predicted 0-3/8 SOLVED, accuracy 0.90-0.97; the selected best RNN cell.
- LSTM_L1_h313 at 3e-3 (Amendment 1 A3): as at 1e-3 (pilot 0.165 vs 0.160 at epochs 275-300): 0-3/8 SOLVED,
  accuracy 0.90-0.97; which of the two 4x cells is selected best is not predicted.
- Primary: **PATH WINS** (binding cell one of the two LSTM_L1_h313 cells; via Fisher if it solves <= 3/8, via accuracy
  otherwise -- if via Fisher only with d < 0.05 the label carries Amendment 1 A2's qualifier). Matched-parameter reading:
  PATH WINS.
- S1: RNN ABOVE INDEX 1L for every RNN cell (+0.10 to +0.25). S2: small cells within +/-0.07 of retrace, the large cell
  above it. S3: short-gap revisits near ceiling (>= 0.95); long-gap and wrap-only revisits clearly below path (0.6-0.9),
  above the blank level -- a recency-limited memory. S4: T = 2048 below T = 1024 for every RNN cell. S5: own map >= held-out
  by 0.00-0.05. S6: r(loss, acc) strongly negative. S8: CONTEXT USED for every cell, anchors included. S9: large minus
  small LSTM +0.03 to +0.08.
- Stated risk: the large LSTM was still descending at epoch 300 (0.186 -> 0.158 over epochs 200-300) under a cosine
  schedule with 600 epochs left. If it solves 4-7/8 with accuracy within 0.05 of path the primary reads UNMEASURED (or
  RNN MATCHES at >= 6/8 and within 0.02), and the matched-parameter reading carries the one-layer claim.

## Power (rule 5)
n = 8 vs 8: exact permutation minimum p 0.00016; Fisher 8/8 vs 3/8 p 0.026, vs 4/8 p 0.077. The primary's IUT has the
power of its hardest comparison (path vs the best RNN cell). An RNN cell that solves 4-6/8 with accuracy within 0.05 of
path makes the primary UNMEASURED (not "RNN MATCHES", which needs >= 6/8 AND within 0.02), and that UNMEASURED is not
evidence for either side.

## Pilot (seed 3390, outside the batch; `run_rnn_torus_pilot.sh`, `runs/rnn_torus_pilot/`)
Run 2026-10-26 01:31-03:01 on kalman GPUs 0-1 (no other user), code at 9db4899 plus this batch's new files (uncommitted).
**Timing and memory** (`runs/rnn_torus_pilot/timing/`, 5-10-epoch runs and a cuDNN microbenchmark, fp32, TF32 off):
- Solo s / epoch (98 batches of 16 x 2,047 tokens): LSTM_L1_h157 2.9, GRU_L1_h181 2.8-2.9, LSTM_L2_h111 1.6, GRU_L2_h128
  0.7, LSTM_L1_h313 3.8-4.0 (4x GRU 1L 3.9, 4x 2-layer shapes 4.7-5.0); anchors 0.9 (Vanilla_r4) and 0.8 (RoPE), both
  with --fast-attn. Peak memory 1.3-1.4 GB per RNN job.
- **Packing gives no throughput.** 2 / 4 / 6 concurrent LSTM_L1_h157 processes on one 4090 ran 60 / 121 / 179 ms per
  training step vs 29 ms alone: the GPU time-slices between processes, so the RNN cost adds up serially (the cost below is
  serial GPU time divided by the number of GPUs, independent of DRV_MAXPG).
- cuDNN speed cliff: hidden <= 128 runs 3-8 ms / step, hidden 129-160 26-29 ms (LSTM 1L); the arms keep their matched
  sizes (the cliff is engineering, not a reason to change the comparison), noted because GRU_L2_h128 sits below it.
**Learning rate** (`runs/rnn_torus_pilot/lr/`): every run uses the batch recipe and its 900-epoch cosine schedule and is
stopped by PID at epoch 300 (the logged loss at 275-300 is the readout; a few were stopped up to 20 epochs late, outside
the window). Pick rule, written into `analyze_rnn_torus.pilot_pick` before any readout: lowest mean logged training loss
over epochs 275-300; a rate is excluded as UNSTABLE if after epoch 50 its loss rises > 0.5 above its running minimum or is
non-finite; a rate within 0.01 of the lowest that is closer to 1e-3 wins. Readout (`python3 -m mapformer.analyze_rnn_torus
--pilot`); no run UNSTABLE, none exited early:
| arm | 3e-4 | 1e-3 | 3e-3 | 1e-2 | pick |
|---|---|---|---|---|---|
| LSTM_L1_h313 | -- | **0.160** | 0.165 | -- | 1e-3 |
| LSTM_L1_h157 | 0.634 | 0.334 | **0.303** | 0.332 | 3e-3 |
| GRU_L1_h181 | -- | 0.329 | 0.327 | -- | 1e-3 (tie within 0.01) |
| LSTM_L2_h111 | -- | 0.329 | **0.308** | -- | 3e-3 |
| GRU_L2_h128 | 0.631 | 0.372 | **0.347** | 0.388 | 3e-3 |
Grid: {1e-3, 3e-3} for every arm, bracketed with 3e-4 and 1e-2 on the two cheapest-to-read arms (LSTM_L1_h157,
GRU_L2_h128), whose optima are interior (3e-3). The rate differences at 275-300 are small (0.002-0.03) except 3e-4; one
seed and 300 epochs cannot rank 1e-3 vs 3e-3 reliably, which is why the batch does not claim the RNNs are at their best
rate, only at a pilot-chosen one. For reference at epoch 300 (FACTORIAL_N8 logs, seeds 2800-2803): path 0.06-0.12, RoPE
1L 1.00-1.05, RoPE 4L 0.57-0.61 (final 0.25-0.46).
**Surprise:** every RNN arm reaches a far lower training loss than index RoPE at 4 layers by epoch 300 (0.30-0.33 small,
0.16 large vs 0.57-0.61), i.e. the recurrent baseline is much stronger than the index baselines on this task; whether it
reaches path's regime (final loss 0.001-0.005) is the open question.
A full-length pilot (`run_rnn_torus_pilot2.sh`, 900 epochs on seed 3391 then the whole eval path on GPU) was written and
NOT run (GPUs go to EXPTORUS); the GPU eval path was exercised only on CPU (`test_rnn_torus` smoke).

## Cost
Serial GPU time (packing gives no throughput, Pilot), 900 epochs, n = 8 per cell, solo rates from the pilot:
| cell | min / run | runs | GPU-h |
|---|---|---|---|
| Path_L1, RoPE_L1 (anchors) | 13.5 / 12 | 16 | 3.4 |
| LSTM_L1_h313 @ 1e-3 | 59 | 8 | 7.9 |
| LSTM_L1_h313 @ 3e-3 (Amendment 1 A3) | 59 | 8 | 7.9 |
| LSTM_L1_h157 | 44 | 8 | 5.8 |
| GRU_L1_h181 | 43 | 8 | 5.8 |
| LSTM_L2_h111 | 24 | 8 | 3.2 |
| GRU_L2_h128 | 10.5 | 8 | 1.4 |
| total training | | 64 | ~35.4 |
Evaluation (T = 512 / 1024 / 2048 x 100 walks, strata, 8 own-map evals per root, context destruction: B = 1 forwards)
~0.5 h on one GPU. **Wall ~18 h on both kalman GPUs** (DRV_GPUS="0 1" DRV_MAXPG=2; 2 jobs per GPU keeps each GPU busy while
one job waits on data; more per GPU only time-slices). Trimming (drop GRU_L1_h181 and LSTM_L2_h111: ~18.5 GPU-h, ~9.5 h)
was considered and rejected: it would remove the 1-vs-2-layer and LSTM-vs-GRU coverage the reviewer's question asks for.
**Ordering on kalman:** the EXPTORUS batch (`EXPTORUS_PREREG.md`, ~9.8 h on both GPUs) runs FIRST; RNN_TORUS launches
after `.exptorus_done` (or after EXPTORUS's trainers have all exited), not beside it -- lib_driver's slot count is shared
across drivers, so concurrent batches would stretch both. Together ~28 h of both GPUs. Enforced by the driver
(Amendment 1 A1).

## Process
- Independent code check (rule 29), blind to results; its findings go into an amendment here before launch.
- Code freeze from launch to `.rnn_torus_done`: the md5-guarded files (the import closure of train_variant, train_rnn,
  model_rnn, data_parallel, eval_rnn_torus, eval_noise_refine, eval_rank_strata, analyze_rnn_torus, analyze_factorial_n8,
  plus ckpt_guard.py, lib_driver.sh, run_rnn_torus.sh and THIS FILE) are not edited, moved or checked out by any session.
  Note for the other builders in this tree: `train_variant.py`, `train.py`, `model.py`, `environment.py`,
  `data_parallel.py`, `stats_core.py`, `analyze_lrsweep.py` and `analyze_factorial_n8.py` are in the closure.
- Driver `run_rnn_torus.sh` (sources `lib_driver.sh`): flock single instance; refuses on modified tracked files and on an
  uncommitted guarded file; md5 guard re-checked before every launch, before evaluation and before the analysis; DRV_GPUS
  and DRV_MAXPG required; seed-major launch, longest cell first within a seed; then per root `eval_rnn_torus noise_refine`
  (T = 512 / 1024 / 2048, 100 walks) and `strata`, own-map evals, context destruction, the analysis; artifacts before the
  `.rnn_torus_done` marker. Outputs `RNN_TORUS_{anchor,lr1e-03,...}.json/.md`, `..._STRATA.json`, `..._OWN_s*.json`,
  `RNN_TORUS_CD.json`, `RNN_TORUS_ANALYSIS.txt`, `RNN_TORUS_VERDICTS.json`.
- Before reading any run: `python3 -m mapformer.experiment_audit --runs-dir runs/rnn_torus/<root>` (check [1]: convergence
  on the loss curves; exit 2 expected on this layout, as FACTORIAL_N8 A4).
- Dry run: `cd /home/prashr/mapformer && DRV_DRYRUN=1 DRV_GPUS="0 1" DRV_MAXPG=2 bash run_rnn_torus.sh` (since Amendment 1
  A1 the dry run does not wait on real GPU slots and turns the sequencing refusals into logged warnings, so it completes
  while another batch trains; it writes `runs/rnn_torus_dryrun/` and `rnn_torus_dryrun.log`, to be deleted afterwards).
- Launch (after commit and the check's amendment):
  `cd /home/prashr/mapformer && DRV_GPUS="0 1" DRV_MAXPG=2 setsid nohup bash run_rnn_torus.sh > /dev/null 2>&1 &`
  (after EXPTORUS has finished; see Cost)

## What this can and cannot show
It can say whether a standard recurrent network, parameter-matched to the path model (1 and 2 layers) or at 4x its
parameters, learns this map within the same 900-epoch budget and recipe, where it fails (strata), and whether it reads
its context. It cannot: say anything at other lengths, maps or budgets; rule out an RNN with a different recipe (one rate
per arm, chosen on a 300-epoch pilot on one seed); separate memory capacity from optimisation (a failing arm is
"not learned within budget", not "cannot represent"; no existence construction is attempted -- rule 14 would be the next
step if an RNN fails and the claim needs "cannot"); speak to gated SSMs (Mamba) or to the 4x GRU and 4x 2-layer shapes.

## Amendment 1 (2026-10-26, after the independent code check `docs/audits/2026-10-26/rnn_torus_audit.md`; no batch result exists or was seen)
The check found no BLOCKER (3 MAJOR, 5 MINOR, 13 NOTE). It ran the unit tests (11/11), the selftest (ALL PASS), the pilot
readout (reproduced the picks exactly) and a dry run in a scratch clone (56 launches, no real launch). Changes, all made
before launch and before any batch run:
- **A1. Sequencing (M1).** `run_rnn_torus.sh` refuses to start unless `.exptorus_done` exists (override
  `RNN_SKIP_EXPTORUS=1`, logged as OVERRIDE), refuses while any `python3 mapformer.train_*` process of this user whose argv
  lacks `runs/rnn_torus/` is running (logged with the first one found; rule-23 `ps`/awk form, never pgrep), and refuses
  while a GPU compute process belongs to another user. The checks run once at start (a restart of this driver is allowed
  while its own trainers run). Under `DRV_DRYRUN=1` all three are logged warnings and slot choice does not wait on real GPU
  occupancy (audit m4). Verified in a scratch clone (fake nvidia-smi; a python3 wrapper that refuses any `mapformer.train_`
  launch): the dry run completes with the three warnings while the EXPTORUS pilot trains; a real-mode start refuses on the
  missing marker, and with the override refuses on the running EXPTORUS trainers; a fake foreign-user GPU process is caught.
- **A2. Route reporting (M2).** The registered rules are unchanged. Every RNN cell's line prints the route by which path
  beats it (`accuracy`, `SOLVED`, `both`), and the binding cell's route is printed in the label. If any RNN cell is
  separated on SOLVED only with accuracy d < 0.05, the PATH WINS label reads "PATH WINS (on SOLVED only for <cells>;
  accuracy within 0.05: a within-budget convergence difference, not an accuracy separation)", and any write-up reports it
  that way. SOLVED is a training-loss class on the run's own training map, so this route says "path converges within the
  budget and the RNN does not", not "path is more accurate". Selftest: the `fisher_route` case asserts the qualified label
  and the SOLVED route; the `path_wins` case asserts the unqualified label and route `both`.
- **A3. Learning rate of the 4x LSTM (M3; decision by the coordinator for the user: add the cell).** The 4x LSTM's pilot
  difference (1e-3 0.1602 vs 3e-3 0.1650) was inside the tie band and its curves crossed (epoch 200: 3e-3 lower). **LSTM_L1_h313
  also runs at 3e-3** (cell `LSTM_L1_h313@0.003`, root `runs/rnn_torus/lr3e-03`, seeds 3300-3307, +8 runs, +7.9 GPU-h). It
  is a sixth RNN cell in the primary's intersection-union test (no change of level: PATH WINS must now beat it too), the
  Holm family of RNN WINS has 6 cells, the selected best cell is chosen among all six by the same training-loss rule (so
  between the two 4x cells too), the matched-parameter reading is unchanged (four ~205k cells), and S9 uses the 4x cell at
  1e-3. The other four cells stay at their pilot-chosen rates ("each RNN arm at a pilot-chosen rate: one seed, 300 of 900
  epochs"). **Rate and decay are coupled:** AdamW's decoupled decay is lr x wd per step, so the 3e-3 cells carry 3x the
  effective weight decay of the 1e-3 cells and of the anchors (passive shrinkage over the schedule ~e^-7 vs ~e^-2.4).
  "Rate" in this batch means the (rate, effective decay) pair; RANKWD showed decay can decide the rank-4 path result, so no
  RNN-vs-path difference is attributed to the rate alone. Cost and wall: 64 runs, ~35.4 GPU-h, ~18 h on both kalman GPUs
  (Cost table updated). Predictions for the new cell added (pilot-informed).
- **A4. Context destruction in fp32 (m1).** `run_cd` sets both TF32 switches off, as every other RNN forward in the batch.
- **A5. S1 note (m2).** RoPE_L1 is read in eval mode, as in FACTORIAL_N8 (its 0.724). Eval mode under-reports one-layer
  attention models below ceiling (the 1/(1-p) attention-dropout scale; `docs/audits/2026-10-04/DROPOUT_RESCORE.md`), so
  S1's "RNN ABOVE INDEX 1L" margins are upper bounds. The primary is unaffected (path at ceiling; an under-report could only
  hurt path).
- **A6. S4 wording (m3).** "A T = 2048 drop is a memory-length effect" is withdrawn: a recurrence trained only on 2,047-token
  sequences from a zero state can also drift in state distribution past that length. A drop at 2048 is "memory length or
  recurrent-state length generalisation", not separated.
- **A7. Power sentence (m5).** With path's sd ~0.002 the contrast's variance is almost all the RNN cell's (Welch df ~7), so
  the exact-t multiplier is ~3.26, not 3.01: the 0.05 margin is at or above the MDE for an RNN sd <= ~0.043 (not 0.047).
  No rule changes.
- **A8. Clarifications (n2, n3).** The IUT is level .05 because each cell's beats() is a union of two two-sided tests each
  used only in path's direction (each <= .025 one-sided). Given the UNREADABLE rule (path >= 6/8 SOLVED), RNN WINS is
  practically unreachable; the readable outcomes are PATH WINS (possibly qualified by A2), RNN MATCHES and UNMEASURED.
  S8 "CONTEXT USED" means "reads its context", not "uses a map" (a short-gap retracer also reads it); S3 carries the
  map-vs-retrace question (n5).
Checks after this amendment: selftest ALL PASS; `test_rnn_torus` (driver table = analysis cells with the sixth cell);
dry run in a scratch clone, 64 DRYRUN launches (8 cells x 8 seeds) and the eval / strata / own-map / CD / analysis steps
for the three roots, md5 guard over 107 files.
Launch (after the commit that freezes this file and after EXPTORUS's `.exptorus_done`):
`cd /home/prashr/mapformer && DRV_GPUS="0 1" DRV_MAXPG=2 setsid nohup bash run_rnn_torus.sh > /dev/null 2>&1 &`
