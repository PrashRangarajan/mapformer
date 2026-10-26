# RNN_TORUS -- independent code check (rule 29), 2026-10-26

Checker: a separate session that did not write the code. It has not seen any batch result (none exists; seeds 3300-3307 have
not been run). Only the lr-pilot training logs (seed 3390) were read, to check the pick rule. No GPU work, no driver run
without DRV_DRYRUN=1, no commit, no edits to the files under review or to tracked files.

Files checked: `RNN_TORUS_PREREG.md`, `model_rnn.py`, `train_rnn.py`, `eval_rnn_torus.py`, `analyze_rnn_torus.py`,
`run_rnn_torus.sh`, `run_rnn_torus_pilot.sh`, `run_rnn_torus_pilot2.sh`, `test_rnn_torus.py`, `rnn_torus_pilot.log`,
`runs/rnn_torus_pilot/lr/*.log`. Context files: `train_variant.py`, `train.py`, `eval_noise_refine.py`,
`eval_rank_strata.py`, `stats_core.py`, `lib_driver.sh`, `run_factorial_n8.sh`, `runs/factorial_n8/code_md5.txt`, the
FACTORIAL_N8 reference checkpoint.

**Verdict: no BLOCKER. Two MAJOR findings (M1, M2), both fixable without changing a registered rule. M3 is a scope decision
for the user. Launch once M1 is fixed in the driver, M2 in the analysis's printed label, and Amendment 1 (below) is in the
prereg before the commit that freezes it.**

## What was run
- `python3 -m mapformer.test_rnn_torus` (CPU, smoke included): **11/11 passed** (parameter counts, causality with a
  bidirectional negative control, forget-gate bias, registration reaching both evaluators, train_rnn refusals, driver table
  = analysis cells, fresh seeds, the context-destruction controls, CD walks = eval walks, and a CPU smoke of all 8 arms through
  train_rnn, both evaluators and run_cd).
- `python3 -m mapformer.analyze_rnn_torus --selftest`: **ALL PASS**; all five primary labels are reached, and each case
  checks the labels it must not print.
- `python3 -m mapformer.analyze_rnn_torus --pilot`: reproduces the prereg's pilot table and picks exactly
  (LSTM_L1_h313 0.1602 / 0.1650 -> 1e-3; LSTM_L1_h157 3e-3; GRU_L1_h181 0.3285 / 0.3265 -> 1e-3 by the tie rule; LSTM_L2_h111
  3e-3; GRU_L2_h128 3e-3). Every pilot log prints `cudnn.allow_tf32 False, matmul.allow_tf32 False`.
- Dry run in a scratch clone (`git clone` of HEAD a8ab5f7 plus the untracked files under review, fake `nvidia-smi` on PATH,
  `DRV_DRYRUN=1 DRV_GPUS="0 1"`): exit 0; 56 launches logged (seed-major, 7 cells per seed, flags as registered: anchors
  `train_variant ... --fast-attn --lr 1e-3`, RNNs `train_rnn` with no `--fast-attn`), then per root noise_refine / strata /
  8 own-map evals, CD and the analysis, logged only; "md5 guard over 107 files"; the not-yet-committed guarded files produce
  the expected DRYRUN WARNING. **Zero real launches** (no `mapformer.train_` process carrying `rnn` before or after). The clone
  needed `DRV_MAXPG=8`: `drv_ntrain` reads the real `ps`, which shows the 4 EXPTORUS pilot trainers (2 per GPU), so with
  `DRV_MAXPG=2` the dry run waits in `drv_wait_slot` forever (m4).
- Code identity: `train.py`, `train_variant.py`, `model.py`, `environment.py`, `data_parallel.py`, `eval_noise_refine.py`,
  `eval_rank_strata.py`, `stats_core.py`, `analyze_factorial_n8.py`, `lib_driver.sh` all have the md5s recorded in
  `runs/factorial_n8/code_md5.txt`. The FACTORIAL_N8 path checkpoint's CLI args equal the current parser's keys (no new
  keys), and its RoPE checkpoint has the same recipe.

## Findings

### MAJOR

**M1. The driver does not refuse while EXPTORUS (or any other batch) is training.** `run_rnn_torus.sh:23-66`.
The prereg (Cost, "Ordering on kalman") says RNN_TORUS starts after `.exptorus_done` or after EXPTORUS's trainers have all
exited, "not beside it". The driver has no such check. `drv_wait_slot` counts every `mapformer.train_*` process, so an early
launch would not overfill a GPU. Instead it takes slots as EXPTORUS frees them, and the two batches interleave for their
whole length. This was confirmed live: the four `mapformer.train_exptorus` pilot processes are running now, and
`.exptorus_done` is absent. *Scenario:* the RNN driver is started after the EXPTORUS pilot ends but before the EXPTORUS
batch starts (or by a session that misreads the state). Both batches then run at about half speed for about 24 h, and
EXPTORUS's planned wall time no longer holds. *Fix:* before `drv_lock`'s first launch (after line 28), refuse unless
`$REPO/.exptorus_done` exists or `RNN_TORUS_AFTER_EXPTORUS_OVERRIDE=1` is set. Also refuse while any `python3`
`mapformer.train_` process whose argv lacks `$R/` is running (rule-23 awk form), logging the first one found. Skip both
checks with a logged warning under `DRV_DRYRUN=1`.

**M2. On the SOLVED route alone, PATH WINS can be printed when held-out accuracy is within the MDE.**
`analyze_rnn_torus.py:173` (beats), `:187` (label), `:465` (selftest case `fisher_route`, which asserts PATH WINS with the
RNN cell at mean accuracy 0.994 vs path 1.000). SOLVED is a training-loss class (final-5% loss < 0.05 on the run's own
training map). An RNN that is still DESCENDING at 900 epochs with loss around 0.06-0.10 can sit at held-out accuracy
0.97-0.99. On this task such losses went with 0.973-0.990 accuracy in the r=2 loop and depth arms (LOOP_RANK). The rule then
reads PATH WINS through Fisher (8/8 vs <= 3/8) on an accuracy gap of 0.01-0.03. That is the rule as registered, inherited
from FACTORIAL, and it is legitimate as "path converges within the budget, the RNN does not". The printed label, though,
says "path beats every RNN cell" and does not say which route fired. This is the scenario the prereg itself predicts for
the binding cell (LSTM_L1_h313, "via Fisher if it solves <= 3/8"). *Fix (reporting only; the verdict rule is unchanged):*
print each cell's route (`accuracy`, `SOLVED`, or both). When the binding cell is separated on SOLVED only with d < 0.05,
append "(binding cell separated on SOLVED only: accuracy d +X, below the 0.05 margin -- a within-budget convergence
difference, not an accuracy separation)". Add a selftest assertion on the route text in `fisher_route`.

### MAJOR (scope; the user decides)

**M3. The binding cell's learning rate is a coin flip, and the IUT would let both rates in at no cost to the level.**
`RNN_TORUS_PREREG.md:76-79, 193-203`; `analyze_rnn_torus.py:53`. LSTM_L1_h313 was picked on one seed at epoch 300 of 900:
0.1602 (1e-3) vs 0.1650 (3e-3). That gap is inside the 0.01 tie band, so the tie rule chose 1e-3. The curves cross: at epoch
200, 3e-3 was lower, 0.1841 vs 0.1864. The other two LSTM arms preferred 3e-3. Readouts at epoch 300 also come while the
cosine LR is still at ~0.82 of peak. Because PATH WINS is an intersection-union test, adding cells cannot inflate its level;
it only makes PATH WINS harder. Adding `LSTM_L1_h313@3e-3` as a sixth RNN cell would make the most likely verdict immune to
"the 4x LSTM was run at the wrong rate". Cost: +8 runs, +7.9 GPU-h, about +4 h wall on both GPUs (total ~18 h). If it is
not added, the label and the prereg must say "each RNN arm at a pilot-chosen rate (one seed, 300 of 900 epochs)". The
prereg's Pilot section says this, but the Recipe section and the printed labels do not. Note also that AdamW's decoupled
decay is lr x wd, so the 3e-3 arms carry 3x the effective decay of the 1e-3 arms and anchors. Passive shrinkage over the
schedule is ~e^-7 at 3e-3 vs ~e^-2.4 at 1e-3. The "rate" is therefore a (rate, effective decay) pair. RANKWD showed decay
decides the rank-4 path result, so say so.

### MINOR

**m1. Context destruction runs with cuDNN TF32 ON.** `analyze_rnn_torus.py:336-338` (`run_cd`), called from `main()` with
the default `torch.backends.cudnn.allow_tf32 = True`. Every other RNN forward in the batch (train_rnn, eval_rnn_torus) turns
it off. The gate's margin (0.27 vs 0.51) makes a wrong label very unlikely, but `clean` would then not be the fp32 number,
contrary to the prereg's "every arm is fp32". *Fix:* set both TF32 switches off at the top of `run_cd`, as in
`eval_rnn_torus.main`.

**m2. S1 (RNN vs index RoPE 1L) reads RoPE in eval mode, which under-reports one-layer attention runs below ceiling.**
`RNN_TORUS_PREREG.md:128-129, 145`. CLAUDE.md (TW_NORMSTEP, DROPOUT_RESCALE): eval mode under-reports one-layer models
below ceiling through the 1/(1-p) attention-dropout scale, by up to +0.17. RoPE_L1 (~0.72) is such a model, so "RNN ABOVE
INDEX 1L" margins may be inflated. The prereg's "Not declared: the dropout re-score (RNN dropout is between layers only...)"
deals only with the RNN side. The primary is unaffected: path is at ceiling, and an under-report could only hurt path.
*Fix:* say so in S1, and quote FACTORIAL_N8's convention (eval mode, the same as the 0.724 there).

**m3. S4 wording.** `RNN_TORUS_PREREG.md:135-136`. "An RNN has no position extrapolation problem, so a T = 2048 drop is a
memory-length effect" is too strong. A recurrence trained only on 2,047-token sequences from a zero state can drift in
state distribution past that length. A drop at 2048 is "memory length or state-dynamics length generalisation", and S4
cannot tell them apart.

**m4. The registered dry-run command hangs while another batch holds the slots.** `run_rnn_torus.sh:66, 84`. In DRYRUN,
`drv_wait_slot` still polls real `ps` counts. With EXPTORUS at 2 per GPU and `DRV_MAXPG=2`, the command in the prereg
(Process, "Dry run") waits forever. *Fix:* in DRYRUN, replace `G=$(drv_wait_slot)` with a logged `drv_pick 99 0` or a fixed
`0`, or document the hang.

**m5. The MDE line uses a pooled-df multiplier.** `RNN_TORUS_PREREG.md:121-124`. With path's sd ~0.002, the contrast's
variance is almost all the RNN arm's (Welch df ~7). The exact-t multiplier is then ~3.26 rather than 3.01, so the 0.05 margin
is at or above the MDE for an RNN sd <= ~0.043, not 0.047. No rule changes; correct the sentence.

### NOTE

- **n1. Data stream and recipe: verified identical.** `train_rnn` sets TF32 off, registers the arms and calls
  `train_variant.main()` with the same argv. `torch.manual_seed(seed)` comes before model construction, and the parallel
  generator's base seed is `torch.initial_seed()`, so the model's init draws do not touch the data. At a seed, every arm
  gets the same GridWorld map and the same batch stream. The loss positions are `revisit_mask[:, 1:]`: the output at action
  a_t predicts o_t, and the RNN is unidirectional (`nn.LSTM/GRU`, `bidirectional=False`; tested by perturbation). There is
  no hidden-state reset or special handling at the interleave, and none is needed. Eval at 512 / 2048 works (no
  length-specific state). The remaining asymmetries are by design and in the prereg: no `--fast-attn` for the RNNs, TF32 off
  everywhere (the anchors' matmuls are fp32 by default anyway), no dropout in 1-layer RNNs vs attention dropout in the
  anchors (this makes SOLVED easier for the RNN, so it is conservative for PATH WINS), and weight decay on every parameter,
  including the LSTM forget bias of 1.
- **n2. Level of the IUT: correct.** Each `beats(c)` is a union of two two-sided tests (permutation, Fisher), each used
  only in path's direction, so each is <= .025 one-sided and the union is <= .05. The IUT over the cells is then level .05
  without correction. State that reason in the prereg rather than just "no correction". The matched-parameter companion is
  a separate registered claim, legitimately so: it is an IUT over a subset.
- **n3. RNN WINS is practically unreachable.** Path must be >= 6/8 SOLVED, or the label is UNREADABLE. Given that, the
  Fisher route needs Holm p < .05, which 8/8 vs 6/8 (p 0.47) cannot give. The accuracy route needs acc(c) >= acc(path) +
  0.05 with raw p ~<= 0.01, so path's mean must already be <= ~0.95. The selftest reaches it only with path at 0.9 and 6/8.
  Say in the prereg that the readable outcomes are PATH WINS / RNN MATCHES / UNMEASURED.
- **n4. Selection by training loss can pick a memoriser.** SOLVED and tail loss are on the run's own training map. An RNN
  that stores that map in its weights would be selected and then fail RNN MATCHES on the held-out map, giving UNMEASURED even
  if another cell matched. This is conservative (it never produces a false RNN MATCHES), and S5 shows it. No change needed.
- **n5. CD gate.** It does destroy context-based prediction for RNNs: the walk, actions and revisit mask are asserted
  identical, the np RNG state after each walk is asserted equal, and the walks are eval_noise_refine's, tested. A model that
  only retraces short gaps also reads CONTEXT USED, so S8 says "reads its context", not "uses a map". The predicted "CONTEXT
  USED for every cell" is nearly certain; S3 carries the map-vs-retrace question. `cd_gate` takes the blank floor from the
  100 eval walks and the destroyed accuracy from the first 50, which is harmless.
- **n6. Recipe assert (builder's d).** It is safe. A CLI key absent from FACTORIAL_N8's checkpoint only prints a NOTE, but
  every run in the batch must share one recipe (asserted), so a new key cannot create a within-batch asymmetry. Today there
  are no new keys, and the code md5s equal FACTORIAL_N8's. Optionally, have the analysis print that md5 comparison beside
  the recipe line.
- **n7. md5 guard (builder's e).** The import closure (107 files) includes train_variant's model modules, train, data_parallel,
  stats_core, analyze_lrsweep, analyze_factorial_n8, the evaluators and model_rnn. The lazy imports are all inside it
  (`ckpt_guard` is added explicitly; `environment_nd` / `minigrid_env` are not reached on this path). It is checked before
  every launch, before the evals and before the analysis. The clean-tree refusal covers tracked files only, plus the
  "guarded file committed" check. Both refuse in a real run and warn in DRYRUN. Correct.
- **n8. `--d-model` / `--n-heads` (builder's f)** are recorded (128 / 2) and ignored by `RNNBaseline`. `n_layers` is
  asserted equal to the arm's depth, so a rebuild from config gets the trained shape. Harmless. `n_params_of("RoPE")` is
  hard-coded (203,925) and untested, but it is used only in a printed line.
- **n9. Parameter counts** match the prereg exactly (module count = closed form = table; within 0.5% of 204,757 / 0.2% of
  798,741). State sizes are as stated.
- **n10. Seeds.** 3300-3307 and 3390 are disjoint from EXPTORUS (3400-3407, pilot 3490-3499; EXPTORUS_PREREG confirms it
  moved off 3300-3307 for this batch). No run dir in runs/ carries a batch seed. Per-batch data seeds s x 1000003 + i,
  i < 88,200, cannot overlap across seeds.
- **n11. Cost.** The arithmetic is consistent (27.5 GPU-h training; ~14 h on both GPUs, assuming time-slicing, which the
  pilot measured on LSTM_L1_h157 only; GRU_L2_h128 at 0.7 s/epoch is probably data-bound and may gain from packing, which
  only shortens the wall time). 2 per GPU is reasonable.
- **n12. Disclosure (builder's h).** The RNN predictions are labelled "PILOT-INFORMED, NOT BLIND", and the anchors'
  predictions are sourced to FACTORIAL / FACTORIAL_N8. Adequate.
- **n13. Interpretation.** At 181-626 state floats, an RNN failing on long-gap and wrap-only revisits is close to predicted
  by state size. A PATH WINS answers the reviewer's specific attack ("an RNN integrates by construction") and shows,
  through S3, where the RNN fails. It does not show that attention integrates better than recurrence. The prereg's "cannot"
  paragraph covers this; keep that wording in any write-up.

## Proposed "## Amendment 1" text (to be added to RNN_TORUS_PREREG.md before the freezing commit)

```
## Amendment 1 (2026-10-26, after the independent code check `docs/audits/2026-10-26/rnn_torus_audit.md`; no batch
result exists or was seen)
A1. Sequencing (M1). run_rnn_torus.sh refuses to start unless .exptorus_done exists (override
    RNN_TORUS_AFTER_EXPTORUS_OVERRIDE=1, logged), and refuses while any python3 mapformer.train_* process outside
    runs/rnn_torus/ is running. Both checks are warnings under DRV_DRYRUN=1, and the dry run does not wait on GPU slots.
A2. Route reporting (M2). The registered rules are unchanged. The primary line prints, for every RNN cell, the route by
    which path beats it (accuracy / SOLVED / both). If the binding cell is separated on SOLVED only with accuracy
    d < 0.05, the PATH WINS label carries "(binding cell separated on SOLVED only: a within-budget convergence difference,
    not an accuracy separation)", and any write-up reports it that way.
A3. Learning rate (M3). [EITHER] LSTM_L1_h313 is also run at 3e-3 (cell LSTM_L1_h313@0.003, root lr3e-03, seeds
    3300-3307, +7.9 GPU-h). It enters the primary's IUT as a sixth RNN cell (no change of level: PATH WINS must beat it
    too), the Holm family of RNN WINS becomes 6, and S9 uses the 1e-3 cell. [OR] No added cell: every label and S9 read
    "each RNN arm at a pilot-chosen rate (one seed, 300 of 900 epochs; the 4x LSTM's 1e-3 vs 3e-3 difference, 0.005,
    was inside the tie band)". In both cases: the rate is a (rate, AdamW decoupled decay = rate x 0.05) pair, so the 3e-3
    arms carry 3x the effective weight decay of the 1e-3 arms and the anchors.
A4. Context destruction runs with both TF32 switches off (m1).
A5. S1 note (m2): RoPE_L1 is read in eval mode, as in FACTORIAL_N8. Eval mode under-reports one-layer attention models
    below ceiling (attention-dropout 1/(1-p) scale; DROPOUT_RESCALE), so S1's "RNN ABOVE INDEX 1L" margins are upper
    bounds. The primary is unaffected (path at ceiling).
A6. S4 wording (m3): a T = 2048 drop is "memory length or recurrent-state length generalisation", not separated.
A7. Power sentence (m5): with path's sd ~0.002 the exact-t multiplier is ~3.26 (Welch df ~7); the 0.05 margin is at or
    above the MDE for an RNN sd <= ~0.043.
A8. Clarifications (n2, n3): the IUT is level .05 because each cell's test is a union of two directional uses of two-sided
    tests (each <= .025 one-sided). Given the UNREADABLE rule, RNN WINS is practically unreachable; the readable outcomes
    are PATH WINS, RNN MATCHES and UNMEASURED.
```
