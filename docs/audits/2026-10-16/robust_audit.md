# ROBUST independent code check (rule 29), 2026-10-16, before launch

Scope: `ROBUST_PREREG.md`, `run_robust.sh`, `analyze_robust.py`, the `--weight-decay` change to `train_gauge.py`, at HEAD
839727e; precedents `run_gauge.sh` / `analyze_gauge.py` / `GAUGE_PREREG.md` Amendment 1, `run_generality.sh`. Blind to ROBUST
results (none exist: `runs/robust/` absent, not created). Test runs in the scratchpad (`robust_check/`) on seeds 1480-1482, plus
a full reproduction of the already-published GAUGE m cell at seed 1300 (a stored, already-read run, used only as a bitwise
reference). Check scripts copied to `docs/audits/2026-10-16/robust_check/`. No code edited.

No BLOCKER. The code does what the pre-registration says; every cell's command line sets what the text says; the default cell
reproduces GAUGE bitwise. Two MAJOR points are about the decision rules (a weak NO SIGN COST and the reference cell), one about
the unpiloted torus cell.

## Verified (no action)

1. **Default cell = GAUGE m cell, bitwise, over 300 epochs.** The ROBUST `def` command line (`train_gauge --arm A --deltas=1,-1
   --seed 1300 --lr 1e-3 --weight-decay 0.05`, HEAD code) against the stored `runs/gauge/p0/{SO,AO}_m_s1300`: per-epoch losses
   identical (300/300), every state-dict tensor `torch.equal`, `eval.json` identical, both arms (`cmp_ckpt.py`). Stored GAUGE
   args equal ROBUST's apart from the new `weight_decay` key (size 32, 300 epochs, 98 batches, batch 128, T 128, lr 1e-3,
   data_workers 3, 400 trials).
2. **HEAD~1 vs HEAD, default flag** (seed 1480, 3 epochs x 5 batches, old on cuda:0, new on cuda:1): losses, state dicts and
   eval.json identical for SO and AO. Confirms the author's check.
3. **Weight decay reaches the optimiser, all parameters** (AdamW instrumented, `wrap_gauge.py`, seed 1481, the four driver
   command lines, 20 epochs x 5 batches): ONE param group of 24 tensors (204,243 params = every parameter, including omega,
   W_in / W_out, embeddings, LayerNorm gains and biases) with weight_decay 0.05 (def, lrx3, lrd3) and **0.0 (wd0)**. Saved args
   record lr / wd per cell.
4. **lr and schedule.** Recorded per-step lr: 5% linear warmup from lr/warm, peak = lr, cosine to 0.1 x lr
   (def/wd0 1e-3 -> 1e-4; lrx3 3e-3 -> 3e-4; lrd3 3.333e-4 -> 3.33e-5). Grad-norm clip 1.0 in every cell. lrd3 is 3.333333e-4
   (1e-7 relative from 1e-3/3; irrelevant); `analyze_robust` matches it exactly.
5. **Torus model at d 256, 2 layers** (`torus_struct.py`, train_variant's construction order, seeds 1480-1481): 1,592,469
   params (d 128 / 1 layer: 204,757; both as stated); 2 layers, d_head 128, 64 blocks per head, omega (2, 64) geometric
   2 pi -> 2 pi/64 (ratio 64, monotone decreasing, heads identical); W_in (4, 256), W_out (128, 4) (one shared rank-4 latent);
   act = (0, 1, 2, 3) (torus moves; pairs (0,1), (2,3) are opposites in `environment.ACTION_DELTAS`). Step table: every
   non-move row exactly 0 in both arms; AO == |SO| exactly; no step at any observation position on a real batch. Both layers
   receive the identical cos/sin (forward pre-hooks). `--fast-attn`: 2 plain `WMTransformerLayer`s, no fallback note. Nothing
   falls back to 1 layer or ignores `--d-model`: the checkpoint config records d_model 256, n_layers 2, and the state dict has
   `layers.0` and `layers.1`.
6. **Init scale is comparable to d 128** (`init_scale.py`, 8 seeds): mean |step| 0.251 (d 128) vs 0.274 (d 256); share of
   (move, channel) angles > pi at init 0.010 vs 0.012 (MECHANISM's start lever is not accidentally engaged). The ladder has
   the same range and twice the channels; identical across arms, so it cannot create an arm difference.
7. **Pairing at a seed.** Torus (`wrap_variant.py`, seed 1482, driver command line with 2 epochs x 4 batches on different
   GPUs): SO and AO initial state dicts identical (all tensors), the md5 of all 8 training batches (tokens + revisit mask)
   identical, AdamW one group of 40 tensors, wd 0.05. Ring: init and stream do not depend on lr / wd (built before `train()`),
   so all four recipes x two arms share them at a seed (also implied by item 1).
8. **Seeds / streams.** No directory under `runs/` uses seeds 1400-1407 (nor 1490s). Batch-seed streams `s*1000003 + i`
   (ring <= 29,400 batches, torus <= 88,200) cannot overlap across seeds 1300-1307 / 1400-1407. Eval maps (env seed 10000)
   differ from training maps.
9. **analyze_robust.** `cell_label` is analyze_gauge's G1/G2 rule verbatim (gate SO < 4; Fisher p < .05 with SO ahead, or
   perm p < .05 with d = AO - SO <= -0.05; NSC AO >= SO - 1 and mean AO >= mean SO - 0.01; else UNMEASURED). `reading`
   checked against an independent implementation written from the prereg text over all 4^5 = 1024 label combinations: 0
   mismatches; every reading reached (`reading_enum.py`). `--selftest` passes. Missing run -> FileNotFoundError (not dropped);
   a recipe directory swapped for another -> AssertionError on lr/wd; wrong epoch count -> AssertionError; torus config keys
   (d_model, n_layers, n_heads, n_steps, lr, batch, n_batches, schedule, fast_attn, epochs, variant, seed) asserted. SOLVED =
   `classify_run` (final 5% of per-epoch training loss < 0.05). Seeds parse 1400-1407 (n = 8). Verdict JSON is written
   before the secondaries.
10. **Driver.** Clean-tree and committed-file checks precede every wait; `flock -n`; md5 guard over 96 imported modules (every
    executed file checked present: train_gauge, environment_steps, model_wrap_exist, model_sign, model, train, data_parallel,
    train_cancel, environment, eval_noise_refine, eval_rank_strata, stats_core, analyze_robust, ...) + 2 driver files, rechecked
    before the evals; rule-23 awk everywhere; resume skips done (`.pt` / `eval.json`) and running runs; artifacts-then-marker.
    Dry run in a `git clone --shared` of HEAD (`DRV_DRYRUN=1`): 80 launches in the stated order, no duplicate output dir,
    guard over 98 files. Incidentally it waited 90 s for my two scratch `train_gauge` jobs to finish before the first torus
    launch: the one-per-GPU slot counts any `mapformer.train_*` of the user, as intended.

## Findings

### MAJOR 1 -- NO SIGN COST is readable when the signed arm itself barely learns, so "lr / 3 under-trains both" reads RECIPE-DEPENDENT
The gate admits SO = 4/8. With SO 4/8 and AO 3/8 (or 4/8) at equal accuracy, `cell_label` returns NO SIGN COST (checked:
`cell_label([0.9]*8, [0.9]*8, 3, 4, 8)` -> NSC), and the reading then says RECIPE-DEPENDENT -- exactly the reviewer's
objection (a recipe that slows both arms looks like a recipe without a sign cost). At SO 4/8 no AO count can reach Fisher
p < .05 (0/8 gives 0.077), so the cell could not have shown the cost by SOLVED; reading its non-detection as "no cost" is
asymmetric. Also NSC includes reversals (AO 8/8 vs SO 4/8 -> NSC), which would deserve their own name.
**Fix (amendment):** NO SIGN COST additionally requires SO >= 6/8 (the smallest SO count at which Fisher can fire: 6/8 vs
1/8, p 0.041). A cell with SO 4-5/8 that is not SIGNED LEARNS BETTER reads UNMEASURED (signed arm weak). Optional: label
AO SOLVED > SO SOLVED with Fisher p < .05 "NON-NEGATIVE LEARNS BETTER" (counts as NSC for the reading).

### MAJOR 2 -- RECIPE-DEPENDENT does not require the default cell to show the cost
Recipe dependence means "cost under the default, none under X". The reading only gates on def being UNREADABLE. If def reads
NO SIGN COST (`reading` -> "RECIPE-DEPENDENT (NO SIGN COST in ring|def)") that is a failure to replicate GAUGE G2 / E3 on
fresh seeds, not a recipe effect; if def reads UNMEASURED and wd0 NSC, the reading still says RECIPE-DEPENDENT with no
in-batch reference cost to depend on.
**Fix:** insert after the UNREADABLE branch: **DEFAULT NOT REPLICATED** if ring|def reads NSC (the headline; the other cells
reported descriptively), and **REFERENCE UNMEASURED** if ring|def reads UNMEASURED; RECIPE-DEPENDENT then requires
ring|def = SIGNED LEARNS BETTER. SCALE-DEPENDENT keeps its torus-only condition (see MINOR 3 for its reference).

### MAJOR 3 -- the torus cell (7.2 of 10 GPU-hours) has no full-budget pilot of the signed arm
The only d 256 / 2-layer runs were 10-epoch timing schedules. Whether SignedAO4_r4 at 2 layers, d 256, lr 1e-3, batch 16
solves within 900 epochs is unknown (GENERALITY's 8/8 is at 1 layer, d 128; wider and deeper models usually want a lower lr,
if anything, and 2-layer path models on this torus have not been trained at r=4). If SO < 4/8 the cell is UNREADABLE and the
batch's most expensive part says nothing; the prediction section names the lr cells' risk but not this one.
**Fix:** before launch, one pilot pair on an outside seed (SO and AO, seed 1490, the driver's exact command line, 900
epochs, one per GPU: ~55 min wall, both GPUs). Proceed if the pilot SO is SOLVED; if not, decide the budget now (rule 4),
e.g. 1800 epochs or lr 5e-4, not after an UNREADABLE. If the cost of the pilot is not wanted, state the risk explicitly in
the prediction ("torus SO at d 256 / 2 layers unpiloted at full budget; UNREADABLE is a live outcome").

### MINOR 1 -- the lr cells also change the effective weight decay
PyTorch AdamW decays p <- p (1 - lr wd) per step, so lr x3 triples and lr / 3 thirds the per-step decay (and the 0.1 x lr
cosine floor moves with lr). lrx3 / lrd3 are therefore "lr and lr x wd", not lr alone; wd0 isolates decay at the default lr.
Decay is plausibly relevant to the mechanism (the wrap solution needs large steps, W_out grows; decay pulls angles toward
zero), so wd0 is the cell with a mechanistic prediction. **Fix:** one sentence in Design and in "What this can and cannot
show"; a NO SIGN COST in lrd3 alone, with wd0 SIGNED LEARNS BETTER, is then not attributable to lr.

### MINOR 2 -- power sentence is wrong for SO 5/8
"With SO 4-5/8 no AO count reaches p < .05": SO 5/8 vs AO 0/8 gives Fisher p 0.026 (stats_core `fisher_solved`). Table:
SO 8: AO <= 3 (0.026); SO 7: AO <= 2 (0.041); SO 6: AO <= 1 (0.041); SO 5: AO = 0 (0.026); SO 4: none (0/8 -> 0.077).
**Fix:** correct the sentence.

### MINOR 3 -- SCALE-DEPENDENT compares with an out-of-batch reference (rule 12)
The torus cell has no in-batch d 128 / 1-layer pair; "scale-dependent" would be read against GENERALITY E3 (seeds 700-707,
another batch). SIGNED LEARNS BETTER at d 256 stands alone; only the dependence claim needs the reference.
**Fix:** either disclose ("SCALE-DEPENDENT is relative to GENERALITY E3, out of batch") or add SO4 / AO4 at d 128, 1 layer on
seeds 1400-1407 (16 runs, the GENERALITY command line; ~1 GPU-s/epoch -> ~2 h of GPU time, can share GPUs with the ring
phase). Disclosure is enough if MAJOR 2's logic is adopted for the ring only.

### MINOR 4 -- 2 layers can buy a non-path partial workaround; read accuracy with care
With a monotone (clock-like) phase, a second layer can build content features from the first (index RoPE gained from depth on
short-gap revisits: FACTORIAL 0.72 -> 0.86 at 4 layers, never SOLVED). That can narrow the torus accuracy gap without
touching the sign cost. SOLVED (loss < 0.05) is unaffected in practice. **Fix:** say so in the predictions; the declared
strata secondary (lag < 128 / >= 128 / wrap-only) is the readout that separates it.

### MINOR 5 -- analysis config asserts are incomplete
Ring: `size`, `n_steps`, `batch_size`, `n_batches`, `data_workers` are not asserted (only arm, seed, deltas, lr, wd,
epochs). Torus: `grid_size` (64), `n_obs_types` (16), `n_landmarks` (0), `args.data_workers` (3), `deterministic` (False),
`init_from` (None), `data_seed_offset` (0) are not asserted. The driver sets them correctly; the asserts guard a resumed or
hand-run checkpoint. **Fix:** add them to the two assert lines (no design content).

### MINOR 6 -- eval device hard-coded
`eval_noise_refine` / `eval_rank_strata` run on `cuda:0` while training honours `DRV_GPUS`. Launched with `DRV_GPUS="1"`
(GPU 0 busy), the evals would land on GPU 0. **Fix:** `EVDEV="cuda:${DRV_GPUS%% *}"` and use it in both eval calls.

### MINOR 7 -- the last torus wave trains beside the ring phase
The ring loop starts only after the 16th torus launch, so seeds 1407 SO / AO train beside up to 5 ring jobs per GPU
(measured 11.9 s/epoch beside 2 ring runs vs 3.6 alone). Numerics are unaffected (concurrency does not change the stream or
init); only wall time. The ~10 h estimate is roughly right (~6.3 h for waves 1-7, then ~3.5 h for the ring phase with wave 8).
No fix needed; mention it so a slow wave 8 is not mistaken for a hung run.

## Proposed amendment text (Amendment 1, before launch)

> **Amendment 1 (2026-10-16, after the independent check `docs/audits/2026-10-16/robust_audit.md`; before launch).** The check
> found the code correct: the default cell reproduces GAUGE's SO_m / AO_m at seed 1300 bitwise over 300 epochs; weight decay
> reaches the single AdamW group (all parameters; 0 in wd0); lr and cosine schedule as stated; at d 256 / 2 layers the step map
> is shared, both layers get the same phase, observation steps are 0, AO = |SO| exactly, initial weights and training stream
> are shared by the arms; every reading reachable and matched to this text over all 1024 label combinations. Changes:
> - **NO SIGN COST requires SO SOLVED >= 6/8** (the smallest count at which the Fisher route could have shown a cost). A cell
>   with SO 4-5/8 that is not SIGNED LEARNS BETTER reads UNMEASURED. (MAJOR 1)
> - **Reading, new branches after UNREADABLE:** DEFAULT NOT REPLICATED if ring|def reads NO SIGN COST; REFERENCE UNMEASURED if
>   ring|def reads UNMEASURED. RECIPE-DEPENDENT requires ring|def = SIGNED LEARNS BETTER. SCALE-DEPENDENT is relative to
>   GENERALITY E3 (out of batch, rule 12; stated). (MAJOR 2, MINOR 3)
> - **Torus pilot** (seed 1490, SO and AO, the driver's command line, 900 epochs) before launch; the batch launches at the
>   registered budget only if the pilot SO is SOLVED, otherwise the torus budget is changed here before launch. [or: the risk
>   is stated and accepted] (MAJOR 3)
> - Disclosed: the lr cells also scale AdamW's per-step decay (lr x wd) and the cosine floor; wd0 isolates decay. Power
>   sentence corrected (SO 5/8 vs AO 0/8 gives p 0.026). At 2 layers a non-path second-layer route can narrow the accuracy gap;
>   the strata secondary separates it. (MINOR 1, 2, 4)
> - Code (no design content): extra config asserts in `analyze_robust`; eval device from `DRV_GPUS`. (MINOR 5, 6)

## Commands (all in the scratchpad `robust_check/`; scripts in `docs/audits/2026-10-16/robust_check/`)
- bitwise HEAD~1 vs HEAD: `git archive 839727e~1` / `839727e` to `old/` `new/`; `wrap_gauge.py OUT -- --arm A --deltas=1,-1
  --seed 1480 --epochs 3 --n-batches 5 --n-trials 50` (old without flags, new with `--lr 1e-3 --weight-decay 0.05`);
  `cmp_ckpt.py` -> "losses equal: True | tensors differing: 0", "eval.json equal: True" (both arms).
- GAUGE reproduction: `python3 -m mapformer.train_gauge --arm {SO,AO} --deltas=1,-1 --seed 1300 --lr 1e-3 --weight-decay 0.05`
  (HEAD copy, 300 epochs, ~5 min) vs `runs/gauge/p0/{SO,AO}_m_s1300` -> identical losses, tensors, eval.json.
- recipes: four driver command lines at seed 1481, 20 x 5, AdamW instrumented -> groups [(0.05, 24)] / [(0.0, 24)]; lr first /
  peak / last 2.0e-4 / 1.0e-3 / 1.0e-4 (def, wd0), 6.0e-4 / 3.0e-3 / 3.0e-4 (lrx3), 6.7e-5 / 3.3e-4 / 3.3e-5 (lrd3).
- torus: `torus_struct.py`, `init_scale.py` (CPU); `wrap_variant.py` with the driver's torus line at seed 1482, 2 epochs x 4
  batches -> "batches 8 identical stream: True", groups [[0.05, 0.001, 40]] both arms.
- analysis: `reading_enum.py` -> "combos 1024, mismatches 0"; loaders on the scratch ring runs (missing file, swapped recipe,
  epoch mismatch all raise); `python3 -m mapformer.analyze_robust --selftest` passes.
- driver: `git clone --shared` of HEAD in scratch, `DRV_DRYRUN=1 DRV_GPUS="0 1" bash run_robust.sh` -> 80 DRYRUN lines, no
  duplicate `--output-dir`, md5 guard over 98 files.
