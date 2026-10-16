# Is the sign learnability cost an artefact of one recipe or one tiny scale? -- pre-registration (2026-10-16, before the batch)

## Question
Every sign contrast so far used one optimiser recipe per task and one model size (1 layer, 2 heads, d 128): the ring at lr 1e-3,
weight decay 0.05 (WRAP_EXISTENCE E3 0/8 vs 8/8, GAUGE G2 1/8 vs 8/8), the torus at the same size (GENERALITY E3 0/8 vs 8/8).
A reviewer's first question for a learnability claim is whether it is an AdamW / learning-rate artefact or a property of a
model too small to have slack (`docs/paper/READ_draft4.md` (e) weak point 1; advisor plan `docs/plans/2026-10-14_next_steps.md`
items 11-12). This batch varies the recipe on the ring one factor at a time and the model size on the torus, with every
contrast in-batch (rule 12).

## Design
**R1, recipe robustness, ring** (`train_gauge.py --deltas=1,-1`, the GAUGE m cell = WRAP_EXISTENCE E3 setting: ring 32 cells,
moves +1/-1, p_plus 0.5, T = 128, batch 128, 98 batches, cosine (5% warmup, floor 0.1), 300 epochs, 1 layer, 2 heads, d 128,
AdamW). Arms SO / AO = `MapFormerWM_SignedAO_r4` / `MapFormerWM_AbsAO_r4` (observation-step-free: only the two moves step).

| cell | lr | weight decay | note |
|---|---|---|---|
| def | 1e-3 | 0.05 | the RING_DIAL / GAUGE recipe: `train_gauge`'s lr default 1e-3, `train()`'s weight-decay default 0.05; in-batch reference |
| wd0 | 1e-3 | 0 | no weight decay |
| lrx3 | 3e-3 | 0.05 | three times the default lr |
| lrd3 | 3.333333e-4 | 0.05 | a third of the default lr |

4 recipes x 2 arms x 8 seeds = 64 runs. Evaluation inside the trainer (`train_cancel.evaluate`, fresh stream, np seed 10**6, maps
redrawn, 400 trajectories) at T = 128 (primary) and 4T = 512.

**R2, scale, torus** (`train_variant`, GENERALITY's torus recipe: lr 1e-3, batch 16, 98 batches, T = 1024, 900 epochs, cosine,
`--fast-attn`, `--data-workers 3`, OMP 2, weight decay 0.05). Arms SO / AO = `SignedAO4_r4` / `AbsAO4_r4` at **2 layers,
d_model 256, 2 heads (d_head 128, 64 rotary blocks per head, 128 rotary channels in all, vs 32 per head / 64 at d 128)**;
1,592,469 parameters (vs 204,757 at GENERALITY's 1 layer, d 128). 2 arms x 8 seeds = 16 runs. Evaluation as GENERALITY:
`eval_noise_refine` (held-out map, env seed 10000, 100 trajectories, seed 1234 + s) at T = 1024 (primary) and 2048, and
`eval_rank_strata` at 1024 (checked against the former).

What the code does at 2 layers (read, `model_wrap_exist._AO.forward`, `model.MapFormerWM`, `model.WMTransformerLayer`): ONE step
map (`ActionToLieAlgebra`, rank 4 shared latent; |.| per channel in AO; the four move tokens step, every other token 0 by
construction) and ONE `PathIntegrator` (learned omega, geometric init over the 64 blocks, grid 64) compute the angles once; the
SAME cos/sin is applied to Q and K in EVERY layer (score Q^T R(dtheta) K in both layers). Layers do not get separate step maps;
layer 2 is a second content-attention layer over the same path phase. `--fast-attn` applies to both layers (both plain
`WMTransformerLayer`). Heads kept at 2 so per-head rank (the rank-4 latent read by each head) is unchanged from GENERALITY;
only width and depth change.

**Seeds** 1400-1407 (new block). At a seed every cell and arm shares the initial weights (AO differs from SO only by |.|) and
the training data stream; ring recipe cells differ only in the optimiser hyperparameter. Pilots and smoke on seeds outside
the block (1490-1496).

## Registered decisions (`analyze_robust.py`; labels as GAUGE G1-G2 / MECHANISM M4)
SOLVED = `stats_core.classify_run` (final 5% of training loss < 0.05). Fisher exact on SOLVED (`fisher_solved`), unpaired
permutation on accuracy (`perm2_p`, CI `perm2_ci`). For each of the five cells (ring|def, ring|wd0, ring|lrx3, ring|lrd3,
torus|d256L2), **primary: AO vs SO**, rule in this order:
- **UNREADABLE** if SO solves < 4/8 (the signed arm does not learn under that recipe / size within the budget; the cell says
  nothing about sign);
- **SIGNED LEARNS BETTER** if Fisher p < .05 with SO ahead, or permutation p < .05 with d = AO - SO <= -0.05;
- **NO SIGN COST** if AO SOLVED >= SO - 1 and mean AO accuracy >= mean SO - 0.01;
- **UNMEASURED** otherwise.

**Overall reading, fixed now** (`analyze_robust.reading`):
- **UNREADABLE** if the in-batch default ring cell is UNREADABLE (the reference itself failed; then nothing is read);
- **RECIPE-DEPENDENT** if any readable ring cell reads NO SIGN COST; **SCALE-DEPENDENT** if the torus cell does (both named if
  both); the cells are listed. This branch wins over the others;
- **ROBUST** if all five cells read SIGNED LEARNS BETTER;
- **INSUFFICIENT** if two or more cells are UNREADABLE (too few readable cells to speak of robustness);
- **ROBUST WHERE READABLE** if exactly one cell is UNREADABLE and every readable cell reads SIGNED LEARNS BETTER (the claim then
  carries the scope "where the signed arm learns at all");
- **PARTLY UNMEASURED** otherwise (some readable cell UNMEASURED, no NO SIGN COST).

What UNMEASURED means: at n = 8 the batch can neither show the cost nor its absence in that cell (e.g. AO 4-6/8 SOLVED with a
small accuracy gap). It is NOT evidence against the claim and NOT counted as a recipe dependence; the claim is not extended to
that recipe / size, and the paper says so.

Power (Fisher, n = 8 per arm): with SO 8/8, AO <= 3/8 reads SIGNED LEARNS BETTER (p 0.026; 4/8 gives 0.077); with SO 7/8, AO <=
2/8 (0.041); with SO 6/8, AO <= 1/8 (0.041); with SO 4-5/8 no AO count reaches p < .05 and only the accuracy route can fire. The
ring accuracy gap has been small (GAUGE G2 d -0.069; RING_DIAL ~ -0.03), so SOLVED carries most of the power on the ring.

**Predictions.** Overall ROBUST. Per cell: def SIGNED LEARNS BETTER (fresh-seed replication of GAUGE G2 / WRAP_EXISTENCE E3);
wd0 SIGNED LEARNS BETTER (ACCOUNTS T1: weight decay irrelevant to the held code); lrx3 and lrd3 SIGNED LEARNS BETTER, each at
risk (the pilot's AO at lr x3 was still DESCENDING at epoch 300, tail loss 0.19, so more AO runs may cross 0.05 than at the
default lr; at lr / 3 the signed arm must still solve within 300 epochs -- the pilot's did by epoch 35); torus SIGNED LEARNS
BETTER (GENERALITY E3 0/8 vs 8/8 at 1 layer d 128; wider and deeper models could in principle find the wrap).

**Declared secondaries (no verdict):** accuracy at 4T on the ring (512) and 2T on the torus (2048; GENERALITY's OOD length --
4T = 4096 was not chosen, see Deviations); loss AUC (mean per-epoch training loss); run classes (SOLVED / STALLED / DESCENDING /
RISING); SOLVED at a loss threshold of 0.1 with its Fisher p (threshold sensitivity, `READ_draft4.md` (e) weak point 3); torus
strata (plain lag < 128, >= 128, wrap-only) from `eval_rank_strata`; the wrap-aware cancelling share (`wrap_share`, ring pair
(+1, -1), torus pairs (0,1), (2,3)) and the opposite-sign share (wrapped angles, |angle| > 0.05; GAUGE's same-sign readout,
complemented); AO and SO SOLVED counts across the four recipes at matched seeds (descriptive).

Floors: predict-blank ~0.50 (p_empty 0.5); CANCEL's measured +1/-1 ring floor 0.554; torus always-blank 0.512.

## What this can and cannot show
It tests three one-factor perturbations of two AdamW hyperparameters (no lr x weight-decay interaction, no other optimiser family,
no other schedule) on the ring at the default model size, and one larger model (2 layers, d 256) on the torus at the default
recipe. It does not test scale on the ring, recipe on the torus, or more heads. Every verdict is budget-scoped (300 / 900
epochs). The eval-mode accuracy under-reports one-layer runs below ceiling (attention-dropout scale, `DROPOUT_RESCORE.md`); the
correction is not valid for 2-layer models, so no re-score is declared; both arms of a cell are read the same way and the SOLVED
contrast does not depend on it.

## Process, code and cost
- Trainer change: `train_gauge.py` gains `--weight-decay` (default 0.05 = `train()`'s default, passed through to `train()`); no
  other change. **Bitwise default check** (seed 1490, SO and AO, deltas +1/-1, 3 epochs x 5 batches, cuda:0): HEAD vs edited
  file without the flag -- per-epoch losses identical, every state-dict tensor `torch.equal`, eval.json identical, both arms.
  `--weight-decay 0` changes the run (AO losses differ from epoch 1, at ~1e-6 relative after 5 steps), so the flag takes
  effect.
- Driver `run_robust.sh` (sources `lib_driver.sh`): `flock` lock; refuses a dirty tree (tracked files) and uncommitted
  `analyze_robust.py` / `run_robust.sh` / `train_gauge.py` BEFORE any wait; md5 guard over every `mapformer` module the trainers,
  evaluators and analysis import (computed from `sys.modules`: 96 modules, `data_parallel.py` among them because the
  list-builder imports it explicitly -- `train()` imports it lazily) plus `lib_driver.sh` and `run_robust.sh`, re-checked before the evaluation; torus first at ONE
  run per GPU (`drv_wait_slot 1`), then ring at <= 6 trainers per GPU (`drv_wait_slot 6`, counting the torus runs);
  artifacts-then-marker (`.robust_done` after `ROBUST_ANALYSIS.txt` and `ROBUST_VERDICTS.json`, 144 training artifacts
  required). Dry run (`DRV_DRYRUN=1`, a committed copy of the repo in scratch): 80 launches in the stated order (torus seed-outer
  SO, AO; then ring seed-outer, recipe, arm), no duplicates, guard over 98 files, fails closed at "missing runs".
- Smoke (scratch, seeds 1495-1496, the driver's exact command lines with 2 epochs x 2 batches): all 16 cell/arm combinations
  train and evaluate; torus evals and strata run (strata 'all' matches eval_noise_refine); `analyze_robust` reads the output end
  to end (UNREADABLE everywhere at n = 2, as it must). Every label and reading reached: `--selftest` (synthetic accuracy / SOLVED
  vectors through the same functions) and two synthetic run trees built from the smoke checkpoints (8 fake seeds, rewritten
  losses and accuracies) read through the real file path: SIGNED LEARNS BETTER / NO SIGN COST / UNMEASURED / UNREADABLE ->
  RECIPE-DEPENDENT, and all SIGNED LEARNS BETTER -> ROBUST.
- **Measured cost** (both GPUs RTX 4090, idle, no other users on 2026-10-16 02:30):
  - torus d 256, 2 layers: 3.6 s/epoch alone (GPU 99%, 2.2 GB); 16.0 s/epoch each at 4 per GPU (worse than serial); 11.9 s/epoch
    beside 2 ring runs. GPU-bound, so ONE torus run per GPU: 900 x 3.6 s = 54 min per run, 8 waves of 2 = **~7.2 h**;
  - ring: 2.0 s/epoch at 2 per GPU (GAUGE: ~5-6 s/epoch at 6 per GPU; 40 runs in 1 h 40 min), ~1.0 GPU-s per ring-epoch at any
    concurrency: 64 x 300 x 1.0 s / 2 GPUs = **~2.7 h**;
  - torus evals and analysis ~15 min. **Total ~10 h wall on GPUs 0 and 1.**
  - Trimmed alternatives (measured, alone): torus at 2 layers d 128 1.8 s/epoch (~3.6 h for R2, total ~6.5 h) or 1 layer d 256
    2.0 s/epoch (~4.0 h, total ~7 h). Not chosen: a single d 256 2-layer run takes 54 min (< 2 h), and only the brief's
    configuration changes both width and depth.

## Pilot (outside seeds, never pooled)
- Ring, seed 1490, deltas +1/-1, 300 epochs:
  - lr x3 (3e-3): SO SOLVED (loss < 0.05 from epoch 33; acc 1.000 at T, 0.975 at 4T; wrap-cancel share 1.00); AO DESCENDING
    (tail loss 0.194, acc 0.968 / 0.867; cancel share 0.10).
  - lr / 3 (3.33e-4; added to the brief's pilot because it carries the UNREADABLE risk): SO SOLVED (from epoch 35; acc 1.000 /
    0.999); AO STALLED (tail 0.411, acc 0.888 / 0.844; cancel share 0.05).
  The signed arm learns under both perturbed learning rates; the non-negative arm does not within 300 epochs. One seed per
  cell: no estimate.
- Torus timing, d 256, 2 layers, 10-epoch schedules (seeds 1490-1492; losses at a 10-epoch cosine schedule are not comparable
  with 900 epochs and are not reported as results).

## Deviations from the brief (each with its reason)
1. Torus OOD accuracy at 2T = 2048, not 4T = 4096: GENERALITY's evaluated length, so the d 256 numbers sit beside the d 128 ones.
2. Pilot extended by an lr / 3 ring pair (seed 1490): cheap (~10 min), and it is the cell most at risk of UNREADABLE.
3. Torus at one run per GPU (not "fewer than ring" in general): measured GPU-bound; concurrency lowers throughput.
4. Readings INSUFFICIENT and ROBUST WHERE READABLE added to the brief's ROBUST / RECIPE-DEPENDENT, and SCALE-DEPENDENT split from
   RECIPE-DEPENDENT, so every combination of cell labels maps to exactly one reading.
