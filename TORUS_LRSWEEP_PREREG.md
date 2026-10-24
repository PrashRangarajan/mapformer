# Is the sign learnability cost present on the T = 1024 torus when each step parameterisation is at its own best learning rate? -- pre-registration (G-4, 2026-10-23, before any batch result)

## 0. Question
LRSWEEP (`LRSWEEP_PREREG.md` + Amendment 1, registered) answered the learning-rate objection on the ring only: each arm at
its own best of four rates, the |.| step solves 2/8 (best rate 3e-3, bracketed) against the signed step's 8/8 (1e-3),
reading **COST AT BEST RATES**. On the torus, where the paper's accuracy claims live, every sign contrast compared the two
parameterisations at ONE rate, the signed model's default 1e-3: GENERALITY E3 (observation-step-free, this recipe) |.| 0/8
(0.880) vs signed 8/8 (1.000); ROBUST torus (2 layers, d 256) 2/8 vs 8/8, -0.032, with 6/8 |.| runs still descending. A
parameterisation change is an optimiser change (rule 15), so a one-rate torus contrast is not neutral. The draft-2 plan
(`docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md` G-4) and the runbook (`docs/RUNBOOK_NEXT_EXPERIMENTS.md` Sec. 3) list
this as the limitation every reviewer raised; the paper currently says "on a ring" for the best-rate result.

## 1. Design
Torus, GENERALITY's torus recipe unchanged (`run_generality.sh` `torus`): `train_variant`, 64 x 64 torus, 16 object types,
p_empty 0.5, T = 1024 moves (2048 tokens), batch 16, 98 batches per epoch, 900 epochs, cosine schedule (5% linear warm-up,
cosine to 0.1 x lr), 1 layer, 2 heads, d 128, AdamW, gradient clip 1.0, `--fast-attn`, `--data-workers 3`, OMP 2.
Arms SO / AO = `SignedAO4_r4` / `AbsAO4_r4` (observation-step-free: only the four move tokens step; AO = |SO| per channel
after W_out, before omega).

| rate | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|
| role | GENERALITY E3 / ROBUST default (in-batch fresh-seed replication) | the ring's |.| optimum (LRSWEEP) | the ring's upper useful rate (signed 4/8 there; 3e-2 broke both ring arms) |

2 arms x 3 rates x 8 seeds = **48 runs**. Seeds **2400-2407**: no run directory, driver or results file uses 2400-2499
(checked 2026-10-23: `find runs -maxdepth 4 -name '*s24[0-9][0-9]*'` empty; no driver passes such a seed; the only "24xx"
hits in `*.sh` are buffer sizes). Training streams (`ParallelBatchGenerator`, batch seed `s * 1000003 + i`, i < 88,200 per
run) cannot overlap across seeds 2400-2407, the pilot seeds 2490-2491, the timing seed 2490 or the smoke seeds 2495-2496.
At a seed every rate and arm shares the initial weights (checked: `SignedAO4_r4` and `AbsAO4_r4` built after
`torch.manual_seed(2400)` have identical state dicts) and the training stream (both are fixed before `train()`, which draws
nothing that depends on lr); AO differs from SO only by |.|, rates only by lr. No trainer flag is added: `--lr` and
`--weight-decay` exist, and `--weight-decay 0.05` is train()'s default, so the 1e-3 cell is GENERALITY's command line with
the default written out (same code path; no bitwise check needed because no code changes).
Evaluation (after training, as GENERALITY): `eval_noise_refine` at T = 1024 (primary accuracy) and 2048, 100 walks, held-out
map (env seed 10000, walk seed 1234 + s), and `eval_rank_strata` at T = 1024 (revisit strata).

**Weight decay.** Held at 0.05, as a standard learning-rate sweep and LRSWEEP do. AdamW's decoupled decay is lr_t x wd per
step, so the total decay exponent over the run is sum_t lr_t x wd = **2415 x lr** for this schedule (2.4 / 7.2 / 24.1 at
1e-3 / 3e-3 / 1e-2; 72 at 3e-2), three times the ring's 805 x lr (900 vs 300 epochs). A rate here therefore means "the recipe
at that lr", decay included (LRSWEEP Amendment 1 MINOR 1 / MAJOR 2). Decay matters on this task: RANKWD (registered) found
shared rank 4 needs AdamW decay to learn the T = 1024 torus within 900 epochs (8/8 vs 1/8 at wd 0), and decay shrinks omega
to ~0.15 there. Whether decay limits |.| at the high rates is handled by the conditional stage 2 (Sec. 3); no wd-0 torus cell
exists for the observation-step-free arms (ROBUST's wd-0 control is ring-only).

## 2. Registered decisions (`analyze_torus_lrsweep.py`; machinery imported unchanged from `analyze_lrsweep.py`)
SOLVED = `stats_core.classify_run` (final-5% mean training loss < 0.05). A non-finite loss is DIVERGED (not SOLVED, kept,
accuracy 0 -- below the 0.506 blank floor; declared). Accuracy = held-out accuracy at T = 1024 (non-finite counts 0).
**UNSTABLE** (new; G-3 Amendment 1.4 in torus form plus a collapse clause): RUNAWAY = any epoch after the 5% warm-up with
training loss above the memoryless 2.08 nats (marginal revisit-observation entropy: 0.5 blank + 0.5 over 16 types), or a
final move step |W_out W_in e(move)| above 1e3; or COLLAPSED = final-5% mean loss above 1.9 (the timing pilot's |.| run at
1e-2 sat at 2.03-2.04 after warm-up, below 2.08). Counted per cell and marked "!" per run.

**Primary B1, best vs best** (LRSWEEP's rule, unchanged):
1. *Selection* (training-side only): each arm's best rate = most SOLVED (of 8); ties -> lower mean selection score
   (final-5% mean training loss, capped at the first-epoch loss; a non-finite tail scores the first-epoch loss); then the
   lower rate. Held-out accuracy is never used to select.
2. *Contrast*: AO@best vs SO@best with the ROBUST cell rule (`label_from` = `analyze_robust.cell_label`, checked equal in the
   selftest), in order: **UNREADABLE** if SO@best solves < 4/8 (gate on the signed arm: it learns at no rate); **SIGNED
   LEARNS BETTER** if p_SOLVED < .05 with SO ahead, or p_acc < .05 with d = AO - SO <= -0.05; **NON-NEGATIVE LEARNS BETTER**
   the mirror; **NO SIGN COST** if SO@best >= 6/8, AO@best >= SO@best - 1 and mean AO accuracy >= SO - 0.01; else
   **UNMEASURED**.
3. *Selection bias*: picking each arm's best of three rates on the seeds then compared inflates both best values. The test
   redoes the selection inside the null: the 16 per-seed rate profiles (8 SO, 8 AO; one run per rate) are split into two
   groups of 8 in all C(16, 8) = 12,870 ways, each group's best rate re-selected by rule 1, the statistic recomputed at the
   re-selected rates (T_k = SOLVED difference, T_acc = mean accuracy difference). **Decision p per route = max(selection-aware
   p, naive p at the selected rates)** (naive = Fisher on SOLVED, unpaired permutation on accuracy), so the verdict never
   rests on a p smaller than the one obtained by pretending the selected rates were fixed in advance (LRSWEEP Amendment 1
   MINOR 2: with SO at ceiling the decision p reduces to Fisher at the selected rates; unpaired although the arms share
   initial weights and stream at a seed -- valid, conservative under positive within-seed correlation). A held-out-seed
   split is a declared secondary (cross-fit), not the primary: at n = 4 per arm Fisher reaches p < .05 only at 4/4 vs 0/4.

**Reading, fixed now** (`analyze_torus_lrsweep.reading`), with what it means for draft 2 (`docs/paper/MAIN_DRAFT1.md` ->
`SYNTHESIS_DRAFT2_PLAN.md` Sec. 2.2: "on a ring" in the best-rate sentence; the evidence table's lr column):
- **COST AT BEST RATES** (B1 = SIGNED LEARNS BETTER): the best-rate result extends to the torus; "on a ring" becomes "on the
  ring and the T = 1024 torus, at each parameterisation's best of three learning rates (1e-3 to 1e-2)". If AO's selected rate
  is >= 1e-2 the reading carries "[STAGE 2 TRIGGERED]" and is final only after stage 2 (Sec. 3).
- **COST CLOSES AT HIGHER LR** (B1 = NO SIGN COST or NON-NEGATIVE LEARNS BETTER, AO's best rate > the LOWEST rate at which SO
  reaches its best SOLVED count): on the torus the barrier learns the map at its own larger rate; the torus claim becomes a
  learning-rate-sensitivity (conditioning) result and the paper says the best-rate cost is ring-only. "(reversed)" is
  appended for NON-NEGATIVE LEARNS BETTER; NO SIGN COST is worded "cost not detectable at AO's best tested rate
  (selection-inflated)".
- **NO COST AT BEST RATES** (same labels, AO's best rate <= SO's lowest best rate): the torus cost does not replicate even at
  the shared rate; "contradicts GENERALITY E3" is printed only if the naive per-rate 1e-3 contrast also reads NO SIGN COST.
- **UNMEASURED**: the paper may not claim the torus cost is robust to the rate; it says the cost "may shrink with the rate"
  only when AO's best rate is above SO's lowest best rate.
- **UNREADABLE** (SO < 4/8 at every rate): nothing is read.
- Qualifiers on any reading: "[AO's best rate is the grid's top rate: not bracketed; 'best' = best TESTED]";
  "[k AO run(s) still DESCENDING at AO's best rate: budget-scoped]"; and, new, "[k AO runs UNSTABLE at AO's best rate: no
  learnability sentence rests on this rate]" when k >= 2.

## 3. Conditional stage 2 (registered now; LRSWEEP Amendment 1 MAJOR 2, torus form)
IF B1 reads SIGNED LEARNS BETTER AND AO's selected rate is >= 1e-2, run SO and AO at AO's selected rate with
**wd = 5e-5 / lr** (lr x wd = the default's; decay exponent 2.4 as at 1e-3), seeds 2400-2407 (16 runs; `MODE=stage2` of the
driver, which refuses unless `TORUS_LRSWEEP_VERDICTS.json` says `stage2_triggered`, and requires the batch's md5 record),
evaluated as above, labelled by the ROBUST rule (naive Fisher / permutation): SIGNED LEARNS BETTER -> the reading stands;
NO SIGN COST or NON-NEGATIVE LEARNS BETTER -> "COST AT BEST RATES IS COUPLED DECAY"; UNMEASURED or UNREADABLE -> the reading
is scoped to "AdamW wd 0.05 coupled to lr". Stage 2 gets its own short independent check before launch. If AO's selected
rate is <= 3e-3 its decay exponent is <= 7.2 (3x the default's) and no decay control is run; this is stated as a limit
(the ROBUST wd-0 control is ring-only).

## 4. Predictions (written after the timing pilot below, before any full-length torus run at these seeds)
- SO solves 8/8 at 1e-3 (replicates GENERALITY E3) and most likely 7-8/8 at 3e-3; at 1e-2 uncertain (ring: 4/8; the timing
  pilot's signed run was at loss 0.41 at epoch 70, healthy).
- AO: 0-1/8 at 1e-3 (GENERALITY 0/8); its best rate most likely 3e-3 with <= 3/8; 1e-2 is unlikely to be its best (timing
  pilot, one seed, 70 epochs: |.| collapsed to the memoryless loss 2.03-2.04 after warm-up).
- Hence **COST AT BEST RATES most likely** (~0.6), **UNMEASURED** next (~0.3, AO 4-6/8 at 3e-3: ROBUST's larger torus model
  had 2/8 solved and 6/8 still descending at 1e-3, so the torus |.| runs are slow rather than stuck), **COST CLOSES AT HIGHER
  LR** least likely (~0.1). Stage 2 unlikely to trigger.

## 5. Power (`analyze_torus_lrsweep --power`, synthetic)
- Deterministic profiles, SO 8/8 at its best: the cost label fires on SOLVED when AO@best <= 3/8 (decision p 0.0256);
  AO@best 4/8 gives 0.077 (UNMEASURED unless the accuracy route fires); AO@best 8/8 with accuracy within 0.01 reads NO SIGN
  COST; 5-7/8 read UNMEASURED. Same whether SO's third rate solves 8, 4 or 0 (the selection-aware p equals Fisher here).
- Simulated seeds (400 replicates; SO true solve rates 1.0 / 1.0 / 0.5; AO true rate p at 3e-3 and p/2 elsewhere; unsolved
  accuracy ~ N(0.88, 0.07)): SIGNED LEARNS BETTER fires 0.99 / 0.94 / 0.76 / 0.29 at p = 0.1 / 0.2 / 0.3 / 0.5; at p = 0.75
  UNMEASURED 0.79, NO SIGN COST 0.20; at p = 1.0 NO SIGN COST 1.00.
- Size: both arms at true rate 0.75 at every rate: SIGNED LEARNS BETTER 0/400 (NO SIGN COST 0.68, UNMEASURED 0.32).
- So the design detects a torus cost of ring size (true |.| best rate <= 0.3) with >= 76% power and cannot call a cost
  "absent" unless |.| solves at least 7/8 at its best rate.

## 6. Declared secondaries (no verdict, no reading depends on them)
- Per-rate AO vs SO at each rate, ROBUST rule on Holm-adjusted p (Holm over the 3 rates, separately for the Fisher and the
  permutation family), raw p beside. The 1e-3 cell is the fresh-seed replication of GENERALITY E3.
- Threshold sensitivity: B1 redone with SOLVED at loss 0.1 (selection redone, same max rule).
- Accuracy at T = 2048 (2x the training length; robustness, not capability, rule 10) at the B1-selected rates
  (selection-aware p).
- Revisit strata at T = 1024 (`eval_rank_strata`: plain lag < 128, plain lag >= 128, wrap-only), per rate and arm.
- Wrap-cancel share (`model_wrap_exist.wrap_share`, wrap-aware, pairs (0, 1), (2, 3)), the usage-weighted share
  (`wrap_share_weighted` x `usage_weights` on fig_wrap_share's torus stream: GridWorld(64, 16, 0.5, seed 10000), np seed 1234,
  32 walks of T = 1024) and the opposite-sign share (`analyze_robust.opp_share`).
- The clock / map ratio with the audit-fixed definition (`docs/paper/expansion/scripts/sign/clock_table.py`
  `ratios`, loaded from its file: per inverse pair, clock wrap(a_x + a_y), symmetric map, usage-weighted, one ratio per run
  and per head; ANALYSIS_SCRIPT_AUDIT B3 / M-SG1), split SOLVED vs unsolved, and the near-pi usage share. The selftest checks
  its move angles equal the model's own (`action_angles`) for both arms.
- Loss AUC (mean per-epoch training loss), mean selection score, run classes (SOLVED / STALLED / DESCENDING / RISING /
  DIVERGED) and UNSTABLE counts per cell; the SOLVED-vs-lr curve per arm.
- Cross-fit: rates selected on seeds 2400-2403 and tested on 2404-2407, and the reverse (Fisher / permutation, n = 4).
- Cross-batch references printed beside, labelled, never pooled (rule 12): GENERALITY E3, ROBUST torus, LRSWEEP ring.

Floors (T = 1024 eval stream, post hoc, `docs/audits/2026-10-20/WHAT_WHERE_T1024.md`): blank 0.506, best n-gram 0.575,
retrace 0.843. The eval-mode accuracy under-reports one-layer runs below ceiling (`DROPOUT_RESCORE.md`); no re-score is
declared; both arms are read the same way and SOLVED does not depend on it.

## 7. What this can and cannot show
One recipe axis (AdamW peak lr with its coupled decay and cosine floor) over one decade, on the T = 1024 torus at the
default size (1 layer, d 128), observation-step-free arms, one budget (900 epochs), n = 8. Not tested: other optimisers,
schedules, warm-up lengths, lr x wd beyond the conditional stage 2, larger models (ROBUST covers one at 1e-3), rank 2 (the
plan's optional r2 vs r4 extension is not part of this batch). "Best rate" means best of the rates tested; every verdict is
budget-scoped.

## 8. Pilots and the grid-extension rule (fixed now)
**Timing pilot (done, kalman GPU 1, shared with another user's job at 100% utilisation; seed 2490, outside the batch, never
pooled; 2026-10-23 22:09-22:14).** The driver's command line at lr 1e-2 for both arms, `--epochs 900`, stopped at epoch 70
(process group killed). Both arms 3.7 s/epoch with two of ours running beside the foreign job; ~1.0 GB each. Losses
(descriptive, one seed, warm-up ends at epoch 45): SO 1.57 (epoch 5) -> 1.12 (20) -> 0.46 (60) -> 0.41 (70); AO 1.86 (5) ->
1.96 (15) -> 2.05 (55) -> 2.03 (70), i.e. |.| at 1e-2 sat at the memoryless level after the peak rate was reached. This
motivated the COLLAPSED clause of UNSTABLE (Sec. 2) and enters the predictions (Sec. 4); it changes nothing else.
**Extension pilot (to run on koopman first: `MODE=pilot DRV_GPUS=1 DRV_MAXPG=2`).** AO at 1e-2 and 3e-2, seeds 2490-2491,
full 900 epochs, the driver's command line (4 runs, ~1.9 h at 2 jobs). Rule (`analyze_torus_lrsweep.pilot_rule`): if the
pilot AO is better at 3e-2 than at 1e-2 (more SOLVED over the two seeds; equal SOLVED -> lower mean selection score), AO may
still be improving at the top rate and **3e-2 is added** (64 runs) by an amendment before launch that edits `RATES` in the
driver; otherwise the grid stays {1e-3, 3e-3, 1e-2}. The signed arm's behaviour does not enter the rule. If the user skips
the pilot, the batch runs as registered and a top-of-grid AO best carries the "not bracketed" qualifier.
An independent code-verification agent, blind to results, checks this file, the driver and the analysis before launch
(rule 29); its findings go into Amendment 1.

## 9. Process, code and cost
- `analyze_torus_lrsweep.py` (new): imports `label_from`, `select`, `b1_test`, `b1_decide`, `holm`, `sel_score` from
  `analyze_lrsweep` (unchanged since LRSWEEP's audited run) and `opp_share`, `cell_label` from `analyze_robust`; torus loader
  asserting the recipe per checkpoint (variant, seed, lr, wd, epochs, len(losses), T, batch, n_batches, size, heads, grid,
  objects, landmarks, schedule, init_from, data_seed_offset, fast_attn, deterministic, data_workers, environment knobs,
  snapshot_every); verdict JSON written before the secondaries; `--selftest` (every B1 label, every reading, every qualifier
  incl. STAGE 2 TRIGGERED, every stage-2 reading, the pilot rule, the UNSTABLE rule, tie-breaks, the one-rate null = Fisher,
  `label_from` = `cell_label`, clock_table angles = model angles, the decay exponent reproducing LRSWEEP's 805 x lr on the
  ring); `--power`; `--pilot`; `--stage2`.
- `run_torus_lrsweep.sh` (new; sources `lib_driver.sh`): refuses to start unless `DRV_GPUS` and `DRV_MAXPG` are given at
  launch (no defaults; MAXPG 1-4), assigns them after sourcing (rule 27), exports `CUDA_DEVICE_ORDER=PCI_BUS_ID` (torch's
  cuda:N = nvidia-smi's N); `flock` lock shared by all modes; before any wait, refuses modified tracked `*.py` / `*.sh` or a
  modified prereg (other docs may be modified) and any uncommitted G-4 file; checks every directory tag against
  `rate_tag`; md5 guard over every imported `mapformer` module (from `sys.modules`, lazily imported `data_parallel` and
  `ckpt_guard` explicit) plus `clock_table.py`, `lib_driver.sh` and the driver, re-checked before evaluation (stage 2 reuses
  the batch's record); seed outer, rate, arm inner; `drv_wait_slot $DRV_MAXPG` (counts every `mapformer.train_*` of the user
  on the GPU); evals after training (strata failures logged, not fatal: a declared secondary); artifacts-then-marker
  (`.torus_lrsweep_done` after the analysis, the verdict JSON and the six accuracy JSONs).
- Layout: `runs/torus_lrsweep/<lr tag>/<SO|AO>/p0/<V>_s<S>/<V>.pt`, evals `runs/torus_lrsweep/eval/TORUS_LRSWEEP_<tag>_<arm>
  .{md,json}` and `_STRATA.json`; analysis `TORUS_LRSWEEP_ANALYSIS.txt`, `TORUS_LRSWEEP_VERDICTS.json`.
- **Cost.** koopman (one unshared RTX 4090, GPU 1, 2 jobs): RANKWD measured the T = 1024 torus at 2.0 s/epoch alone and 4.2
  s/epoch with two sharing (no packing gain; without `--fast-attn`); here 3.7 s/epoch each with two beside a foreign job. Per
  run 900 x (3.3-4.2) s = 50-63 min; 48 runs in 24 pairs = **~20-25 h** (upper end if `--fast-attn` buys nothing on koopman),
  plus evals ~0.5 h and the analysis ~5 min CPU. Extension pilot ~1.9 h first; a 3e-2 extension adds ~7-8 h; stage 2 (if
  triggered) ~7-8 h. The plan's 12.5-15 h assumed two GPUs.

## 10. Smoke (scratch `torus_lrsweep_smoke/`, CPU, seeds 2495-2496; 2026-10-23)
- The driver's exact `train_variant` command line with `--epochs 2 --n-batches 2 --device cpu` for every arm at 1e-3, 3e-3,
  1e-2 and 3e-2 (16 runs, `smoke.sh`): all train and save. The driver's `eval_noise_refine` (T 1024 / 2048) and
  `eval_rank_strata` (`--check-json`) command lines at `--n-trials 3 --device cpu`: all pass. `analyze_torus_lrsweep` reads
  the tree end to end (B1 UNREADABLE at n = 2 with 2-epoch runs, as it must; every secondary prints, incl. strata, both
  wrap-cancel shares, the opposite-sign share and the clock / map ratio; 2-epoch runs are flagged UNSTABLE, as they should
  be). `--pilot` on the smoke's 1e-2 / 3e-2 AO runs prints the rule's decision.
- `--selftest` passes (Sec. 9 list). `--power` gives Sec. 5.
- Synthetic run trees (`synth_tree.py`: the smoke checkpoints copied to fake seeds 2400-2407, losses / seeds / accuracies
  rewritten) read through the real file path: COST AT BEST RATES; COST AT BEST RATES with STAGE 2 TRIGGERED and the edge
  qualifier; COST CLOSES AT HIGHER LR; NO COST AT BEST RATES ("contradicts GENERALITY E3"); UNMEASURED; UNREADABLE (one SO
  run with a NaN loss classed DIVERGED, accuracy 0, kept). Missing strata print "strata unavailable". `--stage2` on synthetic
  stage-2 trees (wd 5e-5 / lr in the checkpoint) reaches "stands", "COUPLED DECAY" and "scoped (UNMEASURED)", and refuses
  when the verdict JSON does not say `stage2_triggered`.
- Driver (`git clone --shared` of HEAD with the three G-4 files committed in the scratch clone): refuses without DRV_GPUS /
  DRV_MAXPG, with DRV_MAXPG 9, with a modified tracked `.py` (a modified tracked doc passes through), and stage 2 without the
  batch's md5 record or without a trigger. `DRV_DRYRUN=1 DRV_GPUS=1 DRV_MAXPG=2`: md5 guard over 102 files; 48 launches,
  seed outer / rate / arm inner, no duplicate output dir, every command line as Sec. 1 on cuda:1; then fails closed
  ("MISSING" x 48, "FAILED: missing runs -- done marker NOT set"). `MODE=pilot`: 4 launches (AO, 1e-2 / 3e-2, s2490-2491),
  fails closed. `MODE=stage2` with a triggered verdict at 1e-2: 16 launches at `--weight-decay 0.005`, fails closed.

## Amendment 1 (2026-10-24, after the independent check `docs/audits/2026-10-24/g4_torus_lrsweep_audit.md`; before any batch run, no batch result exists)
The check found the registered machinery (LRSWEEP's, imported unchanged) correctly applied to three rates, the command lines
identical to GENERALITY's torus recipe except lr, lr and decay reaching the optimiser (decay exponent 2414.5 x lr), shared
init and stream per seed, `--fast-attn` valid for both classes, the analysis inputs as read, and the driver sound. Verdict:
launchable after this amendment. This amendment supersedes the body where they differ. Changes:
- **MAJOR 1 (decay in the closing branch).** Stage 2 (SO / AO at AO's selected rate, wd = 5e-5 / lr, seeds 2400-2407, 16
  runs) now triggers whenever AO's selected rate is >= 1e-2 and B1 reads SIGNED LEARNS BETTER, NO SIGN COST or NON-NEGATIVE
  LEARNS BETTER, i.e. whenever a reading rests on AO's high-rate cell (RANKWD shows decay can help on this task, so a closing
  cost at 1e-2 may be the 10x coupled decay, not the rate). `stage2_kind` records which reading stage 2 qualifies:
  - COST (B1 SLB): unchanged -- stands / "COST AT BEST RATES IS COUPLED DECAY" / scoped to "AdamW wd 0.05 coupled to lr".
  - CLOSES (NSC / NLB, AO's rate above SO's lowest best): stage 2 NSC or NLB -> "COST CLOSES AT HIGHER LR (holds at matched
    decay)"; SLB -> "COST CLOSES WITH COUPLED DECAY (lr alone does not close it)"; UNMEASURED / UNREADABLE -> "COST CLOSES AT
    HIGHER LR with its coupled decay; lr and decay not separated".
  - NOCOST (NSC / NLB at AO's rate <= SO's lowest best, only possible if SO's best count is first reached at >= 1e-2): the
    same three forms with "NO COST AT BEST RATES".
  Until stage 2 reports, COST CLOSES and UNMEASURED's clause read "with the rate and its coupled decay". UNMEASURED does not
  trigger stage 2 (no reading rests on AO's cell there).
- **MAJOR 2 (UNSTABLE).** RUNAWAY's loss clause is now: an epoch after the 5% warm-up above **H_map(seed) + 0.05**, H_map =
  the entropy of the seed's own training map (blank + 16 types, `GridWorld(64, 16, 0.5, seed).obs_map`, built as
  train_variant builds it; 2.061 / 2.073 / 2.102 / 2.051 / 2.054 / 2.070 / 2.094 / 2.086 on seeds 2400-2407, checked equal to
  the audit's values in the selftest). The fixed 2.08 sat inside this per-seed spread. The step clause (> 1e3) and COLLAPSED
  (final-5% mean loss > 1.9) are unchanged. The reading's qualifier counts only UNSTABLE runs that are NOT SOLVED; every
  flagged run prints its max post-warm-up loss minus H_map, H_map, its final-5% loss, step maximum and class.
- **MINOR 1 (pilot rule).** EXTEND iff the pilot AO solves more at 3e-2 than at 1e-2, or solves as many with a mean selection
  score lower by >= 0.05 AND no 3e-2 pilot run UNSTABLE (so two collapsed rates cannot extend the grid by a coin flip). Pilot
  seeds are now **2492-2493** (unused; no pilot run repeats the partly observed timing run on 2490).
- **MINOR 2.** The model-derived secondaries (usage stream, clock_table, wrap / opposite-sign / clock shares) are wrapped in
  try / except -> NaN plus a logged warning, so they can never block the verdict (tested by forcing a failure).
- **MINOR 3.** The driver re-checks the md5 guard before EVERY launch (`drv_fail` on a change). Runbook: no `git pull` /
  checkout on koopman while any mode of this driver runs.
- **MINOR 4 (stage-2 md5 policy).** `MODE=stage2` first verifies every training module (all `mapformer` modules imported by
  `train_variant` + `data_parallel`, 89 files) against the batch's `runs/torus_lrsweep/code_md5.txt` and refuses on any
  change; it then keeps its own md5 record under `runs/torus_lrsweep/stage2`, so fixes from stage 2's own check may touch the
  analysis, the evaluators or the driver, never the training path. Dry-run tested: an analysis edit passes, a `train.py`
  edit is refused.
- **MINOR 5.** `pilot()` applies the batch's recipe asserts (`check_ckpt`: variant, seed, lr, wd, epochs, T, batch,
  n_batches, size, heads, grid, objects, landmarks, schedule, init_from, data_seed_offset, fast_attn, deterministic,
  data_workers, environment knobs).
- **MINOR 6.** n_trials (100), the eval env seed (10000) and the walk seed (1234 + s) are fixed by the driver's command
  lines only; the eval JSON does not record them and the evaluator (a guarded module) is not changed to.
- **MINOR 7 (cross-references).** Pilot = Sec. 8, stage 2 = Sec. 3, UNSTABLE = Sec. 2 (fixed in the driver and analysis).
- **MINOR 8.** Sec. 1's "no code changes" is replaced by: train.py, train_variant.py and model_wrap_exist.py changed since
  GENERALITY (47a766b / 62334d9); the changes are additive (an `epoch_callback=None` default tested bitwise, `--weight-decay`
  default 0.05 = train()'s old default, `--snapshot-every` off, `freeze_latent` read with a False default, new eval-only
  functions) and leave this path unchanged (audit item 2). No trainer flag is added by G-4.
- **MINOR 9.** DRV_MAXPG stays 2 on koopman: RANKWD's timing (2.0 s/epoch alone, 4.2 with two) puts packing ~5% slower than
  serial (~25.2 h vs ~24.0 h without an SDPA gain); accepted for consistency with RANKCLOCK / RANKWD.
- **Incident during the check (recorded).** A driver test in the checker's scratch `git clone --shared` was run once without
  `DRV_DRYRUN=1` and launched two real 900-epoch jobs (SO / AO, seed 2400, lr 1e-3) on kalman GPU 1 at 22:43 on 2026-10-23;
  both were killed by PID within ~2 min with their data workers, no orphans, and the clone and its `runs/` deleted. Nothing
  was written under `/home/prashr/mapformer/runs/` and no checkpoint, log or loss of those runs was kept or read, so seed 2400
  is unaffected (a run is a fresh computation from the seed; no information about its outcome exists).
- Tests after the changes: `--selftest` passes (every B1 label and reading; STAGE 2 TRIGGERED on the SLB edge case and on
  both COST CLOSES cases, not on UNMEASURED / UNREADABLE / NO COST at 1e-3; every stage-2 reading of the three kinds; the
  new pilot rule; the H_map RUNAWAY rule and COLLAPSED; H_map = the audit's values; tie-breaks; one-rate null = Fisher;
  clock_table angles = model angles; decay exponents). `--power` unchanged (Sec. 5). Synthetic trees through the file path:
  COST AT BEST RATES; COST with STAGE 2 TRIGGERED; COST CLOSES with STAGE 2 TRIGGERED (`stage2_kind` CLOSES, wd 0.005);
  stage 2 on it reaching "holds at matched decay" / "COST CLOSES WITH COUPLED DECAY" / "not separated"; NO COST; UNMEASURED;
  UNREADABLE with a DIVERGED run. Driver dry runs (`DRV_DRYRUN=1 DRV_GPUS=1 DRV_MAXPG=2`, scratch clone): pilot 4 launches
  (AO, 1e-2 / 3e-2, s2492-2493), batch 48 (seed / rate / arm order, no duplicates), stage 2 16 at wd 0.005 with the training
  check passing; each fails closed; stage 2 refused after a training-module edit; the batch refused after a guarded edit.
- Launch (koopman, after commit; GPU 1, two jobs): pilot first,
  `cd /home/prashr/mapformer && MODE=pilot DRV_GPUS=1 DRV_MAXPG=2 setsid nohup bash run_torus_lrsweep.sh > /dev/null 2>&1 &`
  (~1.9 h; result `runs/torus_lrsweep_pilot/PILOT_ANALYSIS.txt` recorded here before the batch; EXTEND -> an amendment adding
  3e-2 to `RATES`), then the batch,
  `cd /home/prashr/mapformer && DRV_GPUS=1 DRV_MAXPG=2 setsid nohup bash run_torus_lrsweep.sh > /dev/null 2>&1 &`
  (~20-25 h; marker `.torus_lrsweep_done`).
