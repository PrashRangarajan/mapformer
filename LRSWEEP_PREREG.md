# Is the sign learnability cost present when each arm is compared at its own best learning rate? -- pre-registration (2026-10-16, before any pilot or batch result)

## Question
ROBUST (`ROBUST_RESULTS.md`, registered) found the barrier arm's SOLVED count rising with the learning rate on the
observation-step-free ring at matched seeds 1400-1407: |.| (AO) 0/8 at lr 3.3e-4, 1/8 at 1e-3, 4/8 at 3e-3, while the signed
arm (SO) solved 6/8, 8/8, 8/8. Every sign contrast so far compared the two parameterisations at ONE learning rate, the signed
model's default (1e-3). A parameterisation change is an optimiser change (rule 15), so that comparison is not neutral: the
mock review of draft 5 makes it its top weakness (`docs/paper/review_draft5/MOCK_REVIEW_draft5.md` W1; `SYNTHESIS.md` A1) and
asks for "a per-arm learning-rate sweep, reporting best-of-lr per arm". If |.| learns the map at a larger rate as well as
signed steps do at theirs, the paper's message becomes "barrier parameterisations need larger steps", not "rarely learn".

## Design
Ring, the GAUGE m cell = ROBUST `def` setting, unchanged code: `train_gauge.py --deltas=1,-1` (32 cells, moves +1/-1,
p_first 0.5, T = 128, batch 128, 98 batches, cosine with 5% warmup to 0.1 x lr, 300 epochs, 1 layer, 2 heads, d 128, AdamW,
weight decay 0.05, gradient-norm clip 1.0). Arms SO / AO = `MapFormerWM_SignedAO_r4` / `MapFormerWM_AbsAO_r4`
(observation-step-free: only the two moves step; AO = |SO| per channel).

| rate | 1e-3 | 3e-3 | 1e-2 | 3e-2 |
|---|---|---|---|---|
| note | ROBUST def (in-batch fresh-seed replication) | ROBUST lrx3 | new | new |

2 arms x 4 rates x 8 seeds = **64 runs**. Seeds **1500-1507**: no directory under `runs/`, no driver and no results file uses
any seed 1500-1599 (checked 2026-10-16: `find runs -name '*s15[0-9][0-9]*'` empty; grep of `*.sh` / `*.md` finds no such
seed). Per-seed batch streams (`s * 1000003 + i`, <= 29,400 batches per run) cannot overlap those of seeds 1400-1407 or the
pilot / smoke seeds. At a seed every rate and arm shares the initial weights and the training stream (both are built before
`train()`, independent of lr; ROBUST audit item 7); AO differs from SO only by |.|, rates only by lr. Evaluation inside the
trainer (`train_cancel.evaluate`, fresh stream, np seed 10**6, maps redrawn, env seed 10000, 400 trajectories) at T = 128
(primary) and 4T = 512. `train_gauge.py` is NOT changed (lr and weight decay are existing flags; the default cell reproduced
GAUGE bitwise in the ROBUST audit).

Weight decay is held at 0.05, as in a standard learning-rate sweep. PyTorch AdamW decays by lr x wd per step, so the
effective decay grows with the rate (x3, x10, x30 of the default) and so does the cosine floor (0.1 x lr); a rate here means
"the recipe at that lr", as in ROBUST (Amendment 1 MINOR 1). ROBUST wd0 (cost present with no decay, 0/8 vs 8/8) is the
evidence that decay is not what produces the cost at 1e-3; whether decay limits AO at the high rates is not separated.

## Registered decisions (`analyze_lrsweep.py`)
SOLVED = `stats_core.classify_run` (final 5% of training loss < 0.05); a run with a non-finite loss is DIVERGED, not SOLVED,
and is kept (divergence is data). Accuracy = held-out accuracy at T = 128 (a non-finite value would count as 0; declared).

**Primary B1, best vs best.**
1. *Selection (training-side, fixed now).* For each arm its best rate is the rate with the **most SOLVED runs** (of 8); ties
   are broken by the lower mean selection score (per run: final-5% mean training loss, capped at the run's first-epoch loss;
   a non-finite tail scores the first-epoch loss), then by the lower rate. Held-out accuracy is never used to select.
   *Why SOLVED first, not mean loss:* SOLVED is the outcome on which the registered rule mostly turns (on the ring the accuracy
   gap has been small; ROBUST lrx3 -0.013), so selecting on it gives the barrier arm its best rate on the decisive metric --
   the most generous comparison for |.| and the one W1 asks for; a mean loss can prefer a rate with many runs at 0.06 over one
   with half the runs solved. Both are training-side; the loss enters only to break ties (SO is expected at 8/8 at several
   rates).
2. *Contrast.* AO@best vs SO@best with the ROBUST cell rule (Amendment 1 form, `analyze_robust.cell_label`, reimplemented as
   `label_from` with the p-values supplied and checked equal to it in the selftest), in this order:
   - **UNREADABLE** if SO@best solves < 4/8 (the signed arm learns at no rate);
   - **SIGNED LEARNS BETTER** if p_SOLVED < .05 with SO ahead, or p_acc < .05 with d = AO - SO <= -0.05;
   - **NON-NEGATIVE LEARNS BETTER** if p_SOLVED < .05 with AO ahead, or p_acc < .05 with d >= +0.05;
   - **NO SIGN COST** if SO@best >= 6/8, AO@best >= SO@best - 1 and mean AO accuracy >= mean SO - 0.01;
   - **UNMEASURED** otherwise.
3. *Selection bias: the main design question, resolved by a selection-aware permutation null.* Picking each arm's best of
   four rates on the same seeds that are then compared inflates each arm's best value. The test therefore re-does the
   selection inside the null: the 16 per-seed **rate profiles** (one run per rate; 8 SO, 8 AO) are split into two groups of 8
   in all C(16, 8) = 12,870 ways; in each split each group's best rate is re-selected by rule 1 and the statistic recomputed at
   the re-selected rates (T_k = SOLVED difference, T_acc = mean accuracy difference, both AO - SO); the two-sided p is the
   share of splits with |T| >= |T_obs|. Under the null (arm labels exchangeable per profile) this is exact whatever the
   selection does. **Decision p (each route) = max(selection-aware p, naive p at the selected rates)** (naive = Fisher on SOLVED,
   unpaired permutation on accuracy, the ROBUST tests). The max is valid because the selection-aware p is valid; it is used
   because selection inflates BOTH groups under the null and can make the selection-aware p the smaller one (selftest case
   "AO at edge": 0.0014 vs Fisher 0.0070), so the verdict never rests on a p smaller than the one obtained by pretending the
   selected rates had been fixed in advance. Unpaired, like ROBUST, although the arms share initial weights and stream at a
   seed (valid; conservative under positive within-seed correlation).
   Considered and not chosen: a held-out-seed split (select rates on 4 seeds, test on the other 4): at n = 4 per arm Fisher
   reaches p < .05 only at 4/4 vs 0/4 (0.029), so the primary could fire only on a perfect separation; it is kept as a
   declared secondary (cross-fit, both directions).
   Power (`analyze_lrsweep --power`, synthetic profiles, SO 8/8 at its best): B1 fires on SOLVED when AO@best <= 3/8 (p
   0.026); AO@best 4/8 gives 0.077 (UNMEASURED unless the accuracy gap is >= 0.05); AO@best >= 7/8 with accuracy within 0.01
   reads NO SIGN COST. Where SO's runs are not all solved at every rate the selection-aware p can be smaller than Fisher
   (e.g. 0.031 vs 0.077 at AO 4/8); the max rule discards that gain.

**Reading, fixed now** (`analyze_lrsweep.reading`), with what each means for the paper (`docs/paper/DRAFT5_workshop.md`,
title "Rotary phase steps parameterised to be non-negative rarely learn a map, even when they can represent it"):
- **COST AT BEST RATES** (B1 = SIGNED LEARNS BETTER). Title and Sec. 4.1 stand; the scope gains "at each parameterisation's
  best of four learning rates (1e-3 to 3e-2)"; W1 is answered. The per-rate contrasts say where the cost is largest.
- **COST CLOSES AT HIGHER LR** (B1 = NO SIGN COST or NON-NEGATIVE LEARNS BETTER, and AO's best rate > SO's). The title's
  "rarely learn a map" is withdrawn: barrier parameterisations learn the map at their own (larger) learning rate as reliably
  as signed steps at theirs. The paper becomes a learning-rate-sensitivity (conditioning) result: at a shared learning rate
  the barrier costs (E3, GAUGE, ROBUST), at its own best rate it does not; title e.g. "... need larger learning rates to learn
  a map"; Sec. 4.1's heading and contributions 1-2 are rewritten; the mechanism sections (through-zero route) are presented
  as what signed steps use at a shared rate. "(reversed: NON-NEGATIVE LEARNS BETTER)" is appended when that label fired.
- **NO COST AT BEST RATES** (B1 = NO SIGN COST or NON-NEGATIVE LEARNS BETTER, AO's best rate <= SO's). The cost does not
  replicate at the best rates even without a rate shift (that would contradict ROBUST def and E3 at 1e-3; the per-rate 1e-3
  contrast shows whether the in-batch default replicated). The title claim is withdrawn pending an account.
- **UNMEASURED** (B1 = UNMEASURED). The paper may not claim the cost is robust to the learning rate: the title is scoped to
  "at a shared learning rate"; the abstract states that the cost shrinks with the rate and is unmeasured at each arm's best
  rate (with AO@best SOLVED and the gap).
- **UNREADABLE** (SO solves < 4/8 at every rate): nothing is read.
- Any reading carries the qualifier "[AO's best rate is the grid's top rate: the AO optimum is not bracketed; 'best' = best
  TESTED]" when AO's selected rate is the highest rate of the grid; the paper then says "best tested rate".

**Predictions** (from ROBUST's trend; stated before the pilot): B1 UNMEASURED most likely (AO@best 4-6/8 at 3e-3 or 1e-2,
SO 8/8), COST CLOSES AT HIGHER LR next (if AO reaches >= 7/8 at 1e-2 or 3e-2), COST AT BEST RATES least likely (it needs
AO@best <= 3/8, below ROBUST's 4/8 at 3e-3). SO may degrade or diverge at 3e-2 (kept; it can only lower SO's options, and SO
is selected at its best rate).

**Declared secondaries (no verdict, no reading depends on them):**
- Per-rate AO vs SO at each of the four rates, labelled by the ROBUST rule on **Holm-adjusted** p (Holm over the 4 rates,
  separately for the Fisher and the permutation family); raw p printed beside. The 1e-3 cell is the in-batch fresh-seed
  replication of ROBUST def / GAUGE G2.
- Threshold sensitivity: B1 recomputed with SOLVED at loss 0.1 (selection redone, same max rule).
- Accuracy at 4T = 512 at the B1-selected rates (selection-aware p).
- Loss AUC (mean per-epoch training loss) and mean selection score per arm and rate; run classes (SOLVED / STALLED /
  DESCENDING / RISING / DIVERGED).
- Wrap-cancel share (`wrap_share`, unweighted, wrap-aware, pair (+1, -1)), the **usage-weighted** wrap-cancel share
  (`wrap_share_weighted` x `usage_weights` on 32 held-out trajectories, ACCOUNTS T1's readout; added because the mock review
  W4 notes the unweighted share is dominated by unused channels), and the opposite-sign share (`analyze_robust.opp_share`).
- The SOLVED-vs-lr curve for each arm (descriptive; with ROBUST's 3.3e-4 / 1e-3 / 3e-3 points beside it only as an
  out-of-batch reference, never pooled, rule 12).
- Cross-fit: rates selected on seeds 1500-1503 and tested on 1504-1507, and the reverse (Fisher / permutation at n = 4).

Floors: predict-blank ~0.50 (p_empty 0.5); +1/-1 ring n-gram 0.554; retrace 0.860 (`docs/audits/2026-10-16/ring_floors_out.txt`,
post hoc, ROBUST).

## What this can and cannot show
It tests one axis of the recipe (AdamW peak learning rate with its coupled decay and floor) over 1.5 decades on the ring at
the default model size, for the +1/-1 move pair, at one budget (300 epochs) and n = 8. It does not test the torus (ROBUST's
torus cell is at 1e-3 only), other optimisers, schedules or warmup lengths, or lr x weight-decay interactions; a COST CLOSES
verdict on the ring would not show the same on the torus, and a COST AT BEST RATES would not extend to it either. Every
verdict is budget-scoped. "Best rate" means best of the rates tested. The eval-mode accuracy under-reports one-layer runs below
ceiling (attention-dropout scale, `DROPOUT_RESCORE.md`); no re-score is declared; both arms are read the same way and the
SOLVED contrast does not depend on it.

## Process, code and cost
- `analyze_lrsweep.py` (new; imports `analyze_robust.opp_share` / `cell_label`, `stats_core`): registered analysis,
  `--selftest` (every B1 label and every reading reached from synthetic profiles through the same functions; `label_from`
  checked equal to `analyze_robust.cell_label` on ROBUST's selftest cases; the selection-aware null reduces to Fisher exactly
  when there is one rate; tie-breaks), `--power` (the table above). Verdict JSON is written before the secondaries.
- `run_lrsweep.sh` (new; sources `lib_driver.sh`): `flock` lock; refuses a dirty tree (tracked files) and an uncommitted
  `analyze_lrsweep.py` / `run_lrsweep.sh` / `train_gauge.py` / `LRSWEEP_PREREG.md` BEFORE any wait; checks every rate's
  directory tag against `analyze_lrsweep.rate_tag`; md5 guard over every `mapformer` module the trainer and analysis import
  (from `sys.modules`, `data_parallel` imported explicitly) plus `lib_driver.sh` and `run_lrsweep.sh`, re-checked before the
  analysis; seed outer, rate, arm inner; <= 6 trainers per GPU (`drv_wait_slot 6`, counting every `mapformer.train_*` of the
  user); artifacts-then-marker (`.lrsweep_done` after `LRSWEEP_ANALYSIS.txt` and `LRSWEEP_VERDICTS.json`; 128 training
  artifacts required).
- **Cost** (measured on ROBUST's ring phase, the same trainer and model: 6.2 s/epoch per run at 6 per GPU = ~1.0 GPU-s per
  ring-epoch at any concurrency; learning rate does not change the cost): 64 x 300 x 1.0 s = 5.3 GPU-h, **~2.7 h wall on two
  idle GPUs** (5.3 waves of 12 x 31 min); ~5.3 h on one GPU. ROBUST's 64 ring runs took ~3.7 h beside a torus run. Analysis
  ~2 min CPU. With the 1e-1 extension (below): 80 runs, ~3.3 h on two GPUs.

## Pilot plan and launch rules (fixed now)
Pilot on seeds **1590-1591** (outside the batch; never pooled): SO and AO at 1e-2, 3e-2 and 1e-1, full 300 epochs, the
driver's exact command line (12 runs; ~31 min on two GPUs at 6 per GPU, ~1 h on one).
- **Extension rule.** If the pilot AO is better at 3e-2 than at 1e-2 (more SOLVED over the two seeds; equal SOLVED -> lower
  mean selection score), AO is still improving at the top rate and **1e-1 is added to the grid** (80 runs) by an amendment
  before launch. If AO at 1e-1 is better still than at 3e-2 (same criterion), 3e-1 is piloted on the same two seeds and
  added by the same rule; the grid stops at the first rate at which the pilot AO is not better than at the rate below, and at
  3e-1 at the most.
- The signed arm's behaviour at the high rates (including divergence) does not change the grid.
- If the pilot AO solves 0/2 at every pilot rate and is not better at 3e-2 than at 1e-2, the grid is launched as registered.
- Smoke on seeds 1595-1596 (below). An independent code-verification agent, blind to results, checks this file, the driver and
  the analysis before launch (rule 29); its findings go into Amendment 1.

## Smoke (scratch `lrsweep_smoke/`, seeds 1595-1596, CPU; 2026-10-16)
- The driver's exact `train_gauge` command lines with `--epochs 2 --n-batches 2 --n-trials 20 --device cpu`: all 16
  rate / arm / seed combinations train, evaluate and write `eval.json` + checkpoint. `analyze_lrsweep` reads them end to end
  (B1 UNREADABLE at n = 2, as it must; every secondary prints, including the usage-weighted share; a missing run raises
  FileNotFoundError, not a silent drop). At 2 epochs lr 3e-2 already shows a RISING run (seed 1595, accuracy 0.033): the
  high end of the grid is where the recipe can break.
- `--selftest` passes: `label_from` equals `analyze_robust.cell_label` on ROBUST's cases; every B1 label (SIGNED LEARNS
  BETTER, NON-NEGATIVE LEARNS BETTER, NO SIGN COST, UNMEASURED, UNREADABLE) and every reading (COST AT BEST RATES, COST
  CLOSES AT HIGHER LR incl. "(reversed)", NO COST AT BEST RATES, UNMEASURED, UNREADABLE, and the edge qualifier) reached;
  with one rate the selection-aware null equals Fisher exactly; tie-breaks as registered.
- Synthetic run trees (`lrsweep_smoke/synth_tree.py`: the smoke checkpoints copied to 8 fake seeds, losses and accuracies
  rewritten) read through the real file path: COST AT BEST RATES (AO 3/8 at 3e-3 vs SO 8/8), COST CLOSES AT HIGHER LR (AO
  8/8 at 3e-2, with the edge qualifier; one SO run with a NaN loss classed DIVERGED and kept), COST CLOSES AT HIGHER LR
  (reversed), NO COST AT BEST RATES, UNMEASURED (AO 5/8), UNREADABLE (SO 3/8).
- Driver dry run (`DRV_DRYRUN=1`, a `git clone --shared` of HEAD plus the three new files committed in the scratch clone):
  tag check passes; md5 guard over 96 files (94 imported modules + 2 driver files); 64 launches, seed outer / rate / arm
  inner, no duplicate output dir; then fails closed ("MISSING" x 128, "FAILED: missing runs -- done marker NOT set").

## Pilot (outside seeds, never pooled)
**Not run (2026-10-16 21:50).** Both GPUs are occupied by another user's jobs (danmuir, `tma_standalone.py` cross-validation
folds 6 and 7, ~12 GB and 50-70% utilisation on each GPU, started 21:31; earlier folds ran before them). The standing rule is
no launches beside another user's substantial jobs without asking (`feedback_gpu_sharing_and_cost`). The pilot is ready:
`scratchpad/lrsweep_pilot.sh` (12 runs, seeds 1590-1591, SO / AO at 1e-2, 3e-2, 1e-1, the driver's command line, `flock`,
<= 6 per GPU, output `runs/lrsweep_pilot/`, analysis `PILOT_ANALYSIS.txt` via `analyze_lrsweep --seeds 1590-1591 --rates
1e-2,3e-2,1e-1 --no-json`). Its result decides the extension rule above and is recorded here before launch. The batch is not
to be launched before the pilot and the independent check.

## Amendment 1 (2026-10-16, after the independent check `docs/audits/2026-10-16/lrsweep_audit.md`; before launch, no batch result exists)
The check found the code doing what this file says: 64 launches with the registered lr / wd / deltas / seeds; the 1e-3 and
3e-3 cells byte-identical in command line to ROBUST def / lrx3 (94 shared modules md5-identical to ROBUST's; bitwise on CPU
at one thread); lr reaches the single AdamW group and the cosine schedule at every rate incl. 1e-1 / 3e-1, unclamped; forced
NaN runs are classed DIVERGED, kept, and crash neither the driver nor the analysis; the selection rule, the 12,870-split
null and the reading match independent reimplementations; every label and reading reachable, none guaranteed; size of the
decision p <= 0.5% (SOLVED) and <= 2.2% (accuracy) under the null of equal profiles. Changes:
- **MAJOR 1 (the CLOSES / NO COST split must not rest on SO's loss tie-break).** The split now compares AO's best rate with
  the LOWEST rate at which SO reaches its best SOLVED count. "Contradicts ROBUST def / WRAP_EXISTENCE E3" is printed only if
  the per-rate 1e-3 contrast (naive Fisher / permutation, the ROBUST rule) also reads NO SIGN COST.
- **MAJOR 2 (weight decay coupled to lr).** AdamW's decay per step is lr x wd, so the decay exponent over the run is ~805 x lr
  (0.8 at 1e-3, 8 at 1e-2, 24 at 3e-2) and acts on omega, W_out and the LayerNorm gains. Registered conditional stage 2, fixed
  now: IF B1 reads SIGNED LEARNS BETTER (COST AT BEST RATES) AND AO's selected rate is >= 1e-2, run SO and AO at AO's selected
  rate with wd = 5e-5 / lr (lr x wd equal to the default's), seeds 1500-1507 (16 runs), and label it with the ROBUST rule
  (naive Fisher / permutation). If that decay-matched contrast reads SIGNED LEARNS BETTER, the reading stands; if it reads NO
  SIGN COST or NON-NEGATIVE LEARNS BETTER, the reading becomes "COST AT BEST RATES IS COUPLED DECAY"; if UNMEASURED, the
  reading is scoped to "AdamW wd 0.05 coupled to lr". Stage-2 code gets its own short independent check before launch. If
  AO's selected rate is <= 3e-3 the decay is at most 3x the default's and ROBUST's wd-0 cell (0/8 vs 8/8) stands as the
  decay control. (The audit's option (a), 32 runs at 1e-2 and 3e-2 unconditionally, was not taken: cost on a shared GPU.)
- **MINOR 1.** NO SIGN COST at B1 is worded "cost not detectable at AO's best tested rate (selection-inflated)": with SO at
  ceiling and AO's true solve rate 0.75 at two rates it fires ~33% of the time. The cross-fit is reported beside it, not gated.
- **MINOR 2 (validity statement corrected).** The selection-aware null is exact only if all 16 rate profiles are
  exchangeable; the within-seed pairing of the arms makes it conservative there. Under B1's broader null (best vs best
  equal) the selection-aware p alone can be anti-conservative (0.088 in a simulated case); the max(selection-aware, naive)
  rule is what keeps the false-cost rate <= ~5.2%. With SO at ceiling the decision p reduces to Fisher at the selected rates.
- **MINOR 3 (power in true solve rates, SO at 1.0).** The cost label fires 25 / 67 / 89 / 99% when AO's true best-rate solve
  rate is 0.5 / 0.3 / 0.2 / 0.1.
- **MINOR 4.** UNMEASURED says "the cost may shrink with the rate" only when AO's best rate is above SO's lowest best rate.
- **MINOR 5.** A DIVERGED run scores accuracy 0.0 (below the 0.50 blank floor); disclosed, not re-scored.
- **MINOR 6.** The reading carries a "still DESCENDING at AO's best rate: budget-scoped" qualifier when any AO run there is
  DESCENDING (rule 4).
- **MINOR 7.** run_record asserts n_trials = 400.
- **MINOR 8.** The pilot script has no running() guard; it is not restarted. It ran on code at 9005a48, on GPU 1 shared with
  another user's job (concurrency 4, not 6).
- Selftest re-run after the changes: every B1 label and reading reached.
- GPU: the batch runs on GPU 1 only, beside another user's job, at <= 4 concurrent runs (user's instruction).

## Pilot (seeds 1590-1591, outside the batch, never pooled; 2026-10-16 21:54-23:02 on GPU 1 shared, code 9005a48)
`runs/lrsweep_pilot/PILOT_ANALYSIS.txt`. Accuracy at T = 128 (class), SO / AO:
- lr 1e-2: s1590 1.000 SOLVED / 1.000 SOLVED; s1591 0.946 STALLED / 0.851 STALLED.
- lr 3e-2: both arms at the blank floor or near it (SO 0.496 / 0.496 STALLED; AO 0.757 DESCENDING, 0.496 STALLED).
- lr 1e-1: every run at the blank floor (0.496, STALLED).
**Extension rule:** AO is not better at 3e-2 than at 1e-2 (0/2 vs 1/2 SOLVED; mean selection score 1.44 vs 0.36), so the
grid is NOT extended: rates {1e-3, 3e-3, 1e-2, 3e-2}, 64 runs. 3e-2 breaks the recipe for both arms in the pilot; it stays
in the grid as the registered upper bracket. Note (descriptive, two seeds): the signed arm also degrades at 1e-2.
Launched as amended.
