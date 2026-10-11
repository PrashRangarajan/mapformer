# Rank tests: the 1D ring and the escape test -- pre-registration (2026-10-11, before any run)

Plan: `docs/plans/2026-10-11_next_gpu.md` items 1-2 (after the critical review `docs/audits/2026-10-11/CRITICAL_REVIEW.md`,
which ranked them first). Both test the one claim that looks new (CLAUDE.md "Rank": training finds MapFormer's map only
when each head's content-to-angle rank exceeds the space's dimension D) and the rank theory T1
(`docs/theory/2026-10-04/00_PLAN.md`): at per-head rank r = D, any step shared by every move (a clock component c) is a
false drift of the decoded position in every channel, so a rank-D head is clean only if c = 0 exactly; at r >= D + 1 a
clock can sit off the map. T1 is a statement about end states; whether drift is the CAUSE of failure, and whether the
failure is a search pathology or an early choice of basin, is untested. Code (`model_ring_rank.py`, `train_ring_rank.py`, `ring_rank_readouts.py`, `build_rank_lift.py`, `analyze_rank_tests.py`,
`run_rank_tests.sh`, `docs/audits/2026-10-11/rank_tests_{checks,secondary}.py`, `rank_tests_pilot.sh`), checks (ALL PASS) and the
branch smoke test (`rank_tests_smoke_out.txt`, ALL PASS) are committed with this file before any GPU run; a pilot (seeds outside the batch) and an independent code audit (rule 29) follow as amendments.

## Experiment 1: per-head rank 1 vs 2 on the 1D ring
**Task.** `environment_cancel.GridWorldCancel` (CANCEL's ring: 32 cells, map redrawn per trajectory, directed walk with
runs of 1-10 repeats), p_plus = 0.5 (symmetric), T = 128 moves. Recipe = CANCEL's (`run_cancel.sh`): 300 epochs x 98
batches, batch 128, lr 1e-3, 5% warmup + cosine, 1 layer, 2 heads, d 128, attention dropout 0.1. Trainer
`train_ring_rank.py` (a new entry point; `train_cancel.py` unchanged). Evaluation on a fresh stream (np seed 10^6; every
map unseen) at T = 128 and 512, 400 trajectories.
**Gate** (`docs/audits/2026-10-11/rank_tests_checks.py` / `_out.txt`): floor 0.554 (better of best constant and word
n-gram 1-5, CANCEL_PREREG); revisits 0.759 of steps; wrap-only revisits 0.111 of revisits (so this task tests T1's drift
claim, mostly not the periodic-code reading). Stored reference: shared rank 2 (`Vanilla`, which is rank 2 per head here)
solved 8/8 at p_plus 0.5 in CANCEL.
**Arms** (`model_ring_rank.py`; n = 8, seeds 20-27, fresh for this task; matched initialisation: every non-bottleneck
weight equals MapWM's base at the seed, check C1):
| arm | per-head rank | step |
|---|---|---|
| R1 | 1 | learned, every token: Delta_h = W_out^h W_in^h x, W_in^h: d -> 1 |
| R2 | 2 | learned, every token (= `model_rank_perhead.MapFormerWM_PerHead`, check C2) |
| R1DF | 1 | drift FORBIDDEN by construction: +1 action steps W_out^h W_in^h a with a = (e(+1) - e(-1)) / 2, the -1 action its exact negative, every other token 0 (checks C3, C4); same initial weights as R1 |
**Primary readout: SOLVED** = final-5% training loss < 0.05 (`stats_core.classify_run`), per run. Training loss is
unaffected by the eval-mode dropout-scale issue. Fisher exact test, two-sided.
**Decisions** (k_X = SOLVED count of arm X out of 8):
- **E1 T1 IN 1D.** **HOLDS**: k_R1 <= 2 AND k_R2 >= 6 AND Fisher(R2 vs R1) p < .05. **BROKEN**: k_R1 >= 6 (rank D
  solves; T1's "rank D is hard" fails in 1D). Otherwise **MIXED (k_R1/8 vs k_R2/8, Fisher p)**.
- **E2 DRIFT AS CAUSE** (read only if E1 is HOLDS or MIXED with k_R1 <= 2). **DRIFT IS THE OBSTACLE**: k_R1DF >= 6 AND
  Fisher(R1DF vs R1) p < .05. **DRIFT IS NOT THE OBSTACLE**: k_R1DF <= 2. Otherwise **PARTIAL (k_R1DF/8)**. If E1 is
  BROKEN or k_R1 > 2: **E2 NOT READ (rank 1 does not fail)**, R1DF reported descriptively.
**Power (exact).** Fisher two-sided at 8 vs 8: 0 vs 6 p 0.007, 1 vs 6 p 0.041, 2 vs 7 p 0.041, 2 vs 6 p 0.13. So
HOLDS needs a split at least as clean as 1 vs 6 or 2 vs 7; a 2 vs 6 outcome reads MIXED by design.
**Declared secondaries (no verdict).** Accuracy at T = 128 and 512 in eval mode and dropout-scale re-scored
(`rescore_hook`, one layer); per-head drift kappa (`ring_rank_readouts.drift_kappa`, weighted as SCORE_RANK's head_stats;
CLEAN if kappa <= 0.01 on some head, else CLOCK) and "SOLVED iff CLEAN" per arm (descriptive; thresholds from SCORE_RANK);
epochs to SOLVED (first epoch the 10-epoch running mean < 0.05); r(final loss, accuracy).

## Experiment 2: the rank escape test
**Question.** Is rank 2's failure on the T = 1024 torus a search pathology at exact rank (a spare rank lets a stuck model
escape: the Burer-Monteiro / over-parameterisation reading) or an early choice of basin (a stuck model stays stuck even
with the capacity rank 4 uses)?
**Sources.** The 8 stored rank-2 runs `runs/rank_mi/p0/Vanilla_s0-7` (shared rank 2, T = 1024, 900 epochs, 0/8 SOLVED;
final losses 0.05-0.67).
**Arms** (n = 8, one per source seed; same seed, same map, same data stream across arms: `--data-seed-offset 1`):
| arm | start | rank | how |
|---|---|---|---|
| E (escape) | the stalled rank-2 weights, lifted | 4 (shared, `Vanilla_r4mi`) | `build_rank_lift.py`: W_in gets 2 rows at the default init scale, W_out 2 columns at exactly 0, so the function is unchanged (float64 max logit diff <= 2.4e-11, check C5) |
| C (control) | the stalled rank-2 weights | 2 (`Vanilla`) | the same warm restart without the extra rank |
| S (scratch) | fresh init at the seed | 4 (`Vanilla_r4mi`, matched init to rank 2's base) | the same budget from scratch |
Recipe: `train_variant` with RANK_MI's flags (T = 1024, batch 16, 98 batches, lr 1e-3, cosine with 5% warmup, 1 layer,
2 heads, d 128, `--n-landmarks 0`, `--data-workers 3`) and **300 epochs**; E and C via `--init-from` (fresh optimizer
and schedule: a warm restart). Evaluation: `eval_noise_refine` (held-out map, T = 512 / 1024 / 2048, 100 trials) and
`eval_rank_strata` (T = 1024) as RANK_MI.
**Budget rule (fixed now).** Stored rank-4 runs reach a 10-epoch running loss < 0.05 at epochs 481-579 under the
900-epoch schedule (`runs/rank_mi/p0/Vanilla_r4mi_s*`), so 300 epochs may be too short for S. The pilot trains S at
seed 100 for 300 epochs: if it does not reach SOLVED, the budget for all three arms becomes **600 epochs** (written as
an amendment before launch; nothing else changes).
**Primary readout: SOLVED** (as experiment 1) per run; Fisher two-sided.
**Decisions** (k_X out of 8):
- **X1 DOES A SPARE RANK RESCUE A STALLED MODEL?** **RESCUES**: k_E >= 6 AND Fisher(E vs C) p < .05. **NO RESCUE**:
  k_E <= 2 AND Fisher(E vs C) not < .05. Otherwise **PARTIAL (k_E/8 vs k_C/8)**.
- **X2 ESCAPE OR RELEARN** (read only if X1 is RESCUES). **ESCAPE (faster than from scratch)**: Fisher(E vs S) p < .05
  with k_E > k_S, OR geometric-mean epochs-to-SOLVED E / S <= 0.8 with permutation p < .05 on log epochs (unsolved runs
  censored at the budget + 1). **RELEARN (no faster than from scratch)**: otherwise.
- **X3 IS THE BASIN CHOSEN EARLY?** (read only if X1 is NO RESCUE). **BASIN CHOSEN EARLY**: k_S >= 6 AND Fisher(S vs E)
  p < .05 (rank 4 solves from scratch in the budget, but not from the stalled rank-2 state). **UNMEASURED AT THIS
  BUDGET**: k_S <= 5. **NOT SEPARATED**: k_S >= 6 but Fisher(S vs E) not < .05.
**Power.** As experiment 1 (Fisher 8 vs 8). The control C is expected near 0-1/8 (rank 2 0/8 at 900 and 1800 epochs;
seed 1 ended at loss 0.051 and may cross 0.05 in C: that is what C is for).
**Declared secondaries (no verdict).** Accuracy (held-out map, T = 1024; eval mode and re-scored) and the wrap-only /
plain revisit strata per arm; the T1 head class (`analyze_score_rank.head_stats` / `basin`) at the end of each run and
"SOLVED iff CLEAN"; the norm of W_out's two new columns at the end (does E use the extra rank?); epochs to SOLVED.

## Seeds, pilot, audit, void
- Batch seeds: ring 20-27; escape 0-7 (the source seeds; the escape arms have never been trained). Pilot (outside the
  batch): ring seed 120 (R1, R2, R1DF, 300 epochs); escape: S at seed 100 for 300 epochs (the budget rule) plus one E/C
  pair is NOT possible outside the batch without a new 900-epoch rank-2 source (~1 h), so the escape pilot is S only,
  plus a 3-epoch dry run of E and C at seed 0 to check the --init-from path end to end (not used for any result).
- Independent read-only code audit after this file and the pilot, as an amendment, before the batch.
- Void: any run missing or not at its epoch count; the md5 guard trips; check C5 fails on a rebuilt lift.

## What it can and cannot show
- Can: whether rank = D is hard in 1D (the cheapest test that can break T1); whether forbidding drift by construction
  removes rank 1's difficulty (drift as cause, not only end state); whether extra rank rescues a stalled rank-2 model
  and whether that is faster than starting over.
- Cannot: say anything about multi-layer models or other tasks; the escape arms are warm restarts (fresh optimizer), so
  "stalled" is with respect to the original schedule; R1DF also removes observation steps, so "drift" here means the
  whole per-move common component, including what observations contribute.

## Cost (estimate; the pilot measures)
Ring: 24 runs x ~10 min at full concurrency ~45 min wall. Escape: 24 runs x 300 epochs = 7,200 run-epochs at ~2,700 per
hour (SCORE_RANK) ~2.7 h wall (600 epochs: ~5.3 h). Pilot ~30 min. Total ~4 h (or ~6.5 h at 600 epochs).
