# RANK_TESTS independent code audit (rule 29), 2026-10-11

Read-only and blind to results. Nothing under `runs/rank_tests_pilot/` or `runs/rank_tests/` was read (only the
existence of the eight lifted checkpoints and the absence of `runs/rank_tests/code_md5.txt` were checked), no pilot
output was read, and no conclusions from `CRITICAL_REVIEW.md` were used. CPU tests wrote only to the session
scratchpad. Stored runs read: `runs/rank_mi/p0/{Vanilla,Vanilla_r4mi}_s0-7` (the sources and the stored rank-4 reference).

## Verified correct (with evidence)

- Ring token ids: vocab 19, actions 0 (+1) and 1 (-1), objects 2-17, blank 18 (`environment_nd.py:125-135`,
  `environment_cancel.py:41-48`). `train_ring_rank.build` passes `action_ids=(0, 1)`; `drift_kappa` defaults match.
- R1DF forward = `MapFormerWM.forward` except the step: CPU test, `MapFormerWM.forward` run on R1DF's weights with
  its `action_to_lie` replaced by R1DF's step table gives max |logit diff| 0.0.
- Per-head rank of the step table (numerical `matrix_rank` per head): R1 [1, 1], R2 [2, 2], R1DF [1, 1].
- Matched initialisation C1/C2 reproduce. R1 and R1DF have identical state dicts (the `act` buffer is non-persistent).
- Lift: rebuilt in scratch for seeds 0-1, fp64 diff 4.7e-12 / 1.3e-12. It loads through `train_variant --init-from`
  (variant assert, config-mismatch check, strict load). A CPU dry run of E (lifted) and C (rank-2 source) at seed 0
  with the driver's flags completed and saved checkpoints.
- Sources: E lifts `runs/rank_mi/p0/Vanilla_s<S>/Vanilla.pt`. C loads the same file. S is `Vanilla_r4mi` at seed S,
  so its base equals the source's initial base.
- Data stream: all three escape arms pass `--seed S --data-seed-offset 1`, so `base_seed = torch.initial_seed() + 1 =
  S + 1` (`train.py:109`) and the walks are identical across E/C/S. That stream is fresh with respect to the source's
  own (base S); it replays source S+1's walks on map S, which is harmless. Ring arms share `base_seed = seed`. The eval
  stream (np seed 10^6) collides with no training batch seed (`(s*1_000_003 + i)`, s >= 20).
- Driver flags equal `run_rank_mi.sh` (T 1024, B 16, 98 batches, lr 1e-3, cosine, 1 layer, 2 heads, d 128,
  landmarks 0, data-workers 3), apart from `--data-seed-offset 1` and the dropped `--save-full-state`. Seeds, epochs and
  dirs match the prereg. The done marker comes after `drv_require`. The eval layout `<runs-dir>/p0/<V>_s<S>/<V>.pt`
  matches `eval_noise_refine` (tag `p0`) and `eval_rank_strata`. The analysis reads `RANK_TESTS_ESCAPE_<arm>.json`,
  which `eval_noise_refine` writes (`.md` -> `.json`), under key `0.0|<V>|1024`.
- Decision functions match the prereg text (E1, E2, X1, X2, X3), and the branch smoke passes. The prereg's Fisher values
  are correct.
- G1 recomputed by calling the task code (`GridWorldCancel.generate_trajectory`, np seed 10^6, 400 walks): revisits
  0.759 of steps, wrap-only 0.106 of revisits (the gate reports 0.111 from its re-implementation).

## BUG

**B1. `epochs_to_solved` gives a finite epoch count to runs that are not SOLVED. X2 uses it.**
`analyze_rank_tests.py:68-73` returns the first epoch at which the 10-epoch running mean is below 0.05, whatever
`classify_run` says. Evidence: the stored source `rank_mi/p0/Vanilla_s1` is not SOLVED (final-5% 0.0511, class
DESCENDING) but `epochs_to_solved` returns **829**, because its minimum loss is 0.0287 and it oscillates around the
threshold. In X2 (`x2`, line 49) an unsolved E run with a transient dip counts as fast, which biases toward ESCAPE. The
ring's "epochs to SOLVED" secondary has the same problem.
*Fix:* `return budget + 1` unless `classify_run(losses)["registered"] == "SOLVED"`. Add a smoke case: a dip below 0.05
followed by a final tail above it should return budget + 1.

**B2. A declared secondary is not computed: the re-scored accuracy of the escape arms.** The prereg (exp. 2
secondaries) declares "Accuracy (held-out map, T = 1024; eval mode and re-scored)". The driver runs only eval-mode
`eval_noise_refine`. `rank_tests_secondary.py` re-scores only the ring.
*Fix:* add `python3 -m mapformer.rescore_hook --scale auto -- mapformer.eval_noise_refine ...` per arm to the driver
(writing `RANK_TESTS_ESCAPE_<arm>_RESCORED.md`), or remove the word from the prereg by amendment.

**B3. The analysis does not check run provenance.** `analyze_rank_tests.main` checks only that every escape run has the
same length (`budget = budget or len(L)`). If every run were trained at the wrong epoch count, or E/C without
`--init-from`, or with another offset, the analysis would still pass. The void rule ("not at its epoch count") is not
enforced against the registered value.
*Fix:* for each escape checkpoint assert `config["epochs"] == ESC_EPOCHS` (300, or 600 after the amendment),
`config["data_seed_offset"] == 1`, `config["init_from"]` ending in the lifted / `rank_mi` path for E / C and `None`
for S, and `seed == S`. For the ring assert `config["args"]["epochs"] == 300` and the arm name.

**B4 (ops). Restarting the driver launches duplicates.** `run_rank_tests.sh:95` and `:105` skip a run only when its
final artifact exists. If the driver dies while children run (rule 21's exact case) and is restarted, every in-flight
run is launched again into the same output dir. `flock` covers only concurrent drivers.
*Fix:* also skip when a `python3` process carries `--output-dir $OUT` in its argv (the rule-23 awk on
`index($0, "--output-dir " o)`).

## DESIGN major

**D1. X1 "NO RESCUE" is not conditioned on the budget being enough for rank 4 to solve at all.**
Stored rank-4 runs (`rank_mi/p0/Vanilla_r4mi_s0-7`) cross a 10-epoch running loss of 0.05 at epochs
**481 / 514 / 541 / 514 / 481 / 530 / 557 / 579** of a 900-epoch schedule (recomputed). At 300 epochs, k_E <= 2 can
simply mean rank 4 has not had time. X3 has the guard (k_S <= 5 -> UNMEASURED), but X1 is read unconditionally, so the
likely outcome at a short budget is the pair "NO RESCUE" + "X3 UNMEASURED", and the first label would be a budget
artefact (rule 4). The budget rule rests on one pilot seed (S at seed 100), which is weak evidence that 8 seeds solve.
*Fix:* (a) X1 NO RESCUE requires k_S >= 6. Otherwise label it "UNMEASURED AT THIS BUDGET (S k_S/8)". (b) Prefer 600 or
900 epochs outright (900 matches the sources' own schedule and the stored r4 evidence: 8/8 at 481-579). Or tighten the
budget rule: the pilot S must reach SOLVED with epochs-to-SOLVED <= 0.6 x budget, or two pilot seeds must both solve.

**D2. X2 can barely go the other way: RELEARN is effectively unreachable.**
E starts from a network trained for 900 epochs (loss 0.05-0.67, with trained embeddings, attention and readout). S
starts from scratch (loss ~2.8). Even if E discards its rank-2 phase code and relearns the map, it does so on top of a
trained content pathway, and it starts close to the threshold. So "faster than from scratch" is close to guaranteed
whenever X1 is RESCUES, and the ESCAPE / RELEARN distinction is not tested (rule 5: check that the verdict cell could
have gone the other way).
*Fix:* add a relearn reference arm R: the trained rank-2 network with a fresh default-init rank-4 bottleneck (W_in and
W_out redrawn; everything else from the source; same data and schedule). Then ESCAPE requires E faster than R. Or
define ESCAPE by representation: E's final W_out[:, :2] W_in[:2] step table stays aligned with the source's (for
example cosine > 0.9 on the action rows), against RELEARN where it is replaced. Otherwise make X2 descriptive.

**D3. R1DF removes three things, not one: the action-common component c, every observation step, and its
per-object variance.**
The project has a registered, citable result that zeroing observation steps alone speeds convergence: ActOnly /
NormStep converge 16/16 where MapWM is still descending (`LEAK_RESULTS.md`). "DRIFT IS THE OBSTACLE" can therefore fire
for the LEAK reason, independent of T1's rank argument. The prereg's "Cannot" paragraph widens "drift" to include what
observations contribute, but observation-identity variance is not a common component. The verdict name still asserts
drift. Also, R1DF's initial step scale differs from R1's: the |step(+1)| ratio R1DF/R1 has median **0.68** (range
0.14-4.6) over 20 seeds, because `a = (e+ - e-)/2` has 1/sqrt(2) of the std of e+ (rule 15).
*Fix:* add an arm **R1AO** (rank 1; observation and blank steps forced to 0; action steps free, so c is free). Then
(R1DF vs R1AO) isolates the action-common component and (R1AO vs R1) isolates the observation leak. It costs 8 runs
(~10 min each). At minimum, rename the E2 verdicts "COMMON + OBSERVATION STEP IS / IS NOT THE OBSTACLE" and cite
LEAK as the competing explanation. Use `a = (e+ - e-)/sqrt(2)` to match the initial step variance, or state the
mismatch.

**D4. The sources are not all "stalled", and C's likely partial success limits which X1 branches can be reached.**
By `classify_run` the eight sources are 4 STALLED (s0, s5, s6, s7) and 4 DESCENDING (s1-s4; s1 final 0.0511 with
minimum 0.0287, s4 0.177). A warm restart at lr 1e-3 may solve s1 (and possibly s4) in C. The ceiling this puts on
RESCUES: 6 vs 1 p 0.041, 7 vs 2 p 0.041, 8 vs 3 p 0.026, **7 vs 3 p 0.119, 8 vs 4 p 0.077**. If k_C >= 4, RESCUES
cannot be reached. E and C are paired by source, but the Fisher test is unpaired.
*Fix:* state the source classes in the prereg and replace "stalled" with "unsolved at 900 epochs (4 STALLED,
4 DESCENDING)". Report X1 per source class and add an exact McNemar on the 8 (E, C) pairs as a declared secondary.
Alternatively, register "k_E - k_C >= 5" as the effective condition so readers see the ceiling.

## DESIGN minor

**M1. E1 confounds per-head rank with total latent dimension.** R1 has 2 latent dims (1 per head) and R2 has 4. The
clean per-head-rank contrast at fixed total latent is R1 against shared `Vanilla` (r=2: per-head rank 2, total 2). R1
is exactly Vanilla with a block-diagonal W_out, the RANK_SEP C_bd vs C design. *Fix:* add `Vanilla` at seeds 20-27
(8 runs) as a declared contrast. CANCEL's 8/8 is stored and comes from other seeds, so rule 12 applies.

**M2. E2 is read even when R2 also fails.** When E1 is MIXED with k_R1 <= 2 and k_R2 < 6, the task is not solved at
rank 2 within the budget, so "rank 1 fails because of drift" is not rank-specific. *Fix:* also require k_R2 >= 6 for
E2. Otherwise report "E2 NOT READ (rank 2 does not solve)".

**M3. The kappa readout cannot see per-object step variance.** `ring_rank_readouts.drift_kappa` uses the mean over
objects (line 28). CPU test: a hand-built clean R1 plus a zero-mean leak on two object types (+/-0.5 along W_in) still
reads kappa ~6e-07, so CLEAN. The 0.01 threshold is also carried over from the T = 1024 torus to T = 128, where gaps
are shorter. *Fix:* add per head `sd_over_objects(A[obj]) / |U|`, weighted the same way, and report it beside kappa.
Label the 0.01 threshold as borrowed.

**M4. The verdict labels overreach.** "T1 IN 1D HOLDS / BROKEN" tests "rank = D is hard within 300 epochs", a search
claim. T1 itself, by the prereg's own text, is an end-state statement. "BASIN CHOSEN EARLY" does not measure "early":
the lift is from the end of 900 epochs. *Fix:* rename to "RANK-D DIFFICULTY IN 1D HOLDS / BROKEN (budget-scoped)" and
"BASIN COMMITTED BY THE END OF RANK-2 TRAINING". Print the DESCENDING count beside each k.

**M5. Gate G1 re-implements the walk (rule 11: the gate must call the task code).**
`rank_tests_checks.py:50-64` draws its own walk. My recomputation through `GridWorldCancel` gives wrap-only 0.106,
not 0.111, which is immaterial. The 0.554 floor is quoted from CANCEL, not recomputed on the 10^6 stream. *Fix:* call
`env.generate_trajectory` in G1 and recompute the floor on the eval stream.

**M6. Prereg precision.** C5 claims "the function is unchanged (float64 <= 2.4e-11)". In float32, which is how training
runs, the lift changes logits by 4.6e-03 (s0) and 7.7e-04 (s1) (`build_rank_lift.py` prints this). E and C also
differ in dropout RNG, because building `Vanilla_r4mi` draws extra global-RNG numbers before `--init-from`. Both are
harmless for an unpaired count test, but the prereg should state both.

## Cosmetic

- `MapFormerWM_Ring_R1DF` inherits `get_position_state`, which returns the unconstrained `action_to_lie` steps, not the
  step table. Any future probe that calls it on R1DF reads the wrong thing. Override it.
- E checkpoints get `losses_prior == []` (the lift saves `losses: []`), while C's carry the 900 source epochs. Copy
  `source_losses` into `losses_prior` for symmetry.
- `x1` labels "C solves, E does not" (k_E <= 2, Fisher p < .05) as PARTIAL. `e2`'s "rank 1 does not fail" is shown at
  k_R1 = 3-5. Name both cases.
- The md5 GUARD omits `docs/audits/2026-10-11/rank_tests_secondary.py`, `lib_driver.sh` and `run_rank_tests.sh`.
- `rank_tests_secondary.py` pools r(final loss, acc) over all 24 ring runs. Rule 2 asks for it per eval length, and
  per arm is more informative.
