# B. State correction -- inventory

## Overview

The line asked whether explicit state correction on MapFormer's path integral -- an invariant
EKF on SO(2) (Level 1 / Level 1.5 / Level 2), predictive coding (PC) and its gradient-isolated
variants, a Gaussian sum filter, a two-timescale cascade, depth-axis theta refinement -- does
Bayesian inference work (sharp corrections from informative observations), and whether it pays
under drift, landmarks, long OOD length, and in hex/place-cell probes. What survives is narrow:
(1) Level 1.5 improves accuracy only at OOD length, and only after loss-matching at n=5
(+0.062 t 3.08 at T=512, +0.124 t 3.83 at T=1024; `L15_ABLATION.md`), replicated in direction at
n=12 (`L15_LOOP_2X2.md`); (2) it improves likelihood/stability on every task measured and
accuracy almost never (`CORRECTION_COMPOSITIONAL.md`); (3) no named component (measurement head,
per-token gate, learned Pi vs DARE) is shown load-bearing at n=5; (4) on Match-Query with
stochastic transitions -- the task built to supply its premise -- the filter's benefit at p=0.10 is
bounded below MDE ~0.065-0.068 and does not grow with drift, in two recipes (pre-registered
prediction refuted twice); (5) refining theta on the depth axis does nothing detectable; (6) lm200
passes context destruction but a filter-free extra head ties Level15 there. Everything built on
April lm200 checkpoints is void, which removes the whole R_t-saturation mechanism story for PC,
the NoDrop/GSF/Beta/Cascade "wins", and all cross-scale/multi-env landmark results. Hex/DoG tests
are vacuous or single-seed. The line's own interpretation: correction = stabilisation (bounded
state) plus token-type gating, not inference.

Config key used throughout (from `orchestrator.py`): `clean` = p_action_noise 0.0, no landmarks;
`noise` = p_action_noise 0.10; `lm200` = p_action_noise 0.10 + 200 landmarks. Torus 64x64,
16 obs types, p_empty 0.5, T=128 training. "OOD" in the April-May files = fresh obs_map.
Measured always-predict-blank floor on the paper task: 0.484 (`LM200_ABLATION.md`) / 0.506
(`RESULTS_INDEX.md`).

---

### B1 Level 1.5 decomposition at n=5 (component ablation)
- Dates: 2026-09-01
- Question: which parts of Level 1.5 (wrap, measurement head, per-token R gate, learned Pi) carry its effect? Re-tests a published single-seed decomposition.
- Task / environment: clean torus paper task, held-out map; eval T=128 (train), 512, 1024. Floor not restated in file (paper-task blank floor 0.484-0.506 elsewhere).
- Arms: Vanilla; Level15 (wrap + measurement + per-token R); L15_DARE (Pi fixed by DARE); L15_ConstR (no per-token gate); L15_NoMeas (wrap alone, z==0); L15_NoCorr (correction zeroed, == Vanilla).
- Seeds / batch: n=5 per arm, one batch; 300 epochs, 98 batches x 128, warmup+cosine, lr 3e-4 (train_variant default; `run_l15_ablation.sh`).
- Validity gates: rule 9 r(final loss, acc) = -0.930 / -0.897 / -0.812 (T=128/512/1024, 30 runs); convergence per arm (|slope|<5e-4 last 10%): DARE 4/5, ConstR 3/5, Vanilla 3/5, Level15 4/5, NoMeas 1/5, NoCorr 2/5. Harness check NoCorr - Vanilla -0.029/-0.029/-0.020 (unmeasured, as designed).
- Result (raw acc mean +/- sd): Vanilla 0.966/0.891/0.768; Level15 0.979/0.948/0.888; L15_DARE 0.989/0.940/0.843; L15_ConstR 0.989/0.948/0.889; L15_NoMeas 0.964/0.875/0.815; L15_NoCorr 0.936/0.862/0.748.
  - Raw contrasts (T=128/512/1024, MDE): Level15-NoMeas +0.015 (0.033) / +0.074 (0.137) / +0.073 (0.105); Level15-ConstR -0.010/+0.000/-0.002; Level15-DARE -0.010/+0.008/+0.045; Level15-Vanilla +0.014 (0.104) / +0.057 (0.150) / +0.120 (0.180); ConstR-NoCorr +0.053 (0.049) / +0.086 (0.046) / +0.141 (0.050) -- the only raw-detectable contrast, 5/5 seeds, t 3.0/5.2/7.9.
  - Loss-matched: ConstR-NoCorr +0.016 (t 1.24) / +0.014 (t 0.51) / +0.062 (t 1.56); **Level15-Vanilla +0.016 (t 1.32) / +0.062 (t 3.08) / +0.124 (t 3.83)**; Level15-NoMeas -0.004 / +0.036 (t 1.23) / +0.031 (t 1.39).
- Status: Level15 - Vanilla at OOD length, loss-matched: CITABLE (t > 2.8 at n=5, T=512 and T=1024 only). Every component contrast: unmeasured (not a powered negative).
- Pre-registered? yes (branches in `run_l15_ablation.sh` header / verdict section). Branch C fired with sign inverted (n=1 "ConstR worse than nothing" REFUTED); Branch A did not fire (NoMeas << Level15 unmeasured); Branch D fires only after loss-matching; DARE == Level15.
- Caveats: loss-matched analysis is a regression control, not randomised; clean only; n=5, one width; the design "cannot separate the arms", not "the arms are equal".
- Sources: L15_ABLATION.md, run_l15_ablation.sh, CLAUDE.md (2026-08-31/09-01 section 4)
- Bears on: correction = stabilisation at OOD length, not inference; withdraws the named decomposition.

### B2 Filter x loop 2x2 on the clean torus (filter rows)
- Dates: 2026-09-01
- Question: are the InEKF filter and a 4-pass shared-block loop complementary? (Loop line covered elsewhere; filter results recorded here.)
- Task / environment: clean torus paper task, held-out map, eval T=128/512/1024.
- Arms: Vanilla 204,373; Level15 253,973; Looped 204,373; Level15Looped 253,973; LoopedSampled 204,373. Level15Looped verified bit-identical to Level15 at n_loops=1 (max|diff| 0.000e+00), causal leak 0.00e+00; filter adds exactly 49,600 on both rows.
- Seeds / batch: n=12, one batch (two schedulers; 6 of 66 launches duplicated, same seed/code, evals agreed byte-for-byte -- CLAUDE.md infra note); 300 ep, warmup+cosine, lr 3e-4.
- Validity gates: rule 9 r(final loss, acc) = -0.956 / -0.471 / -0.326 at T=128/512/1024 (60 runs). Mean final loss: Vanilla 0.1549, Level15 0.0420, Looped 0.0076, Level15Looped 0.0189, LoopedSampled 0.0180.
- Result (acc mean +/- sd, T=128/512/1024): Vanilla 0.947/0.876/0.749; **Level15 0.990/0.953/0.878**; Looped 0.999/0.872/0.730; Level15Looped 0.994/0.929/0.830; LoopedSampled 0.997/0.905/0.745.
  - Level15 - Vanilla (filter main effect): T=128 raw +0.044 (MDE 0.060, 11/12), LM +0.008; T=512 raw +0.077 (sd 0.096, t +2.79, MDE 0.077, 11/12), LM +0.036 (t +2.33, MDE 0.043, 10/12); T=1024 raw **+0.129 (sd 0.127, t +3.51, MDE 0.103, 10/12)**, LM +0.083 (sd 0.102, t +2.79, MDE 0.083, 9/12). File labels all UNMEASURED (verdict keyed to loss-matched).
  - Filter inside the loop (Level15Looped - Looped): -0.004 / +0.058 (MDE 0.113) / +0.100 (MDE 0.163), unmeasured.
  - Interaction loss-matched: -0.009 (MDE 0.017) / +0.026 (MDE 0.099) / +0.022 (MDE 0.147), unmeasured; raw -0.048/-0.020/-0.029.
  - Levels: the combination is below Level15 alone at OOD (0.929 vs 0.953; 0.830 vs 0.878).
  - Level15Looped - LoopedSampled at T=1024 +0.085 (MDE 0.173), unmeasured; free fix understated by ~0.017 (evaluated at 4 passes, not its best count).
- Status: filter main effect at T=1024 DIRECTIONAL/borderline (raw above MDE; loss-matched t 2.79 at the 2.8 bar); replicates B1's size and sign. Interaction: unmeasured.
- Pre-registered? yes (in-file verdict): super-additivity not found at T=512 or T=1024.
- Caveats: loss-matching is weakly justified at OOD here (r -0.471 / -0.326); noise rejected as a condition (at p=0.25 every arm within 0.11 of the 0.500 floor at T=512); one loop count.
- Sources: L15_LOOP_2X2.md, _L15_LOOP_RAW.md (same accuracy table), CLAUDE.md "Filter x loop" section
- Bears on: correction (OOD stabilisation replicates at n=12); loops (not complementary with filter).

### B3 Refining theta under action noise (depth-axis correction), torus
- Dates: 2026-08-31
- Question: does carrying and correcting theta each loop pass (`LoopedRefine`) beat a fixed theta, with a gain that GROWS with action noise?
- Task / environment: torus paper task, held-out map, p_action_noise in {0, 0.10, 0.25} (record corrupted, agent moves per true action), evaluated at the training noise; T=128 and 512.
- Arms: Vanilla; Looped x4 (theta once); LoopedRefine x4 (`theta = theta_0 + gate*tanh(refine(x))`, +385 params, gate init 0, bit-identical at init, gate gradient 1.9e-03 at 0); Level15.
- Seeds / batch: n=3 per cell, one batch; 300 ep x 98 batches x 128, cosine, lr 3e-4 (`run_noise_refine.sh`).
- Validity gates: none beyond held-out map; no rule-9 table in file.
- Result (acc mean +/- sd):
  - T=128: p=0 Vanilla 0.954, Looped 1.000, LoopedRefine 0.999, Level15 0.966; p=0.1 0.764 / 0.901 / 0.891 / 0.787; p=0.25 0.638 / 0.843 / 0.848 / 0.641.
  - T=512: p=0 0.914 / 0.780 / 0.786 / 0.931; p=0.1 0.677 / 0.629 / 0.631 / 0.705; p=0.25 0.569 / 0.606 / 0.601 / 0.593.
  - refine - fixed (se, t): T=128 -0.001 (0.000, -1.95) / -0.011 (0.008, -1.35) / +0.005 (0.020, 0.25); T=512 +0.006 (0.091, 0.07) / +0.003 (0.027, 0.09) / -0.005 (0.055, -0.10). No slope in noise.
  - Level15 vs Vanilla under noise (t values from CLAUDE.md, not in the results file): T=128 p=0.10 +0.023 (t=1.9), p=0.25 +0.004 (t=0.2); T=512 p=0.25 +0.025 (t=3.82). Loop vs Vanilla at T=128: +0.138 (t=12.1) / +0.205 (t=8.8).
  - Learned gate on this run (CLAUDE.md CORRECTED 2026-09-06 only; not in NOISE_REFINE.md/json): mean|g| 0.144, max 0.320, 7 of 9 positive.
- Status: EXPLORATORY (n=3); refinement gain flat at zero with small se at T=128.
- Pre-registered? yes (`run_noise_refine.sh` header): positive slope predicted -- NOT observed ("flat slope, gain ~0" branch: depth-axis Kalman idea dead).
- Caveats: n=3; "loop beats Kalman under noise ~9x" came from reading the control column, needs its own pre-registered replication (CLAUDE.md); gate statistics exist only in CLAUDE.md.
- Sources: NOISE_REFINE.md, NOISE_REFINE.json, run_noise_refine.sh, CLAUDE.md (2026-08-31/09-01 sections 1-2)
- Bears on: correction (refinement dead on depth axis); Level15 under noise only detectable at OOD length (stabilisation signature).

### B4 Refining theta on Match-Query (premise-invalid first test)
- Dates: 2026-08-31
- Question: does LoopedRefine beat Looped on Match-Query 128^2?
- Task / environment: Match-Query 128^2, TQ=256, chance 0.0625; actions clean, query blind (neither half of the InEKF premise holds).
- Arms: Looped (theta fixed), LoopedRefine (+385 params).
- Seeds / batch: n=8 each, one batch.
- Validity gates: cross-batch repro of Looped FAILED (mean per-seed drift 0.185; seed 2 0.772 -> 0.123), so analysis redone unpaired.
- Result: Looped 0.735 (sd 0.256), LoopedRefine 0.893 (sd 0.082); paired refine - fixed +0.158 (sd 0.302, MDE 0.299, 6/8). Unpaired with loop pooled over 2 batches (n=16, 0.803 +/- 0.200): refine vs fixed +0.090, se 0.058, t 1.56, not significant. Gate per seed +0.058, +0.135, +0.143, -0.029, -0.046, -0.126, +0.059, -0.069 (mean|gate| 0.083, max 0.143, correction capped at 0.14 rad).
- Status: EXPLORATORY (unmeasured; premise does not apply to this task).
- Pre-registered? no.
- Caveats: task has no drift to correct and nothing to correct with; the null replicated a known negative (CLAUDE.md rule 17). Retracts "loop arm never fails 8/8" (loop line).
- Sources: REFINE_RESULTS.md, CLAUDE.md rule 17
- Bears on: correction; Match-Query batch reproducibility (method).

### B5 Filter where its premise holds: Match-Query + stochastic transitions, take 1
- Dates: 2026-09-02
- Question: does Level15 - Vanilla GROW with drift when explore-phase transitions are stochastic (drift present, observations carry true position, task has headroom)?
- Task / environment: Match-Query 128^2, T_explore=512, T_query=256, chance 0.0625; `--p-transition-noise` p in {0, 0.10} on explore only, eval at training noise; drift 0 -> 13.05 cells at p=0.10.
- Arms: Vanilla, Level15, Looped, Level15Looped (3-layer Match-Query models as in the trainer; params not stated).
- Seeds / batch: n=8, one batch; 300 ep, lr 3e-4 (log: "recipe from step 1: lr=3e-4 epochs=300" -- a broken Pareto rule selected C0), 48 batches x 16, cosine, fast-attn.
- Validity gates: pre-flight gates at p=0.10 (`MATCH_QUERY_GATES_P010.md`, 600 episodes): marginal 0.0685, n-gram o1 0.0647 / o3 0.0628 / o5 0.0648, never-moved 0.1042, answerable rate 0.027. Context destruction on the noisy variant (B7): query-actions destroyed keeps ~45% of score -> real floor ~0.15. Convergence: final loss at p=0.10 Vanilla 3.823, Level15 3.443, Looped 2.976, Level15Looped 3.101 (not converged).
- Result (acc mean +/- sd, p=0 / p=0.10): Vanilla 0.505/0.215; Level15 0.540/0.253; Looped 0.878/0.337; Level15Looped 0.737/0.312.
  - **Primary Level15 - Vanilla: p=0 +0.035 (sd 0.267, MDE 0.264, 5/8); p=0.10 +0.038 (sd 0.069, t +1.56, MDE 0.068, 5/8). Change with drift +0.003.**
  - Level15Looped - Looped: -0.141 (MDE 0.155) / -0.025 (MDE 0.057). Looped - Vanilla: +0.373 (MDE 0.251, 7/8) / +0.121 (t +5.22, MDE 0.065, 8/8, DETECTABLE). Interaction -0.176 (MDE 0.340) / -0.063 (MDE 0.106), unmeasured.
- Status: POWERED NEGATIVE for a filter benefit larger than ~0.068 at p=0.10 on this task; the "grows with drift" slope itself is unmeasured (p=0 half has MDE 0.264).
- Pre-registered? yes (in-file and `run_mq_noise_c2.sh` header): effect must grow with drift -- REFUTED (flat).
- Caveats: arms far from converged (rule 10) -- motivated take 2; floor for p=0.10 columns is ~0.15, not 0.0625 (B7); two noise levels only; query transitions clean by design.
- Sources: MQ_NOISE_2X2.md, MQ_NOISE_2X2.json, run_mq_noise_2x2.sh, runs/mq_noise log, MATCH_QUERY_GATES_P010.md
- Bears on: correction is not inference -- the sharpest test available.

### B6 Same 2x2, take 2 (converging recipe)
- Dates: 2026-09-03
- Question: as B5, on arms trained toward convergence.
- Task / environment: identical to B5.
- Arms: identical to B5.
- Seeds / batch: n=8, one batch; 600 ep, lr 1e-3, cosine, fast-attn (`run_mq_noise_c2.sh`).
- Validity gates: as B5. Final loss p=0: Vanilla 1.492, Level15 0.786, Looped 0.668, Level15Looped 0.505; p=0.10: 3.284 / 3.107 / 2.784 / 2.900 (still unconverged at p=0.10).
- Result (acc, p=0 / p=0.10): Vanilla 0.644/0.308; Level15 0.790/0.313; Looped 0.789/0.365; Level15Looped 0.855/0.312.
  - **Primary Level15 - Vanilla: p=0 +0.146 (sd 0.329, MDE 0.326, 5/8); p=0.10 +0.005 (sd 0.065, t +0.23, MDE 0.065, 5/8). Change with drift -0.141.**
  - Level15Looped - Looped: +0.066 (MDE 0.114) / -0.053 (t -2.57, MDE 0.058, 2/8). Looped - Vanilla: +0.146 (MDE 0.309) / +0.057 (t +6.31, MDE 0.025, 8/8, DETECTABLE). Interaction -0.081 (MDE 0.304) / -0.058 (MDE 0.079).
- Status: POWERED NEGATIVE (filter benefit at p=0.10 bounded below MDE 0.065), replicating B5 on a second recipe; slope unmeasured.
- Pre-registered? yes (unchanged from take 1): REFUTED a second time.
- Caveats: recipe transfer failed on variance (Vanilla p=0 sd 0.263 -> 0.261, CLAUDE.md); p=0.10 not converged at 2x budget; non-positional floor ~0.15 (B7) means the primary contrast is a difference of two near-floor numbers.
- Sources: MQ_NOISE_2X2_C2.md, MQ_NOISE_2X2_C2.json, run_mq_noise_c2.sh, CLAUDE.md "The InEKF does not pay even where its premise holds"
- Bears on: correction; also the only landed stochastic-transition experiment (torus `STOCHASTIC_TRANSITION_RESULTS.md` never landed).

### B7 Context destruction on clean and noisy Match-Query
- Dates: 2026-09-03
- Question: does the noisy Match-Query variant still measure localisation?
- Task / environment: as B5/B6; inference only; answers from the true walk.
- Arms: Vanilla, Looped at p=0 and p=0.10.
- Seeds / batch: n=4 seeds per cell.
- Validity gates: this IS the gate.
- Result (control / explore-obs destroyed / query-actions destroyed): p0 Vanilla 0.493/0.050/0.086; p0 Looped 0.824/0.061/0.079; p010 Vanilla 0.302/0.069/0.142; p010 Looped 0.366/0.079/0.167. Chance 0.0625, never-moved floor 0.1042. Clean passes decisively; noisy variant keeps 47%/46% under query-action destruction -- "roughly 45% of the p=0.10 score is non-positional" (local observation prior). Automatic PASS verdict at a 50% line was wrong.
- Status: CITABLE as a validity diagnostic (clean passes; noisy floor ~0.15).
- Pre-registered? no.
- Caveats: n=4, 2 arms; a 120-episode never-moved estimate (0.1204) was biased high vs 600-episode 0.1042.
- Sources: MATCH_QUERY_NOISE_ABLATION.md, MATCH_QUERY_GATES_P010.md
- Bears on: interpretation of B5/B6 and the loop's noisy-task gain; "gate every variant, not every family".

### B8 lm200 context-destruction gate + capacity control (fresh retrain)
- Dates: 2026-08-21
- Question: (1) is lm200 a shortcut? (2) does a filter-free extra head match Level15 there?
- Task / environment: 64x64 torus with 200 landmarks, retrained with `--n-landmarks 200` and DEFAULT p_action_noise 0.0 (`run_correction_gaps.sh`); eval T=128, fresh obs_map (env seed 10000), 12x64 sequences; scored at revisited observations; measured blank floor 0.484.
- Arms: Vanilla, Level15, Vanilla_ExtraHead (generic extra attention head, more params than Level15).
- Seeds / batch: n=3, one batch, 50 ep x 156 batches (LinearLR default).
- Validity gates: shuffle vs on-manifold resample of actions/obs; final train loss mean/worst seed: ExtraHead 0.018/0.046, Level15 0.068/0.188, Vanilla 0.682/0.988.
- Result: intact / resample actions / resample obs: Vanilla 0.8427 / 0.3428 / 0.2829; Level15 0.9850 / 0.1553 / 0.2724; ExtraHead 0.9952 / 0.2871 / 0.2629. Drops (resample actions): Level15 -0.830 (Match-Query reference -0.842), ExtraHead -0.708, Vanilla -0.500. **ExtraHead - Level15 +0.010, t=0.79 (tie); Level15 - Vanilla +0.142, t=2.30.** Vanilla seed 2 finished at loss 0.988 (April non-convergence reproduces under current code), intact 0.722 vs 0.894/0.913.
- Status: gate CITABLE (validity); capacity tie EXPLORATORY (n=3) -- but it is what withdraws the Kalman interpretation of lm200.
- Pre-registered? no (design rationale in script header).
- Caveats: T=128 only (the +24.8pp headline is at T=512 and still lacks a same-length capacity control); this retrain has p_action_noise 0.0, whereas the orchestrator's lm200 config has 0.10 -- not the same condition as B9; the file title says "paper's own task" but the checkpoints are landmark runs. "Level15 beats Vanilla on lm200" is partly "Vanilla sometimes fails to converge on lm200".
- Sources: LM200_ABLATION.md, run_correction_gaps.sh
- Bears on: correction (lm200 gap is capacity/optimisation, not measurement); validity of the corrected leaderboard.

### B9 Corrected lm200 leaderboard (fresh, current code)
- Dates: 2026-07-15 (seed 0) and 2026-07-22 (n=3)
- Question: re-rank lm200 variants after the April non-convergence retraction.
- Task / environment: lm200 torus; T=128 and T=512 OOD.
- Arms: Level15, TEMFaithful, Level15GSF, Level15NoDrop, Level15EM, Vanilla, VanillaEM, PC, MambaLike, RoPE.
- Seeds / batch: seed 0 fresh (CORRECTED_LM200_LEADERBOARD; GSF fresh retrain OOM'd, stored converged checkpoint used); n=3 fresh (LM200_CORRECTED_MULTISEED). Batch membership not stated.
- Validity gates: fresh mean final loss listed (Level15 0.0083, TEMFaithful 0.0009, GSF 0.0860, NoDrop 0.0918, Level15EM 0.3537, Vanilla 0.7741, VanillaEM 0.9719, PC 0.5377, MambaLike 1.8174, RoPE 1.5318) -- several arms still unconverged. No context destruction on these checkpoints (B8 gated a different retrain at T=128).
- Result (n=3, T=128 / T=512): Level15 0.996/0.990 +/- 0.005; TEMFaithful 1.000/0.974 +/- 0.008; Level15GSF 0.982/0.967 +/- 0.034; Level15NoDrop 0.981/0.956 +/- 0.029; Level15EM 0.938/0.823 +/- 0.110; Vanilla 0.814/0.742 +/- 0.075; PC 0.888/0.716 +/- 0.012; VanillaEM 0.830/0.656 +/- 0.130; MambaLike 0.562/0.549 +/- 0.013; RoPE 0.636/0.482 +/- 0.023. Seed 0: Level15 0.996, TEMFaithful 0.982, NoDrop 0.915, Level15EM 0.860, Vanilla 0.835, VanillaEM 0.807, PC 0.721, MambaLike 0.567, RoPE 0.513.
- Status: EXPLORATORY (n=3). Numbers stand; interpretation withdrawn (B8). Diagnostic value: reverses every April lm200 ranking.
- Pre-registered? no.
- Caveats: loss column shows Vanilla/VanillaEM/PC/Level15EM still in bad basins -- the ranking partly tracks convergence again; no capacity control at T=512.
- Sources: LM200_CORRECTED_MULTISEED.md, CORRECTED_LM200_LEADERBOARD.md, NOISE_CLEAN_REVALIDATION.md, RESULTS_INDEX.md note
- Bears on: correction; RoPE/Mamba weakness on landmark torus reproduces (baseline line).

### B10 Clean/noise revalidation (bit-identical retrain)
- Dates: 2026-07-15 (file), cited in all PARTIAL banners
- Question: are April/May clean and noise checkpoints valid?
- Task / environment: noise T=512 (env seed 0, 120 trials, eval seed 1000); clean T=512 spot check.
- Arms: Vanilla, Level15, VanillaEM, Level15EM, Level15NoDrop, TEMFaithful (noise); RoPE, Vanilla (clean).
- Seeds / batch: seed 0 fresh vs stored.
- Result: fresh == old bit-identical on every row (e.g. noise Level15 loss 0.814 acc 0.853 both; clean Vanilla 0.119/0.868 both). Root cause pinned to landmark-cell-selection RNG, which runs only when n_landmarks > 0.
- Status: determinism check (not replication) -- licenses using clean/noise rows from April-May.
- Pre-registered? no.
- Caveats: seed 0 only; this is determinism (rule C27).
- Sources: NOISE_CLEAN_REVALIDATION.md
- Bears on: every clean/noise block below (B11-B18).

### B11 Capacity control per regime + length sweep (clean/noise rows)
- Dates: 2026-05-22
- Question: is Level15 > Vanilla architecture or parameters, per regime and across length?
- Task / environment: torus, eval_single_env, T=512/1024/2048 OOD; NumberLine arithmetic chain (train T=128, OOD chain 512).
- Arms: Vanilla (~256K), Vanilla_ExtraHead (322K, > Level15's 305K), Level15 (305K).
- Seeds / batch: n=3; ExtraHead trained May, Vanilla/Level15 likely stored April checkpoints (mitigated by B10 for clean/noise); 50 ep LinearLR.
- Validity gates: none in file.
- Result (acc T=512 / 1024 / 2048):
  - clean: Vanilla 0.918 / 0.802 / 0.627; ExtraHead 0.742 / 0.601 / 0.526; Level15 0.995 / 0.968 / 0.886.
  - noise: Vanilla 0.634 / 0.574 / 0.536; ExtraHead 0.648 / 0.577 / 0.535; Level15 0.707 / 0.655 / 0.596.
  - NLL clean T=512/1024/2048: Vanilla 0.410/1.352/3.084; ExtraHead 1.620/3.023/3.978; Level15 0.031/0.178/0.648. Noise: 1.610/2.434/3.017; 1.770/2.685/3.266; 0.952/1.210/1.457.
  - NumberLine (in-dist T=128 / OOD chain T=512 / NLL): Vanilla 0.925/0.633/2.024; ExtraHead 0.986/0.662/2.542; Level15 0.902/0.841/0.521.
- Status: EXPLORATORY (n=3, no MDE, 50-ep LinearLR recipe).
- Pre-registered? verdict rule stated (CAPACITY if ExtraHead within 2pp of Level15) -- ARCHITECTURE on clean and noise.
- Caveats: recipe is the LinearLR-from-step-one budget (rule 10); the 300-ep cosine re-test (B1/B2) found a much smaller clean OOD gap, raw-unmeasured at T=512 -- the +8pp at T=512 here does not survive at matched convergence as a raw effect; lm200 rows void; possibly cross-batch arms.
- Sources: CAPACITY_PERREGIME.md, CAPACITY_CONTROL.md (superseded lm200-only version), run_capacity_perregime.sh
- Bears on: correction (length/calibration signature; ExtraHead is not a substitute on clean/noise).

### B12 Level 1.5 on the paper task at 16 vs 50 epochs (with capacity control)
- Dates: 2026-08-17 / 2026-08-18
- Question: does Level15 match Vanilla on the clean paper task, and is it capacity?
- Task / environment: paper config (1 layer, 2 heads, d=128, T=128); same-map and fresh-map held-out revisit accuracy; floor 0.506.
- Arms: Vanilla, Vanilla_ExtraHead (270,934), Level15 (254,230).
- Seeds / batch: n=3, one batch, 16 ep and 50 ep (98 batches x 128).
- Validity gates: final train loss at 50 ep: Vanilla 0.1126, ExtraHead 0.1927, Level15 0.0068.
- Result (fresh-map): 16 ep Vanilla 0.989 +/- 0.010, ExtraHead 0.972 +/- 0.039, Level15 0.938 +/- 0.080; 50 ep 0.993 +/- 0.009, 0.985 +/- 0.023, **1.000 +/- 0.000**.
- Status: EXPLORATORY (n=3, ceiling). Claim supported: "matches or slightly exceeds Vanilla at 50 ep, with ~16x lower training loss".
- Pre-registered? no.
- Caveats: +0.007 against a 0.506 floor is a ceiling effect; the 16-epoch reading is a budget false negative (rule 5).
- Sources: LEVEL15_MEETS_GATED_paper.md, LEVEL15_MEETS_GATED_paper50.md, run_level15_meets_gated.sh
- Bears on: correction (likelihood, not accuracy); rule 5.

### B13 Level 1.5 on Match-Query (no measurements at query time)
- Dates: 2026-08-17
- Question: does the correction survive when the query phase feeds MASK (uninformative innovation)?
- Task / environment: Match-Query, T_explore=512, T_query=256 (train) / 512 (OOD); chance 0.0625; never-moved 0.0893; 3 layers, 200 epochs.
- Arms: Vanilla 601,174; Vanilla_ExtraHead 667,478; Level15 650,774.
- Seeds / batch: n=3, one batch.
- Validity gates: Vanilla retrain matches `MATCH_QUERY_RESULTS.md` to four decimals (0.8884 vs 0.888; same seeds -- determinism).
- Result (TQ=256 / TQ=512): Vanilla 0.8884 +/- 0.1401 / 0.9024; ExtraHead 0.7562 +/- 0.2160 / 0.7055; Level15 0.8764 +/- 0.2128 / 0.8528. Per-seed TQ=256: Vanilla 0.7312/1.0000/0.9340; ExtraHead 0.5886/1.0000/0.6801; Level15 0.6306/1.0000/0.9986.
- Status: EXPLORATORY ("no advantage", cannot order the arms).
- Pre-registered? prediction stated before run (collapse to plain path integration) -- consistent.
- Caveats: n=3, sd 0.14-0.26; Match-Query base rate moved 0.888 -> 0.730 at n=5 elsewhere.
- Sources: LEVEL15_MEETS_GATED_matchq.md
- Bears on: correction needs informative observations (premise), B5/B6 context.

### B14 Level 1.5 on compositional transfer
- Dates: 2026-08-21
- Question: does the correction help cross-instance compositional transfer?
- Task / environment: compositional motif task, n_templates=4, T=256 train, eval 256/1024; metrics exact_acc, cross_nb, cross_nll.
- Arms: Level15, MapWM-Flat (3 layers).
- Seeds / batch: n=3, one batch, 50 epochs x 156 batches.
- Validity gates: an aggregation bug (results keyed by variant, only seed 2 reported) found and fixed here.
- Result: exact_acc T=256 0.940 +/- 0.051 vs 0.919 +/- 0.028; cross_nb T=256 0.368 +/- 0.220 vs 0.260 +/- 0.034 (per-seed Delta -0.043, +0.401, -0.034: 2/3 negative); cross_nb T=1024 0.180 +/- 0.215 vs 0.082 +/- 0.009; cross_nll T=256 0.939 vs 1.385 and T=1024 1.419 vs 2.160, **better on 3/3 seeds at both lengths** (1.269/0.443/1.105 vs 1.365/1.499/1.290; 1.791/0.713/1.752 vs 2.515/1.746/2.219).
- Status: EXPLORATORY (n=3). Five-task synthesis in file: likelihood and stability reliably, accuracy almost never.
- Pre-registered? no.
- Caveats: accuracy mean is one outlier seed; family-tree row of the synthesis is n=3 unmeasured (B15).
- Sources: CORRECTION_COMPOSITIONAL.md, run_correction_gaps.sh
- Bears on: correction = stabilisation/likelihood; hierarchy/compositional line.

### B15 Level 1.5 on the family tree
- Dates: 2026-08-21
- Question: correction on a non-landmark structured task.
- Task / environment: family tree depth 5, 8 obs types, 2 layers, 100 epochs; T=64 train, T=128 OOD; floor 0.163 (hub baseline; chance 0.125).
- Arms: Level15, MapWM-Flat, MapEM-NC-NL, MapEM-os, Plain-Flat.
- Seeds / batch: n=3, one batch; the three republished arms reproduce to 3 d.p.
- Result: Level15 0.843 +/- 0.015 (T=128 0.789) vs MapWM-Flat 0.805 +/- 0.072 (0.746). Per-seed Delta -0.005, +0.001, +0.117; paired t~0.89. Level15 - MapWM-Flat +0.038, sd 0.069, MDE 0.111, 2/3 (N3_AUDIT).
- Status: EXPLORATORY (unmeasured); "variance reduction" (+/-0.015 vs +/-0.072) is one bad MapWM seed.
- Pre-registered? no.
- Caveats: n=3 never extended (N3_AUDIT item 2).
- Sources: FAMILY_TREE_WM_GAP.md, N3_AUDIT.md
- Bears on: correction; the EM/non-commutativity axis is another line.

### B16 Predictive coding + Level 1.5 variant series (clean rows)
- Dates: 2026-04-27 .. 2026-04-30
- Question: do PC (forward model g(theta)->x) and the InEKF (inverse model h(x)->z) compose, and which gradient-isolation fixes prevent degradation?
- Task / environment: torus clean, T=128 / T=512, in-dist and fresh obs_map (seed+1000).
- Arms: Level15; Level15PC (PC aux loss); Level15PC_NoBypass (Fix 5 stop-grad on correction inside aux + Fix 6 mask landmarks); Level15PC_v3 (+ Fix 7 log_R clamp [-1,5]); Level15PC_v4 (+ Fix 8 detach theta_hat AND target embedding: PC gradient touches only forward_model); Level15PC_v4_control (v4 with aux_coef=0).
- Seeds / batch: single seed s0 for the series; n=3 (seeds 0-2) for v4 and control; 50 ep LinearLR.
- Validity gates: per-parameter gradient trace on v4 (aux gradient only on forward_model.*, norm sum 0.062; all other params CE-only, 8.86 -- CLAUDE.md 2026-04-30).
- Result (clean, T=512 OOD, s0): Level15 0.991 (NLL 0.050); Level15PC 0.985 (0.086); NoBypass 0.872 (0.634); v3 0.948 (0.286); v4 0.964 (0.217). T=128 OOD all 0.998-1.000.
  - v4 multi-seed clean T=512 OOD: Level15 0.995 +/- 0.003 (0.991/0.994/0.998); v4 0.985 +/- 0.015 (0.964/0.998/0.994) (V4_MULTISEED).
  - Control (clean T=512 OOD): Level15 0.991/0.990/0.998 = 0.993 +/- 0.003; v4 0.964/0.995/0.992 = 0.984 +/- 0.014; **v4_control 0.991/0.990/0.998 -- byte-identical to Level15 on every seed** (V4_CONTROL_RESULTS).
- Status: EXPLORATORY (s0 / n=3). The control is a determinism result: an unused forward_model shifts no RNG; "init drift" hypothesis dead; no surviving v4 win (clean v4 slightly below Level15).
- Pre-registered? decision rules in files (v4 == Level15 if PC is a passive observer) -- clean v4 is below, not equal, and the control is equal.
- Caveats: the PC/Kalman "duality" and R-saturation autoencoder-bypass mechanism were diagnosed ONLY on April lm200 checkpoints (void, see Excluded); surviving evidence for that mechanism is the clean accuracy ordering at s0 plus the control; "v4 win from grad-clip side effects" (SESSION_2026-05-01) was never shown.
- Sources: NOBYPASS_RESULTS.md, V3_RESULTS.md, V4_RESULTS.md, V4_MULTISEED.md, V4_CONTROL_RESULTS.md, CLAUDE.md (2026-04-27..30, CORRECTED 2026-09-05)
- Bears on: correction (PC coupling degrades length generalisation on clean; full isolation needed); "PC and Kalman are duals" live negative in RESULTS_INDEX.

### B17 NoDrop / learnable beta / GSF / GSF_NoDrop (clean + noise rows)
- Dates: 2026-05-10 .. 2026-05-11
- Question: do removing post-attention residual dropout (Level15NoDrop), a learnable softmax temperature (Level15Beta), or a K=8 Gaussian sum filter (Level15GSF, +NoDrop) change clean/noise results?
- Task / environment: torus clean and noise (p_action_noise 0.10), T=128 / T=512 fresh map.
- Arms: Vanilla, Level15, Level15NoDrop, Level15Beta, Level15GSF, Level15GSF_NoDrop, TEMFaithful.
- Seeds / batch: n=3; Level15/Vanilla rows are April checkpoints reused (valid per B10); new arms May; 50 ep LinearLR.
- Result (T=512 OOD acc; NLL):
  - clean (NODROP/BETA files): Vanilla 0.911 +/- 0.035 (0.458); Level15 0.993 +/- 0.004 (0.039); NoDrop 0.985 +/- 0.016 (0.070); Beta 0.986 +/- 0.005 (0.078); TEMFaithful 0.961 +/- 0.013 (0.213).
  - clean (GSF file, separate eval): Vanilla 0.912 +/- 0.038; Level15 0.992 +/- 0.005; NoDrop 0.984 +/- 0.015; GSF 0.965 +/- 0.014 (0.129); GSF_NoDrop 0.989 +/- 0.002 (0.044); TEMFaithful 0.959 +/- 0.016.
  - noise (NODROP/BETA): Vanilla 0.638 +/- 0.035 (1.637); Level15 0.702 +/- 0.011 (0.994); NoDrop 0.699 +/- 0.027 (0.984); Beta 0.722 +/- 0.006 (0.930); TEMFaithful 0.706 +/- 0.005 (1.224).
  - noise (GSF): Vanilla 0.629 +/- 0.033; Level15 0.692 +/- 0.011; NoDrop 0.692 +/- 0.026; GSF 0.703 +/- 0.014; GSF_NoDrop 0.717 +/- 0.005 (0.946); TEMFaithful 0.696 +/- 0.006.
  - Learned beta per seed: clean 0.1816/0.1492/0.1517, noise 0.1503/0.1485/0.1632 (init 1/sqrt(64) = 0.125).
- Status: EXPLORATORY (n=3, no MDE). Readable conclusion: on clean/noise none of these moves accuracy materially relative to Level15; NoDrop raises clean NLL (0.039 -> 0.070).
- Pre-registered? GSF file states a prediction (marginal clean, modest noise) -- roughly consistent, unmeasured.
- Caveats: every lm200 claim these experiments were built for (NoDrop +13pp, Beta +12pp, GSF closes TEM gap, "dropout not beta was load-bearing") is void; clean vs noise tables differ between files by up to 0.010 for the same checkpoints (eval sampling).
- Sources: NODROP_PARETO_RESULTS.md, LEVEL15BETA_RESULTS.md, GSF_FULL_RESULTS.md, MULTISEED_FOLLOWUP.md (torus clean/noise rows duplicate these)
- Bears on: correction; dropout/Pareto story is gone with lm200.

### B18 Level15Cascade (two-timescale filter), clean + noise
- Dates: 2026-07-12 (design) / 2026-07-15 (results)
- Question: does a slow per-chunk filter on fast-filter residual innovations improve length generalisation?
- Task / environment: torus clean and noise; long_sequence_eval T=128/512/2048 (eval protocol of that script); zero-shot fresh obs_map seeds 10000-10002.
- Arms: Level15; Level15Cascade (chunk 32, slow log_R bias +3.0, K_slow ~0.05 at init).
- Seeds / batch: n=3 model seeds (long-seq); zero-shot model seed 0 only x 3 test seeds; Cascade trained July vs Level15 April checkpoints (not one batch; mitigated for clean/noise by B10); 50 ep x 156 batches.
- Result: clean acc T=128/512/2048: Level15 1.000/0.994 +/- 0.003/0.879 +/- 0.013; Cascade 0.999/0.989 +/- 0.012/0.881 +/- 0.044; NLL 0.000/0.032/0.705 vs 0.002/0.066/0.795. Noise: Level15 0.948/0.866 +/- 0.026/0.676 +/- 0.043; Cascade 0.959/0.881 +/- 0.016/0.700 +/- 0.033; NLL 0.242/0.532/1.183 vs 0.201/0.491/1.149. Zero-shot clean s0 T=512: Level15 0.990 +/- 0.000, Cascade 0.997 +/- 0.000 (NLL 0.062 vs 0.015).
- Status: EXPLORATORY (n=3 / n=1 model). No detectable clean or noise effect claimed.
- Pre-registered? diagnostics specified in design doc (K_slow growth, |d_slow|/|d_fast|, T=2048 slope); none reported.
- Caveats: lm200 cascade "win" void (compared against stuck Level15; fresh Level15 and CascadeNoSlow both 0.996); zero-shot "+/-" is over test seeds of one model.
- Sources: CASCADE_MULTISEED_RESULTS.md, CASCADE_ZEROSHOT_S0.md, SESSION_HIERARCHICAL_CASCADE.md, CASCADE_REPRO_TEST.md
- Bears on: correction (no multi-timescale benefit shown); hierarchy line only by name.

### B19 Test-time omega rescaling across grid sizes (clean)
- Dates: 2026-04-24
- Question: do trained frequencies transfer to unseen grid sizes, and does rescaling omega by train/test size help?
- Task / environment: clean torus, trained grid 64, T=128; test grid 32/48/64/96/128.
- Arms: Vanilla, VanillaEM, Level1, Level15, Level15EM, PC (LSTM, MambaLike rows empty).
- Seeds / batch: 3 model seeds x 3 fresh test seeds (9 runs per cell), April clean checkpoints.
- Result (acc orig / rescaled): Vanilla 32: 0.954/0.310, 128: 0.988/0.681; VanillaEM 32: 0.970/0.938, 128: 0.999/0.870; Level1 32: 0.913/0.824, 128: 0.936/0.732; Level15 32: 0.964/0.990, 48: 0.995/1.000, 96: 1.000/0.985, 128: 1.000/0.886; Level15EM 32: 0.969/0.698, 128: 1.000/0.624; PC 32: 0.930/0.493, 128: 0.962/0.747.
- Status: EXPLORATORY (n=3 model seeds; test-seed replicates are not independent).
- Pre-registered? no.
- Caveats: untouched omega already generalises across sizes for every variant (orig column); rescaling hurts all variants except Level15 at smaller grids. Uses the pre-cosine LinearLR checkpoints.
- Sources: OMEGA_RESCALE_clean.md
- Bears on: environment/map-size transfer; correction's bounded state tolerating frequency mismatch (associated, not intervened).

### B20 Clone-separation transfer (clean, s0)
- Dates: 2026-04-27
- Question: is PC's clone-structure (per-cell clustering of theta_hat) transferable to a fresh map or memorised?
- Task / environment: clean, model seed 0, OOD env seed 10000, T=128, 200 trials.
- Arms: Vanilla, Level15, PC, Level15PC (despite the filename, no NoBypass row).
- Result (in-dist sep / OOD sep / drop): Vanilla +0.271/+0.262/+0.009; Level15 -0.187/-0.172/-0.015; PC +0.235/+0.221/+0.014; Level15PC -0.493/-0.484/-0.009.
- Status: EXPLORATORY (n=1).
- Pre-registered? decision rule in file: PC's separation persists OOD (small drop) -- but PC does not lead here; Vanilla is higher.
- Caveats: n=1; the earlier "PC best separation 0.619" (CLAUDE.md, clone_analysis.py) is not reproduced by this metric/file; CLONE_ANALYSIS_LEVEL15PC.md is empty.
- Sources: CLONE_TRANSFER_NOBYPASS.md, CLONE_ANALYSIS_LEVEL15PC.md (0 lines)
- Bears on: CSCG/clone correspondence -- unsupported.

### B21 Grid-cell probes on discrete torus (block and hidden-state rate maps)
- Dates: 2026-04-24 .. 2026-04-27
- Question: do path-integrator blocks or hidden units show hexagonal grid scores (Sargolini; >0.3 grid-like)? Does trained omega keep geometric module spacing?
- Task / environment: clean config, T=512 (hidden eval default), test seed 12345, seed 0 checkpoints.
- Arms: Vanilla, VanillaEM, Level1, Level15, Level15EM, MambaLike, PC, Level15PC, Grid (fixed hex orientations), Grid_Free (learnable), GridL15PC_Free.
- Result:
  - Block rate maps (Test A, HIPPOCAMPAL_ANALYSIS): mean/max grid score Vanilla -0.034/+0.145; VanillaEM -0.030/+0.258; Level1 -0.069/+0.088; Level15 -0.043/+0.222; Level15EM -0.041/+0.124. Grid_Free run: Vanilla -0.028/+0.225, Level15 -0.053/+0.146, Grid_Free -0.059/+0.142. Blocks are 1D phase clocks by construction.
  - Hidden-state max grid score (dims >0.3 = 0 in every run): HIPPOCAMPAL_HIDDEN Vanilla +0.092, VanillaEM +0.071, Level1 +0.058, Level15 +0.029, Level15EM +0.065, MambaLike +0.142; HIPPOCAMPAL_GRID Grid +0.053, Level15 +0.141, Vanilla +0.088; HIDDEN_GRIDFREE Vanilla +0.058, Level15 +0.081, Grid_Free +0.090; GRIDL15PC Vanilla +0.055, Level15 +0.031, Grid_Free +0.095, GridL15PC_Free +0.052; LEVEL15PC Vanilla +0.062, Level15 +0.155, PC +0.055, Level15PC +0.102, Grid_Free +0.124.
  - Test C (omega spectrum): figure only; "approximately geometric, largely inherited from init".
- Status: EXPLORATORY negative (n=1, no power). No unit in any run crosses 0.3.
- Pre-registered? no.
- Caveats: the same checkpoint (e.g. Level15 s0) gives max hidden grid scores from +0.029 to +0.155 across runs -- sampling noise dominates; Sorscher's non-negativity/DoG conditions absent, so hex was not expected.
- Sources: HIPPOCAMPAL_ANALYSIS.md (Test A, C), HIPPOCAMPAL_GRID.md, HIPPOCAMPAL_GRID_FREE.md, HIPPOCAMPAL_GRIDL15PC.md, HIPPOCAMPAL_HIDDEN.md, HIPPOCAMPAL_HIDDEN_GRIDFREE.md, HIPPOCAMPAL_LEVEL15PC.md, hippocampal_analysis.py, hippocampal_hidden_eval.py
- Bears on: "hex emergence does not follow from architecture or correction stacking" (RESULTS_INDEX live negative) -- n=1 support only.

### B22 Continuous 2D navigation (Cueva/Wei/Sorscher-style) -- position decoding and hex probe
- Dates: 2026-05-03
- Question: on continuous velocity-driven navigation with a 256-unit ReLU bottleneck, does Level15 hold position at long T, and do grid-like units emerge?
- Task / environment: continuous torus size 64, 256 place cells, v_noise = omega_noise = 0.05; trained with hard_ce (argmax nearest place cell) after the MSE-on-DoG run collapsed to chance (~20-cell error; commit 57084cc); eval T=128/256/512/1024 x eval-noise 0/0.05/0.10/0.20; hex probe 200 trajectories T=256.
- Arms: Vanilla, Level15, VanillaEM, Level15EM (seed 0 each).
- Result (mean position error in cells, eval-noise 0.00, T=128 / 512 / 1024; p90 at 512): Vanilla 1.83 / 7.31 / 9.44 (p90 28.02); Level15 1.84 / 2.62 / 5.01 (p90 2.99); VanillaEM 1.83 / 1.71 / 1.70 (p90 2.61); Level15EM 1.83 / 1.90 / 1.96 (p90 2.68). Eval noise changes nothing. Hex max grid score: Vanilla 0.299, VanillaEM 0.001, Level15 0.083, Level15EM -0.024; frac>0.3 = 0.00% for all.
- Status: EXPLORATORY (n=1 per arm).
- Pre-registered? no.
- Caveats: MSE column meaningless under hard_ce; hard_ce targets are not DoG regression, so this is not a test of Sorscher's conditions; single seed; the EM backbone holding at T=1024 is one seed each.
- Sources: CNAV_RESULTS.md, CNAV_HEX_Vanilla.md, CNAV_HEX_VanillaEM.md, CNAV_HEX_Level15.md, CNAV_HEX_Level15EM.md, git log 57084cc, SESSION_2026-05-01.md
- Bears on: correction (length stabilisation on continuous state, n=1); EM vs WM (n=1); hex.

### B23 Allocentric recoding at 12 headings with actuation noise
- Dates: 2026-08-20 .. 2026-08-23
- Question: at Habitat-like 12 headings with real-valued position, does allocentric recoding survive, and does 0.15 rad actuation noise on executed turns break it?
- Task / environment: H=12 headings, floor 0.509, scored rate 0.022 (vs torus 0.225).
- Arms: Vanilla (path-integrated), RoPE (index); conditions commanded / allocentric / allocnoise.
- Seeds / batch: n=3; 980 batches (budget later extended to 2000 and 4000).
- Result: 980 batches -- commanded +0.110 (Vanilla 0.618 vs 0.509), allocentric +0.263 (0.772 +/- 0.099), allocnoise +0.230 (0.739 +/- 0.045): noise costs -0.033. Budget: allocentric +0.264 (980), **+0.383 (2000; Vanilla 0.891 +/- 0.005)**, +0.286 (4000; bimodal basins, RoPE leaves floor 0.551). r = -0.996 acc vs final loss over 18 runs.
- Status: EXPLORATORY (n=3). The noise contrast is at the undertrained 980 budget only.
- Pre-registered? no.
- Caveats: "partial recovery" and "still climbing" both withdrawn; actuation-noise cost never re-measured at nb=2000; Habitat navmesh sliding (69-91% of forward moves) not modelled.
- Sources: CONTINUOUS_ALLOC.md, H12_BUDGET_CURVE.md (referenced, not read)
- Bears on: environment line (allocentric recoding); noise robustness of path integration without correction.

### B24 Drift probe: does the r=2 skewed basis inject accumulating drift?
- Dates: 2026-09-04
- Question: does the state residual after net displacement grow with t, and does it explain the rank effect's length dependence?
- Task / environment: torus, trajectories from env seed 10000, 24 per seed; exact cumsum of the linear map (no forward pass); accuracy from RANK_SWEEP.json at T=1024.
- Arms: Vanilla (r=2), Vanilla_r4.
- Seeds / batch: n=8 each (RANK_SWEEP checkpoints).
- Result: residual (normalised) t=128/512/1024/2048: Vanilla 15.802/63.500/128.674/254.718 (growth 16.12x); Vanilla_r4 1.389/5.560/11.184/22.213 (15.99x). Ratio 11.38x-11.51x at every length (predicted 5.4x from opposition errors 0.495 vs 0.092). Within-arm r(drift, acc): Vanilla -0.363 (n=8; 95% CI ~ -0.83 to +0.42), r4 +0.086. Pooled r -0.614 flagged "must not be cited".
- Status: linear growth and 11x ratio CITABLE as a descriptive diagnostic (n=8, tight); drift-explains-accuracy UNMEASURED.
- Pre-registered? yes (point prediction 5.4x) -- wrong by 2x; direction and linearity held; link to accuracy not supported.
- Caveats: counterexample seed 0 (highest drift 319.5, accuracy 0.869); ~n=20 needed.
- Sources: DRIFT_PROBE.md
- Bears on: rank line (why r=4 wins with length is still unexplained); drift framing of correction.

### B25 Bump tokens in a walled maze -- dead-reckoning diagnostic only
- Dates: 2026-07-22
- Question: open-loop theta over commanded actions in a maze where 28.1% of moves are blocked; can directional bump tokens restore exact dead reckoning?
- Task / environment: maze_varying (the accuracy metric is a planner-demonstration task, VOID -- see Excluded).
- Result surviving as an environment diagnostic: dead-reckoning error after 256 steps 5.99 cells (random guess ~6.0) vs free torus 0.00; generic bump token 5.77 vs 5.76 (provably cannot help: context-free per-token map); directional BUMP_a -> -delta(a) gives 0.00 by construction.
- Status: EXPLORATORY (diagnostic; accuracy rows void).
- Pre-registered? no.
- Caveats: the nobump/bump accuracy (0.503 vs 0.522) is on a task an action-only 1-gram solves at 0.650 (PLANNER_TASK_AUDIT.md).
- Sources: BUMP_TOKEN_RESULTS.md, PLANNER_TASK_AUDIT.md
- Bears on: environment line (open-loop cumsum under blocked moves -- same premise as allocentric recoding).

### B26 Cross-environment-class training (torus-with-landmarks + DoorKey)
- Dates: 2026-05-13 / 2026-05-15
- Question: can one architecture handle torus and MiniGrid DoorKey-8x8 in a 50/50 mix?
- Task / environment: 30 train / 30 held-out envs per class; torus built with 200 landmarks (`environment_multiclass.py`, torus_n_landmarks=200); T=128 train, T=512 OOD; 50 ep x 128 batches x 64, lr 3e-4.
- Arms: RoPE, Vanilla, Level15, Level15GSF_NoDrop_K16.
- Seeds / batch: n=3 (seed 0 earlier, seeds 1-2 later run).
- Result (torus T=128 / T=512; DoorKey T=128 / T=512): RoPE 0.548/0.497; 0.890/0.788. Vanilla 0.813/0.681 +/- 0.091; 0.957/0.841. Level15 0.925/0.879 +/- 0.039; 0.946/0.888. GSF_NoDrop_K16 0.925/0.865; 0.945/0.891.
- Status: EXPLORATORY, recommend OMIT: torus half is the May landmark regime whose multi-size and multi-env siblings were voided (archive/void/MULTISIZE_RESULTS.md, MULTIENV_*), and Vanilla non-convergence on landmark torus reproduces under current code (B8).
- Pre-registered? no.
- Caveats: seeds not in one batch; no convergence reported.
- Sources: MULTICLASS_RESULTS.md, MULTICLASS_MULTISEED_RESULTS.md, run_multiclass_multiseed.sh, environment_multiclass.py
- Bears on: environment transfer (weak).

---

## Excluded

| file / claim | reason | killed by |
|---|---|---|
| All lm200 rows in NOBYPASS, V3, V4, V4_MULTISEED, V4_CONTROL, NODROP_PARETO, LEVEL15BETA, GSF_FULL, CAPACITY_PERREGIME, MULTISEED_FOLLOWUP (incl. "v4 +3.4pp on lm200", "NoDrop +13pp", "Beta +12pp", "GSF closes 95% of TEM gap") | April lm200 checkpoints never converged (CE ~1.0) | CLAUDE.md RETRACTION 2026-07-16; CORRECTED_LM200_LEADERBOARD.md; file banners |
| CAPACITY_CONTROL.md "CAPACITY" verdict | lm200-only, stuck baselines; superseded | CAPACITY_PERREGIME.md banner; RETRACTION |
| "TEMFaithful is the lm200 leader"; "multiple fixes close the TEM gap" | no gap: fresh Level15 0.990 > TEMFaithful 0.974 | LM200_CORRECTED_MULTISEED.md |
| Level15Cascade lm200 win (0.949 vs 0.841; zero-shot 0.991 vs 0.785) | compared against stuck April Level15; fresh Level15 = CascadeNoSlow = 0.996 | CASCADE_REPRO_TEST.md; archive/void/CASCADE_NOSLOW_CONTROL.md; CLAUDE.md |
| R_T_DISTRIBUTION_3WAY.md and V3's R_t table; "PC drives R_t to the -5 clamp (autoencoder bypass)"; NoBypass |theta_hat| ~3840 | lm200 seed-0 April checkpoints | archive/void/R_T_DISTRIBUTION.md, archive/void/LENGTH_DIAGNOSTIC.md (same data) |
| AUX_COEF_SWEEP, CLONE_TRANSFER_TEST (lm200), GSF_MODES_DIAGNOSTIC, GSF_NODROP_RESULTS, GSF_RESULTS, DROPOUT_ABLATION_RESULTS, VANILLANODROP_CONTROL, SPARSE_LANDMARKS, OMEGA_RESCALE_lm200, PER_VISIT_lm200, LONG_SEQ_lm200, ZERO_SHOT_TRANSFER_lm200*, MODEOMEGA | lm200 | archive/void/README.md |
| HIPPOCAMPAL_ANALYSIS.md Test B (R_t at landmarks: ordering aliased<landmark<blank) | run on `--config-R lm200` April checkpoints | hippocampal_analysis.py defaults; RETRACTION |
| EXTRAHEAD_CONTROL.md (Hopfield vs ExtraHead cross-scale), LEVEL15EM_CROSSSCALE.md, PERSCALE_OMEGA_RESULTS.md, MULTISEED_FOLLOWUP_RESULTS.md (cross-topology/scale/multi-env) | all use multi-size / multi-env torus with 200 landmarks (`run_level15em_crossscale.sh`, `run_multiseed_followup.sh`), sharing the Level15/Vanilla rows of voided MULTISIZE_RESULTS / SINGLE_SIZE_CONTROL / MULTIENV_*. NOTE: RESULTS_INDEX lists these as "other current" -- unresolved classification | archive/void/MULTISIZE_RESULTS.md, SINGLE_SIZE_CONTROL.md, MULTIENV_RESULTS.md |
| VECTOR_NAV_V2_RESULTS.md | April/May lm200 checkpoints; also SUSPECT (no action-only n-gram control) | file banner; RETRACTION |
| BUMP_TOKEN_RESULTS.md accuracy rows | maze_varying planner task: action-only n-gram 0.650 vs chance 0.250 | PLANNER_TASK_AUDIT.md |
| DOG_RESULTS.md (max grid score 0.036) | DoG targets all-zero (unnormalised Gaussians cancel at d=0) -- VACUOUS, not negative; DOG_RESULTS_FIXED.md never produced | SESSION_2026-05-01.md; RESULTS_INDEX Known-open #3 |
| Original CNAV run (MSE on DoG) | degenerate near-zero minimum, all variants ~20-cell error (chance) | commit 57084cc message |
| STOCHASTIC_TRANSITION_RESULTS.md (torus equivalence of action-record vs execution noise) | never landed; the equivalence is asserted, not measured | RESULTS_INDEX Known-open #4 |
| "Kalman win is inference / measurement-driven" | no component load-bearing at n=5; benefit flat in drift on MQ (x2); ExtraHead ties on lm200 | L15_ABLATION.md; MQ_NOISE_2X2*.md; LM200_ABLATION.md; RESULTS_INDEX Live negatives |
| n=1 L15 decomposition: "ConstR worse than nothing (0.672 < 0.833)", "Level15 does not reduce to clamping theta (0.831 vs 0.993)", "per-token gate is load-bearing" | sign inverted / unmeasured at n=5 | L15_ABLATION.md |
| "Level 1.5 +8pp clean OOD T=512" as an accuracy effect at matched recipe | at 300-ep cosine raw +0.057 (MDE 0.150, n=5) / +0.077 (MDE 0.077, n=12), unmeasured; only loss-matched survives | L15_ABLATION.md, L15_LOOP_2X2.md |
| "v4 win is RNG init drift" / "v4 win from grad-clip side effects" | control byte-identical to Level15 on clean; no surviving v4 win (lm200 void) | V4_CONTROL_RESULTS.md; CLAUDE.md CORRECTED 2026-09-05 |
| "Level15 is 0.938 on the paper task" (16 ep) | budget false negative; 1.000 at 50 ep | LEVEL15_MEETS_GATED_paper50.md |
| MQ noise p=0.10 floor = chance 0.0625; automatic PASS of noisy context destruction | ~45% of score non-positional; floor ~0.15 | MATCH_QUERY_NOISE_ABLATION.md |
| "Loop arm never fails 8/8 >= 0.77" (Match-Query) | one lucky batch; pooled 1/16 failures | REFINE_RESULTS.md corrected analysis |
| "Allocentric recoding generalises only partially at H=12" / "still climbing with budget" | budget artifact; nb=4000 goes back down | CONTINUOUS_ALLOC.md corrections; H12_BUDGET_CURVE.md |
| Drift probe pooled r = -0.614 | pooled across the manipulation, confounded | DRIFT_PROBE.md |
| SESSION_2026-05-01 "defensible claims" (+8pp clean, +11pp landmarks, +10pp noise, NLL 2x) | rest on LinearLR / lm200 numbers since retracted or unmeasured at matched recipe | L15_ABLATION.md; RETRACTION; RESULTS_INDEX |

## Cross-line dependencies

- **B7 (noisy Match-Query floor ~0.15)** conditions the loop line's noisy-task gain (Looped - Vanilla +0.121 / +0.057 at p=0.10).
- **B2 (L15_LOOP_2X2)** is shared with the loop line; the loop's T=128 win being convergence (loss-matched +0.006) lives there.
- **B3** is the source of "loop beats Kalman under noise" (+0.138 / +0.205 vs +0.023 / +0.004), a loop-line claim flagged as needing a pre-registered replication.
- **B4** (REFINE_RESULTS) supplies the Match-Query cross-batch non-reproducibility finding (mean per-seed drift 0.185) used by method notes and the loop/rank lines; it conflicts with B13's exact Vanilla reproduction.
- **B8/B9**: RoPE 0.482 and MambaLike 0.549 on fresh lm200 are used by the baselines/necessity line; lm200 context destruction (-0.830) is the only rule-2 gate on landmark torus.
- **B10** licenses every April-May clean/noise number used anywhere (TEM line, EM/WM early rows, baseline tables).
- **B16** (v4 control byte-identical) is cited by the forget-gate line (`run_forget_control.sh` header cites V4_MULTISEED as the source of the control idea).
- **B24** (DRIFT_PROBE) belongs to the rank line: the r=4 length effect remains unexplained.
- **B23** belongs to the environment/allocentric line (H12_BUDGET_CURVE.md).
- **B14/B15** feed the compositional/hierarchy and family-tree lines' baseline tables.
- **B21/B22** support RESULTS_INDEX's "hex emergence does not follow..." live negative, at n=1 only; the Sorscher test itself (DoG) was never validly run.
- Correction-as-"stabilisation" (B1, B2, B11, B14) is the same OOD-length signature that the clock/map, rank, PoPE and forget-gate lines report; no line explains that axis.

## Unresolved source disagreements

1. **lm200 context destruction**: `RESULTS_INDEX.md` (regenerated 2026-09-06) says lm200 "has never been through rule 2 ... Blocked"; `LM200_ABLATION.md` (2026-08-21) ran it on fresh checkpoints and it passed. The index is stale, but the gate was at T=128 with p_action_noise 0.0 (run_correction_gaps.sh), not on the p=0.10 lm200 config or at T=512.
2. **Capacity control citation**: `RESULTS_INDEX.md` cites `EXTRAHEAD_CONTROL.md` for "a filter-free capacity control ties it on lm200"; that file is the cross-scale Hopfield control (void-sibling data). The actual lm200 tie is in `LM200_ABLATION.md`.
3. **Match-Query reproducibility**: `LEVEL15_MEETS_GATED_matchq.md` reproduces Vanilla to four decimals across batches; `REFINE_RESULTS.md` and CLAUDE.md say Match-Query does not reproduce across batches (fast-attn runs). Plausibly recipe/kernel dependent; not resolved in any file.
4. **NOISE_REFINE gate statistics** (mean|g| 0.144, max 0.320, 7/9 positive) and Level15-under-noise t values exist only in CLAUDE.md, not in NOISE_REFINE.md/.json.
5. **MQ noise convergence numbers**: MQ_NOISE_2X2.md reports p=0.10 final loss 3.823/3.443/2.976/3.101; `run_mq_noise_c2.sh` header quotes "final MATCH loss" 2.526/2.367/2.067/2.198 for the same take; CLAUDE.md quotes take-2 p=0.10 match loss "1.97-2.23" vs file 2.543-3.442. Likely total vs match-only loss; not stated.
6. **MQ noise gates**: CLAUDE.md quotes ngram1 0.044 -> 0.059, ngram3 0.080 -> 0.085, never-moved 0.089 -> 0.120; `MATCH_QUERY_GATES_P010.md` (600 episodes) gives 0.0647 / 0.0628 / 0.1042 at p=0.10. MATCH_QUERY_NOISE_ABLATION notes the 120-episode estimate (0.1204) was biased high.
7. **Same checkpoints, different numbers**: clean T=512 Level15 per-seed 0.991/0.994/0.998 (V4_MULTISEED) vs 0.991/0.990/0.998 (V4_CONTROL); v4 0.998/0.994 vs 0.995/0.992; noise Level15 T=512 0.702 (NODROP/BETA) vs 0.692 (GSF_FULL) vs 0.707 (CAPACITY_PERREGIME) vs 0.866 (CASCADE long-seq) vs 0.853 (revalidation, seed 0, in-env). Eval protocol/sampling differences; no file reconciles them.
8. **Cross-scale / multiclass classification**: RESULTS_INDEX lists EXTRAHEAD_CONTROL, LEVEL15EM_CROSSSCALE, PERSCALE_OMEGA, MULTISEED_FOLLOWUP_RESULTS, MULTICLASS as current, while the archive voided MULTISIZE/SINGLE_SIZE/MULTIENV built on the same landmark regime and shared rows. Treated here as excluded (cross-scale) / omit-recommended (multiclass).
9. **Clone separation**: CLAUDE.md "PC best theta_hat separation 0.619 vs 0.573 vs 0.395" vs CLONE_TRANSFER_NOBYPASS (clean s0) where Vanilla +0.271 > PC +0.235. Different scripts/metric versions; n=1 both.

## Files read

L15_ABLATION.md, _L15_LOOP_RAW.md, L15_LOOP_2X2.md, LEVEL15BETA_RESULTS.md, LEVEL15EM_CROSSSCALE.md, NOBYPASS_RESULTS.md, NODROP_PARETO_RESULTS.md, V3_RESULTS.md, V4_RESULTS.md, V4_MULTISEED.md, V4_CONTROL_RESULTS.md, GSF_FULL_RESULTS.md, CASCADE_MULTISEED_RESULTS.md, CASCADE_REPRO_TEST.md, CASCADE_ZEROSHOT_S0.md, SESSION_HIERARCHICAL_CASCADE.md, CAPACITY_CONTROL.md, CAPACITY_PERREGIME.md, EXTRAHEAD_CONTROL.md, LM200_ABLATION.md, LM200_CORRECTED_MULTISEED.md, CORRECTED_LM200_LEADERBOARD.md, NOISE_REFINE.md, NOISE_REFINE.json (keys), REFINE_RESULTS.md, MQ_NOISE_2X2.md, MQ_NOISE_2X2.json, MQ_NOISE_2X2_C2.md, MQ_NOISE_2X2_C2.json, MATCH_QUERY_NOISE_ABLATION.md, MATCH_QUERY_GATES_P010.md, R_T_DISTRIBUTION_3WAY.md, CLONE_ANALYSIS_LEVEL15PC.md (empty), CLONE_TRANSFER_NOBYPASS.md, HIPPOCAMPAL_ANALYSIS.md, HIPPOCAMPAL_GRID.md, HIPPOCAMPAL_GRID_FREE.md, HIPPOCAMPAL_GRIDL15PC.md, HIPPOCAMPAL_HIDDEN.md, HIPPOCAMPAL_HIDDEN_GRIDFREE.md, HIPPOCAMPAL_LEVEL15PC.md, DOG_RESULTS.md, CNAV_RESULTS.md, CNAV_HEX_Vanilla.md, CNAV_HEX_VanillaEM.md, CNAV_HEX_Level15.md, CNAV_HEX_Level15EM.md, CONTINUOUS_ALLOC.md, DRIFT_PROBE.md, VECTOR_NAV_V2_RESULTS.md, BUMP_TOKEN_RESULTS.md, MULTICLASS_RESULTS.md, MULTICLASS_MULTISEED_RESULTS.md, MULTISEED_FOLLOWUP.md, MULTISEED_FOLLOWUP_RESULTS.md, LEVEL15_MEETS_GATED_paper.md, LEVEL15_MEETS_GATED_paper50.md, LEVEL15_MEETS_GATED_matchq.md, OMEGA_RESCALE_clean.md, PERSCALE_OMEGA_RESULTS.md, SESSION_2026-05-01.md, KNOWN_BUGS.md.
Exclusion reading: RESULTS_INDEX.md, N3_AUDIT.md, archive/void/README.md (+ listing), CLAUDE.md (grep and relevant sections), paper/RETRACTED.md.
Followed references: CORRECTION_COMPOSITIONAL.md, NOISE_CLEAN_REVALIDATION.md, FAMILY_TREE_WM_GAP.md (first 80 lines), PLANNER_TASK_AUDIT.md (maze_varying row), archive/void/{CASCADE_NOSLOW_CONTROL, CLONE_TRANSFER_TEST, R_T_DISTRIBUTION, HEX_EMERGENCE_RESULTS, MULTISIZE_RESULTS, SINGLE_SIZE_CONTROL, MULTIENV_RESULTS, MULTIENV_CLEAN_2x2, MODEOMEGA_RESULTS, OMEGA_RESCALE_lm200, DROPOUT_ABLATION_RESULTS, VANILLANODROP_CONTROL, GSF_MODES_DIAGNOSTIC, GSF_NODROP_RESULTS, AUX_COEF_SWEEP, LENGTH_DIAGNOSTIC, SPARSE_LANDMARKS_RESULTS, ACTIVE_INFERENCE_RESULTS, STATE_PROBES, VECTOR_NAV_RESULTS}.md (banners/heads); scripts: run_l15_ablation.sh, run_noise_refine.sh (header), run_l15_loop_2x2.sh, run_mq_noise_2x2.sh, run_mq_noise_c2.sh, run_correction_gaps.sh, run_multiclass_multiseed.sh, run_level15em_crossscale.sh, run_multiseed_followup.sh (grep), orchestrator.py (config), train_variant.py (defaults), train_multiclass.py, environment_multiclass.py, hippocampal_analysis.py, hippocampal_hidden_eval.py, r_t_distribution_test.py (grep); RECIPE_CHOICE.json; mq_noise logs; git logs for dates and commit 57084cc.

## Files in scope not covered

None of the listed files were skipped. Related but not in the brief's list and not read: RESULTS_PAPER.md (PARTIAL; source of the n=3 April "+11pp noise T=512 0.851 vs 0.739" figures, which differ from the 0.702 vs 0.638 in B17), LONG_SEQ_clean.md, PER_VISIT_clean.md, ZERO_SHOT_TRANSFER_clean*.md, NUMBERLINE_RESULTS.md, H12_BUDGET_CURVE.md, TEM_*.md, MINIGRID_DOORKEY_*.md (Level15 on MiniGrid), RECIPE_POWER.md, RESULTS_LEVEL2 / RESULTS_LEVEL15* (Level 2 / Level 1.5 early files, superseded by RESULTS_PAPER per CLAUDE.md).
