# E. The axes of the phase mechanism (MapWM side)

## Overview

This line asks what the content-dependent phase increment `Delta = W_out W_in x` (accumulated by
cumsum, applied as a rotation) must be able to do, one axis at a time: its **rank** r, its
**sign**, the extra machinery other generators add (Selective RoPE's conv / gate / full rank, an
explicit CoPE-style gate, a forget gate, PoPE's magnitude), and whether its value depends on the
**task** (map vs contextual counting). What survives:
- **Rank**: r=4 beats the paper's r=2 by +0.085 at T=1024 (8/8) for 384 parameters; a step, flat to
  r=32; r=2 learns a skewed action basis. Holds on MapWM (and the geometry replicates on MapEM); does
  NOT transfer measurably to MapPoPE. The D x r "packing geometry" account is withdrawn.
- **Sign**: at matched loss, signed path integration beats an index code (+0.123/+0.195 at
  T=512/1024, 12/12) and a monotone increment beats it nowhere; monotone codes cannot cancel opposite
  actions (opposition 1.85-1.98 vs 0.11-0.13). A PRIOR-ART REPLICATION (Sarrof, Grazzi, Selective
  RoPE Sec 4.2) extended to navigation with a one-operation isolation.
- **Task dependence**: on a k-back task with uncounted filler, an index code fails (+0.750 for path
  integration, 8/8; reproduces CoPE), a monotone constraint costs nothing measurable, the free arm
  learns a ballistic accumulator (alpha 0.967 vs 0.591 on torus), and a content gate is most of the
  counting mechanism by intervention (+0.594, 8/8). Scope corrected by audit: recency does not
  *need* a clock.
- **Add-ons**: Selective RoPE's generator is no better than MapFormer's (per-knob attribution
  confounded); an explicit what/where gate separates (4.16x) and buys nothing; the forget gate's
  +0.086 at r=2 is real but its mechanism is unidentified; PoPE's gain rises with length, not grid.
- Borrowed benchmarks (Flip-Flop, MQAR) do not discriminate this axis. Language: literature only
  plus an underpowered enwik8 pointer.

Common recipe unless stated ("torus recipe"): clean torus paper task, grid 64, 16 obs types,
interleaved stream, revisit-masked loss, train T=128, 1 layer, 2 heads, d=128, 300 epochs, 98
batches x 128, warmup+cosine, lr 1e-3, held-out map (env-seed 10000), eval T=128/512/1024.
Measured blank floor on this task is 0.506 (`RESULTS_INDEX.md`). The torus retrains bit-identically
across batches (`FORGET_CONTROL.md`), so the r=2 `Vanilla` arm appearing with identical numbers
(0.993/0.944/0.834) in RANK_SWEEP, SELECTIVE_ROPE (torus), FORGET_GATE, FORGET_CONTROL and
POPE_WRAPPING grid 64 is the same deterministic run repeated, NOT independent replication (rule C27).

---

### E1 Rank sweep: r in {2,4,8,16,32} on MapWM
- Dates: 2026-09-04 (edited 2026-09-06)
- Question: is the paper's r=2 bottleneck under-provisioned; what is the cheapest route to
  Selective RoPE's torus gain?
- Task / environment: torus recipe (floor 0.506); eval T=128/512/1024, held-out map.
- Arms: `Vanilla` (r=2, 204,373), `Vanilla_r4` (204,757), `Vanilla_r8` (205,525), `Vanilla_r16`
  (207,061), `Vanilla_r32` (210,133). Only `bottleneck_r` differs.
- Seeds / batch: 8 per arm, one batch, torus recipe ("same recipe as the selective run").
- Validity gates: none task-specific beyond the paper-task gates; no rule-9 r(loss,acc) reported in
  the file.
- Result (`RANK_SWEEP.md`):
  | arm | T=128 | T=512 | T=1024 |
  |---|---|---|---|
  | Vanilla (r=2) | 0.993 ± 0.017 | 0.944 ± 0.028 | 0.834 ± 0.064 |
  | Vanilla_r4 | 1.000 ± 0.000 | 0.982 ± 0.005 | 0.919 ± 0.012 |
  | Vanilla_r8 | 1.000 ± 0.000 | 0.981 ± 0.018 | 0.925 ± 0.033 |
  | Vanilla_r16 | 1.000 ± 0.000 | 0.972 ± 0.015 | 0.913 ± 0.018 |
  | Vanilla_r32 | 1.000 ± 0.000 | 0.982 ± 0.010 | 0.928 ± 0.021 |
  Paired vs r=2 at T=1024: r4 +0.085 (t 3.57, 8/8, sign p=0.008), r8 +0.091 (8/8), r16 +0.079 (8/8),
  r32 +0.095 (8/8). At T=512: +0.038 / +0.037 / +0.028 / +0.038 (r4 and r32 at 8/8). Step at r=2,
  flat from r=4 to r=32. r=4 cuts T=1024 seed sd 0.064 -> 0.012. Selective RoPE's gate reaches
  +0.086 for +8,193 params vs +384 for r=4 ("21x fewer").
- Status: CITABLE (T=1024 contrast; T=512 is +0.038).
- Pre-registered? yes (header of `run_rank_sweep.sh`): three branches (flat = paper right; rises =
  under-provisioned; falls = bias load-bearing). The "rises" branch fired, contradicting the paper's
  App. A.7 justification as a training (not expressivity) claim.
- Caveats: torus only; trained T=128, the +0.085 is at 8x training length (at 4x it is +0.038).
  r=3 and r=1 untested ("r=1 -> 0.66" has no experiment anywhere in the repo). No loss-matched
  contrast reported. Parity not run.
- Sources: `RANK_SWEEP.md`, `run_rank_sweep.sh`, `RANK_SWEEP.json`.
- Bears on: rank axis; default r for every later arm (`*_r4` in sign, recency, gated, EM line).

### E2 Action geometry of the learned code by rank
- Dates: 2026-09-04
- Question: does widening r destroy the paper's displacement reading; why does r=2 lose?
- Task / environment: eval-only probe on E1 checkpoints.
- Arms: Vanilla (r=2), Vanilla_r4, Vanilla_r8, Vanilla_r32.
- Seeds / batch: 8 seeds (E1 batch).
- Validity gates: basis-invariant metrics; 2-plane energy is 1.0 by construction at r=2.
- Result (`ACTION_GEOMETRY.md`):
  | arm | opposition \|N+S\|/scale | 2-plane energy | \|cos(N,E)\| | obs/action norm |
  |---|---|---|---|---|
  | Vanilla r=2 | 0.4950 | 1.0000 | 0.7833 | 0.1394 |
  | Vanilla_r4 | 0.0922 | 1.0000 | 0.1754 | 0.0421 |
  | Vanilla_r8 | 0.0972 | 1.0000 | 0.2030 | 0.0470 |
  | Vanilla_r32 | 0.1109 | 0.9996 | 0.0855 | 0.0702 |
- Status: CITABLE as a description of learned codes (8 seeds). Causal route from skew to accuracy
  NOT tested (THEORY_NARRATIVE ledger #8).
- Pre-registered? no.
- Caveats: "the mechanism for the +0.085" is associated, not established by intervention.
  Interpretability recovered only by projecting onto top-2 singular directions.
- Sources: `ACTION_GEOMETRY.md`, `RANK_SWEEP.md`.
- Bears on: rank ("conditioning, not capacity"); links to sign via opposition (E13).

### E3 Effective rank of unconstrained angle maps
- Dates: 2026-09-04
- Question: does a full-rank angle map rediscover rank 2?
- Task / environment: eval-only on the Selective RoPE torus checkpoints (E15).
- Arms: NoBottleneck, SRoPEGen, GateAngle, ConvAngle (W_omega 64x128), Vanilla (r=2 by construction).
- Seeds / batch: 8.
- Result (`LEARNED_RANK.md`): top-2 energy / participation ratio: NoBottleneck 0.388 / 8.35;
  SRoPEGen 0.432 / 7.49; GateAngle 1.000 / 1.58; ConvAngle 1.000 / 1.18; Vanilla 1.000 / 2.00.
  Verdict: "The bias is NOT rediscovered" for the unconstrained arms.
- Status: EXPLORATORY (descriptive).
- Pre-registered? no.
- Caveats: the E15 arms also delete `path_integrator.omega` (confound, see E15). The file's
  open question is answered by E4 and E1 (training-time rank, not post-hoc).
- Sources: `LEARNED_RANK.md`. Bears on: rank.

### E4 Post-hoc SVD truncation of a trained full-rank map
- Dates: 2026-09-04
- Question: are the extra directions of NoBottleneck load-bearing?
- Task / environment: eval-only, torus, held-out map; exact SVD projection of trained W, bias untouched.
- Arms: NoBottleneck truncated to rank 1..64.
- Seeds / batch: 8.
- Result (`RANK_TRUNCATION.md`): rank 1 0.375/0.378; 2 0.418/0.392; 3 0.485/0.445; 4 0.594/0.539;
  8 0.877/0.810; 16 0.992/0.926; 64 0.994/0.966 (T=128/T=512). Truncating to rank 2 costs +0.576
  (T=128) / +0.574 (T=512).
- Status: EXPLORATORY; its reading ("post-hoc truncation is not a sufficiency test") is a method
  finding (rule 24). r=4 trains to 0.919 at T=1024 while full-rank truncated to 4 gives 0.594 (T=128).
- Pre-registered? no.
- Caveats: says nothing about whether rank 2 suffices; `Vanilla` (r=2 by construction) reaches 0.944
  at T=512. Same omega-deletion confound as E15.
- Sources: `RANK_TRUNCATION.md`, `RANK_SWEEP.md`. Bears on: rank; method.

### E5 Paper Fig. 4 reproduction on MapWM (r=2 and r=4)
- Dates: 2026-09-04
- Question: do the paper's four checkable Fig. 4 claims reproduce?
- Task / environment: eval-only on E1 torus checkpoints (`runs/rank_sweep/p0`).
- Arms: Vanilla (r=2), Vanilla_r4.
- Seeds / batch: 8.
- Result (`PAPER_FIG4_REPRO.md`):
  | claim | paper | Vanilla r=2 | Vanilla_r4 |
  |---|---|---|---|
  | C1 \|\|D_act\|\|/\|\|D_obs\|\| | >>1 | 25.0 ± 22.4 | 35.9 ± 19.0 |
  | C2 cos(left,right) | -1 | -0.729 ± 0.373 | -0.996 ± 0.004 |
  | C3 \|cos(left,up)\| | >>0 (paper's own limitation) | 0.779 ± 0.276 | 0.174 ± 0.083 |
  | C4 \|\|v_obs\|\|/\|\|v_act\|\| | >>1 | 0.57 ± 0.06 | 0.60 ± 0.04 |
  C1 reproduces; C2 direction only at r=2 (high variance; file says "reproduces", memory note says
  "weak"); C3 reproduces the paper's reported failure, and r=4 removes it (0.779 -> 0.174) with no
  regulariser (the paper proposes a bounded-energy constraint); C4 does not reproduce (inverted).
- Status: CITABLE as description (n=8).
- Pre-registered? no (C4 follow-up in E6 was).
- Caveats: the file cites "Sec 5.4 / Fig 4"; CLAUDE.md (corrected 2026-09-11) places the EM
  separate-pools discussion in App. C.3. Rule 25 was bought here: C3 is the paper's own reported
  result, not a failure to reproduce.
- Sources: `PAPER_FIG4_REPRO.md`, `.claude-memory/project_rank_and_selective_rope.md`.
- Bears on: reproduction of the paper; rank.

### E6 Paper Fig. 4 on MapEM, and the C4 metric check
- Dates: 2026-09-04
- Question: is C4 an EM-only property the caption does not scope? Does the r=2 skew replicate on EM?
- Task / environment: torus; new EM batch (`runs/em_fig4/p0`), eval-only probe.
- Arms: VanillaEM (r=2), VanillaEM_r4.
- Seeds / batch: 8 per arm, one batch (`run_em_fig4.sh`).
- Result (`PAPER_FIG4_EM.md`): VanillaEM r=2: C1 21.8 ± 25.2, C2 -0.352 ± 0.896, C3 0.868 ± 0.214,
  C4 0.90 ± 0.21; VanillaEM_r4: 26.1 ± 13.0, -0.996 ± 0.002, 0.190 ± 0.115, 0.64 ± 0.04.
  C4 under three readings (with LN / no LN / contextual): MapWM r=2 0.57/0.74/0.58; MapWM r=4
  0.60/0.87/0.60; MapEM r=2 0.90/1.25 ± 0.31/0.98; MapEM r=4 0.64/0.95/0.77. One of eight cells
  crosses 1; none "much bigger".
- Status: CITABLE as description; C4 is a recorded discrepancy with the paper on both backbones.
  The r=2 skew replicates on EM (more skewed than WM), and r=4 lands both on the same geometry.
- Pre-registered? yes (in `run_em_fig4.sh`): branch 2 fired ("EM ratio also < 1 -> genuine
  discrepancy"). Hypothesis "C4 is EM-only" not supported.
- Caveats: the motivating text calls MapWM's attention "additive"; that premise is withdrawn
  (`AUDIT_2026-09-10.md` #1), which does not change the measured numbers. The "no gradient pressure
  on action value vectors" hypothesis is untested. Position half of the double dissociation
  reproduces; content half does not.
- Sources: `PAPER_FIG4_EM.md`, `run_em_fig4.sh`. Bears on: paper reproduction; EM vs WM; rank.

### E7 D x r: rank threshold across dimension (D-dimensional torus)
- Dates: prelim 2026-09-04 (n=5), full 2026-09-05 (n=8)
- Question: is r=2's deficit a packing-geometry effect (predicts deficit growing with D/r, the
  proposed account of v4 Table 6's 5D collapse), or optimisation (r=D -> r=D+2 gap at every D)?
- Task / environment: `environment_nd.py`, directed walk, allocentric aliased obs, revisit-masked.
  D=2 grid 32 (1024 cells), D=3 grid 10 (1000), D=5 grid 4 (1024). Train T=128, eval T=128/512,
  fresh env (seed 10000), 100 trajectories. Measured majority-class chance 0.506 / 0.523 / 0.526.
- Arms: r in {2, D, D+2}: D=2 Vanilla, Vanilla_r4; D=3 Vanilla, Vanilla_r3, Vanilla_r5; D=5 Vanilla,
  Vanilla_r5, Vanilla_r7.
- Seeds / batch: 8, one batch, torus recipe (300 ep, lr 1e-3, cosine, `--data-workers 3`).
- Validity gates (`ND_GATES.md`, before GPU): action-stream n-gram orders 1-5 max 0.513/0.531/0.536
  vs chance 0.506/0.523/0.526; revisit rates 0.233/0.283/0.620; 29.8/36.3/79.3 scored per
  trajectory. All PASS. Rule 9: r(loss,acc) at T=128 -1.000 (D=2), -0.937 (D=3), -0.951 (D=5); at
  T=512 -0.697 / -0.752 / -0.182.
- Result (`DXR_RANK_THRESHOLD.md`), T=512 means: D=2 r2 0.848 ± 0.056, r4 0.959 ± 0.036; D=3 r2
  0.707 ± 0.077, r3 0.793 ± 0.092, r5 0.860 ± 0.055; D=5 r2 0.896 ± 0.076, r5 0.877 ± 0.072, r7
  0.950 ± 0.045. Paired vs r=2 at T=512: D=2 r4 +0.110 (MDE 0.076, 7/8) DETECTABLE; D=3 r3 +0.086
  (MDE 0.088) unmeasured, r5 +0.153 (MDE 0.048, 8/8) DETECTABLE; D=5 r5 -0.019 (MDE 0.099)
  unmeasured, r7 +0.055 (MDE 0.062) unmeasured. At T=128 D=3 and D=5 higher-r arms detectable
  (e.g. D=5 r5 +0.009, MDE 0.002, 8/8). `mapformer_math.tex` (audit): r=D+2 minus r=D = +0.110 /
  +0.067 / +0.073, DETECTABLE at D=2 (t 4.08, 7/8) and D=5 (t 3.35, 8/8), unmeasured at D=3 (t 1.93,
  6/8); at D=2 r=D is r=2, so only ONE independent detectable cell (D=5) supports the optimisation
  half.
- Status: geometric account REFUTED (falsifiers fired on levels: r=2's best score is at D=5; its
  deficit +0.110/+0.153/+0.055 does not grow with D). Optimisation half: EXPLORATORY (one
  independent detectable cell).
- Pre-registered? yes (`run_dxr.sh`, `mapformer_math.tex` sec 2.5/6.2, `paper_rank/sections/
  06_experiments_nd.tex`). 2 of 3 falsifiers fired. Table-6 account WITHDRAWN.
- Caveats: the motivating premise is counterfactual -- the paper's 5D run used r=5 (sec D.1) at grid
  5, l=16, and the paper's own account is a neuron-budget one. D=2 T=128 column is uninformative
  (r=-1.000). Cross-D revisit rate is a confound; only within-D contrasts are primary. The gate's
  G5 exponent fit disagrees with an un-scripted ladder (-1.04 vs -1.51) per the tex audit note.
  Prelim n=5 (`DXR_PRELIM.md`) is a subset of the same batch; superseded.
- Sources: `DXR_RANK_THRESHOLD.md`, `DXR_PRELIM.md`, `ND_GATES.md`, `run_dxr.sh`,
  `mapformer_math.tex` (sec "A prediction, and the paper's own 5D negative").
- Bears on: rank (scope: not geometry), paper's Table 6.

### E8 MapPoPE at r=4, plus the gated-P4 repair
- Dates: 2026-09-07
- Question: (Q-A) does r=4 help MapPoPE, the best arm on the paper task (which defaulted to r=2)?
  (P4) does an explicit gate help r=2 more than r=4?
- Task / environment: torus recipe (T=512/1024); recency (k_max=64, T=1024/2048) for P4.
- Arms: torus `MapPoPE_r4` (205,205, w_in.out_features==4 asserted), `MapPoPE-Flat` (r=2, 204,693),
  `Gated_r2`, `Vanilla` (r=2); recency `Gated_r2`, `Vanilla`.
- Seeds / batch: 8; one batch per task; recipes verbatim from `run_sign.sh` / `run_recency.sh`.
- Validity gates: rank survived construction (param difference 512). Rule 9: r(loss,acc) -0.81 /
  -0.68; flat seeds MapPoPE_r4 8/8, MapPoPE-Flat 6/8, Gated_r2 5/8, Vanilla 2/8.
- Result (`MAPPOPE_R4_RESULTS.md`):
  | arm | final loss | T=512 | T=1024 |
  |---|---|---|---|
  | MapPoPE_r4 | 0.00021 | 0.982 ± 0.018 | 0.941 ± 0.028 |
  | MapPoPE-Flat (r=2) | 0.00635 | 0.974 ± 0.011 | 0.921 ± 0.023 |
  | Gated_r2 | 0.03954 | 0.962 ± 0.048 | 0.887 ± 0.105 |
  | Vanilla (r=2) | 0.08863 | 0.915 ± 0.052 | 0.777 ± 0.094 |
  Q-A: MapPoPE_r4 - MapPoPE-Flat +0.009 raw / +0.006 matched (T=512), +0.019 / +0.015 (T=1024),
  5/8, unmeasured. MapPoPE-Flat - Vanilla: +0.058 raw (8/8) DETECTABLE, +0.024 matched (MDE 0.034);
  +0.144 raw (8/8) DETECTABLE, +0.082 matched (MDE 0.104) at T=1024. P4: Gated_r2 - Vanilla +0.046 /
  +0.026 (T=512), +0.110 / +0.073 (T=1024), recency T=2048 +0.015 (3/8): all unmeasured; cross-batch
  interaction (recency) +0.054.
- Status: Q-A DIRECTIONAL/unmeasured -- "use r=4" is a MapWM-family recommendation, not general.
  PoPE-over-path-integration: raw detectable, loss-matched unmeasured (partly a convergence gap).
  P4 unmeasured.
- Pre-registered? yes (`MAPPOPE_R4_PREREG.md`). Q-A: the pre-registered alternative ("does not
  transfer to PoPE's one-frequency-per-element layout") is the live reading. P4: author's recorded
  negative prior; data mildly against it, not established.
- Caveats: proposed reason (PoPE has n_blocks=64 vs 32) untested. The Vanilla baseline is the
  worst-converging arm. Gated_r2's torus numbers are identical to `GATED_TORUS.md`'s (determinism).
  P4 interaction half comes from a different batch.
- Sources: `MAPPOPE_R4_RESULTS.md`, `MAPPOPE_R4_PREREG.md`, `MAPPOPE_R4.md`.
- Bears on: rank scope; PoPE magnitude on top of path integration; the gated variant (E18).

### E9 MapPoPE vs PoPE across every task where PoPE was run (compilation)
- Dates: 2026-09-09
- Question: does adding path integration to PoPE ever hurt?
- Task / environment: compiled from existing batches (torus, paper OOD protocol, MiniGrid raw and
  allocentric, MiniWorld fixed/fresh/oracle, bounded memory, compositional). No new runs. Void files
  and the ungated CLOCK_SCAN excluded.
- Arms: PoPE (index phase) vs MapPoPE (path-integrated), matched on hierarchy.
- Seeds / batch: per source batch (torus n=8 stated; others not restated).
- Result (`MAPPOPE_VS_POPE.md`): wins torus 0.509 vs 0.994 (+0.485); paper OOD 0.508/0.226/0.799/0.804
  vs 1.000/0.995/0.999/0.996; small wins MiniGrid allocentric, MiniWorld fixed, bounded memory.
  Losses: MiniGrid raw flat T=1024 0.953 vs 0.919 (-0.034); MiniWorld fresh raw 0.384 vs 0.308,
  allocentric 0.364 vs 0.232; MiniWorld oracle 0.938 vs 0.324 (-0.614); compositional longer T.
- Status: EXPLORATORY compilation; the MiniWorld MapPoPE cells carry the non-convergence flag (train
  loss > 1.5) and were never re-run at the cosine / lr 1e-3 recipe.
- Pre-registered? no.
- Caveats: cross-batch; MapPoPE here is r=2 throughout. Losses concentrate on rotation-action
  environments (a diagnosed failure owned by the environment line).
- Sources: `MAPPOPE_VS_POPE.md`. Bears on: PoPE/encoding axis; environment line (rotation actions).

### E10 PoPE wrapping: does PoPE's gain scale with octaves or with length?
- Dates: 2026-09-05
- Question: omega spans [2pi/N, 2pi] over 32 blocks (log2 N octaves); does MapPoPE's gain over
  Vanilla rise with grid size (octaves) and with eval length?
- Task / environment: torus at grids 16/32/64/128 (grid 8 excluded by gates: 82 scored per
  trajectory, different majority rate 0.472), T=128/512/1024.
- Arms: Vanilla (204,373) vs MapPoPE (204,693), +320 params at every grid.
- Seeds / batch: 8, one batch, torus recipe.
- Validity gates: revisit rate plateaus 0.225 from grid 32 up; ceiling rows flagged; rule 9 per grid.
- Result (`POPE_WRAPPING.md`) raw gain at T=1024: grid 16 +0.139 (MDE 0.455), 32 +0.122 (MDE
  0.137), 64 +0.101 (MDE 0.065, 8/8, DETECTABLE), 128 +0.123 (MDE 0.139). Loss-matched at T=1024:
  grid 16 -0.077 (1/8; r -0.933), 32 +0.077 (6/8), 64 +0.095 (8/8 DETECTABLE; r -0.167), 128 +0.079
  (5/8). Length at fixed grid: grid 32 +0.001/+0.057/+0.122; grid 64 +0.007/+0.037/+0.101; grid 128
  +0.004/+0.042/+0.123 (T=128/512/1024). Grid 16 bimodal (Vanilla converges 2/8, MapPoPE 4/8).
- Status: octave prediction REFUTED (flat loss-matched gains across 5/6/7 octaves); length trend
  holds 3/3 (only grid 64 T=1024 individually detectable). DIRECTIONAL for the length trend.
- Pre-registered? yes (`run_popewrap.sh` header): rise with grid size (failed), rise with length
  (held 3/3).
- Caveats: "helps more at OOD length" is the project's universal unexplained signature; the one
  distinctive prediction failed. Corrected account (wraps set by path length, not map extent) is
  post hoc. MapPoPE at r=2.
- Sources: `POPE_WRAPPING.md`, `run_popewrap.sh`. Bears on: PoPE magnitude; OOD-length axis.

### E11 Sign ablation: may the phase increment be negative?
- Dates: 2026-09-06
- Question: does a non-negative (monotone) increment, as in CARoPE / GRAPE-AP / CoPE, destroy the
  map? Isolation: `|W_out W_in x|` vs `W_out W_in x`, nothing else.
- Task / environment: torus recipe (T=128 train; eval 128/512/1024), floor 0.506.
- Arms: `Signed_r4` (baseline), `Vanilla_r4` (RNG/construction control), `Abs_r4` (primary),
  `Pos_r4` (softplus, GRAPE-AP style), `CARoPE_r4` (1/(softplus+1), CARoPE verbatim), `RoPE` (index).
  Params 204,757 in all five MapFormer arms per training log (the prereg says 205,785; see
  disagreements).
- Seeds / batch: 12 per arm, one batch, 300 ep, cosine, lr 1e-3.
- Validity gates: identical params; Signed_r4 bit-identical to Vanilla_r4 on shared weights (0.0);
  Delta >= 0 at init and after training (`SIGN_PROBE.md`: nonneg True, frac_neg 0.000); 8-epoch smoke
  of all arms. Rule 9: r(loss,acc) -0.978 / -0.856 / -0.729 at T=128/512/1024 over 72 runs.
  Control Vanilla_r4 - Signed_r4: +0.000 / +0.006 / +0.006, unmeasured at every length.
- Result (`SIGN_ABLATION.md`):
  | arm | T=128 | T=512 | T=1024 | final loss |
  |---|---|---|---|---|
  | Signed_r4 | 1.000 ± 0.000 | 0.978 ± 0.014 | 0.922 ± 0.027 | 0.0002 |
  | Vanilla_r4 | 1.000 ± 0.000 | 0.984 ± 0.009 | 0.927 ± 0.015 | 0.0002 |
  | Abs_r4 | 0.946 ± 0.070 | 0.675 ± 0.040 | 0.558 ± 0.027 | 0.1708 |
  | Pos_r4 | 0.977 ± 0.035 | 0.798 ± 0.080 | 0.584 ± 0.080 | 0.0676 |
  | CARoPE_r4 | 0.900 ± 0.138 | 0.809 ± 0.133 | 0.645 ± 0.090 | 0.2929 |
  | RoPE | 0.799 ± 0.018 | 0.449 ± 0.083 | 0.345 ± 0.118 | 0.7844 |
  Primary Abs - Signed, loss-matched: T=128 -0.006 (MDE 0.018) unmeasured; T=512 -0.215 (MDE 0.055,
  12/12 neg) DETECTABLE; T=1024 -0.280 (MDE 0.061, 12/12) DETECTABLE (raw -0.303 / -0.363).
  Pos - Signed matched -0.145 (T=512), -0.305 (T=1024) DETECTABLE; CARoPE - Signed matched -0.018
  (T=512, unmeasured), -0.134 (T=1024, MDE 0.118) DETECTABLE.
  Signed - RoPE matched: -0.021 (T=128, MDE 0.013, detectable negative), +0.123 (T=512, MDE 0.075,
  0/12 neg), +0.195 (T=1024, MDE 0.107, 0/12 neg). Abs - RoPE matched: -0.028 (T=128, MDE 0.025,
  detectable negative), -0.092 (T=512, MDE 0.103), -0.085 (T=1024, MDE 0.140), unmeasured.
  Training loss: every constrained arm worse than Signed on 12/12 seeds (Abs +0.1706, Pos +0.0674,
  CARoPE +0.2927). Degradation T=128 -> 1024: Signed -0.078, Abs -0.387, Pos -0.393, CARoPE -0.255,
  RoPE -0.454.
- Status: CITABLE (Abs - Signed at OOD; Signed - RoPE at OOD) and PRIOR-ART REPLICATION.
- Pre-registered? yes (`SIGN_ABLATION_PREREG.md`), with an AMENDMENT written at 48/72 checkpoints
  before any result was read. H1 mechanically fired the "OOD length only -> NOT the predicted result"
  branch, but that branch required a deficit in a T=128 cell at ceiling (1.000 ± 0.000; headroom
  0.054 vs MDE 0.057) -- a design error recorded in the file. The training-length effect is in the
  loss (12/12). H2 (Pos/CARoPE track Abs, not below RoPE) held in direction. H3 reported.
- Caveats: Pos and CARoPE carry an init confound (they start as RoPE; signed arms start near NoPE).
  Loss-matching at T=128 partials out the constraint's own effect. OOD-only accuracy effect is also
  the project's generic "helps at OOD length" signature. Torus only; language half of the
  prediction not tested.
- PRIOR ART: the axis and the parity theorem are Sarrof et al. (2405.17394), Grazzi et al.
  (2411.12537, ICLR 2025; negative eigenvalues), adopted by PaTH, discussed by RWKV-7, and carried
  into content-dependent rotation in attention by Selective RoPE Sec 4.2 (single-layer parity).
  CARoPE, GRAPE-AP and CoPE are non-negative and cite none of it. New here: navigation regime and the
  one-operation isolation at identical parameters with an RNG-path control.
- Sources: `SIGN_ABLATION.md`, `SIGN_ABLATION_PREREG.md`, `_SIGN_RAW.md`, `run_sign.sh`,
  `papers/INDEX.md`, `.claude-memory/project_sign_axis.md`.
- Bears on: sign axis; clock vs map; "path integration, not encoding" headline requires the sign.

### E12 Sign probe: constraint integrity and opposition
- Dates: 2026-09-06
- Question: do monotone arms actually fail to cancel opposite actions, and do signed arms use the sign?
- Task / environment: eval-only on E11 checkpoints, 512-step streams.
- Arms: Signed_r4, Abs_r4, Pos_r4, CARoPE_r4, Vanilla_r4.
- Seeds / batch: 12.
- Result (`SIGN_PROBE.md`): opposition x/y and |cos(x,y)|: Signed 0.125/0.106/0.218 (frac_neg
  0.503); Vanilla_r4 0.128/0.130/0.133 (0.512); Abs 1.849/1.855/0.587; Pos 1.905/1.885/0.723; CARoPE
  1.981/1.977/0.930 (Delta range +0.028..+1.000). All constrained arms nonneg True.
- Status: CITABLE (description of learned code; pre-stated reading fired).
- Pre-registered? yes (reading stated in the probe before results: signed near 0, constrained near 2).
- Caveats: opposition 2.0 = identical; monotone codes cannot reach 0 by construction, so the result
  is partly definitional for Abs; the informative part is that the signed arm DOES reach ~0.1.
- Sources: `SIGN_PROBE.md`. Bears on: sign; clock/map.

### E13 Localisation: does the OOD benefit live in under-trained channels? (and alpha)
- Dates: 2026-09-06
- Question: import of the long-context "critical dimension" account -- low-frequency channels that
  never complete a cycle at training length are read at unseen phases OOD.
- Task / environment: eval-only, torus; phase ablation theta_c = 0 on k lowest / highest / random
  channels, k in {4,8,16,32}, at T=128 and T=1024.
- Arms: Signed_r4, Abs_r4 (sign batch, 12 seeds); Vanilla (r=2), Vanilla_r4 (rank sweep, 8 seeds).
- Seeds / batch: as stated; two independent batches.
- Result (`LOCALISATION.md`, `LOCALISATION_RANK.md`):
  P1: under-trained channels at T=128: Signed 8.2/64, Abs 17.0, Vanilla_r4 7.5, Vanilla 13.7 (PASS).
  P2: ablate k=32 lowest-frequency channels: Signed_r4 -0.001 (T=128) vs -0.139 ± 0.092 (T=1024);
  Vanilla_r4 -0.001 vs -0.170 ± 0.084; Vanilla r=2 -0.030 vs -0.254 ± 0.164; Abs_r4 -0.280 vs -0.211
  (floor artifact: baseline 0.558 vs floor 0.506). High-frequency k=32 costs 0.42-0.57.
  P3: range(S) growth exponent alpha: Signed_r4 0.518, Vanilla_r4 0.524, Vanilla r=2 0.619, Abs_r4
  0.943; r(opposition, alpha) = +0.9995 across the four arms from two batches.
- Status: P2 = critical-dimension import REFUTED (sign backwards on all clean arms). P3 = EXPLORATORY
  (four points, correlational); per `.claude-memory/project_clock_vs_map.md` alpha is a
  re-description of opposition (economy, not a third cause), and "vary alpha" is malformed (a fitted
  statistic, not a parameter). The association alpha <-> OOD degradation is established; its route
  is not.
- Pre-registered? yes (`LOCALISATION_PREREG.md`). P1 passed; P2 refuted; P3 confirmed as ordering.
- Caveats: LOCALISATION's suggested direct test ("the InEKF's wrap bounds the accumulator") failed in
  E14. Signed_r4 torus alpha is 0.518 here and 0.591 in `RECENCY_H2.md` (different analyses; see
  disagreements).
- Sources: `LOCALISATION.md`, `LOCALISATION_RANK.md`, `LOCALISATION_PREREG.md`,
  `.claude-memory/project_clock_vs_map.md`.
- Bears on: sign and rank as one finding; OOD-length axis (still unexplained).

### E14 Accumulator: do the forget gate, PoPE or the InEKF bound the accumulator?
- Dates: 2026-09-06
- Question: does alpha explain the other OOD-length mechanisms?
- Task / environment: eval-only on existing torus checkpoints.
- Arms (paired within batch): Forget vs Vanilla (forget batch), MapPoPE vs Vanilla (popewrap grid 64),
  Level15 vs Vanilla (L15 ablation batch).
- Seeds / batch: n=6 / 6 / 5 in the paired contrasts.
- Result (`ACCUMULATOR.md`): delta alpha: Forget -0.051 (sd 0.081, MDE 0.092); MapPoPE +0.006 (sd
  0.155, MDE 0.177); Level15 +0.009 (sd 0.152, MDE 0.191) -- all unmeasured. P1 positive control:
  Level15 range(theta_hat) 285.6 vs range(theta_path) 283.9 at T=1024, alpha 0.621. P4: forget gate's
  sum(log gamma) grows 0.38 -> 2.75, alpha +0.956.
- Status: alpha changes UNMEASURED for all three; positive control FAILED (the InEKF wraps the
  innovation, not theta_hat) -> "a wrapped filter bounds the accumulator" WITHDRAWN. Accumulator
  account covers 2 of 4 mechanisms.
- Pre-registered? yes (docstring of `probe_accumulator.py`): P1 failed; P2, P3 "no change" consistent
  but unmeasured; P4 measured.
- Caveats: between-batch Vanilla alphas differ (file text 0.665 vs 0.608, larger than either
  effect); the table's pooled alphas differ from the text's per-seed figures (0.710 vs 0.631).
  The proposed intervention "bound the accumulator" remains untested.
- Sources: `ACCUMULATOR.md`, `probe_accumulator.py`, `run_accumulator.sh`.
- Bears on: OOD-length axis; correction line (InEKF); forget gate as a second accumulator.

### E15 Selective RoPE's angle generator vs MapFormer's
- Dates: 2026-09-03 (confound block 2026-09-05)
- Question: does Selective RoPE's generator (causal conv, no bottleneck, sigmoid gate) beat
  MapFormer's, and which knob matters?
- Task / environment: parity (train L=16, eval to 256, chance 0.5; path-int - index +0.078 at L=128
  per `run_selective.sh`) and torus recipe.
- Arms (parity params): RoPE 199,042; Vanilla 199,490; ConvAngle +193; NoBottleneck +7,873;
  GateAngle +8,193; SRoPEGen +16,385. Not parameter-matched by design. Placement kept MapFormer's
  (angle from token embeddings once before blocks), so not a faithful Selective RoPE.
- Seeds / batch: parity 16, torus 8; one batch; 300 ep, lr 1e-3, cosine.
- Result (`SELECTIVE_ROPE.md`): parity L=128 vs Vanilla (0.598): RoPE -0.078, ConvAngle -0.020 (MDE
  0.013), NoBottleneck -0.020 (MDE 0.018), GateAngle -0.030 (MDE 0.014) all DETECTABLE NEGATIVE;
  SRoPEGen -0.009 unmeasured. Torus T=512 vs Vanilla 0.944: ConvAngle -0.031 (MDE 0.048), NoBottleneck
  +0.022 (MDE 0.025), GateAngle +0.040 (MDE 0.030, 7/8), SRoPEGen +0.031 (MDE 0.030, 7/8); T=1024:
  conv -0.064 (t -1.99), no bottleneck +0.058 (t 2.13, 7/8), gate +0.086 (t 3.05, 8/8, sign p=0.008),
  all three +0.048 (t 1.36). GateAngle - NoBottleneck +0.018 (MDE 0.026) / +0.028 (MDE 0.045),
  unmeasured. Raw torus table in `_SELECTIVE_TORUS.md` (T=1024: RoPE 0.412, Vanilla 0.834, ConvAngle
  0.770, NoBottleneck 0.892, GateAngle 0.920, SRoPEGen 0.882).
- Status: full generator vs MapFormer UNMEASURED on parity and at torus T=1024 (+0.031 at T=512 is at
  its MDE). Per-knob attribution CONFOUNDED: every "single-knob" arm also deletes
  `path_integrator.omega`, `w_in`, `w_out`, swapping the readout diag(omega) W_out -> tau I. The
  sign flip between tasks survives only as an observation about the arms as built. EXPLORATORY.
- Pre-registered? partly (in `run_selective.sh`): conv should hurt on torus and be neutral on parity;
  hurt on both (neutral-on-parity half failed).
- Caveats: missing arm (keep diag(omega) W_out, add only conv or only gate) never run. Parity
  param counts differ from torus (torus Vanilla 204,373). A negative here does not refute Selective
  RoPE (different placement, different tasks). RANK_SWEEP is unaffected by the confound.
- PRIOR ART: Selective RoPE (2511.17388, ICLR 2026) and MapFormer (2511.19279) put the same
  content-dependent cumsum in the phase, posted 21 vs 24 Nov 2025, neither citing the other; Mamba-3
  (2603.15569) Prop 3 publishes data-dependent RoPE equivalence.
- Sources: `SELECTIVE_ROPE.md`, `_SELECTIVE_TORUS.md` (header text is a stale template from the
  refine-theta experiment; the numbers are the selective arms), `run_selective.sh`,
  `ALGORITHMIC_RESULTS.md`.
- Bears on: generator design; motivated E1 and E16.

### E16 Gate probe: does Selective RoPE's gate suppress observation tokens?
- Dates: 2026-09-03
- Question: explanation of GateAngle's torus win as token suppression.
- Task / environment: eval-only, 24 trajectories per seed.
- Arms: GateAngle (torus, parity).
- Seeds / batch: torus 8; parity 6 seeds (0-5) probed.
- Result (`GATE_PROBE.md`): torus gate on actions 0.5602 vs observations 0.4155, 1.35x (per seed
  1.15-1.58); parity bit=1 0.5267 vs bit=0 0.3419, 1.54x.
- Status: hypothesis FALSIFIED (descriptive; larger ratio on the task where the gate hurts).
- Pre-registered? yes ("gate(action) >> gate(observation) on torus, no split on parity", stated before
  looking). Failed.
- Caveats: the arm also carries the omega-deletion confound. 1.35x became the floor for E18.
- Sources: `GATE_PROBE.md`, `probe_gate.py`. Bears on: generator design; what/where separation.

### E17 Conv kernel probe
- Dates: 2026-09-04
- Question: what did Selective RoPE's causal conv (K=4) learn: identity (accumulate) or first
  difference (their Sec 3 derivation)?
- Task / environment: eval-only, projections of unit-normalised kernels.
- Arms: ConvAngle and SRoPEGen, parity (16) and torus (8).
- Result (`CONV_KERNEL_PROBE.md`): |.identity| 0.423-0.470, |.differencer| 0.345-0.463, |DC| 0.777-1.063
  vs a random-kernel baseline 0.424 / 0.424 / 0.798. Kernels sit at the random baseline.
- Status: EXPLORATORY (descriptive).
- Pre-registered? no.
- Caveats: K=4 is an assumption (paper states no width); K=2 fixed difference kernel not run.
- Sources: `CONV_KERNEL_PROBE.md`. Bears on: generator design.

### E18 Gated signed increment (CoPE's selection on MapFormer's direction)
- Dates: 2026-09-07
- Question: does an explicit per-token sigmoid gate `Delta = sigmoid(W_g x + b) * (W_out W_in x)`
  (+258 params) help on the torus or recency, and does it separate what from where?
- Task / environment: torus recipe; recency (k_max=64, p_filler 0.5, T=1024 train, eval 1024/2048).
- Arms: Vanilla_r4, Gated_r4, Gated_r4_frozen (gate untrainable, constant 0.982 rescale), Gated_r2.
- Seeds / batch: 8; one batch per task; recipes verbatim from sign / recency runs.
- Validity gates: bit-identical to Vanilla_r4 with gate forced open (0.0); gradient 1.9e-01 at init;
  gate starts token-independent; zero causal leak. Torus all arms 8/8 flat except Gated_r2 (5/8).
  Rule 9: torus -0.999/-0.734/-0.702; recency -0.949/-0.657.
- Result (`GATED_RESULTS.md`, `GATED_SEPARATION.md`, `GATED_TORUS.md`): separation Gated_r4 actions
  0.987 ± 0.003, observations 0.244 ± 0.043, 4.16x (per seed 3.33-5.44, 8/8); Gated_r2 2.55x
  (0.92-5.30); frozen 1.00x. Torus Gated_r4 - Vanilla_r4 +0.004 (MDE 0.008, 5/8) T=512, +0.003 (MDE
  0.024) T=1024; Gated_r4 - frozen +0.009 (MDE 0.008, 7/8) DETECTABLE at T=512; frozen - Vanilla_r4
  -0.005 (MDE 0.013). Recency T=2048: Vanilla_r4 0.9773 ± 0.015 (loss 0.0119), frozen 0.9397 ± 0.057,
  Gated_r4 0.9384 ± 0.083 (loss 0.0406); Gated_r4 - Vanilla_r4 -0.039 raw (MDE 0.079), -0.016
  matched (MDE 0.055).
- Status: separation CITABLE (8/8 above a pre-set 1.35x floor). Accuracy: torus T=512 a POWERED
  NEGATIVE against any gain above MDE 0.008 (smaller than rank's +0.038 there); torus T=1024 and
  recency UNMEASURED, never positive.
- Pre-registered? yes (`GATED_PREREG.md`): P3 (separation) PASSED; P1 (torus gain) FAILED; P2
  (recency >=) FAILED in direction; P4 NOT TESTABLE as run (Vanilla_r2 missing; repaired in E8,
  unmeasured). Falsifier "no effect at any length on either task" fired -> the review's borrow
  recommendation withdrawn.
- Caveats: one gate granularity (per token, per head), one init (bias 4.0), one placement. The only
  detectable contrast (over the frozen twin) equals the frozen twin's own rescale deficit.
  Interpretation "corroborates the paper's design" = the bottleneck already separates (~5x action vs
  observation movement).
- PRIOR ART: the gate is CoPE's (2405.18719) selection, per token rather than per query-key pair.
- Sources: `GATED_RESULTS.md`, `GATED_PREREG.md`, `GATED_SEPARATION.md`, `GATED_TORUS.md`,
  `run_gated.sh`. Bears on: what/where separation; "load-bearing does not imply needs help".

### E19 Forget gate x rank 2x2
- Dates: 2026-09-05
- Question: MapFormer plus a learned decay bias (the "empty Re G cell"); does it help on a map task,
  and does it work by forgetting?
- Task / environment: torus recipe.
- Arms: Vanilla (r=2), Vanilla_r4, Forget (r=2, +259), Forget_r4. Parameter-matched 2x2.
- Seeds / batch: 8, one batch.
- Validity gates: at lambda=0 bit-identical to Vanilla (0.0); |grad| 5.0e-03 vs 4.8e-04 median;
  lambda leaves zero at step 1. Rule 9: r(final train loss, acc) -0.716 (T=512), -0.311 (T=1024);
  loss-matched on last-5-epoch training loss, not eval NLL.
- Result (`FORGET_GATE.md`): T=1024 Vanilla 0.834 ± 0.064, Vanilla_r4 0.919 ± 0.012, Forget 0.914 ±
  0.041, Forget_r4 0.918 ± 0.018. Gate at r=2: +0.081 raw (7/8), +0.086 matched (8/8) at T=1024;
  +0.022 / +0.029 at T=512. Gate at r=4: -0.002. Rank without gate +0.085 raw (8/8) / +0.078 matched.
  Interaction -0.082 (2/8). Learned lambda: Forget +0.0118 ± 0.0334 (5/8 seeds < 0); Forget_r4
  +0.0141 ± 0.0048. r(lambda, gain) = -0.516; seeds with lambda <= 0 gain +0.102, decaying seeds
  +0.045; most-decaying seed (lambda +0.0916) is the only loser (-0.015). Gain +0.119 on worst four
  Vanilla seeds vs +0.042 on best four.
- Status: gate at r=2 CITABLE (T=1024; see E20 for MDE 0.080 against +0.081 raw). Gate x rank as
  substitutes DIRECTIONAL (no MDE given). Decay mechanism falsified as the explanation (by the arm's
  own lambda); mechanism UNIDENTIFIED.
- Pre-registered? yes, with a sign (`run_forget.sh`): "neutral-to-negative on accuracy" -> WRONG;
  "not by forgetting" -> right.
- Caveats: effect concentrated where the baseline converges worst; r=2 only. Vanilla / Vanilla_r4
  numbers are deterministic repeats of E1's.
- PRIOR ART: forget gate is FoX (2503.02130) style; Selective RoPE's "recall needs rotation and
  decay" principle; GRAPE proves FoX an exact additive instance.
- Sources: `FORGET_GATE.md`, `run_forget.sh`. Bears on: magnitude slot; OOD-length axis; rank.

### E20 Forget-gate control: frozen lambda
- Dates: 2026-09-05
- Question: is +0.086 the 259 parameters / init shift, or a trainable lambda?
- Task / environment: torus recipe; all three arms retrained in one batch.
- Arms: Vanilla, Forget, Forget_Frozen (gate parameters present, init matched at 0.0e+00, decay bias
  identically zero).
- Seeds / batch: 8, one batch.
- Result (`FORGET_CONTROL.md`), T=1024: Forget - Vanilla +0.081 (sd 0.080, MDE 0.080, 7/8) DETECTABLE;
  Forget_Frozen - Vanilla -0.016 (sd 0.119, MDE 0.118, 2/8) unmeasured; Forget - Forget_Frozen +0.097
  (sd 0.106, MDE 0.105, 7/8) unmeasured. Incidental: Vanilla and Forget retrain with 0.0000 maximum
  per-seed drift vs the earlier batch (torus is bit-reproducible).
- Status: DIRECTIONAL for "needs a live lambda" (flanking contrasts; the direct contrast is under
  MDE). The +0.081 is borderline detectable.
- Pre-registered? yes (second branch fired: Frozen lands on Vanilla).
- Caveats: the "transient training aid" hypothesis written here was post hoc and was refuted in E21.
  The CORRECTED block in this file (SDPA is NOT the cause of Match-Query non-reproducibility; the
  task's landscape is) belongs to the Match-Query line.
- Sources: `FORGET_CONTROL.md`, `run_forget_control.sh`. Bears on: forget gate; reproducibility.

### E21 Lambda trace over training
- Dates: 2026-09-05
- Question: does the gate work through its trajectory (rise then anneal)?
- Task / environment: torus, Forget retrained with lambda logged (bit-identical to stored checkpoints).
- Arms: Forget; gain vs Vanilla at T=1024 from the same seeds.
- Seeds / batch: 8.
- Result (`LAMBDA_TRACE.md`): peak decay at 0.01 of training in 6/8 seeds; r(peak decay, gain) =
  -0.531 (pre-registered > 0); r(final decay, gain) = -0.511; peak interior in 2/8.
- Status: transient-aid hypothesis REFUTED against its pre-registered sign (correlational, n=8).
  Mechanism behind +0.081 remains unidentified.
- Pre-registered? yes (`run_lambda_trace.sh`).
- Caveats: effective decay is lambda * E[sigmoid]. The file labels seeds with a 0.01-fraction peak
  "monotone".
- Sources: `LAMBDA_TRACE.md`, `run_lambda_trace.sh`. Bears on: forget gate.

### E22 Forget gate as a clock (pre-registered; batch deleted, not re-run)
- Dates: 2026-09-08
- Question: does the gate help MORE on recency (needs a clock) than on the torus? The live account:
  sum(log gamma) is a second, monotone accumulator (alpha +0.956) restoring the clock a signed phase
  gives up.
- Task / environment: torus and recency (T=4096 added as primary clock-side length because recency
  is at 1.000 at T=1024).
- Arms: Vanilla (204,373), Forget (205,660), Forget_Frozen (205,659), all r=2; 8 seeds per task.
- Result: NONE. The batch ran to completion (driver log "missing=0, finished Tue Sep 8 10:25:29") and
  the run directory was deleted by mistake; no results file exists. `run_forget_clock.sh` and the
  prereg are intact; must be re-run.
- Status: no result (UNTESTED). The clock account of the forget gate remains a candidate only.
- Pre-registered? yes (`FORGET_CLOCK_PREREG.md`): P1 interaction gain(recency) - gain(torus) > 0; P2
  torus benefit at OOD length only; P3 Frozen lands on Vanilla; P4 sum log gamma alpha ~0.95 on both.
- Caveats: note that the 2026-09-10 audit says recency does not need a clock (E24), which changes the
  premise's framing but not the testable interaction.
- Sources: `FORGET_CLOCK_PREREG.md`, `runs_forget_clock_driver.log`, `.claude-memory/project_state.md`,
  `CLAUDE.md` (START HERE block). Bears on: forget gate; clock vs map; OOD-length axis.

### E23 Recency (k-back) task construction, gates and pilots
- Dates: 2026-09-07 (K4SET/K16SET gate files 2026-09-11)
- Question: build a task where k is a CONTEXTUAL position.
- Task / environment: retrieve the k-th most recent content symbol; n_symbols 16, chance 0.0625;
  filler tokens emitted but not counted (p_filler 0.5); k_max 64, min_gap 64, T=1024.
- Validity gates: `RECENCY_GATES.md` (k_max=8): only min_gap = k_max rows pass (o1 up to 0.1494 at
  min_gap 0), establishing min_gap = k_max as necessary. `RECENCY_GATES_K64.md` (800 episodes): all
  rows PASS at min_gap 4/16/64; most-recent floor 0.0771; G8 token distance to the answer at k=64
  128.3 ± 11.8 (min_gap 64); G7 a +/-1 signed walk would be ambiguous on 0.975 of queries (an upper
  bound on difficulty, not a prediction). `RECENCY_GATES_K4SET.md` / `_K16SET.md` (k from a small
  set, used by the EM/SPREAD line): n-gram and oracle pass, most-recent FAIL (0.2958 / 0.1123) against
  the k_max=64 floor the script states -- the file does not state a floor for a k set.
- Pilots (`RECENCY_PREREG.md`, n=1, not results): without filler (k_max 8/32/64) everything at ceiling
  or RoPE best (1.000) with signed and monotone tied 0.946 -> "index retrieval in disguise"; with
  p_filler 0.5 RoPE falls to 0.366 / 0.315.
- Status: task validity CITABLE (gates pass; context manipulation built in via filler).
- Pre-registered? yes, with a pre-launch AMENDMENT after pilots: H4 (index does well) refuted and
  withdrawn before launch; H1 ("monotone beats signed") declared MALFORMED (signed contains monotone)
  and replaced by the cost of constraining; H2 promoted to primary.
- Caveats: the first prereg was written after a CPU smoke pilot and says so. CoPE arXiv id cited as
  2405.11582 in recency files; corpus copy reads 2405.18719.
- Sources: `RECENCY_PREREG.md`, `RECENCY_GATES.md`, `RECENCY_GATES_K64.md`, `RECENCY_GATES_K4SET.md`,
  `RECENCY_GATES_K16SET.md`, `validate_recency.py`. Bears on: every recency result (this line and
  the EM/WM line).

### E24 Recency batch: index vs path integration, and the cost of constraining (crossover)
- Dates: 2026-09-07
- Question: can an index code count contextually? Does forcing monotone cost anything on a counting
  task (vs -0.280 on the torus)?
- Task / environment: E23 config; chance 0.0625, most-recent floor 0.0771; train T=1024, eval
  1024/2048; 300 ep cosine, lr 1e-3, 1 layer, d=128, fast-attn.
- Arms: Signed_r4 (unconstrained), Abs_r4, Pos_r4, CARoPE_r4 (monotone), RoPE, PlainFlat (index).
- Seeds / batch: 8, one batch.
- Validity gates: convergence checked -- final-10% slope -0.002 to -0.005/epoch; 2x budget (600 ep,
  3 seeds) moves nothing (RoPE 0.251 -> 0.242, PlainFlat 0.247 -> 0.248, Signed 1.000 -> 1.000, index
  loss 2.15 -> 2.12) -> 0.234 is a capability limit. Rule 9: r(loss,acc) -0.999 over 48 runs, -0.875
  within path-integrated arms at T=2048. T=1024 at ceiling for two arms; move to T=2048 pre-registered.
- Result (`RECENCY_RESULTS.md`):
  | arm | final loss | T=1024 | T=2048 |
  |---|---|---|---|
  | Signed_r4 | 0.025 | 1.000 ± 0.000 | 0.940 ± 0.050 |
  | CARoPE_r4 | 0.020 | 1.000 ± 0.000 | 0.973 ± 0.028 |
  | Pos_r4 | 0.061 | 0.979 ± 0.041 | 0.947 ± 0.069 |
  | Abs_r4 | 0.110 | 0.961 ± 0.068 | 0.889 ± 0.126 |
  | RoPE | 2.150 | 0.234 ± 0.025 | 0.234 ± 0.011 |
  | PlainFlat | 2.143 | 0.236 ± 0.019 | 0.233 ± 0.009 |
  Path-integrated - index: +0.750 (sd 0.030, MDE 0.030, 8/8) at T=1024; +0.704 (MDE 0.056, 8/8) at
  T=2048. Per-offset at T=1024: index 0.99-1.00 at k=1, 0.27-0.33 at k=8, 0.11-0.13 at k=64;
  path-integrated flat in k. Cost of constraining at T=2048: Abs - Signed -0.052 raw (MDE 0.105) /
  +0.016 matched (MDE 0.046); Pos +0.007 / +0.035 (MDE 0.031, detectable); CARoPE +0.032 / +0.028;
  mean(monotone) - Signed -0.004 (MDE 0.055) / +0.026 (MDE 0.027) UNMEASURED. Interaction with torus
  (-0.215/-0.280): "~ +0.28".
- Status: index-cannot-count CITABLE and PRIOR-ART REPLICATION (CoPE's claim; not loss-matchable by
  construction). Torus half of the crossover CITABLE (E11); recency half UNMEASURED; the interaction
  is cross-batch, cross-task arithmetic -> DIRECTIONAL.
- Pre-registered? yes (amended before launch). Amended H1 (cost ~0 on recency): consistent, unmeasured.
  H3 (signed decays in k, monotone flat): REFUTED (both flat; decay is in index arms). H4 refuted
  pre-launch.
- Caveats: AUDIT 2026-09-10 scope correction -- recency does not REQUIRE a clock: a signed
  accumulator whose query token rewinds by k-1 solves it (single-p0 kernel 1423/1423; full EM model
  frozen install 1.000 on 8/8, `WARM_RESULTS.md`). The recency half shows a monotone increment is
  HARMLESS, not REQUIRED. Theorem "no single accumulator is both" holds only for scalar or fully
  constrained accumulators (the model's is rank r per head). One task, one k_max, one p_filler.
- PRIOR ART: CoPE (2405.18719) "relative PE's best is a decaying attention"; Flip-Flop / counting
  results in CoPE Sec 5.
- Sources: `RECENCY_RESULTS.md`, `RECENCY_PREREG.md`, `AUDIT_2026-09-10.md` #2,
  `THEORY_NARRATIVE.md` Sec 3.2/3.4 and ledger #4/#6. Bears on: clock vs map; sign; EM/WM line
  (recency task and Signed_r4 as WM baseline).

### E25 H2: the unconstrained arm learns a different accumulator per task
- Dates: 2026-09-07
- Question: is alpha diagnostic of what the task demanded?
- Task / environment: eval-only probe on E11 (torus, n=12) and E24 (recency, n=8) checkpoints.
- Result (`RECENCY_H2.md`): alpha torus -> recency (negative fraction of Delta): Signed_r4 0.591 ±
  0.028 -> 0.967 ± 0.009, delta +0.376, se 0.009 (0.498 -> 0.478); Abs_r4 1.010 -> 0.976 (-0.033, se
  0.007); Pos_r4 1.005 -> 1.005; CARoPE_r4 1.003 -> 0.992 (-0.011, se 0.001).
- Status: CITABLE as description (t ~ 42 per `RECENCY_RESULTS.md`; constrained arms are the control).
  Mechanism is E26, not a one-signed code (negative fraction barely moves).
- Pre-registered? yes (primary claim after amendment). Held.
- Caveats: alpha is a fitted statistic nearly collinear with opposition; it cannot be varied as a
  parameter. Cross-task comparison (different batches, tasks, n). Signed_r4 torus alpha disagrees
  with E13's 0.518.
- Sources: `RECENCY_H2.md`, `RECENCY_RESULTS.md` Sec 3. Bears on: clock vs map.

### E26 Recency gate ablation: is the content gate the counting mechanism?
- Dates: 2026-09-07
- Question: causal test of how the unconstrained arm counts.
- Task / environment: eval-only interventions on Delta, Signed_r4, T=1024; index arms at 0.234.
- Arms: Signed_r4 under 8 conditions.
- Seeds / batch: 8 (E24 checkpoints).
- Result (`RECENCY_GATE_ABLATION.md`): none 1.0000; zero_filler 0.9997 (-0.0003, MDE 0.0008);
  equalize 0.0855; scale_match 0.1096; uniform_content 0.7826 ± 0.1150; uniform_all 0.1886 ± 0.0248;
  zero_content 0.0680; zero_query 0.8232 ± 0.0797. uniform_content - uniform_all +0.594 at 8/8
  (magnitude-matched). Correlation r(log gate ratio, acc) = -0.34 (no power near ceiling). Per-head
  gate present in 8/8 seeds.
- Status: CITABLE, established by intervention.
- Pre-registered? the original criterion was (zero_filler no-op AND equalize collapses AND
  zero_content destroys); the equalize leg is CONFOUNDED (scale_match collapses as hard) and not
  counted. The decisive magnitude-matched pair was added AFTER scale_match failed (control for a
  confound, not pre-registered; labelled as such).
- Caveats: the model is acutely sensitive to theta's absolute scale. uniform_all falls below the index
  arms. The E18 follow-up shows the gate being load-bearing does not mean supplying it helps.
- PRIOR ART: the mechanism "an unconstrained MapFormer discovers CoPE's gate" relates to CoPE's
  counted-token gate.
- Sources: `RECENCY_GATE_ABLATION.md`, `ablate_recency_gate.py`, `RECENCY_RESULTS.md`.
- Bears on: clock vs map; what/where separation; motivated E18.

### E27 Flip-Flop LM (external check)
- Dates: 2026-09-07
- Question: does the recency result replicate on a published benchmark?
- Task / environment: Flip-Flop (Liu et al. 2023, definition from CoPE Sec 5.1); chance 0.500; train
  p_i=0.80, OOD dense 0.98, OOD sparse 0.10; T=512; scoring the FINAL read only (removes the
  consecutive-reads order-1 shortcut); 4 layers, d=256, 4 heads, 200 epochs.
- Arms: Signed_r4, Pos_r4, Abs_r4, RoPE, PlainFlat, CARoPE_r4.
- Seeds / batch: 8, one batch.
- Validity gates (`FLIPFLOP_GATES.md`): final-read scoring puts n-gram orders near chance; all-reads
  order 1 = 0.730 (published task's own property). OOD sparse: last bit = governing bit 0.978.
- Result (`FLIPFLOP_RESULTS.md`), error %: in-dist all 0.00 (CARoPE 0.03); OOD dense Signed 6.69 ±
  3.12, Pos 7.00, Abs 8.09, RoPE 8.59 ± 4.54, PlainFlat 10.84, CARoPE 12.16. Path-integrated - index
  -1.23pp (MDE 3.99); monotone - signed +2.40pp (MDE 3.97); all pairwise MDE 4.2-7.0 -> unmeasured.
  OOD sparse favours index arms (0.00-0.13%) via the last-bit shortcut.
- Status: UNMEASURED (not null). Predicted by E24's per-offset curve: Flip-Flop only asks k=1, where
  index is 0.99.
- Pre-registered? no formal prereg; the gate file records the shortcut in advance.
- Caveats: OOD sparse is a shortcut, not a result. Different architecture size from E24.
- Sources: `FLIPFLOP_RESULTS.md`, `FLIPFLOP_GATES.md`, `run_flipflop.sh`.
- Bears on: scope of borrowed benchmarks; clock vs map (ordinal depth is the discriminating property).

### E28 MQAR pilots (not runnable at this scale)
- Dates: 2026-09-08
- Question: does MQAR discriminate positional mechanisms?
- Task / environment: MQAR, 2 layers d=128, ~150K sequences; raw chance 0.0078.
- Arms: Vanilla_r4 (n_kv 16 at 100/500 ep; n_kv 4 at 200 ep), RoPE and PlainFlat (n_kv 4).
- Seeds / batch: 4 pilots, n=1; no batch launched.
- Result (`MQAR_RESULTS.md`): n_kv=16: 0.075 / 0.089 vs "guess among episode values" 0.063; n_kv=4:
  Vanilla_r4 0.273, RoPE 0.264, PlainFlat 0.274 vs 0.250. Loss 2.96 vs ln16 2.77; 1.85 vs ln4 1.39.
  No induction circuit forms; index arms fail identically.
- Status: EXPLORATORY feasibility negative (a floor, not a positional result).
- Pre-registered? yes (`MQAR_PREREG.md`): predicted a CEILING; got a FLOOR (prediction wrong).
- Caveats: capacity/data, not mechanism; published MQAR uses vocab 8192 and its own sweep.
- Sources: `MQAR_RESULTS.md`, `MQAR_PREREG.md`. Bears on: scope of borrowed benchmarks.

### E29 Language landscape (literature check) and the enwik8 pointer
- Dates: 2026-08-27 (landscape); enwik8 numbers from the language line
- Question: which language experiments are already done?
- Result (`LANGUAGE_LANDSCAPE.md`, literature, not our experiments): MapFormer v4 Sec 5.5: 12-layer
  MapWM (r=4) on OpenWebText, RoPE 19.14 ± 0.14 vs MapWM 18.79 ± 0.15 ppl, 5 seeds; BLiMP 0.78 vs
  0.79 (no gain); length extrapolation loses to CoPE / PathAtt on NarrativeQA; nominates code modeling.
  PoPE's angle is not content-dependent (magnitude only), 124M/253M/774M ppl gains vs RoPE. CoPE
  Wikitext-103 23.81 (relative) vs 23.46; Flip-Flop OOD 20.3 -> 4.9 error. Selective RoPE: GLA 1.3B
  Wiki ppl 18.50 -> 17.87; FoX 370M regresses 25.29 -> 33.87; NoPE has best avg acc (55.2) in its own
  table. PaTH strongest (RULER 16K 0.0 -> 18.7). Published negatives for content-dependent phase on
  language: HGRN Table 11, HGRN2 Table 1 (`papers/INDEX.md`).
- Our enwik8 (from `.claude-memory/reference_language_and_pope.md`, flat 9-layer ~28.6M, 36k iters,
  seq 512): MapPoPE-Flat 1.3740 (n=3), PoPE-Flat 1.3746 (n=1), Vanilla/MapWM 1.3758 (n=1), RoPE 1.3799
  (n=3); MapPoPE - RoPE -0.0058, t=3.49. Position effect -0.0041 vs MDE 0.0041 at n=3: UNDERPOWERED,
  not null; paper's effect ~0.0067 bits/byte.
- Status: literature = PRIOR ART context; enwik8 EXPLORATORY (n=1 singles).
- Pre-registered? no.
- Caveats: `ENWIK8_LONG.md` (n=1 table) gives RoPE 1.3817 and MapPoPE-Flat_r4 1.3723, differing from
  the n=3 means above. The language trainer saves no checkpoints, so alpha has never been measured on
  text. Enwik8 belongs to the language/hierarchy line.
- Sources: `LANGUAGE_LANDSCAPE.md`, `papers/INDEX.md`, `.claude-memory/reference_language_and_pope.md`,
  `ENWIK8_LONG.md`. Bears on: scope of every claim (navigation regime is the surviving novelty).

### E30 MONOTONE (in flight, no results)
- Dates: prereg 2026-09-13
- Question: the sign ablation outside MapFormer-WM: monotone MapFormer-EM (P0, r=4) on recency (whose
  solution is a rewind) and monotone Selective RoPE generator (`SRoPEGen_Abs`, 238,234 params).
- Status: TRAINING (`runs/monotone`); no results. Mention as in-flight only.
- Sources: `MONOTONE_PREREG.md`.

---

## Excluded

| file / claim | reason | killed by |
|---|---|---|
| "A5 (sign) is the one axis nobody varies deliberately" (`SIGN_ABLATION_PREREG.md` main text, `SIGN_ABLATION.md` header) | false; Sarrof, Grazzi, PaTH, RWKV-7, Selective RoPE Sec 4.2 | `SIGN_ABLATION_PREREG.md` AMENDMENT |
| Selective RoPE per-knob attribution (conv / rank / gate rows) | every arm also deletes omega and action_to_lie (readout swap) | `SELECTIVE_ROPE.md` CONFOUND block |
| Gate-as-token-suppressor | 1.35x torus, 1.54x parity | `GATE_PROBE.md` |
| r=2 deficit is packing geometry; account of v4 Table 6 | r=2 best at D=5; deficit not growing with D; paper used r=D in 5D | `mapformer_math.tex`, `DXR_RANK_THRESHOLD.md` |
| Optimisation half "stands at three cells" | D=3 unmeasured; D=2 cell is the r=2 deficit; one independent cell | `mapformer_math.tex` audit note |
| "C4 is an EM-only property" | EM also < 1 (0.90) | `PAPER_FIG4_EM.md` |
| "MapWM's additive attention" (motivation in `PAPER_FIG4_EM.md`, CLAUDE.md) | MapWM rotates content Q,K; not additive | `AUDIT_2026-09-10.md` #1 |
| "use r=4" as a general recommendation | +0.019 unmeasured on MapPoPE | `MAPPOPE_R4_RESULTS.md` |
| Critical-dimension / under-trained low-frequency channel account of OOD degradation | ablating low channels costs MORE at OOD | `LOCALISATION.md` P2 |
| "The InEKF's wrap bounds the accumulator" | wraps the innovation; range 285.6 vs 283.9 | `ACCUMULATOR.md` P1 |
| "One quantity (alpha) explains almost everything" | covers sign and rank only | `ACCUMULATOR.md` |
| alpha as an independently variable diagnostic ("vary alpha and watch degradation") | fitted statistic; r(opposition, alpha) +0.9995 | `.claude-memory/project_clock_vs_map.md` |
| Recency needs a clock; Thm 1 "no single accumulator is both map and clock" (unscoped) | rewind construction; rank-r accumulator | `AUDIT_2026-09-10.md` #2 |
| THEORY_KERNEL labelling +0.123/+0.195 as signed vs monotone | they are signed vs index | `AUDIT_2026-09-10.md` stale list |
| Recency H1 "monotone beats signed" | malformed (signed contains monotone) | `RECENCY_PREREG.md` amendment |
| Recency H4 (index at or above path-integrated) | refuted, sign inverted | `RECENCY_PREREG.md` amendment |
| Recency H3 (signed decays in k) | both flat | `RECENCY_RESULTS.md` |
| `equalize` as proof of the content gate | confounded with theta scale | `RECENCY_GATE_ABLATION.md` |
| Forget gate works by decay | anti-correlated with lambda | `FORGET_GATE.md` |
| Forget gate as parameters / init shift | frozen lambda lands on Vanilla | `FORGET_CONTROL.md` |
| Forget gate as a transient training aid | r(peak, gain) -0.531 | `LAMBDA_TRACE.md` |
| Explicit gate should help (review borrow recommendation) | 4.16x separation, no gain | `GATED_RESULTS.md` |
| "The two ~8k knobs show the gate is special" | GateAngle ~ NoBottleneck | `SELECTIVE_ROPE.md` |
| Conv "neutral on parity" | hurts on parity too | `SELECTIVE_ROPE.md` |
| PoPE gain scales with octaves (grid size) | flat loss-matched | `POPE_WRAPPING.md` |
| Grid-16 PoPE gain | bimodal convergence; sign flips loss-matched | `POPE_WRAPPING.md` |
| MQAR "ceiling" prediction; recommendation to run MQAR | floor; wrong target class | `MQAR_RESULTS.md` |
| Flip-Flop as a replication test of recency | only asks k=1 | `FLIPFLOP_RESULTS.md` |
| SDPA as cause of cross-batch Match-Query non-reproducibility | bitwise identical gradients | `FORGET_CONTROL.md` CORRECTED block |
| "The r=1 -> 0.66" claim | no experiment in repo | `RANK_SWEEP.md` |
| Forget-clock batch results | directory deleted; never re-run | `.claude-memory/project_state.md` |
| enwik8 "inside the noise floor" (position effect) | rule-11 violation; underpowered | `.claude-memory/reference_language_and_pope.md` |

## Cross-line dependencies

- **r=4 default** (E1) is used by the EM/WM line (`VanillaEM_P0_r4`, recency EM arms), the loop line
  (`MQ_RANK_2X2.md`, r=4 + loop x4 = 0.986 on Match-Query) and every `*_r4` arm here. E6 is the
  evidence that the r=2 skew also afflicts MapEM.
- **Recency task and gates** (E23) and `Signed_r4` / Vanilla WM baselines (E24) underlie the EM/WM
  line (`RECENCY_EM_RESULTS.md`, WARM, UNFREEZE, NOLEAK, SEARCH, SPREAD, PAIRORIGIN); K4SET / K16SET
  gate files belong to that line.
- **Sign ablation** (E11) is the torus half of the clock/map crossover and the basis for "path
  integration, not the encoding" requiring the sign (headline line). The AUDIT recency-rewind
  correction from the EM line rescopes E24.
- **Localisation / accumulator** (E13, E14) bear on the OOD-length axis claimed by the correction line
  (Level 1.5 stabilisation), the loop line and PoPE; E14's failed positive control withdraws a
  correction-line claim.
- **MapPoPE at r=2** caveat applies to `PAPER_OOD_WITH_POPE.md` (MapPoPE-Flat best on the paper task);
  E8 says r=4 does not measurably change that standing.
- **Torus bit-reproducibility** (E20) licenses cross-batch identical Vanilla arms and warns that they
  are not replications; the Match-Query landscape diagnosis there is used by the Match-Query / loop
  lines.
- **Parity** (`ALGORITHMIC_RESULTS.md`, loop line) supplies the parity task for E15.
- **Rotation-action losses** in E9 depend on the environment line (allocentric recoding, MiniWorld).

## Files read

RESULTS_INDEX.md, N3_AUDIT.md, KNOWN_BUGS.md, archive/void/README.md, CLAUDE.md (START HERE and grep
for corrections), report/INVENTORY_BRIEF.md, RANK_SWEEP.md, RANK_TRUNCATION.md, LEARNED_RANK.md,
ACTION_GEOMETRY.md, PAPER_FIG4_REPRO.md, PAPER_FIG4_EM.md, DXR_PRELIM.md, DXR_RANK_THRESHOLD.md,
ND_GATES.md, mapformer_math.tex (sec. predict / D x r result and audit notes, lines ~515-540,
2255-2400), MAPPOPE_R4.md, MAPPOPE_R4_PREREG.md, MAPPOPE_R4_RESULTS.md, MAPPOPE_VS_POPE.md,
SIGN_ABLATION_PREREG.md, SIGN_ABLATION.md, _SIGN_RAW.md, SIGN_PROBE.md, ACCUMULATOR.md,
LOCALISATION_PREREG.md, LOCALISATION.md, LOCALISATION_RANK.md, SELECTIVE_ROPE.md, _SELECTIVE_TORUS.md,
CONV_KERNEL_PROBE.md, GATE_PROBE.md, GATED_PREREG.md, GATED_RESULTS.md, GATED_SEPARATION.md,
GATED_TORUS.md, FORGET_GATE.md, FORGET_CONTROL.md, FORGET_CLOCK_PREREG.md, LAMBDA_TRACE.md,
POPE_WRAPPING.md, RECENCY_PREREG.md, RECENCY_RESULTS.md, RECENCY_GATES.md, RECENCY_GATES_K64.md,
RECENCY_GATES_K4SET.md, RECENCY_GATES_K16SET.md, RECENCY_H2.md, RECENCY_GATE_ABLATION.md,
FLIPFLOP_GATES.md, FLIPFLOP_RESULTS.md, MQAR_PREREG.md, MQAR_RESULTS.md, LANGUAGE_LANDSCAPE.md,
THEORY_NARRATIVE.md (Sec 2-3, 9, 10), AUDIT_2026-09-10.md (grep + lines 100-130), MONOTONE_PREREG.md
(head), ALGORITHMIC_RESULTS.md (head), ENWIK8_LONG.md (head), papers/INDEX.md (lines 1-110),
.claude-memory/reference_positional_landscape.md, .claude-memory/project_sign_axis.md,
.claude-memory/project_clock_vs_map.md, .claude-memory/project_rank_and_selective_rope.md,
.claude-memory/project_state.md (lines 60-95), .claude-memory/reference_language_and_pope.md;
script headers: run_rank_sweep.sh, run_selective.sh, run_dxr.sh, run_sign.sh, run_recency.sh,
run_flipflop.sh, run_gated.sh, run_mappope_r4.sh, run_forget.sh, run_forget_control.sh,
run_lambda_trace.sh, run_popewrap.sh, run_accumulator.sh, run_localisation.sh, probe_accumulator.py
(docstring); runs_forget_clock_driver.log (tail); runs/sign training log (param count).

## Files in scope not covered

None missing. Not read in full: `mapformer_math.tex` outside the D x r sections, `AUDIT_2026-09-10.md`
beyond the items cited, `EM_WM_STATE.md` (EM line), JSON sidecars.

## Source disagreements recorded (unresolved)

1. Sign-ablation parameter count: `SIGN_ABLATION_PREREG.md` 205,785 vs training log / RESULTS_INDEX /
   THEORY_NARRATIVE 204,757 (log is authoritative for the trained runs).
2. Signed_r4 torus alpha: 0.518 (`LOCALISATION.md`) vs 0.591 ± 0.028 (`RECENCY_H2.md`), same arm,
   different analyses; `THEORY_NARRATIVE.md` notes it without resolving.
3. `ACCUMULATOR.md` between-batch Vanilla alpha: text 0.665 vs 0.608, table 0.710 vs 0.631.
4. Fig. 4 C2 at r=2 (-0.729 ± 0.373): file says "reproduces", memory note says "weak".
5. Fig. 4 location: files say Sec 5.4; CLAUDE.md corrected to App. C.3.
6. CoPE arXiv id: 2405.11582 (recency files) vs 2405.18719 (corpus).
7. enwik8: n=3 means (RoPE 1.3799, MapPoPE 1.3740) vs `ENWIK8_LONG.md` n=1 (1.3817, MapPoPE_r4 1.3723).
8. Forget gate at r=2: +0.086 loss-matched 8/8 (`FORGET_GATE.md`, no MDE) vs +0.081 raw, MDE 0.080
   (`FORGET_CONTROL.md`, deterministic rerun of the same runs).
9. ND gate G5 decay exponent at D=3 r=2: -1.04 (`ND_GATES.md`) vs -1.51 (un-scripted ladder, per the
   tex audit note).
