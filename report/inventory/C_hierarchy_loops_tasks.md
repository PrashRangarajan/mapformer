# Inventory C: hierarchy, recursion/loops, and the task suite

Built 2026-09-13 from first-hand reads of the files listed at the end. Dates are file mtimes
(git was not used) unless the file states its own date. All numbers are copied from the named
source; where a number exists only in `CLAUDE.md` that is said.

## Overview

This line asked three questions. (1) Does a time-hierarchy (Hourglass pooling, hierarchical /
routed / recursive attention) buy anything, and where? (2) Does recursion (one weight-shared block
applied k times) substitute for depth, and does it compose with path integration? (3) Which tasks
in the repo actually measure a cognitive map, i.e. survive the action-stream n-gram gate and
context destruction?

What survived:
- **Tasks.** Match-Query (blind continuation), the compositional-motif task, parity, and the lap
  task passed their gates; Match-Query and compositional passed context destruction. Hier-goal and
  all four planner-demonstration tasks are void by audit; the audits themselves are current. Map-Query
  passes gates but was never trained adequately. CSCG stitch/schema tasks are NOT READY.
- **Loops.** On Match-Query 128^2 the loop adds +0.414 to path integration and +0.099 to an index
  code, interaction +0.315 (n=8, all detectable); `r=4 + loop x4` is 0.986 +/- 0.020 (n=8). On parity
  (n=16) the loop substitutes for depth equally in both codes (interaction +0.014, unmeasured), so the
  super-additivity is task-specific. The loop's torus training-length win is convergence (loss-matched
  +0.006). "Loop beats three real layers" is retracted (it matches). MoR routing has nothing to route on.
- **Path integration outside navigation.** Parity: +0.316 / +0.326 / +0.167 / +0.083 / +0.041 at
  L=16..256, 8/8 seeds at every length (copy control failed).
- **Hierarchy.** The compositional gain reproduces in size at the better recipe (+0.136, 7/8) but is
  under MDE (0.173): directional. The recipe itself (+0.160, detectable) is the largest effect on that
  task. On parity at L=512 hierarchy helps 12/12 and 11/12 seeds (sign test) but is under the t-MDE.
  On text hierarchy is an efficiency property (1.23x throughput, -14.1% memory), not a quality win.
  Every retrieval-task hierarchy variant loses to flat; oracle segmentation and frame reset do not help.
- **Recipe.** lr 1e-3 cuts Vanilla's torus seed sd 3.5x at T=512 (n=8); the same change on
  compositional doubled seed variance.

---

## Task-validity summary (for the report writer)

| task | action/answer-stream n-gram gate | context destruction | other gates | usable? |
|---|---|---|---|---|
| Match-Query 64^2 n_obs=16 | PASS (o1 0.0625, o3 0.0689, o5 0.0664 vs chance 0.0625) | PASS (0.918 -> 0.074 / 0.076) | never-moved floor CORRECTED to 0.0893 | YES |
| Match-Query 128^2 n_obs=16 | PASS (0.0640 / 0.0620 / 0.0731) | inherits design; not separately run | never-moved 0.0490 (uncorrected scorer) | YES |
| Match-Query 64^2 n_obs=4 | PASS vs chance 0.25 | not run | separation breaks | boundary only |
| Compositional motif | no action n-gram reported; oracle 1.00, majority ~0.50 | PASS (cross_nb collapses 67-80% of headroom) | lag medians 26/72/160 | YES |
| Compositional Match-Query | n-gram at chance for both categories | not run | `exact` never-moved 0.092-0.178 (up to 2.8x chance) | exploratory only |
| Parity (algorithmic) | PASS (worst excess +0.0122) | n/a | repeat-prev at chance | YES |
| Copy (algorithmic) | PASS | n/a | no dynamic range (ceiling at L=16, chance at L>=32) | NO as control |
| Lap (variable loop length) | n-gram boundary acc at always-no level (0.750) | not run | positional shortcut 0.163 < random 0.250 | YES (exploratory) |
| Map-Query | PASS at T_explore >= 256 (n-gram 0.485-0.507 vs ~0.50) | not run | FAILS at T_explore=64/16 (assume-start 0.623) | gates only; no valid trained result |
| Torus paper task (horizon, looped pilot, recipe, L15xloop) | validated elsewhere (PAPER_VALIDATION, other line) | -- | blank floor 0.506 | YES |
| Retrieval task T=256 (HierAttn/RouteAttn/Recursive/SpaceTime/Bounded) | torus revisit task | -- | -- | exploratory era |
| Aggregate (windowed majority) | none recorded | none | chance ~0.11 | exploratory |
| hier-goal (both versions) | FAIL (o1 0.969 raw; o3 0.971 interleaved) | FAIL (0.912 -> 0.913) | closed-loop 0.013-0.037 vs 0.010 floor | VOID |
| goal / rooms_goal / rooms_maze / maze_varying | FAIL (0.969 / 0.969 / 0.791 / 0.650 vs 0.250) | -- | -- | VOID |
| DoorKey BC / DAgger | never checked (SUSPECT banner) | -- | -- | not usable |
| CSCG stitch / schema | stitch control defeatable 0.617 balanced acc | -- | NOT READY | NO |
| MiniGrid MemoryS13 | none recorded | none | -- | exploratory |

---

## Experiments

### C01 Compositional-motif task: design and validity
- Dates: design 2026-07-25 (file text); ablation `ABLATE_COMPOSITIONAL.md` "new 2026-08-18" (RESULTS_INDEX).
- Question: is there a task where a lossy hierarchical summary is a sufficient statistic, so hierarchy should pay?
- Task / environment: 64x64 torus tiled into 8x8 rooms, `n_templates=4` motif templates assigned to several rooms; templates and assignment redrawn every episode (`fresh_per_episode=True`). Metrics: `exact_acc` (exact revisit), `cross_nb_acc` (motif-cell seen in another copy, exact cell not seen, non-blank). Majority/blank ~50%; `cross_nb` floor ~0.072 (COMP_HEADROOM, BASELINE_TABLE). Train T=256; eval T in {256,512,1024,2048}; held-out env seed 10000.
- Arms: validator only; ablation on Vanilla, Hourglass_k2, PlainFlat checkpoints.
- Seeds / batch: validator CPU; ablation on n=3 checkpoints (per-arm +/- reported).
- Validity gates: cross-instance label mass 14.3/23.2/34.4% at T=128/256/512; consistency failures 0; copy-nearest-motif oracle 100%; lag median 26/72/160; solvable in a 32-step window 57/30/16%. Context destruction (cross_nb): Vanilla 0.2708 -> shuffle_actions 0.0234 / resample_actions 0.0658 / shuffle_obs 0.0197 / resample_obs 0.0516; Hourglass_k2 0.4279 -> 0.0294 / 0.0854 / 0.0238 / 0.0622; PlainFlat 0.2162 -> 0.0075 / 0.0724 / 0.0161 / 0.0551. Headroom lost 76% / 80% / 67%. Verdict PASSES.
- Result: task is valid; the hierarchy gap lives in the part of the signal destruction removes (destroyed Hourglass_k2 0.085 vs Vanilla 0.066; intact 0.428 vs 0.271).
- Status: CITABLE (as a validity result).
- Pre-registered? The design lists H1-H3 before Phase 1 (see C02, C03).
- Caveats the report must carry: no action-stream n-gram gate is recorded for this task (it is an observation-prediction task, not a demonstration task). `shuffle` and `resample` disagree in ordering (resample less destructive here, opposite of the paper task); both must be reported.
- Sources: COMPOSITIONAL_EXPERIMENT.md, ABLATE_COMPOSITIONAL.md, HOURGLASS_README.md.
- Bears on: task validity; hierarchy; "evaluations are transfer measurements".

### C02 Compositional multi-seed: hierarchy x backbone (old recipe)
- Dates: n=3 run 2026-07-25; table extended to n=8 for hierarchy arms (COMPOSITIONAL_MULTISEED.md mtime 2026-08-10).
- Question: does an Hourglass (k=2 fixed stride) beat flat on compositional transfer; does MapFormer beat a plain transformer; H1 (flat already transfers; WM > EM on cross), H2 (absolute-theta Hourglass ~ flat).
- Task / environment: C01. Floor ~0.072.
- Arms (display names; old keys): MapWM-Flat (Vanilla), MapEM-Flat (VanillaEM), MapWM-Hier (Hourglass_k2, 600,917 params per C03), MapWM-FlatHG (HourglassFlat3, parameter-matched flat scaffold), Plain-Hier (PlainHourglass), Plain-Flat (PlainFlat), MapWM-Hier-CoarseIdx (coarse level uses index position), MapWM-Hier-CoarsePI (coarse level runs its own path integration, +~1K params), PoPE-Flat, MapPoPE-Flat, MapPoPE-Hier, MapPoPE-Hier-CoarseIdx.
- Seeds / batch: n=8 for all hierarchy arms plus Plain-Flat and MapWM-FlatHG; n=3 for MapWM-Flat, MapEM-Flat, PoPE-Flat, MapPoPE-Flat. Accumulated over several batches ("the weakest provenance here", BASELINE_TABLE). Recipe: 50 epochs, 156 batches, 3 layers, LinearLR 1.0->0.0 at lr 3e-4 (COMP_HEADROOM_PREREG).
- Validity gates: C01. No convergence / rule-9 analysis in the file.
- Result (`cross_nb_acc` @T=256, COMPOSITIONAL_MULTISEED.md):

  | arm | T=256 | T=512 | T=1024 | T=2048 | n |
  |---|---|---|---|---|---|
  | MapWM-Hier | 0.415 +/- 0.096 | 0.289 +/- 0.098 | 0.170 +/- 0.113 | 0.108 +/- 0.118 | 8 |
  | MapWM-Hier-CoarseIdx | 0.556 +/- 0.123 | 0.429 +/- 0.116 | 0.308 +/- 0.126 | 0.232 +/- 0.134 | 8 |
  | MapWM-Hier-CoarsePI | 0.498 +/- 0.109 | 0.371 +/- 0.115 | 0.258 +/- 0.124 | 0.166 +/- 0.106 | 8 |
  | MapPoPE-Hier | 0.429 +/- 0.117 | 0.318 +/- 0.119 | 0.232 +/- 0.126 | 0.190 +/- 0.129 | 8 |
  | MapPoPE-Hier-CoarseIdx | 0.528 +/- 0.117 | 0.401 +/- 0.131 | 0.302 +/- 0.143 | 0.243 +/- 0.152 | 8 |
  | MapWM-FlatHG | 0.285 +/- 0.067 | 0.172 +/- 0.061 | 0.085 +/- 0.061 | 0.053 +/- 0.054 | 8 |
  | Plain-Hier | 0.318 +/- 0.029 | 0.196 +/- 0.032 | 0.088 +/- 0.019 | 0.042 +/- 0.010 | 8 |
  | Plain-Flat | 0.216 +/- 0.004 | 0.100 +/- 0.006 | 0.038 +/- 0.002 | 0.018 +/- 0.002 | 8 |
  | MapWM-Flat | 0.270 +/- 0.030 | 0.164 +/- 0.021 | 0.081 +/- 0.006 | 0.048 +/- 0.012 | 3 |
  | MapEM-Flat | 0.097 +/- 0.013 | 0.047 +/- 0.012 | 0.026 +/- 0.011 | 0.015 +/- 0.010 | 3 |
  | PoPE-Flat | 0.319 +/- 0.032 | 0.234 +/- 0.027 | 0.162 +/- 0.022 | 0.123 +/- 0.016 | 3 |
  | MapPoPE-Flat | 0.366 +/- 0.076 | 0.232 +/- 0.061 | 0.137 +/- 0.048 | 0.090 +/- 0.038 | 3 |

  `exact_acc` @T=256 / T=2048: MapWM-Hier 0.959 / 0.710; MapWM-FlatHG 0.929 / 0.641; Plain-Hier 0.918 / 0.592; Plain-Flat 0.904 / 0.538; MapWM-Flat 0.924 / 0.646 (n=3); MapEM-Flat 0.788 +/- 0.168 / 0.519 (n=3); MapPoPE-Flat 0.979 / 0.924 (n=3).
  - Hierarchy gap to cite: MapWM-Hier - MapWM-FlatHG = +0.130 (0.415 vs 0.285, both n=8); plain family +0.102 (0.318 vs 0.216, both n=8) (MATCH_QUERY_RESULTS finding 4, BASELINE_TABLE D, N3_AUDIT item 3). No MDE computed in any source for these pairs (cross-batch).
  - n=3 paired read (COMPOSITIONAL_EXPERIMENT): MapWM-Hier > MapWM-FlatHG on all 3 seeds (+0.09/+0.28/+0.06), Plain-Hier > Plain-Flat on all 3 (+0.15/+0.12/+0.07). H2 FALSIFIED (fixed-stride Hourglass beats flat).
  - Path integration vs plain on exact recall widens with T (n=3 era numbers, COMPOSITIONAL_EXPERIMENT): flat +0.02/+0.10/+0.14/+0.11, hier +0.03/+0.10/+0.18/+0.17 over T=256..2048.
  - MapEM-Flat (0.097) below Plain-Flat (0.213 at n=3; 0.216 at n=8): H1's WM > EM on cross holds; "EM >= WM on exact" did not (0.788 vs 0.924), flagged unstable.
- Status: DIRECTIONAL for the hierarchy gap (re-measured in C05, under MDE). CoarseIdx/CoarsePI rows EXPLORATORY (no matched contrast with MDE anywhere). n=3 rows EXPLORATORY.
- Pre-registered? Yes, hypotheses H1-H3 in COMPOSITIONAL_EXPERIMENT.md. H1 held (flat transfers; WM > EM on cross), EM >= WM on exact failed; H2 falsified.
- Caveats the report must carry: seed counts are not randomly assigned (every flat MapFormer/PoPE arm n=3, every hierarchy arm n=8) - cite the MapWM-FlatHG pair, not MapWM-Flat 0.270 (+0.145). MapWM-Hier high variance (one seed 0.625 at n=3). Old recipe undertrains (C04, +0.160 recipe effect > +0.130). The "plain ~ 0.06 chance floor" prediction was falsified: a plain transformer path-integrates via attention because actions are in the input.
- Sources: COMPOSITIONAL_MULTISEED.md, COMPOSITIONAL_EXPERIMENT.md, COMPOSITIONAL_RESULTS.md (superseded single seed), COMPOSITIONAL_PLAIN_RESULTS.md (single seed plain pair), BASELINE_TABLE.md sec. D, N3_AUDIT.md sec. 3.
- Bears on: hierarchy (compositional transfer); path integration buys exact recall; EM vs WM (EM worse on transfer); coarse-position form (CoarseIdx).

### C03 Motif segmentation (H3): oracle room pooling and local frame reset
- Dates: v1 2026-07-25, v2 2026-07-26 (file text).
- Question: does a hierarchy that segments at room boundaries (v1) and resets the local frame at room entry so identical motifs collapse (v2) beat flat on transfer?
- Task / environment: C01, T=256.
- Arms: MapWM-MotifSeg (Hourglass_MotifSeg; identical to MapWM-Hier, 600,917 params, pools on ORACLE room boundaries); MapWM-MotifSeg-FR (segmentation + frame reset); MapWM-Flat-FR (FrameResetFlat, reset only, 3-layer flat).
- Seeds / batch: n=3; old recipe; causality verified (max leak 4.8e-7, CLAUDE.md).
- Validity gates: C01; v2 frame reset verified to zero the angle at room entry.
- Result: v1 `cross_nb` 0.254 +/- 0.014 @T=256 (below MapWM-FlatHG 0.281 at n=3; `exact_acc` 0.943). v2: `cross_nb` 0.157 (MotifSeg-FR) / 0.151 (Flat-FR); `exact_acc` 0.94 -> 0.77. Both metrics dropped; MotifSeg-FR ~= Flat-FR.
- Status: EXPLORATORY (n=3, no MDE).
- Pre-registered? Yes (H3 in COMPOSITIONAL_EXPERIMENT.md). H3 falsified for both "room-aligned pooling helps" (v1) and "collapse-by-structure helps" (v2).
- Caveats the report must carry: n=3, old recipe; comparison is to the n=3 FlatHG value 0.281, not the n=8 0.285. The mechanism reading (reset destroys the absolute position cross-instance retrieval also needs) is interpretation, not intervention-tested beyond the two arms.
- Sources: COMPOSITIONAL_EXPERIMENT.md findings 6-7; CLAUDE.md (session 2026-07-25).
- Bears on: hierarchy (generic compression, not structure alignment); where/what factorisation (a fully relative frame over-aliases).

### C04 Compositional headroom: recipe vs capability
- Dates: 2026-09-08.
- Question: is 0.415 a capability limit or a recipe limit?
- Task / environment: C01; `cross_nb` on held-out env seed 10000, n_traj=200; floor ~0.072.
- Arms: A Hourglass_k2 linear 3e-4 50 ep (published recipe, reproduction control); B cosine 1e-3 50 ep; C cosine 1e-3 150 ep; D LoopedHourglass cosine 1e-3 150 ep (209,256 params vs 605,800).
- Seeds / batch: 8 per arm, one batch.
- Validity gates: A reproduces the published 0.415 +/- 0.096 at 0.354 +/- 0.070 (t = 1.45, "same distribution, not a bit-reproduction"). Final losses A 0.7466, B 0.5579, C 0.4760, D 0.4095. No r(loss,acc) table in the file.
- Result:

  | contrast | delta | sd | MDE | seeds+ | verdict |
  |---|---|---|---|---|---|
  | C - A @256 | +0.160 | 0.141 | 0.140 | 7/8 | DETECTABLE |
  | B - A @256 | +0.107 | 0.151 | 0.150 | 6/8 | unmeasured |
  | C - B @256 | +0.053 | 0.133 | 0.132 | 6/8 | unmeasured |
  | D - C @256 | +0.029 | 0.168 | 0.166 | 4/8 | unmeasured |
  | C - A @512 | +0.153 | 0.144 | 0.143 | 7/8 | DETECTABLE |
  | B - A @512 | +0.115 | 0.132 | 0.131 | 7/8 | unmeasured |
  | C - B @512 | +0.038 | 0.101 | 0.100 | 6/8 | unmeasured |
  | D - C @512 | +0.050 | 0.211 | 0.209 | 4/8 | unmeasured |

  Arm means @256: A 0.354 +/- 0.070, B 0.460 +/- 0.151, C 0.514 +/- 0.126, D 0.542 +/- 0.170. Seed sd A 0.070 -> B 0.151 -> C 0.126 -> D 0.170.
- Status: CITABLE (C - A); P2 split and D - C unmeasured.
- Pre-registered? Yes (COMP_HEADROOM_PREREG.md). P1 held (C > A). P2 unresolved. P3 REFUTED (sd was to fall below 0.05; it roughly doubled - by the pre-stated criterion "not pure optimisation", mechanism unidentified). P4 as predicted (loop matches 3x its parameters; unmeasured).
- Caveats the report must carry: recipe effect (+0.160) exceeds every architectural effect on this task, so all old-recipe compositional conclusions need re-measurement. Variance widens rather than compresses, unlike the torus. D is not parameter-matched.
- Sources: COMP_HEADROOM_PREREG.md, COMP_HEADROOM.md.
- Bears on: recipe/power; hierarchy claim (C05); loops (parameter efficiency).

### C05 Hierarchy recheck at the better recipe
- Dates: 2026-09-08.
- Question: does MapWM-Hier - MapWM-FlatHG = +0.130 survive at cosine/1e-3/150 ep?
- Task / environment: C01; n_traj=200.
- Arms: Hourglass_k2 (MapWM-Hier), HourglassFlat3 (MapWM-FlatHG), both 3-block scaffold, parameter-identical.
- Seeds / batch: 8 each; run_hier_recheck.sh retrains both arms (150 ep, lr 1e-3, cosine).
- Validity gates: the retrained Hourglass_k2 per-seed accuracies are identical to COMP_HEADROOM arm C to all printed digits (HIER_RECHECK.json vs COMP_HEADROOM.json, verified by reading both JSONs) - a same-seed retrain, i.e. determinism, not replication. This also shows the compositional pipeline is bit-deterministic across batches.
- Result: MapWM-Hier 0.514 +/- 0.126 (T=256), 0.390 +/- 0.126 (T=512); MapWM-FlatHG 0.378 +/- 0.103, 0.239 +/- 0.086. Hierarchy +0.136 (sd 0.174, MDE 0.173, 7/8) at T=256; +0.151 (sd 0.157, MDE 0.155, 7/8) at T=512. Both unmeasured. File estimates ~n=13 needed for MDE < 0.136.
- Status: DIRECTIONAL.
- Pre-registered? No separate prereg; motivated by COMP_HEADROOM's "consequence" section.
- Caveats the report must carry: cite as "directional, n=8, unmeasured", never as +0.130 settled. Both arms rose together (hier 0.415 -> 0.514, flat 0.285 -> 0.378).
- Sources: HIER_RECHECK.md, HIER_RECHECK.json, COMP_HEADROOM.json, run_hier_recheck.sh.
- Bears on: hierarchy (the one navigation-side positive for hierarchy).

### C06 Dissociation sweep over motif structure (n_templates)
- Dates: 2026-08-23 (mtime).
- Question: is "which ingredient pays" a property of the task? Sweep `n_templates` in {2,4,8,16}.
- Task / environment: C01 with varying n_templates; each model evaluated on the environment it trained on; floor 0.072/0.072/0.071/0.081.
- Arms: MapWM-Flat, MapWM-Hier, Plain-Flat, Plain-Hier.
- Seeds / batch: n=3, one batch (`runs/dissociation`), 50 epochs, 156 batches, 3 layers (old recipe).
- Validity gates: aliasing covariate measured (`ALIASING_COVARIATE.md`, not read here): per-cell aliasing flat.
- Result (`cross_nb` @T=256, hierarchy effect / path-int effect averaged over the other factor): nt=2 +0.163 / +0.081; nt=4 +0.135 / +0.078; nt=8 +0.034 / +0.174; nt=16 +0.006 / +0.114. `exact_acc` @T=1024: nt=2 +0.120 / +0.060; nt=4 +0.076 / +0.158; nt=8 +0.014 / +0.251; nt=16 -0.015 / +0.154. At nt=16 hierarchy is -0.072 on MapWM and +0.084 on Plain (averaging cancels).
- Status: EXPLORATORY (n=3, no MDE, old recipe).
- Pre-registered? Yes (in `sweep_dissociation.py`). P1 (hierarchy's cross_nb advantage falls with n_templates) confirmed; P2 (path-int exact advantage flat) REFUTED; P3 (the ranking crosses) confirmed on both metrics. The proposed aliasing confound for P2 was ruled out; remaining account untested.
- Caveats the report must carry: MapWM arms seed sd up to +/-0.216; nt=16 column unreliable; MapWM-Flat here is a 3-layer flat, not FlatHG; old recipe.
- Sources: DISSOCIATION_SWEEP.md; BASELINE_TABLE.md sec. E.
- Bears on: hierarchy vs path integration trade-off; task decides the lever.

### C07 Level 1.5 on compositional (correction family's last gap)
- Dates: 2026-08-23.
- Question: does Level15 improve compositional transfer?
- Task / environment: C01, n_templates=4, 50 epochs, 3 layers.
- Arms: Level15, MapWM-Flat, one batch.
- Seeds / batch: n=3, one batch.
- Validity gates: C01. A reporting bug in `eval_compositional.py` (results keyed by variant, silently reporting seed 2 only) was found and fixed here.
- Result: `cross_nb` @T=256 Level15 0.368 +/- 0.220 vs MapWM-Flat 0.260 +/- 0.034; per-seed paired delta -0.043 / +0.401 / -0.034 (2 of 3 negative). `cross_nll` better on 3/3 seeds at both lengths (T=256: 1.269/0.443/1.105 vs 1.365/1.499/1.290; T=1024: 1.791/0.713/1.752 vs 2.515/1.746/2.219).
- Status: EXPLORATORY.
- Pre-registered? No.
- Caveats the report must carry: accuracy mean is one outlier seed; likelihood-not-accuracy pattern across five tasks (table in file) is the correction line's claim.
- Sources: CORRECTION_COMPOSITIONAL.md.
- Bears on: correction line (stabilisation/likelihood, not accuracy).

### C08 Compositional Match-Query (blind continuation in a motif world)
- Dates: gates/results 2026-08-23; writeup 2026-08-24.
- Question: blind continuation where `exact` needs path-integration matching and `cross` needs path integration AND motif abstraction.
- Task / environment: repeated-motif world, explore with observations then blind query; chance 1/16 = 0.0625; oracle 1.000 both categories. Trained T_explore=512, T_query=256, 200 epochs; held-out env seed 10000.
- Arms: Vanilla, Hourglass_k2, Hourglass_CoarseIdx, PlainFlat, PlainHourglass.
- Seeds / batch: three separate sweeps: no-warmup n=3 (COMPOSITIONAL_MATCH_QUERY_RESULTS), warmup 0.05 n=6 (STAB), warmup + blind-horizon curriculum (T_query 16 -> 256 over first half) n=6 (CURRIC).
- Validity gates: n-gram o1/o3/o5 and marginal near chance for both categories at all (TE,TQ); consistency fails 0. **`exact` never-moved baseline 0.0924-0.1777 (up to 2.8x chance)** and `cross` never-moved 0.0709-0.0952 - above chance, not discussed in the writeup. No context destruction run.
- Result (pre-declared bars exact > 0.40, cross > 0.20): warmup-only MapFormer clears exact 7/18, cross 1/18; plain 0/12 both. Curriculum: MapFormer exact 16/18 (Hourglass_k2 6/6), cross 4/18 (Vanilla 1/6, Hourglass_k2 1/6, CoarseIdx 2/6); plain 0/12 both. Curriculum means @TQ=256: exact Vanilla 0.607 +/- 0.284, Hourglass_k2 0.778 +/- 0.179, CoarseIdx 0.709 +/- 0.304, PlainFlat 0.185 +/- 0.025, PlainHourglass 0.184 +/- 0.028; cross 0.153 / 0.177 / 0.227 / 0.115 / 0.113.
- Status: EXPLORATORY (hit-rate counts; no MDE; warmup vs curriculum comparison is cross-batch).
- Pre-registered? Hit-rate bars pre-declared; no prereg file.
- Caveats the report must carry: the exact never-moved gate sits above chance, so plain-model exact ~0.18-0.20 is partly that shortcut; "curriculum converted bimodal to reliable" compares two batches; cross-blind is a frontier (existence), not a result.
- Sources: COMPOSITIONAL_MATCH_QUERY.md, _RESULTS.md, _GATES.md, _STAB.md, _CURRIC.md.
- Bears on: path integration necessary for blind matching; hierarchy (CoarseIdx) exploratory; recipe (curriculum).

### C09 Match-Query 64^2: path integration is necessary for an in-context map
- Dates: 2026-08-09 (file text); mtime 2026-09-07 (MATCH_QUERY_RESULTS).
- Question: with observations withheld in the query phase, can a model predict the observation at revisited cells, and what makes the difference?
- Task / environment: 64^2 torus, n_obs=16; explore T_explore=512 with observations; blind T_query=256; scored at explore-visited non-blank cells, once per episode. Chance 0.0625; marginal 0.068; corrected never-moved floor 0.0893. Held-out env seed 10000. Eval TQ=256 (train) and 512.
- Arms: MapWM-Flat (Vanilla), MapWM-Hier, Plain-Flat, Plain-Hier, PoPE-Flat, MapPoPE-Hier.
- Seeds / batch: n=3 for all arms; MapWM-Flat and PlainFlat extended to n=5. Recipe: 200 epochs, 48 batches, 3 layers, trainer default schedule (LinearLR; `--schedule cosine` added later).
- Validity gates: n-gram o1 0.0625 / o3 0.0689 / o5 0.0664; marginal 0.0677; never-moved CORRECTED 0.0516 -> 0.0893 (1.43x chance). Context destruction (n=3): MapWM-Flat 0.918 -> 0.074 (explore obs shuffled) / 0.076 (query actions shuffled); Plain-Flat 0.146 -> 0.084 / 0.110. PASSES.
- Result: MapWM-Flat **0.730 +/- 0.247 (n=5)**, per seed 0.731 / 1.000 / 0.934 / 0.398 / 0.589; PlainFlat 0.154 +/- 0.018 (n=5), per seed 0.164/0.155/0.139/0.178/0.135. No seed overlap (worst PI 0.398 vs best index 0.178), 10/10. n=3 arms @TQ=256: MapPoPE-Hier 0.847 +/- 0.132, MapWM-Hier 0.786 +/- 0.227, Plain-Hier 0.155 +/- 0.020, PoPE-Flat 0.117 +/- 0.011. Axis is path integration, not encoding (MapPoPE-Hier 0.847 vs PoPE-Flat 0.117, n=3). Hierarchy paired (n=3): MapWM -0.183 / +0.000 / -0.125; Plain +0.012 / -0.002 / -0.003 - "no benefit, possibly a small cost".
- Status: CITABLE (the separation, n=5 vs n=5, no overlap). Hierarchy null and PoPE rows EXPLORATORY (n=3).
- Pre-registered? No.
- Caveats the report must carry: use n=5 0.730, not n=3 0.888. Read against the 0.0893 never-moved floor, not only 0.0625 (index models are 1.7x that floor). Match-Query is not reproducible across batches (epoch-1 losses agree to 7 s.f., final losses 1.91 vs 3.48; CLAUDE.md 2026-09-05/06) - part of its seed sd is landscape noise. Index controls are PlainFlat/PoPE-Flat, not the architecture-matched RoPE (RESULTS_INDEX known-open #5). Five of six arms never extended beyond n=3 (N3_AUDIT item 1).
- Sources: MATCH_QUERY_RESULTS.md, MATCH_QUERY_SCALE.md, MATCH_QUERY_GATES.md, MATCH_GATES_64_16.md, N3_AUDIT.md, RESULTS_INDEX.md.
- Bears on: the headline where/what claim (path integration builds the map); encoding vs position; hierarchy on exact recall.

### C10 Match-Query scale, aliasing boundary, and long blind queries
- Dates: 2026-08-09 (file text).
- Question: does the separation survive a 4x map, heavy aliasing, and blind query phases up to 8x training length?
- Task / environment: 128^2 n_obs=16 (chance 0.0625); 64^2 n_obs=4 (chance 0.25); TQ up to 2048.
- Arms: Vanilla (path-int), PlainFlat (index).
- Seeds / batch: 128^2 and n_obs=4 n=3; LONGQ inference-only on seeds 0-2 of the base config.
- Validity gates: gates re-run per config (128^2 n-gram 0.0640/0.0620/0.0731, never-moved 0.0490; n_obs=4 n-gram 0.2479/0.2484/0.2428, never-moved 0.1515). No context destruction at these configs.
- Result: 128^2 Vanilla 0.823 +/- 0.043 / 0.747 +/- 0.064 / 0.720 +/- 0.075 at TQ=256/512/1024 vs PlainFlat 0.192 +/- 0.022 / 0.164 / 0.150; per seed 0.827/0.778/0.864 vs 0.199/0.209/0.167 (no overlap). n_obs=4: Vanilla 0.510 +/- 0.187 vs 0.332 +/- 0.012; per seed Vanilla 0.512/0.321/0.696 vs PlainFlat 0.345/0.328/0.321 - OVERLAP. LONGQ (base, seeds 0-2): Vanilla 0.904 +/- 0.098 / 0.894 +/- 0.125 / 0.831 +/- 0.197 / 0.693 +/- 0.282 at TQ=256/512/1024/2048; PlainFlat 0.150 / 0.118 / 0.103 / 0.093.
- Status: EXPLORATORY (n=3). The 128^2 separation is re-measured at n=8 under a different recipe in C12 (Q1 +0.348, MDE 0.215, 8/8).
- Pre-registered? No.
- Caveats the report must carry: base-row TQ=1024 cell in MATCH_QUERY_SCALE (0.352 +/- 0.077) is n=2 (weak seeds only) - use LONGQ for length. "No OOD degradation" is withdrawn: correct claim is "degrades gracefully to 8x" (0.904 -> 0.693). Heavy aliasing (n_obs=4) breaks per-seed separation. LONGQ TQ=256 (0.904) differs from MATCH_QUERY_RESULTS n=3 (0.888) on the same seeds (different eval sample; not reconciled in any file).
- Sources: MATCH_QUERY_SCALE.md, MATCH_QUERY_LONGQ.md, MATCH_GATES_128_16.md, MATCH_GATES_64_4.md.
- Bears on: where/what map; length generalisation; aliasing boundary (cross-line with the map-extent/aliasing line).

### C11 Map-Query (absolute position decode): gates and an undertrained run
- Dates: 2026-08-09/10.
- Question: after exploring, can the model answer "which room am I in" / "which direction is the goal"?
- Task / environment: torus, fixed start; room query 64 classes (chance 0.016, best constant 0.023-0.026); direction query chance ~0.50.
- Arms: MapWM-Flat, MapWM-Hier, Plain-Flat, Plain-Hier, PoPE-Flat, MapPoPE-Hier (results table); Vanilla 3-layer diagnostic.
- Seeds / batch: table n=3 at 25 epochs; diagnostic single run at 200 epochs.
- Validity gates: PASS at T_explore >= 256 (n-gram 0.485-0.507, goal-only/explore-only/assume-start 0.487-0.510, mean distance 32.0-32.7 vs uniform 32); FAIL at T_explore=64 (assume-start 0.623, goal-only 0.546). Two gate calibration bugs fixed (set-intersection scoring; saturated H(pos)).
- Result: n=3 table at 25 epochs is UNDERTRAINED - NOT A RESULT (room term at chance until ~epoch 30). Diagnostic (single run, 200 ep): T_explore=16 (fails gate) room 0.994, direction 0.969; T_explore=256 (passes) room 0.1211 (7.6x chance), direction 0.4966 with loss flat at 0.70.
- Status: EXPLORATORY (gates valid; single-run diagnostic only).
- Pre-registered? No.
- Caveats the report must carry: shortcut-free regime and learnable regime do not overlap for the direction query. `train_map_query.py` still defaults to 25 epochs (RESULTS_INDEX). Matching-based redesign proposed, not built.
- Sources: MAP_QUERY_GATES.md, MAP_QUERY_RESULTS.md.
- Bears on: MapFormer's code makes positions comparable, not readable (interpretation); task suite.

### C12 Loop x path integration on Match-Query 128^2 (headroom test)
- Dates: 2026-08-31.
- Question: the torus loop null was at ceiling - does a loop add to path integration where there is headroom?
- Task / environment: Match-Query 128^2, n_obs=16, TE=512, TQ=256, chance 0.0625.
- Arms: index no loop (204,182); index + loop x4 (204,182); path-int 1 layer (204,630); path-int + loop x4 (204,630); path-int 3 real layers (601,174).
- Seeds / batch: n=8 all arms, one batch; 300 ep warmup + cosine, 48 batches, batch 16, fast-attn.
- Validity gates: Match-Query gates (C09/C10); no rule-9 table in the file.
- Result:

  | arm | mean | sd | min |
  |---|---|---|---|
  | index, no loop | 0.108 | 0.025 | 0.06 |
  | index + loop x4 | 0.207 | 0.032 | 0.15 |
  | path-int, 1 layer | 0.456 | 0.220 | 0.11 |
  | path-int + loop x4 | 0.870 | 0.099 | 0.77 |
  | path-int, 3 real layers | 0.771 | 0.263 | 0.14 |

  Q1 path-int - index (no loop) +0.348 (sd 0.218, MDE 0.215, 8/8). Q2 loop on path-int +0.414 (sd 0.279, MDE 0.277, 7/8) DETECTABLE. Q3 loop on index +0.099 (sd 0.045, MDE 0.045, 8/8) DETECTABLE. Q4 loop vs 3 real layers +0.099 (sd 0.255, MDE 0.252, 5/8) underpowered. 2x2 interaction +0.315 (sd 0.283, MDE 0.281) DETECTABLE.
- Status: CITABLE (Q1, Q2, Q3, interaction); Q4 unmeasured.
- Pre-registered? Four questions stated in the file before the n=8 table; the n=3 read ("loop beats 3 layers by +0.273") was retracted at n=8.
- Caveats the report must carry: the loop's contribution is mostly to the floor (8/8 seeds >= 0.77); Q2 clears MDE narrowly because the baseline's variance enters the paired differences. The one seed where 1-layer trained well (0.800) is the one where the loop did not help (-0.029). Super-additivity does NOT generalise to parity (C18, C19). Cross-batch non-reproducibility: the same r=2/no-loop and r=2/loop arms in C13 scored 0.416 and 0.833 against 0.456 and 0.870 here. Scope: one task, one loop count, ALBERT-style sharing, theta computed once.
- Sources: LOOP_HEADROOM.md.
- Bears on: loops; path integration; "loop matches depth at a third of the parameters".

### C13 Match-Query rank x loop
- Dates: 2026-09-04.
- Question: does r=4 remove the same failure mode as the loop?
- Task / environment: Match-Query 128^2, TQ=256, chance 0.0625.
- Arms: r=2 no loop (204,373); r=4 no loop (204,757); r=2 loop x4 (204,373); r=4 loop x4 (204,757). Loop free on both rows; rank costs +384 on both.
- Seeds / batch: n=8, one batch, LOOP_HEADROOM's recipe.
- Validity gates: as C12.
- Result: means (sd, min): r2 no loop 0.416 (0.181, 0.122); r4 no loop 0.421 (0.209, 0.150); r2 loop 0.833 (0.082, 0.713); **r4 loop 0.986 (0.020, 0.941)**. Rank without loop +0.005 (sd 0.092, MDE 0.091, 4/8) unmeasured; rank with loop +0.154 (sd 0.085, MDE 0.084, 8/8) DETECTABLE; loop at r=2 +0.417 (MDE 0.165, 8/8); loop at r=4 +0.565 (MDE 0.219, 8/8); interaction +0.149 (sd 0.113, MDE 0.112, 7/8) DETECTABLE.
- Status: CITABLE.
- Pre-registered? The prediction ("r=4 compresses variance more than it lifts the mean", i.e. the same failure mode as the loop, implying a negative interaction) is stated in the file. It failed: rank alone moves nothing and the interaction is positive (CLAUDE.md: "my pre-registration had the sign backwards").
- Caveats the report must carry: no verdict section in the file itself; best arm on Match-Query measured anywhere. Cross-batch arm values differ from C12 (see C12).
- Sources: MQ_RANK_2X2.md; RESULTS_INDEX.md.
- Bears on: rank (r=4) - cross-line; loops.

### C14 Per-token recursion depth (Mixture-of-Recursions strata)
- Dates: 2026-09-02.
- Question: do tokens want different recursion depths, i.e. is there anything for a MoR router to exploit?
- Task / environment: Match-Query 128^2, T_explore=512, T_query=256, chance 0.0625; strata = 5 equal slices of query-phase position.
- Arms: `Looped` checkpoints evaluated at k in {1,2,3,4,5,6,8} (inference only).
- Seeds / batch: n=8 existing checkpoints.
- Validity gates: eval-only; branch against noise floor after a spread != 0 misread was caught.
- Result: overall k=1 0.384, k=2 0.835, k=3 0.866, k=4 0.865, k=5 0.865, k=6 0.863, k=8 0.856. Per-stratum oracle router 0.854 vs best global count 0.847: upper bound +0.007 against seed sd 0.152.
- Status: POWERED NEGATIVE (an oracle upper bound 22x below seed sd, for the per-position axis).
- Pre-registered? No.
- Caveats the report must carry: per-POSITION heterogeneity only (a hidden-state router could find another axis). "Recursion substitutes for depth" is the premise of the MoR literature, not a finding of this repo.
- Sources: LOOP_DEPTH_STRATA.md.
- Bears on: loops; prior-art positioning.

### C15 Looped pilot on the torus: recursion buys depth's horizon
- Dates: 2026-08-30.
- Question: does a weight-shared block applied 4x buy the attention horizon that real depth buys?
- Task / environment: torus paper task T=128, revisit accuracy bucketed by recurrence interval (1-2 ... 65+); blank floor ~0.50 per bucket.
- Arms: RoPE L1 / L4 / Looped x4; MapFormer (Vanilla) L1 / L4 / Looped x4. Params: Looped 207,457 = L1 207,457; L4 802,273.
- Seeds / batch: n=3, one batch, 300 ep, 5% warmup + cosine, 98 batches, batch 128.
- Validity gates: paper task validated elsewhere; per-bucket blank rates reported.
- Result: RoPE horizon L1 9-16 (17-32: 0.492), L4 17-32 (0.878), Looped 17-32 (0.855); 33-64/65+ all at floor (L1 0.496/0.499, L4 0.523/0.508, Looped 0.512/0.481). MapFormer L1 0.951/0.945, L4 0.999/0.997, Looped 0.999/0.989; long-range means L1 0.948, L4 0.998, Looped 0.994. Loop on index at 17-32: +0.363 (sd 0.018, MDE 0.029, 3/3), vs L4 -0.023 (HORIZON_RESULTS follow-up). Loop on path-int: +0.046 (sd 0.074, MDE 0.120, one seed negative) - at ceiling, uninterpretable.
- Status: EXPLORATORY (n=3; the index-row contrast is detectable at n=3 but was not re-run at higher n on this task).
- Pre-registered? No (Q1/Q2 stated in file).
- Caveats the report must carry: n=3; one width/loop count; the path-int null is a ceiling artifact resolved on Match-Query (C12). This run retracts "scale hurts path integration at long range" (see Excluded) and shows the published horizon table values are lower bounds.
- Sources: LOOPED_PILOT.md, LOOPED_L1.md, LOOPED_L4.md, LOOPED_Loop4.md, HORIZON_RESULTS.md (follow-up section).
- Bears on: loops (iteration vs specialisation); the wall past interval ~32 for index codes.

### C16 Attention horizon grid (index vs path integration by recurrence interval)
- Dates: 2026-08-23 (mtime); retraction block 2026-08-30.
- Question: is the ~2-step index-model horizon architectural or capacity?
- Task / environment: torus paper task T=128, recurrence-interval buckets, floor ~0.50.
- Arms: RoPE and Vanilla at L1 d128 16 ep (204K), L1 d128 50 ep, L2 d128 16 ep (402K), L2 d256 16 ep (1.59M), L4 d128 16 ep (799K), L4 d256 16 ep (3.17M).
- Seeds / batch: n=3; 16 or 50 epochs LinearLR 1.0->0.0.
- Validity gates: per-bucket blank rate reported; budget-limited (rule 10).
- Result: RoPE horizon (largest bucket > floor + 0.10): ~2 / ~8 / ~16 / ~16 / ~16-32 / ~32. RoPE L4 d256 (3.17M) at 33-64 / 65+: 0.485 / 0.503. Vanilla L1 d128 16 ep (204K): 0.935 / 0.880. Under the fair budget (C15) RoPE L1's horizon is 9-16 and index long-range means are 0.498/0.515/0.497 while Vanilla L1 holds 0.945 at 65+.
- Status: EXPLORATORY (budget-limited; horizon values are lower bounds). The "wall" is supported at n=3 under a fair budget by C15.
- Pre-registered? No.
- Caveats the report must carry: every number trained with LinearLR from step one; the "architectural ~2-step bound" is refuted (capacity); the non-monotone Vanilla-in-capacity section is RETRACTED.
- Sources: HORIZON_RESULTS.md, HORIZON_L1d128e16.md, HORIZON_L1d128e50.md, HORIZON_L2d128e16.md, HORIZON_L2d256e16.md, HORIZON_L4d128e16.md, HORIZON_L4d256e16.md.
- Bears on: path integration vs attention reach; loops (C15).

### C17 Cross-task horizon account (CPU analysis)
- Dates: 2026-08-23.
- Question: does the share of scored events within an index model's ~2-step horizon predict its cross-task performance?
- Task / environment: generators of paper task (T=128/512), Match-Query (TE=512 TQ=256), family tree (depth 5, T=64); no model.
- Arms: none (event-lag distributions).
- Seeds / batch: n/a.
- Validity gates: n/a.
- Result: within-horizon share paper T=128 0.191, Match-Query 0.000 (median lag 517), family tree 0.362. Index recovery of headroom: family tree 79%, paper 1.7%, Match-Query 8.1% - not monotone. The single-variable horizon account FAILS.
- Status: CITABLE as a refutation (analytic, no seeds involved) - but see caveat.
- Pre-registered? The prediction is stated at the top of the file before the table.
- Caveats the report must carry: uses stale Match-Query anchors (ceiling 0.888 n=3, floor 0.089) and the ~2-step horizon, which C15/C16 later showed is budget-dependent (9-16 under a fair budget). Floors differ in construction across tasks.
- Sources: HORIZON_TASK_DISTANCES.md.
- Bears on: why index models fail on some tasks; task suite.

### C18 Algorithmic tasks: parity and copy, path integration x loop
- Dates: 2026-09-03.
- Question: does path integration help parity (a cumsum mod 2 register) more than copy; does the loop improve length generalisation; does sampling the loop count help?
- Task / environment: parity (chance 0.5) and copy (chance 0.125); train L=16; eval L=16..256.
- Arms: RoPE (index flat, 199,042), Vanilla (path-int flat, 199,490), RoPELooped (199,042), Looped (199,490), LoopedSampled (199,490).
- Seeds / batch: 8 per arm per task, one batch; 300 ep, 50 batches, lr 1e-3, cosine, 1 layer.
- Validity gates: CPU gates PASS - worst trivial-baseline excess +0.0122 (marginal, n-gram 1/2/3/5, echo-input, repeat-prev) at L=16/32/64.
- Result: parity path-int - index: +0.316 (sd 0.167, MDE 0.165, 8/8) L=16; +0.326 (sd 0.079, MDE 0.078, 8/8) L=32; +0.167 (sd 0.037, MDE 0.036, 8/8) L=64; +0.083 (sd 0.020, MDE 0.020, 8/8) L=128; +0.041 (sd 0.011, MDE 0.011, 8/8) L=256. Copy: no dynamic range (1.000 at L=16, 0.124-0.211 at L>=32). Loop x path-int on raw scale: interaction +0.028 / +0.003 / +0.007 at L=64/128/256, inside MDEs; loop main effect L=128 +0.062 path-int / +0.058 index (additive). At L=32 interaction -0.150 detectable but ceiling-compressed. H3 LoopedSampled - Looped on parity +0.027 (MDE 0.109) / +0.038 (MDE 0.147) / +0.015 (MDE 0.066), unmeasured.
- Status: CITABLE (parity path-int effect, every length). Copy control FAILED. H2 additivity DIRECTIONAL (interactions under MDE). H3 unmeasured.
- Pre-registered? H1-H3 stated in the file. H1 confirmed on parity (control failed). H2: loop helps retention in both rows, additively. H3 unmeasured.
- Caveats the report must carry: the copy control cannot rule out a pipeline artifact. Retention ratios are a normalisation artifact (look super-additive; raw is additive). Prior art: single-layer parity from signed rotations is Selective RoPE sec. 4.2 / Sarrof / Grazzi (cross-line sign inventory). Models far smaller than the looped literature's.
- Sources: ALGORITHMIC_GATES.md, ALGORITHMIC_RESULTS.md.
- Bears on: path integration outside navigation (sign/cancellation); loops (super-additivity is Match-Query-specific).

### C19 Depth vs loop frontier on parity
- Dates: 2026-09-03.
- Question: does looping substitute for depth in both position codes, and is there an interaction?
- Task / environment: parity, accuracy at L=128 (extrapolation point); copy also run.
- Arms: {index, path-int} x {1, 2, 3 real layers, loop x4}. Parity params: index 199,042 / 397,314 / 595,586 / 199,042; path-int 199,490 / 397,762 / 596,034 / 199,490.
- Seeds / batch: n=16 per arm; 300 ep (some 600), lr 1e-3, cosine, fast-attn; venue chosen by `decide_frontier.py` rule fixed before seeing numbers.
- Validity gates: C18 gates.
- Result: parity accuracy index L1 0.519 +/- 0.019, L2 0.541 +/- 0.014, L3 0.565 +/- 0.006, loop 0.575 +/- 0.004; path-int L1 0.598 +/- 0.013, L2 0.632 +/- 0.033, L3 0.641 +/- 0.090, loop 0.667 +/- 0.077. Loop - L1: index +0.056 (sd 0.021, MDE 0.014, 16/16); path-int +0.070 (sd 0.080, MDE 0.056, 14/16); difference +0.014 (not detectable). Loop - L3: index +0.010 (sd 0.005, MDE 0.003, 16/16) DETECTABLE; path-int +0.026 (sd 0.146, MDE 0.102, 8/16) unmeasured. Stacking: index L1 0.519 -> + path-int 0.598 (+448 params) -> + loop 0.575 (0 params) -> both 0.667 (additive prediction 0.653). Copy: every arm at chance (0.124-0.144) - vacuous.
- Status: CITABLE (loop gains, index loop > L3, no detectable interaction); path-int loop vs L3 unmeasured.
- Pre-registered? Venue-choice rule pre-fixed; contrasts not separately preregistered.
- Caveats the report must carry: parameters-and-memory result, NOT FLOPs (four passes cost four). Path-int row much noisier (sd 0.077-0.146 vs 0.004-0.021). Three depth points. The file's claim that LOOP_HEADROOM's interaction "does not survive" is about parity; the Match-Query interaction (C12) stands as task-specific.
- Sources: FRONTIER_ALGORITHMIC.md.
- Bears on: loops ("best arm is also the smallest"); path integration per parameter.

### C20 Hierarchy on parity at L=16
- Dates: 2026-09-03.
- Question: parity is a tree reduction (pooled partial parity is a sufficient statistic) - does hierarchy help?
- Task / environment: parity, train L=16, eval to L=256.
- Arms: index flat 3-layer / index hier (595,586); path-int flat 3-block / path-int hier (596,034).
- Seeds / batch: n=16, 300 ep, lr 1e-3, cosine.
- Validity gates: C18 gates.
- Result: hier - flat at L=128: index +0.001 (sd 0.009, MDE 0.006, 9/16); path-int -0.014 (sd 0.156, MDE 0.109, 8/16). At L=16: index +0.020 (MDE 0.049, 15/16); path-int +0.012 (sd 0.002, MDE 0.002, 16/16).
- Status: POWERED NEGATIVE for the index row at L=16 training (MDE 0.006); path-int row unmeasured. The conclusion drawn ("the sufficient-statistic principle is not predictive") is WITHDRAWN (C21).
- Pre-registered? Prediction stated in file (tree reduction should decay flatter); failed at this length.
- Caveats the report must carry: at L=16 k=2 pooling leaves 8 coarse tokens - the mechanism cannot operate; superseded in part by C21.
- Sources: HIER_PARITY.md.
- Bears on: hierarchy (sufficiency is necessary; length also required).

### C21 Loop x hierarchy on parity at L=512
- Dates: 2026-09-03.
- Question: is combining weight sharing and hierarchy free on accuracy; does hierarchy help parity when trained long?
- Task / environment: parity trained at L=512, eval L=512/1024/2048.
- Arms: unshared FLAT (HourglassFlat3, 596,034), unshared HIER (Hourglass_k2, 596,034), SHARED FLAT (LoopedHourglassFlat, 199,490), SHARED HIER (LoopedHourglass, 199,490).
- Seeds / batch: n=12 per arm (per the sign counts x/12 and MDE arithmetic in the tables); 150 ep, lr 1e-3, cosine, 30 batches.
- Validity gates: C18 gates (at shorter L).
- Result: means L=512/1024/2048: unshared flat 0.936/0.889/0.806; unshared hier 0.995/0.948/0.832; shared flat 0.891/0.813/0.699; shared hier 0.962/0.897/0.800. Hier - flat: unshared +0.060 (sd 0.093, MDE 0.075, 12/12) L=512, +0.059 (MDE 0.150, 8/12), +0.026 (MDE 0.174, 7/12); shared +0.071 (sd 0.100, MDE 0.081, 11/12), +0.085 (MDE 0.176, 9/12), +0.101 (MDE 0.230, 9/12). At L=512 t +2.23 (sign-test p=0.0005) and t +2.44 (p=0.006). LoopedHourglass vs unshared flat: +0.026 / +0.009 / -0.006, not detectable.
- Status: DIRECTIONAL by the MDE rule (both hierarchy contrasts at L=512 under MDE) but sign-consistent 12/12 and 11/12 (sign test p=0.0005 / 0.006). Pairing-is-free equivalence DIRECTIONAL (bounded only by MDE ~0.180 at L=2048 per file).
- Pre-registered? The "null on accuracy is the success case" framing is stated up front.
- Caveats the report must carry: internal inconsistency - the Scope boilerplate says "n=16" and "no looped-hierarchical variant exists", which is stale text from HIER_PARITY; tables are n=12 and include looped-hierarchical arms. The report must state which significance rule it uses. Hierarchy washes out at L=1024/2048.
- Sources: LOOP_HIER_PARITY.md.
- Bears on: hierarchy (sufficient statistic + long sequence); loops x hierarchy efficiency.

### C22 Loop x hierarchy compute benchmark
- Dates: 2026-09-03.
- Question: do parameter savings (sharing) and compute savings (hierarchy) compose?
- Task / environment: forward+backward, batch 64, timed alone on an idle device with synchronisation; L in {16,128,512,2048}.
- Arms: HourglassFlat3 (596,034), Hourglass_k2 (596,034), LoopedHourglassFlat (199,490), LoopedHourglass (199,490).
- Seeds / batch: single timing per cell.
- Validity gates: warmup + explicit synchronisation.
- Result: at L=2048 Hourglass_k2 -22.9% time, -19.9% memory; LoopedHourglass -22.8% time, -19.9% memory, -66.5% parameters. At L=512 -21.8% / -17.0% (hier) and -21.7% / -17.2% (both). At L=16 hierarchy costs +12.2% time; both +9.0% time.
- Status: CITABLE (as a direct measurement; no repeat timings).
- Pre-registered? No.
- Caveats the report must carry: length-dependent - hierarchy is slower at L=16; single timing per cell.
- Sources: LOOP_HIER_COMPUTE.md.
- Bears on: hierarchy as efficiency; loops save parameters not compute.

### C23 Sampled loop count (LoopedSampled) on the torus
- Dates: 2026-09-01.
- Question: can the loop's length trade-off (fixed-4 model peaks at 4 passes at T=128, 2 at T=512) be trained away by sampling the count from {2..6}?
- Task / environment: torus paper task, held-out map, p_action_noise in {0, 0.1}, eval T=128/512/1024 at loops {1,2,3,4,6}.
- Arms: Vanilla, Looped, LoopedSampled.
- Seeds / batch: n=5 (3 arms x 2 noise x 5 seeds; run_loop_sampled.sh), 300 ep, 98 batches, batch 128, T=128.
- Validity gates: none beyond the paper task's.
- Result (p=0): Looped at 1 loop 0.821 +/- 0.191 (T=128) vs LoopedSampled 0.998 +/- 0.003; count spread at T=128 Looped 0.821-1.000 vs LoopedSampled 0.998-0.999. Best counts: Looped T=512 2 loops 0.823, T=1024 2 loops 0.651; LoopedSampled T=512 2 loops 0.915, T=1024 2 loops 0.736. Vanilla 0.965 / 0.892 / 0.767. OOD gain +0.092 (T=512) / +0.085 (T=1024) with t=1.67 / 1.80 at n=5 (numbers only in CLAUDE.md). p=0.1: best LoopedSampled 0.878 / 0.695 / 0.600 vs Looped 0.892 / 0.678 / 0.585.
- Status: DIRECTIONAL (OOD gain); the flattening of the count curve is descriptive and large, EXPLORATORY in the formal sense (no paired test in file).
- Pre-registered? No.
- Caveats the report must carry: n=5; t values and "+0.092/+0.085" appear only in CLAUDE.md; does not transfer to noise; even repaired the loop does not beat Vanilla at T=1024 clean (0.736 vs 0.767). Sampling vs fixed on parity is unmeasured (C18 H3).
- Sources: LOOP_SAMPLED.md, LOOP_SAMPLED.json, run_loop_sampled.sh, CLAUDE.md (2026-08-31/09-01 section 3).
- Bears on: loops (runtime depth knob; cheaper inference).

### C24 Filter x loop 2x2 on the torus
- Dates: 2026-09-01.
- Question: are the Level 1.5 filter and the loop complementary?
- Task / environment: clean torus paper task, held-out map, T=128/512/1024.
- Arms: Vanilla 204,373; Level15 253,973; Looped 204,373; Level15Looped 253,973 (bit-identical to Level15 at n_loops=1, causal leak 0); LoopedSampled 204,373.
- Seeds / batch: n=12, one batch, 300 ep warmup + cosine.
- Validity gates: rule 9 - r(final loss, acc) -0.956 / -0.471 / -0.326 at T=128/512/1024 over 60 runs; mean final loss Vanilla 0.1549, Level15 0.0420, Looped 0.0076, Level15Looped 0.0189, LoopedSampled 0.0180.
- Result: accuracies T=128/512/1024 Vanilla 0.947/0.876/0.749; Level15 0.990/0.953/0.878; Looped 0.999/0.872/0.730; Level15Looped 0.994/0.929/0.830; LoopedSampled 0.997/0.905/0.745. Interaction raw / loss-matched: T=128 -0.048 (MDE 0.064) / -0.009 (MDE 0.017); T=512 -0.020 (MDE 0.136) / +0.026 (MDE 0.099); T=1024 -0.029 (MDE 0.190) / +0.022 (MDE 0.147) - all UNMEASURED. Looped - Vanilla T=128 raw +0.052 (sd 0.060, MDE 0.048, 12/12) but loss-matched +0.006 (MDE 0.017). Level15 - Vanilla T=1024 raw +0.129 (sd 0.127, MDE 0.103, 10/12), loss-matched +0.083 (sd 0.102, MDE 0.083, 9/12; file labels UNMEASURED).
- Status: Interaction unmeasured; loop's training-length advantage is CONVERGENCE (loss-matched +0.006) - DIRECTIONAL-to-null; combination is below the filter alone at OOD (levels).
- Pre-registered? Yes (verdict "against the pre-registration" in file). Not super-additive.
- Caveats the report must carry: loss-matching is well justified only at T=128 (r collapses with length). LoopedSampled evaluated at 4 passes, not its best count (understated by ~0.017). Filter main effect carries a 49,600-param capacity gap.
- Sources: L15_LOOP_2X2.md.
- Bears on: loops (convergence aid); correction line (cross-line).

### C25 Recipe power on the torus (lr 1e-3)
- Dates: 2026-09-02.
- Question: which existing-knob recipe cuts seed variance (the quantity setting every MDE)?
- Task / environment: clean torus paper task, T=128/512/1024.
- Arms: Vanilla, Looped under C0 (300 ep, lr 3e-4), C1 (300 ep, lr 1e-3), C2 (600 ep, lr 1e-3).
- Seeds / batch: n=8 per cell, one batch, parallel data path (`--data-workers 3`), cosine.
- Validity gates: converged fraction (final loss < 0.05, flat tail): Vanilla 3/8, 4/8, 6/8; Looped 8/8 in all.
- Result: Vanilla sd C0 -> C1: T=128 0.086 -> 0.017 (4.9x), T=512 0.096 -> 0.028 (3.5x), T=1024 0.110 -> 0.064 (1.7x); means 0.936 -> 0.993, 0.863 -> 0.944, 0.735 -> 0.834. C2 Vanilla sd at T=1024 0.158 (worst). Looped roughly unaffected (T=512 sd 0.073 -> 0.091).
- Status: CITABLE (as a variance measurement; no formal test of sd differences).
- Pre-registered? Yes (primary metric converged fraction declared in file). The decision rule was broken (required gains on every arm; fixed to Pareto) and the primary metric was the wrong proxy (converged fraction ranks C2 best; accuracy sd ranks C1 best, C2 worst at T=1024).
- Caveats the report must carry: bimodality not eliminated (Vanilla 4/8 converged). Recipe transfer fails: on Match-Query p=0 lr 1e-3 moved Vanilla sd 0.263 -> 0.261 (MQ_NOISE_2X2_C2, cross-line, CLAUDE.md); on compositional it doubled variance (C04). Data-parallel stream differs from the serial path's.
- Sources: RECIPE_POWER.md, _RECIPE_C0.md, _RECIPE_C1.md, _RECIPE_C2.md (per-condition renders with a stale "refining theta" header).
- Bears on: power for every torus claim; recipe-before-architecture method.

### C26 enwik8: hierarchy in the MapFormer family (parameter-identical pair)
- Dates: 2026-08-28 (runs); ENWIK8_HIERARCHY.md 2026-09-07.
- Question: does pooling help next-byte prediction at exact parameter parity?
- Task / environment: byte-level enwik8, 36k iters, seq 512, batch 16, lr 2e-4, dim 880, r=4, seed 42, deterministic val (fixed generator).
- Arms: MapWM-Hier (28,371,016), MapWM-FlatHG (28,371,016); exploratory MapPoPE-Hier (28,372,336), PoPE-Hier (28,367,936).
- Seeds / batch: n=1 per arm.
- Validity gates: deterministic val verified bit-identical (KNOWN_BUGS); checkpoint sd 0.003-0.007.
- Result: final val bpc MapWM-Hier 1.4537 vs MapWM-FlatHG 1.4506 (+0.0032, hierarchy worse, inside checkpoint noise). Mean of last 5 checkpoints (ENWIK8_HIER): 1.4609 vs 1.4598 (+0.0010). Efficiency, measured alone on an idle GPU: 20.10 vs 16.36 it/s (1.23x, -18.6% step time), peak memory 2.62 vs 3.05 GiB (-14.1%), analytic block FLOPs -17.4%; attention is 8.8% of a block at d=880, L=512; FLOP saving -19.0% at 2048, -21.7% at 8192, ceiling -25%.
- Status: EXPLORATORY on quality (n=1; consistent with a null, not proof); efficiency is a direct measurement.
- Pre-registered? No.
- Caveats the report must carry: do not quote the earlier -8.6% wall time (GPU co-tenancy). The saving is linear (FFN tokens), not the quadratic attention win. PoPE arms are rank-confounded (MapPoPE-Hier trained at r=2 via the `_widen_to_d` bug) and have no flat control. Exact-recall objective is the regime where hierarchy is expected to lose.
- Sources: ENWIK8_HIERARCHY.md, ENWIK8_HIER.md, enwik8_long/*_h880.json, KNOWN_BUGS.md.
- Bears on: hierarchy as efficiency on text.

### C27 enwik8: plain Hourglass scaffold (Gate B)
- Dates: 2026-07-23 (JSON mtime).
- Question: does the Hourglass scaffold reproduce the Nawrot et al. efficiency property at equal params?
- Task / environment: enwik8, seq 2048, 8000 iters.
- Arms: hourglass (shorten 4) vs flat10, both 31,787,264 params.
- Seeds / batch: n=1.
- Validity gates: none; predates the deterministic-val fix.
- Result: val bpc hourglass 1.4844 vs flat10 1.4727 (+0.0117 hierarchy worse) with -18.75% FLOPs and -17.6% wall time (CLAUDE.md CORRECTED 2026-08-28; JSON `hourglass_enwik8_long/{hourglass,flat10}.json` final val_bpc 1.48438 / 1.47273). Earlier partial run 1.5099 vs 1.4973.
- Status: EXPLORATORY.
- Pre-registered? No.
- Caveats the report must carry: the "hourglass ~2.00 vs flat ~2.07, hierarchy better" statement in COMPOSITIONAL_EXPERIMENT.md finding 5 and HOURGLASS_README is WRONG (no data; sign inverted). Val was resampled per checkpoint and per model at the time (KNOWN_BUGS), swings 0.02-0.07; wall time may be co-tenancy-contaminated.
- Sources: hourglass_enwik8_long/hourglass.json, flat10.json; CLAUDE.md; ENWIK8_HIERARCHY.md; KNOWN_BUGS.md.
- Bears on: hierarchy as efficiency on text.

### C28 enwik8: PoPE x path integration on text
- Dates: 12k run 2026-08-27; 36k and seeds 2026-08-27/28.
- Question: do PoPE and path integration each help, and compose, on byte-level text?
- Task / environment: enwik8, seq 512, batch 16, lr 2e-4, 36k iters, r=4 for path-integrating arms, param-matched to 0.03% (~28.6M).
- Arms: RoPE (index/RoPE), PoPE-Flat (index/PoPE), Vanilla_r4 (path-int/RoPE), MapPoPE-Flat_r4 (path-int/PoPE).
- Seeds / batch: RoPE and MapPoPE-Flat_r4 n=3; PoPE-Flat and Vanilla_r4 n=1.
- Validity gates: deterministic val stated in ENWIK8_SEEDS; final-checkpoint slopes still negative (-0.0020 to -0.0081 per 1000 iters, ENWIK8_LONG).
- Result: ENWIK8_SEEDS (mean of last 5 checkpoints): MapPoPE-Flat r4 - RoPE = -0.0078 / -0.0043 / -0.0036, mean -0.0052, sd 0.0023, 3/3. Single-seed arms: PoPE-Flat 1.3806, Vanilla r4 1.3841 (last-5 means); MapPoPE vs PoPE-Flat -0.0020 at n=1. Final-checkpoint means (CLAUDE.md, from enwik8_long JSONs): MapPoPE-Flat 1.3740 (n=3), PoPE-Flat 1.3746 (n=1), Vanilla 1.3758 (n=1), RoPE 1.3799 (n=3); MapPoPE - RoPE -0.0058, t=3.49. CLAUDE.md converts the paper's RoPE 19.14 vs MapWM 18.79 ppl to ~0.0067 bits/byte, above this setup's MDE ~0.0041 at n=3.
- Status: MapPoPE vs RoPE DIRECTIONAL-to-CITABLE (3/3; no MDE stated in the source files; CLAUDE.md reports t=3.49). Composition NOT ESTABLISHED (components n=1).
- Pre-registered? No.
- Caveats the report must carry: two metrics in use (final checkpoint vs mean of last 5) give different numbers; still improving at 36k; 295M tokens vs the paper's 100B; the path-int-alone effect is underpowered, not null.
- Sources: ENWIK8_SEEDS.md, ENWIK8_LONG.md, ENWIK8_HIER.md (flat reference rows), enwik8_long/*.json, CLAUDE.md (2026-08-31/09-01 section 5).
- Bears on: language (cross-line with encoding/PoPE line).

### C29 Early hierarchical-attention variants on the retrieval task
- Dates: 2026-07-16..22.
- Question: do pooled (HierAttn), routed (RouteAttn), learned-readout recursive, space-time, or bounded-memory hierarchies beat flat Level15 on revisit retrieval at long T?
- Task / environment: torus revisit prediction, clean, trained n_steps=256, eval T=256..4096.
- Arms: Level15 (253,973 / 254K, 1-layer), Level15_L2 (452K), HierAttn, RouteAttn (253,973), RouteAttn_NoBias, RouteAttn_K4, Recursive (485K), SpaceTimeHier (485K), BoundedFlat / BoundedHier (read budget M=128).
- Seeds / batch: mostly n=1-3 (+/- where shown); ROUTE_ATTN "all arms same batch"; April-July recipe (not stated in files).
- Validity gates: coarse-contribution diagnostic: zeroing coarse_proj changes SpaceTimeHier accuracy by 0.0000 at T=1024, ||coarse||/||fine|| = 0.030 - the coarse level is INERT; SpaceTimeHier and Recursive byte-identical. HIER_ATTN_LONGT final train loss Level15 0.0013 vs HierAttn 0.1197 (unconverged comparison).
- Result: T=4096: Level15 0.861 vs HierAttn 0.769 (HIER_ATTN_LONGT, seed 0); ROUTE_ATTN Level15 0.849 +/- 0.011, RouteAttn 0.708 +/- 0.069, HierAttn 0.764 +/- 0.007, RouteAttn_NoBias 0.832 +/- 0.018. T=2048: SpaceTimeHier 0.833 vs Level15 1-layer 0.955; Recursive 0.833 vs Level15_L2 0.945 +/- 0.071. Bounded: BoundedHier 0.752 vs BoundedFlat 0.770 at T=4096 (Level15 unbounded 0.860, n=1).
- Status: EXPLORATORY (every hierarchy variant loses to flat on retrieval).
- Pre-registered? No.
- Caveats the report must carry: design flaw owned - the fine level was local, so the coarse path had no job; unconverged HierAttn; n<=3, old recipe; comparator is Level15 (correction line), not plain MapWM.
- Sources: HIER_ATTN_LONGT.md, ROUTE_ATTN_RESULTS.md, RECURSIVE_RESULTS.md, SPACETIME_HIER_RESULTS.md, BOUNDED_MEMORY_RESULTS.md.
- Bears on: hierarchy costs precise retrieval.

### C30 Aggregate (windowed-majority) task
- Dates: 2026-07-17/18.
- Question: does hierarchy win when the target is a long-window aggregate rather than a retrieval?
- Task / environment: windowed-majority obs type (W_agg=128), clean, trained n_steps=256; chance ~0.11.
- Arms: Level15 (flat), HierAttn, HierAttn_CoarseOnly, HierAttn_LocalOnly, Level15 trained at n_steps=512.
- Seeds / batch: main pair n=3; ablations and training-length control n=1.
- Validity gates: none recorded (no n-gram, no context destruction).
- Result: n=3 T=256/512/1024/2048: Level15 0.865 / 0.694 / 0.553 / 0.401; HierAttn 0.831 / 0.718 / 0.620 / 0.537. Training-length control (n=1): Level15 trained at 512 scores 0.746 / 0.629 / 0.539 at T=512/1024/2048 - matching HierAttn's 0.537 at T=2048. Ablation (n=1): CoarseOnly 0.522, LocalOnly 0.453 at T=2048.
- Status: EXPLORATORY; the hierarchy "win" at T=2048 is attributed to a training-length confound (RESULTS_INDEX live negatives).
- Pre-registered? No.
- Caveats the report must carry: see Unresolved disagreements (still listed as a hierarchy win in CLAUDE.md and ENWIK8_HIERARCHY.md); control is n=1.
- Sources: AGGREGATE_TASK_RESULTS.md, AGGREGATE_MULTISEED.md, AGGREGATE_EXTRAS.md, RESULTS_INDEX.md.
- Bears on: hierarchy (long-horizon aggregation claim).

### C31 Lap task (CSCG port) and lap-transfer forgetting controls
- Dates: 2026-08-09 (file text).
- Question: can MapFormer distinguish same-place-different-lap; does lap training degrade the map?
- Task / environment: circuit with exactly zero net displacement, K=4 laps, one reward boundary; variable loop length; headline `exact` (hit right boundary, no false alarms); random-boundary floor 0.250; always-say-no boundary accuracy 0.750. Transfer: phase 1 Match-Query, phase 2 lap / lap-without-reward / more Match-Query, phase 3 re-measure.
- Arms: Vanilla (MapWM-Flat), ~600K params in transfer.
- Seeds / batch: probe single seed (60 ep); transfer n=3 per arm.
- Validity gates: positional shortcut fixed-length 1.000 (invalid) vs variable 0.163 (valid operating point); n-gram boundary acc o1/o2/o4/o8 0.250/0.673/0.750/0.750 (at always-no level). The no-reward control differs at exactly one token per episode (verified).
- Result: probe exact 1.000 at K=4, 0.000 at K=6 (OOD). Transfer MQ before -> after: lap 0.377 -> 0.083 (-0.293; -0.317/-0.308/-0.255); lap without reward 0.377 -> 0.085 (-0.291; -0.292/-0.321/-0.260); same-task control +0.002. Lap exact after phase 2 0.993.
- Status: EXPLORATORY (probe n=1; transfer n=3). Established: degradation is catastrophic forgetting under distribution shift, not lap counting.
- Pre-registered? Yes (predictions in LAP_GATES/LAP_TRANSFER): "MapFormer cannot distinguish laps" REFUTED; "theta drift rises" REFUTED (obs/act Delta fell 0.252 -> 0.060).
- Caveats the report must carry: mechanism of lap solving unknown; theta-drift and obs/act Delta metrics are NOT diagnostic; phase-1 map was mediocre (0.377, constant LR).
- Sources: LAP_GATES.md, LAP_TRANSFER.md, LAP_TRANSFER_NOREWARD.md.
- Bears on: event vs place coding (CSCG); probe calibration method.

### C32 Hier-goal context-destruction and closed-loop audit (DIAGNOSTIC)
- Dates: 2026-08-09 (file text).
- Question: are hier-goal models navigating or exploiting a shortcut?
- Task / environment: hier-goal `[room_goal, local_goal, explore, navigate(BFS)]`, T_explore=128 OOD, T_navigate=64; copy-previous-action floor 0.327 (interleaved); closed-loop success within 96 steps, random floor 0.010, BFS oracle 1.000.
- Arms: MapWM-Flat, MapWM-Hier, Plain-Flat, Plain-Hier, PoPE-Flat, MapPoPE-Hier.
- Seeds / batch: n=3; closed-loop n_trials=200.
- Validity gates: this IS the gate.
- Result: destroy_context (goal + all explore actions and observations randomised) leaves accuracy unchanged: MapWM-Flat 0.912 -> 0.913, Plain-Flat 0.915 -> 0.916, PoPE-Flat 0.935 -> 0.938. Action-only Markov predictors: raw BFS o1/o3/o5 0.969/0.969/0.969; interleaved 0.320/0.971/0.974. Closed-loop success 0.013-0.037 across variants and T_explore. Raw-BFS per seed at T_explore=128: MapWM-Flat 0.526/0.942/0.470 (bimodal), MapWM-Hier 0.869/0.891/0.934 - all below the 0.969 copy-previous baseline.
- Status: CITABLE (as the audit that voids the task).
- Pre-registered? No.
- Caveats the report must carry: validating a fix at order 1 only certified a stronger order-3 shortcut.
- Sources: HIERGOAL_ABLATION.md, HIERGOAL_CLOSEDLOOP.md, archive/void/README.md.
- Bears on: task suite; rules 1-2.

### C33 Planner-task audit (DIAGNOSTIC)
- Dates: 2026-08-09.
- Question: are planner-demonstration tasks solvable from the action stream alone?
- Task / environment: goal, rooms_goal, rooms_maze, maze_varying, hier_goal as positive control; chance 0.250.
- Arms: none (n-gram orders 1-5).
- Seeds / batch: n/a.
- Validity gates: hier_goal positive control 0.969.
- Result: goal 0.969/0.968/0.967/0.967/0.966; rooms_goal 0.969/0.968/0.968/0.968/0.968; rooms_maze 0.791-0.793; maze_varying 0.650/0.645/0.643/0.623/0.567. All VOID (threshold chance + 0.25).
- Status: CITABLE (audit).
- Pre-registered? Thresholds stated in file.
- Caveats the report must carry: voids 13 result files including the +7.5pp frozen-probe result; ROOMS_GOAL_RESULTS.md (not archived) is on the rooms_goal task and is void by this audit.
- Sources: PLANNER_TASK_AUDIT.md.
- Bears on: task suite.

### C34 MiniGrid MemoryS13 (single seed)
- Dates: 2026-05-02.
- Question: Vanilla vs Level15 vs RoPE on a MiniGrid memory topology.
- Task / environment: MiniGrid-MemoryS13 (start room, hallway, choice room), revisit prediction, OOD T=128/512/1024.
- Arms: Vanilla, Level15, RoPE.
- Seeds / batch: single seed; cached buffer; May-era recipe.
- Validity gates: none recorded.
- Result: T=512 Vanilla 0.763 (NLL 1.561), Level15 0.897 (0.329), RoPE 0.796 (0.796); T=1024 0.674 / 0.809 / 0.731.
- Status: EXPLORATORY.
- Pre-registered? No.
- Caveats the report must carry: n=1, no gates, no context destruction; LinearLR era.
- Sources: MINIGRID_MEMORY_RESULTS.md.
- Bears on: correction line, environment line (cross-line).

---

## Excluded

| file / claim | reason | killed by |
|---|---|---|
| hier-goal: "MapWM-Hier best at OOD explore length", +0.09-0.10 super-additive interaction, "MapFormer x hierarchy synergy" (COMPOSITIONAL_EXPERIMENT follow-up section, CLAUDE.md) | task solvable from navigate-phase action prefix; destroy_context 0.912 -> 0.913 | HIERGOAL_ABLATION.md |
| HIERGOAL_RESULTS/MULTISEED/FIXED/FIXED_LONGT (archive/void) | same | HIERGOAL_ABLATION.md, archive/void/README.md |
| HIERGOAL_LONGT.md (not archived) | evaluates hier-goal checkpoints; "PoPE-Flat holds 0.95 to T=2048" explained as order-3 continuation | HIERGOAL_ABLATION.md |
| HIERGOAL_CLOSEDLOOP.md as a model comparison | kept only as diagnostic (all variants at ~random floor) | HIERGOAL_ABLATION.md |
| BOUNDED_MEMORY.md (sliding-window eval) | runs on hier-goal checkpoints/task (goal prefix, T_explore) | HIERGOAL_ABLATION.md |
| goal / rooms_goal / rooms_maze / maze_varying results incl. frozen-probe +7.5pp | action-only n-gram 0.650-0.969 vs 0.250 | PLANNER_TASK_AUDIT.md |
| ROOMS_GOAL_RESULTS.md (not archived) | rooms_goal task, action-only n-gram 0.969 | PLANNER_TASK_AUDIT.md |
| DOORKEY_BC_RESULTS.md, DAGGER_RESULTS.md, DAGGER_DK6_RESULTS.md | SUSPECT planner-demonstration tasks never n-gram-validated; single seed | banners in each file (2026-08-09) |
| DAGGER_EMPTY_RESULTS.md | pipeline sanity check only (all 1.00), same SUSPECT banner | banner |
| "EM wins on DoorKey match-acc, mechanism-consistent" (CLAUDE.md 2026-05-10) | rests on unvalidated BC task, n=1 | DOORKEY_BC banner |
| Loop beats three real layers by +0.273 (n=3) | at n=8 Q4 +0.099, MDE 0.252 - matches, does not beat | LOOP_HEADROOM.md |
| "Scale hurts the path-integrated model at long range" (Vanilla L4 d256 0.782 at 65+) | 16-epoch LinearLR artifact; 300 ep cosine gives L1 0.948, L4 0.998, Looped 0.994 | LOOPED_PILOT.md, HORIZON_RESULTS.md retraction block |
| "Attention has a fixed ~2-step horizon" | capacity/budget; RoPE L1 9-16 under fair budget | HORIZON_RESULTS.md, LOOPED_PILOT.md |
| "Sufficient-statistic principle is not predictive" (HIER_PARITY verdict) | L=16 cannot exercise pooling; L=512 hierarchy helps 23/24 seeds | LOOP_HIER_PARITY.md |
| Match-Query base 0.888 +/- 0.140 (n=3) | n=5 gives 0.730 +/- 0.247 | MATCH_QUERY_SCALE.md, MATCH_QUERY_RESULTS.md header |
| Match-Query "no OOD degradation" | 0.904 -> 0.693 at TQ=2048 | MATCH_QUERY_SCALE.md, MATCH_QUERY_LONGQ.md |
| Match-Query base TQ=1024 0.352 +/- 0.077 | n=2 weak-seed selection | MATCH_QUERY_SCALE.md correction |
| Match-Query never-moved gate 0.0516 "PASS" | scorer bug; true floor 0.0893 | MATCH_QUERY_GATES.md correction |
| Map-Query n=3 chance-level table | undertrained (25 epochs) - not a result | MAP_QUERY_RESULTS.md banner |
| Lap: "MapFormer solved laps by abandoning path integration"; theta-drift / obs-Delta as diagnostics; "lap task conflicts with cognitive maps" | Match-Query model scores 0.252/4.37 vs lap 0.188/3.86; no-reward control -0.291 ~= -0.293 | LAP_GATES.md retraction, LAP_TRANSFER_NOREWARD.md |
| CSCG stitch and schema tasks | stitch negative control defeatable (0.617); ~14% of shared events need stitching; schema env is not schema transfer; 71.9% border steps | CSCG_TASK_GATES.md banner |
| enwik8 "hourglass ~2.00 vs flat ~2.07, hierarchy better" (COMPOSITIONAL_EXPERIMENT finding 5, HOURGLASS_README) | no saved data; actual 1.4844 vs 1.4727, sign inverted | CLAUDE.md CORRECTED 2026-08-28, ENWIK8_HIERARCHY.md, KNOWN_BUGS.md |
| enwik8 hierarchy "-8.6% wall time" | GPU co-tenancy; isolated 1.23x | ENWIK8_HIERARCHY.md correction |
| ENWIK8_2X2.md (12k-iter PoPE x path-int table) | non-deterministic val "invalidated the entire 12k language result"; "every arm still improving" overstated | KNOWN_BUGS.md, ENWIK8_2X2.md correction |
| enwik8 "PoPE and path integration compose on language" | components at n=1; vs PoPE-Flat -0.0020 < checkpoint sd | ENWIK8_SEEDS.md correction |
| COMPOSITIONAL_RESULTS.md (single seed) | superseded by multi-seed; lead inflated by one seed | its own banner |
| "hierarchy +0.145 over MapWM-Flat" | crosses seed counts (n=8 vs n=3); cite +0.130 vs FlatHG | N3_AUDIT.md sec. 3, BASELINE_TABLE.md D |
| "Hierarchy buys compositional transfer" as settled | recipe effect +0.160 > +0.130; recheck +0.136 under MDE 0.173 | COMP_HEADROOM.md, HIER_RECHECK.md |
| "Aggregate T=2048 0.537 vs 0.401" as a hierarchy win | flat trained at 512 reaches 0.539 (n=1) - training-length confound | AGGREGATE_EXTRAS.md, RESULTS_INDEX.md live negatives (disputed, see below) |
| "Recipe fixes power" as general | Match-Query sd unchanged; compositional sd doubled | MQ_NOISE_2X2_C2 via CLAUDE.md, COMP_HEADROOM.md P3 |
| COMPOSITIONAL_EXPERIMENT's "WM (additive OR)" rationale for H1 | MapWM is not additive | AUDIT_2026-09-10 (via CLAUDE.md) |
| HierAttn/SpaceTimeHier/Recursive coarse level as a contributing module | coarse contribution 0.0000 when zeroed; inert | SPACETIME_HIER_RESULTS.md / RECURSIVE_RESULTS.md post-hoc |
| LOOP_SAMPLED "adaptivity rescues length generalisation" as general | does not replicate on parity at n=8 (H3 unmeasured) | ALGORITHMIC_RESULTS.md |

## Unresolved source disagreements

1. **Aggregate task as a hierarchy win.** RESULTS_INDEX.md (regenerated 2026-09-06) calls it a training-length confound; ENWIK8_HIERARCHY.md (2026-09-07) and CLAUDE.md's hierarchy summary still list "aggregate T=2048 0.537 vs 0.401" among hierarchy's wins. The data (AGGREGATE_EXTRAS, flat trained at 512 -> 0.539) supports the confound reading, but that control is n=1.
2. **LOOP_HIER_PARITY seed count.** Tables are n=12 (x/12 sign counts, MDE arithmetic); the Scope paragraph says n=16 and "no looped-hierarchical variant exists" (stale text from HIER_PARITY). Treated as n=12.
3. **Match-Query base TQ=256 on seeds 0-2.** MATCH_QUERY_RESULTS gives 0.888 +/- 0.140; MATCH_QUERY_LONGQ gives 0.904 +/- 0.098 for the same checkpoints (different eval sample). Not reconciled.
4. **Same arm, two batches, Match-Query 128^2.** LOOP_HEADROOM path-int 1-layer 0.456 / +loop 0.870 vs MQ_RANK_2X2 r=2 no loop 0.416 / loop 0.833, same recipe. Consistent with the documented cross-batch non-reproducibility of Match-Query; no file reconciles them.
5. **enwik8 metric definitions.** Final-checkpoint values (ENWIK8_LONG, ENWIK8_HIERARCHY, CLAUDE.md -0.0058 t=3.49) vs mean of last 5 checkpoints (ENWIK8_SEEDS -0.0052; ENWIK8_HIER +0.0010 pooling). ENWIK8_LONG does not state deterministic val and its "12k value" column is from the invalidated 12k run; whether its 36k values were produced before the val fix is not stated (JSON mtimes all 08-28 23:2x).
6. **FRONTIER_ALGORITHMIC vs LOOP_HEADROOM.** FRONTIER says LOOP_HEADROOM's interaction "does not survive having a depth baseline on the index row"; LOOP_HEADROOM's +0.315 was measured on a different task (Match-Query) and is detectable. CLAUDE.md resolves as "task-specific"; the report should say that, not "does not survive".
7. **Compositional Match-Query gates.** The `exact` never-moved baseline is up to 2.8x chance, but the writeup calls plain-model failure "airtight" without addressing it.
8. **HORIZON_TASK_DISTANCES** uses the retracted n=3 Match-Query ceiling (0.888) and a horizon (~2) later shown to be budget-limited; its refutation of the cross-task account likely survives but was not recomputed.

## Cross-line dependencies

- **Headline where/what claim (all lines):** C09/C10 Match-Query separation (0.730 n=5 vs 0.154; context destruction 0.918 -> 0.074) is one of the three supports cited in RESULTS_INDEX's headline.
- **Encoding vs position line:** C09 (MapPoPE-Hier 0.847 vs PoPE-Flat 0.117, n=3); C28 enwik8 MapPoPE vs RoPE.
- **Rank line:** C13 (r=4 + loop 0.986; rank effect appears only with the loop) is cited in RESULTS_INDEX next to RANK_SWEEP.
- **Sign / clock-vs-map line:** C18 parity (a signed cumsum mod 2pi is a parity register) is the non-navigation instance; prior art Sarrof/Grazzi/Selective RoPE sec. 4.2 applies.
- **Correction (InEKF) line:** C24 filter x loop 2x2 (filter's OOD-only effect, non-complementarity), C07 Level15 on compositional (likelihood 3/3, accuracy 1/3), C34 MiniGrid Memory. NOISE_REFINE.md and MQ_NOISE_2X2*.md contain loop-under-noise contrasts (+0.138, +0.121/+0.057) that belong to that line and were not read here.
- **Recipe/power (all torus lines):** C25 sets the lr 1e-3 default that later torus batches use; C04 shows the old compositional recipe undertrained. EM/WM, sign and rank batches depend on this.
- **Task validity (all lines):** C32/C33 audits and the gate records above decide which tasks the report may use.
- **EM vs WM line:** C02 MapEM-Flat compositional 0.097 (n=3) and C09 MapEM rows are referenced there (MATCH_QUERY_EM.md, EM_COMP_SAMEBATCH.md - not read here).
- **Environment / map-extent line:** C10's n_obs=4 aliasing boundary and 128^2 scaling.
- **Methods (GUARDS.md):** `stats_guard.paired` / `interaction` (MDE conventions used by C12, C13) and `ckpt_guard.compare_checkpoints` (determinism vs replication, relevant to C05's bit-identical Hourglass_k2 arm) are code implementations of the statistics used across this inventory.

## Files read

COMPOSITIONAL_EXPERIMENT.md, COMPOSITIONAL_RESULTS.md, COMPOSITIONAL_MULTISEED.md, COMPOSITIONAL_PLAIN_RESULTS.md, COMP_HEADROOM_PREREG.md, COMP_HEADROOM.md, COMP_HEADROOM.json (per-seed, for C05 check), HIER_RECHECK.md, HIER_RECHECK.json, HIER_PARITY.md, HIER_ATTN_LONGT.md, HIERGOAL_ABLATION.md, HIERGOAL_CLOSEDLOOP.md, HIERGOAL_LONGT.md, SPACETIME_HIER_RESULTS.md, ROUTE_ATTN_RESULTS.md, HOURGLASS_README.md, AGGREGATE_TASK_RESULTS.md, AGGREGATE_MULTISEED.md, AGGREGATE_EXTRAS.md, CORRECTION_COMPOSITIONAL.md, ENWIK8_HIERARCHY.md, ENWIK8_HIER.md, ENWIK8_2X2.md, ENWIK8_LONG.md, ENWIK8_SEEDS.md, enwik8_long/*.json (final values), hourglass_enwik8_long/{hourglass,flat10}.json, LOOPED_PILOT.md, LOOPED_L1.md, LOOPED_L4.md, LOOPED_Loop4.md, LOOP_HEADROOM.md, LOOP_HIER_COMPUTE.md, LOOP_HIER_PARITY.md, LOOP_SAMPLED.md, LOOP_SAMPLED.json, LOOP_DEPTH_STRATA.md, L15_LOOP_2X2.md, MQ_RANK_2X2.md, RECURSIVE_RESULTS.md, ALGORITHMIC_GATES.md, ALGORITHMIC_RESULTS.md, FRONTIER_ALGORITHMIC.md, HORIZON_RESULTS.md, HORIZON_TASK_DISTANCES.md, HORIZON_L1d128e16.md, HORIZON_L1d128e50.md, HORIZON_L2d128e16.md, HORIZON_L2d256e16.md, HORIZON_L4d128e16.md, HORIZON_L4d256e16.md, MATCH_QUERY_RESULTS.md, MATCH_QUERY_SCALE.md, MATCH_QUERY_LONGQ.md, MATCH_QUERY_GATES.md, MATCH_GATES_128_16.md, MATCH_GATES_64_16.md, MATCH_GATES_64_4.md, COMPOSITIONAL_MATCH_QUERY.md, COMPOSITIONAL_MATCH_QUERY_RESULTS.md, COMPOSITIONAL_MATCH_QUERY_GATES.md, COMPOSITIONAL_MATCH_QUERY_CURRIC.md, COMPOSITIONAL_MATCH_QUERY_STAB.md, MAP_QUERY_GATES.md, MAP_QUERY_RESULTS.md, BOUNDED_MEMORY.md, BOUNDED_MEMORY_RESULTS.md, MINIGRID_MEMORY_RESULTS.md, LAP_GATES.md, LAP_TRANSFER.md, LAP_TRANSFER_NOREWARD.md, CSCG_TASK_GATES.md, PLANNER_TASK_AUDIT.md, ROOMS_GOAL_RESULTS.md, DAGGER_RESULTS.md, DAGGER_DK6_RESULTS.md, DAGGER_EMPTY_RESULTS.md, DOORKEY_BC_RESULTS.md, RECIPE_POWER.md, _RECIPE_C0.md, _RECIPE_C1.md, _RECIPE_C2.md, GUARDS.md.
Exclusion/context files: report/INVENTORY_BRIEF.md, RESULTS_INDEX.md, N3_AUDIT.md, KNOWN_BUGS.md, archive/void/README.md (and directory listing), CLAUDE.md (in context), ABLATE_COMPOSITIONAL.md, DISSOCIATION_SWEEP.md, BASELINE_TABLE.md (secs. D-E).
Scripts checked for recipe/seed counts only: run_loop_sampled.sh, run_hier_recheck.sh, run_algorithmic.sh, run_frontier.sh, run_hier_parity.sh, run_loop_hier.sh, run_loop_headroom.sh, run_l15_loop_2x2.sh, run_recipe_power.sh, run_comp_headroom.sh, run_match_scale.sh, run_match_query.sh, run_horizon.sh, run_looped_pilot.sh, run_dissociation.sh, model_hourglass.py (CoarseIdx/CoarsePI/FR docstrings), train_hourglass_enwik8.py (val generator).

## Files in scope not covered

None missing from the in-scope list. Referenced but not read (belong to other lines): ALIASING_COVARIATE.md, MATCH_QUERY_EM.md, EM_COMP_SAMEBATCH.md, NOISE_REFINE.md, MQ_NOISE_2X2.md, MQ_NOISE_2X2_C2.md, LEVEL15_MEETS_GATED_matchq.md, REVISIT_DISTANCE.md, PAPER_VALIDATION.md, sweep_dissociation.py (prereg text of C06), validate_compositional.py (gate internals).
