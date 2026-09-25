# Results index (regenerated 2026-09-24; rank, documents and statistics updated 2026-09-25)

A catalogue with statuses, not a narrative. **`CLAUDE.md` is the authority** for conventions, the
standing rules (numbered 1-28 there; this file keeps no rule list of its own), the withdrawn list and
the invariants. Live state is `.claude-memory/project_state.md`; history is `docs/LOG.md`. A CORRECTED
or AUDIT block at the top of a results file supersedes its body.

**Status** is from the 2026-09-24 experiment audit (a 58-claim ledger), checked against CLAUDE.md;
every number below was re-read from the file named beside it.
SOLID = holds as stated at its scope; NEEDS CONTROL = real measurement, but the claim outruns it
(usually: read only past the training length or depth, or at a superseded recipe); UNDERPOWERED =
inside a t-based MDE or n <= 4. "OOD" = read past the training length or depth, i.e. robustness until a
matched control exists (CLAUDE.md rule 10). The house "DETECTABLE" (|mean| > 2.8 sd/sqrt(n), i.e. |t| > 2.8)
is a 10.7% false-positive test at n=3 (2.7% at n=8), so an n < 8 DETECTABLE is read with its t-test p
(CLAUDE.md rule 5; e.g. code C1 MapPoPE - PoPE +0.0033 has p ~0.1).

Documents: `positional_review.pdf` (review), `axes_measured.pdf` (results paper), `mapformer_math.pdf`
(record), `report/report.pdf` and `report/report_short.pdf`; all brought into line with this index on
2026-09-25.

## Citable results

| claim | key number | n | floor / chance | status | file |
|---|---|---|---|---|---|
| Path integration helps on the paper's torus task, at training length | position **+0.243** (MDE 0.038, 8/8); index RoPE 0.805, path 0.971; +0.359 at 8x (OOD). The often-quoted +0.461 is the 16-epoch recipe (index arm on the floor) | 8 | blank floor 0.506 | SOLID at this budget; the index arm was still slowly descending, so its ceiling is unmeasured | `PAPER2X2_RESULTS.md` |
| ...and is necessary for in-context maps (Match-Query) | 0.730 +/- 0.247 vs index 0.154; context destruction 0.918 -> 0.074 | 5 | chance 0.0625 | SOLID; its index control is PlainFlat, never the architecture-matched RoPE arm | `MATCH_QUERY_SCALE.md`, `MATCH_QUERY_RESULTS.md` |
| Shared r=4 finds the torus solution where r=2 does not (trained and tested at T=1024, 900 ep, matched initialisation; why -- per-head rank, cross-head sharing or W_out scale -- is unseparated) | SOLVED within 900 ep: our shared r=2 **0/8**, a per-head r=2 **2/8**, shared r=4 **8/8** (Fisher p 0.0002 / 0.007); accuracy 0.894 / 0.885 / 0.998. A rank-2 projection of each solved r=4 scores 0.9955 frozen and is held under training (7/8 vs the r=4 control's 7/8): a SEARCH deficit, in the paper's own design | 8 | constant floor 0.506 (wrap-only 0.507) | SOLID, budget-scoped (within 900 epochs); r=4's smaller `W_out` init (bound 0.5 vs 0.707) is unseparated; one task, one width; MapWM family only | `RANK_MI_RESULTS.md`, `RANK_PROJ_RESULTS.md`, `RANK_MATCHED_RESULTS.md` |
| Dyck-2 depth ladder at the training cell (L32 D4), width fixed | position +0.293 / +0.081 / +0.048 / +0.019 at 1-4 layers (8/8); index RoPE 0.979 at 4L | 8 | chance 0.5 | SOLID (matched length and depth) | `DYCK_LADDER_RESULTS.md` |
| Index code cannot count contextually (reproduces CoPE) | +0.750 at T=1024 (8/8) | 8 | chance 0.0625 | SOLID (matched length) | `RECENCY_RESULTS.md` |
| Loop on path integration (Match-Query) | loop main effect **unpaired** +0.346 (t 3.75); loop arm pooled 0.803 +/- 0.200, 1/16 failures | 8 / 16 | chance 0.0625 | SOLID for the main effect. The paired interaction +0.315 and "r=4 + loop x4 0.986, 8/8 >= 0.941" are paired / one-batch statistics on a task whose same-seed retrains drift 0.185: CONTRADICTED (see Withdrawn) | `REFINE_RESULTS.md`, `LOOP_HEADROOM.md`, `MQ_RANK_2X2.md` |
| EM's recency deficit is search | EM - WM -0.375 (0/8, MDE 0.154); installed rewind frozen 1.000 (8/8); per-pair origins +0.215 = pathway +0.124 + freedom +0.091 (n=48) | 8 / 48 | chance 0.0625 | SOLID as a fixed-budget learnability result | `RECENCY_EM_RESULTS.md`, `WARM_RESULTS.md`, `SEARCH_RESULTS.md`, `PAIRSPLIT_RESULTS.md`; `EM_WM_STATE.md` Sec 3-4 gives the status of every EM/WM file |
| Phase freedom in q0/k0 | +0.146 vs a matched-optimiser control (22/24); fresh seeds +0.113 | 24 | -- | SOLID; mechanism unidentified | `MAGONLY_RESULTS.md`, `D5_RESULTS.md` |
| Paper replications | MapFormer v4 Dyck-2: ordering MapWM-1L - RoPE-2L +0.370 (8/8, F1; on Hewitt closing accuracy +0.064, 0.638 vs 0.574 at L128 D12, same direction); levels do not replicate and sit at the F1 floor 0.884. PoPE's Indirect Indexing at 200k iters 7/8 (1/8 at their 100k). PoPE's Bach: PoPE - RoPE -0.032 NLL (5/5) | 8 / 8 / 5 | F1 no-stack 0.884 | SOLID | `DYCK_RESULTS_bs128.md`, `INDIRECT_RESULTS_200k.md`, `JSB_RESULTS.md` |
| Our PoPE is faithful; non-negativity is not what extrapolates | 1.7e-06 max logit difference vs the authors' code; NoSigma penalty -0.0108 vs RoPE +3.5885; 80.7% of `pope_delta` frozen in its clamp | 3 | batch floor 0.0021 / 0.0028 bpc | SOLID as corrected (the NoSigma Table-5 cell is unmeasured) | `ABLATE_RESULTS.md` |
| PoPE's encoding helps the path row (Bach) | MapPoPE - MapWM -0.0165 NLL (5/5, MDE 0.0111) | 5 | -- | SOLID on Bach. Dyck 2L +0.050 is depth-OOD (~0 at L32 D4); code -0.0052 and MapPoPE - PoPE +0.0033 are UNDERPOWERED | `JSB_RESULTS.md`, `.claude-memory/project_mappope_asymmetry.md` |
| Code: the OOD encoding "win" is extrapolation cost | at matched 2048 the encoding effect is -0.0030 (MDE 0.0046) against -3.694 extrapolating from 512 | 3 | no-memory brackets 0.858 | SOLID as a retraction | `CODE_RESULTS.md`, `CODE_GATES.md` |
| Recipe beats architecture on the compositional task | warmup + cosine +0.160 (7/8) | 8 | floor 0.072 | SOLID (hierarchy's +0.136 on the same task is UNDERPOWERED) | `COMP_HEADROOM.md`, `HIER_RECHECK.md` |
| Parallel scan | 2.6-3.3x over a 16x length increase; MapEM-NC 14.5x; TEMFaithful 120x | -- | -- | SOLID | `TIMING_BENCHMARK.md` |
| CSCG stitching control reproduces | paired +0.131 +/- 0.024 vs index -0.005 | 3 | exactly 0 | SOLID, n=3 | `STITCH_ATTENTION.md` |
| Metric and data findings | Dyck F1 has a 0.88 no-stack floor (n-gram 0.857 at the hardest cell; use Hewitt closing accuracy); Bach is overfitting-limited: transposition 0.107 NLL vs the 0.032 PoPE-RoPE gap | -- / 5 | -- | SOLID | `DYCK_LITERATURE_METRICS.md`, `AUG_RESULTS.md` |
| Robustness repairs (OOD, labelled as such) | 48-parameter decay envelope: Bach MapPoPE 4.616 -> 0.622, PoPE 1.597 -> 0.626 NLL at 2-4x (5/5); MapWM - RoPE -0.662 at 2-4x beyond a 512 context | 5 | -- | SOLID as robustness | `DECAY_RESULTS.md`, `JSB_LENGTH_RESULTS.md` |
| Hierarchy on text is efficiency only | 1.4537 vs 1.4506 bpc at parameter parity; 1.23x throughput, -14.1% peak memory | 1 | checkpoint sd 0.003-0.007 | a null at n=1 (consistent with, not proof of) | `ENWIK8_HIERARCHY.md` |

## Listed as citable in CLAUDE.md, but NEEDS CONTROL

| claim | key number | what is missing | file |
|---|---|---|---|
| Dyck ladder at L32 D12 | +0.290 / +0.209 / +0.159 / +0.168 at 1-4L (8/8); index plateaus 0.76-0.78 | matched LENGTH but **3x the training depth**: a matched-depth control, and an index-arm budget extension ("stop climbing" rests on a slope rule read after LR decay). The results file now carries a correction banner (2026-09-24) | `DYCK_LADDER_RESULTS.md` |
| Sign of the increment (a replication of Sarrof / Grazzi / Selective RoPE in navigation) | signed beats index +0.123 / +0.195 at T=512/1024 (12/12); monotone does not; opposition 0.11 vs 1.85-1.98 | the accuracy cost is extrapolation-only (at T=128 monotone 0.90-0.98 vs index 0.80; loss 12/12 worse). Needs a monotone arm trained and tested at T=1024 | `SIGN_ABLATION.md`, `SIGN_PROBE.md` |
| Clock/map crossover | monotone costs -0.280 on the torus, -0.004 on recency; magnitude-matched content increment +0.594 (8/8) | both halves are extrapolation readouts at different ratios | `RECENCY_RESULTS.md`, `RECENCY_GATE_ABLATION.md` |
| Map extent is a threshold | -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells, matched aliasing | the index arms behind +0.305 and +0.015 were still DESCENDING under a flat-slope "converged" label | `ALIASING_CONTROLLED.md`, `VISITS_TEST.md` |
| Rotation actions; allocentric recoding fixes it | +0.050 -> +0.488 (8/8); 12 headings +0.26..+0.38 (bimodal budget curve) | all at the 16-epoch recipe with the index arm on the 0.508 floor; rerun under the converged recipe | `KNOB_SWEEP_n8.md`, `ALLOCENTRIC_RECODING.md` (n=3), `H12_BUDGET_CURVE.md` |
| Decay envelope vs long-range retrieval | steepness +0.136 and metric +0.145 of +0.281 (8/8) | residual collinear with convergence (r = -0.995, non-overlapping losses) | `CROSS_RESULTS.md` |

## Open: underpowered, OOD-only or unfinished

- **Rank, the old numbers.** r=4 +0.085 at T=1024 when trained at T=128 is OOD only (94% from
  short-gap revisits late in the sequence; `RANK_SWEEP.md`); on Bach the order inverts (r=1 best,
  `JSB_LENGTH_RESULTS_RANK.md`). The matched-length result is now citable (table above). The
  registered matched-length verdicts of `RANK_MATCHED_RESULTS.md` (900 ep, and 900 + 900 with a warm
  restart) are UNREADABLE (four r=2 runs still descending); `RANK_MI_RESULTS.md` is the readable test.
  "r=2 loses because its basis is skewed" is withdrawn: within r=2 skew does not predict accuracy, and a
  rank-2 solution exists. Our bottleneck is shared across heads, the paper's is per head (2 heads: our
  r=2 has 2 latent dims, the paper's 4).
- **Level 1.5 / InEKF**: loss-matched +0.062 / +0.124 at T=512/1024, OOD only; no matched-length arm
  (`L15_ABLATION.md`).
- **Forget gate** +0.086 at r=2, T=1024, OOD only, mechanism unidentified; the forget-clock batch was
  deleted and, as registered, reads OOD lengths only (`FORGET_GATE.md`, `FORGET_CONTROL.md`,
  `FORGET_CLOCK_PREREG.md`).
- **PoPE wrapping**: the length half holds 3/3, OOD only (`POPE_WRAPPING.md`).
- **EM_P0 - WM on the paper task** +0.035 / +0.070 / +0.085 at l=512/1024/2048, OOD only
  (`EM_WM_THEORY.md`).
- **"The loop's torus gain is convergence, not representation"** is a loss-matched residual at
  matched length, the inference withdrawn for rank; the LoopedSampled count-curve flattening itself is
  solid (`L15_LOOP_2X2.md`, `LOOP_SAMPLED.md`).
- **Code C1 "reversal"** (position +0.0055, MapPoPE - PoPE +0.0033) and the C2 envelope contrasts:
  n=3, paired t-test p 0.04-0.10, read on `best_val_bpc` (min over 36 evals of 40 windows). Rescore the
  `.final.pt` checkpoints on the full val file before adding seeds. "RoPE + envelope is the best of
  eight" is p 0.35 and cross-batch (`CODE_RESULTS.md`, `CODE_DECAY_RESULTS.md`).
- **MiniGrid 2x2x2**: index arms best (0.955 / 0.953), MapWM last (0.823); convergence unreported
  (`MINIGRID_FULL_2X2X2.md`).
- **Indirect Indexing**: path integration faster (6/7, directional) and more padding-robust (0.430 vs
  0.149, uncontrolled) (`INDIRECT_RESULTS_200k.md`, `INDIRECT_OOD.md`).
- **Family tree** non-commutativity +0.013 (MDE 0.008 z-based; fails the t-based MDE at n=3)
  (`N3_AUDIT.md`, `FAMILY_TREE_RESULTS.md`).
- **COUNTER** (n=4): with an identical installed counter MapWM 1.000, EM 0.740, TEM 0.337
  (`COUNTER_RESULTS.md`).
- **Addition**: signed MapFormer is the only learned code to learn 30 digits (3/3), holds to ~33-38;
  the control gate FAILED (role-format oracle 0.697) (`SAMEBLOCK_RESULTS.md`).
- **Map-Query**: the room query is learnable at 7.6x chance on one variant and one seed; the
  multi-seed table is 25-epoch undertrained (`MAP_QUERY_GATES.md`, `MAP_QUERY_RESULTS.md`).
- **lm200, corrected**: Level15 0.990 vs Vanilla 0.742 stands as numbers; the interpretation is
  withdrawn (a filter-free capacity control ties it), and lm200 never had a context-destruction
  ablation (`LM200_CORRECTED_MULTISEED.md`, `EXTRAHEAD_CONTROL.md`).

## Live negatives and withdrawn claims

**Live negatives** (do not re-run): CLAUDE.md, "Live negatives". **Withdrawn** (do not cite):
CLAUDE.md, "Withdrawn -- do not cite", plus `archive/void/` (54 files, each bannered) and
`archive_stale/` (35 files).

Contradicted by the 2026-09-24 audit and **not yet on CLAUDE.md's withdrawn list**:
- Loop x path integration "super-additive" (paired interaction +0.315), "`r=4 + loop x4` 0.986,
  8/8 >= 0.941" and "the loop raises the floor, 8/8 >= 0.77": paired or one-batch statistics on a task
  whose same-seed retrains drift 0.185 per seed (`REFINE_RESULTS.md`). CLAUDE.md's loop row now marks
  the interaction and "never fails" withdrawn and the 0.986 as one batch.
- "The encoding moves the torus result ~0.003 / PoPE is inert without path integration (0.509)":
  16-epoch recipe. Converged, the encoding main effect is detectable at every length (-0.049 / +0.114
  / +0.189, `PAPER2X2_RESULTS.md`).
- "Index models exceed the floor only at recurrence interval 1-2" (`REVISIT_DISTANCE.md`): under the
  converged recipe index RoPE leads index PoPE at 5-16-step revisits (`REVISIT_2X2.md`).
- The shared report's "on text, code and music, plain PoPE still wins" (its own code row and
  `ABLATE_RESULTS.md` say otherwise) and its unsourced "six cases" sentence.
- Indirect Indexing "0.965 against the paper's 0.948" as like for like: 0.965 is the mean among
  solvers; the paper's figure is over all runs.

## Two readout notes carried over from the previous index (not rules)

- **`shuffle` and `resample` are not interchangeable** in a context-destruction ablation: permuting
  slots destroys the walk's autocorrelation and puts the input off-manifold; substituting the stream
  from an independent episode does not. Report both (`ABLATE_COMPOSITIONAL.md`).
- **An ablation landing BELOW the floor** means the model fails confidently rather than hedging. Check
  with an on-manifold resample before blaming the manipulation.

## Catalogue of results files, by line

Top-level `*.md` only (444 files). `*` = a CORRECTED / RETRACTED / WITHDRAWN / SUPERSEDED / VOID /
STALE marker in the first 12 lines (a correction block further down also supersedes the body).
Files starting with `_` are raw per-seed dumps; names ending `_PREREG` are pre-registrations and
`_GATES` task gates.

**Torus paper task, recipe and reproduction** (54)

`ALLOCENTRIC_RECODING`, `AUDIT_HEADLINE`, `BASELINE_TABLE`, `CLOCK_SCAN`, `DETAILED_RESULTS`, `DRIFT_PROBE`, `FREQ_CONTROL`, `GENERALIZATION_REPORT`, `H12_BUDGET_CURVE`, `HORIZON_L1d128e16`, `HORIZON_L1d128e50`, `HORIZON_L2d128e16`, `HORIZON_L2d256e16`, `HORIZON_L4d128e16`, `HORIZON_L4d256e16`, `HORIZON_RESULTS`, `HORIZON_TASK_DISTANCES`, `INDEX_BASELINE_PAPER_TASK`, `INDEX_BASELINE_PAPER_TASK_n8`, `KNOB_SWEEP`, `KNOB_SWEEP_n8`, `LONG_SEQ_clean`, `N3_AUDIT`, `NOISE_CLEAN_REVALIDATION`, `OMEGA_RESCALE_clean`, `OOD_GRID_RESULTS`, `PAPER2X2_PREREG`, `PAPER2X2_RESULTS`, `PAPERTASK_PREREG`, `PAPERTASK_RESULTS`, `PAPER_OOD_EXTENDED`, `PAPER_OOD_EXTENDED_n8`, `PAPER_OOD_PROTOCOL`, `PAPER_OOD_RERUN`, `PAPER_OOD_WITH_POPE`, `PAPER_TASK_ABLATION`, `PAPER_TASK_ACCURACY`*, `PAPER_TASK_FLOORS`, `PAPER_VALIDATION`, `PERSCALE_OMEGA_RESULTS`, `PER_VISIT_clean`, `RECIPE_POWER`, `REVISIT_2X2`, `REVISIT_DISTANCE`, `ROPE_CANONICAL`, `ROPE_CONVERGE`, `TIMING_BENCHMARK`, `TOPOLOGY_RESULTS`, `ZERO_SHOT_TRANSFER_clean`, `ZERO_SHOT_TRANSFER_clean_brokeninit`, `_PAPER2X2_RAW`, `_RECIPE_C0`, `_RECIPE_C1`, `_RECIPE_C2`

**Rank, generator and accumulator** (44)

`ACCUMULATOR`, `ACTION_GEOMETRY`, `CONV_KERNEL_PROBE`, `DXR_PRELIM`, `DXR_RANK_THRESHOLD`, `FAST_ATTN_RANK`, `GATE_PROBE`, `LEARNED_RANK`, `LOCALISATION`, `LOCALISATION_PREREG`, `LOCALISATION_RANK`, `MAPPOPE_R4`, `MAPPOPE_R4_PREREG`, `MAPPOPE_R4_RESULTS`, `ND_GATES`, `PAPER_FIG4_EM`, `PAPER_FIG4_REPRO`, `RANK_MATCHED`, `RANK_MATCHED_GEOMETRY`, `RANK_MATCHED_PREREG`, `RANK_MATCHED_RESULTS`, `RANK_MATCHED_e900`, `RANK_MATCHED_e900_GEOMETRY`, `RANK_MATCHED_e900c`, `RANK_MATCHED_e900c_GEOMETRY`, `RANK_MI`, `RANK_MI_GEOMETRY`, `RANK_MI_PREREG`, `RANK_MI_RESULTS`, `RANK_PERHEAD_PILOT`, `RANK_PERHEAD_PILOT_GEOMETRY`, `RANK_PERHEAD_PILOT_RESULTS`, `RANK_PERHEAD_PREREG`, `RANK_PROJ_FROZEN`, `RANK_PROJ_PREREG`, `RANK_PROJ_RESULTS`, `RANK_PROJ_TRAIN`, `RANK_PROJ_TRAIN_GEOMETRY`, `RANK_SWEEP`, `RANK_TRUNCATION`, `SELECTIVE_ROPE`, `THEORY_NARRATIVE`*, `THEORY_NUMBERS`, `THEORY_SEARCH_AND_LENGTH`, `_SELECTIVE_TORUS`

**Sign, clock/map and recency** (32)

`COUNTER_BATCH`, `COUNTER_RESULTS`, `FLIPFLOP_GATES`, `FLIPFLOP_RESULTS`, `FORGET_CLOCK_PREREG`, `FORGET_CONTROL`, `FORGET_GATE`, `GATED_PREREG`, `GATED_RESULTS`, `GATED_SEPARATION`, `GATED_TORUS`, `LAMBDA_TRACE`, `MONOTONE_PREREG`, `MONOTONE_RAW`, `MONOTONE_RESULTS`, `MQAR_PREREG`, `MQAR_RESULTS`, `RECENCY_GATES`, `RECENCY_GATES_K16SET`, `RECENCY_GATES_K4SET`, `RECENCY_GATES_K64`, `RECENCY_GATE_ABLATION`, `RECENCY_H2`, `RECENCY_PREREG`, `RECENCY_RESULTS`, `SIGN_ABLATION`, `SIGN_ABLATION_PREREG`, `SIGN_PROBE`, `TEM_RECENCY_DIAG`, `TEM_RECENCY_PILOT`, `_MONOTONE_TORUS`, `_SIGN_RAW`

**EM vs WM and the position kernel** (49)

`AP_KERNEL_DIAGNOSTIC`, `AUDIT_2026-09-10`, `D5_PREREG`, `D5_RESULTS`, `DOF_PREREG`, `DOF_RESULTS`, `EM_COMP_SAMEBATCH`, `EM_FIX_COMP`, `EM_HOPFIELD_CROSSSCALE`, `EM_P0_COMP`, `EM_P0_PAPER`, `EM_WM_STATE`*, `EM_WM_THEORY`, `HOPFIELD_NOMAINAP_RESULTS`, `MAGONLY_PREREG`, `MAGONLY_RESULTS`, `MATCH_QUERY_EM`, `MINIGRID_EM`, `MINIGRID_EM_FIX`, `MINIGRID_EM_FIX_PREREG`, `MINIGRID_EM_PREREG`, `N5_PREREG`, `N5_RESULTS`, `NOLEAK_PREREG`, `NOLEAK_RESULTS`, `PAIRCONST_PREREG`, `PAIRCONST_RESULTS`, `PAIRORIGIN_PREREG`, `PAIRORIGIN_RESULTS`, `PAIRSPLIT_PREREG`, `PAIRSPLIT_RESULTS`, `RECENCY_EM_RESULTS`, `REC_EM_PREREG`, `SEARCH_PREREG`, `SEARCH_RESULTS`, `SPREAD2_PREREG`, `SPREAD2_RESULTS`, `SPREAD_PREREG`, `SPREAD_RESULTS`, `TALE_OF_TWO_ALGORITHMS`, `THEORY_KERNEL`, `UNFREEZE_PREREG`, `UNFREEZE_RESULTS`*, `VOCAB_EM`, `VOCAB_EM_PREREG`, `WARM_PREREG`, `WARM_RESULTS`*, `_DOF_TORUS_RAW`, `_N5_TORUS_RAW`

**Match-Query, loop and algorithmic tasks** (38)

`ADDITION_CHO_REPRO`, `ADDITION_DESIGN`, `ADDITION_GATES`, `ADDITION_PILOT`, `ADDITION_PILOT2`, `ALGORITHMIC_GATES`, `ALGORITHMIC_RESULTS`, `FRONTIER_ALGORITHMIC`, `HIER_PARITY`*, `L15_LOOP_2X2`, `LOOPED_L1`, `LOOPED_L4`, `LOOPED_Loop4`, `LOOPED_PILOT`, `LOOP_DEPTH_STRATA`, `LOOP_HEADROOM`, `LOOP_HIER_COMPUTE`, `LOOP_HIER_PARITY`, `LOOP_SAMPLED`, `MATCH_GATES_128_16`, `MATCH_GATES_64_16`, `MATCH_GATES_64_4`, `MATCH_QUERY_GATES`*, `MATCH_QUERY_GATES_P010`, `MATCH_QUERY_LONGQ`, `MATCH_QUERY_NOISE_ABLATION`, `MATCH_QUERY_RESULTS`, `MATCH_QUERY_SCALE`, `MQ_NOISE_2X2`, `MQ_NOISE_2X2_C2`, `MQ_RANK_2X2`, `RECURSIVE_RESULTS`, `REFINE_RESULTS`, `SAMEBLOCK_COMPILE_CHECK`, `SAMEBLOCK_PREREG`, `SAMEBLOCK_RAW`, `SAMEBLOCK_RESULTS`, `_L15_LOOP_RAW`

**Hierarchy, compositional and planner tasks** (35)

`ABLATE_COMPOSITIONAL`, `AGGREGATE_EXTRAS`, `AGGREGATE_MULTISEED`, `AGGREGATE_TASK_RESULTS`, `BOUNDED_MEMORY`, `BOUNDED_MEMORY_RESULTS`, `COMPOSITIONAL_EXPERIMENT`, `COMPOSITIONAL_MATCH_QUERY`, `COMPOSITIONAL_MATCH_QUERY_CURRIC`, `COMPOSITIONAL_MATCH_QUERY_GATES`, `COMPOSITIONAL_MATCH_QUERY_RESULTS`, `COMPOSITIONAL_MATCH_QUERY_STAB`, `COMPOSITIONAL_MULTISEED`, `COMPOSITIONAL_PLAIN_RESULTS`, `COMPOSITIONAL_RESULTS`*, `COMP_HEADROOM`, `COMP_HEADROOM_PREREG`, `CORRECTION_COMPOSITIONAL`, `CSCG_TASK_GATES`, `DISSOCIATION_SWEEP`, `HIERGOAL_ABLATION`, `HIERGOAL_CLOSEDLOOP`, `HIERGOAL_LONGT`, `HIER_ATTN_LONGT`, `HIER_RECHECK`, `LAP_GATES`, `LAP_TRANSFER`, `LAP_TRANSFER_NOREWARD`, `MAP_QUERY_GATES`, `MAP_QUERY_RESULTS`, `PLANNER_TASK_AUDIT`*, `ROOMS_GOAL_RESULTS`, `ROUTE_ATTN_RESULTS`, `SPACETIME_HIER_RESULTS`, `STITCH_ATTENTION`

**Family tree** (6)

`ABLATE_FAMILY_TREE`, `FAMILY_TREE_D7_GATES`, `FAMILY_TREE_D7_RESULTS`, `FAMILY_TREE_GATES`*, `FAMILY_TREE_RESULTS`, `FAMILY_TREE_WM_GAP`

**MiniGrid, MiniWorld, Habitat** (54)

`ALIASING_CONTROLLED`, `ALIASING_COVARIATE`, `ALIASING_GATES`, `CONTINUOUS_ALLOC`, `CROSSOVER_CONVERGED`, `DAGGER_DK6_RESULTS`, `DAGGER_EMPTY_RESULTS`, `DAGGER_RESULTS`, `DOORKEY_BC_RESULTS`, `HABITAT_BUILD`, `MINIGRID_2X2`, `MINIGRID_2X2X2`, `MINIGRID_2X2X2_n8`, `MINIGRID_ALLOCENTRIC_2X2X2`, `MINIGRID_ALLOCENTRIC_8CELL`, `MINIGRID_ALLO_8THCELL`, `MINIGRID_DK16_RESULTS`, `MINIGRID_DOORKEY_CACHED`, `MINIGRID_DOORKEY_LONGT`, `MINIGRID_DOORKEY_RESULTS`, `MINIGRID_DOORKEY_ROPE_DIAG`, `MINIGRID_FULL_2X2X2`, `MINIGRID_MEMORY_RESULTS`, `MINIGRID_REPRO_CONTROL`, `MINIWORLD_ENDPOINTS`, `MINIWORLD_FIXED_FINDINGS`, `MINIWORLD_FIXED_RESULTS`, `MINIWORLD_FIXED_RESULTS_T1024`, `MINIWORLD_FRESH_ABLATION`, `MINIWORLD_FRESH_FINDINGS`, `MINIWORLD_FRESH_GATES_ALLO`, `MINIWORLD_FRESH_GATES_RAW`, `MINIWORLD_FRESH_RESULTS`, `MINIWORLD_FRESH_RESULTS_T1024`, `MINIWORLD_GATES_ALLO`, `MINIWORLD_GATES_FIXED`, `MINIWORLD_GATE_CONTROL`, `MINIWORLD_GRID16_GATES`, `MINIWORLD_GRID24_GATES`, `MINIWORLD_GRID32_GATES`, `MINIWORLD_GRID_SWEEP`, `MINIWORLD_GRID_SWEEP_HIER`, `MINIWORLD_HIER_ABLATION`, `MINIWORLD_ORACLE_ABLATION`, `MINIWORLD_ORACLE_GATES`, `MINIWORLD_ORACLE_RESULTS_T1024`, `MINIWORLD_ORACLE_RESULTS_T512`, `MINIWORLD_PROBE3`, `MINIWORLD_RESULTS`, `MINIWORLD_SROPE_COMPONENTS`, `MINIWORLD_TODO`, `PERCEPTION_EXPERIMENT_PLAN`, `POSITION_EFFECT_CONVERGED`, `VISITS_TEST`

**Dyck-2** (16)

`DYCK_DECAY_PREREG`, `DYCK_DECAY_PROBE`, `DYCK_DECAY_RESULTS`, `DYCK_DEPTH_PREREG`, `DYCK_DEPTH_RESULTS`, `DYCK_FAR_PROBE`, `DYCK_GATES`, `DYCK_LADDER_PREREG`, `DYCK_LADDER_RESULTS`, `DYCK_LITERATURE_METRICS`, `DYCK_PREREG`, `DYCK_RESULTS_POPE`, `DYCK_RESULTS_bs128`, `DYCK_SHAREABLE`, `DYCK_STACK_PROBE`, `DYCK_T3_RESULTS`

**Indirect Indexing** (5)

`INDIRECT_ARITHMETIC`, `INDIRECT_OOD`, `INDIRECT_PREREG`, `INDIRECT_RESULTS`*, `INDIRECT_RESULTS_200k`

**Bach chorales, decay envelope and the MapPoPE collapse** (27)

`AUG_PREREG`, `AUG_RESULTS`, `CROSS_PREREG`, `CROSS_RESULTS`, `DECAY_PREREG`, `DECAY_RESULTS`, `JSBLEN_PREREG`, `JSB_LENGTH_RESULTS`, `JSB_LENGTH_RESULTS_BASE`, `JSB_LENGTH_RESULTS_RANK`, `JSB_PREREG`, `JSB_RESULTS`, `MAESTRO_PLAN`, `MAPPOPE_VS_POPE`, `POPE_WRAPPING`, `RECENCY_T3_PREREG`, `RECENCY_T3_RESULTS`*, `T1_PREREG`, `T1_RESULTS`, `T2_PREREG`, `T2_RESULTS`, `T3GEN_PREREG`, `T3GEN_RESULTS`*, `T3_PREREG`, `T3_RESULTS`, `THEORY_MAPPOPE`, `TORUS_T3_RESULTS`

**PoPE ablation, code and enwik8** (15)

`ABLATE_PREREG`, `ABLATE_RESULTS`*, `BF16_RESULTS`, `CODE_DECAY_RESULTS`, `CODE_GATES`, `CODE_PREREG`, `CODE_RESULTS`*, `CODE_RESULTS_OOD`, `ENWIK8_2X2`, `ENWIK8_COMPOSITION_PREREG`, `ENWIK8_HIER`, `ENWIK8_HIERARCHY`, `ENWIK8_LONG`, `ENWIK8_SEEDS`, `LANGUAGE_LANDSCAPE`

**Level 1.5 / InEKF / PC / TEM / grid cells (April-August lines)** (56)

`BUMP_TOKEN_RESULTS`, `CAPACITY_CONTROL`*, `CAPACITY_PERREGIME`*, `CASCADE_MULTISEED_RESULTS`, `CASCADE_REPRO_TEST`, `CASCADE_ZEROSHOT_S0`, `CLONE_ANALYSIS_LEVEL15PC`, `CLONE_TRANSFER_NOBYPASS`, `CNAV_HEX_Level15`, `CNAV_HEX_Level15EM`, `CNAV_HEX_Vanilla`, `CNAV_HEX_VanillaEM`, `CNAV_RESULTS`, `CORRECTED_LM200_LEADERBOARD`, `DOG_RESULTS`, `EXTRAHEAD_CONTROL`, `GSF_FULL_RESULTS`*, `HIPPOCAMPAL_ANALYSIS`, `HIPPOCAMPAL_GRID`, `HIPPOCAMPAL_GRIDL15PC`, `HIPPOCAMPAL_GRID_FREE`, `HIPPOCAMPAL_HIDDEN`, `HIPPOCAMPAL_HIDDEN_GRIDFREE`, `HIPPOCAMPAL_LEVEL15PC`, `L15_ABLATION`, `LEVEL15BETA_RESULTS`*, `LEVEL15EM_CROSSSCALE`, `LEVEL15_MEETS_GATED_matchq`, `LEVEL15_MEETS_GATED_paper`, `LEVEL15_MEETS_GATED_paper50`, `LM200_ABLATION`, `LM200_CORRECTED_MULTISEED`*, `MULTICLASS_MULTISEED_RESULTS`, `MULTICLASS_RESULTS`, `MULTISEED_FOLLOWUP`, `MULTISEED_FOLLOWUP_RESULTS`, `NOBYPASS_RESULTS`*, `NODROP_PARETO_RESULTS`*, `NOISE_REFINE`, `NUMBERLINE_RESULTS`, `RESULTS_PAPER`*, `R_T_DISTRIBUTION_3WAY`, `SESSION_HIERARCHICAL_CASCADE`, `TEM_BACKGROUND_BASELINES`, `TEM_CROSSSCALE_DIAGNOSTIC`, `TEM_NOISE_FFN_RESULTS`*, `TEM_RESULTS`*, `TEM_T_MULTISEED`*, `TEM_T_RESULTS`*, `V3_RESULTS`*, `V4_CONTROL_RESULTS`*, `V4_MULTISEED`*, `V4_RESULTS`*, `VECTOR_NAV_V2_RESULTS`, `VOCAB_SWEEP_MULTISEED`, `VOCAB_SWEEP_RESULTS`*

**Project meta, reports and infrastructure** (12)

`CLAUDE`, `GUARDS`, `HOURGLASS_README`, `KNOWN_BUGS`, `PUBLICATION_VENUES`, `README`, `REPORT`, `REPORT_ADDENDUM`*, `REPORT_v2`*, `RESULTS_INDEX`, `RESULTS_SUMMARY_2026-05-10`*, `SESSION_2026-05-01`

