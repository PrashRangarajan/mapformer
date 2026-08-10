# Results index — authoritative status (audited 2026-08-09)

Generated from the status banners in the files themselves, so it cannot drift
from them. Re-run the audit rather than hand-editing this file.

## What is safe to cite

| area | status | source |
|---|---|---|
| Paper replication, WM + EM, paper's own OOD protocol | **VERIFIED** | `PAPER_OOD_PROTOCOL.md`, `PAPER_TASK_ACCURACY.md` |
| Separate-q0/k0 conjecture refuted (App. A.4) | **VERIFIED** | `PAPER_VALIDATION.md` |
| Clean / noise regimes | **VALID** (retrain bit-identically) | `NOISE_CLEAN_REVALIDATION.md` |
| Vocab sweep, multi-seed, same batch | **VALID** | `VOCAB_SWEEP_MULTISEED.md` |
| Compositional, same training batch | **VALID** | `EM_COMP_SAMEBATCH.md` |
| lm200 / landmarks | **VOID** | `CLAUDE.md` RETRACTION, `CORRECTED_LM200_LEADERBOARD.md` |
| hier-goal navigation | **VOID** | `HIERGOAL_ABLATION.md`, `HIERGOAL_CLOSEDLOOP.md` |
| WM-vs-EM regime narrative | **WITHDRAWN**, replacement falsified | `AP_KERNEL_DIAGNOSTIC.md` |
| Goal-directed / planner-demo tasks | **SUSPECT**, control not run | see below |

## Standing rules learned the hard way

1. Validate a demonstration task against an **n-gram on the action stream
   alone at orders 1-5**, not a single copy-previous baseline. Hier-goal passed
   the order-1 check (0.327) while carrying an order-3 shortcut at 0.971.
2. Never compare a fresh variant against a **stored baseline checkpoint**;
   retrain every arm in the same batch.
3. Report the **measured chance rate** next to every headline; it is often not
   what it looks like (0.50 not 0.25 for torus direction; 0.500 = the
   always-predict-blank floor at n_obs=4096).
4. Check the **training budget suffices** before reading a chance-level table
   as a negative result (Map-Query needed 8x its initial budget).

## VOID — lm200 (whole file) (26)
- `AUX_COEF_SWEEP.md`
- `CASCADE_NOSLOW_CONTROL.md`
- `CLONE_TRANSFER_TEST.md`
- `DROPOUT_ABLATION_RESULTS.md`
- `GSF_MODES_DIAGNOSTIC.md`
- `GSF_NODROP_RESULTS.md`
- `GSF_RESULTS.md`
- `HEX_EMERGENCE_RESULTS.md`
- `HIER_ATTN_MULTIENV.md`
- `LENGTH_DIAGNOSTIC.md`
- `LONGT_EVAL_RESULTS.md`
- `LONG_SEQ_lm200.md`
- `MODEOMEGA_RESULTS.md`
- `MULTIENV_CLEAN_2x2.md`
- `MULTIENV_RESULTS.md`
- `MULTISIZE_RESULTS.md`
- `OMEGA_RESCALE_lm200.md`
- `PER_VISIT_lm200.md`
- `R_T_DISTRIBUTION.md`
- `SINGLE_SIZE_CONTROL.md`
- `SPARSE_LANDMARKS_RESULTS.md`
- `TEM_NOVEL_ENV_RESULTS.md`
- `TEM_SCALING_RESULTS.md`
- `VANILLANODROP_CONTROL.md`
- `ZERO_SHOT_TRANSFER_lm200.md`
- `ZERO_SHOT_TRANSFER_lm200_brokeninit.md`

## VOID — hier-goal (8)
- `DIMSWEEP_d128.md`
- `DIMSWEEP_d256.md`
- `DIMSWEEP_d512.md`
- `EM_FIX_HIERGOAL.md`
- `HIERGOAL_FIXED.md`
- `HIERGOAL_FIXED_LONGT.md`
- `HIERGOAL_MULTISEED.md`
- `HIERGOAL_RESULTS.md`

## PARTIAL — lm200 rows void, clean/noise VALID (14)
- `CAPACITY_PERREGIME.md`
- `GSF_FULL_RESULTS.md`
- `LEVEL15BETA_RESULTS.md`
- `NOBYPASS_RESULTS.md`
- `NODROP_PARETO_RESULTS.md`
- `RESULTS_PAPER.md`
- `TEM_NOISE_FFN_RESULTS.md`
- `TEM_RESULTS.md`
- `TEM_T_MULTISEED.md`
- `TEM_T_RESULTS.md`
- `V3_RESULTS.md`
- `V4_CONTROL_RESULTS.md`
- `V4_MULTISEED.md`
- `V4_RESULTS.md`

## SUSPECT — planner demos (18)
- `ACTIVE_INFERENCE_RESULTS.md`
- `CLOCK_SCAN.md`
- `DAGGER_DK6_RESULTS.md`
- `DAGGER_EMPTY_RESULTS.md`
- `DAGGER_RESULTS.md`
- `DOORKEY_BC_RESULTS.md`
- `GOAL_CLOSEDLOOP_RESULTS.md`
- `GOAL_DIRECTED_RESULTS.md`
- `GOAL_TASKS_RESULTS.md`
- `PROBE_GOAL_DISTANCE.md`
- `PROBE_GOAL_RESULTS.md`
- `ROOMS_MAZE_RESULTS.md`
- `ROOMS_TASK_RESULTS.md`
- `SR_PRETRAIN_RESULTS.md`
- `SR_PROBE_RESULTS.md`
- `VARYING_MAZE_RESULTS.md`
- `VECTOR_NAV_RESULTS.md`
- `VECTOR_NAV_V2_RESULTS.md`

## CONTAINS RETRACTED CLAIMS (3)
- `REPORT_ADDENDUM.md`
- `REPORT_v2.md`
- `RESULTS_SUMMARY_2026-05-10.md`

## SUPERSEDED / UNDERTRAINED (4)
- `CAPACITY_CONTROL.md`
- `COMPOSITIONAL_RESULTS.md`
- `MAP_QUERY_RESULTS.md`
- `VOCAB_SWEEP_RESULTS.md`

## CURRENT (82)
- `AGGREGATE_EXTRAS.md`
- `AGGREGATE_MULTISEED.md`
- `AGGREGATE_TASK_RESULTS.md`
- `AP_KERNEL_DIAGNOSTIC.md`
- `BOUNDED_MEMORY.md`
- `BOUNDED_MEMORY_RESULTS.md`
- `BUMP_TOKEN_RESULTS.md`
- `CASCADE_MULTISEED_RESULTS.md`
- `CASCADE_REPRO_TEST.md`
- `CASCADE_ZEROSHOT_S0.md`
- `CLAUDE.md`
- `CLONE_ANALYSIS_LEVEL15PC.md`
- `CLONE_TRANSFER_NOBYPASS.md`
- `CNAV_HEX_Level15.md`
- `CNAV_HEX_Level15EM.md`
- `CNAV_HEX_Vanilla.md`
- `CNAV_HEX_VanillaEM.md`
- `CNAV_RESULTS.md`
- `COMPOSITIONAL_EXPERIMENT.md`
- `COMPOSITIONAL_MULTISEED.md`
- `CORRECTED_LM200_LEADERBOARD.md`
- `DETAILED_RESULTS.md`
- `DOG_RESULTS.md`
- `EM_COMP_SAMEBATCH.md`
- `EM_FIX_COMP.md`
- `EM_HOPFIELD_CROSSSCALE.md`
- `EM_P0_COMP.md`
- `EM_P0_PAPER.md`
- `EXTRAHEAD_CONTROL.md`
- `GENERALIZATION_REPORT.md`
- `HIERGOAL_ABLATION.md`
- `HIERGOAL_CLOSEDLOOP.md`
- `HIERGOAL_LONGT.md`
- `HIER_ATTN_LONGT.md`
- `HIPPOCAMPAL_ANALYSIS.md`
- `HIPPOCAMPAL_GRID.md`
- `HIPPOCAMPAL_GRIDL15PC.md`
- `HIPPOCAMPAL_GRID_FREE.md`
- `HIPPOCAMPAL_HIDDEN.md`
- `HIPPOCAMPAL_HIDDEN_GRIDFREE.md`
- `HIPPOCAMPAL_LEVEL15PC.md`
- `HOPFIELD_NOMAINAP_RESULTS.md`
- `HOURGLASS_README.md`
- `LEVEL15EM_CROSSSCALE.md`
- `LM200_CORRECTED_MULTISEED.md`
- `LONG_SEQ_clean.md`
- `MAP_QUERY_GATES.md`
- `MATCH_QUERY_GATES.md`
- `MINIGRID_DK16_RESULTS.md`
- `MINIGRID_DOORKEY_CACHED.md`
- `MINIGRID_DOORKEY_LONGT.md`
- `MINIGRID_DOORKEY_RESULTS.md`
- `MINIGRID_DOORKEY_ROPE_DIAG.md`
- `MINIGRID_MEMORY_RESULTS.md`
- `MULTICLASS_MULTISEED_RESULTS.md`
- `MULTICLASS_RESULTS.md`
- `MULTISEED_FOLLOWUP.md`
- `MULTISEED_FOLLOWUP_RESULTS.md`
- `NOISE_CLEAN_REVALIDATION.md`
- `NUMBERLINE_RESULTS.md`
- `OMEGA_RESCALE_clean.md`
- `OOD_GRID_RESULTS.md`
- `PAPER_OOD_PROTOCOL.md`
- `PAPER_TASK_ACCURACY.md`
- `PAPER_VALIDATION.md`
- `PERSCALE_OMEGA_RESULTS.md`
- `PER_VISIT_clean.md`
- `README.md`
- `RECURSIVE_RESULTS.md`
- `REPORT.md`
- `ROUTE_ATTN_RESULTS.md`
- `R_T_DISTRIBUTION_3WAY.md`
- `SESSION_2026-05-01.md`
- `SESSION_HIERARCHICAL_CASCADE.md`
- `SPACETIME_HIER_RESULTS.md`
- `STATE_PROBES.md`
- `TEM_BACKGROUND_BASELINES.md`
- `TEM_CROSSSCALE_DIAGNOSTIC.md`
- `TOPOLOGY_RESULTS.md`
- `VOCAB_SWEEP_MULTISEED.md`
- `ZERO_SHOT_TRANSFER_clean.md`
- `ZERO_SHOT_TRANSFER_clean_brokeninit.md`

