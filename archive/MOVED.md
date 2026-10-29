# Moved and removed files (cleanup of 2026-10-29)

One row per file the 2026-10-29 cleanup moved or removed (plan: `docs/cleanup/CLEANUP_PLAN.md`; log: `docs/cleanup/EXECUTION_LOG.md`).
Historical documents (docs/LOG.md, audits, sessions, old reports, pre-registrations) were NOT rewritten: when one of them names a
file below by its old path, look the name up here. Archived files keep their git history (`git log --follow <new path>`).

Folders: `archive/withdrawn/` withdrawn or void results (CLAUDE.md "Withdrawn"); `archive/superseded/` files a newer named
version replaces; `archive/orphan/` results no live document, result file or reader script refers to (each read before the
move); `archive/legacy_level15/` the April-August Level 1.5 / InEKF / PC / TEM / lm200-era results (withdrawn as a line);
`archive/code/2026-10/` dead and superseded scripts; `docs/paper/old/` superseded drafts, outlines and reviews.
`archive/void/` and `archive_stale/` predate this cleanup and are unchanged.

| old path | new path | step | date |
|---|---|---|---|
| `CLONE_ANALYSIS_LEVEL15PC.md` | deleted; byte-identical copy at `archive_stale/CLONE_ANALYSIS_LEVEL15PC.md` | A1 | 2026-10-29 |
| `CLONE_TRANSFER_NOBYPASS.md` | deleted; byte-identical copy at `archive_stale/CLONE_TRANSFER_NOBYPASS.md` | A1 | 2026-10-29 |
| `CNAV_HEX_Level15.md` | deleted; byte-identical copy at `archive_stale/CNAV_HEX_Level15.md` | A1 | 2026-10-29 |
| `CNAV_HEX_Level15EM.md` | deleted; byte-identical copy at `archive_stale/CNAV_HEX_Level15EM.md` | A1 | 2026-10-29 |
| `CNAV_HEX_Vanilla.md` | deleted; byte-identical copy at `archive_stale/CNAV_HEX_Vanilla.md` | A1 | 2026-10-29 |
| `CNAV_HEX_VanillaEM.md` | deleted; byte-identical copy at `archive_stale/CNAV_HEX_VanillaEM.md` | A1 | 2026-10-29 |
| `CNAV_RESULTS.md` | deleted; byte-identical copy at `archive_stale/CNAV_RESULTS.md` | A1 | 2026-10-29 |
| `DOG_RESULTS.md` | deleted; byte-identical copy at `archive_stale/DOG_RESULTS.md` | A1 | 2026-10-29 |
| `HIPPOCAMPAL_ANALYSIS.md` | deleted; byte-identical copy at `archive_stale/HIPPOCAMPAL_ANALYSIS.md` | A1 | 2026-10-29 |
| `HIPPOCAMPAL_GRID.md` | deleted; byte-identical copy at `archive_stale/HIPPOCAMPAL_GRID.md` | A1 | 2026-10-29 |
| `HIPPOCAMPAL_GRIDL15PC.md` | deleted; byte-identical copy at `archive_stale/HIPPOCAMPAL_GRIDL15PC.md` | A1 | 2026-10-29 |
| `HIPPOCAMPAL_GRID_FREE.md` | deleted; byte-identical copy at `archive_stale/HIPPOCAMPAL_GRID_FREE.md` | A1 | 2026-10-29 |
| `HIPPOCAMPAL_HIDDEN.md` | deleted; byte-identical copy at `archive_stale/HIPPOCAMPAL_HIDDEN.md` | A1 | 2026-10-29 |
| `HIPPOCAMPAL_HIDDEN_GRIDFREE.md` | deleted; byte-identical copy at `archive_stale/HIPPOCAMPAL_HIDDEN_GRIDFREE.md` | A1 | 2026-10-29 |
| `HIPPOCAMPAL_LEVEL15PC.md` | deleted; byte-identical copy at `archive_stale/HIPPOCAMPAL_LEVEL15PC.md` | A1 | 2026-10-29 |
| `MULTICLASS_RESULTS.md` | deleted; byte-identical copy at `archive_stale/MULTICLASS_RESULTS.md` | A1 | 2026-10-29 |
| `MULTISEED_FOLLOWUP.md` | deleted; byte-identical copy at `archive_stale/MULTISEED_FOLLOWUP.md` | A1 | 2026-10-29 |
| `OOD_GRID_RESULTS.md` | deleted; byte-identical copy at `archive_stale/OOD_GRID_RESULTS.md` | A1 | 2026-10-29 |
| `R_T_DISTRIBUTION_3WAY.md` | deleted; byte-identical copy at `archive_stale/R_T_DISTRIBUTION_3WAY.md` | A1 | 2026-10-29 |
| `ZERO_SHOT_TRANSFER_clean_brokeninit.md` | deleted; byte-identical copy at `archive_stale/ZERO_SHOT_TRANSFER_clean_brokeninit.md` | A1 | 2026-10-29 |
| `docs/prepared/dyck_mdepth/pkg/` (373 tracked files: a 2026-09-25 copy of the package used for the Dyck matched-depth gates) | deleted; unreferenced outside itself; recoverable from git history (`git show 233eb9b:docs/prepared/dyck_mdepth/pkg/...`) | C1 | 2026-10-29 |
| `ACCUMULATOR.md` | `archive/withdrawn/ACCUMULATOR.md` | A2 | 2026-10-29 |
| `ACCUMULATOR.json` | `archive/withdrawn/ACCUMULATOR.json` | A2 | 2026-10-29 |
| `DXR_RANK_THRESHOLD.md` | `archive/withdrawn/DXR_RANK_THRESHOLD.md` | A2 | 2026-10-29 |
| `DXR_RANK_THRESHOLD.json` | `archive/withdrawn/DXR_RANK_THRESHOLD.json` | A2 | 2026-10-29 |
| `DYCK_T3_RESULTS.md` | `archive/withdrawn/DYCK_T3_RESULTS.md` | A2 | 2026-10-29 |
| `ENWIK8_HIER.md` | `archive/withdrawn/ENWIK8_HIER.md` | A2 | 2026-10-29 |
| `HIERGOAL_ABLATION.md` | `archive/withdrawn/HIERGOAL_ABLATION.md` | A2 | 2026-10-29 |
| `HIERGOAL_ABLATION.json` | `archive/withdrawn/HIERGOAL_ABLATION.json` | A2 | 2026-10-29 |
| `HORIZON_RESULTS.md` | `archive/withdrawn/HORIZON_RESULTS.md` | A2 | 2026-10-29 |
| `HORIZON_L1d128e16.md` | `archive/withdrawn/HORIZON_L1d128e16.md` | A2 | 2026-10-29 |
| `HORIZON_L1d128e16.json` | `archive/withdrawn/HORIZON_L1d128e16.json` | A2 | 2026-10-29 |
| `HORIZON_L1d128e50.md` | `archive/withdrawn/HORIZON_L1d128e50.md` | A2 | 2026-10-29 |
| `HORIZON_L1d128e50.json` | `archive/withdrawn/HORIZON_L1d128e50.json` | A2 | 2026-10-29 |
| `HORIZON_L2d128e16.md` | `archive/withdrawn/HORIZON_L2d128e16.md` | A2 | 2026-10-29 |
| `HORIZON_L2d128e16.json` | `archive/withdrawn/HORIZON_L2d128e16.json` | A2 | 2026-10-29 |
| `HORIZON_L2d256e16.md` | `archive/withdrawn/HORIZON_L2d256e16.md` | A2 | 2026-10-29 |
| `HORIZON_L2d256e16.json` | `archive/withdrawn/HORIZON_L2d256e16.json` | A2 | 2026-10-29 |
| `HORIZON_L4d128e16.md` | `archive/withdrawn/HORIZON_L4d128e16.md` | A2 | 2026-10-29 |
| `HORIZON_L4d128e16.json` | `archive/withdrawn/HORIZON_L4d128e16.json` | A2 | 2026-10-29 |
| `HORIZON_L4d256e16.md` | `archive/withdrawn/HORIZON_L4d256e16.md` | A2 | 2026-10-29 |
| `HORIZON_L4d256e16.json` | `archive/withdrawn/HORIZON_L4d256e16.json` | A2 | 2026-10-29 |
| `HORIZON_TASK_DISTANCES.md` | `archive/withdrawn/HORIZON_TASK_DISTANCES.md` | A2 | 2026-10-29 |
| `HORIZON_TASK_DISTANCES.json` | `archive/withdrawn/HORIZON_TASK_DISTANCES.json` | A2 | 2026-10-29 |
| `LAP_TRANSFER_NOREWARD.md` | `archive/withdrawn/LAP_TRANSFER_NOREWARD.md` | A2 | 2026-10-29 |
| `LAP_TRANSFER_NOREWARD.json` | `archive/withdrawn/LAP_TRANSFER_NOREWARD.json` | A2 | 2026-10-29 |
| `N3_AUDIT.md` | `archive/withdrawn/N3_AUDIT.md` | A2 | 2026-10-29 |
| `PLANNER_TASK_AUDIT.md` | `archive/withdrawn/PLANNER_TASK_AUDIT.md` | A2 | 2026-10-29 |
| `PLANNER_TASK_AUDIT.json` | `archive/withdrawn/PLANNER_TASK_AUDIT.json` | A2 | 2026-10-29 |
| `REPORT_ADDENDUM.md` | `archive/withdrawn/REPORT_ADDENDUM.md` | A2 | 2026-10-29 |
| `REPORT_v2.md` | `archive/withdrawn/REPORT_v2.md` | A2 | 2026-10-29 |
| `REPORT_v2.pdf` | `archive/withdrawn/REPORT_v2.pdf` | A2 | 2026-10-29 |
| `RESULTS_PAPER.md` | `archive/withdrawn/RESULTS_PAPER.md` | A2 | 2026-10-29 |
| `RESULTS_SUMMARY_2026-05-10.md` | `archive/withdrawn/RESULTS_SUMMARY_2026-05-10.md` | A2 | 2026-10-29 |
| `RESULTS_SUMMARY_2026-05-10.pdf` | `archive/withdrawn/RESULTS_SUMMARY_2026-05-10.pdf` | A2 | 2026-10-29 |
| `ADDITION_PILOT.md` | `archive/superseded/ADDITION_PILOT.md` | A3 | 2026-10-29 |
| `COMPOSITIONAL_RESULTS.md` | `archive/superseded/COMPOSITIONAL_RESULTS.md` | A3 | 2026-10-29 |
| `DXR_PRELIM.md` | `archive/superseded/DXR_PRELIM.md` | A3 | 2026-10-29 |
| `DXR_PRELIM.json` | `archive/superseded/DXR_PRELIM.json` | A3 | 2026-10-29 |
| `INDEX_BASELINE_PAPER_TASK.md` | `archive/superseded/INDEX_BASELINE_PAPER_TASK.md` | A3 | 2026-10-29 |
| `INDEX_BASELINE_PAPER_TASK.json` | `archive/superseded/INDEX_BASELINE_PAPER_TASK.json` | A3 | 2026-10-29 |
| `MINIGRID_2X2X2.md` | `archive/superseded/MINIGRID_2X2X2.md` | A3 | 2026-10-29 |
| `MINIGRID_2X2X2.json` | `archive/superseded/MINIGRID_2X2X2.json` | A3 | 2026-10-29 |
| `MINIWORLD_FIXED_RESULTS.md` | `archive/superseded/MINIWORLD_FIXED_RESULTS.md` | A3 | 2026-10-29 |
| `MINIWORLD_FRESH_RESULTS.md` | `archive/superseded/MINIWORLD_FRESH_RESULTS.md` | A3 | 2026-10-29 |
| `MINIWORLD_ORACLE_RESULTS_T512.md` | `archive/superseded/MINIWORLD_ORACLE_RESULTS_T512.md` | A3 | 2026-10-29 |
| `PAPER_OOD_EXTENDED.md` | `archive/superseded/PAPER_OOD_EXTENDED.md` | A3 | 2026-10-29 |
| `PAPER_OOD_EXTENDED.json` | `archive/superseded/PAPER_OOD_EXTENDED.json` | A3 | 2026-10-29 |
| `REPORT.md` | `archive/superseded/REPORT.md` | A3 | 2026-10-29 |
| `REPORT.pdf` | `archive/superseded/REPORT.pdf` | A3 | 2026-10-29 |
| `COMPOSITIONAL_MATCH_QUERY_GATES.md` | `archive/orphan/COMPOSITIONAL_MATCH_QUERY_GATES.md` | A3 | 2026-10-29 |
| `COMPOSITIONAL_MATCH_QUERY_GATES.json` | `archive/orphan/COMPOSITIONAL_MATCH_QUERY_GATES.json` | A3 | 2026-10-29 |
| `COMPOSITIONAL_MATCH_QUERY_RESULTS.md` | `archive/orphan/COMPOSITIONAL_MATCH_QUERY_RESULTS.md` | A3 | 2026-10-29 |
| `COMPOSITIONAL_PLAIN_RESULTS.md` | `archive/orphan/COMPOSITIONAL_PLAIN_RESULTS.md` | A3 | 2026-10-29 |
| `GENERALIZATION_REPORT.md` | `archive/orphan/GENERALIZATION_REPORT.md` | A3 | 2026-10-29 |
| `GENERALIZATION_REPORT.pdf` | `archive/orphan/GENERALIZATION_REPORT.pdf` | A3 | 2026-10-29 |
| `HIERGOAL_LONGT.md` | `archive/orphan/HIERGOAL_LONGT.md` | A3 | 2026-10-29 |
| `HIERGOAL_LONGT.json` | `archive/orphan/HIERGOAL_LONGT.json` | A3 | 2026-10-29 |
| `MAPPOPE_VS_POPE.md` | `archive/orphan/MAPPOPE_VS_POPE.md` | A3 | 2026-10-29 |
| `MINIGRID_2X2.md` | `archive/orphan/MINIGRID_2X2.md` | A3 | 2026-10-29 |
| `MINIGRID_2X2.json` | `archive/orphan/MINIGRID_2X2.json` | A3 | 2026-10-29 |
| `MINIWORLD_RESULTS.md` | `archive/orphan/MINIWORLD_RESULTS.md` | A3 | 2026-10-29 |
| `MINIWORLD_TODO.md` | `archive/orphan/MINIWORLD_TODO.md` | A3 | 2026-10-29 |
| `OMEGA_RESCALE_clean.md` | `archive/orphan/OMEGA_RESCALE_clean.md` | A3 | 2026-10-29 |
| `PUBLICATION_VENUES.md` | `archive/orphan/PUBLICATION_VENUES.md` | A3 | 2026-10-29 |
| `RECURSIVE_RESULTS.md` | `archive/orphan/RECURSIVE_RESULTS.md` | A3 | 2026-10-29 |
| `ROOMS_GOAL_RESULTS.md` | `archive/orphan/ROOMS_GOAL_RESULTS.md` | A3 | 2026-10-29 |
| `ROUTE_ATTN_RESULTS.md` | `archive/orphan/ROUTE_ATTN_RESULTS.md` | A3 | 2026-10-29 |
| `SESSION_HIERARCHICAL_CASCADE.md` | `archive/orphan/SESSION_HIERARCHICAL_CASCADE.md` | A3 | 2026-10-29 |
| `SPACETIME_HIER_RESULTS.md` | `archive/orphan/SPACETIME_HIER_RESULTS.md` | A3 | 2026-10-29 |
| `THEORY_NARRATIVE.md` | `archive/orphan/THEORY_NARRATIVE.md` | A3 | 2026-10-29 |
| `BUMP_TOKEN_RESULTS.md` | `archive/legacy_level15/BUMP_TOKEN_RESULTS.md` | A4 | 2026-10-29 |
| `CAPACITY_CONTROL.md` | `archive/legacy_level15/CAPACITY_CONTROL.md` | A4 | 2026-10-29 |
| `CAPACITY_PERREGIME.md` | `archive/legacy_level15/CAPACITY_PERREGIME.md` | A4 | 2026-10-29 |
| `CASCADE_MULTISEED_RESULTS.md` | `archive/legacy_level15/CASCADE_MULTISEED_RESULTS.md` | A4 | 2026-10-29 |
| `CASCADE_REPRO_TEST.md` | `archive/legacy_level15/CASCADE_REPRO_TEST.md` | A4 | 2026-10-29 |
| `CASCADE_ZEROSHOT_S0.md` | `archive/legacy_level15/CASCADE_ZEROSHOT_S0.md` | A4 | 2026-10-29 |
| `CORRECTED_LM200_LEADERBOARD.md` | `archive/legacy_level15/CORRECTED_LM200_LEADERBOARD.md` | A4 | 2026-10-29 |
| `GSF_FULL_RESULTS.md` | `archive/legacy_level15/GSF_FULL_RESULTS.md` | A4 | 2026-10-29 |
| `LEVEL15_MEETS_GATED_matchq.md` | `archive/legacy_level15/LEVEL15_MEETS_GATED_matchq.md` | A4 | 2026-10-29 |
| `LEVEL15_MEETS_GATED_paper.md` | `archive/legacy_level15/LEVEL15_MEETS_GATED_paper.md` | A4 | 2026-10-29 |
| `LEVEL15_MEETS_GATED_paper.json` | `archive/legacy_level15/LEVEL15_MEETS_GATED_paper.json` | A4 | 2026-10-29 |
| `LEVEL15_MEETS_GATED_paper50.md` | `archive/legacy_level15/LEVEL15_MEETS_GATED_paper50.md` | A4 | 2026-10-29 |
| `LEVEL15_MEETS_GATED_paper50.json` | `archive/legacy_level15/LEVEL15_MEETS_GATED_paper50.json` | A4 | 2026-10-29 |
| `LEVEL15BETA_RESULTS.md` | `archive/legacy_level15/LEVEL15BETA_RESULTS.md` | A4 | 2026-10-29 |
| `LEVEL15EM_CROSSSCALE.md` | `archive/legacy_level15/LEVEL15EM_CROSSSCALE.md` | A4 | 2026-10-29 |
| `LM200_ABLATION.md` | `archive/legacy_level15/LM200_ABLATION.md` | A4 | 2026-10-29 |
| `LM200_ABLATION.json` | `archive/legacy_level15/LM200_ABLATION.json` | A4 | 2026-10-29 |
| `LM200_CORRECTED_MULTISEED.md` | `archive/legacy_level15/LM200_CORRECTED_MULTISEED.md` | A4 | 2026-10-29 |
| `MULTICLASS_MULTISEED_RESULTS.md` | `archive/legacy_level15/MULTICLASS_MULTISEED_RESULTS.md` | A4 | 2026-10-29 |
| `MULTISEED_FOLLOWUP_RESULTS.md` | `archive/legacy_level15/MULTISEED_FOLLOWUP_RESULTS.md` | A4 | 2026-10-29 |
| `NOBYPASS_RESULTS.md` | `archive/legacy_level15/NOBYPASS_RESULTS.md` | A4 | 2026-10-29 |
| `NODROP_PARETO_RESULTS.md` | `archive/legacy_level15/NODROP_PARETO_RESULTS.md` | A4 | 2026-10-29 |
| `NUMBERLINE_RESULTS.md` | `archive/legacy_level15/NUMBERLINE_RESULTS.md` | A4 | 2026-10-29 |
| `TEM_BACKGROUND_BASELINES.md` | `archive/legacy_level15/TEM_BACKGROUND_BASELINES.md` | A4 | 2026-10-29 |
| `TEM_CROSSSCALE_DIAGNOSTIC.md` | `archive/legacy_level15/TEM_CROSSSCALE_DIAGNOSTIC.md` | A4 | 2026-10-29 |
| `TEM_NOISE_FFN_RESULTS.md` | `archive/legacy_level15/TEM_NOISE_FFN_RESULTS.md` | A4 | 2026-10-29 |
| `TEM_RESULTS.md` | `archive/legacy_level15/TEM_RESULTS.md` | A4 | 2026-10-29 |
| `TEM_T_MULTISEED.md` | `archive/legacy_level15/TEM_T_MULTISEED.md` | A4 | 2026-10-29 |
| `TEM_T_RESULTS.md` | `archive/legacy_level15/TEM_T_RESULTS.md` | A4 | 2026-10-29 |
| `V3_RESULTS.md` | `archive/legacy_level15/V3_RESULTS.md` | A4 | 2026-10-29 |
| `V4_CONTROL_RESULTS.md` | `archive/legacy_level15/V4_CONTROL_RESULTS.md` | A4 | 2026-10-29 |
| `V4_MULTISEED.md` | `archive/legacy_level15/V4_MULTISEED.md` | A4 | 2026-10-29 |
| `V4_RESULTS.md` | `archive/legacy_level15/V4_RESULTS.md` | A4 | 2026-10-29 |
| `VECTOR_NAV_V2_RESULTS.md` | `archive/legacy_level15/VECTOR_NAV_V2_RESULTS.md` | A4 | 2026-10-29 |
| `VOCAB_SWEEP_MULTISEED.md` | `archive/legacy_level15/VOCAB_SWEEP_MULTISEED.md` | A4 | 2026-10-29 |
| `VOCAB_SWEEP_MULTISEED.json` | `archive/legacy_level15/VOCAB_SWEEP_MULTISEED.json` | A4 | 2026-10-29 |
| `VOCAB_SWEEP_RESULTS.md` | `archive/legacy_level15/VOCAB_SWEEP_RESULTS.md` | A4 | 2026-10-29 |
| `_sign_smoke.sh` | `archive/code/2026-10/_sign_smoke.sh` | B1 | 2026-10-29 |
| `grid_module_hex_test.py` | `archive/code/2026-10/grid_module_hex_test.py` | B1 | 2026-10-29 |
| `make_inekf_report_figure.py` | `archive/code/2026-10/make_inekf_report_figure.py` | B1 | 2026-10-29 |
| `maze_rl_env.py` | `archive/code/2026-10/maze_rl_env.py` | B1 | 2026-10-29 |
| `orchestrator_gridl15pc.sh` | `archive/code/2026-10/orchestrator_gridl15pc.sh` | B1 | 2026-10-29 |
| `probe_coarse_angles.py` | `archive/code/2026-10/probe_coarse_angles.py` | B1 | 2026-10-29 |
| `run_addition_cho.sh` | `archive/code/2026-10/run_addition_cho.sh` | B1 | 2026-10-29 |
| `run_dimsweep.sh` | `archive/code/2026-10/run_dimsweep.sh` | B1 | 2026-10-29 |
| `run_dropout_ablation.sh` | `archive/code/2026-10/run_dropout_ablation.sh` | B1 | 2026-10-29 |
| `run_goal_closedloop.sh` | `archive/code/2026-10/run_goal_closedloop.sh` | B1 | 2026-10-29 |
| `run_goal_directed.sh` | `archive/code/2026-10/run_goal_directed.sh` | B1 | 2026-10-29 |
| `run_goal_tasks.sh` | `archive/code/2026-10/run_goal_tasks.sh` | B1 | 2026-10-29 |
| `run_gsf.sh` | `archive/code/2026-10/run_gsf.sh` | B1 | 2026-10-29 |
| `run_habitat_install.sh` | `archive/code/2026-10/run_habitat_install.sh` | B1 | 2026-10-29 |
| `run_hex_emergence.sh` | `archive/code/2026-10/run_hex_emergence.sh` | B1 | 2026-10-29 |
| `run_hiergoal_fixed.sh` | `archive/code/2026-10/run_hiergoal_fixed.sh` | B1 | 2026-10-29 |
| `run_hiergoal_multiseed.sh` | `archive/code/2026-10/run_hiergoal_multiseed.sh` | B1 | 2026-10-29 |
| `run_hiergoal_scan.sh` | `archive/code/2026-10/run_hiergoal_scan.sh` | B1 | 2026-10-29 |
| `run_modeomega.sh` | `archive/code/2026-10/run_modeomega.sh` | B1 | 2026-10-29 |
| `run_multisize.sh` | `archive/code/2026-10/run_multisize.sh` | B1 | 2026-10-29 |
| `run_mw_fixed_diag.sh` | `archive/code/2026-10/run_mw_fixed_diag.sh` | B1 | 2026-10-29 |
| `run_mw_grid_pilot.sh` | `archive/code/2026-10/run_mw_grid_pilot.sh` | B1 | 2026-10-29 |
| `run_mw_nope.sh` | `archive/code/2026-10/run_mw_nope.sh` | B1 | 2026-10-29 |
| `run_mw_scale_diag.sh` | `archive/code/2026-10/run_mw_scale_diag.sh` | B1 | 2026-10-29 |
| `run_prebuild.sh` | `archive/code/2026-10/run_prebuild.sh` | B1 | 2026-10-29 |
| `run_probe_goal.sh` | `archive/code/2026-10/run_probe_goal.sh` | B1 | 2026-10-29 |
| `run_probe_sr.sh` | `archive/code/2026-10/run_probe_sr.sh` | B1 | 2026-10-29 |
| `run_recency_budget.sh` | `archive/code/2026-10/run_recency_budget.sh` | B1 | 2026-10-29 |
| `run_tem_scaling.sh` | `archive/code/2026-10/run_tem_scaling.sh` | B1 | 2026-10-29 |
| `run_vector_nav.sh` | `archive/code/2026-10/run_vector_nav.sh` | B1 | 2026-10-29 |
| `train_maze_varying.py` | `archive/code/2026-10/train_maze_varying.py` | B1 | 2026-10-29 |
| `train_rooms.py` | `archive/code/2026-10/train_rooms.py` | B1 | 2026-10-29 |
| `train_rooms_goal.py` | `archive/code/2026-10/train_rooms_goal.py` | B1 | 2026-10-29 |
| `validate_clock.py` | `archive/code/2026-10/validate_clock.py` | B1 | 2026-10-29 |
| `verify_theory.py` | `archive/code/2026-10/verify_theory.py` | B1 | 2026-10-29 |
| `probe_vector_nav.py` | `archive/code/2026-10/probe_vector_nav.py` | B1 | 2026-10-29 |
| `run_addition_pilot.sh` | `archive/code/2026-10/run_addition_pilot.sh` | B1 | 2026-10-29 |
| `run_addition_pilot2.sh` | `archive/code/2026-10/run_addition_pilot2.sh` | B1 | 2026-10-29 |
| `run_capacity_rerun.sh` | `archive/code/2026-10/run_capacity_rerun.sh` | B1 | 2026-10-29 |
| `run_cmq_curric_topup.sh` | `archive/code/2026-10/run_cmq_curric_topup.sh` | B1 | 2026-10-29 |
| `run_cmq_stab_topup.sh` | `archive/code/2026-10/run_cmq_stab_topup.sh` | B1 | 2026-10-29 |
| `run_cmq_topup.sh` | `archive/code/2026-10/run_cmq_topup.sh` | B1 | 2026-10-29 |
| `run_cnav_redo.sh` | `archive/code/2026-10/run_cnav_redo.sh` | B1 | 2026-10-29 |
| `run_cont_redo.sh` | `archive/code/2026-10/run_cont_redo.sh` | B1 | 2026-10-29 |
| `run_looped_pilot.sh` | `archive/code/2026-10/run_looped_pilot.sh` | B1 | 2026-10-29 |
| `run_minigrid_allo.sh` | `archive/code/2026-10/run_minigrid_allo.sh` | B1 | 2026-10-29 |
| `run_redo_all.sh` | `archive/code/2026-10/run_redo_all.sh` | B1 | 2026-10-29 |
| `run_rotate_redo.sh` | `archive/code/2026-10/run_rotate_redo.sh` | B1 | 2026-10-29 |
| `run_sameblock_amend.sh` | `archive/code/2026-10/run_sameblock_amend.sh` | B1 | 2026-10-29 |
| `run_t3gen.sh` | `archive/code/2026-10/run_t3gen.sh` | B1 | 2026-10-29 |
| `run_tem_recency_pilot.sh` | `archive/code/2026-10/run_tem_recency_pilot.sh` | B1 | 2026-10-29 |
| `run_topup_n8.sh` | `archive/code/2026-10/run_topup_n8.sh` | B1 | 2026-10-29 |
| `docs/paper/pdf/0_COMBINED_all_documents.pdf` | `docs/paper/old/pdf/0_COMBINED_all_documents.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/1_main_paper.pdf` | `docs/paper/old/pdf/1_main_paper.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/2_main_paper_appendix.pdf` | `docs/paper/old/pdf/2_main_paper_appendix.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/6_main_paper_outline.pdf` | `docs/paper/old/pdf/6_main_paper_outline.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/8_delta_free_results.pdf` | `docs/paper/old/pdf/8_delta_free_results.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/9_what_where_T1024.pdf` | `docs/paper/old/pdf/9_what_where_T1024.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/10_draft2_plan.pdf` | `docs/paper/old/pdf/10_draft2_plan.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/11_expansion_mappope.pdf` | `docs/paper/old/pdf/11_expansion_mappope.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/12_expansion_rank_theory.pdf` | `docs/paper/old/pdf/12_expansion_rank_theory.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/13_expansion_search_theory.pdf` | `docs/paper/old/pdf/13_expansion_search_theory.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/14_expansion_depth_theory.pdf` | `docs/paper/old/pdf/14_expansion_depth_theory.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/15_expansion_recurrence.pdf` | `docs/paper/old/pdf/15_expansion_recurrence.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/16_expansion_language.pdf` | `docs/paper/old/pdf/16_expansion_language.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/17_expansion_paper_level.pdf` | `docs/paper/old/pdf/17_expansion_paper_level.pdf` | C3 | 2026-10-29 |
| `docs/paper/pdf/paper_sources_and_figures.zip` | `docs/paper/old/pdf/paper_sources_and_figures.zip` | C3 | 2026-10-29 |
