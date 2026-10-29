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
