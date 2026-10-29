# Archived code (cleanup B1, 2026-10-29)

Dead and superseded scripts moved here with `git mv` (history kept: `git log --follow archive/code/2026-10/<file>`).
Evidence from `docs/cleanup/INVENTORY_CODE.md` section 4, re-checked on 2026-10-29 with a whole-repo `git grep -w <stem>`
(outside `docs/cleanup/` and `archive*/`): no importer, no live driver, no md5 record (`runs/**/code_md5*.txt`), and any
remaining mention is a comment, a historical patch or inventory, or another file archived here. These scripts ran as
`python3 -m mapformer.<name>` / `bash <name>.sh` from the repo root; to re-run one, check out its last commit before the
move (or move it back), since relative paths and module names assume the root.

Not archived although section 4 listed them (a live reference was found): `analyze_t1.py` (produces T1_RESULTS via
`run_t1_centered.sh`; cited by THEORY_MAPPOPE.md), `environment_textworld_ctx.py`, `environment_textworld_ctx2.py`,
`train_ctxstep.py`, `train_ctxstep2.py` (imported by the `docs/audits/2026-09-27/` gate and swap scripts),
`hyak/mm_gate_v2.slurm` (cited by MEMORY_MAZE_PREREG.md), and `docs/theory/2026-10-04/scripts/{toy,basins2}.py`
(`toy2.py` imports `toy`; the theory folder is an audit record and stays whole).

| file | class | last commit | lines | references outside runs/ | note |
|---|---|---|---:|---|---|
| `_sign_smoke.sh` | DEAD | 2026-09-06 | 12 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `grid_module_hex_test.py` | DEAD | 2026-04-26 | 155 | `.claude/settings.local.json` | permission allow-list entry only |
| `make_inekf_report_figure.py` | DEAD | 2026-06-03 | 176 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `maze_rl_env.py` | DEAD | 2026-09-19 | 138 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `orchestrator_gridl15pc.sh` | DEAD | 2026-04-26 | 63 | `.claude/settings.local.json` | permission allow-list entry only |
| `probe_coarse_angles.py` | DEAD | 2026-08-02 | 95 | `archive/void/HIERGOAL_RESULTS.md` | cited only by void/stale archive |
| `run_addition_cho.sh` | DEAD | 2026-09-14 | 15 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_dimsweep.sh` | DEAD | 2026-08-03 | 39 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_dropout_ablation.sh` | DEAD | 2026-05-10 | 154 | `archive/void/DROPOUT_ABLATION_RESULTS.md` | cited only by void/stale archive |
| `run_goal_closedloop.sh` | DEAD | 2026-05-11 | 43 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_goal_directed.sh` | DEAD | 2026-05-10 | 146 | `archive/void/GOAL_DIRECTED_RESULTS.md` | cited only by void/stale archive |
| `run_goal_tasks.sh` | DEAD | 2026-05-11 | 178 | `archive/void/GOAL_TASKS_RESULTS.md` | cited only by void/stale archive |
| `run_gsf.sh` | DEAD | 2026-05-10 | 125 | `archive_stale/GSF_RESULTS.md`, `archive/void/GSF_RESULTS.md` | cited only by void/stale archive |
| `run_habitat_install.sh` | DEAD | 2026-08-23 | 17 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_hex_emergence.sh` | DEAD | 2026-05-13 | 44 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_hiergoal_fixed.sh` | DEAD | 2026-08-09 | 31 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_hiergoal_multiseed.sh` | DEAD | 2026-07-26 | 39 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_hiergoal_scan.sh` | DEAD | 2026-07-26 | 30 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_modeomega.sh` | DEAD | 2026-05-11 | 189 | `archive_stale/MODEOMEGA_RESULTS.md`, `archive/void/MODEOMEGA_RESULTS.md` | cited only by void/stale archive |
| `run_multisize.sh` | DEAD | 2026-05-11 | 117 | `archive/void/MULTISIZE_RESULTS.md` | cited only by void/stale archive |
| `run_mw_fixed_diag.sh` | DEAD | 2026-08-28 | 53 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_mw_grid_pilot.sh` | DEAD | 2026-09-19 | 51 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_mw_nope.sh` | DEAD | 2026-08-28 | 87 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_mw_scale_diag.sh` | DEAD | 2026-08-28 | 32 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_prebuild.sh` | DEAD | 2026-08-28 | 12 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_probe_goal.sh` | DEAD | 2026-05-10 | 38 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_probe_sr.sh` | DEAD | 2026-05-11 | 54 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_recency_budget.sh` | DEAD | 2026-09-07 | 23 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_tem_scaling.sh` | DEAD | 2026-05-21 | 118 | `archive/void/TEM_SCALING_RESULTS.md` | cited only by void/stale archive |
| `run_vector_nav.sh` | DEAD | 2026-05-13 | 44 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `train_maze_varying.py` | DEAD | 2026-07-22 | 115 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `train_rooms.py` | DEAD | 2026-07-18 | 108 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `train_rooms_goal.py` | DEAD | 2026-07-21 | 121 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `validate_clock.py` | DEAD | 2026-07-28 | 32 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `verify_theory.py` | DEAD | 2026-09-06 | 225 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `probe_vector_nav.py` | SUPERSEDED | 2026-09-24 | 300 | none | newer version exists: probe_vector_nav_v2 |
| `run_addition_pilot.sh` | SUPERSEDED | 2026-09-14 | 22 | none | one-off pilot / top-up / redo / amend |
| `run_addition_pilot2.sh` | SUPERSEDED | 2026-09-14 | 27 | `run_tem_recency_pilot.sh` | one-off pilot / top-up / redo / amend |
| `run_capacity_rerun.sh` | SUPERSEDED | 2026-05-22 | 136 | `CAPACITY_CONTROL.md` | one-off pilot / top-up / redo / amend |
| `run_cmq_curric_topup.sh` | SUPERSEDED | 2026-08-24 | 36 | none | one-off pilot / top-up / redo / amend |
| `run_cmq_stab_topup.sh` | SUPERSEDED | 2026-08-23 | 35 | none | one-off pilot / top-up / redo / amend |
| `run_cmq_topup.sh` | SUPERSEDED | 2026-08-23 | 40 | none | one-off pilot / top-up / redo / amend |
| `run_cnav_redo.sh` | SUPERSEDED | 2026-05-03 | 118 | `.claude/settings.local.json` | permission allow-list entry only |
| `run_cont_redo.sh` | SUPERSEDED | 2026-08-23 | 40 | none | one-off pilot / top-up / redo / amend |
| `run_looped_pilot.sh` | SUPERSEDED | 2026-09-24 | 105 | `report/inventory/C_hierarchy_loops_tasks.md` | one-off pilot / top-up / redo / amend |
| `run_minigrid_allo.sh` | SUPERSEDED | 2026-08-24 | 50 | none | newer version exists: run_minigrid_allo8 |
| `run_redo_all.sh` | SUPERSEDED | 2026-08-23 | 48 | none | one-off pilot / top-up / redo / amend |
| `run_rotate_redo.sh` | SUPERSEDED | 2026-08-21 | 39 | none | one-off pilot / top-up / redo / amend |
| `run_sameblock_amend.sh` | SUPERSEDED | 2026-09-15 | 19 | none | one-off pilot / top-up / redo / amend |
| `run_t3gen.sh` | SUPERSEDED | 2026-09-19 | 54 | none | newer version exists: run_t3gen2 |
| `run_tem_recency_pilot.sh` | SUPERSEDED | 2026-09-14 | 19 | none | one-off pilot / top-up / redo / amend |
| `run_topup_n8.sh` | SUPERSEDED | 2026-08-01 | 51 | none | one-off pilot / top-up / redo / amend |
