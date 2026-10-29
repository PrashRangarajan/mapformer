# Code inventory (2026-10-29, read-only)

Scope: every top-level `*.py` (532) and `*.sh` (350), plus `hyak/` (6 `.slurm`/`.sh`), `docs/paper/tools/`,
`docs/tools/`, `docs/paper/expansion/scripts/**` and `docs/theory/2026-10-04/scripts/` (78 scripts): **968 files**.
`paper_figures/` holds no scripts (only `.png`/`.npz` and two sub-directories of outputs). `docs/prepared/dyck_mdepth/pkg/`
is a frozen 389-file snapshot of the package (2026-09-27) and is out of scope: nothing imports it; leave it as the
snapshot it is. Nothing was moved, edited, deleted or committed; no training was run. CPU-only tests were run (Section 7).

## How the graph was built (and what it can miss)

- **Import graph** (1,402 edges): `ast` over every file, `import mapformer.X`, `from mapformer.X import`, `from .X import`,
  `from mapformer import X`, imports inside functions (counted as dynamic), `importlib.import_module("mapformer.X")` and
  `__import__("mapformer.X")` (how `train_variant.VARIANT_MAP` loads 24 of its modules). The package `__init__.py` imports
  `lie_groups`, `prefix_scan`, `environment` and `model`, so every `import mapformer.X` also loads those four.
- **Driver graph** (1,950 edges from shell files): every `mapformer.X`, `X.py` and `X.sh` token in a `.sh`/`.slurm`, and the
  same tokens inside Python strings (subprocess launches, md5 lists).
- **Citations**, four kinds, each split into *live* and *non-live* documents:
  1. *named-in*: the file name in a `.md`/`.tex`/`.html` (top-level, `docs/**`, `report/**`, `.claude-memory/`);
  2. *commit*: a results/pre-registration file cites a commit hash (e.g. "pre-registration ... commit 3a9e49a") and the
     script was added or changed in that commit (commits touching > 30 files ignored);
  3. *run dir*: the script names `runs/<d>` and a document names `runs/<d>`;
  4. *output*: the script reads or writes an upper-case `*.md` (e.g. `run_dyck.sh` writes `DYCK_RESULTS_bs$BS.md`), or its
     run dir is `<d>` and `<D>_RESULTS*.md` exists.
- **md5 records**: 56 `runs/**/code_md5*.txt` files; 244 distinct modules are md5-recorded (Section 7.5).
- **Live documents**: `CLAUDE.md`; every results / pre-registration file named in its *Citable results* and rank sections
  (150); `docs/paper/MAIN_DRAFT2.md`, `MAIN_APPENDIX2.md`, `DRAFT6_workshop.md`, `APPENDIX_draft6.md`,
  `CLAIM_MAP_clock_map.md`, `docs/paper/expansion/**`, `docs/paper/appendix_checks/**`, `docs/audits/2026-10-2*`;
  `docs/WHERE_THINGS_STAND.md`, `project_state.md`, `GUARDS.md`, `README.md`. A run dir is live if a live document names it.
  `RESULTS_INDEX.md` is excluded (it names everything), as are `docs/cleanup/*` and `docs/prepared/*`.

Caveats. The classification is mechanical, then spot-checked by hand (every DEAD and SUPERSEDED file was re-checked with a
whole-word grep over the whole repository outside `runs/`; Section 4). Token matching over-links rather than under-links
(a script that mentions `model.py` in a comment counts as depending on it), so LIVE is generous on purpose: the cost of a
false LIVE is a file kept, the cost of a false DEAD is a broken reproduction. `runs/` (31 GB) was not grepped for
command lines; a script whose only trace is a log line inside a run dir would show as DEAD. None of the 36 DEAD files has
a run dir of its own name.

## Classes

| class | rule |
|---|---|
| CORE | top-level module on a live path and imported by >= 5 modules or by >= 2 live modules |
| LIVE | cited by a live document (any of the four kinds), md5-recorded in a live run dir, or a dependency (import / driver edge, transitively) of such a file |
| HISTORICAL | *added class*: referenced only by non-live material (withdrawn or uncited results files, `docs/LOG.md`, old audits, non-live run dirs, non-live drivers). Not dead -- it reproduces a number that is written down somewhere -- but no current claim rests on it |
| SUPERSEDED | not live, and either a newer member of the same version family exists (`foo` -> `foo2` -> `foo_v3`) or it is a one-off pilot / top-up / redo / amend driver |
| DEAD | no importer, no driver, no document mention, no commit / run-dir / output link, no md5 record |

## Headline findings

1. **Almost nothing is truly dead: 36 files (10 `.py`, 26 `.sh`), ~2,900 lines**, all last touched 2026-04-26..09-19 except
   one 15-line theory toy (2026-10-04). Eight are cited only by bannered `archive/void/` or `archive_stale/` results. A further 24 are
   SUPERSEDED. Together these 60 files are the safe first archive (Section 4).
2. **268 HISTORICAL files** (113 `.py`, 154 `.sh`, 1 `.slurm`) reproduce numbers in non-citable results files. They are
   the bulk of the clutter; archiving them needs a path map because results files name them (Section 6, stage 3).
3. **`train_variant.py` eagerly imports 63 model modules and loads 24 more through `__import__` in `VARIANT_MAP`.**
   Every one of those 87 modules is therefore CORE or LIVE, legacy InEKF / Level-1.5 / PC / TEM models included, and none
   can be archived without editing `train_variant.py`. Import time is not the issue (0.24 s on top of torch's 0.93 s); the
   coupling is. A lazy registry (Section 6, stage 4) is the single change that unlocks archiving the dead model zoo.
4. **Statistics are already mostly consolidated** (64 files import `stats_core`, 18 use `stats_guard`; 0 call scipy's
   t-tests directly). What remains copied: `holm` (6 copies + 3 importers of `analyze_lrsweep.holm`; `stats_core` has no
   `holm`), a two-sample `mde2` (5 copies; `stats_core` has none), the 2.8-rule `mde` (4 copies), `fisher_state` (3),
   n-gram floors (5 implementations), a B=1 `evaluate()` copied verbatim into 6 trainers, and `build_model` loaders (11+31+4
   copies, all in legacy eval scripts). Section 3.
5. **Every registered batch except the most recent cannot be md5-verified against HEAD**: `train_variant.py` and
   `train.py` differ from the md5 recorded in 41 run dirs each (both are edited to add arms for each new batch). The
   reproducibility anchor of an archived batch is therefore its pre-registration commit, not HEAD; any refactor of a
   CORE module is safe for reproducibility only if that remains true (Section 7.5).
6. **Efficiency**: the torus path was already optimised (2026-09-24 audit: vectorised walk 23x, sync-free loop, vectorised
   strata and permutation tests, `--fast-attn`). Remaining hot spots: `GridWorldND`'s per-step Python walk (0.375 s per
   128x512-step batch vs 0.017 s for the torus fast walk), the text-world renderers (0.20-0.24 s per 128x1024 batch), B=1
   evaluation loops in 45 files (canonical `eval_rank_strata.evaluate` included), and 32 trainers that keep a per-step
   `loss.item()` sync (Section 5).
7. Hygiene: two invalid escape sequences (`model_gated.py` line 21, `analyze_signsep.py` docstring lines 24/101);
   198 unused imports (F401) in 136 files, 23 of them in `__init__.py`'s re-exports; 483 hard-coded
   `/home/prashr/mapformer` occurrences in 421 files, by design (rule 25); 11 `test_*.py`, 10 run and pass on CPU,
   `test_mm.py` needs the Memory Maze data (Section 7).

## 1. Counts

### 1a. Counts per class

| class | .py | .sh | .slurm | total |
|---|---:|---:|---:|---:|
| CORE | 126 | 0 | 0 | 126 |
| LIVE | 353 | 157 | 4 | 514 |
| HISTORICAL | 113 | 154 | 1 | 268 |
| SUPERSEDED | 7 | 16 | 1 | 24 |
| DEAD | 10 | 26 | 0 | 36 |
| **total** | 609 | 353 | 6 | 968 |

### 1b. Counts per location

| location | CORE | LIVE | HISTORICAL | SUPERSEDED | DEAD |
|---|---:|---:|---:|---:|---:|
| `docs/paper/expansion/` | 0 | 63 | 3 | 0 | 0 |
| `docs/paper/tools/` | 0 | 0 | 1 | 0 | 0 |
| `docs/theory/2026-10-04/` | 0 | 7 | 1 | 1 | 1 |
| `docs/tools/` | 0 | 1 | 0 | 0 | 0 |
| `hyak/` | 0 | 6 | 1 | 1 | 0 |
| `top-level` | 126 | 437 | 262 | 22 | 35 |

## 2. Per-file table

Columns: **imp-by** modules that import it (static or dynamic); **drv-by** files whose text launches or names it
(drivers, subprocess strings, md5 lists); **named-in** documents that name it; **links** distinct documents linked by
commit hash, run dir or output file (Section "How the graph was built"); **md5 dirs** run dirs whose `code_md5*.txt`
lists it (and how many of those records differ from HEAD); **last commit** date of the last commit touching it.
Actions: *keep*; *keep (md5-recorded)*: do not edit outside a new series; *archive*: `git mv` to `archive/code/2026-10/`
(stage 1); *archive w/ path map*: stage 3, with an index row because a non-live document names it; *keep or archive w/
md5 note*: HISTORICAL but md5-recorded, so a non-live run dir points at it -- archive only with the run dir's note.
Sorted by class, then path.

| file | class | imp-by | drv-by | named-in | links | md5 dirs | last commit | proposed action | reason |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| `__init__.py` | CORE | 0 | 7 | 4 | 0 | 3 | 2026-04-24 | keep (md5-recorded) | package init: executed by every `import mapformer.X`; imports lie_groups, prefix_scan, environment, model; its 23 F401 hits are the package API |
| `analyze_factorial_n8.py` | CORE | 2 | 2 | 4 | 12 | 3 | 2026-10-24 | keep | imported by 2 (2 live), e.g. analyze_rnn_torus, gate_image_torus |
| `analyze_lrsweep.py` | CORE | 4 | 3 | 7 | 5 | 6 | 2026-10-16 | keep | imported by 4 (4 live), e.g. analyze_exptorus, analyze_rankrecipe, analyze_rnn_torus |
| `analyze_rank_matched.py` | CORE | 2 | 2 | 7 | 6 | 0 | 2026-09-24 | keep | imported by 2 (2 live), e.g. analyze_rank_perhead, analyze_rank_proj |
| `analyze_rankclock.py` | CORE | 2 | 2 | 4 | 4 | 0 | 2026-10-23 | keep | imported by 2 (2 live), e.g. analyze_rankrecipe, analyze_rankwd |
| `analyze_robust.py` | CORE | 5 | 2 | 6 | 5 | 10 | 2026-10-16 | keep | imported by 5 (5 live), e.g. analyze_exptorus, analyze_lrsweep, analyze_signroute |
| `analyze_signroute.py` | CORE | 2 | 1 | 2 | 3 | 3 | 2026-10-21 | keep | imported by 2 (2 live), e.g. analyze_signsep, analyze_ssm_sign |
| `analyze_textworld_secondary.py` | CORE | 3 | 0 | 4 | 8 | 0 | 2026-09-28 | keep | imported by 3 (3 live), e.g. docs/paper/expansion/scripts/language/tw_lib.py, docs/theory/2026-10-04/scripts/tw_latent.py, docs/theory/2026-10-04/scripts/tw_weights.py |
| `analyze_torus_lrsweep.py` | CORE | 2 | 2 | 4 | 4 | 4 | 2026-10-23 | keep | imported by 2 (2 live), e.g. analyze_exptorus, analyze_rankrecipe |
| `ckpt_guard.py` | CORE | 21 | 21 | 6 | 27 | 13 | 2026-09-24 | keep | imported by 21 (18 live), e.g. analyze_monotone, analyze_pairconst, analyze_pairorigin |
| `continuous_nav.py` | CORE | 3 | 1 | 2 | 1 | 0 | 2026-05-01 | keep | imported by 3 (3 live), e.g. eval_continuous, probe_hex_continuous, train_continuous |
| `dyck_mdepth_common.py` | CORE | 4 | 1 | 1 | 1 | 1 | 2026-09-25 | keep | imported by 4 (4 live), e.g. analyze_dyck_mdepth, docs/paper/expansion/scripts/depth/dyck_crosstab.py, docs/paper/expansion/scripts/depth/dyck_distance.py |
| `environment.py` | CORE | 111 | 59 | 22 | 6 | 44 (3 drifted) | 2026-09-24 | keep | imported by 111 (72 live), e.g. __init__, ablate_paper_task, analyze_delta_free |
| `environment_addition.py` | CORE | 6 | 5 | 0 | 1 | 18 | 2026-09-15 | keep | imported by 6 (2 live), e.g. model_coupled_ape, verify_addition_fast |
| `environment_algorithmic.py` | CORE | 2 | 0 | 0 | 0 | 0 | 2026-09-03 | keep | imported by 2 (2 live), e.g. train_algorithmic, validate_algorithmic |
| `environment_cancel.py` | CORE | 21 | 12 | 6 | 1 | 22 | 2026-09-27 | keep | imported by 21 (20 live), e.g. analyze_accounts, analyze_generality, analyze_ring_budget |
| `environment_compositional.py` | CORE | 8 | 0 | 2 | 0 | 0 | 2026-07-23 | keep | imported by 8 (6 live), e.g. ablate_context, eval_compositional, sweep_dissociation |
| `environment_dyck.py` | CORE | 18 | 1 | 0 | 1 | 1 | 2026-09-15 | keep | imported by 18 (17 live), e.g. analyze_cross, analyze_dyck_depth, analyze_dyck_ladder |
| `environment_family_tree.py` | CORE | 4 | 1 | 0 | 0 | 0 | 2026-08-15 | keep | imported by 4 (3 live), e.g. ablate_context, train_family_tree, validate_family_tree |
| `environment_goal.py` | CORE | 14 | 2 | 3 | 0 | 0 | 2026-05-11 | keep | imported by 14 (7 live), e.g. environment_hier_goal, eval_goal_closedloop, probe_active_inference |
| `environment_hier_goal.py` | CORE | 8 | 1 | 1 | 0 | 0 | 2026-08-09 | keep | imported by 8 (1 live), e.g. train_hier_goal |
| `environment_indirect.py` | CORE | 4 | 0 | 0 | 0 | 0 | 2026-09-15 | keep | imported by 4 (2 live), e.g. analyze_indirect, train_indirect |
| `environment_lap.py` | CORE | 5 | 0 | 1 | 0 | 0 | 2026-08-15 | keep | imported by 5 (3 live), e.g. make_figures, probe_lap_theta, validate_lap |
| `environment_match_query.py` | CORE | 7 | 1 | 1 | 0 | 0 | 2026-09-02 | keep | imported by 7 (4 live), e.g. eval_match_longq, loop_depth_strata, train_match_query |
| `environment_nd.py` | CORE | 12 | 15 | 4 | 0 | 24 | 2026-09-04 | keep | imported by 12 (9 live), e.g. analyze_rank_nd_secondary, analyze_rank_wrap, docs/paper/expansion/scripts/rank/init_frame.py |
| `environment_newobj.py` | CORE | 4 | 2 | 5 | 1 | 2 | 2026-10-02 | keep | imported by 4 (4 live), e.g. analyze_leak_secondary, gain_phase_eval, leak_eval |
| `environment_recency.py` | CORE | 9 | 4 | 3 | 5 | 3 | 2026-09-11 | keep | imported by 9 (9 live), e.g. ablate_recency_gate, docs/paper/expansion/scripts/language/crossover_usage.py, eval_clock_side |
| `environment_steps.py` | CORE | 14 | 3 | 3 | 3 | 13 | 2026-10-15 | keep | imported by 14 (14 live), e.g. analyze_gauge, analyze_lrsweep, construct_signroute |
| `environment_stitch.py` | CORE | 4 | 0 | 0 | 0 | 0 | 2026-08-16 | keep | imported by 4 (4 live), e.g. make_figures, probe_stitch_attention, train_stitch |
| `environment_textworld.py` | CORE | 16 | 7 | 5 | 1 | 5 | 2026-09-28 | keep | imported by 16 (14 live), e.g. analyze_textworld_secondary, docs/paper/expansion/scripts/language/tw_lib.py, docs/theory/2026-10-04/scripts/tw_latent.py |
| `environment_textworld_ctx3.py` | CORE | 2 | 2 | 3 | 3 | 1 | 2026-09-30 | keep | imported by 2 (2 live), e.g. environment_tw_ambig, train_ctxstep3 |
| `environment_tw_ambig.py` | CORE | 2 | 1 | 1 | 2 | 0 | 2026-10-08 | keep | imported by 2 (2 live), e.g. train_tw_ambig, tw_ambig_readouts |
| `environment_tw_landmark.py` | CORE | 2 | 2 | 1 | 1 | 1 | 2026-10-08 | keep | imported by 2 (2 live), e.g. train_tw_landmark, tw_landmark_eval |
| `environment_tw_statechange.py` | CORE | 2 | 1 | 1 | 1 | 1 | 2026-10-08 | keep | imported by 2 (2 live), e.g. train_tw_statechange, tw_statechange_readouts |
| `eval_compositional.py` | CORE | 2 | 3 | 2 | 2 | 0 | 2026-08-21 | keep | imported by 2 (2 live), e.g. agg_comp_headroom, agg_comp_multiseed |
| `eval_dyck_literature.py` | CORE | 4 | 1 | 1 | 7 | 1 | 2026-09-19 | keep | imported by 4 (4 live), e.g. analyze_dyck_depth, analyze_dyck_ladder, dyck_mdepth_common |
| `eval_image_torus.py` | CORE | 2 | 4 | 3 | 4 | 2 | 2026-10-25 | keep | imported by 2 (2 live), e.g. gate_image_torus, test_image_torus |
| `eval_noise_refine.py` | CORE | 10 | 45 | 11 | 2 | 17 | 2026-09-24 | keep | imported by 10 (9 live), e.g. analyze_srope_torus, build_torus_exist, construct_rankclock |
| `eval_rank_strata.py` | CORE | 8 | 23 | 6 | 1 | 14 | 2026-09-24 | keep | imported by 8 (8 live), e.g. analyze_factorial_n8, docs/paper/expansion/scripts/depth/torus_depth.py, docs/paper/expansion/scripts/mappope/kernel_probe.py |
| `evaluate.py` | CORE | 2 | 6 | 1 | 0 | 18 | 2026-04-16 | keep | imported by 2 (2 live), e.g. __init__, main |
| `hourglass_plain.py` | CORE | 3 | 5 | 1 | 0 | 18 | 2026-07-23 | keep | imported by 3 (3 live), e.g. probe_stitch_attention, train_hourglass_enwik8, train_variant |
| `image_bank.py` | CORE | 5 | 2 | 3 | 1 | 2 | 2026-10-25 | keep | imported by 5 (5 live), e.g. eval_image_torus, gate_image_torus, model_image |
| `leak_eval.py` | CORE | 3 | 2 | 2 | 1 | 2 | 2026-10-03 | keep | imported by 3 (3 live), e.g. analyze_leak, analyze_leak_secondary, gain_phase_eval |
| `lie_groups.py` | CORE | 3 | 17 | 7 | 0 | 27 | 2026-04-09 | keep | imported by 3 (3 live), e.g. __init__, main, model |
| `minigrid_env.py` | CORE | 6 | 13 | 4 | 1 | 0 | 2026-08-24 | keep | imported by 6 (3 live), e.g. eval_minigrid_2x2, probe_ap_coherence, train_variant |
| `miniworld_env.py` | CORE | 5 | 8 | 2 | 2 | 0 | 2026-08-28 | keep | imported by 5 (3 live), e.g. prebuild_buffers, probe_visits_per_cell, train_miniworld |
| `mm_data.py` | CORE | 3 | 1 | 3 | 3 | 0 | 2026-10-27 | keep | imported by 3 (3 live), e.g. mm_encoder, test_mm, train_mm |
| `mm_gate.py` | CORE | 4 | 5 | 3 | 2 | 0 | 2026-10-26 | keep | imported by 4 (4 live), e.g. mm_data, mm_gate_v2, mm_gate_v3 |
| `mm_gate_v2.py` | CORE | 3 | 4 | 3 | 1 | 0 | 2026-10-27 | keep | imported by 3 (3 live), e.g. mm_data, mm_gate_v3, test_mm |
| `model.py` | CORE | 111 | 68 | 50 | 2 | 54 (3 drifted) | 2026-09-24 | keep | imported by 111 (91 live), e.g. __init__, calibration_analysis, docs/paper/expansion/scripts/language/tw_lib.py |
| `model_ablations.py` | CORE | 8 | 6 | 1 | 0 | 18 | 2026-04-22 | keep | imported by 8 (4 live), e.g. calibration_analysis, long_sequence_eval, train_variant |
| `model_baseline_nope.py` | CORE | 2 | 5 | 2 | 0 | 18 | 2026-08-28 | keep | imported by 2 (2 live), e.g. model_coupled_ape, train_variant |
| `model_baseline_rope.py` | CORE | 25 | 27 | 11 | 1 | 30 | 2026-09-04 | keep | imported by 25 (17 live), e.g. calibration_analysis, eval_long_t, eval_ood_grid |
| `model_baselines_extra.py` | CORE | 12 | 8 | 4 | 0 | 18 | 2026-04-22 | keep | imported by 12 (6 live), e.g. calibration_analysis, long_sequence_eval, test_rnn_torus |
| `model_cho_coupled.py` | CORE | 2 | 5 | 1 | 0 | 18 | 2026-09-14 | keep | imported by 2 (2 live), e.g. model_cho_positions, train_variant |
| `model_codes.py` | CORE | 10 | 8 | 6 | 2 | 6 | 2026-10-03 | keep | imported by 10 (10 live), e.g. analyze_leak_secondary, gain_phase_eval, leak_eval |
| `model_context_step.py` | CORE | 4 | 2 | 4 | 2 | 1 | 2026-09-30 | keep | imported by 4 (2 live), e.g. model_tw_ambig, train_ctxstep3 |
| `model_continuous.py` | CORE | 3 | 0 | 2 | 1 | 0 | 2026-05-01 | keep | imported by 3 (3 live), e.g. eval_continuous, probe_hex_continuous, train_continuous |
| `model_em_fixed.py` | CORE | 6 | 8 | 1 | 1 | 18 | 2026-09-08 | keep | imported by 6 (6 live), e.g. model_counter_installed, model_em_pairorigin, model_em_unfreeze |
| `model_em_pairorigin.py` | CORE | 2 | 5 | 1 | 2 | 18 | 2026-09-11 | keep | imported by 2 (2 live), e.g. model_em_pairconst, train_variant |
| `model_em_pope.py` | CORE | 4 | 2 | 0 | 1 | 2 | 2026-10-05 | keep | imported by 4 (4 live), e.g. eval_gain_grain, gain_phase_eval, model_gain_phase |
| `model_em_unfreeze.py` | CORE | 2 | 6 | 0 | 1 | 18 | 2026-09-10 | keep | imported by 2 (2 live), e.g. model_em_noleak, train_variant |
| `model_em_warm.py` | CORE | 4 | 5 | 1 | 0 | 18 | 2026-09-10 | keep | imported by 4 (3 live), e.g. model_em_noleak, model_em_unfreeze, train_variant |
| `model_exptorus.py` | CORE | 4 | 1 | 2 | 1 | 3 | 2026-10-26 | keep | imported by 4 (4 live), e.g. analyze_exptorus, eval_exptorus, test_exptorus |
| `model_gain_phase.py` | CORE | 2 | 1 | 1 | 2 | 1 | 2026-10-06 | keep | imported by 2 (2 live), e.g. gain_phase_eval, train_gain_phase |
| `model_grid.py` | CORE | 4 | 6 | 4 | 0 | 18 | 2026-04-26 | keep | imported by 4 (2 live), e.g. model_grid_l15_pc, train_variant |
| `model_hier_attn.py` | CORE | 4 | 6 | 1 | 0 | 18 | 2026-07-18 | keep | imported by 4 (4 live), e.g. long_sequence_eval, model_recursive, model_spacetime_hier |
| `model_hourglass.py` | CORE | 5 | 5 | 2 | 2 | 18 | 2026-09-03 | keep | imported by 5 (4 live), e.g. hourglass_plain, model_pope, test_hourglass_causal |
| `model_image.py` | CORE | 4 | 2 | 3 | 2 | 2 | 2026-10-25 | keep | imported by 4 (4 live), e.g. eval_image_torus, gate_image_torus, test_image_torus |
| `model_inekf_cascade.py` | CORE | 4 | 5 | 2 | 1 | 18 | 2026-07-15 | keep | imported by 4 (3 live), e.g. long_sequence_eval, train_variant, zero_shot_eval |
| `model_inekf_gsf.py` | CORE | 5 | 8 | 2 | 0 | 18 | 2026-05-10 | keep | imported by 5 (5 live), e.g. eval_long_t, model_inekf_gsf_modeomega, model_inekf_gsf_nodrop |
| `model_inekf_gsf_nodrop.py` | CORE | 7 | 7 | 3 | 0 | 18 | 2026-05-15 | keep | imported by 7 (5 live), e.g. eval_long_t, eval_ood_grid, probe_hex_emergence |
| `model_inekf_level15.py` | CORE | 46 | 33 | 3 | 0 | 18 | 2026-04-24 | keep | imported by 46 (32 live), e.g. calibration_analysis, eval_long_t, eval_ood_grid |
| `model_inekf_level15_em.py` | CORE | 12 | 13 | 3 | 0 | 18 | 2026-04-24 | keep | imported by 12 (6 live), e.g. calibration_analysis, long_sequence_eval, model_inekf_level15_em_perscale |
| `model_inekf_level15_hopfield.py` | CORE | 2 | 6 | 0 | 0 | 18 | 2026-05-18 | keep | imported by 2 (2 live), e.g. model_inekf_level15_hopfield_nomainap, train_variant |
| `model_inekf_level15_nodrop.py` | CORE | 6 | 12 | 3 | 0 | 18 | 2026-05-10 | keep | imported by 6 (6 live), e.g. eval_long_t, model_inekf_gsf_modeomega, model_inekf_gsf_nodrop |
| `model_inekf_level2.py` | CORE | 11 | 7 | 1 | 0 | 18 | 2026-04-18 | keep | imported by 11 (4 live), e.g. calibration_analysis, long_sequence_eval, train_variant |
| `model_inekf_parallel.py` | CORE | 15 | 8 | 2 | 0 | 18 | 2026-04-16 | keep | imported by 15 (5 live), e.g. calibration_analysis, long_sequence_eval, model_predictive_coding |
| `model_level15_dog.py` | CORE | 2 | 6 | 3 | 2 | 18 | 2026-05-01 | keep | imported by 2 (2 live), e.g. probe_hex, train_variant |
| `model_level15_pc.py` | CORE | 11 | 10 | 2 | 0 | 18 | 2026-04-26 | keep | imported by 11 (4 live), e.g. calibration_analysis, long_sequence_eval, train_variant |
| `model_level15_pc_v2.py` | CORE | 10 | 8 | 2 | 0 | 18 | 2026-04-27 | keep | imported by 10 (6 live), e.g. calibration_analysis, long_sequence_eval, model_level15_pc_v3 |
| `model_looped.py` | CORE | 3 | 11 | 3 | 3 | 21 | 2026-09-26 | keep | imported by 3 (3 live), e.g. docs/paper/expansion/scripts/recurrence/grad_at_init.py, model_rank, train_variant |
| `model_mm.py` | CORE | 2 | 2 | 3 | 4 | 0 | 2026-10-27 | keep | imported by 2 (2 live), e.g. test_mm, train_mm |
| `model_pope.py` | CORE | 16 | 10 | 9 | 2 | 21 | 2026-09-07 | keep | imported by 16 (16 live), e.g. analyze_delta_free, analyze_gain_grain, analyze_score_rank |
| `model_pope_decay.py` | CORE | 3 | 6 | 4 | 3 | 18 | 2026-09-20 | keep | imported by 3 (3 live), e.g. train_dyck, train_jsb, train_variant |
| `model_pope_delta.py` | CORE | 2 | 1 | 2 | 2 | 13 | 2026-10-20 | keep | imported by 2 (2 live), e.g. analyze_delta_free, train_variant |
| `model_pope_pair.py` | CORE | 7 | 3 | 2 | 2 | 3 | 2026-10-04 | keep | imported by 7 (7 live), e.g. docs/paper/expansion/scripts/mappope/kernel_probe.py, eval_pair, eval_score_rank |
| `model_pope_pair_mi.py` | CORE | 3 | 1 | 0 | 2 | 1 | 2026-10-05 | keep | imported by 3 (3 live), e.g. docs/paper/expansion/scripts/mappope/kernel_probe.py, eval_score_rank, train_score_rank |
| `model_pope_t3.py` | CORE | 5 | 5 | 3 | 2 | 18 | 2026-09-19 | keep | imported by 5 (5 live), e.g. model_dyck_monotone, train_dyck, train_indirect |
| `model_predictive_coding.py` | CORE | 14 | 8 | 2 | 0 | 18 | 2026-04-17 | keep | imported by 14 (4 live), e.g. calibration_analysis, long_sequence_eval, train_variant |
| `model_rank.py` | CORE | 5 | 21 | 6 | 1 | 32 | 2026-09-04 | keep | imported by 5 (5 live), e.g. model_codes, model_counter_installed, model_gain_phase |
| `model_rank_perhead.py` | CORE | 5 | 15 | 6 | 4 | 27 (3 drifted) | 2026-09-27 | keep | imported by 5 (3 live), e.g. docs/paper/expansion/scripts/rank/train_nd_cpu.py, model_ring_rank, train_variant |
| `model_rankclock.py` | CORE | 4 | 2 | 1 | 1 | 10 | 2026-10-23 | keep | imported by 4 (4 live), e.g. analyze_rankclock, construct_rankclock, test_rankclock |
| `model_recursive.py` | CORE | 2 | 5 | 0 | 0 | 18 | 2026-07-22 | keep | imported by 2 (2 live), e.g. model_spacetime_hier, train_variant |
| `model_rnn.py` | CORE | 4 | 2 | 2 | 3 | 1 | 2026-10-26 | keep | imported by 4 (4 live), e.g. analyze_rnn_torus, eval_rnn_torus, test_rnn_torus |
| `model_selective.py` | CORE | 3 | 7 | 6 | 1 | 19 | 2026-09-04 | keep | imported by 3 (2 live), e.g. model_monotone, train_variant |
| `model_sign.py` | CORE | 4 | 22 | 16 | 0 | 33 | 2026-09-06 | keep | imported by 4 (4 live), e.g. model_dyck_monotone, model_monotone, model_wrap_exist |
| `model_signroute.py` | CORE | 8 | 4 | 3 | 2 | 8 | 2026-10-20 | keep | imported by 8 (8 live), e.g. construct_signroute, model_exptorus, model_signsep |
| `model_signsep.py` | CORE | 3 | 3 | 1 | 3 | 3 (1 drifted) | 2026-10-23 | keep | imported by 3 (3 live), e.g. construct_signsep, test_signsep, train_signsep |
| `model_ssm_sign.py` | CORE | 3 | 1 | 4 | 1 | 2 | 2026-10-23 | keep | imported by 3 (3 live), e.g. construct_ssm_sign, test_ssm_sign, train_ssm_sign |
| `model_tem_faithful.py` | CORE | 8 | 15 | 3 | 1 | 18 | 2026-05-07 | keep | imported by 8 (7 live), e.g. eval_long_t, eval_ood_grid, model_tem_ffn |
| `model_textstep.py` | CORE | 5 | 2 | 3 | 2 | 2 | 2026-10-03 | keep | imported by 5 (5 live), e.g. docs/paper/expansion/scripts/language/tw_lib.py, train_tw_normstep, train_tw_statechange |
| `model_tw_ambig.py` | CORE | 2 | 1 | 1 | 2 | 0 | 2026-10-08 | keep | imported by 2 (2 live), e.g. train_tw_ambig, tw_ambig_readouts |
| `model_wrap_exist.py` | CORE | 24 | 11 | 16 | 7 | 23 (3 drifted) | 2026-10-14 | keep | imported by 24 (24 live), e.g. analyze_accounts, analyze_exptorus, analyze_gauge |
| `prefix_scan.py` | CORE | 2 | 6 | 2 | 0 | 18 | 2026-04-09 | keep | imported by 2 (2 live), e.g. __init__, main |
| `probe_anatomy.py` | CORE | 6 | 0 | 0 | 15 | 0 | 2026-09-12 | keep | imported by 6 (6 live), e.g. analyze_monotone, analyze_pairorigin, analyze_search |
| `probe_dyck_stack.py` | CORE | 6 | 2 | 0 | 6 | 1 | 2026-09-21 | keep | imported by 6 (5 live), e.g. analyze_cross, dyck_standard_metrics, eval_dyck_literature |
| `probe_rewind.py` | CORE | 9 | 0 | 1 | 17 | 0 | 2026-09-11 | keep | imported by 9 (9 live), e.g. analyze_monotone, analyze_pairorigin, analyze_search |
| `rescore_hook.py` | CORE | 8 | 13 | 11 | 0 | 10 (8 drifted) | 2026-10-14 | keep | imported by 8 (7 live), e.g. analyze_generality, analyze_ring_budget, analyze_ring_dial |
| `stats_core.py` | CORE | 72 | 28 | 17 | 1 | 38 | 2026-09-24 | keep | imported by 72 (68 live), e.g. analyze_accounts, analyze_cancel, analyze_clock_side |
| `stats_guard.py` | CORE | 15 | 0 | 2 | 2 | 0 | 2026-09-24 | keep | imported by 15 (11 live), e.g. analyze_monotone, analyze_pairconst, analyze_pairorigin |
| `train.py` | CORE | 27 | 73 | 37 | 6 | 53 (41 drifted) | 2026-10-23 | keep | imported by 27 (23 live), e.g. __init__, main, test_epoch_callback |
| `train_cancel.py` | CORE | 19 | 12 | 7 | 3 | 17 (1 drifted) | 2026-09-30 | keep | imported by 19 (19 live), e.g. analyze_generality, analyze_ring_budget, analyze_ring_dial |
| `train_dyck.py` | CORE | 14 | 13 | 8 | 2 | 1 | 2026-09-25 | keep | imported by 14 (13 live), e.g. analyze_cross, analyze_dyck_depth, analyze_dyck_ladder |
| `train_hourglass_enwik8.py` | CORE | 7 | 15 | 3 | 1 | 0 | 2026-09-21 | keep | imported by 7 (7 live), e.g. check_bf16, docs/paper/expansion/scripts/language/tw_alpha_drift.py, docs/paper/expansion/scripts/language/usage_tick.py |
| `train_image.py` | CORE | 2 | 3 | 3 | 4 | 2 | 2026-10-25 | keep | imported by 2 (2 live), e.g. gate_image_torus, test_image_torus |
| `train_match_query.py` | CORE | 4 | 13 | 3 | 0 | 0 | 2026-09-02 | keep | imported by 4 (2 live), e.g. eval_match_longq, loop_depth_strata |
| `train_newobj.py` | CORE | 2 | 2 | 1 | 1 | 2 | 2026-10-03 | keep | imported by 2 (2 live), e.g. leak_eval, train_gain_phase |
| `train_score_rank.py` | CORE | 2 | 1 | 1 | 2 | 1 | 2026-10-05 | keep | imported by 2 (2 live), e.g. analyze_score_rank, docs/paper/expansion/scripts/rank/sweep_final.py |
| `train_textworld.py` | CORE | 2 | 6 | 6 | 3 | 3 (1 drifted) | 2026-09-30 | keep | imported by 2 (2 live), e.g. train_tw_normstep, train_tw_statechange |
| `train_tinystories.py` | CORE | 2 | 1 | 2 | 2 | 0 | 2026-10-09 | keep | imported by 2 (2 live), e.g. docs/paper/expansion/scripts/language/tw_alpha_drift.py, docs/paper/expansion/scripts/language/usage_tick.py |
| `train_variant.py` | CORE | 160 | 180 | 46 | 30 | 51 (41 drifted) | 2026-10-23 | keep | imported by 160 (118 live), e.g. ablate_context, ablate_paper_task, analyze_delta_free |
| `train_wrap_exist.py` | CORE | 2 | 2 | 7 | 6 | 2 (1 drifted) | 2026-10-14 | keep | imported by 2 (2 live), e.g. analyze_accounts, analyze_wrap_existence |
| `train_wrap_mech.py` | CORE | 4 | 4 | 8 | 11 | 5 (1 drifted) | 2026-10-14 | keep | imported by 4 (4 live), e.g. docs/paper/expansion/scripts/sign/path_profiles.py, docs/paper/expansion/scripts/sign/step_basin.py, model_signsep |
| `tw_normstep_readouts.py` | CORE | 2 | 1 | 3 | 2 | 1 | 2026-10-03 | keep | imported by 2 (2 live), e.g. analyze_tw_normstep, docs/paper/expansion/scripts/language/tw_lib.py |
| `validate_dyck.py` | CORE | 5 | 1 | 2 | 1 | 1 | 2026-09-15 | keep | imported by 5 (5 live), e.g. dyck_standard_metrics, eval_dyck_literature, make_dyck_table |
| `ablate_context.py` | LIVE | 0 | 0 | 0 | 4 | 0 | 2026-08-17 | keep | reads/writes FAMILY_TREE_RESULTS.md |
| `ablate_paper_task.py` | LIVE | 0 | 2 | 0 | 2 | 0 | 2026-08-17 | keep | dependency of run_correction_gaps.sh |
| `ablate_recency_gate.py` | LIVE | 0 | 0 | 0 | 5 | 0 | 2026-09-07 | keep | reads/writes RECENCY_GATE_ABLATION.md |
| `agg_alias.py` | LIVE | 0 | 2 | 1 | 1 | 0 | 2026-08-29 | keep | dependency of run_alias_sweep.sh |
| `agg_comp_headroom.py` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | keep | reads/writes COMP_HEADROOM.md |
| `agg_comp_multiseed.py` | LIVE | 0 | 15 | 3 | 2 | 0 | 2026-09-24 | keep | named in COMPOSITIONAL_EXPERIMENT.md |
| `agg_hiergoal.py` | LIVE | 3 | 10 | 0 | 0 | 0 | 2026-09-24 | keep | dependency of run_pope_redo.sh |
| `agg_l15_loop.py` | LIVE | 0 | 2 | 0 | 0 | 0 | 2026-09-01 | keep | dependency of run_l15_loop_2x2.sh |
| `agg_lambda_trace.py` | LIVE | 0 | 1 | 0 | 0 | 0 | 2026-09-05 | keep | dependency of run_lambda_trace.sh |
| `agg_loop_headroom.py` | LIVE | 0 | 3 | 1 | 0 | 0 | 2026-08-30 | keep | dependency of run_loop_headroom.sh |
| `agg_miniworld.py` | LIVE | 0 | 5 | 2 | 1 | 0 | 2026-08-28 | keep | dependency of run_alias_sweep.sh |
| `agg_mq_rank.py` | LIVE | 0 | 1 | 1 | 0 | 0 | 2026-09-04 | keep | dependency of run_mq_rank.sh |
| `agg_popewrap.py` | LIVE | 0 | 1 | 0 | 1 | 0 | 2026-09-05 | keep | reads/writes ALIASING_CONTROLLED.md |
| `agg_refine.py` | LIVE | 0 | 1 | 2 | 0 | 0 | 2026-08-31 | keep | dependency of run_refine.sh |
| `agg_rope_canonical.py` | LIVE | 0 | 1 | 0 | 0 | 0 | 2026-09-03 | keep | dependency of run_rope_canonical.sh |
| `agg_selective.py` | LIVE | 0 | 1 | 0 | 0 | 0 | 2026-09-03 | keep | dependency of run_selective.sh |
| `agg_sign.py` | LIVE | 0 | 2 | 0 | 1 | 0 | 2026-09-06 | keep | reads/writes SIGN_ABLATION_PREREG.md |
| `agg_visits.py` | LIVE | 0 | 1 | 0 | 0 | 0 | 2026-08-30 | keep | dependency of run_visits_test.sh |
| `analyze_accounts.py` | LIVE | 0 | 1 | 2 | 12 | 1 | 2026-10-14 | keep (md5-recorded) | commit cited in ACCOUNTS_RESULTS.md |
| `analyze_aug.py` | LIVE | 0 | 0 | 0 | 1 | 0 | 2026-09-20 | keep | reads/writes AUG_RESULTS.md |
| `analyze_cancel.py` | LIVE | 0 | 1 | 3 | 6 | 0 | 2026-09-27 | keep | named in CANCEL_RESULTS.md |
| `analyze_clock_side.py` | LIVE | 0 | 1 | 3 | 4 | 1 (1 drifted) | 2026-10-12 | keep (md5-recorded) | commit cited in docs/audits/2026-10-12/ring_dial_audit.md |
| `analyze_code_fullval.py` | LIVE | 0 | 1 | 1 | 16 | 0 | 2026-09-27 | keep | named in CODE_FULLVAL_RESULTS.md |
| `analyze_cross.py` | LIVE | 0 | 0 | 1 | 1 | 0 | 2026-09-20 | keep | reads/writes CROSS_RESULTS.md |
| `analyze_delta_free.py` | LIVE | 0 | 2 | 3 | 3 | 2 (1 drifted) | 2026-10-20 | keep (md5-recorded) | named in DELTA_FREE_PREREG.md |
| `analyze_dyck.py` | LIVE | 0 | 2 | 1 | 2 | 0 | 2026-09-15 | keep | dependency of run_dyck.sh |
| `analyze_dyck_depth.py` | LIVE | 0 | 0 | 2 | 9 | 0 | 2026-09-21 | keep | run dir in DYCK_LADDER_PREREG.md |
| `analyze_dyck_ladder.py` | LIVE | 0 | 0 | 2 | 7 | 0 | 2026-09-24 | keep | run dir in DYCK_MDEPTH_PREREG.md |
| `analyze_dyck_mdepth.py` | LIVE | 0 | 1 | 2 | 12 | 1 | 2026-09-25 | keep (md5-recorded) | named in DYCK_MDEPTH_PREREG.md |
| `analyze_exptorus.py` | LIVE | 1 | 1 | 3 | 4 | 3 | 2026-10-26 | keep (md5-recorded) | named in EXPTORUS_PREREG.md |
| `analyze_factorial.py` | LIVE | 1 | 2 | 6 | 8 | 4 | 2026-10-12 | keep (md5-recorded) | named in FACTORIAL_N8_PREREG.md |
| `analyze_gain_grain.py` | LIVE | 0 | 1 | 3 | 5 | 1 | 2026-10-05 | keep (md5-recorded) | named in DELTA_FREE_PREREG.md |
| `analyze_gain_phase.py` | LIVE | 1 | 1 | 2 | 2 | 1 | 2026-10-06 | keep (md5-recorded) | named in GAIN_PHASE_PREREG.md |
| `analyze_gauge.py` | LIVE | 0 | 2 | 5 | 5 | 1 | 2026-10-15 | keep (md5-recorded) | named in GAUGE_PREREG.md |
| `analyze_generality.py` | LIVE | 0 | 1 | 4 | 12 | 1 | 2026-10-14 | keep (md5-recorded) | named in GENERALITY_PREREG.md |
| `analyze_image_torus.py` | LIVE | 1 | 1 | 3 | 4 | 1 | 2026-10-25 | keep (md5-recorded) | named in IMAGE_TORUS_PREREG.md |
| `analyze_indirect.py` | LIVE | 0 | 3 | 0 | 1 | 0 | 2026-09-16 | keep | dependency of run_indirect.sh |
| `analyze_jsb.py` | LIVE | 0 | 1 | 0 | 1 | 0 | 2026-09-15 | keep | reads/writes JSB_PREREG.md |
| `analyze_leak.py` | LIVE | 0 | 1 | 2 | 6 | 1 | 2026-10-03 | keep (md5-recorded) | named in LEAK_RESULTS.md |
| `analyze_leak_secondary.py` | LIVE | 0 | 0 | 3 | 6 | 0 | 2026-10-03 | keep | named in LEAK_PREREG.md |
| `analyze_loop_rank.py` | LIVE | 0 | 1 | 2 | 22 | 0 | 2026-09-26 | keep | named in LOOP_RANK_RESULTS.md |
| `analyze_loop_rank_e1800.py` | LIVE | 0 | 1 | 1 | 3 | 0 | 2026-09-27 | keep | run dir in LOOP_RANK_E1800_P1_RESULTS.md |
| `analyze_loop_rank_e1800_budget.py` | LIVE | 0 | 1 | 2 | 21 | 0 | 2026-09-28 | keep | named in LOOP_RANK_E1800_P1_RESULTS.md |
| `analyze_mappope_pair.py` | LIVE | 0 | 1 | 1 | 4 | 1 | 2026-10-04 | keep (md5-recorded) | named in MAPPOPE_PAIR_RESULTS.md |
| `analyze_mm.py` | LIVE | 0 | 1 | 2 | 4 | 0 | 2026-10-27 | keep | named in MEMORY_MAZE_PREREG.md |
| `analyze_mm_pilot.py` | LIVE | 0 | 2 | 1 | 2 | 0 | 2026-10-27 | keep | named in docs/audits/2026-10-27/memory_maze_audit.md |
| `analyze_monotone.py` | LIVE | 0 | 1 | 2 | 10 | 0 | 2026-09-13 | keep | run dir in PAPER2X2_PREREG.md |
| `analyze_pairconst.py` | LIVE | 0 | 0 | 0 | 6 | 0 | 2026-09-12 | keep | run dir in PAIRSPLIT_PREREG.md |
| `analyze_pairorigin.py` | LIVE | 0 | 0 | 1 | 5 | 0 | 2026-09-11 | keep | run dir in PAIRSPLIT_PREREG.md |
| `analyze_pairsplit.py` | LIVE | 0 | 0 | 0 | 21 | 0 | 2026-09-12 | keep | run dir in EM_WM_STATE.md |
| `analyze_paper2x2.py` | LIVE | 0 | 1 | 1 | 20 | 0 | 2026-09-13 | keep | run dir in MAPPOPE_PAIR_PREREG.md |
| `analyze_rank3.py` | LIVE | 0 | 1 | 2 | 24 | 0 | 2026-09-27 | keep | named in RANK3_RESULTS.md |
| `analyze_rank_mi.py` | LIVE | 0 | 1 | 2 | 30 | 0 | 2026-09-24 | keep | named in RANK_MI_RESULTS.md |
| `analyze_rank_nd.py` | LIVE | 0 | 2 | 3 | 7 | 0 | 2026-09-30 | keep | named in RANK_ND_PREREG.md |
| `analyze_rank_nd_secondary.py` | LIVE | 1 | 1 | 3 | 7 | 1 | 2026-09-30 | keep (md5-recorded) | named in RANK_ND_PREREG.md |
| `analyze_rank_perhead.py` | LIVE | 0 | 1 | 1 | 15 | 0 | 2026-09-24 | keep | run dir in .claude-memory/project_state.md |
| `analyze_rank_proj.py` | LIVE | 0 | 1 | 2 | 3 | 0 | 2026-09-24 | keep | named in RANK_PROJ_RESULTS.md |
| `analyze_rank_sep.py` | LIVE | 0 | 1 | 2 | 21 | 0 | 2026-09-25 | keep | named in RANK_SEP_RESULTS.md |
| `analyze_rank_tests.py` | LIVE | 1 | 2 | 3 | 22 | 2 | 2026-10-11 | keep (md5-recorded) | named in RANK_TESTS_PREREG.md |
| `analyze_rank_wrap.py` | LIVE | 0 | 1 | 3 | 10 | 1 | 2026-10-01 | keep (md5-recorded) | named in RANK_WRAP_PREREG.md |
| `analyze_rank_wrap_secondary.py` | LIVE | 0 | 0 | 2 | 5 | 0 | 2026-10-01 | keep | named in RANK_WRAP_PREREG.md |
| `analyze_rankrecipe.py` | LIVE | 0 | 1 | 2 | 2 | 0 | 2026-10-24 | keep | named in RANKRECIPE_PREREG.md |
| `analyze_rankwd.py` | LIVE | 0 | 1 | 2 | 4 | 0 | 2026-10-23 | keep | named in RANKWD_PREREG.md |
| `analyze_ring_budget.py` | LIVE | 0 | 1 | 3 | 3 | 1 | 2026-10-12 | keep (md5-recorded) | dependency of run_ring_budget.sh |
| `analyze_ring_dial.py` | LIVE | 1 | 2 | 3 | 7 | 2 | 2026-10-12 | keep (md5-recorded) | named in docs/audits/2026-10-12/ring_dial_audit.md |
| `analyze_rnn_torus.py` | LIVE | 1 | 1 | 2 | 6 | 1 | 2026-10-26 | keep (md5-recorded) | named in RNN_TORUS_PREREG.md |
| `analyze_score_rank.py` | LIVE | 0 | 2 | 2 | 22 | 2 | 2026-10-05 | keep (md5-recorded) | named in DELTA_FREE_PREREG.md |
| `analyze_search.py` | LIVE | 0 | 0 | 0 | 15 | 0 | 2026-09-11 | keep | run dir in EM_WM_STATE.md |
| `analyze_sign_matched.py` | LIVE | 0 | 1 | 2 | 10 | 0 | 2026-09-27 | keep | named in SIGN_MATCHED_RESULTS.md |
| `analyze_signsep.py` | LIVE | 0 | 1 | 2 | 4 | 1 | 2026-10-23 | keep (md5-recorded) | named in SIGNSEP_PREREG.md; INVALID ESCAPE (DeprecationWarning on 3.10, SyntaxWarning on 3.12) |
| `analyze_spread.py` | LIVE | 0 | 0 | 0 | 15 | 0 | 2026-09-11 | keep | run dir in EM_WM_STATE.md |
| `analyze_srope_torus.py` | LIVE | 0 | 1 | 2 | 2 | 1 | 2026-10-20 | keep (md5-recorded) | named in SROPE_TORUS_PREREG.md |
| `analyze_ssm_sign.py` | LIVE | 0 | 1 | 6 | 7 | 2 (1 drifted) | 2026-10-24 | keep (md5-recorded) | named in FACTORIAL_N8_PREREG.md |
| `analyze_textworld.py` | LIVE | 0 | 1 | 2 | 7 | 0 | 2026-09-28 | keep | named in TEXTWORLD_RESULTS.md |
| `analyze_torus_start.py` | LIVE | 0 | 1 | 3 | 3 | 1 | 2026-10-14 | keep (md5-recorded) | named in TORUS_START_PREREG.md |
| `analyze_tw_ambig.py` | LIVE | 0 | 1 | 0 | 2 | 0 | 2026-10-08 | keep | commit cited in .claude-memory/project_state.md |
| `analyze_tw_landmark.py` | LIVE | 0 | 2 | 0 | 2 | 1 | 2026-10-08 | keep (md5-recorded) | commit cited in .claude-memory/project_state.md |
| `analyze_tw_normstep.py` | LIVE | 0 | 1 | 3 | 7 | 1 | 2026-10-03 | keep (md5-recorded) | named in TW_NORMSTEP_PREREG.md |
| `analyze_tw_statechange.py` | LIVE | 0 | 1 | 1 | 2 | 1 | 2026-10-08 | keep (md5-recorded) | named in TW_STATECHANGE_PREREG.md |
| `analyze_wrap_cope.py` | LIVE | 0 | 1 | 3 | 4 | 1 | 2026-10-14 | keep (md5-recorded) | dependency of run_wrap_cope.sh |
| `analyze_wrap_existence.py` | LIVE | 0 | 1 | 4 | 8 | 1 | 2026-10-13 | keep (md5-recorded) | named in WRAP_EXISTENCE_PREREG.md |
| `analyze_wrap_mech.py` | LIVE | 1 | 2 | 4 | 9 | 2 | 2026-10-14 | keep (md5-recorded) | named in MECHANISM_PREREG.md |
| `audit_headline_numbers.py` | LIVE | 0 | 0 | 0 | 5 | 0 | 2026-08-23 | keep | reads/writes BASELINE_TABLE.md |
| `benchmark_timing.py` | LIVE | 0 | 1 | 0 | 1 | 0 | 2026-08-16 | keep | reads/writes TIMING_BENCHMARK.md |
| `build_rank_lift.py` | LIVE | 0 | 1 | 2 | 21 | 1 | 2026-10-11 | keep (md5-recorded) | named in RANK_TESTS_PREREG.md |
| `build_rank_proj.py` | LIVE | 0 | 0 | 2 | 17 | 0 | 2026-09-24 | keep | named in RANK_PROJ_PREREG.md |
| `build_torus_exist.py` | LIVE | 0 | 1 | 4 | 4 | 1 | 2026-10-14 | keep (md5-recorded) | named in GENERALITY_PREREG.md |
| `calibration_analysis.py` | LIVE | 0 | 2 | 3 | 0 | 0 | 2026-04-27 | keep | named in README.md |
| `check_bf16.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-09-23 | keep | named in BF16_RESULTS.md |
| `construct_rankclock.py` | LIVE | 0 | 2 | 1 | 4 | 0 | 2026-10-23 | keep | named in RANKCLOCK_PREREG.md |
| `construct_signroute.py` | LIVE | 0 | 3 | 1 | 3 | 0 | 2026-10-20 | keep | named in SIGNROUTE_PREREG.md |
| `construct_signsep.py` | LIVE | 1 | 3 | 1 | 7 | 2 (1 drifted) | 2026-10-23 | keep (md5-recorded) | named in SIGNSEP_PREREG.md |
| `construct_ssm_sign.py` | LIVE | 1 | 2 | 3 | 6 | 2 | 2026-10-23 | keep (md5-recorded) | named in G3_SSM_PREREG.md |
| `data_parallel.py` | LIVE | 2 | 53 | 11 | 0 | 50 | 2026-09-01 | keep (md5-recorded) | named in RNN_TORUS_PREREG.md |
| `docs/paper/expansion/scripts/depth/dyck_crosstab.py` | LIVE | 0 | 0 | 3 | 6 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/dyck_distance.py` | LIVE | 0 | 1 | 3 | 6 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/ring_class_floors.py` | LIVE | 0 | 0 | 2 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/ring_depth.py` | LIVE | 0 | 7 | 5 | 6 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/ring_plain_bins.py` | LIVE | 0 | 2 | 2 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/ring_probe_window.py` | LIVE | 0 | 0 | 2 | 6 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/summarize_ring.py` | LIVE | 0 | 2 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/summarize_torus.py` | LIVE | 0 | 1 | 4 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/torus_depth.py` | LIVE | 0 | 3 | 6 | 7 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/torus_probe.py` | LIVE | 0 | 0 | 4 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/depth/torus_runlen.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/crossover_usage.py` | LIVE | 0 | 0 | 4 | 12 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_alpha_drift.py` | LIVE | 0 | 1 | 3 | 12 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_attention_asides.py` | LIVE | 0 | 1 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_attention_asides_dironly.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_clock_unit.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_common_used.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_lib.py` | LIVE | 0 | 0 | 3 | 11 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_retrace_floor.py` | LIVE | 0 | 2 | 4 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_step_account.py` | LIVE | 0 | 3 | 4 | 11 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_strata.py` | LIVE | 0 | 1 | 4 | 7 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/language/tw_strata_200.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-21 | keep | named in docs/audits/2026-10-28/FULL_REVIEW.md |
| `docs/paper/expansion/scripts/language/usage_tick.py` | LIVE | 0 | 1 | 4 | 12 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/mappope/kernel_argmax_odd.py` | LIVE | 0 | 0 | 2 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/mappope/kernel_probe.py` | LIVE | 0 | 2 | 6 | 15 | 0 | 2026-10-21 | keep | named in DELTA_FREE_PREREG.md |
| `docs/paper/expansion/scripts/mappope/probe_stats.py` | LIVE | 0 | 0 | 4 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/analyze_snap.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/c1_runlevel.py` | LIVE | 0 | 1 | 4 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/c1_runlevel_once.py` | LIVE | 0 | 0 | 2 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/clock_decomp.py` | LIVE | 0 | 0 | 2 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/escape_start.py` | LIVE | 0 | 0 | 4 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/grad_hiding.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/init_frame.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/rank_geom.py` | LIVE | 0 | 1 | 4 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/summarise_final.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/sweep_final.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/train_nd_cpu.py` | LIVE | 0 | 1 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/rank/train_snap.py` | LIVE | 0 | 1 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/attn_offsets.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/common.py` | LIVE | 0 | 1 | 5 | 23 | 0 | 2026-10-20 | keep | named in DYCK_MDEPTH_PREREG.md |
| `docs/paper/expansion/scripts/recurrence/geometry.py` | LIVE | 0 | 4 | 12 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/grad_at_init.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/index_loop_passes.py` | LIVE | 0 | 0 | 3 | 7 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/pass_strata.py` | LIVE | 0 | 1 | 4 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/pass_strata_besthead.py` | LIVE | 0 | 0 | 2 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/phase_code.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/phase_reader_soft.py` | LIVE | 0 | 1 | 4 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/reader_mde.py` | LIVE | 0 | 0 | 2 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/recurrence/summarize.py` | LIVE | 0 | 0 | 2 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/expansion/RECURRENCE.md |
| `docs/paper/expansion/scripts/recurrence/training_curves.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/aobal_geometry.py` | LIVE | 0 | 0 | 3 | 6 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/basin_summary.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/clock_table.py` | LIVE | 0 | 3 | 5 | 0 | 0 | 2026-10-21 | keep | named in TORUS_LRSWEEP_PREREG.md |
| `docs/paper/expansion/scripts/sign/latent_geometry.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/latent_reach.py` | LIVE | 0 | 0 | 3 | 6 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/lattice_p.py` | LIVE | 0 | 0 | 3 | 3 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/path_profiles.py` | LIVE | 0 | 0 | 4 | 6 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/prop_checks.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/rank_clock.py` | LIVE | 0 | 2 | 4 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/run_basin.sh` | LIVE | 0 | 0 | 2 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/expansion/SEARCH_THEORY.md |
| `docs/paper/expansion/scripts/sign/sign_clock.py` | LIVE | 0 | 2 | 4 | 0 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/sign/step_basin.py` | LIVE | 0 | 2 | 4 | 6 | 0 | 2026-10-20 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/paper/expansion/scripts/whatwhere/whatwhere_stats.py` | LIVE | 0 | 0 | 3 | 0 | 0 | 2026-10-21 | keep | named in docs/paper/MAIN_APPENDIX2.md |
| `docs/theory/2026-10-04/scripts/basins.py` | LIVE | 0 | 7 | 3 | 31 | 0 | 2026-10-04 | keep | named in SCORE_RANK_PREREG.md |
| `docs/theory/2026-10-04/scripts/basins_loop.py` | LIVE | 0 | 0 | 1 | 7 | 0 | 2026-10-04 | keep | run dir in LOOP_RANK_E1800_P1_RESULTS.md |
| `docs/theory/2026-10-04/scripts/basins_rt.py` | LIVE | 0 | 0 | 1 | 37 | 0 | 2026-10-04 | keep | run dir in .claude-memory/project_state.md |
| `docs/theory/2026-10-04/scripts/basins_sign.py` | LIVE | 0 | 0 | 0 | 15 | 0 | 2026-10-04 | keep | run dir in LOOP_RANK_E1800_P1_RESULTS.md |
| `docs/theory/2026-10-04/scripts/per_head.py` | LIVE | 0 | 0 | 1 | 31 | 0 | 2026-10-04 | keep | run dir in .claude-memory/project_state.md |
| `docs/theory/2026-10-04/scripts/tw_latent.py` | LIVE | 0 | 0 | 1 | 7 | 0 | 2026-10-04 | keep | run dir in TEXTWORLD_PREREG.md |
| `docs/theory/2026-10-04/scripts/tw_weights.py` | LIVE | 0 | 0 | 1 | 7 | 0 | 2026-10-04 | keep | run dir in TEXTWORLD_PREREG.md |
| `docs/tools/catalog_results_index.py` | LIVE | 0 | 0 | 2 | 1 | 0 | 2026-10-09 | keep | named in CLAUDE.md |
| `dyck_standard_metrics.py` | LIVE | 0 | 0 | 0 | 6 | 0 | 2026-09-15 | keep | run dir in DYCK_RESULTS_bs128.md |
| `environment_clock.py` | LIVE | 2 | 1 | 0 | 0 | 0 | 2026-07-28 | keep | dependency of environment_recency |
| `environment_jsb.py` | LIVE | 3 | 0 | 0 | 0 | 0 | 2026-09-15 | keep | dependency of train_jsb |
| `environment_map_query.py` | LIVE | 2 | 3 | 6 | 0 | 0 | 2026-08-09 | keep | dependency of validate_map_query |
| `environment_multienv.py` | LIVE | 1 | 2 | 2 | 0 | 0 | 2026-05-10 | keep | named in README.md |
| `environment_multisize.py` | LIVE | 1 | 1 | 1 | 0 | 0 | 2026-05-11 | keep | named in README.md |
| `environment_schema.py` | LIVE | 1 | 0 | 2 | 1 | 0 | 2026-08-16 | keep | named in RANKWD_PREREG.md |
| `environment_topology.py` | LIVE | 1 | 2 | 2 | 0 | 0 | 2026-05-13 | keep | named in README.md |
| `eval_clock_side.py` | LIVE | 0 | 2 | 2 | 4 | 1 | 2026-10-12 | keep (md5-recorded) | commit cited in docs/audits/2026-10-12/ring_dial_audit.md |
| `eval_code_delims.py` | LIVE | 0 | 1 | 3 | 12 | 0 | 2026-09-24 | keep | run dir in ABLATE_PREREG.md |
| `eval_code_long.py` | LIVE | 0 | 3 | 6 | 2 | 0 | 2026-09-22 | keep | named in ABLATE_PREREG.md |
| `eval_continuous.py` | LIVE | 0 | 3 | 1 | 6 | 0 | 2026-05-01 | keep | dependency of run_p4_p5_fix.sh |
| `eval_exptorus.py` | LIVE | 0 | 2 | 2 | 1 | 3 | 2026-10-26 | keep (md5-recorded) | named in EXPTORUS_PREREG.md |
| `eval_gain_grain.py` | LIVE | 0 | 1 | 1 | 1 | 1 | 2026-10-05 | keep (md5-recorded) | named in GAIN_GRAIN_PREREG.md |
| `eval_gain_phase.py` | LIVE | 0 | 1 | 1 | 1 | 1 | 2026-10-06 | keep (md5-recorded) | named in GAIN_PHASE_PREREG.md |
| `eval_goal_closedloop.py` | LIVE | 0 | 2 | 0 | 0 | 0 | 2026-09-24 | keep | dependency of probe_active_inference |
| `eval_h12_budget.py` | LIVE | 1 | 0 | 2 | 2 | 0 | 2026-08-23 | keep | named in H12_BUDGET_CURVE.md |
| `eval_h12_perseed.py` | LIVE | 0 | 1 | 2 | 1 | 0 | 2026-08-23 | keep | named in H12_BUDGET_CURVE.md |
| `eval_long_t.py` | LIVE | 0 | 2 | 2 | 0 | 0 | 2026-09-24 | keep | named in README.md |
| `eval_match_longq.py` | LIVE | 0 | 1 | 1 | 4 | 0 | 2026-09-24 | keep | run dir in BASELINE_TABLE.md |
| `eval_minigrid_2x2.py` | LIVE | 0 | 6 | 0 | 2 | 0 | 2026-08-24 | keep | dependency of run_n8_evals.sh |
| `eval_nd.py` | LIVE | 0 | 5 | 2 | 0 | 2 | 2026-09-04 | keep (md5-recorded) | named in RANK_ND_PREREG.md |
| `eval_ood_grid.py` | LIVE | 0 | 1 | 2 | 1 | 0 | 2026-09-24 | keep | named in README.md |
| `eval_pair.py` | LIVE | 0 | 1 | 1 | 2 | 1 | 2026-10-04 | keep (md5-recorded) | named in MAPPOPE_PAIR_PREREG.md |
| `eval_paper_ood.py` | LIVE | 1 | 5 | 0 | 2 | 0 | 2026-08-19 | keep | dependency of run_pope_ood.sh |
| `eval_paper_task.py` | LIVE | 2 | 5 | 1 | 3 | 0 | 2026-08-17 | keep | dependency of run_level15_paper_50ep.sh |
| `eval_rnn_torus.py` | LIVE | 0 | 3 | 1 | 1 | 1 | 2026-10-26 | keep (md5-recorded) | named in docs/audits/2026-10-26/rnn_torus_audit.md |
| `eval_score_rank.py` | LIVE | 0 | 1 | 1 | 0 | 1 | 2026-10-05 | keep (md5-recorded) | named in SCORE_RANK_PREREG.md |
| `experiment_audit.py` | LIVE | 0 | 0 | 17 | 12 | 0 | 2026-09-24 | keep | named in CLAUDE.md |
| `gain_phase_eval.py` | LIVE | 1 | 1 | 1 | 1 | 1 | 2026-10-06 | keep (md5-recorded) | named in GAIN_PHASE_PREREG.md |
| `gate_image_torus.py` | LIVE | 0 | 2 | 3 | 1 | 1 | 2026-10-25 | keep (md5-recorded) | named in IMAGE_TORUS_PREREG.md |
| `gate_ssm_sign.py` | LIVE | 0 | 1 | 2 | 1 | 0 | 2026-10-23 | keep | named in G3_SSM_PREREG.md |
| `hyak/mm_download.sh` | LIVE | 0 | 0 | 1 | 1 | 0 | 2026-10-26 | keep | commit cited in docs/audits/2026-10-26/rnn_torus_audit.md |
| `hyak/mm_floors.slurm` | LIVE | 0 | 1 | 1 | 2 | 0 | 2026-10-27 | keep | named in MEMORY_MAZE_PREREG.md |
| `hyak/mm_gate.slurm` | LIVE | 0 | 0 | 1 | 2 | 0 | 2026-10-26 | keep | commit cited in docs/audits/2026-10-26/rnn_torus_audit.md |
| `hyak/mm_prep.slurm` | LIVE | 0 | 1 | 1 | 2 | 0 | 2026-10-27 | keep | named in MEMORY_MAZE_PREREG.md |
| `hyak/mm_runs.slurm` | LIVE | 0 | 1 | 1 | 2 | 0 | 2026-10-27 | keep | named in docs/audits/2026-10-27/memory_maze_audit.md |
| `hyak/mm_sync.sh` | LIVE | 0 | 0 | 2 | 4 | 0 | 2026-10-27 | keep | named in MEMORY_MAZE_PREREG.md |
| `lib_driver.sh` | LIVE | 0 | 68 | 38 | 4 | 31 | 2026-10-09 | keep (md5-recorded) | named in CLAUDE.md |
| `long_sequence_eval.py` | LIVE | 0 | 3 | 5 | 0 | 0 | 2026-09-24 | keep | named in README.md |
| `loop_depth_strata.py` | LIVE | 0 | 0 | 0 | 3 | 0 | 2026-09-24 | keep | run dir in docs/paper/expansion/RECURRENCE.md |
| `main.py` | LIVE | 0 | 0 | 6 | 0 | 0 | 2026-04-16 | keep | named in CLAUDE.md |
| `make_dyck_table.py` | LIVE | 0 | 0 | 0 | 6 | 0 | 2026-09-15 | keep | run dir in DYCK_RESULTS_bs128.md |
| `make_figures.py` | LIVE | 0 | 0 | 0 | 5 | 0 | 2026-08-16 | keep | run dir in BASELINE_TABLE.md |
| `master_finish_v3.sh` | LIVE | 0 | 1 | 3 | 2 | 0 | 2026-04-24 | keep | named in README.md |
| `measure_paper_floor.py` | LIVE | 0 | 0 | 0 | 3 | 0 | 2026-09-12 | keep | reads/writes BASELINE_TABLE.md |
| `mm_encoder.py` | LIVE | 0 | 4 | 2 | 4 | 0 | 2026-10-27 | keep | named in MEMORY_MAZE_PREREG.md |
| `mm_floors.py` | LIVE | 0 | 1 | 2 | 2 | 0 | 2026-10-27 | keep | named in MEMORY_MAZE_PREREG.md |
| `mm_gate_v3.py` | LIVE | 1 | 1 | 2 | 1 | 0 | 2026-10-27 | keep | named in docs/audits/2026-10-27/memory_maze_audit.md |
| `model_bounded_mem.py` | LIVE | 1 | 5 | 1 | 0 | 18 | 2026-07-20 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_centered.py` | LIVE | 2 | 0 | 0 | 1 | 0 | 2026-09-18 | keep | dependency of train_jsb |
| `model_cho_positions.py` | LIVE | 1 | 5 | 1 | 2 | 18 | 2026-09-15 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_code_decay.py` | LIVE | 1 | 5 | 0 | 1 | 18 | 2026-09-21 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_counter_installed.py` | LIVE | 1 | 5 | 0 | 1 | 18 | 2026-09-14 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_coupled_ape.py` | LIVE | 1 | 5 | 1 | 0 | 18 | 2026-09-14 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_coupled_rope.py` | LIVE | 1 | 5 | 0 | 0 | 18 | 2026-09-14 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_dyck_monotone.py` | LIVE | 1 | 0 | 0 | 2 | 0 | 2026-09-20 | keep | dependency of train_dyck |
| `model_em_dof.py` | LIVE | 1 | 5 | 1 | 3 | 18 | 2026-09-10 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_em_magonly.py` | LIVE | 1 | 5 | 1 | 1 | 18 | 2026-09-10 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_em_noleak.py` | LIVE | 1 | 5 | 1 | 1 | 18 | 2026-09-11 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_em_pairconst.py` | LIVE | 1 | 5 | 1 | 2 | 18 | 2026-09-12 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_em_phase.py` | LIVE | 1 | 5 | 1 | 1 | 18 | 2026-09-09 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_fixed_omega.py` | LIVE | 1 | 5 | 2 | 3 | 18 | 2026-08-21 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_forget.py` | LIVE | 1 | 6 | 0 | 2 | 18 | 2026-09-06 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_gated.py` | LIVE | 1 | 5 | 2 | 1 | 18 | 2026-09-07 | keep (md5-recorded) | named in G3_SSM_PREREG.md; INVALID ESCAPE (DeprecationWarning on 3.10, SyntaxWarning on 3.12) |
| `model_grid_l15_pc.py` | LIVE | 3 | 7 | 2 | 0 | 18 | 2026-04-26 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_inekf_gsf_modeomega.py` | LIVE | 1 | 7 | 0 | 0 | 18 | 2026-05-11 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_inekf_level15_beta.py` | LIVE | 1 | 7 | 2 | 0 | 18 | 2026-05-10 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_inekf_level15_em_perscale.py` | LIVE | 1 | 6 | 0 | 0 | 18 | 2026-05-18 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_inekf_level15_extrahead.py` | LIVE | 1 | 6 | 0 | 0 | 18 | 2026-05-21 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_inekf_level15_hopfield_nomainap.py` | LIVE | 1 | 6 | 0 | 0 | 18 | 2026-05-21 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_inekf_level15_perscale.py` | LIVE | 1 | 6 | 1 | 1 | 18 | 2026-05-17 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_inekf_level15_sr.py` | LIVE | 1 | 6 | 1 | 0 | 18 | 2026-05-17 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_level15_pc_v3.py` | LIVE | 1 | 7 | 2 | 0 | 18 | 2026-04-28 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_level15_pc_v4.py` | LIVE | 1 | 8 | 2 | 0 | 18 | 2026-04-28 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_mapformer_nc.py` | LIVE | 1 | 5 | 0 | 0 | 18 | 2026-08-15 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_mono_baselines.py` | LIVE | 1 | 5 | 11 | 9 | 18 (1 drifted) | 2026-10-14 | keep (md5-recorded) | named in GENERALITY_PREREG.md |
| `model_monotone.py` | LIVE | 1 | 6 | 1 | 4 | 18 | 2026-09-13 | keep (md5-recorded) | reads/writes SEARCH_RESULTS.md |
| `model_pope_ablate.py` | LIVE | 1 | 5 | 4 | 0 | 18 | 2026-09-22 | keep (md5-recorded) | named in DELTA_FREE_PREREG.md |
| `model_pope_index_hier.py` | LIVE | 1 | 5 | 0 | 2 | 18 | 2026-08-28 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_ring_rank.py` | LIVE | 1 | 1 | 1 | 1 | 1 | 2026-10-11 | keep (md5-recorded) | named in RANK_TESTS_PREREG.md |
| `model_rope_canonical.py` | LIVE | 1 | 7 | 0 | 0 | 18 | 2026-09-03 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_route_attn.py` | LIVE | 1 | 5 | 0 | 0 | 18 | 2026-07-22 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_spacetime_hier.py` | LIVE | 1 | 5 | 0 | 0 | 18 | 2026-07-22 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_srope.py` | LIVE | 1 | 1 | 1 | 1 | 11 | 2026-10-20 | keep (md5-recorded) | named in SROPE_TORUS_PREREG.md |
| `model_srope_components.py` | LIVE | 1 | 5 | 0 | 0 | 18 | 2026-08-28 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_tem.py` | LIVE | 1 | 10 | 1 | 0 | 18 | 2026-05-01 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_tem_ffn.py` | LIVE | 1 | 6 | 0 | 0 | 18 | 2026-05-21 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_tem_recency.py` | LIVE | 1 | 5 | 1 | 0 | 18 | 2026-09-14 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_tem_scaling.py` | LIVE | 1 | 6 | 0 | 0 | 18 | 2026-05-21 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_tem_t.py` | LIVE | 1 | 8 | 1 | 0 | 18 | 2026-05-03 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_torus_start.py` | LIVE | 1 | 1 | 2 | 2 | 1 | 2026-10-14 | keep (md5-recorded) | named in TORUS_START_PREREG.md |
| `model_vanilla_extrahead.py` | LIVE | 1 | 6 | 0 | 0 | 18 | 2026-05-21 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `model_vanilla_nodrop.py` | LIVE | 1 | 6 | 1 | 0 | 18 | 2026-05-15 | keep (md5-recorded) | md5-recorded in runs/delta_free_pilot (cited run dir) |
| `orchestrator.py` | LIVE | 0 | 1 | 2 | 0 | 0 | 2026-04-22 | keep | named in README.md |
| `orchestrator_cope_resume.py` | LIVE | 0 | 1 | 0 | 0 | 0 | 2026-04-24 | keep | dependency of master_finish_v3.sh |
| `orchestrator_finalize.sh` | LIVE | 0 | 4 | 4 | 2 | 0 | 2026-04-26 | keep | dependency of master_finish_v3.sh |
| `orchestrator_mambalike.py` | LIVE | 0 | 1 | 1 | 0 | 0 | 2026-04-24 | keep | dependency of master_finish_v3.sh |
| `prebuild_buffers.py` | LIVE | 0 | 5 | 3 | 0 | 0 | 2026-08-28 | keep | named in CLAUDE.md |
| `probe_accumulator.py` | LIVE | 0 | 2 | 2 | 0 | 0 | 2026-09-06 | keep | named in RECENCY_PREREG.md |
| `probe_achievable.py` | LIVE | 0 | 0 | 1 | 16 | 0 | 2026-09-12 | keep | run dir in EM_WM_STATE.md |
| `probe_action_geometry.py` | LIVE | 0 | 8 | 2 | 2 | 0 | 2026-09-24 | keep | dependency of run_rank_matched.sh |
| `probe_active_inference.py` | LIVE | 0 | 3 | 1 | 0 | 0 | 2026-05-15 | keep | dependency of model_inekf_level15_sr |
| `probe_ap_coherence.py` | LIVE | 0 | 1 | 2 | 9 | 0 | 2026-09-09 | keep | run dir in EM_WM_STATE.md |
| `probe_code_accum.py` | LIVE | 0 | 1 | 2 | 1 | 0 | 2026-09-24 | keep | dependency of ckpt_guard |
| `probe_code_depth0.py` | LIVE | 0 | 1 | 2 | 1 | 0 | 2026-09-24 | keep | dependency of ckpt_guard |
| `probe_collisions.py` | LIVE | 0 | 0 | 1 | 4 | 0 | 2026-09-12 | keep | run dir in SEARCH_PREREG.md |
| `probe_conv_kernel.py` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-04 | keep | reads/writes SELECTIVE_ROPE.md |
| `probe_drift.py` | LIVE | 0 | 0 | 0 | 6 | 0 | 2026-09-04 | keep | reads/writes RANK_SWEEP.md |
| `probe_dyck_alpha.py` | LIVE | 0 | 0 | 1 | 6 | 0 | 2026-09-20 | keep | run dir in DYCK_RESULTS_bs128.md |
| `probe_dyck_far.py` | LIVE | 0 | 0 | 1 | 7 | 0 | 2026-09-20 | keep | run dir in DYCK_RESULTS_bs128.md |
| `probe_dyck_metric.py` | LIVE | 0 | 0 | 2 | 6 | 0 | 2026-09-20 | keep | run dir in CROSS_RESULTS.md |
| `probe_gate_separation.py` | LIVE | 0 | 1 | 0 | 4 | 0 | 2026-09-07 | keep | reads/writes GATED_PREREG.md |
| `probe_goal_distance.py` | LIVE | 2 | 5 | 0 | 0 | 0 | 2026-05-15 | keep | dependency of model_inekf_level15_sr |
| `probe_goal_linear.py` | LIVE | 0 | 1 | 3 | 0 | 0 | 2026-09-24 | keep | named in README.md |
| `probe_gsf_modes.py` | LIVE | 0 | 1 | 3 | 0 | 0 | 2026-09-24 | keep | named in README.md |
| `probe_hex.py` | LIVE | 2 | 3 | 2 | 1 | 0 | 2026-04-30 | keep | dependency of probe_hex_continuous |
| `probe_hex_continuous.py` | LIVE | 0 | 3 | 1 | 6 | 0 | 2026-05-01 | keep | dependency of run_p4_p5_fix.sh |
| `probe_hex_emergence.py` | LIVE | 1 | 2 | 1 | 0 | 0 | 2026-09-24 | keep | named in README.md |
| `probe_init_grad.py` | LIVE | 0 | 0 | 0 | 15 | 0 | 2026-09-11 | keep | run dir in EM_WM_STATE.md |
| `probe_lap_theta.py` | LIVE | 0 | 1 | 2 | 0 | 0 | 2026-09-24 | keep | dependency of environment_lap |
| `probe_localisation.py` | LIVE | 0 | 1 | 0 | 1 | 0 | 2026-09-06 | keep | dependency of run_localisation.sh |
| `probe_paper_fig4.py` | LIVE | 0 | 1 | 0 | 4 | 0 | 2026-09-24 | keep | reads/writes RANK_SWEEP.md |
| `probe_phase_spread.py` | LIVE | 1 | 0 | 2 | 10 | 0 | 2026-09-12 | keep | run dir in EM_WM_STATE.md |
| `probe_rank_projection.py` | LIVE | 0 | 1 | 2 | 0 | 0 | 2026-09-23 | keep | named in RANK_PROJ_PREREG.md |
| `probe_recency_accumulator.py` | LIVE | 1 | 0 | 0 | 14 | 0 | 2026-09-24 | keep | run dir in PAPER2X2_PREREG.md |
| `probe_revisit_2x2.py` | LIVE | 0 | 0 | 1 | 14 | 0 | 2026-09-14 | keep | run dir in MAPPOPE_PAIR_PREREG.md |
| `probe_revisit_distance.py` | LIVE | 0 | 2 | 0 | 3 | 0 | 2026-08-17 | keep | dependency of run_horizon.sh |
| `probe_sign.py` | LIVE | 0 | 4 | 10 | 1 | 0 | 2026-09-06 | keep | named in RECENCY_PREREG.md |
| `probe_stitch_attention.py` | LIVE | 0 | 3 | 0 | 1 | 0 | 2026-08-16 | keep | reads/writes STITCH_ATTENTION.md |
| `probe_textworld.py` | LIVE | 1 | 1 | 1 | 1 | 0 | 2026-09-28 | keep | named in TEXTWORLD_PREREG.md |
| `probe_vector_nav_v2.py` | LIVE | 0 | 1 | 2 | 1 | 0 | 2026-09-24 | keep | named in README.md |
| `probe_visits_per_cell.py` | LIVE | 0 | 2 | 2 | 0 | 0 | 2026-09-24 | keep | named in VISITS_TEST.md |
| `ring_rank_readouts.py` | LIVE | 0 | 1 | 1 | 1 | 1 | 2026-10-11 | keep (md5-recorded) | named in RANK_TESTS_PREREG.md |
| `run_ablate.sh` | LIVE | 0 | 6 | 0 | 4 | 0 | 2026-09-22 | keep | run dir in ABLATE_RESULTS.md |
| `run_accounts.sh` | LIVE | 0 | 0 | 4 | 12 | 1 | 2026-10-14 | keep (md5-recorded) | named in docs/paper/APPENDIX_draft6.md |
| `run_alias_finish.sh` | LIVE | 0 | 0 | 0 | 4 | 0 | 2026-09-24 | keep | reads/writes ALIASING_CONTROLLED.md |
| `run_alias_sweep.sh` | LIVE | 0 | 1 | 0 | 4 | 0 | 2026-08-28 | keep | reads/writes ALIASING_CONTROLLED.md |
| `run_cancel.sh` | LIVE | 0 | 1 | 2 | 6 | 0 | 2026-09-27 | keep | named in CANCEL_PREREG.md |
| `run_clock_side.sh` | LIVE | 0 | 0 | 5 | 4 | 1 | 2026-10-12 | keep (md5-recorded) | named in docs/paper/APPENDIX_draft6.md |
| `run_code.sh` | LIVE | 0 | 0 | 0 | 12 | 0 | 2026-09-21 | keep | run dir in ABLATE_PREREG.md |
| `run_code2048.sh` | LIVE | 0 | 0 | 0 | 10 | 0 | 2026-09-22 | keep | run dir in .claude-memory/project_mappope_asymmetry.md |
| `run_code2048_fill.sh` | LIVE | 0 | 0 | 3 | 10 | 0 | 2026-09-22 | keep | run dir in .claude-memory/project_mappope_asymmetry.md |
| `run_code2048_seeds.sh` | LIVE | 0 | 2 | 0 | 10 | 0 | 2026-09-22 | keep | run dir in .claude-memory/project_mappope_asymmetry.md |
| `run_code_controls.sh` | LIVE | 0 | 0 | 0 | 6 | 0 | 2026-09-22 | keep | run dir in CODE_FULLVAL_PREREG.md |
| `run_code_decay.sh` | LIVE | 0 | 0 | 0 | 12 | 0 | 2026-09-22 | keep | run dir in ABLATE_PREREG.md |
| `run_code_fullval.sh` | LIVE | 0 | 0 | 1 | 16 | 0 | 2026-09-27 | keep | named in CODE_FULLVAL_RESULTS.md |
| `run_cognitive_tier.sh` | LIVE | 0 | 0 | 1 | 0 | 0 | 2026-05-10 | keep | named in README.md |
| `run_comp_headroom.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-08 | keep | reads/writes COMP_HEADROOM.md |
| `run_comp_multiseed.sh` | LIVE | 0 | 1 | 3 | 2 | 0 | 2026-07-25 | keep | named in COMPOSITIONAL_EXPERIMENT.md |
| `run_continuous_alloc.sh` | LIVE | 0 | 0 | 0 | 3 | 0 | 2026-08-23 | keep | reads/writes ALLOCENTRIC_RECODING.md |
| `run_correction_gaps.sh` | LIVE | 0 | 0 | 0 | 4 | 0 | 2026-08-21 | keep | run dir in BASELINE_TABLE.md |
| `run_cross.sh` | LIVE | 0 | 0 | 0 | 3 | 0 | 2026-09-20 | keep | run dir in CROSS_RESULTS.md |
| `run_d5_seeds.sh` | LIVE | 0 | 0 | 0 | 15 | 0 | 2026-09-10 | keep | run dir in EM_WM_STATE.md |
| `run_delta_free.sh` | LIVE | 0 | 0 | 2 | 3 | 2 (1 drifted) | 2026-10-20 | keep (md5-recorded) | named in DELTA_FREE_PREREG.md |
| `run_dissociation.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-08-19 | keep | run dir in BASELINE_TABLE.md |
| `run_dof.sh` | LIVE | 0 | 0 | 1 | 16 | 0 | 2026-09-10 | keep | run dir in EM_WM_STATE.md |
| `run_dog_fix_and_continuous.sh` | LIVE | 0 | 0 | 2 | 11 | 0 | 2026-05-01 | keep | run dir in README.md |
| `run_dyck.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-15 | keep | reads/writes DYCK_RESULTS_bs128.md |
| `run_dyck_decay.sh` | LIVE | 0 | 0 | 0 | 6 | 0 | 2026-09-19 | keep | run dir in CROSS_RESULTS.md |
| `run_dyck_depth.sh` | LIVE | 0 | 0 | 0 | 3 | 0 | 2026-09-22 | keep | run dir in DYCK_LADDER_PREREG.md |
| `run_dyck_ladder.sh` | LIVE | 0 | 0 | 0 | 6 | 0 | 2026-09-22 | keep | run dir in DYCK_MDEPTH_PREREG.md |
| `run_dyck_mdepth.sh` | LIVE | 0 | 0 | 4 | 12 | 0 | 2026-09-25 | keep | named in DYCK_MDEPTH_PREREG.md |
| `run_em_p0.sh` | LIVE | 0 | 1 | 0 | 4 | 0 | 2026-08-09 | keep | dependency of eval_paper_task |
| `run_exptorus.sh` | LIVE | 0 | 0 | 2 | 4 | 3 | 2026-10-26 | keep (md5-recorded) | named in EXPTORUS_PREREG.md |
| `run_extrahead_control.sh` | LIVE | 0 | 0 | 0 | 1 | 0 | 2026-05-21 | keep | reads/writes EXTRAHEAD_CONTROL.md |
| `run_factorial.sh` | LIVE | 0 | 1 | 7 | 38 | 1 | 2026-10-12 | keep (md5-recorded) | named in FACTORIAL_N8_PREREG.md |
| `run_factorial_n8.sh` | LIVE | 0 | 0 | 3 | 22 | 1 | 2026-10-24 | keep (md5-recorded) | named in FACTORIAL_N8_PREREG.md |
| `run_factorial_n8_pilot.sh` | LIVE | 0 | 0 | 2 | 4 | 0 | 2026-10-24 | keep | named in FACTORIAL_N8_PREREG.md |
| `run_family_tree.sh` | LIVE | 0 | 1 | 0 | 2 | 0 | 2026-08-15 | keep | reads/writes FAMILY_TREE_RESULTS.md |
| `run_forget_control.sh` | LIVE | 0 | 0 | 0 | 3 | 0 | 2026-09-05 | keep | reads/writes FORGET_CONTROL.md |
| `run_fr.sh` | LIVE | 0 | 0 | 1 | 3 | 0 | 2026-07-26 | keep | named in COMPOSITIONAL_EXPERIMENT.md |
| `run_frozen.sh` | LIVE | 0 | 0 | 0 | 3 | 0 | 2026-09-20 | keep | run dir in CROSS_RESULTS.md |
| `run_ft_extra.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-08-21 | keep | run dir in BASELINE_TABLE.md |
| `run_gain_grain.sh` | LIVE | 0 | 0 | 1 | 4 | 0 | 2026-10-05 | keep | named in GAIN_GRAIN_PREREG.md |
| `run_gain_phase.sh` | LIVE | 0 | 0 | 1 | 2 | 0 | 2026-10-06 | keep | named in GAIN_PHASE_PREREG.md |
| `run_gated.sh` | LIVE | 0 | 1 | 0 | 2 | 0 | 2026-09-07 | keep | reads/writes GATED_PREREG.md |
| `run_gauge.sh` | LIVE | 0 | 0 | 4 | 5 | 1 | 2026-10-15 | keep (md5-recorded) | named in docs/paper/APPENDIX_draft6.md |
| `run_generality.sh` | LIVE | 0 | 0 | 10 | 26 | 1 | 2026-10-14 | keep (md5-recorded) | named in SROPE_TORUS_PREREG.md |
| `run_hier_recheck.sh` | LIVE | 0 | 0 | 0 | 1 | 0 | 2026-09-08 | keep | reads/writes HIER_RECHECK.md |
| `run_horizon.sh` | LIVE | 0 | 0 | 0 | 11 | 0 | 2026-08-19 | keep | run dir in BASELINE_TABLE.md |
| `run_image_torus.sh` | LIVE | 0 | 0 | 3 | 5 | 1 | 2026-10-25 | keep (md5-recorded) | named in IMAGE_TORUS_PREREG.md |
| `run_image_torus_pilot.sh` | LIVE | 0 | 0 | 2 | 2 | 0 | 2026-10-25 | keep | named in IMAGE_TORUS_PREREG.md |
| `run_image_torus_pilot2.sh` | LIVE | 0 | 0 | 2 | 3 | 0 | 2026-10-25 | keep | named in IMAGE_TORUS_PREREG.md |
| `run_indirect.sh` | LIVE | 0 | 0 | 0 | 5 | 0 | 2026-09-15 | keep | reads/writes INDIRECT_RESULTS_200k.md |
| `run_indirect_200k.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-16 | keep | reads/writes INDIRECT_RESULTS_200k.md |
| `run_indirect_more.sh` | LIVE | 0 | 0 | 0 | 6 | 0 | 2026-09-15 | keep | reads/writes INDIRECT_RESULTS_200k.md |
| `run_jsb.sh` | LIVE | 0 | 0 | 0 | 8 | 0 | 2026-09-15 | keep | reads/writes INDIRECT_RESULTS_200k.md |
| `run_l15_ablation.sh` | LIVE | 0 | 0 | 1 | 4 | 0 | 2026-09-24 | keep | reads/writes L15_ABLATION.md |
| `run_l15_loop_2x2.sh` | LIVE | 0 | 1 | 0 | 3 | 0 | 2026-09-24 | keep | reads/writes L15_ABLATION.md |
| `run_l15_loop_supp.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | keep | reads/writes L15_LOOP_2X2.md |
| `run_lambda_trace.sh` | LIVE | 0 | 1 | 0 | 2 | 0 | 2026-09-05 | keep | reads/writes FORGET_CONTROL.md |
| `run_leak.sh` | LIVE | 0 | 1 | 1 | 6 | 0 | 2026-10-03 | keep | named in GAIN_PHASE_PREREG.md |
| `run_level15_meets_gated.sh` | LIVE | 0 | 1 | 1 | 5 | 0 | 2026-08-17 | keep | run dir in BASELINE_TABLE.md |
| `run_level15_paper_50ep.sh` | LIVE | 0 | 0 | 0 | 3 | 0 | 2026-08-17 | keep | run dir in BASELINE_TABLE.md |
| `run_localisation.sh` | LIVE | 0 | 0 | 0 | 17 | 0 | 2026-09-06 | keep | run dir in PAPER2X2_PREREG.md |
| `run_longt.sh` | LIVE | 0 | 1 | 0 | 0 | 0 | 2026-05-10 | keep | dependency of run_cognitive_tier.sh |
| `run_loop_headroom.sh` | LIVE | 0 | 0 | 1 | 3 | 0 | 2026-09-24 | keep | run dir in docs/paper/expansion/RECURRENCE.md |
| `run_loop_rank.sh` | LIVE | 0 | 0 | 0 | 7 | 0 | 2026-09-26 | keep | run dir in LOOP_RANK_RESULTS.md |
| `run_loop_rank_e1800.sh` | LIVE | 0 | 0 | 0 | 3 | 0 | 2026-09-27 | keep | run dir in LOOP_RANK_E1800_P1_RESULTS.md |
| `run_loop_rank_e1800_budget.sh` | LIVE | 0 | 0 | 1 | 4 | 0 | 2026-09-28 | keep | run dir in LOOP_RANK_E1800_P1_RESULTS.md |
| `run_loop_topup.sh` | LIVE | 0 | 0 | 1 | 3 | 0 | 2026-09-24 | keep | run dir in docs/paper/expansion/RECURRENCE.md |
| `run_loop_topup2.sh` | LIVE | 0 | 0 | 1 | 3 | 0 | 2026-09-24 | keep | run dir in docs/paper/expansion/RECURRENCE.md |
| `run_lrsweep.sh` | LIVE | 0 | 1 | 2 | 5 | 1 | 2026-10-16 | keep (md5-recorded) | named in LRSWEEP_PREREG.md |
| `run_magonly.sh` | LIVE | 0 | 0 | 0 | 15 | 0 | 2026-09-10 | keep | run dir in EM_WM_STATE.md |
| `run_mappope_pair.sh` | LIVE | 0 | 1 | 2 | 6 | 0 | 2026-10-04 | keep | named in GAIN_GRAIN_PREREG.md |
| `run_match_query.sh` | LIVE | 0 | 0 | 1 | 4 | 0 | 2026-08-09 | keep | run dir in BASELINE_TABLE.md |
| `run_match_scale.sh` | LIVE | 0 | 0 | 1 | 5 | 0 | 2026-08-16 | keep | run dir in BASELINE_TABLE.md |
| `run_minigrid_2x2.sh` | LIVE | 0 | 1 | 0 | 2 | 0 | 2026-08-21 | keep | dependency of run_minigrid_2x2x2.sh |
| `run_minigrid_2x2x2.sh` | LIVE | 0 | 2 | 1 | 2 | 0 | 2026-08-21 | keep | reads/writes EXTRAHEAD_CONTROL.md |
| `run_miniworld.sh` | LIVE | 0 | 1 | 1 | 4 | 0 | 2026-08-28 | keep | dependency of train_miniworld |
| `run_mm_pilot.sh` | LIVE | 0 | 0 | 1 | 2 | 0 | 2026-10-27 | keep | named in MEMORY_MAZE_PREREG.md |
| `run_mm_pilot3.sh` | LIVE | 0 | 0 | 1 | 2 | 0 | 2026-10-27 | keep | named in MEMORY_MAZE_PREREG.md |
| `run_motifseg.sh` | LIVE | 0 | 0 | 2 | 3 | 0 | 2026-07-25 | keep | named in COMPOSITIONAL_EXPERIMENT.md |
| `run_mq_rank.sh` | LIVE | 0 | 0 | 0 | 5 | 0 | 2026-09-04 | keep | run dir in docs/paper/expansion/RECURRENCE.md |
| `run_multienv.sh` | LIVE | 0 | 1 | 0 | 0 | 0 | 2026-05-10 | keep | dependency of run_cognitive_tier.sh |
| `run_n8_evals.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-08-22 | keep | reads/writes KNOB_SWEEP_n8.md |
| `run_noise_refine.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | keep | reads/writes NOISE_REFINE.md |
| `run_p0_extend.sh` | LIVE | 0 | 1 | 0 | 16 | 0 | 2026-09-12 | keep | run dir in EM_WM_STATE.md |
| `run_p4_p5_fix.sh` | LIVE | 0 | 0 | 0 | 9 | 0 | 2026-05-02 | keep | run dir in README.md |
| `run_pairconst.sh` | LIVE | 0 | 0 | 0 | 7 | 0 | 2026-09-12 | keep | run dir in PAIRSPLIT_PREREG.md |
| `run_pairorigin.sh` | LIVE | 0 | 1 | 1 | 4 | 0 | 2026-09-11 | keep | run dir in PAIRSPLIT_PREREG.md |
| `run_pairsplit.sh` | LIVE | 0 | 0 | 0 | 5 | 0 | 2026-09-12 | keep | commit cited in PAIRSPLIT_RESULTS.md |
| `run_paper2x2.sh` | LIVE | 0 | 1 | 2 | 13 | 0 | 2026-09-13 | keep | named in DELTA_FREE_PREREG.md |
| `run_papertask_rerun.sh` | LIVE | 0 | 0 | 0 | 7 | 0 | 2026-09-12 | keep | run dir in PAIRSPLIT_PREREG.md |
| `run_pope_ood.sh` | LIVE | 0 | 0 | 0 | 4 | 0 | 2026-08-17 | keep | run dir in BASELINE_TABLE.md |
| `run_pope_redo.sh` | LIVE | 0 | 0 | 1 | 0 | 0 | 2026-08-09 | keep | named in RANKWD_PREREG.md |
| `run_rank3.sh` | LIVE | 0 | 0 | 0 | 6 | 0 | 2026-09-27 | keep | run dir in RANK3_RESULTS.md |
| `run_rank_matched.sh` | LIVE | 0 | 0 | 6 | 23 | 0 | 2026-09-24 | keep | run dir in GUARDS.md |
| `run_rank_mi.sh` | LIVE | 0 | 1 | 3 | 33 | 0 | 2026-09-24 | keep | named in SCORE_RANK_PREREG.md |
| `run_rank_nd.sh` | LIVE | 0 | 2 | 1 | 7 | 0 | 2026-09-30 | keep | commit cited in RANK_ND_PREREG.md |
| `run_rank_perhead_pilot.sh` | LIVE | 0 | 0 | 0 | 7 | 0 | 2026-09-24 | keep | run dir in .claude-memory/project_state.md |
| `run_rank_proj.sh` | LIVE | 0 | 0 | 2 | 8 | 0 | 2026-09-24 | keep | run dir in RANK_PROJ_PREREG.md |
| `run_rank_sep.sh` | LIVE | 0 | 0 | 1 | 9 | 0 | 2026-09-25 | keep | named in SCORE_RANK_PREREG.md |
| `run_rank_sweep.sh` | LIVE | 0 | 1 | 1 | 6 | 0 | 2026-09-24 | keep | reads/writes RANK_SWEEP.md |
| `run_rank_tests.sh` | LIVE | 0 | 0 | 3 | 28 | 1 | 2026-10-11 | keep (md5-recorded) | named in RANK_TESTS_PREREG.md |
| `run_rank_wrap.sh` | LIVE | 0 | 0 | 0 | 5 | 0 | 2026-10-01 | keep | run dir in RANK_WRAP_RESULTS.md |
| `run_rankclock.sh` | LIVE | 0 | 0 | 2 | 4 | 0 | 2026-10-23 | keep | named in RANKCLOCK_PREREG.md |
| `run_rankclock_existence.sh` | LIVE | 0 | 2 | 1 | 4 | 0 | 2026-10-23 | keep | named in RANKCLOCK_PREREG.md |
| `run_rankrecipe.sh` | LIVE | 0 | 0 | 1 | 6 | 0 | 2026-10-24 | keep | named in RANKRECIPE_PREREG.md |
| `run_rankwd.sh` | LIVE | 0 | 0 | 2 | 4 | 0 | 2026-10-23 | keep | named in RANKWD_PREREG.md |
| `run_recency.sh` | LIVE | 0 | 2 | 7 | 5 | 0 | 2026-09-07 | keep | named in GATED_PREREG.md |
| `run_recency_em.sh` | LIVE | 0 | 1 | 1 | 10 | 0 | 2026-09-09 | keep | run dir in EM_WM_STATE.md |
| `run_refine.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | keep | reads/writes REFINE_RESULTS.md |
| `run_ring_budget.sh` | LIVE | 0 | 0 | 3 | 3 | 1 | 2026-10-12 | keep (md5-recorded) | named in docs/paper/APPENDIX_draft6.md |
| `run_ring_dial.sh` | LIVE | 0 | 0 | 3 | 6 | 1 | 2026-10-12 | keep (md5-recorded) | named in docs/audits/2026-10-12/ring_dial_audit.md |
| `run_rnn_torus.sh` | LIVE | 0 | 1 | 4 | 30 | 1 | 2026-10-26 | keep (md5-recorded) | named in EXPTORUS_PREREG.md |
| `run_rnn_torus_pilot.sh` | LIVE | 0 | 0 | 2 | 3 | 0 | 2026-10-26 | keep | named in RNN_TORUS_PREREG.md |
| `run_rnn_torus_pilot2.sh` | LIVE | 0 | 0 | 2 | 3 | 0 | 2026-10-26 | keep | named in RNN_TORUS_PREREG.md |
| `run_robust.sh` | LIVE | 0 | 0 | 7 | 7 | 1 | 2026-10-16 | keep (md5-recorded) | named in ROBUST_PREREG.md |
| `run_rope_canonical.sh` | LIVE | 0 | 2 | 1 | 2 | 0 | 2026-09-24 | keep | dependency of run_rank_sweep.sh |
| `run_score_rank.sh` | LIVE | 0 | 0 | 1 | 6 | 0 | 2026-10-05 | keep | named in SCORE_RANK_PREREG.md |
| `run_search.sh` | LIVE | 0 | 0 | 0 | 15 | 0 | 2026-09-11 | keep | commit cited in SEARCH_RESULTS.md |
| `run_seed_scaleup.sh` | LIVE | 0 | 1 | 1 | 4 | 0 | 2026-08-19 | keep | run dir in BASELINE_TABLE.md |
| `run_selective.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | keep | reads/writes SELECTIVE_ROPE.md |
| `run_sign.sh` | LIVE | 0 | 4 | 6 | 13 | 0 | 2026-09-06 | keep | named in GATED_PREREG.md |
| `run_sign_matched.sh` | LIVE | 0 | 0 | 3 | 11 | 0 | 2026-09-27 | keep | named in docs/paper/APPENDIX_draft6.md |
| `run_signroute.sh` | LIVE | 0 | 1 | 1 | 2 | 0 | 2026-10-20 | keep | named in SIGNROUTE_PREREG.md |
| `run_signsep.sh` | LIVE | 0 | 0 | 1 | 3 | 1 | 2026-10-23 | keep (md5-recorded) | named in SIGNSEP_PREREG.md |
| `run_signsep_pilot.sh` | LIVE | 0 | 1 | 1 | 3 | 1 (2 drifted) | 2026-10-23 | keep (md5-recorded) | named in SIGNSEP_PREREG.md |
| `run_signsep_pilot2.sh` | LIVE | 0 | 0 | 1 | 3 | 1 | 2026-10-23 | keep (md5-recorded) | named in SIGNSEP_PREREG.md |
| `run_sparse_landmarks.sh` | LIVE | 0 | 1 | 0 | 0 | 0 | 2026-05-10 | keep | dependency of run_cognitive_tier.sh |
| `run_spread.sh` | LIVE | 0 | 0 | 0 | 15 | 0 | 2026-09-11 | keep | run dir in EM_WM_STATE.md |
| `run_srope_torus.sh` | LIVE | 0 | 0 | 2 | 8 | 1 | 2026-10-20 | keep (md5-recorded) | named in SROPE_TORUS_PREREG.md |
| `run_ssm_sign.sh` | LIVE | 0 | 0 | 6 | 7 | 2 (1 drifted) | 2026-10-24 | keep (md5-recorded) | named in FACTORIAL_N8_PREREG.md |
| `run_stitch_probe.sh` | LIVE | 0 | 0 | 0 | 1 | 0 | 2026-08-16 | keep | reads/writes STITCH_ATTENTION.md |
| `run_t3.sh` | LIVE | 0 | 1 | 1 | 2 | 0 | 2026-09-18 | keep | dependency of run_accounts.sh |
| `run_tem_t.sh` | LIVE | 0 | 0 | 1 | 3 | 0 | 2026-05-03 | keep | run dir in README.md |
| `run_textworld.sh` | LIVE | 0 | 1 | 3 | 7 | 0 | 2026-09-28 | keep | named in TEXTWORLD_PREREG.md |
| `run_timing_rerun.sh` | LIVE | 0 | 0 | 0 | 1 | 0 | 2026-08-16 | keep | reads/writes TIMING_BENCHMARK.md |
| `run_torus_lrsweep.sh` | LIVE | 0 | 0 | 3 | 4 | 1 | 2026-10-23 | keep (md5-recorded) | named in TORUS_LRSWEEP_PREREG.md |
| `run_torus_start.sh` | LIVE | 0 | 0 | 3 | 3 | 1 | 2026-10-14 | keep (md5-recorded) | named in docs/paper/APPENDIX_draft6.md |
| `run_tw_ambig.sh` | LIVE | 0 | 0 | 1 | 2 | 0 | 2026-10-08 | keep | commit cited in .claude-memory/project_state.md |
| `run_tw_landmark.sh` | LIVE | 0 | 0 | 1 | 2 | 0 | 2026-10-08 | keep | named in TW_LANDMARK_PREREG.md |
| `run_tw_landmark_pilot.sh` | LIVE | 0 | 0 | 1 | 3 | 0 | 2026-10-08 | keep | named in TW_LANDMARK_PREREG.md |
| `run_tw_normstep.sh` | LIVE | 0 | 0 | 0 | 7 | 0 | 2026-10-03 | keep | commit cited in TW_NORMSTEP_RESULTS.md |
| `run_tw_statechange.sh` | LIVE | 0 | 0 | 1 | 7 | 0 | 2026-10-08 | keep | named in TW_STATECHANGE_PREREG.md |
| `run_unfreeze.sh` | LIVE | 0 | 1 | 0 | 3 | 0 | 2026-09-10 | keep | dependency of model_em_unfreeze |
| `run_v4_control.sh` | LIVE | 0 | 2 | 3 | 1 | 0 | 2026-04-30 | keep | dependency of model_srope_components |
| `run_visits_test.sh` | LIVE | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | keep | reads/writes VISITS_TEST.md |
| `run_warm.sh` | LIVE | 0 | 0 | 0 | 19 | 0 | 2026-09-10 | keep | run dir in EM_WM_STATE.md |
| `run_wrap_cope.sh` | LIVE | 0 | 0 | 3 | 4 | 1 | 2026-10-14 | keep (md5-recorded) | named in docs/paper/APPENDIX_draft6.md |
| `run_wrap_existence.sh` | LIVE | 0 | 0 | 3 | 6 | 1 | 2026-10-13 | keep (md5-recorded) | named in docs/paper/APPENDIX_draft6.md |
| `run_wrap_mech.sh` | LIVE | 0 | 0 | 3 | 9 | 1 | 2026-10-14 | keep (md5-recorded) | named in docs/paper/APPENDIX_draft6.md |
| `safe_clear.sh` | LIVE | 0 | 0 | 5 | 5 | 0 | 2026-09-24 | keep | named in CLAUDE.md |
| `sweep_dissociation.py` | LIVE | 0 | 1 | 1 | 3 | 0 | 2026-08-19 | keep | run dir in BASELINE_TABLE.md |
| `test_epoch_callback.py` | LIVE | 0 | 3 | 3 | 0 | 0 | 2026-10-23 | keep | named in FACTORIAL_N8_PREREG.md |
| `test_exptorus.py` | LIVE | 0 | 2 | 2 | 1 | 0 | 2026-10-26 | keep | named in EXPTORUS_PREREG.md |
| `test_guards.py` | LIVE | 0 | 2 | 7 | 23 | 0 | 2026-09-11 | keep | named in CLAUDE.md |
| `test_hourglass_causal.py` | LIVE | 0 | 2 | 1 | 0 | 0 | 2026-07-23 | keep | dependency of hourglass_plain |
| `test_image_torus.py` | LIVE | 0 | 1 | 3 | 2 | 1 | 2026-10-25 | keep (md5-recorded) | named in IMAGE_TORUS_PREREG.md |
| `test_mm.py` | LIVE | 0 | 1 | 1 | 4 | 0 | 2026-10-27 | keep | named in docs/audits/2026-10-27/memory_maze_audit.md |
| `test_rankclock.py` | LIVE | 0 | 2 | 3 | 0 | 0 | 2026-10-23 | keep | named in RANKCLOCK_PREREG.md |
| `test_rnn_torus.py` | LIVE | 0 | 2 | 2 | 3 | 0 | 2026-10-26 | keep | named in RNN_TORUS_PREREG.md |
| `test_signroute.py` | LIVE | 0 | 3 | 1 | 1 | 0 | 2026-10-20 | keep | named in SIGNROUTE_PREREG.md |
| `test_signsep.py` | LIVE | 0 | 2 | 1 | 3 | 0 | 2026-10-23 | keep | named in SIGNSEP_PREREG.md |
| `test_ssm_sign.py` | LIVE | 0 | 2 | 3 | 0 | 0 | 2026-10-23 | keep | named in G3_SSM_PREREG.md |
| `tinystories_data.py` | LIVE | 0 | 1 | 1 | 0 | 0 | 2026-10-09 | keep | dependency of train_tinystories |
| `toy_wrap.py` | LIVE | 0 | 0 | 5 | 6 | 0 | 2026-10-14 | keep | named in ACCOUNTS_PREREG.md |
| `train_algorithmic.py` | LIVE | 0 | 6 | 0 | 0 | 0 | 2026-09-03 | keep | dependency of run_selective.sh |
| `train_compositional.py` | LIVE | 0 | 18 | 1 | 1 | 0 | 2026-09-08 | keep | dependency of run_comp_multiseed.sh |
| `train_continuous.py` | LIVE | 0 | 3 | 1 | 6 | 0 | 2026-05-03 | keep | dependency of run_p4_p5_fix.sh |
| `train_ctxstep3.py` | LIVE | 0 | 1 | 0 | 8 | 1 | 2026-09-30 | keep (md5-recorded) | run dir in TEXTWORLD_PREREG.md |
| `train_exptorus.py` | LIVE | 0 | 3 | 3 | 1 | 3 | 2026-10-26 | keep (md5-recorded) | named in EXPTORUS_PREREG.md |
| `train_family_tree.py` | LIVE | 0 | 4 | 0 | 0 | 0 | 2026-08-16 | keep | dependency of run_family_tree.sh |
| `train_gain_grain.py` | LIVE | 1 | 1 | 1 | 1 | 1 | 2026-10-05 | keep (md5-recorded) | named in GAIN_GRAIN_PREREG.md |
| `train_gain_phase.py` | LIVE | 0 | 1 | 1 | 1 | 1 | 2026-10-06 | keep (md5-recorded) | named in GAIN_PHASE_PREREG.md |
| `train_gauge.py` | LIVE | 0 | 3 | 13 | 6 | 3 (1 drifted) | 2026-10-16 | keep (md5-recorded) | named in G3_SSM_PREREG.md |
| `train_goal.py` | LIVE | 0 | 3 | 2 | 0 | 0 | 2026-05-11 | keep | dependency of probe_goal_linear |
| `train_hier_goal.py` | LIVE | 3 | 11 | 1 | 0 | 0 | 2026-08-09 | keep | dependency of run_pope_redo.sh |
| `train_indirect.py` | LIVE | 1 | 4 | 1 | 1 | 0 | 2026-09-18 | keep | named in DELTA_FREE_PREREG.md |
| `train_jsb.py` | LIVE | 2 | 10 | 1 | 1 | 0 | 2026-09-20 | keep | named in DELTA_FREE_PREREG.md |
| `train_miniworld.py` | LIVE | 2 | 24 | 1 | 0 | 0 | 2026-08-29 | keep | dependency of run_alias_sweep.sh |
| `train_mm.py` | LIVE | 1 | 5 | 3 | 4 | 0 | 2026-10-27 | keep | named in MEMORY_MAZE_PREREG.md |
| `train_multienv.py` | LIVE | 0 | 7 | 2 | 0 | 0 | 2026-05-10 | keep | named in README.md |
| `train_multisize.py` | LIVE | 0 | 10 | 1 | 0 | 0 | 2026-05-17 | keep | named in README.md |
| `train_pair.py` | LIVE | 0 | 1 | 1 | 2 | 1 | 2026-10-04 | keep (md5-recorded) | named in MAPPOPE_PAIR_PREREG.md |
| `train_recency.py` | LIVE | 1 | 31 | 7 | 5 | 3 | 2026-09-11 | keep (md5-recorded) | commit cited in SEARCH_RESULTS.md |
| `train_ring_dial.py` | LIVE | 0 | 3 | 3 | 5 | 2 | 2026-10-12 | keep (md5-recorded) | named in docs/audits/2026-10-12/ring_dial_audit.md |
| `train_ring_rank.py` | LIVE | 1 | 1 | 1 | 1 | 1 | 2026-10-11 | keep (md5-recorded) | named in RANK_TESTS_PREREG.md |
| `train_ring_variant.py` | LIVE | 0 | 3 | 6 | 12 | 3 (1 drifted) | 2026-10-14 | keep (md5-recorded) | named in ACCOUNTS_PREREG.md |
| `train_rnn.py` | LIVE | 1 | 3 | 2 | 2 | 1 | 2026-10-26 | keep (md5-recorded) | named in RNN_TORUS_PREREG.md |
| `train_signroute.py` | LIVE | 0 | 4 | 3 | 2 | 0 | 2026-10-20 | keep | named in G3_SSM_PREREG.md |
| `train_signsep.py` | LIVE | 1 | 3 | 1 | 3 | 3 (3 drifted) | 2026-10-23 | keep (md5-recorded) | named in SIGNSEP_PREREG.md |
| `train_ssm_sign.py` | LIVE | 0 | 1 | 4 | 5 | 2 | 2026-10-23 | keep (md5-recorded) | named in FACTORIAL_N8_PREREG.md |
| `train_stitch.py` | LIVE | 0 | 1 | 1 | 0 | 0 | 2026-08-16 | keep | dependency of run_stitch_probe.sh |
| `train_topology.py` | LIVE | 0 | 4 | 1 | 0 | 0 | 2026-05-13 | keep | named in README.md |
| `train_torus_start.py` | LIVE | 0 | 1 | 5 | 3 | 1 | 2026-10-14 | keep (md5-recorded) | named in TORUS_START_PREREG.md |
| `train_tw_ambig.py` | LIVE | 1 | 1 | 1 | 2 | 0 | 2026-10-08 | keep | commit cited in .claude-memory/project_state.md |
| `train_tw_landmark.py` | LIVE | 0 | 2 | 1 | 1 | 1 | 2026-10-08 | keep (md5-recorded) | named in TW_LANDMARK_PREREG.md |
| `train_tw_normstep.py` | LIVE | 1 | 2 | 2 | 2 | 1 | 2026-10-03 | keep (md5-recorded) | named in TW_NORMSTEP_PREREG.md |
| `train_tw_statechange.py` | LIVE | 1 | 1 | 2 | 1 | 1 | 2026-10-08 | keep (md5-recorded) | named in TW_STATECHANGE_PREREG.md |
| `tw_ambig_readouts.py` | LIVE | 1 | 1 | 1 | 2 | 0 | 2026-10-08 | keep | commit cited in .claude-memory/project_state.md |
| `tw_landmark_eval.py` | LIVE | 0 | 4 | 1 | 2 | 1 | 2026-10-08 | keep (md5-recorded) | named in TW_LANDMARK_PREREG.md |
| `tw_statechange_readouts.py` | LIVE | 1 | 1 | 1 | 11 | 1 | 2026-10-08 | keep (md5-recorded) | named in TW_STATECHANGE_PREREG.md |
| `validate_algorithmic.py` | LIVE | 0 | 1 | 0 | 1 | 0 | 2026-09-24 | keep | dependency of environment_algorithmic |
| `validate_code.py` | LIVE | 1 | 0 | 3 | 1 | 0 | 2026-09-21 | keep | dependency of eval_code_long |
| `validate_compositional.py` | LIVE | 0 | 0 | 2 | 0 | 0 | 2026-07-23 | keep | named in COMPOSITIONAL_EXPERIMENT.md |
| `validate_cscg_tasks.py` | LIVE | 0 | 1 | 0 | 1 | 0 | 2026-09-24 | keep | dependency of environment_stitch |
| `validate_dyck_mdepth.py` | LIVE | 0 | 1 | 2 | 2 | 1 | 2026-09-25 | keep (md5-recorded) | named in DYCK_MDEPTH_PREREG.md |
| `validate_family_tree.py` | LIVE | 0 | 4 | 2 | 1 | 0 | 2026-09-24 | keep | dependency of run_family_tree.sh |
| `validate_lap.py` | LIVE | 0 | 2 | 1 | 1 | 0 | 2026-09-24 | keep | dependency of make_figures |
| `validate_map_query.py` | LIVE | 0 | 3 | 0 | 1 | 0 | 2026-09-24 | keep | dependency of validate_match_query |
| `validate_match_query.py` | LIVE | 0 | 7 | 1 | 1 | 0 | 2026-09-24 | keep | dependency of train_match_query |
| `validate_nd.py` | LIVE | 0 | 1 | 1 | 0 | 0 | 2026-09-06 | keep | named in RANK_ND_PREREG.md |
| `validate_recency.py` | LIVE | 0 | 2 | 2 | 3 | 0 | 2026-09-24 | keep | commit cited in SEARCH_RESULTS.md |
| `verify_addition_fast.py` | LIVE | 0 | 1 | 0 | 0 | 0 | 2026-09-15 | keep | dependency of environment_addition |
| `verify_comp_speedups.py` | LIVE | 0 | 1 | 0 | 2 | 0 | 2026-09-08 | keep | reads/writes COMP_HEADROOM.md |
| `zero_shot_eval.py` | LIVE | 0 | 4 | 6 | 0 | 0 | 2026-09-24 | keep | named in README.md |
| `ablate_hiergoal.py` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: HIERGOAL_ABLATION.md |
| `ablate_match_noise.py` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: MATCH_QUERY_NOISE_ABLATION.md |
| `agg_algorithmic.py` | HISTORICAL | 0 | 1 | 0 | 1 | 0 | 2026-09-03 | archive w/ path map | non-live evidence: ALGORITHMIC_GATES.md |
| `agg_clock.py` | HISTORICAL | 0 | 4 | 0 | 0 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: driven by run_bestofboth.sh |
| `agg_cmq.py` | HISTORICAL | 0 | 7 | 0 | 1 | 0 | 2026-08-23 | archive w/ path map | non-live evidence: COMPOSITIONAL_MATCH_QUERY_RESULTS.md |
| `agg_frontier.py` | HISTORICAL | 0 | 1 | 0 | 0 | 0 | 2026-09-03 | archive w/ path map | non-live evidence: driven by run_frontier.sh |
| `agg_hier_parity.py` | HISTORICAL | 0 | 1 | 0 | 1 | 0 | 2026-09-03 | archive w/ path map | non-live evidence: ALGORITHMIC_RESULTS.md |
| `agg_loop_hier.py` | HISTORICAL | 0 | 1 | 0 | 3 | 0 | 2026-09-03 | archive w/ path map | non-live evidence: ALGORITHMIC_RESULTS.md |
| `agg_looped.py` | HISTORICAL | 0 | 1 | 0 | 1 | 0 | 2026-08-29 | archive w/ path map | non-live evidence: HORIZON_RESULTS.md |
| `agg_mq_noise.py` | HISTORICAL | 0 | 2 | 0 | 0 | 0 | 2026-09-02 | archive w/ path map | non-live evidence: driven by run_mq_noise_2x2.sh |
| `agg_recipe.py` | HISTORICAL | 0 | 1 | 0 | 0 | 0 | 2026-09-02 | archive w/ path map | non-live evidence: driven by run_recipe_power.sh |
| `aliasing_covariate.py` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-19 | archive w/ path map | non-live evidence: ALIASING_COVARIATE.md |
| `analyze_clock_speed.py` | HISTORICAL | 1 | 2 | 2 | 4 | 2 | 2026-10-12 | keep or archive w/ md5 note | non-live evidence: CLOCK_SPEED_PREREG.md |
| `analyze_clock_speed_rep.py` | HISTORICAL | 0 | 1 | 2 | 3 | 1 | 2026-10-13 | keep or archive w/ md5 note | non-live evidence: CLOCK_SPEED_REP_PREREG.md |
| `analyze_code.py` | HISTORICAL | 0 | 0 | 1 | 3 | 0 | 2026-09-21 | archive w/ path map | non-live evidence: CODE_GATES.md |
| `analyze_code_long.py` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-21 | archive w/ path map | non-live evidence: CODE_PREREG.md |
| `analyze_ctxstep.py` | HISTORICAL | 0 | 1 | 0 | 1 | 0 | 2026-09-30 | archive w/ path map | non-live evidence: CTXSTEP_PREREG.md |
| `analyze_jsb_base.py` | HISTORICAL | 0 | 1 | 0 | 4 | 0 | 2026-09-17 | archive w/ path map | non-live evidence: JSBLEN_PREREG.md |
| `analyze_jsb_len.py` | HISTORICAL | 0 | 2 | 0 | 1 | 0 | 2026-09-17 | archive w/ path map | non-live evidence: JSBLEN_PREREG.md |
| `analyze_noleak.py` | HISTORICAL | 0 | 0 | 1 | 4 | 0 | 2026-09-11 | archive w/ path map | non-live evidence: NOLEAK_PREREG.md |
| `analyze_papertask.py` | HISTORICAL | 0 | 1 | 0 | 2 | 0 | 2026-09-12 | archive w/ path map | non-live evidence: PAPERTASK_PREREG.md |
| `analyze_rank_nowrap.py` | HISTORICAL | 0 | 1 | 1 | 1 | 0 | 2026-10-08 | archive w/ path map | non-live evidence: RANK_NOWRAP_PREREG.md |
| `analyze_sameblock.py` | HISTORICAL | 0 | 0 | 1 | 3 | 0 | 2026-09-15 | archive w/ path map | non-live evidence: SAMEBLOCK_PREREG.md |
| `analyze_spread2.py` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-12 | archive w/ path map | non-live evidence: REPORT.md |
| `analyze_t2.py` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-20 | archive w/ path map | non-live evidence: T2_RESULTS.md |
| `ap_kernel_diagnostic.py` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: AP_KERNEL_DIAGNOSTIC.md |
| `audit_planner_tasks.py` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: PLANNER_TASK_AUDIT.md |
| `autocommit_bob.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-07-29 | archive w/ path map | non-live evidence: COMPOSITIONAL_MULTISEED.md |
| `autocommit_longt.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-08 | archive w/ path map | non-live evidence: EM_FIX_COMP.md |
| `aux_coef_sweep.py` | HISTORICAL | 0 | 1 | 1 | 0 | 0 | 2026-04-27 | archive w/ path map | non-live evidence: docs/LOG.md |
| `bench_compute.py` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: LOOP_HIER_COMPUTE.md |
| `build_code_corpus.py` | HISTORICAL | 0 | 0 | 2 | 0 | 0 | 2026-09-21 | archive w/ path map | non-live evidence: CODE_PREREG.md |
| `clone_analysis.py` | HISTORICAL | 0 | 2 | 2 | 0 | 0 | 2026-04-20 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `clone_transfer_test.py` | HISTORICAL | 0 | 2 | 2 | 0 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: CLONE_TRANSFER_NOBYPASS.md |
| `decide_frontier.py` | HISTORICAL | 0 | 2 | 1 | 1 | 0 | 2026-09-03 | archive w/ path map | non-live evidence: ALGORITHMIC_RESULTS.md |
| `diagnose.py` | HISTORICAL | 0 | 0 | 2 | 0 | 0 | 2026-04-16 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `docs/paper/expansion/scripts/depth/depthlib.py` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-10-20 | archive w/ path map | non-live evidence: docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md |
| `docs/paper/expansion/scripts/depth/segments_bins.py` | HISTORICAL | 0 | 1 | 2 | 0 | 0 | 2026-10-21 | archive w/ path map | non-live evidence: docs/paper/MAIN_DRAFT2_notes.md |
| `docs/paper/expansion/scripts/depth/walk_stats.py` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-10-20 | archive w/ path map | non-live evidence: docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md |
| `docs/paper/tools/build_pdfs.py` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-10-20 | archive w/ path map | non-live evidence: docs/RUNBOOK_NEXT_EXPERIMENTS.md |
| `docs/theory/2026-10-04/scripts/toy2.py` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-10-04 | archive w/ path map | non-live evidence: docs/theory/2026-10-04/03_formal.md |
| `doorkey_solver.py` | HISTORICAL | 2 | 1 | 2 | 0 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: RESULTS_SUMMARY_2026-05-10.md |
| `dp_reference.py` | HISTORICAL | 0 | 1 | 0 | 0 | 0 | 2026-09-01 | archive w/ path map | non-live evidence: driven by wire_data_parallel |
| `environment_compositional_match_query.py` | HISTORICAL | 2 | 0 | 1 | 0 | 0 | 2026-08-23 | archive w/ path map | non-live evidence: COMPOSITIONAL_MATCH_QUERY.md |
| `environment_flipflop.py` | HISTORICAL | 2 | 0 | 0 | 0 | 0 | 2026-09-07 | archive w/ path map | non-live evidence: imported by train_flipflop |
| `environment_maze_varying.py` | HISTORICAL | 3 | 0 | 0 | 0 | 0 | 2026-07-22 | archive w/ path map | non-live evidence: imported by audit_planner_tasks |
| `environment_mqar.py` | HISTORICAL | 1 | 0 | 0 | 1 | 0 | 2026-09-08 | archive w/ path map | non-live evidence: MQAR_PREREG.md |
| `environment_multiclass.py` | HISTORICAL | 1 | 1 | 0 | 0 | 0 | 2026-05-13 | archive w/ path map | non-live evidence: imported by train_multiclass |
| `environment_nd_redraw.py` | HISTORICAL | 1 | 1 | 0 | 0 | 0 | 2026-10-06 | archive w/ path map | non-live evidence: imported by train_rank_nowrap |
| `environment_rooms.py` | HISTORICAL | 1 | 0 | 0 | 1 | 0 | 2026-07-18 | archive w/ path map | non-live evidence: HIER_ATTN_LONGT.md |
| `environment_rooms_goal.py` | HISTORICAL | 3 | 0 | 0 | 0 | 0 | 2026-07-21 | archive w/ path map | non-live evidence: imported by audit_planner_tasks |
| `environment_rooms_maze.py` | HISTORICAL | 3 | 0 | 0 | 0 | 0 | 2026-07-21 | archive w/ path map | non-live evidence: imported by audit_planner_tasks |
| `eval_ablation_miniworld.py` | HISTORICAL | 0 | 2 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_FRESH_ABLATION.md |
| `eval_bounded_memory.py` | HISTORICAL | 0 | 1 | 0 | 1 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: BOUNDED_MEMORY.md |
| `eval_hier_goal_closedloop.py` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: HIERGOAL_CLOSEDLOOP.md |
| `eval_hiergoal_longT.py` | HISTORICAL | 0 | 1 | 0 | 1 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: HIERGOAL_LONGT.md |
| `eval_indirect_ood.py` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-17 | archive w/ path map | non-live evidence: INDIRECT_OOD.md |
| `eval_knob_sweep.py` | HISTORICAL | 0 | 2 | 0 | 1 | 0 | 2026-08-21 | archive w/ path map | non-live evidence: KNOB_SWEEP.md |
| `eval_loop_sweep.py` | HISTORICAL | 0 | 1 | 0 | 0 | 0 | 2026-08-31 | archive w/ path map | non-live evidence: driven by run_loop_sampled.sh |
| `eval_rank_nowrap.py` | HISTORICAL | 0 | 1 | 1 | 0 | 0 | 2026-10-06 | archive w/ path map | non-live evidence: RANK_NOWRAP_PREREG.md |
| `eval_single_env.py` | HISTORICAL | 0 | 7 | 2 | 1 | 0 | 2026-05-21 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `eval_vocab_sweep.py` | HISTORICAL | 0 | 1 | 0 | 2 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: VOCAB_SWEEP_MULTISEED.md |
| `gaussian_noise_test.py` | HISTORICAL | 0 | 0 | 2 | 0 | 0 | 2026-04-20 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `hand_indirect.py` | HISTORICAL | 0 | 0 | 2 | 2 | 0 | 2026-09-23 | archive w/ path map | non-live evidence: INDIRECT_ARITHMETIC.md |
| `held_out_obs_eval.py` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: SESSION_HIERARCHICAL_CASCADE.md |
| `hippocampal_analysis.py` | HISTORICAL | 1 | 1 | 2 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: HIPPOCAMPAL_ANALYSIS.md |
| `hippocampal_hidden_eval.py` | HISTORICAL | 0 | 2 | 5 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: HIPPOCAMPAL_GRID.md |
| `horizon_task_distances.py` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-19 | archive w/ path map | non-live evidence: HORIZON_TASK_DISTANCES.md |
| `hyak/mm_gate_v3.slurm` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-10-27 | archive w/ path map | non-live evidence: MEMORY_MAZE_GATE_V3.md |
| `landmark_eval.py` | HISTORICAL | 0 | 0 | 2 | 0 | 0 | 2026-04-20 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `length_diagnostic.py` | HISTORICAL | 0 | 1 | 2 | 0 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `make_paper_figures.py` | HISTORICAL | 1 | 1 | 2 | 0 | 0 | 2026-04-24 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `make_per_visit_figure.py` | HISTORICAL | 0 | 2 | 0 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: PER_VISIT_clean.md |
| `make_place_cell_figure.py` | HISTORICAL | 0 | 1 | 1 | 0 | 0 | 2026-05-14 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `master_level15pc.sh` | HISTORICAL | 0 | 0 | 0 | 7 | 0 | 2026-04-26 | archive w/ path map | non-live evidence: CLONE_ANALYSIS_LEVEL15PC.md |
| `model_inekf_proper.py` | HISTORICAL | 3 | 0 | 1 | 0 | 0 | 2026-04-16 | archive w/ path map | non-live evidence: docs/LOG.md |
| `number_line_env.py` | HISTORICAL | 1 | 1 | 0 | 0 | 0 | 2026-05-21 | archive w/ path map | non-live evidence: imported by train_numberline |
| `orchestrator_baselines.py` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-04-22 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `orchestrator_em.py` | HISTORICAL | 0 | 0 | 2 | 0 | 0 | 2026-04-22 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `orchestrator_level15_em.py` | HISTORICAL | 0 | 1 | 2 | 0 | 0 | 2026-04-22 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `orchestrator_level15pc.py` | HISTORICAL | 0 | 1 | 1 | 0 | 0 | 2026-04-26 | archive w/ path map | non-live evidence: docs/LOG.md |
| `orchestrator_multilayer.py` | HISTORICAL | 0 | 0 | 2 | 0 | 0 | 2026-04-22 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `per_visit_eval.py` | HISTORICAL | 0 | 3 | 1 | 0 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: PER_VISIT_clean.md |
| `probe_forget.py` | HISTORICAL | 0 | 1 | 0 | 0 | 0 | 2026-09-05 | archive w/ path map | non-live evidence: driven by run_forget.sh |
| `probe_gate.py` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: GATE_PROBE.md |
| `probe_goal_distance_state.py` | HISTORICAL | 0 | 1 | 2 | 0 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: KNOWN_BUGS.md |
| `probe_position_decode.py` | HISTORICAL | 0 | 1 | 2 | 0 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: KNOWN_BUGS.md |
| `probe_rank_learned.py` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: LEARNED_RANK.md |
| `probe_rank_truncate.py` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: LEARNED_RANK.md |
| `probe_sr.py` | HISTORICAL | 0 | 1 | 0 | 0 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: driven by run_probe_sr.sh |
| `probe_theory_numbers.py` | HISTORICAL | 0 | 0 | 0 | 8 | 0 | 2026-09-19 | archive w/ path map | non-live evidence: JSB_LENGTH_RESULTS.md |
| `r_t_distribution_test.py` | HISTORICAL | 0 | 1 | 2 | 0 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `rescale_eval.py` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: OMEGA_RESCALE_clean.md |
| `retrain_level15em.sh` | HISTORICAL | 0 | 0 | 2 | 1 | 0 | 2026-04-24 | archive w/ path map | non-live evidence: DETAILED_RESULTS.md |
| `run_accumulator.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-06 | archive w/ path map | non-live evidence: ACCUMULATOR.md |
| `run_algorithmic.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: ALGORITHMIC_GATES.md |
| `run_alias_followup.sh` | HISTORICAL | 0 | 1 | 0 | 3 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: MINIWORLD_ENDPOINTS.md |
| `run_alias_gates.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: ALIASING_GATES.md |
| `run_alias_waveb.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: MINIWORLD_ENDPOINTS.md |
| `run_allocentric.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-22 | archive w/ path map | non-live evidence: KNOB_SWEEP.md |
| `run_aug.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-20 | archive w/ path map | non-live evidence: docs/LOG.md |
| `run_bestofboth.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-07-28 | archive w/ path map | non-live evidence: CLOCK_SCAN.md |
| `run_bestofboth_seeds.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-07-29 | archive w/ path map | non-live evidence: COMPOSITIONAL_MULTISEED.md |
| `run_bounded_mem.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: BOUNDED_MEMORY.md |
| `run_capacity_control.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-05-21 | archive w/ path map | non-live evidence: CAPACITY_CONTROL.md |
| `run_capacity_perregime.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-05-22 | archive w/ path map | non-live evidence: CAPACITY_CONTROL.md |
| `run_clock_multiseed.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-07-28 | archive w/ path map | non-live evidence: CLOCK_SCAN.md |
| `run_clock_speed.sh` | HISTORICAL | 0 | 0 | 2 | 4 | 1 | 2026-10-12 | keep or archive w/ md5 note | non-live evidence: CLOCK_SPEED_PREREG.md |
| `run_clock_speed_rep.sh` | HISTORICAL | 0 | 0 | 1 | 3 | 1 | 2026-10-12 | keep or archive w/ md5 note | non-live evidence: CLOCK_SPEED_REP_PREREG.md |
| `run_cmq_curric.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-23 | archive w/ path map | non-live evidence: COMPOSITIONAL_MATCH_QUERY.md |
| `run_cmq_curric_full.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-24 | archive w/ path map | non-live evidence: COMPOSITIONAL_MATCH_QUERY.md |
| `run_cmq_stab.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-23 | archive w/ path map | non-live evidence: COMPOSITIONAL_MATCH_QUERY.md |
| `run_cmq_sweep.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-23 | archive w/ path map | non-live evidence: COMPOSITIONAL_MATCH_QUERY_RESULTS.md |
| `run_coarseidx.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-07-27 | archive w/ path map | non-live evidence: COMPOSITIONAL_MULTISEED.md |
| `run_coarsepi.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-07-27 | archive w/ path map | non-live evidence: COMPOSITIONAL_MULTISEED.md |
| `run_comprehensive_followup.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-05-07 | archive w/ path map | non-live evidence: MULTISEED_FOLLOWUP.md |
| `run_counter_batch.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-14 | archive w/ path map | non-live evidence: COUNTER_BATCH.md |
| `run_cross_jsb.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-20 | archive w/ path map | non-live evidence: docs/LOG.md |
| `run_ctxstep.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-30 | archive w/ path map | non-live evidence: CTXSTEP_PREREG.md |
| `run_dagger.sh` | HISTORICAL | 0 | 0 | 2 | 1 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: DAGGER_RESULTS.md |
| `run_dagger_dk6.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: DAGGER_DK6_RESULTS.md |
| `run_dagger_empty.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: DAGGER_EMPTY_RESULTS.md |
| `run_decay.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-19 | archive w/ path map | non-live evidence: DECAY_PREREG.md |
| `run_diagnostic_and_v3.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-04-28 | archive w/ path map | non-live evidence: V3_RESULTS.md |
| `run_dog_test.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-04-30 | archive w/ path map | non-live evidence: DOG_RESULTS.md |
| `run_doorkey_bc.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: DOORKEY_BC_RESULTS.md |
| `run_dxr.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-04 | archive w/ path map | non-live evidence: DXR_RANK_THRESHOLD.md |
| `run_em_comp_samebatch.sh` | HISTORICAL | 0 | 1 | 0 | 2 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: EM_COMP_SAMEBATCH.md |
| `run_em_fig4.sh` | HISTORICAL | 0 | 0 | 3 | 4 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: .claude-memory/project_rank_and_selective_rope.md |
| `run_em_fix.sh` | HISTORICAL | 0 | 1 | 0 | 2 | 0 | 2026-08-08 | archive w/ path map | non-live evidence: EM_FIX_COMP.md |
| `run_em_perscale_and_hopfield.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-18 | archive w/ path map | non-live evidence: EM_HOPFIELD_CROSSSCALE.md |
| `run_enwik8_2x2.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: ENWIK8_2X2.md |
| `run_enwik8_comp.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-22 | archive w/ path map | non-live evidence: ENWIK8_COMPOSITION_PREREG.md |
| `run_enwik8_hier.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: ENWIK8_HIER.md |
| `run_enwik8_long.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: ENWIK8_LONG.md |
| `run_enwik8_seeds.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: ENWIK8_SEEDS.md |
| `run_family_tree_d7.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-16 | archive w/ path map | non-live evidence: FAMILY_TREE_D7_GATES.md |
| `run_final_evals.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-23 | archive w/ path map | non-live evidence: CONTINUOUS_ALLOC.md |
| `run_fixup.sh` | HISTORICAL | 0 | 0 | 2 | 1 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_flipflop.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-07 | archive w/ path map | non-live evidence: FLIPFLOP_GATES.md |
| `run_forget.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-05 | archive w/ path map | non-live evidence: FORGET_GATE.md |
| `run_forget_clock.sh` | HISTORICAL | 0 | 0 | 3 | 1 | 0 | 2026-09-08 | archive w/ path map | non-live evidence: FORGET_CLOCK_PREREG.md |
| `run_freq_control.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-21 | archive w/ path map | non-live evidence: FREQ_CONTROL.md |
| `run_frontier.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: FRONTIER_ALGORITHMIC.md |
| `run_g8_converge.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: CROSSOVER_CONVERGED.md |
| `run_gsf_clean_noise.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-11 | archive w/ path map | non-live evidence: GSF_FULL_RESULTS.md |
| `run_gsf_modes.sh` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: docs/LOG.md |
| `run_gsf_nodrop.sh` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: docs/LOG.md |
| `run_h12_budget.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-08-23 | archive w/ path map | non-live evidence: CONTINUOUS_ALLOC.md |
| `run_hier_parity.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: HIER_PARITY.md |
| `run_hopfield_nomainap.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-21 | archive w/ path map | non-live evidence: HOPFIELD_NOMAINAP_RESULTS.md |
| `run_hourglass_experiments.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-07-23 | archive w/ path map | non-live evidence: COMPOSITIONAL_RESULTS.md |
| `run_index_baseline_paper.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-08-17 | archive w/ path map | non-live evidence: INDEX_BASELINE_PAPER_TASK.md |
| `run_interference_tests.sh` | HISTORICAL | 0 | 1 | 1 | 0 | 0 | 2026-04-27 | archive w/ path map | non-live evidence: docs/LOG.md |
| `run_jsb_base.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-17 | archive w/ path map | non-live evidence: JSB_LENGTH_RESULTS_BASE.md |
| `run_jsb_len.sh` | HISTORICAL | 0 | 0 | 0 | 5 | 0 | 2026-09-17 | archive w/ path map | non-live evidence: JSBLEN_PREREG.md |
| `run_jsb_rank.sh` | HISTORICAL | 0 | 0 | 0 | 4 | 0 | 2026-09-17 | archive w/ path map | non-live evidence: JSB_LENGTH_RESULTS_RANK.md |
| `run_knob_eval.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-21 | archive w/ path map | non-live evidence: KNOB_SWEEP.md |
| `run_knob_sweep.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-08-21 | archive w/ path map | non-live evidence: FREQ_CONTROL.md |
| `run_lam.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-20 | archive w/ path map | non-live evidence: docs/LOG.md |
| `run_lap_transfer.py` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-08-15 | archive w/ path map | non-live evidence: LAP_TRANSFER.md |
| `run_level15_beta.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: LEVEL15BETA_RESULTS.md |
| `run_level15_novelenv_fillin.sh` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_level15em_crossscale.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-18 | archive w/ path map | non-live evidence: LEVEL15EM_CROSSSCALE.md |
| `run_loop_hier.sh` | HISTORICAL | 0 | 0 | 0 | 4 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: HIER_PARITY.md |
| `run_loop_sampled.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: LOOP_SAMPLED.md |
| `run_map_query.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: MAP_QUERY_GATES.md |
| `run_mappope_r4.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-07 | archive w/ path map | non-live evidence: MAPPOPE_R4.md |
| `run_match_query_em.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-15 | archive w/ path map | non-live evidence: MATCH_QUERY_EM.md |
| `run_minigrid_allo8.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-08-26 | archive w/ path map | non-live evidence: AUDIT_HEADLINE.md |
| `run_minigrid_cached_full.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-05-01 | archive w/ path map | non-live evidence: MINIGRID_DOORKEY_CACHED.md |
| `run_minigrid_cached_test.sh` | HISTORICAL | 0 | 1 | 0 | 1 | 0 | 2026-05-01 | archive w/ path map | non-live evidence: SESSION_2026-05-01.md |
| `run_minigrid_dk16.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-05-01 | archive w/ path map | non-live evidence: MINIGRID_DK16_RESULTS.md |
| `run_minigrid_doorkey.sh` | HISTORICAL | 0 | 1 | 0 | 2 | 0 | 2026-05-01 | archive w/ path map | non-live evidence: MINIGRID_DOORKEY_RESULTS.md |
| `run_minigrid_em.sh` | HISTORICAL | 0 | 1 | 0 | 2 | 0 | 2026-09-08 | archive w/ path map | non-live evidence: MINIGRID_EM.md |
| `run_minigrid_em_fix.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-08 | archive w/ path map | non-live evidence: MINIGRID_EM_FIX.md |
| `run_minigrid_longt.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-01 | archive w/ path map | non-live evidence: MINIGRID_DOORKEY_LONGT.md |
| `run_minigrid_rope_diag.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-05-01 | archive w/ path map | non-live evidence: MINIGRID_DOORKEY_RESULTS.md |
| `run_monotone.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-13 | archive w/ path map | non-live evidence: MONOTONE_PREREG.md |
| `run_mq_noise_2x2.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: MQ_NOISE_2X2.md |
| `run_mq_noise_c2.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: MQ_NOISE_2X2_C2.md |
| `run_multiclass.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-13 | archive w/ path map | non-live evidence: MULTICLASS_RESULTS.md |
| `run_multiclass_multiseed.sh` | HISTORICAL | 0 | 0 | 2 | 1 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: MULTICLASS_MULTISEED_RESULTS.md |
| `run_multienv_clean_baselines.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_multiseed_followup.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-14 | archive w/ path map | non-live evidence: MULTISEED_FOLLOWUP_RESULTS.md |
| `run_mw_fresh_factorial.sh` | HISTORICAL | 0 | 0 | 1 | 4 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_FRESH_GATES_ALLO.md |
| `run_mw_fresh_probe.sh` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_TODO.md |
| `run_mw_gatectl.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_GATE_CONTROL.md |
| `run_mw_grid_sweep.sh` | HISTORICAL | 0 | 5 | 0 | 6 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_GRID16_GATES.md |
| `run_mw_grid_sweep_hier.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_GRID_SWEEP_HIER.md |
| `run_mw_hier_ablation.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: KNOWN_BUGS.md |
| `run_mw_missing.sh` | HISTORICAL | 0 | 0 | 1 | 2 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_FIXED_RESULTS.md |
| `run_mw_oracle.sh` | HISTORICAL | 0 | 0 | 0 | 4 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_ORACLE_ABLATION.md |
| `run_mw_probe3.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_PROBE3.md |
| `run_mw_probe3_seeds.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_SROPE_COMPONENTS.md |
| `run_mw_srope_components.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIWORLD_SROPE_COMPONENTS.md |
| `run_n5_phase.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-09 | archive w/ path map | non-live evidence: N5_PREREG.md |
| `run_nobypass_test.sh` | HISTORICAL | 0 | 0 | 2 | 3 | 0 | 2026-04-27 | archive w/ path map | non-live evidence: CLONE_TRANSFER_NOBYPASS.md |
| `run_nodrop_multiseed.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: NODROP_PARETO_RESULTS.md |
| `run_noleak.sh` | HISTORICAL | 0 | 0 | 0 | 4 | 0 | 2026-09-11 | archive w/ path map | non-live evidence: NOLEAK_PREREG.md |
| `run_numberline.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-21 | archive w/ path map | non-live evidence: NUMBERLINE_RESULTS.md |
| `run_ood_grid.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-05-11 | archive w/ path map | non-live evidence: OOD_GRID_RESULTS.md |
| `run_paper_validation.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: PAPER_VALIDATION.md |
| `run_papertask_eval.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-12 | archive w/ path map | non-live evidence: PAPERTASK_PREREG.md |
| `run_perscale_omega.sh` | HISTORICAL | 0 | 0 | 2 | 1 | 0 | 2026-05-17 | archive w/ path map | non-live evidence: PERSCALE_OMEGA_RESULTS.md |
| `run_planning_and_scale_controls.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_pope.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-07-28 | archive w/ path map | non-live evidence: COMPOSITIONAL_MULTISEED.md |
| `run_pope_faithful.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-07-29 | archive w/ path map | non-live evidence: CLOCK_SCAN.md |
| `run_pope_hier.sh` | HISTORICAL | 0 | 1 | 0 | 0 | 0 | 2026-08-23 | archive w/ path map | non-live evidence: driven by run_pope_hier_allo.sh |
| `run_pope_hier_allo.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: MINIGRID_ALLOCENTRIC_2X2X2.md |
| `run_pope_paper.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-17 | archive w/ path map | non-live evidence: INDEX_BASELINE_PAPER_TASK.md |
| `run_popewrap.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-05 | archive w/ path map | non-live evidence: POPE_WRAPPING.md |
| `run_post_pipeline_analysis.sh` | HISTORICAL | 0 | 0 | 2 | 2 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_rank_nowrap.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-10-06 | archive w/ path map | non-live evidence: RANK_NOWRAP_PREREG.md |
| `run_recency_t3.sh` | HISTORICAL | 0 | 0 | 1 | 4 | 0 | 2026-09-19 | archive w/ path map | non-live evidence: RECENCY_T3_PREREG.md |
| `run_recipe_power.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: RECIPE_POWER.md |
| `run_rope_converge.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: ROPE_CONVERGE.md |
| `run_sameblock.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-15 | archive w/ path map | non-live evidence: SAMEBLOCK_PREREG.md |
| `run_seeds_n8.sh` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-08-22 | archive w/ path map | non-live evidence: report/report.tex |
| `run_spread2.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-12 | archive w/ path map | non-live evidence: PAPERTASK_PREREG.md |
| `run_spread_cal.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-11 | archive w/ path map | non-live evidence: PAIRORIGIN_RESULTS.md |
| `run_sr_pretrain_and_probe.sh` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-05-17 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_state_probes.sh` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_t1_centered.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-18 | archive w/ path map | non-live evidence: T1_PREREG.md |
| `run_t2.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-19 | archive w/ path map | T2 test driver (not a version of run_t*); non-live |
| `run_t2b.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-20 | archive w/ path map | non-live evidence: T2_RESULTS.md |
| `run_t2c.sh` | HISTORICAL | 0 | 0 | 0 | 3 | 0 | 2026-09-20 | archive w/ path map | non-live evidence: T2_RESULTS.md |
| `run_t3gen2.sh` | HISTORICAL | 0 | 0 | 0 | 4 | 0 | 2026-09-19 | archive w/ path map | non-live evidence: DYCK_T3_RESULTS.md |
| `run_tem_background_baselines.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_tem_diag.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-14 | archive w/ path map | non-live evidence: TEM_RECENCY_DIAG.md |
| `run_tem_noise_and_ffn.sh` | HISTORICAL | 0 | 0 | 2 | 4 | 0 | 2026-05-21 | archive w/ path map | non-live evidence: NODROP_PARETO_RESULTS.md |
| `run_tem_novel_envs.sh` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_tem_t_multiseed.sh` | HISTORICAL | 0 | 0 | 2 | 3 | 0 | 2026-05-04 | archive w/ path map | non-live evidence: RESULTS_PAPER.md |
| `run_tinystories.sh` | HISTORICAL | 0 | 0 | 1 | 3 | 0 | 2026-10-09 | archive w/ path map | non-live evidence: TINYSTORIES_PILOT.md |
| `run_topology.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-13 | archive w/ path map | non-live evidence: TOPOLOGY_RESULTS.md |
| `run_v4_multiseed.sh` | HISTORICAL | 0 | 0 | 2 | 3 | 0 | 2026-04-29 | archive w/ path map | non-live evidence: SESSION_2026-05-01.md |
| `run_v4_test.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-04-28 | archive w/ path map | non-live evidence: V4_RESULTS.md |
| `run_vanilla_converge.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-08-28 | archive w/ path map | non-live evidence: POSITION_EFFECT_CONVERGED.md |
| `run_vanilla_nodrop_control.sh` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-05-15 | archive w/ path map | non-live evidence: REPORT_ADDENDUM.md |
| `run_vector_nav_v2.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-05-13 | archive w/ path map | non-live evidence: VECTOR_NAV_V2_RESULTS.md |
| `run_vocab_em.sh` | HISTORICAL | 0 | 0 | 0 | 2 | 0 | 2026-09-09 | archive w/ path map | non-live evidence: VOCAB_EM.md |
| `run_vocab_samebatch.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: VOCAB_SWEEP_MULTISEED.md |
| `run_vocab_sweep.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: VOCAB_SWEEP_RESULTS.md |
| `run_wire_dp.sh` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: docs/audits/2026-09-24/README.md |
| `train_addition.py` | HISTORICAL | 1 | 5 | 0 | 3 | 0 | 2026-09-15 | archive w/ path map | non-live evidence: ADDITION_DESIGN.md |
| `train_aggregate.py` | HISTORICAL | 1 | 0 | 0 | 1 | 0 | 2026-07-17 | archive w/ path map | non-live evidence: HIER_ATTN_LONGT.md |
| `train_clock.py` | HISTORICAL | 0 | 4 | 0 | 0 | 0 | 2026-07-28 | archive w/ path map | non-live evidence: driven by run_bestofboth.sh |
| `train_compositional_match_query.py` | HISTORICAL | 0 | 7 | 0 | 0 | 0 | 2026-08-23 | archive w/ path map | non-live evidence: driven by run_cmq_curric.sh |
| `train_doorkey_bc.py` | HISTORICAL | 0 | 3 | 2 | 0 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: RESULTS_SUMMARY_2026-05-10.md |
| `train_doorkey_dagger.py` | HISTORICAL | 0 | 3 | 2 | 0 | 0 | 2026-05-10 | archive w/ path map | non-live evidence: RESULTS_SUMMARY_2026-05-10.md |
| `train_flipflop.py` | HISTORICAL | 0 | 1 | 0 | 1 | 0 | 2026-09-07 | archive w/ path map | non-live evidence: FLIPFLOP_GATES.md |
| `train_lap.py` | HISTORICAL | 1 | 0 | 1 | 0 | 0 | 2026-08-14 | archive w/ path map | non-live evidence: docs/LOG.md |
| `train_map_query.py` | HISTORICAL | 0 | 1 | 0 | 0 | 0 | 2026-08-09 | archive w/ path map | non-live evidence: driven by run_map_query.sh |
| `train_mqar.py` | HISTORICAL | 0 | 0 | 0 | 1 | 0 | 2026-09-08 | archive w/ path map | non-live evidence: MQAR_PREREG.md |
| `train_multiclass.py` | HISTORICAL | 0 | 2 | 0 | 0 | 0 | 2026-05-13 | archive w/ path map | non-live evidence: driven by run_multiclass.sh |
| `train_numberline.py` | HISTORICAL | 0 | 2 | 0 | 0 | 0 | 2026-05-21 | archive w/ path map | non-live evidence: driven by run_capacity_perregime.sh |
| `train_rank_nowrap.py` | HISTORICAL | 2 | 1 | 1 | 1 | 0 | 2026-10-06 | archive w/ path map | non-live evidence: RANK_NOWRAP_PREREG.md |
| `validate_addition.py` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-09-14 | archive w/ path map | non-live evidence: ADDITION_DESIGN.md |
| `validate_compositional_match_query.py` | HISTORICAL | 0 | 2 | 0 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: COMPOSITIONAL_MATCH_QUERY_GATES.md |
| `validate_flipflop.py` | HISTORICAL | 0 | 1 | 0 | 1 | 0 | 2026-09-07 | archive w/ path map | non-live evidence: FLIPFLOP_GATES.md |
| `validate_hier_goal.py` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-07-26 | archive w/ path map | non-live evidence: docs/LOG.md |
| `validate_miniworld.py` | HISTORICAL | 0 | 4 | 1 | 0 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: MINIWORLD_TODO.md |
| `validate_task.py` | HISTORICAL | 0 | 0 | 2 | 0 | 0 | 2026-07-21 | archive w/ path map | non-live evidence: .claude-memory/feedback_validate_task_first.md |
| `verify_addition_compile.py` | HISTORICAL | 0 | 0 | 3 | 0 | 0 | 2026-09-15 | archive w/ path map | non-live evidence: .claude-memory/feedback_scheduler_and_measurement_traps.md |
| `verify_data_parallel.py` | HISTORICAL | 0 | 0 | 1 | 0 | 0 | 2026-09-01 | archive w/ path map | non-live evidence: docs/LOG.md |
| `verify_inclusion.py` | HISTORICAL | 0 | 1 | 0 | 0 | 0 | 2026-09-04 | archive w/ path map | non-live evidence: driven by verify_theory |
| `watch_minigrid_doorkey.sh` | HISTORICAL | 0 | 0 | 1 | 1 | 0 | 2026-05-01 | archive w/ path map | non-live evidence: MINIGRID_DOORKEY_RESULTS.md |
| `wire_data_parallel.py` | HISTORICAL | 0 | 1 | 1 | 1 | 0 | 2026-09-24 | archive w/ path map | non-live evidence: docs/audits/2026-09-24/README.md |
| `analyze_t1.py` | SUPERSEDED | 0 | 1 | 1 | 3 | 0 | 2026-09-18 | archive | newer version exists: analyze_t2 |
| `docs/theory/2026-10-04/scripts/toy.py` | SUPERSEDED | 0 | 0 | 1 | 0 | 0 | 2026-10-04 | archive | newer version exists: toy2 |
| `environment_textworld_ctx.py` | SUPERSEDED | 1 | 0 | 1 | 1 | 0 | 2026-09-28 | archive | newer version exists: environment_textworld_ctx2, environment_textworld_ctx3 |
| `environment_textworld_ctx2.py` | SUPERSEDED | 1 | 0 | 2 | 1 | 0 | 2026-09-28 | archive | newer version exists: environment_textworld_ctx3 |
| `hyak/mm_gate_v2.slurm` | SUPERSEDED | 0 | 0 | 1 | 1 | 0 | 2026-10-27 | archive | newer version exists: mm_gate_v3 |
| `probe_vector_nav.py` | SUPERSEDED | 0 | 1 | 0 | 0 | 0 | 2026-09-24 | archive | newer version exists: probe_vector_nav_v2 |
| `run_addition_pilot.sh` | SUPERSEDED | 0 | 0 | 0 | 2 | 0 | 2026-09-14 | archive | one-off pilot / top-up / redo / amend |
| `run_addition_pilot2.sh` | SUPERSEDED | 0 | 0 | 0 | 1 | 0 | 2026-09-14 | archive | one-off pilot / top-up / redo / amend |
| `run_capacity_rerun.sh` | SUPERSEDED | 0 | 0 | 1 | 1 | 0 | 2026-05-22 | archive | one-off pilot / top-up / redo / amend |
| `run_cmq_curric_topup.sh` | SUPERSEDED | 0 | 0 | 0 | 2 | 0 | 2026-08-24 | archive | one-off pilot / top-up / redo / amend |
| `run_cmq_stab_topup.sh` | SUPERSEDED | 0 | 0 | 0 | 2 | 0 | 2026-08-23 | archive | one-off pilot / top-up / redo / amend |
| `run_cmq_topup.sh` | SUPERSEDED | 0 | 0 | 0 | 1 | 0 | 2026-08-23 | archive | one-off pilot / top-up / redo / amend |
| `run_cnav_redo.sh` | SUPERSEDED | 0 | 0 | 0 | 5 | 0 | 2026-05-03 | archive | one-off pilot / top-up / redo / amend |
| `run_cont_redo.sh` | SUPERSEDED | 0 | 0 | 0 | 1 | 0 | 2026-08-23 | archive | one-off pilot / top-up / redo / amend |
| `run_looped_pilot.sh` | SUPERSEDED | 0 | 0 | 0 | 6 | 0 | 2026-09-24 | archive | one-off pilot / top-up / redo / amend |
| `run_minigrid_allo.sh` | SUPERSEDED | 0 | 1 | 0 | 1 | 0 | 2026-08-24 | archive | newer version exists: run_minigrid_allo8 |
| `run_redo_all.sh` | SUPERSEDED | 0 | 0 | 0 | 1 | 0 | 2026-08-23 | archive | one-off pilot / top-up / redo / amend |
| `run_rotate_redo.sh` | SUPERSEDED | 0 | 0 | 0 | 1 | 0 | 2026-08-21 | archive | one-off pilot / top-up / redo / amend |
| `run_sameblock_amend.sh` | SUPERSEDED | 0 | 0 | 0 | 2 | 0 | 2026-09-15 | archive | one-off pilot / top-up / redo / amend |
| `run_t3gen.sh` | SUPERSEDED | 0 | 0 | 0 | 6 | 0 | 2026-09-19 | archive | newer version exists: run_t3gen2 |
| `run_tem_recency_pilot.sh` | SUPERSEDED | 0 | 0 | 0 | 1 | 0 | 2026-09-14 | archive | one-off pilot / top-up / redo / amend |
| `run_topup_n8.sh` | SUPERSEDED | 0 | 0 | 0 | 3 | 0 | 2026-08-01 | archive | one-off pilot / top-up / redo / amend |
| `train_ctxstep.py` | SUPERSEDED | 0 | 0 | 0 | 1 | 0 | 2026-09-28 | archive | newer version exists: train_ctxstep2, train_ctxstep3 |
| `train_ctxstep2.py` | SUPERSEDED | 0 | 0 | 0 | 1 | 0 | 2026-09-28 | archive | newer version exists: train_ctxstep3 |
| `_sign_smoke.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-09-06 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `docs/theory/2026-10-04/scripts/basins2.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-10-04 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `grid_module_hex_test.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-04-26 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `make_inekf_report_figure.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-06-03 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `maze_rl_env.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-09-19 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `orchestrator_gridl15pc.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-04-26 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `probe_coarse_angles.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-08-02 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_addition_cho.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-09-14 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_dimsweep.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-08-03 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_dropout_ablation.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-10 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_goal_closedloop.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-11 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_goal_directed.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-10 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_goal_tasks.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-11 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_gsf.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-10 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_habitat_install.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-08-23 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_hex_emergence.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-13 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_hiergoal_fixed.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-08-09 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_hiergoal_multiseed.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-07-26 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_hiergoal_scan.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-07-26 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_modeomega.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-11 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_multisize.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-11 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_mw_fixed_diag.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-08-28 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_mw_grid_pilot.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-09-19 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_mw_nope.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-08-28 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_mw_scale_diag.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-08-28 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_prebuild.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-08-28 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_probe_goal.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-10 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_probe_sr.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-11 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_recency_budget.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-09-07 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_tem_scaling.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-21 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `run_vector_nav.sh` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-05-13 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `train_maze_varying.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-07-22 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `train_rooms.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-07-18 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `train_rooms_goal.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-07-21 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `validate_clock.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-07-28 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `verify_theory.py` | DEAD | 0 | 0 | 0 | 0 | 0 | 2026-09-06 | archive | no importer, driver, doc mention, run-dir/commit link or md5 record |
## 3. Duplicated logic

Found two ways: (a) identical function bodies (normalised AST) in more than one file -- 38 groups of >= 5 lines, ~705
redundant lines; (b) same-purpose functions under a family of names (Fisher, permutation, sign-flip, MDE, Holm, run
classes, floors, strata, evaluators, loaders, md5 helpers, aggregation helpers). `stats_core.py` is the declared canonical
statistics module; `stats_guard.py` wraps it for verdict tables; `eval_rank_strata.py` is the de-facto canonical torus
strata evaluator (14 importers); `analyze_lrsweep.py` is the de-facto library for best-rate selection (`select`, `b1_test`,
`b1_decide`, `sel_score`, `holm`; imported by 4 analyses, `holm` by 3).

"Consolidate" below never means editing a md5-recorded analysis in place: a caller changes only when it is next touched
for a new batch, or is verified (Section 6). Equivalence notes are from reading the code, not from running it.

| # | logic | copies (file: function) | canonical | consolidation | callers that would change | numerics risk |
|---|---|---|---|---|---|---|
| D1 | Holm step-down adjustment | `analyze_image_torus`, `analyze_tw_landmark`, `analyze_lrsweep`, `analyze_delta_free` (returns list), `analyze_signroute`, `analyze_mm` (own copy "so this module imports only numpy / scipy / stats_core"); imported from `analyze_lrsweep` by `analyze_rnn_torus`, `analyze_exptorus`, `analyze_torus_lrsweep`; inline Holm text in 15 analyses | none in `stats_core` | add `stats_core.holm` (the `analyze_lrsweep` body); new analyses import it | 6 definers + 3 importers, only when next edited | none: the two variants (min inside vs outside the running max) are algebraically identical because min(1, .) is monotone; verify on every stored p-vector |
| D2 | two-sample MDE, pooled sd, exact t (`mde2`) | `analyze_tw_landmark`, `analyze_mappope_pair`, `analyze_score_rank`, `analyze_gain_phase`, `analyze_gain_grain` (two spellings: `x.var` vs `np.var`) | none (`stats_core.mde` is one-sample) | add `stats_core.mde2(x, y)` | 5 | none (identical formula) |
| D3 | house-rule MDE 2.8 sd / sqrt(n) | `analyze_code.mde`, `analyze_code_long.mde`, `analyze_dyck_depth.mde`, `eval_nd.mde`, `stats_guard.mde` | `stats_core.mde(sd, n, exact_t=False)` | replace with `stats_core.mde(..., exact_t=False)`; rule 5 says report the exact-t MDE beside it | 4 (`analyze_code`, `analyze_code_long` HISTORICAL; `analyze_dyck_depth`, `eval_nd` LIVE): leave all four | none |
| D4 | Fisher on SOLVED with a direction label (`fisher_state`) | `analyze_score_rank`, `analyze_gain_phase`, `analyze_gain_grain` (each wraps `stats_core.fisher_solved`); raw `scipy.stats.fisher_exact` in `analyze_textworld`, `analyze_indirect` | `stats_core.fisher_solved` | add `stats_core.fisher_state` | 3 | none |
| D5 | sign-flip p | `stats_core.signflip_p` (canonical); wrappers `stats_guard.signflip_p` / `p_signflip` / `sign_flip`; copy in `docs/paper/expansion/scripts/whatwhere/whatwhere_stats.py` | `stats_core.signflip_p` | collapse the three `stats_guard` names to one alias; the expansion script imports `stats_core` | 1 script | Monte-Carlo above n=20: same seed argument needed |
| D6 | permutation tests | `stats_core.perm2_p` / `perm2_ci` (vectorised 2026-09-24); local `analyze_rank_matched.perm` / `perm_ci`, `basin_summary.perm_p`; `np.random.permutation` in 15 files (mostly data shuffles, not tests) | `stats_core.perm2_p` | none needed: `analyze_rank_matched` is CORE (imported by `analyze_rank_perhead`, `analyze_rank_proj`) and md5-adjacent -- leave | 0 | -- |
| D7 | run classification SOLVED / STALLED / DESCENDING | `stats_core.classify_run` (named in 22 of the 23 files that print run classes); `analyze_rank_proj` classifies inline; `analyze_rank_matched.classify`, `agg_recipe.converged`, `experiment_audit.check_convergence` (slope rule; prints the registered windowed classes too) | `stats_core.classify_run` | none for registered analyses (their thresholds are pre-registered); document that `experiment_audit`'s slope rule is not a convergence test (rule 3) | 0 | changing a threshold changes a registered verdict: never |
| D8 | trivial-predictor floors | n-gram: `validate_flipflop.ngram_acc`, `validate_recency.ngram_acc`, `validate_algorithmic.ngram_acc`, `validate_code.fit_ngram`/`predict_ngram`, `gate_image_torus.fit_ngrams`; retrace: `analyze_delta_free.retrace_class` = `docs/paper/expansion/scripts/mappope/kernel_probe.retrace_class` (self-tested equal), `gate_ssm_sign.retrace`, `tw_strata.retrace_ok`; per-cell constant: `eval_rank_strata.evaluate` (`floor` field). Six `floors()` functions are task-specific (Indirect, text world, image torus, rank-nowrap, paper floor) and are NOT duplicates | none | a `floors.py` with `ngram_floor(stream, orders)` and `retrace_class(actions)`, adopted by new gates only (rule 11 wants the gate to call task code, which these do) | 0 now | floors are reported numbers: verify byte-equal on stored gate outputs before any switch |
| D9 | revisit strata (plain lag < 128 / >= 128 / wrap-only) | `eval_rank_strata.kinds` / `kinds_vec` (canonical, verified byte-identical 2026-09-24, `docs/audits/2026-09-24/efficiency/verify_strata.py`); own copies: `analyze_rank_nowrap.strat` / `all_strata`, `analyze_rank_nd_secondary.strat_acc` (imported by `analyze_rank_wrap`), `analyze_torus_start`, `analyze_image_torus`, `gate_image_torus`, `mm_gate_v2.stratum_mask` (different task), `loop_depth_strata.by_stratum` (LIVE: MoR negative), `docs/.../recurrence/common.stratum` | `eval_rank_strata.kinds_vec` | new torus analyses import `kinds_vec`; ND / ring / Memory Maze strata stay separate (different geometry) | 0 now | strata define registered secondaries: verify per-stratum counts equal |
| D10 | B=1 revisit evaluator `evaluate(model, env, T, n_trials, dev, seed)` | identical 13-line body in `train_cancel`, `train_ctxstep`, `train_ctxstep2`, `train_ctxstep3`, `train_newobj`, `train_textworld`; `eval_revisit` in 7 legacy eval scripts (`long_sequence_eval`, `eval_ood_grid`, `aux_coef_sweep`, `zero_shot_eval`, `rescale_eval`, `held_out_obs_eval`, `eval_long_t`); `evaluate_run` in `eval_image_torus`, `leak_eval`, `gain_phase_eval`; `evaluate` defined in 38 files in total | `eval_rank_strata.evaluate` for torus strata; none for the text-world family | one `revisit_eval(model, gen_fn, T, n_trials, dev, seed)` in a new module; of the 6 trainers, `train_ctxstep` / `train_ctxstep2` are SUPERSEDED (archive), the other 4 are CORE/LIVE and keep their copy until next touched | 4 | accuracy counts are integers (exact); NLL summation order must be kept (see `eval_rank_strata`'s float64 cumsum trick) |
| D11 | checkpoint -> model builders | `build_model` x11 (`long_sequence_eval`, `calibration_analysis`, `r_t_distribution_test`, `hippocampal_hidden_eval`, `zero_shot_eval`, `rescale_eval`, `per_visit_eval`, `clone_transfer_test`, `hippocampal_analysis`, `held_out_obs_eval`, `make_paper_figures`; 6 byte-identical), `build` x31, `build_model_from_config` x4 (two identical pairs), `load_model` x2 | `train_variant.VARIANT_MAP` + `ckpt_guard.load_checkpoint` | a `ckpt_guard.build_from_checkpoint(path)` helper; 8 of the 11 `build_model` definers are HISTORICAL (archive them); `long_sequence_eval`, `calibration_analysis`, `zero_shot_eval` are LIVE -- leave | 0 | none |
| D12 | aggregation helpers of the Aug-Sep `agg_*.py` scripts | `stat` x8, `vd` x6 (identical), `pair` x5, `final_loss` x4 (`agg_l15_loop`, `agg_sign` identical) | `stats_guard.Contrast` | none: 5 of these 8 `agg_*` are HISTORICAL (archive); `agg_l15_loop`, `agg_selective`, `agg_sign` are LIVE (cited live negatives) -- leave | 0 | -- |
| D13 | model code | `forward` identical (21 lines) in `model_ablations`, `model_inekf_level15`, `model_inekf_level15_beta`, `model_inekf_level15_nodrop`, `model_inekf_level2`; `__init__` identical in `model.py` and `model_inekf_level15_nodrop`; `forward` (8 lines) in `model_rankclock`, `model_signroute`, `model_wrap_exist`; `__init__`+`forward` in `model_code_decay` / `model_pope_decay`; `__init__` in `model_inekf_level15_extrahead` / `model_vanilla_extrahead` | -- | **do not consolidate**: model files are md5-recorded (up to 54 run dirs for `model.py`) and a shared base class changes the md5 and the state-dict layout of every arm | 0 | high |
| D14 | driver helpers | local `on_gpu()` in 81 drivers, `busy()` 65, `pick()` 49 (50 of these drivers are LIVE); 68 drivers source `lib_driver.sh` | `lib_driver.sh` | none for existing drivers (rule 22: never edit a driver in place; md5-recorded drivers would change); every new driver sources it. Zero drivers run `pgrep -f` / `pkill -f` (the 10 hits are comments citing rule 23) | 0 | -- |
| D15 | md5 helpers | `mm_encoder.md5`, `train_mm.md5_file`, `image_bank._md5`, `tw_landmark_eval.ckpt_md5`; `lib_driver.drv_md5_guard` (shell) | none in Python | add `ckpt_guard.md5_file` for new code | 0 | none |
| D16 | small analysis helpers | `basin` (5 identical: `analyze_gain_grain`, `analyze_rank_nowrap`, `analyze_score_rank`, `docs/theory/.../basins.py`, `basins_rt.py`), `bin_of` (3: `eval_code_delims`, `eval_code_long`, `validate_code`), `cells` (`analyze_signroute`, `analyze_signsep`), `compare` (`analyze_torus_start`, `analyze_wrap_cope`) | -- | leave; lift only when a new analysis needs them | 0 | -- |
| D17 | launch / harness boilerplate | `launch_experiment` (3 orchestrators, identical), `launch` (2), `main` of `train_ring_dial` = `train_ring_variant` (35 lines), test `main` of `test_guards` = `test_rankclock` = `test_signroute` (16 lines) | -- | a 16-line `testlib.run_tests()` for new tests; orchestrators are HISTORICAL | 0 | -- |

## 4. Dead and superseded code safe to archive (60 files, ~4,600 lines)

Evidence per file: the graph shows no importer, driver, document mention, commit / run-dir / output link or md5 record
(DEAD), or a newer version / one-off role (SUPERSEDED); then a whole-word grep for the file stem over the whole
repository outside `runs/`, `papers/`, `.git/`, `docs/prepared/` and `docs/cleanup/` (the "references" column). A hit in
`.claude/settings.local.json` is a permission allow-list entry, not a use. A hit in `archive/void/` or `archive_stale/`
is a bannered void/stale result: archive the script next to it. No DEAD file has a run dir named after it; no
SUPERSEDED file except `hyak/mm_gate_v2.slurm` (named in `MEMORY_MAZE_PREREG.md` as history) is cited by a live document.

Proposed destination: `archive/code/2026-10/` (keeps git history via `git mv`; nothing deleted), with a one-line
`archive/code/2026-10/INDEX.md` row per file (old path, class, evidence). Before the move, re-run
`python3 -m mapformer.test_guards` and `python3 -c "import mapformer.train_variant"`; after it, run both again and
`git grep -n` each moved stem to confirm zero code references.

| file | class | last commit | lines | references found anywhere outside runs/ (whole-word grep) | note |
|---|---|---|---:|---|---|
| `_sign_smoke.sh` | DEAD | 2026-09-06 | 12 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
| `docs/theory/2026-10-04/scripts/basins2.py` | DEAD | 2026-10-04 | 15 | none | no importer, driver, doc mention, run-dir/commit link or md5 record |
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
| `analyze_t1.py` | SUPERSEDED | 2026-09-18 | 94 | none | newer version exists: analyze_t2 |
| `docs/theory/2026-10-04/scripts/toy.py` | SUPERSEDED | 2026-10-04 | 96 | none | newer version exists: toy2 |
| `environment_textworld_ctx.py` | SUPERSEDED | 2026-09-28 | 78 | none | newer version exists: environment_textworld_ctx2, environment_textworld_ctx3 |
| `environment_textworld_ctx2.py` | SUPERSEDED | 2026-09-28 | 93 | `CTXSTEP_PILOT2.md`, `CONTEXT_STEP_DESIGN.md`, `train_ctxstep2.py` | newer version exists: environment_textworld_ctx3 |
| `hyak/mm_gate_v2.slurm` | SUPERSEDED | 2026-10-27 | 19 | `MEMORY_MAZE_PREREG.md`, `mm_data.py`, `mm_gate_v2.py` | newer version exists: mm_gate_v3 |
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
| `train_ctxstep.py` | SUPERSEDED | 2026-09-28 | 82 | none | newer version exists: train_ctxstep2, train_ctxstep3 |
| `train_ctxstep2.py` | SUPERSEDED | 2026-09-28 | 83 | `docs/audits/2026-09-27/swap_ctxstep2.py`, `docs/audits/2026-09-27/swap_test.py` | newer version exists: train_ctxstep3 |
## 5. Efficiency opportunities, ranked by payoff and risk

Already done (2026-09-24 audit, `docs/audits/2026-09-24/efficiency/REPORT.md`, applied and verified under
`docs/audits/2026-09-24/applied/`): vectorised torus walk (`environment.FAST_WALK`, byte-identical, 23x), sync-free
`train.train` loop (bitwise), scale fold, vectorised strata scoring in `eval_rank_strata` (byte-identical JSON),
vectorised exact permutation test in `stats_core`, `lib_driver.sh`, `--fast-attn` for new series. Not repeated here.

Timings below are CPU, measured for this inventory (`CUDA_VISIBLE_DEVICES=""`, one process, the shared server idle):
torus `GridWorld.generate_batch(128, 512)` 0.017 s (fast walk) vs 0.30 s (loop); `GridWorldND` 0.375 s;
`GridWorldCancel` 0.059 s; `TextWorld.generate_batch(128, 1024)` 0.196 s; `StateChangeWorld` 0.237 s.

Rule 19 applies to every row: a fast path is used by a batch only after it is shown row-exact (data) or loss-exact
(training) against the slow path, and a change to a md5-recorded module means the batches that recorded the old md5 can
no longer be re-run bitwise from HEAD (acceptable for archived batches only if written into the batch's results file;
they remain reproducible from their pre-registration commit).

| rank | where | what | payoff | risk / verification | live users |
|---|---|---|---|---|---|
| E1 | `environment_nd.GridWorldND.generate_trajectory` (also the parent of `GridWorldCancel` / `GridWorldSteps`) | per-step Python loop with `tuple(pos)` dict keys and per-step `np.random.randint`; the torus had the same shape before `FAST_WALK` | ~20x on data generation (by analogy with the torus: 0.30 -> 0.017 s); the ND / rank-wrap runs use B=16 and `--data-workers 3`, so at 98 x 900 batches ~70 CPU-min per run move off the critical path and 3 cores per job are freed | LOW if done the `FAST_WALK` way: keep the exact sequence of `np.random` draws (precompute the same draws in the same order), parse in numpy; verify tokens, masks, locations AND the RNG state byte-identical over many seeds and dims 2/3, as `verify_env_patch.py` did | `RANK_ND`, `RANK_WRAP`, `CANCEL` (H3), `GAUGE`, steps tasks |
| E2 | B=1 evaluation loops (45 files: 14 LIVE, 8 CORE incl. `eval_rank_strata.evaluate`, `evaluate.py`, `train_textworld.evaluate`, `train_cancel.evaluate`, `train_newobj.evaluate`; 20 HISTORICAL) | one trajectory per forward, `.item()` / `float()` per trial; trajectories are fixed length, so a batch of n_trials needs no padding | 5-30x on eval-only re-scores (GPU mostly idle at B=1, 1 layer); matters for eval-only sweeps (rule 13) and the dropout re-score kind of audit, not for training batches | MEDIUM: batched attention / matmul need not be bitwise equal to B=1 (different kernels, reduction order). Hits are integers and survive; NLL may move in the last bits. Keep the B=1 path as default, add `--eval-batch N`, verify hit counts equal and NLL within 1e-6 on stored checkpoints of every live arm before any registered readout uses it | every eval-only re-score |
| E3 | 32 trainers with their own loop and a per-step `loss.item()` (14 LIVE incl. `train_mm`, `train_indirect`, `train_jsb`, `train_recency`, `train_miniworld`; 4 CORE: `train_dyck`, `train_hourglass_enwik8`, `train_match_query`, `train_tinystories`; 11 HISTORICAL; 3 DEAD) | host-device sync every step, the pattern removed from `train.py` | 0-5% at current GPU-bound configs, 10-20% with `--fast-attn` (2026-09-24 estimate for `train.py`) | LOW for numerics (the `train.py` change was bitwise: float64 accumulate in the same order); but each trainer is md5-recorded by its batches. Do it only when a trainer is next used for a NEW series | Dyck, Memory Maze, Indirect, Bach |
| E4 | text-world renderers (`environment_textworld`, `_tw_statechange`, `_tw_ambig`, `_tw_landmark`, `_textworld_ctx3`) | per-step Python rendering with global `np.random` calls, per-sample `generate_trajectory` in `generate_batch` | 0.20-0.24 s per 128x1024 batch; runs use B=16 and `--data-workers`, so ~25-30 ms per batch, near the GPU step time | HIGH for bitwise: a vectorised renderer changes the draw order and therefore the data stream (rule 19: `--data-workers` already changes it). New series only, never beside a reproduction control. Cheaper alternative with no numerics change: more `--data-workers` | text-world line (TW_*), LEAK |
| E5 | `--fast-attn` coverage | supported by 9 trainers (`train_variant`, `train_compositional`, `train_image`, `train_flipflop`, `train_miniworld`, `train_rnn`, `train_match_query`, `train_torus_start`, `train_recency`) via `model.py`; not by the text-world trainers, `train_cancel`, `train_dyck`, `train_mm` | 2.5-4x training (JUDGED 2026-09-24 for the rank config) | not bit-identical (dropout RNG); deterministic backward only with `--deterministic`; invalid for MapEM; new series only (`FAST_ATTN_RANK.md`) | next new series on those trainers |
| E6 | `train_variant.py` eager imports | 87 model modules imported or `__import__`ed at module load | 0.24 s per process: negligible | none for speed; the value is decoupling (Section 6, stage 4) | -- |
| E7 | `kinds()` (slow strata) still present in `eval_rank_strata` beside `kinds_vec` | dead path kept as the reference implementation | none | keep: it is the reference the vectorised path was verified against | -- |

Not recommended: changing `stats_core` internals (already vectorised and verified), consolidating model classes (md5 and
state-dict layout, D13), TF32 or bf16 anywhere (rule 19; `BF16_RESULTS.md`).


## 6. Staged, low-risk refactor plan

Principles: never edit a module a running batch spawns from (rule 22; check with the rule-23 `ps | awk` first); never change
the numerics of a live analysis without a recorded before/after check; every move is a `git mv` (history kept); every
stage ends with `python3 -m mapformer.test_guards` (from `/home/prashr`), the 10 CPU tests of Section 7.4, and
`python3 -c "import mapformer.train_variant"`; one commit per stage, single-author (no `Co-Authored-By`).

| stage | change | files | numerics touched | verification | reversible by |
|---|---|---|---|---|---|
| 0 | Freeze the baseline: record `md5sum` of every CORE and LIVE file and the outputs of the 10 CPU tests into `docs/cleanup/baseline_2026-10.md5` | 640 | none | -- | -- |
| 1 | Archive the 36 DEAD and 24 SUPERSEDED files (Section 4) into `archive/code/2026-10/` with an `INDEX.md` | 60 | none (nothing imports them) | `git grep -nw <stem>` returns only the index and archived docs; tests pass; `import mapformer.train_variant` | `git mv` back |
| 2 | Hygiene with no md5-recorded file touched: remove unused imports only in files that are neither md5-recorded nor CORE. The two invalid escapes are in md5-recorded files (`model_gated.py`: 17 run dirs, current md5 matches all; `analyze_signsep.py`: `runs/signsep/p0`) and are harmless docstring escapes (a DeprecationWarning on 3.10, SyntaxWarning from 3.12; slated to become an error in a later Python): defer them to the next series that edits those files, and then use `r"""` docstrings | <= 100 | none | `ruff check --select F401` on touched files; byte-compare outputs of any touched analysis on its stored inputs | revert commit |
| 3 | Archive HISTORICAL files in batches by topic (e.g. all `agg_*`, the orchestrators, the hier-goal / goal / rooms / MiniGrid DAgger line, the Aug lm200 era), each batch with a path map row in `archive/code/2026-10/INDEX.md` and, for every non-live results file that names a moved script, a one-line note "script moved to archive/code/..." (do not rewrite the results file) | 268, in 6-10 batches | none | as stage 1, plus: no LIVE or CORE file imports a moved module (the graph says none; re-check with `git grep`) | `git mv` back |
| 4 | Lazy `VARIANT_MAP` in a NEW module (`variant_registry.py`: name -> "module:Class" strings, imported on lookup); `train_variant.py` switches to it only at the start of a new series, recorded in that series' pre-registration. Then the model modules whose variants appear in no live driver or live document can be archived (stage 3 rules). Today none qualify only because `train_variant` imports them | 1 new + `train_variant.py` | none if the registry returns the same classes | for every variant name: `VARIANT_MAP[name] is registry.get(name)`; a 1-epoch CPU run of 4 representative arms bitwise-equal before / after (`test_epoch_callback.py` pattern) | keep the old dict for one series |
| 5 | Add to `stats_core` (append-only, no existing function changed): `holm`, `mde2`, `fisher_state`; to `ckpt_guard`: `md5_file`; a `floors.py` with `ngram_floor` and `retrace_class`; a `revisit_eval` helper. New analyses import them; existing analyses are NOT rewritten | 3-4 | none for existing code | unit tests: each new function equals every copy it replaces on recorded inputs (e.g. the p-vectors stored in registered run dirs' verdict / analysis JSON), byte-equal | delete the additions |
| 6 | Efficiency E1 (`GridWorldND` fast walk) behind a flag defaulting to the old path, as `FAST_WALK` was done | 1 | data generation | tokens / masks / locations / RNG state byte-identical over >= 92 checks (the 2026-09-24 standard); then flip the default and record in `GUARDS.md` that ND batches before the flip used the loop | flag off |
| 7 | Efficiency E2 (`--eval-batch`) on `eval_rank_strata` first | 1-3 | NLL last bits only | hits equal, NLL within 1e-6 on stored checkpoints of every live torus arm; default stays B=1 for registered readouts | flag off |
| 8 | Optional, new series only: E3 / E5 in the trainer that the next batch uses; record the trainer's new md5 in the pre-registration | per batch | training loop | loss-exact vs the old loop on 2 seeds x 3 epochs on the GPU before launch (rule 19) | -- |

What is deliberately NOT proposed: moving CORE modules into sub-packages (every `python3 -m mapformer.X` command in 350
drivers and every results file would break), rewriting md5-recorded drivers to source `lib_driver.sh`, sharing model
base classes, or replacing absolute `REPO` constants (rule 25 requires them).


## 7. Hygiene

### 7.1 Hard-coded absolute paths

`/home/prashr/mapformer` occurs **483 times in 421 files**: 181 in 158 top-level `.py`, 232 in 202 `.sh`, 70 in 61
scripts under `docs/` and `hyak/`. This is mostly by design: rule 25 requires an absolute `REPO` constant, and 151
drivers set `REPO=/home/prashr/mapformer` once; 88 Python files set a `REPO`/`ROOT`-style constant, the remaining ~70
embed the path at the use site (mostly HISTORICAL eval / agg scripts: `agg_comp_multiseed`, `agg_clock`,
`ablate_recency_gate`, `long_sequence_eval`, `held_out_obs_eval`, `eval_ood_grid`, ...). Outside the repo:
`/home/prashr/mm_data` in `run_mm_pilot.sh`, `run_mm_pilot3.sh` (Memory Maze data). The `hyak/*.slurm` files carry the
cluster's own paths. Proposal: none for existing files (md5 records, rule 22); new code takes `REPO` from one place
(`ckpt_guard.repo_path` already resolves repo-relative paths).

### 7.2 Invalid escape sequences

| file | where | status |
|---|---|---|
| `model_gated.py` | module docstring line 21 (`\_____ whether _____/`) | DeprecationWarning on Python 3.10, SyntaxWarning on 3.12+; md5-recorded in 17 run dirs (all match HEAD); `train_variant` imports it |
| `analyze_signsep.py` | module docstring line 24 and docstring line 101 (`2 Z \ {0}`) | same; md5-recorded in `runs/signsep/p0` |

No other file in scope raises a Syntax/DeprecationWarning at compile time (checked with `-W error` on all 609 Python files).
Fix both with `r"""` docstrings when the next series touches them (Section 6, stage 2).

### 7.3 Unused code at scale (ruff 0.15.5, `python3 -m ruff`; no config file in the repo)

- F401 unused imports: **198 in 136 files**. 23 are `__init__.py`'s re-exports (the package API: keep). 52 files with
  hits are md5-recorded and 37 are CORE (e.g. `model.py` imports `build_block_diagonal_rotations_fast` unused;
  `train.py` imports `math`, `typing.Optional` unused): do not touch these outside a new series, since removing an
  import changes the md5 that `drv_md5_guard` checks. 78 other files / 96 imports can be cleaned at no risk.
  Most common: `torch.nn.functional` (21), `math` (20), `torch.nn` (13).
- F841 unused local variables: 45. F811 redefinition: 1. F821 undefined names: **0** (no latent NameErrors).

### 7.4 Tests

All 11 `test_*.py` modules, run from `/home/prashr` as `python3 -m mapformer.<test>` with `CUDA_VISIBLE_DEVICES=""`:

| test | result | time | covers |
|---|---|---:|---|
| `test_guards` | 13/13 pass | 3 s | `ckpt_guard`, `stats_guard` / `stats_core` edge cases, determinism pair, layout guard, probe RNG hygiene, `train_variant` manipulation checks |
| `test_signroute` | 12/12 pass | 2 s | `model_signroute` |
| `test_ssm_sign` | 40/40 pass | 2 s | `model_ssm_sign` |
| `test_exptorus` | 7/7 pass | 6 s | `model_exptorus`, init bitwise vs lr-0 checkpoint |
| `test_epoch_callback` | PASS | 3 s | `train.train` snapshot callback bitwise |
| `test_hourglass_causal` | pass | 3 s | `model_hourglass` causality and parameter parity |
| `test_image_torus` | 14/14 pass | 5 s | `image_bank`, `model_image`, `eval_image_torus` |
| `test_rankclock` | 10/10 pass | 10 s | `model_rankclock` |
| `test_signsep` | 9/9 pass | 2 s | `model_signsep` |
| `test_rnn_torus` | 11/11 pass | 84 s | `model_rnn`, `eval_rnn_torus` |
| `test_mm` | not run: needs `--zip` and `--data` (Memory Maze data, not on this machine's repo path) | -- | Memory Maze pipeline |

Gaps: no unit test of `environment.py`'s fast walk vs loop, of `eval_rank_strata.kinds_vec` vs `kinds`, or of
`stats_core` directly (the equivalence checks exist as one-off audit scripts:
`docs/audits/2026-09-24/efficiency/verify_env_patch.py`, `verify_strata.py`). Promoting those two scripts to
`test_fastpaths.py` would guard the two hottest verified fast paths at ~seconds of CPU. 26 further `validate_*` /
`verify_*` / `check_*` scripts are task gates or one-off verifications, not regression tests.

### 7.5 md5 records versus HEAD

56 `code_md5*.txt` files record 244 distinct modules. In 45 of the 56 at least one recorded top-level file differs from HEAD (only 10 match HEAD entirely). Per
file (rows marked "not at top level" are files kept under `docs/` whose basename was looked up at top level -- not drift):

| file | records matching HEAD | records differing from HEAD | run dirs whose recorded md5 differs (first 3) |
|---|---:|---:|---|
| `train_variant` | 10 | 41 | `runs/accounts`, `runs/cancel`, `runs/clock_side` |
| `train` | 13 | 41 | `runs/accounts`, `runs/cancel`, `runs/clock_side` |
| `rescore_hook` | 2 | 8 | `runs/clock_side`, `runs/gain_phase`, `runs/rank_tests` |
| `model_wrap_exist` | 21 | 3 | `runs/generality`, `runs/wrap_existence`, `runs/wrap_mech` |
| `model` | 52 | 3 | `runs/rank_matched_e900`, `runs/rank_matched_e900c`, `runs/rank_perhead_pilot` |
| `environment` | 41 | 3 | `runs/rank_matched_e900`, `runs/rank_matched_e900c`, `runs/rank_perhead_pilot` |
| `model_rank_perhead` | 24 | 3 | `runs/rank_mi`, `runs/rank_perhead_pilot`, `runs/rank_sep` |
| `train_signsep` | 1 | 3 | `runs/signsep_pilot`, `runs/signsep_pilot2` |
| `run_signsep_pilot.sh` | 0 | 2 | `runs/signsep_pilot` |
| `train_wrap_exist` | 1 | 1 | `runs/wrap_existence` |
| `train_wrap_mech` | 5 | 1 | `runs/wrap_mech` |
| `train_ring_variant` | 2 | 1 | `runs/generality` |
| `model_mono_baselines` | 17 | 1 | `runs/generality` |
| `train_cancel` | 17 | 1 | `runs/cancel` |
| `analyze_clock_side` | 0 | 1 | `runs/clock_side` |
| `swap_test (not at top level: recorded under docs/; not drift)` | 0 | 1 | `runs/ctxstep` |
| `analyze_delta_free` | 1 | 1 | `runs/delta_free_pilot` |
| `run_delta_free.sh` | 1 | 1 | `runs/delta_free_pilot` |
| `gain_grain_remap (not at top level: recorded under docs/; not drift)` | 0 | 1 | `runs/gain_grain` |
| `remap_probe (not at top level: recorded under docs/; not drift)` | 0 | 1 | `runs/gain_grain` |
| `probe_whatwhere (not at top level: recorded under docs/; not drift)` | 0 | 1 | `runs/gain_grain` |
| `gain_phase_secondary (not at top level: recorded under docs/; not drift)` | 0 | 1 | `runs/gain_phase` |
| `train_gauge` | 2 | 1 | `runs/gauge/p0` |
| `rank_tests_secondary (not at top level: recorded under docs/; not drift)` | 0 | 1 | `runs/rank_tests` |
| `model_signsep` | 3 | 1 | `runs/signsep_pilot` |
| `construct_signsep` | 2 | 1 | `runs/signsep_pilot` |
| `analyze_ssm_sign` | 1 | 1 | `runs/ssm_sign/p0` |
| `run_ssm_sign.sh` | 1 | 1 | `runs/ssm_sign/p0` |
| `train_textworld` | 2 | 1 | `runs/textworld` |
| `clock_table (not at top level: recorded under docs/; not drift)` | 0 | 1 | `runs/torus_lrsweep` |

Per run dir (live = the run dir is named in a live document):

| md5 file | files listed | differ from HEAD | not at top level | listed analysis / driver files (first 4, excluding model / env / train modules) |
|---|---:|---:|---:|---|
| `runs/accounts/code_md5.txt` (live) | 22 | 2 | 0 | analyze_accounts.py, analyze_wrap_mech.py, run_accounts.sh, rescore_hook.py |
| `runs/cancel/code_md5.txt` (live) | 8 | 3 | 0 |  |
| `runs/clock_side/code_md5.txt` | 13 | 4 | 0 | analyze_clock_side.py, eval_clock_side.py, rescore_hook.py, run_clock_side.sh |
| `runs/clock_speed/code_md5.txt` | 11 | 2 | 0 | analyze_clock_speed.py, run_clock_speed.sh |
| `runs/clock_speed_rep/code_md5.txt` | 12 | 2 | 0 | analyze_clock_speed.py, analyze_clock_speed_rep.py, run_clock_speed_rep.sh |
| `runs/ctxstep/code_md5.txt` | 13 | 2 | 1 |  |
| `runs/delta_free/code_md5.txt` | 92 | 2 | 0 | analyze_delta_free.py, eval_noise_refine.py, evaluate.py, hourglass_plain.py |
| `runs/delta_free_pilot/code_md5.txt` (live) | 92 | 4 | 0 | analyze_delta_free.py, eval_noise_refine.py, evaluate.py, hourglass_plain.py |
| `runs/dyck_mdepth/code_md5.txt` (live) | 11 | 0 | 0 | eval_dyck_literature.py, probe_dyck_stack.py, validate_dyck.py, dyck_mdepth_common.py |
| `runs/exptorus/code_md5.txt` (live) | 106 | 0 | 0 | analyze_exptorus.py, analyze_lrsweep.py, analyze_robust.py, analyze_torus_lrsweep.py |
| `runs/exptorus_pilot/code_md5.txt` (live) | 106 | 0 | 0 | analyze_exptorus.py, analyze_lrsweep.py, analyze_robust.py, analyze_torus_lrsweep.py |
| `runs/exptorus_smoke/code_md5.txt` (live) | 106 | 0 | 0 | analyze_exptorus.py, analyze_lrsweep.py, analyze_robust.py, analyze_torus_lrsweep.py |
| `runs/factorial/code_md5.txt` (live) | 16 | 2 | 0 | eval_noise_refine.py, eval_rank_strata.py, analyze_factorial.py, run_factorial.sh |
| `runs/factorial_n8/code_md5.txt` (live) | 97 | 0 | 0 | analyze_factorial.py, analyze_factorial_n8.py, eval_noise_refine.py, eval_rank_strata.py |
| `runs/gain_grain/code_md5.txt` (live) | 96 | 2 | 3 | analyze_gain_grain.py, eval_noise_refine.py, evaluate.py, hourglass_plain.py |
| `runs/gain_phase/code_md5.txt` (live) | 99 | 3 | 1 | analyze_gain_phase.py, eval_gain_phase.py, evaluate.py, gain_phase_eval.py |
| `runs/gauge/p0/code_md5.txt` | 18 | 3 | 0 | analyze_gauge.py, run_gauge.sh |
| `runs/generality/code_md5.txt` | 21 | 5 | 0 | build_torus_exist.py, eval_noise_refine.py, eval_rank_strata.py, analyze_generality.py |
| `runs/image_torus/code_md5.txt` (live) | 103 | 0 | 0 | analyze_factorial.py, analyze_factorial_n8.py, analyze_image_torus.py, eval_image_torus.py |
| `runs/image_torus_pilot2/code_md5.txt` (live) | 5 | 0 | 0 | eval_image_torus.py, image_bank.py |
| `runs/leak/code_md5.txt` (live) | 12 | 2 | 0 | leak_eval.py, analyze_leak.py |
| `runs/loop_rank/code_md5.txt` (live) | 6 | 2 | 0 |  |
| `runs/loop_rank_e1800/code_md5.txt` (live) | 7 | 2 | 0 |  |
| `runs/lrsweep/code_md5.txt` (live) | 96 | 2 | 0 | analyze_lrsweep.py, analyze_robust.py, evaluate.py, hourglass_plain.py |
| `runs/mappope_pair/code_md5.txt` (live) | 13 | 2 | 0 | eval_pair.py, eval_noise_refine.py, analyze_mappope_pair.py |
| `runs/rank3/code_md5.txt` (live) | 7 | 2 | 0 |  |
| `runs/rank_matched_e900/code_md5.txt` (live) | 6 | 4 | 0 |  |
| `runs/rank_matched_e900c/code_md5.txt` (live) | 6 | 4 | 0 |  |
| `runs/rank_mi/code_md5.txt` (live) | 7 | 3 | 0 |  |
| `runs/rank_nd/code_md5.txt` (live) | 8 | 2 | 0 | eval_nd.py |
| `runs/rank_perhead_pilot/code_md5.txt` (live) | 7 | 5 | 0 |  |
| `runs/rank_sep/code_md5.txt` (live) | 7 | 3 | 0 |  |
| `runs/rank_tests/code_md5.txt` | 23 | 3 | 1 | ring_rank_readouts.py, eval_noise_refine.py, eval_rank_strata.py, rescore_hook.py |
| `runs/rank_wrap/code_md5.txt` (live) | 11 | 2 | 0 | eval_nd.py, analyze_rank_nd_secondary.py, analyze_rank_wrap.py |
| `runs/ring_budget/code_md5.txt` | 18 | 3 | 0 | analyze_ring_budget.py, analyze_ring_dial.py, analyze_rank_tests.py, rescore_hook.py |
| `runs/ring_dial/code_md5.txt` | 16 | 3 | 0 | analyze_ring_dial.py, run_ring_dial.sh, rescore_hook.py |
| `runs/rnn_torus/code_md5.txt` (live) | 106 | 0 | 0 | analyze_factorial.py, analyze_factorial_n8.py, analyze_lrsweep.py, analyze_rnn_torus.py |
| `runs/robust/code_md5.txt` | 98 | 2 | 0 | analyze_robust.py, eval_noise_refine.py, eval_rank_strata.py, evaluate.py |
| `runs/score_rank/code_md5.txt` (live) | 17 | 3 | 0 | eval_score_rank.py, eval_noise_refine.py, eval_rank_strata.py, rescore_hook.py |
| `runs/sign_matched/code_md5.txt` (live) | 6 | 2 | 0 |  |
| `runs/signsep/p0/code_md5.txt` (live) | 103 | 0 | 0 | analyze_robust.py, analyze_signroute.py, analyze_signsep.py, evaluate.py |
| `runs/signsep_pilot/code_md5.txt` (live) | 15 | 2 | 0 | construct_signsep.py, run_signsep_pilot.sh |
| `runs/signsep_pilot/code_md5_first_launch.txt` (live) | 13 | 4 | 0 | construct_signsep.py, run_signsep_pilot.sh |
| `runs/signsep_pilot2/code_md5.txt` (live) | 15 | 1 | 0 | construct_signsep.py, run_signsep_pilot2.sh |
| `runs/srope_torus/code_md5.txt` | 95 | 2 | 0 | analyze_srope_torus.py, eval_noise_refine.py, eval_rank_strata.py, evaluate.py |
| `runs/ssm_sign/p0/A2/code_md5.txt` (live) | 103 | 0 | 0 | analyze_robust.py, analyze_signroute.py, analyze_ssm_sign.py, construct_ssm_sign.py |
| `runs/ssm_sign/p0/code_md5.txt` (live) | 103 | 2 | 0 | analyze_robust.py, analyze_signroute.py, analyze_ssm_sign.py, construct_ssm_sign.py |
| `runs/textworld/code_md5.txt` (live) | 8 | 3 | 0 |  |
| `runs/torus_lrsweep/code_md5.txt` (live) | 102 | 0 | 1 | analyze_lrsweep.py, analyze_robust.py, analyze_torus_lrsweep.py, eval_noise_refine.py |
| `runs/torus_start/p0/code_md5.txt` | 16 | 2 | 0 | eval_noise_refine.py, eval_rank_strata.py, analyze_torus_start.py, run_torus_start.sh |
| `runs/tw_landmark_pilot/code_md5.txt` (live) | 14 | 3 | 0 | tw_landmark_eval.py, rescore_hook.py, analyze_tw_landmark.py |
| `runs/tw_normstep/code_md5.txt` (live) | 14 | 2 | 0 | tw_normstep_readouts.py, analyze_tw_normstep.py |
| `runs/tw_statechange/code_md5.txt` (live) | 94 | 2 | 0 | analyze_tw_statechange.py, evaluate.py, hourglass_plain.py, tw_statechange_readouts.py |
| `runs/wrap_cope/code_md5.txt` | 17 | 2 | 0 | analyze_wrap_cope.py, run_wrap_cope.sh, rescore_hook.py |
| `runs/wrap_existence/code_md5.txt` (live) | 17 | 5 | 0 | analyze_wrap_existence.py, run_wrap_existence.sh, rescore_hook.py |
| `runs/wrap_mech/code_md5.txt` | 16 | 4 | 0 | analyze_wrap_mech.py, run_wrap_mech.sh |


Reading: `train.py` and `train_variant.py` are edited for almost every batch (new arms, new flags), so 41 run dirs each
no longer match HEAD; `model.py` / `environment.py` differ only for three rank dirs (`rank_matched_e900`, `rank_matched_e900c`, `rank_perhead_pilot`).
This is expected and harmless while every registered batch names its pre-registration commit, from which it is
reproducible. It also means a CORE refactor does not newly break bitwise reproduction-from-HEAD for most batches -- that
property is already gone -- but it does change what `drv_md5_guard` compares when a driver resumes from checkpoints in an
existing run dir, so no CORE file may change while a run dir that a driver could resume from is in use.

