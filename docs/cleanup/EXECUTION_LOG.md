# Cleanup execution log (2026-10-29)

Executed groups 1 and 2 of `docs/cleanup/CLEANUP_PLAN.md` as approved: group 1 (A1, A5, B2, C1, C2, C5, D1) and group 2
(A2, A3, A4, B1, C3, C4, C6). Not done (not approved): D1b, D2a, D2b, D3, B3, B4, B5, A6 and the topic-folder layout.
No GPU job was launched; nothing was running before or during (no `python3` process, `nvidia-smi` empty). Every commit
is single-author with no `Co-Authored-By` line, adds only its own paths, and was followed by `git pull --rebase` and a
push. Tracked files were archived with `git mv`; the only deletions are A1 (cmp-identical duplicates), C1 (the
unreferenced package copy) and D1 (untracked checkpoints, from an md5 manifest). Every move has a row in
`archive/MOVED.md` (215 rows: old path -> new path, step, date); historical documents and pre-registrations were not
edited.

Checks used after every code step and at the end (from `/home/prashr`, `CUDA_VISIBLE_DEVICES=""`):
`python3 -m mapformer.test_guards` (13/13), the `--selftest` of all 16 analyses that have one (`analyze_delta_free`,
`_factorial_n8`, `_lrsweep`, `_image_torus`, `_exptorus`, `_rankclock`, `_signroute`, `_rankwd`, `_signsep`, `_mm`,
`_rankrecipe`, `_srope_torus`, `_rnn_torus`, `_robust`, `_ssm_sign`, `_torus_lrsweep`; all CORE or LIVE in
`INVENTORY_CODE.md`) and `import mapformer.train_variant`. Baseline before A1: all pass. After B2, D1, A2, A3, A4, B1 and
at the end: all pass, outputs identical to the baseline apart from temp-dir names and set ordering. After B1 and at the
end also the CPU unit tests `test_signroute`, `test_ssm_sign`, `test_exptorus`, `test_epoch_callback`,
`test_hourglass_causal`, `test_image_torus`, `test_rankclock`, `test_signsep` (and `test_rnn_torus` at the end): all pass.

## Group 1

| step | commit | what was done | counts | checks / notes |
|---|---|---|---|---|
| A1 | `ce9a24c` | deleted the top-level copies of results files byte-identical to `archive_stale/` | 20 files (`CLONE_*` x2, `CNAV_*` x5, `DOG_RESULTS`, `HIPPOCAMPAL_*` x7, `MULTICLASS_RESULTS`, `MULTISEED_FOLLOWUP`, `OOD_GRID_RESULTS`, `R_T_DISTRIBUTION_3WAY`, `ZERO_SHOT_TRANSFER_clean_brokeninit`) | `cmp` re-run immediately before `git rm`; the scripts that name them only write them; README's one link rewritten; `archive/MOVED.md` started |
| A5 | `3bb7a9c` | `docs/tools/catalog_results_index.py` now scans `archive/**` and `archive_stale/**` (archived files listed by path, marked `[a]`; unmatched archived files go to "Archived, other lines" instead of failing) and classifies the batches registered after 2026-10-09 (two new groups and about 28 new name patterns); RESULTS_INDEX catalogue regenerated; CLAUDE.md's catalogue row updated | 133 previously unclassified files now classified; 653 top-level + 84 archived, 0 unclassified (re-run at the end, C6 3/3: 573 + 166) | exits 0 |
| B2 | `89c9bf9` | removed unused imports (ruff F401 `--fix`) | 62 imports in 49 top-level scripts | excluded: any file md5-recorded in `runs/**/code_md5*.txt` (70 imports), CORE (9), named in a `*_PREREG.md` (11), `__init__.py` (23); also checked no md5 of a touched file appears in any tracked or `runs/` record. The inventory's "96 in 78 files" included files under `docs/` that stay untouched (audits, expansion scripts, theory). ruff F401/F811/F821 clean, all compile |
| C1 | `49db9ac` | deleted `docs/prepared/dyck_mdepth/pkg/` | 373 tracked + 2 ignored files, 3.5 MB | whole-repo grep (`dyck_mdepth/pkg`, outside `.git`, `runs`, `papers`, `data`): only the cleanup inventories name it; in git history at `233eb9b` |
| C2 | `a32b56c` | `git rm --cached` of 11 `report/report{,_short}.{aux,bbl,blg,log,out,toc}` and the 4 tracked `paper_figures/cnav_*_rate_maps_s0.npz` (33 MB); `.gitignore` gains `report/*.bbl`, `*.blg`, `*.toc` (`.aux/.log/.out/.npz` were already ignored) | 15 files | all still on disk and now ignored |
| C5 | `06c4943` | wrote `papers/new_2026-10-14/SOURCES.txt` (5 arXiv ids, versions read from the text stamps, URLs); `.gitignore` gains `papers/new_*/**/*.pdf` (the plan's `papers/new_*/*.pdf` would not match the nested PDFs); `papers/INDEX.md` gains a contents paragraph (44 papers with text in `txt/`, 42 PDFs; 28 more under `new_*`: 5 + 3 + 20); CLAUDE.md's corpus count | 28 PDFs now ignored | `git check-ignore` confirmed |
| D1 | `392d3d4` | deleted tier-1 checkpoints from an md5 manifest | 920 `.pt` files, **3,927,239,446 bytes = 3.93 GB (3.66 GiB)**: 1a 311 files in 32 pilot / smoke / void dirs (checkpoints only), 1b 586 files in the 228 pre-August legacy dirs (180 left empty and removed; 48 keep `train.log`, `comp_plain` keeps 2 tracked JSON), 1c 23 `.best.pt` in `code_decay` / `code_ablate` | before: md5 manifest re-verified, no process / open file under `runs/`, no symlink target hit, no candidate read by `test_guards`, `stats_guard` or an `analyze_*`; after: the before/after listing of `runs/` differs by exactly the 920 manifest paths, `git ls-files -d runs` empty, no dangling symlink. Kept as ambiguous (0.11 GB): the planted `PoPE-NoSigma_s2.best.pt.stale` fixture, `tw_landmark_pilot`, `dyck_t2`, `ctxstep_pilot`, `hs_recipe_pilot`, `signsep_pilot`, `textworld_pilot`. Manifest and reasons: `docs/cleanup/D1_MANIFEST.md` |

## Group 2

| step | commit | what was done | counts | checks / notes |
|---|---|---|---|---|
| A2 | `0867a21` | withdrawn / void results to `archive/withdrawn/` | 34 files: ACCUMULATOR, DXR_RANK_THRESHOLD, DYCK_T3_RESULTS, ENWIK8_HIER, HIERGOAL_ABLATION, HORIZON_RESULTS + 6 per-arm tables + TASK_DISTANCES, LAP_TRANSFER_NOREWARD, N3_AUDIT, PLANNER_TASK_AUDIT, REPORT_ADDENDUM, REPORT_v2, RESULTS_PAPER, RESULTS_SUMMARY_2026-05-10 (with .json / .pdf) | none cited in CLAUDE.md's citable region or the drafts; no reader script (writers only). Paths rewritten in CLAUDE.md (Withdrawn list, Dyck run-dir row), README, a RESULTS_INDEX hand row, `project_clock_vs_map.md`. **Skipped:** the LOOPED_PILOT record (`agg_looped.py` reads `LOOPED_*.json` by path); the CODE_* and RECENCY_T3_* families (registered; their pre-registrations may not move and a record is not split) |
| A3 | `241b645` | superseded files to `archive/superseded/`; orphans to `archive/orphan/` after reading each | 15 superseded (ADDITION_PILOT, COMPOSITIONAL_RESULTS, DXR_PRELIM, INDEX_BASELINE_PAPER_TASK, MINIGRID_2X2X2, MINIWORLD_FIXED/FRESH_RESULTS, MINIWORLD_ORACLE_RESULTS_T512, PAPER_OOD_EXTENDED, REPORT; with .json / .pdf); 21 orphans (compositional Match-Query sweep, GENERALIZATION_REPORT, HIERGOAL_LONGT, MAPPOPE_VS_POPE, MINIGRID_2X2, MINIWORLD_RESULTS / _TODO, OMEGA_RESCALE_clean, PUBLICATION_VENUES, RECURSIVE / ROOMS_GOAL / ROUTE_ATTN / SPACETIME_HIER results, SESSION_HIERARCHICAL_CASCADE, THEORY_NARRATIVE) | **Kept as records after the read:** TINYSTORIES_PILOT and TEM_RECENCY_PILOT (readouts declared before the runs), AGGREGATE_MULTISEED and AGGREGATE_TASK_RESULTS (numbers quoted in `project_hierarchy_negative.md`), LEARNED_RANK (the probe behind rule 9's lesson), CONV_KERNEL_PROBE (cited by `model_selective.py`), DYCK_SHAREABLE (Dyck family prefix), MATCH_GATES_64_16 (+json) and MINIWORLD_GATES_FIXED (+json) (task gates of results that stay), MINIWORLD_FRESH_FINDINGS (the write-up behind the MiniWorld memory). KNOWN_BUGS notes the new location (the aggregator default it describes is unchanged) |
| A4 | `1612ee0` | Level 1.5 / InEKF / PC / TEM / lm200-era results to `archive/legacy_level15/` | 37 files (CORRECTED_LM200_LEADERBOARD, LM200_CORRECTED_MULTISEED, LM200_ABLATION, capacity, cascade, GSF, NoDrop, NoBypass, V3 / V4, TEM, number line, vector nav, vocab sweep, LEVEL15_MEETS_GATED; with .json) | no reader script. The cited live negatives (L15_ABLATION, EXTRAHEAD_CONTROL, NOISE_REFINE, MQ_NOISE_2X2*) stay at the root; `archive_stale/` copies untouched (CLAUDE.md cites `archive_stale/V4_CONTROL_RESULTS.md`). Paths rewritten in CLAUDE.md, README, a RESULTS_INDEX hand row and two memory files; MOVED.md gained a folder legend. SUPPORTING files the inventory proposed for `archive/reports_old/` and GENERATED outputs proposed for `archive/generated/` were not in A2-A4 and stay |
| B1 | `7c7e1ac` | dead and superseded scripts to `archive/code/2026-10/` with `INDEX.md` | 52 of the inventory's 60 | re-checked with `git grep -w <stem>` per file. **Kept (live reference found):** `analyze_t1.py` (`run_t1_centered.sh` produces T1_RESULTS with it; cited by THEORY_MAPPOPE), `environment_textworld_ctx{,2}.py` and `train_ctxstep{,2}.py` (imported by `docs/audits/2026-09-27/` gate and swap scripts), `hyak/mm_gate_v2.slurm` (cited by MEMORY_MAZE_PREREG), `docs/theory/2026-10-04/scripts/{toy,basins2}.py` (`toy2.py` imports `toy`; the theory folder is an audit record) |
| C3 | `51d18fb` | `docs/paper/tools/build_pdfs.py` builds `main_paper_draft2.pdf`, `_appendix.pdf`, `_with_appendix.pdf` (MAIN_DRAFT2 + MAIN_APPENDIX2 in one `build()` call) plus the theory appendix, draft 6, its appendix and the figure notes; no longer draft 1 or its outline | 14 superseded PDFs (0_COMBINED, 1_, 2_, 6_, 8_..17_) and `paper_sources_and_figures.zip` to `docs/paper/old/pdf/` (9.4 MB) | rebuilt into a scratch folder: `pdftotext -layout` output identical to the committed PDFs for all three draft-2 files (26 / 80 / 106 pages, same image counts) and for 3_, 4_, 5_. `7_figures_and_captions.pdf` differs (238 lines; 11 vs 12 pages): the committed PDF predates `figs/FIGS_NOTES.md`; left as committed. No reference to any archived PDF anywhere |
| C4 | `834ad52`, `ad31611` | superseded drafts, outlines and reviews to `docs/paper/old/` (with `README.md`); redirect stubs at the old paths of the four files pre-registrations cite | 19 files (MAIN_DRAFT1, MAIN_APPENDIX1, MAIN_DRAFT1_notes, OUTLINE_main_v2 / overall / workshop, DRAFT_workshop, DRAFT3/4/5 workshop and changes, APPENDIX_draft5, APPENDIX_theory v1 / v2, READ_draft4, MOCK_REVIEW_draft3, VENUES_2026-10-15); 4 stubs (MAIN_DRAFT1 for SROPE_TORUS and TORUS_LRSWEEP, DRAFT5_workshop for LRSWEEP, READ_draft4 for ROBUST, MOCK_REVIEW_draft3 for GAUGE) | two commits so the moves are pure renames (`git log --follow` works); stubs keep every pre-registration link resolving. `project_state.md` and the RUNBOOK's "Where things are" rewritten. `docs/plans/` (cited by RANK_TESTS, ROBUST, TORUS_START pre-registrations) left in place |
| C6 | `e5640d0`, `7e2d0c7`, `2c39d0d` | README: the April-May README archived as `archive/superseded/README_2026-05_inekf_framing.md` (pure rename), a short accurate README written (read order CLAUDE.md -> WHERE_THINGS_STAND -> project_state -> drafts 2 / 6 -> RESULTS_INDEX; quick start; layout). `docs/WHERE_THINGS_STAND.md` rewritten as a one-page orientation (thesis, what survives with status, documents, what is open, what is stale, method). CLAUDE.md: documents row names draft 2 and draft 6 with their PDFs, claim map, review and build command, the September documents marked stale; Operations rule 30 (archived batches are reproducible from their pre-registration commit, not HEAD). Memory: 12 files corrected with dated notes (MEMORY.md, project_state, reference_review_documents, reference_positional_landscape, project_hierarchy_negative, project_paper_direction, project_sign_axis, project_rank_and_selective_rope, project_clock_vs_map, project_what_where_and_language, project_mappope_asymmetry, project_loop_and_correction), 1 retired (`feedback_baselines_backfill.md` to `.claude-memory/archive/`, README there); MEMORY.md index lines updated. Catalogue regenerated after the moves | 13 memory files | every backticked path in CLAUDE.md, README, WHERE_THINGS_STAND and memory resolves, except names that pre-date this cleanup (`STOCHASTIC_TRANSITION_RESULTS.md`; `HIER_ATTN_MULTIENV.md`, which is in `archive/void/`; `neuro_design.md`, relative to `docs/theory/2026-10-05/`) and short PDF names relative to `docs/paper/pdf/` |

Commit count: 17 (C4 in two parts and C6 in three, so that moved files stay pure renames for `git log --follow`).

## Incident

During B1 an unquoted heredoc let bash execute backtick-quoted fragments of the INDEX text as commands (`git mv` with no
arguments, a few file paths, `toy`, `analyze_t1.py`). Every one failed harmlessly (usage message, "Permission denied" on
non-executable files, "command not found"); `git status` showed nothing changed besides the intended moves, no file was
created, and the INDEX header was rewritten from a quoted heredoc before the commit.

## Left for the user

- `docs/paper/pdf/7_figures_and_captions.pdf` is older than `figs/FIGS_NOTES.md`; rebuild with the fixed script if wanted.
- Not moved although an inventory proposed it: the LOOPED_PILOT record (read by `agg_looped.py`), the CODE_* and
  RECENCY_T3_* registered families, the five SUPPORTING report files (`AUDIT_HEADLINE`, `DETAILED_RESULTS`,
  `MAESTRO_PLAN`, `PERCEPTION_EXPERIMENT_PLAN`, `SESSION_2026-05-01`), the ~40 GENERATED outputs (`archive/generated/`
  in the inventory), `docs/plans/`, and the 11 orphans kept as records (A3).
- D1 kept seven ambiguous items (0.11 GB; list above and in `D1_MANIFEST.md`).
- `.run_signsep.lock.stale_20261024` (untracked, repo root) and a pre-existing `git stash` entry ("leftover deletions
  from prior cleanup") were not touched.
- Old drivers retrain any run whose checkpoint is missing: do not relaunch the drivers of the D1 dirs.

## D2 (2026-10-29, second pass): disk tiers D2a / D2b / D3, steps R1-R4

Approved: delete the SAFE groups (R1, R2); extract, verify, then delete the SAFE AFTER EXTRACTION groups (R3, R4);
guard the drivers. Not touched: the KEEP groups (`runs/code_ablate/*.final.pt`, `runs/selective`, `runs/em_fig4`,
TinyStories valid text), `runs/code/*.final.pt`, `runs/code_decay/*.final.pt` and `/home/prashr/mm_data`. Plan and
verdicts: `DISK_TIERS_ANALYSIS.md`; manifest: `D2_MANIFEST.md`. Nothing was running (no `python3` process,
`nvidia-smi` empty, `lsof` clean) before or during.

| step | commit | what was done | counts | checks / notes |
|---|---|---|---|---|
| X2 extraction | `f203593` | every non-tensor value of the 2,000 checkpoints of R1-R3 + `em_fig4` (kept) dumped to `runs/<dir>/CKPT_META.jsonl` (format and extractor `docs/cleanup/extract_ckpt_meta.py`): per-epoch `losses`, `config`, stored `results`, variant, seed, sizes, plus the checkpoint's md5 and byte size | 46 sidecars, 6.6 MB, 1,762 entries; 15 dirs of bare state dicts (238 files: `indirect*`, `jsb*`, `dyck_t3`) have nothing to dump | `--check` re-read all 1,762 entries against `torch.load`: 0 mismatches (losses bit-identical); sidecar md5 = manifest md5 on all 1,746 deleted files they cover. JSON lines, not `.json`, so no `*.json` glob in a run dir picks them up |
| X3 readers | `f203593`, `516943d` | `ckpt_sidecar.py` (`load_ckpt_meta`: the `.pt` if present, else the sidecar); fallback added to `agg_alias.py`, `agg_visits.py`, `agg_miniworld.py`, `experiment_audit.py` (MiniWorld layout) and `analyze_papertask.py` (found in the pre-deletion reader check: it would have fallen back to 4-decimal log losses) | 5 readers | none md5-recorded or named in a pre-registration. Output byte-identical for HEAD + `.pt`, patched + `.pt`, patched + sidecar only, on 27 invocations (`agg_alias`, `agg_visits`, `agg_miniworld` x4, `experiment_audit` x13 incl. the documented `--control GateDeltaCtl` example) and again after deletion; `agg_alias` reproduces `ALIASING_CONTROLLED.md` and `agg_visits` `VISITS_TEST.md` byte for byte; `analyze_papertask` leaves the tracked `runs/paper_task_rerun/REPORT.md` unchanged. `run_mw_gatectl.sh`'s inline table still reads `.pt` (driver guarded) |
| verification | `f203593` | `docs/cleanup/verify_d2_sidecars.py` re-derives from the sidecars alone (output `verify_d2_sidecars_out.txt`, identical before and after deletion) | -- | **rule 2**: 51 `mw_grid_sweep` runs alone give r -0.9968, acc = 1.033 - 0.549*loss, sd 0.019; adding `miniworld_oracle` RoPE_oracle and one of three other 3-run cells (MapPoPE-Flat_oracle, PoPE-Flat_allo of `miniworld_oracle` or of `miniworld_fresh`) gives n=57, r -0.9963/-0.9964, acc = 1.039 - 0.555*loss, resid sd 0.021: the documented numbers. The exact set was never recorded, so it stays ambiguous among those three. **Map extent**: all 12 per-seed final losses and final-10% slopes of `MINIWORLD_ENDPOINTS.md` (+0.305 / +0.015 endpoints) and the three rows of `ALIASING_CONTROLLED.md`'s loss table reproduced exactly. Rule 1's twin floor 0.150 (n=9) reproduced by `experiment_audit` from the sidecars |
| X1 backup | -- | every file under `runs/` except `*.pt`, `*.pt.stale`, `*.pkl` tarred to `/home/prashr/mapformer_backups/runs_nonckpt_2026-10-29.tar.gz` | 19,209 entries (6,832 `.log`), 13,958,968 bytes, md5 `f57e41b73cbef05113dfe1ced1e1eeff` | outside the repo; the gitignored logs are the only other loss record |
| guards | `dde0630` | refusal at the top of each driver (exit 3, message points to `DISK_TIERS_ANALYSIS.md` section 4) unless `MAPFORMER_ALLOW_DELETED_RERUN=1`; CLAUDE.md Operations rule 31 (one line) | 54 drivers | `bash -n` clean on all; refusal and override tested. List below |
| X6 TinyStories | `f0507e7` | md5, sha256, size, source URL and download date of `TinyStoriesV2-GPT4-{train,valid}.txt` recorded in `D2_MANIFEST.md` | train md5 `f14749c06624ab6311c8e50ca80f4043`, sha256 `6418d412...0443` | source `huggingface.co/datasets/roneneldan/TinyStories` (`resolve/main/TinyStoriesV2-GPT4-train.txt`), downloaded 2026-10-09; analyses read `train.bin` / `val.bin` / `vocab.json` (kept) |
| R1-R4 deletion | `f0507e7` (manifest, committed before deleting) | deleted from the md5 manifest, re-verified with `md5sum -c` immediately before; then the 120 dangling `runs/vocab_em_link/**` file symlinks (they pointed at `runs/vocab_em` checkpoints) and the 803 directories left empty (rmdir, bottom-up, only ancestors of deleted paths) | **1,985 files, 19,062,795,726 bytes = 17.75 GiB**: R1 652 files 5,146,615,544 B; R2 96 files 7,023,945,509 B; R3 1,236 files 4,664,481,511 B; R4 1 file 2,227,753,162 B | after: the before/after listing of `runs/` differs by exactly the 1,984 manifest files, the 120 symlinks and the 803 empty dirs (nothing added or changed); `git ls-files -d runs` empty; `find runs -xtype l` empty (`n5_phase/torus_eval_stage/p0` still resolves). `df /`: 71,829,139,456 -> 90,898,354,176 bytes free (67G -> 85G) |
| checks | -- | from `/home/prashr`, `CUDA_VISIBLE_DEVICES=""`, before the commits and after deletion | -- | `test_guards` 13/13; the 16 `--selftest` analyses pass; `import mapformer.train_variant` and the five readers OK |

Guarded drivers (54): the 44 top-level `.pt`-gated drivers of `DISK_TIERS_ANALYSIS.md` section 4 (`run_mw_grid_sweep`,
`run_mw_grid_sweep_hier`, `run_mw_hier_ablation`, `run_mw_gatectl`, `run_mw_probe3`, `run_mw_probe3_seeds`,
`run_mw_srope_components`, `run_miniworld`, `run_mw_missing`, `run_mw_fresh_factorial`, `run_mw_oracle`,
`run_alias_sweep`, `run_alias_finish`, `run_alias_followup`, `run_alias_waveb`, `run_rope_converge`,
`run_vanilla_converge`, `run_g8_converge`, `run_visits_test`, `run_flipflop`, `run_frontier`, `run_vocab_em`,
`run_minigrid_em`, `run_minigrid_em_fix`, `run_pope_hier_allo`, `run_n5_phase`, `run_cmq_curric`,
`run_cmq_curric_full`, `run_cmq_stab`, `run_cmq_sweep`, `run_recipe_power`, `run_papertask_rerun`,
`run_mq_noise_2x2`, `run_mq_noise_c2`, `run_l15_loop_2x2`, `run_l15_loop_supp`, `run_l15_ablation`,
`run_noise_refine`, `run_loop_sampled`, `run_forget`, `run_forget_control`, `run_lambda_trace`, `run_popewrap`,
`run_mappope_r4`, `run_t3gen2`); 6 archived (`archive/code/2026-10/run_mw_nope`, `run_minigrid_allo`,
`run_cmq_topup`, `run_cmq_curric_topup`, `run_cmq_stab_topup`, `run_t3gen`); and 3 eval drivers:
`run_code_fullval.sh` (would rewrite the tracked `runs/{code,code_decay}/FULLVAL_512.json`),
`run_papertask_eval.sh` (rewrites `PAPER_OOD_RERUN.json`), `run_accumulator.sh` (loads `forget`, `l15_ablation`,
`popewrap` checkpoints; not in the analysis, found by grepping every `*.sh` for the deleted dirs).

Not guarded, on purpose: `run_selective.sh` and `run_em_fig4.sh` (their dirs are KEEP, checkpoints present); the
JSON-gated drivers, which skip every finished run (`run_t3`, `run_jsb_len`, `run_jsb_rank`, `run_t1_centered`,
`run_cross_jsb`, `run_decay`, `run_aug`, `run_jsb`, `run_indirect`, `run_indirect_more`, `run_indirect_200k`,
`run_code`, `run_code_decay`, `run_tinystories` (its completion check needs `.best.pt`, kept), `run_algorithmic`,
`run_hier_parity`, `run_loop_hier`, `run_rope_canonical`, `run_sameblock`, `run_counter_batch`, archived
`run_recency_budget`, `run_sameblock_amend`); `run_spread2.sh` (only waits on a marker);
`docs/audits/2026-09-24/applied/run_evals_extra.sh` (an audit record, left unedited; it evaluates `noise_refine` /
`l15_ablation` checkpoints into a caller-given directory and would now fail loudly).

Kept and why: everything the user did not approve (above); `runs/em_fig4/*.pt` (KEEP; its sidecar was written anyway,
as the analysis recommended); the run JSON, logs, markers and md5 records of every touched dir; the top-level dirs
themselves (each holds its sidecar or JSON). Deviation from "one commit per step": the deletion step has two commits
(manifest before deleting, as X1 requires; this log after), and the `analyze_papertask.py` fallback is a separate
addendum commit because the need was found during the pre-deletion reader check.

Lost (re-analyses that need weights, listed in `D2_MANIFEST.md`): e.g. `analyze_t1.py`, `analyze_sameblock.py`'s
cosines, `probe_theory_numbers.py`, `eval_indirect_ood.py`, the code-depth / accumulator probes on `code/*.best.pt`,
`eval_noise_refine.py` / `eval_loop_sweep.py` / `probe_forget.py` / `probe_accumulator.py` on these dirs; their stored
outputs stay. CROSS Amendment 1's Bach r = 1.000 (no committed script) can no longer be re-derived.
