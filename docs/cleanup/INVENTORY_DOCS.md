# Inventory: documents, papers, archives, memory (2026-10-28, read-only)

Scope: `docs/` (1,849 files, 41 MB), `report/`, `paper/`, `paper_rank/`, `paper_figures/`, `archive/`,
`archive_stale/`, `.claude-memory/`, the top-level PDFs, and the `papers/` corpus. Nothing was moved, edited,
deleted or committed. This file is the only write. Reference counts come from an exact-path substring search over every
tracked `.md/.py/.sh/.tex/.html/.txt/.json` outside `runs/` (script kept in the session scratchpad, not committed).
"Ignored" means matched by `.gitignore`. Inside these folders, every untracked file is ignored (`git status --ignored`).

Classes: CURRENT, SUPERSEDED, AUDIT-RECORD, ORPHAN, GENERATED, DUPLICATE (definitions as in the brief).

---

## 1. Counts and sizes per class

These are file-level counts from rules applied per path (rules in Section 2). Total: 2,213 files, 94 MB, not counting
`papers/`.

| class | files | of which tracked | size | what dominates |
|---|---|---|---|---|
| CURRENT | 283 | 283 | 3.9 MB | `docs/paper/expansion/scripts/` (337 incl. ignored), draft 2 / draft 6 sources, figure scripts, `docs/lit`, memory |
| AUDIT-RECORD | 644 | 643 | 5.9 MB | `docs/audits/<date>/` (666 files), `review_draft5/`, `appendix_checks/`, `docs/theory/`, `archive/void/` |
| SUPERSEDED | 522 | 522 | 18.3 MB | old drafts, superseded `docs/paper/pdf/*` (9.4 MB), `docs/prepared/` (minus pkg), `paper_figures/` PNGs, `report/` Sept notes, `paper/`, `paper_rank/` |
| GENERATED | 368 | 182 | 63.5 MB | `paper_figures/*.npz` (48 MB, 32 MB of it tracked), current PDFs, figure PNG/PDF, LaTeX aux, ignored run logs/pycache |
| DUPLICATE | 394 | 393 | 2.6 MB | `docs/prepared/dyck_mdepth/pkg/mapformer/` (374-file package copy), 20 `archive_stale/` files byte-identical to top-level files |
| ORPHAN | 2 | 2 | 0.03 MB | `docs/inekf_mapformer_writeup.docx`, `docs/mapformer_implementation_guide.docx` |

Sizes per folder (du):

| folder | size | files | note |
|---|---|---|---|
| `papers/` | 217 MB | 160 | `pdf/` 136 MB ignored; `new_2026-10-1{4,5,6}` 80 MB, PDFs untracked and NOT ignored |
| `paper_figures/` | 53 MB | 169 | 6 x 8 MB `.npz` (4 tracked despite `paper_figures/*.npz` in `.gitignore`) |
| `docs/paper/` | 26 MB | 492 | `pdf/` 17 MB (22 files, tracked), `expansion/` 4.1 MB, `figs/` 3.9 MB |
| `docs/audits/` | 7.5 MB | 666 | 23 date folders; largest 2026-10-14 (1.7 MB), 2026-09-24 (1.6 MB) |
| `docs/prepared/` | 6.5 MB | 650 | `dyck_mdepth/pkg` 3.5 MB, `mock_runs` 2.5 MB |
| `report/` | 2.4 MB | 34 | 11 LaTeX artefacts tracked |
| `paper/` | 1.0 MB | 17 | retracted April draft |
| top-level `*.pdf` | 7.6 MB | 9 | 3 are CLAUDE.md "documents"; 6 others (Section 2.9) |
| `docs/theory/` | 388 KB | 18 | |
| `archive/` | 280 KB | 57 | |
| `paper_rank/` | 256 KB | 15 | |
| `.claude-memory/` | 224 KB | 32 | |
| `docs/lit/` | 176 KB | 6 | |
| `archive_stale/` | 148 KB | 35 | |

Largest files: 6 x `paper_figures/*_rate_maps*_s0.npz` (8 MB each); `docs/paper/pdf/0_COMBINED_all_documents.pdf` 4.0 MB;
`main_paper_draft2_with_appendix.pdf` 2.8 MB; `paper_sources_and_figures.zip` 2.6 MB; `paper_figures/fig10_hidden_rate_maps.png`
2.4 MB; `main_paper_draft2.pdf` 1.8 MB; `paper_figures/fig7_rate_maps.png` 1.7 MB; `1_main_paper.pdf` 1.3 MB.
Git pack is 74 MB. `docs/paper/pdf/` has been committed 16 times, and each draft-2 rebuild adds about 5.8 MB of blobs.

---

## 2. Per-folder tables with proposed action

Actions: **keep** (in place) / **archive** (move under `docs/archive/<topic>/`, with a redirect line) / **delete-regen**
(remove from the tree; it can be regenerated, and git history keeps it) / **untrack** (`git rm --cached`, file stays) / **consolidate**.

### 2.1 `docs/paper/` top level (28 .md)

| file(s) | class | refs (who cites) | action | reason |
|---|---|---|---|---|
| `MAIN_DRAFT2.md`, `MAIN_APPENDIX2.md`, `MAIN_DRAFT2_notes.md` | CURRENT | each other, FULL_REVIEW, CRITICAL_REVIEW 10-26, g3_ssm_audit, FIGS_NOTES | keep | live main paper (updated 2026-10-28) |
| `DRAFT6_workshop.md`, `APPENDIX_draft6.md`, `DRAFT6_changes.md` | CURRENT | project_state, draft-2 notes | keep | live sign companion (2026-10-26) |
| `APPENDIX_theory_v3.md` | CURRENT | MAIN_APPENDIX2, DRAFT6, APPENDIX_draft6 | keep | cited by both live drafts |
| `CLAIM_MAP_clock_map.md` | CURRENT | project_paper_direction, RING_DIAL_PREREG | keep | updated 2026-10-28 |
| `BIG_PICTURE_2026-10-24.md` | CURRENT | RNN_TORUS_PREREG | keep | latest advisor read; de facto orientation (see 2.4 WHERE_THINGS_STAND) |
| `MAIN_DRAFT1.md`, `MAIN_APPENDIX1.md`, `MAIN_DRAFT1_notes.md` | SUPERSEDED | **SROPE_TORUS_PREREG, TORUS_LRSWEEP_PREREG**, RUNBOOK, 7 expansion reports | archive -> `docs/paper/old/` | replaced by draft 2. Registered preregs cite it, so it needs a redirect (Section 6) |
| `OUTLINE_main_v2.md`, `OUTLINE_overall.md`, `OUTLINE_workshop.md` | SUPERSEDED | MAIN_DRAFT1, project_state, DRAFT6_changes | archive -> `old/` | outlines of drafts 1-5 |
| `DRAFT_workshop.md`, `DRAFT3_*`, `DRAFT4_*`, `DRAFT5_*`, `APPENDIX_draft5.md` | SUPERSEDED | project_state; **LRSWEEP_PREREG** (DRAFT5_workshop); review_draft5 | archive -> `old/` | replaced by draft 6 |
| `APPENDIX_theory.md`, `APPENDIX_theory_v2.md` | SUPERSEDED | old drafts only; THEORY_REVIEW cites DRAFT_workshop | archive -> `old/` | replaced by v3 |
| `READ_draft4.md`, `MOCK_REVIEW_draft3.md`, `VENUES_2026-10-15.md` | SUPERSEDED | **ROBUST_PREREG** (READ_draft4), **GAUGE_PREREG** + 2026-10-15/reanalyses (MOCK_REVIEW_draft3), project_state | archive -> `old/` | reviews of superseded drafts |

### 2.2 `docs/paper/` subfolders

| folder | files / size | class | action | reason |
|---|---|---|---|---|
| `figs/` `*.py`, `*_out.txt`, `*.json`, `FIGS_NOTES.md`, `figstyle.py` | ~50 | CURRENT | keep | 15 figures cited by draft 2 / draft 6. Regenerate: from `/home/prashr`, `CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/prashr python3 mapformer/docs/paper/figs/fig_<name>.py` (FIGS_NOTES). 3 scripts read `runs/` dirs, so they can be regenerated only while those runs exist |
| `figs/*.png`, `figs/*.pdf` | ~32 | GENERATED | keep (tracked) | `build_pdfs.py` embeds the PNGs by path. Small (3.9 MB total incl. scripts) |
| `expansion/scripts/` (7 topic dirs, 337 files, 3.8 MB) | CURRENT | keep in place | cited by MAIN_APPENDIX2 and by registered analysis code **at runtime** (`analyze_delta_free.py` loads `scripts/mappope/kernel_probe.py`, `analyze_torus_lrsweep.py` loads `scripts/sign/clock_table.py`); 39 scripts hard-code `/home/prashr/mapformer/docs/paper/expansion/...`. `sign/runs_basin*`, `recurrence/out/logs` are ignored GENERATED (0.8 MB) |
| `expansion/*.md` (8 reports, 0.33 MB) | SUPERSEDED | keep in place | inputs to draft 2, folded in. Cited by draft-2 notes, `model_pope_delta.py` (MAPPOPE.md Prop. 1), DELTA_FREE prereg/results. Moving them saves nothing |
| `review_draft5/` (9, 324 KB) | AUDIT-RECORD | keep | ANALYSIS_SCRIPT_AUDIT, REPRO_AUDIT, theory_check_extra and PRIOR_ART are cited by MAIN_APPENDIX2 / notes. "[A-audited]" labels depend on it |
| `appendix_checks/` (17, 100 KB) | AUDIT-RECORD | keep | numeric checks behind the theory appendix (dir cited 9x in docs) |
| `tools/build_pdfs.py` | CURRENT (stale targets) | keep, fix later | **builds draft 1 into `1_main_paper.pdf` and does NOT build the draft-2 PDFs.** RUNBOOK l.69 tells a new machine to run it. No committed recipe for `main_paper_draft2*.pdf` exists (commits 74f3807, 05e22b5, fd256da only say "rebuilt") |
| `pdf/main_paper_draft2{,_appendix,_with_appendix}.pdf`, `4_sign_workshop_paper.pdf`, `5_sign_workshop_appendix.pdf`, `3_theory_appendix.pdf`, `7_figures_and_captions.pdf` (8.0 MB) | GENERATED (current) | keep | shareable builds of the live sources. Weasyprint from markdown (`build()` in build_pdfs.py) |
| `pdf/0_COMBINED_all_documents.pdf`, `1_main_paper.pdf`, `2_main_paper_appendix.pdf`, `6_main_paper_outline.pdf`, `8_..` to `17_..` (draft 1, outline, expansion reports, two results notes; 2026-10-20/23) | SUPERSEDED + GENERATED | delete-regen | 6.7 MB. `1_main_paper.pdf` is draft 1, and its name invites sharing the wrong paper |
| `pdf/paper_sources_and_figures.zip` (2.6 MB, 2026-10-19) | DUPLICATE | delete | a snapshot of sources that git already holds |

### 2.3 `docs/audits/` (23 date folders, 666 files, 7.5 MB)

All AUDIT-RECORD except the two latest reviews (CURRENT) and the ignored `__pycache__`/logs (GENERATED). Each date folder is
cited by at least one prereg, results file or analysis script: 42 `*_PREREG.md` and 28 `*_RESULTS.md` cite `docs/audits/`.
The exceptions are `2026-09-25/rank_mi_review/` (7 files, 53 KB, no external reference) and, inside 2026-09-24, `context/`,
`hygiene/`, `rank_review*/` and `theory_REPORT.md`. Those are referenced only by `2026-09-24/README.md`, which `CLAUDE.md`
cites via `applied/`.
**Action: keep every folder in place.** Reasons:

- At least 30 audit scripts hard-code absolute paths or `sys.path.insert` into their own or sibling date folders.
  Examples: `2026-10-05/gain_vs_place.py` -> `2026-09-27`; `2026-10-15/reanalyses/capture_by_channel.py` -> `2026-10-14/theory_review`.
- `analyze_mm.py` (the registered Memory Maze analysis) reads `docs/audits/2026-10-27/memory_maze_gate_v3/diag_eval_split.json`
  and `.../memory_maze_batch/floors_held.json` at runtime.
- Amendments in registered preregs cite audit files by path, and preregs must not be edited.

| date | files | size | contents (main) | class / action |
|---|---|---|---|---|
| 2026-09-24 | 129 | 1.6 MB | context / efficiency / hygiene / rank_review(2) / applied (bit-exact checks cited by CLAUDE.md rule 19) | AUDIT-RECORD, keep; `efficiency/proposed`, `*.patch` already applied (could go to an `applied-history` note, low value) |
| 2026-09-25 | 7 | 76 KB | rank_mi_review | AUDIT-RECORD, unreferenced; keep (tiny) or archive |
| 2026-09-27 | 35 | 348 KB | gates for cancel / ctxstep / newobj / rank3, floors | AUDIT-RECORD, keep |
| 2026-10-03..10-13 | 205 | 2.1 MB | what/where, dropout rescore (CLAUDE.md), gain_grain, gain_phase, tw gates, rank_tests, CRITICAL_REVIEW 10-11 (CLAUDE.md), clock/ring audits, wrap existence | AUDIT-RECORD, keep |
| 2026-10-14 | 153 | 1.7 MB | accounts/generality/mechanism/torus_start audits, `cope_diagnosis/`, `theory_review/` (toy_runs ignored, 0.7 MB) | AUDIT-RECORD, keep |
| 2026-10-15..10-27 | 133 | 1.6 MB | gauge, lrsweep, robust, delta_free, g3/g4, factorial_n8, image_torus, exptorus, rnn_torus, memory_maze (gate v2/v3, batch) | AUDIT-RECORD, keep |
| 2026-10-26 `CRITICAL_REVIEW.md`, 2026-10-28 `FULL_REVIEW.md` | 2 | 0.1 MB | latest reviews of draft 2 / 6 | CURRENT, keep |
| `*/__pycache__`, `*.log` (ignored) | ~25 | 0.2 MB | | GENERATED, ignored, leave |

One tracked log: `2026-10-06/gain_phase_pilot.log`, which is tracked even though `*.log` is ignored. Harmless.

### 2.4 `docs/` top level

| file | class | refs | action | reason |
|---|---|---|---|---|
| `LOG.md` (185 KB) | CURRENT (history) | CLAUDE.md, memory, RESULTS_INDEX, audits | keep | CLAUDE.md's history pointer |
| `WHERE_THINGS_STAND.md` | CURRENT, **stale** | CLAUDE.md ("read second"), MEMORY.md, project_state, claim map | keep, refresh | dated 2026-10-03 (+10-09 banner); nothing from 10-11..10-28. BIG_PICTURE_2026-10-24 and FULL_REVIEW are the real orientation now |
| `RUNBOOK_NEXT_EXPERIMENTS.md` | CURRENT, partly stale | project_state, G3_SSM_PREREG, TORUS_LRSWEEP_PREREG, HYAK_NOTES | keep | l.69 points at `build_pdfs.py` (see 2.2) |
| `HYAK_NOTES.md`, `IMAGE_TASKS_SURVEY_2026-10-25.md` | CURRENT | memory, IMAGE_TORUS prereg/results, reviews | keep | |
| `SESSION_2026-09-27_to_10-03.md`, `SESSION_2026-10-04_to_10-09.md` | AUDIT-RECORD (narrative) | CLAUDE.md, MEMORY.md, RESULTS_INDEX, WHERE_THINGS_STAND | consolidate -> `docs/sessions/` (optional) | needs 5 rewrites; low value |
| `SESSION_2026-10-12_autonomous.md` | AUDIT-RECORD, unreferenced | none | move with the other SESSION files | |
| `WHAT_WHERE_ANALYSIS.md`, `WHAT_WHERE_CHECKS.md`, `NORMSTEP_NOTES.md` | AUDIT-RECORD | CLAUDE.md citable row (CHECKS); GAIN_GRAIN/LEAK/MAPPOPE_PAIR/GAIN_PHASE/TW_NORMSTEP preregs; code docstrings | keep | cited by preregs |
| `kalman_mapformer_intuition.md` (2026-04-26) | SUPERSEDED | RESULTS_INDEX only | archive -> `docs/archive/2026-04_kalman/` | Kalman/InEKF line is a live negative |
| `inekf_mapformer_writeup.docx`, `mapformer_implementation_guide.docx` | ORPHAN | none | archive with the Kalman note (or ask the user) | no reference anywhere; binary |
| `plans/2026-10-11_next_gpu.md`, `plans/2026-10-14_next_steps.md` | SUPERSEDED | **RANK_TESTS_PREREG, ROBUST_PREREG, TORUS_START_PREREG**, LIT_REPRESENTABILITY, old drafts | keep in place, or archive with redirect | superseded by BIG_PICTURE / project_state. Cited by preregs |
| `lit/LIT_*.md` (6) | CURRENT | CLAUDE.md ("check before claiming novelty"), 39 files | keep | |
| `tools/catalog_results_index.py` | CURRENT | CLAUDE.md, RESULTS_INDEX | keep | it globs `docs/*.md` and `docs/lit/*.md`. Moving SESSION/kalman notes changes its catalogue output |
| `theory/2026-10-04/`, `theory/2026-10-05/` (18) | AUDIT-RECORD | 9 results files, `analyze_rank_nowrap.py`, `analyze_score_rank.py`, `model_em_pope.py` | keep | `scripts/basins.py` is cited as the definition used by registered analyses |

### 2.5 `docs/prepared/` (650 files, 6.5 MB): no reference from anywhere outside itself

| subfolder | class | action | reason |
|---|---|---|---|
| `dyck_mdepth/` deliverables (PREREG, analyze, validate, common, train_dyck.py, patch) | SUPERSEDED + DUPLICATE | archive -> `docs/archive/2026-09_prepared/` (or delete) | 5 of 7 files are byte-identical to the top-level copies that were launched (`run_dyck_mdepth.sh` and `lib_driver.sh` differ: the top-level ones are later) |
| `dyck_mdepth/pkg/mapformer/` (374 files, 3.5 MB) | DUPLICATE | delete | package copy used for gating 2026-09-25. README says it is identical to the repo except the patched files |
| `dyck_mdepth/mock_runs/`, `mock_out/`, `gate_*`, `gates/` (2.7 MB) | GENERATED (scratch) | delete-regen (`make_mock.py`) | README: "scratch only, never to be reused as runs" |
| `tex_edits_2026-09-25/`, `tex_edits_2026-09-27/` (17) | SUPERSEDED (applied) | archive | edit scripts for the .tex documents, already applied |

### 2.6 `report/`

| file(s) | class | refs | action | reason |
|---|---|---|---|---|
| `language_summary.html` | CURRENT | CLAUDE.md, memory x3, RESULTS_INDEX, LOG | **keep in place, do not touch** | source of the shared artifact https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc (v22, 2026-10-11). Republish only WITH `url=` |
| `navigation_in_words.html` | CURRENT | SESSION_2026-10-04 | keep | source of a second artifact (JmzdmkFup4V3kV3C5YndPm), last edited 2026-10-05 |
| `report.tex`, `report_short.tex`, `refs.bib`, `pending.tex`, `figures/` (make_*.py + 2 PDFs) | SUPERSEDED (CLAUDE.md "documents", stale) | 2026-09-24 audits, tex_edits | keep | CLAUDE.md lists `report/report.pdf`, `report_short.pdf` |
| `report.pdf`, `report_short.pdf` | GENERATED | CLAUDE.md | keep | `pdflatex`+`bibtex` from the .tex |
| `report*.aux/.bbl/.blg/.log/.out/.toc` (11) | GENERATED | none | **untrack** | tracked although `.gitignore` has `*.aux`, `*.log`, `*.out` |
| `STORY.md`, `FIXES.md`, `VERIFY.md`, `VERIFY_2X2.md`, `INVENTORY_BRIEF.md`, `inventory/` (6) (Sept 13-25) | SUPERSEDED | only `docs/audits/2026-09-24/experiments/PROPAGATION.md` | archive -> `docs/archive/2026-09_report_notes/` | working notes behind the September report |

### 2.7 `paper/`, `paper_rank/`, `paper_figures/`

| folder | class | action | reason |
|---|---|---|---|
| `paper/` (April draft; `RETRACTED.md`) | SUPERSEDED | archive -> `docs/archive/2026-04_paper_retracted/` | retracted 2026-09-04. `main.pdf` and aux files are ignored GENERATED |
| `paper_rank/` (Sept draft; `STATUS.md`) | SUPERSEDED | archive -> `docs/archive/2026-09_paper_rank/` | its thesis is the packing bound, which CLAUDE.md withdrew ("Packing geometry as the account of v4 Table 6"). `main.pdf` ignored |
| `paper_figures/` PNGs and subfolders (Apr-May: lm200 calibration, hippocampal / place / grid / hex, capacity, active inference) | SUPERSEDED | archive later (coordinate with the code inventory) | 20 legacy scripts write into it (`hippocampal_analysis.py`, `make_place_cell_figure.py`, `run_capacity_*.sh`, ...), and 28 top-level results files embed its paths |
| `paper_figures/*.npz` (6 x 8 MB) | GENERATED | **untrack the 4 tracked `cnav_*_rate_maps_s0.npz` (32 MB)** | `.gitignore` already says `paper_figures/*.npz`, but they were tracked before the rule. The `dog_*` two are ignored |

### 2.8 `archive/`, `archive_stale/`

| folder | class | action | reason |
|---|---|---|---|
| `archive/void/` (56, bannered) + `archive/RESULTS_INDEX_2026-08-18.md` | AUDIT-RECORD | keep | CLAUDE.md "void / stale results". `ablate_context.py`, `ablate_match_noise.py`, `audit_headline_numbers.py` reference `archive/` |
| `archive_stale/` 20 files **byte-identical to top-level files** (CLONE_ANALYSIS_LEVEL15PC, CLONE_TRANSFER_NOBYPASS, CNAV_HEX_x4, CNAV_RESULTS, DOG_RESULTS, HIPPOCAMPAL_x7, MULTICLASS, MULTISEED_FOLLOWUP, OOD_GRID, R_T_DISTRIBUTION_3WAY, ZERO_SHOT_TRANSFER_clean_brokeninit) | DUPLICATE | consolidate: keep the `archive_stale/` copy, remove the top-level copy (top-level inventory decides) | the 2026-05-21 cleanup (b18c509) copied rather than moved |
| `archive_stale/` 5 files that differ from same-named top-level files (NOBYPASS, V3, V4_RESULTS, V4_MULTISEED, V4_CONTROL) and 10 that also exist, bannered and different, in `archive/void/` | SUPERSEDED | keep; flag | three copies of some results with different banners. CLAUDE.md cites `archive_stale/DOG_RESULTS_FIXED.md` and `archive_stale/V4_CONTROL_RESULTS.md` |
| `archive_stale/CLONE_ANALYSIS_LEVEL15PC.md` | (0 bytes) | delete with its top-level twin | empty file |

### 2.9 Top-level PDFs and documents

| file | class | action | reason |
|---|---|---|---|
| `positional_review.{tex,pdf}`, `axes_measured.{tex,pdf}`, `mapformer_math.{tex,pdf}` | CURRENT source / GENERATED pdf; **stale** per CLAUDE.md (none carries rank 3, H1, H3, text world, ... NormStep) | keep | CLAUDE.md "documents" row. Build: `pdflatex` from source (aux/log ignored). 30+ code and docs references to the .tex |
| `MapFormer.pdf` (2.3 MB, unreferenced) | SUPERSEDED | move to `papers/pdf/mapformer_v1.pdf` (ignored) | it is arXiv **v1** (22 pp). `papers/pdf/mapformer.pdf` is v4 (32 pp), the reproduction target. Not a duplicate |
| `REPORT.pdf`, `REPORT_v2.pdf`, `GENERALIZATION_REPORT.pdf`, `RESULTS_SUMMARY_2026-05-10.pdf` | SUPERSEDED / GENERATED | archive (together with their top-level .md, top-level inventory) | Apr-Jun era; lm200-era content |
| `Prashant_Rangarajan_Generals_Document.pdf` (2.7 MB, tracked) | ORPHAN (personal) | **ask the user** | not project output |
| `artifacts_results_brief.html` (2026-09-03) | SUPERSEDED | ask / archive | an earlier artifact source, not the live one |

### 2.10 `papers/` corpus (217 MB)

- **No duplicates by arXiv id**: none of the 25 `new_2026-10-1{4,5,6}/<id>/` folders repeats an id in `fetch.sh`'s 42-key
  map, and no id repeats across the three `new_*` folders. No two PDFs share an md5.
- **Untracked PDFs (28, about 80 MB)**: `papers/new_*/**/paper.pdf` (and `liu2023.pdf`) show as `??`. Their `.txt` extractions
  are tracked (22 in 10-16 alone). The repo's policy for `papers/pdf/` is "PDFs ignored, txt tracked, `fetch.sh` restores".
  **Proposal: ignore them** (add `papers/new_*/**/*.pdf` to `.gitignore`) and record their URLs. `new_2026-10-15/` and
  `new_2026-10-16/` have `SOURCES.txt`; **`new_2026-10-14/` has none**, so write one, or extend `fetch.sh`, before relying on
  the ignore. Tracking them would add about 80 MB to a 74 MB pack.
- Naming mismatches in the main corpus: `pdf/09_Whittington_2025_Tale_of_Two_Algorithms.pdf` vs `txt/tale_two_algorithms.txt`
  (same paper, different key); `pdf/survey_geom.pdf` has no txt (2511.08243, which memory marks WITHDRAWN); txt without pdf:
  `position_coupling`, `puranik_janestreet`, `zhou_length_gen_not_robust`.
- `papers/INDEX.md` (and CLAUDE.md, memory) say "40 papers". The tree has 42 pdf / 44 txt plus 25 `new_*` papers, and INDEX
  does not list the `new_*` folders. They are cited by `docs/lit/LIT_REPRESENTABILITY_2026-10-14.md`, review_draft5 and the
  DRAFT3-6 changes.

### 2.11 `.claude-memory/` (32 files): CURRENT, keep; staleness in Section 5.

---

## 3. Consolidation proposals (ordered by value / risk)

1. **Fix the PDF build before moving anything** (no moves needed). `docs/paper/tools/build_pdfs.py` should build
   `MAIN_DRAFT2 (+APPENDIX2)`, `DRAFT6 + APPENDIX_draft6`, `APPENDIX_theory_v3` and `FIGS_NOTES` under the names that exist
   now (`main_paper_draft2*.pdf`, `4_/5_sign_workshop_*.pdf`). Then delete-regen the 13 superseded PDFs and the zip (9.4 MB).
2. **One `docs/paper/` with `old/`**: leave every CURRENT file at its path, because preregs, notes, reviews and code cite them.
   Move the 21 superseded `.md` drafts / outlines / reviews into `docs/paper/old/` and add `docs/paper/old/REDIRECTS.md`.
   Six registered preregs cite four of these files (Section 4), and preregs must not be edited, so the redirect file is the
   fix. The lowest-risk alternative is to leave them where they are and add a "superseded by draft 2 / draft 6" index at the
   top of `docs/paper/README.md` (new).
3. **One audits index, no audit moves**: add `docs/audits/INDEX.md` with one row per date folder giving the batch(es), the
   prereg / results file that cites it, and whether scripts in it hard-code paths. Do not move any audit folder.
4. **`docs/archive/`** for the unreferenced or historical material: `docs/prepared/` (drop `pkg/`, `mock_runs/`),
   `report/` Sept notes, `paper/`, `paper_rank/`, the Kalman note + 2 docx. That is about 9 MB out of the working tree, with
   about 10 rewrites (Section 4).
5. **Untrack generated files that `.gitignore` already excludes**: the 11 `report/*.aux|bbl|blg|log|out|toc` and the 4
   `paper_figures/cnav_*.npz` (32 MB in the tree; history keeps them).
6. **`papers/`**: ignore the new PDFs, add `new_2026-10-14/SOURCES.txt`, update INDEX.md's count and add the `new_*` rows.
7. **CLAUDE.md "documents" row** (for the owner to edit, not here): it lists only the September .tex PDFs and `report/`. It
   should name draft 2 (`docs/paper/MAIN_DRAFT2.md` + `pdf/main_paper_draft2*.pdf`), draft 6, the claim map and FULL_REVIEW.
   `WHERE_THINGS_STAND.md` ("read second") should be refreshed or should point at BIG_PICTURE_2026-10-24.
8. Deferred: `paper_figures/` (legacy code writes there), `archive_stale/` top-level twins (top-level inventory), SESSION files
   into `docs/sessions/` (5 rewrites, little gain).

---

## 4. Rewrite map (who cites what would move)

Files marked **(prereg)** are registered and must not be edited. Use a redirect file or leave the target in place.

| would move | cited by |
|---|---|
| `docs/paper/MAIN_DRAFT1.md` | **SROPE_TORUS_PREREG.md, TORUS_LRSWEEP_PREREG.md** (prereg); docs/RUNBOOK_NEXT_EXPERIMENTS.md; MAIN_APPENDIX1, MAIN_DRAFT1_notes; expansion/{DEPTH_THEORY,LANGUAGE,PAPER_LEVEL,RANK_THEORY,RECURRENCE,SEARCH_THEORY,SYNTHESIS_DRAFT2_PLAN}.md; MAIN_DRAFT2_notes |
| `docs/paper/MAIN_APPENDIX1.md`, `MAIN_DRAFT1_notes.md` | MAIN_DRAFT1, MAIN_DRAFT1_notes, MAIN_DRAFT2_notes |
| `docs/paper/DRAFT5_workshop.md` | **LRSWEEP_PREREG.md** (prereg); project_state; APPENDIX_draft5, DRAFT5_changes, OUTLINE_overall; review_draft5/{MOCK_REVIEW_draft5,NUMBERS_AUDIT,PRIOR_ART} |
| `docs/paper/READ_draft4.md` | **ROBUST_PREREG.md** (prereg); project_state; DRAFT4_changes |
| `docs/paper/MOCK_REVIEW_draft3.md` | **GAUGE_PREREG.md** (prereg); project_state; docs/audits/2026-10-15/reanalyses/PREDICTIONS.md |
| `docs/paper/DRAFT_workshop.md`, `DRAFT4_workshop.md`, `OUTLINE_overall.md`, `OUTLINE_workshop.md`, `VENUES_2026-10-15.md` | `.claude-memory/project_state.md`; THEORY_REVIEW (2026-10-14); other old drafts; DRAFT6_changes (OUTLINE_overall); BIG_PICTURE (OUTLINE_overall) |
| `docs/paper/OUTLINE_main_v2.md` | MAIN_DRAFT1, MAIN_DRAFT1_notes, expansion/DEPTH_THEORY; `build_pdfs.py` (6_main_paper_outline) |
| `docs/paper/APPENDIX_theory.md`, `APPENDIX_theory_v2.md`, `DRAFT3_*`, `DRAFT4_changes.md`, `DRAFT5_changes.md`, `APPENDIX_draft5.md` | old drafts only |
| `docs/plans/2026-10-11_next_gpu.md` | **RANK_TESTS_PREREG.md** (prereg) |
| `docs/plans/2026-10-14_next_steps.md` | **ROBUST_PREREG.md, TORUS_START_PREREG.md** (prereg); docs/lit/LIT_REPRESENTABILITY_2026-10-14.md; old drafts |
| `docs/SESSION_2026-09-27_to_10-03.md` | CLAUDE.md ("Where things live"); .claude-memory/MEMORY.md, project_what_where_and_language.md; RESULTS_INDEX.md; WHERE_THINGS_STAND.md |
| `docs/SESSION_2026-10-04_to_10-09.md` | .claude-memory/MEMORY.md, project_state.md; RESULTS_INDEX.md; WHERE_THINGS_STAND.md |
| `docs/kalman_mapformer_intuition.md` | RESULTS_INDEX.md (and the catalogue tool's glob) |
| `docs/prepared/**` | nothing outside itself |
| `paper/` | DETAILED_RESULTS.md, docs/LOG.md, docs/WHERE_THINGS_STAND.md, docs/audits/2026-09-24/context/{HISTORY_PLAN,REPORT}.md, paper_rank/{STATUS.md,main.tex} |
| `paper_rank/` | docs/WHERE_THINGS_STAND.md, paper/{RETRACTED.md,main.tex}, report/inventory/E_phase_axes.md, run_dxr.sh |
| `report/{STORY,VERIFY}.md` | docs/audits/2026-09-24/experiments/PROPAGATION.md |
| `report/language_summary.html` (**do not move**) | CLAUDE.md, .claude-memory/{project_state,reference_review_documents,user_style}.md, RESULTS_INDEX.md, docs/LOG.md, WHERE_THINGS_STAND.md, 2026-09-24 audits, tex_edits ed_index.py |
| `docs/paper/expansion/scripts/**` (**do not move**) | runtime: analyze_delta_free.py (kernel_probe.py), analyze_torus_lrsweep.py (clock_table.py); 39 self-referencing absolute paths; MAIN_APPENDIX2; DELTA_FREE/G3_SSM/RANKCLOCK/SIGNROUTE/SIGNSEP/SROPE_TORUS/TORUS_LRSWEEP preregs |
| `docs/audits/**` (**do not move**) | runtime: analyze_mm.py (2026-10-27 JSONs); about 30 audit scripts with absolute paths; 42 preregs, 28 results files, CLAUDE.md (2026-09-24/applied, 2026-10-04/DROPOUT_RESCORE, 2026-10-11/CRITICAL_REVIEW, 2026-10-12/ring_dial_audit) |
| `docs/theory/**` (**do not move**) | analyze_rank_nowrap.py, analyze_score_rank.py, model_em_pope.py, 9 results files |

---

## 5. `.claude-memory/` staleness (list only; nothing edited)

Each item is stale against CLAUDE.md or a later results file. CLAUDE.md wins (reference_shared_memory.md says so).

| file | stale / contradicted item | current source |
|---|---|---|
| `MEMORY.md` | "Live state only: running (nothing, 2026-10-03)"; "report link (v10)"; "40 papers" | project_state (2026-10-28); CLAUDE.md report v22; corpus 42+25 |
| `reference_review_documents.md` (2026-10-03) | live report "v10 on 2026-09-30"; "40 sources"; names the September .tex PDFs as the documents and does not mention draft 2 / draft 6 / claim map; omits the second artifact `navigation_in_words.html` | CLAUDE.md v22; `docs/paper/` |
| `project_state.md` l.287 | "shared report v17 ... is current" | CLAUDE.md: v22 (2026-10-11) |
| `reference_positional_landscape.md` l.38-42 | "rank ... Nobody constrains it"; "r=4 buys +0.085 ... The cause is a skewed basis" | CLAUDE.md withdrawn: "MapFormer ablates r", "r=2 loses because its basis is skewed", "r=4's old +0.085 was OOD only" |
| `feedback_baselines_backfill.md` (2026-05/09-24) | whole file: backfill RoPE on lm200 tables "lm200 checkpoints exist; just eval" | CLAUDE.md withdrawn: the lm200 era; the RoPE/RNN baselines now exist (RNN_TORUS, SROPE_TORUS, GENERALITY, FACTORIAL) |
| `project_hierarchy_negative.md` l.13, l.34 | lm200 numbers (0.996 lm200; "lm200 flat WINS +5pp") with no void banner | CLAUDE.md withdrawn: lm200 |
| `project_paper_direction.md` (2026-10-13) | venue gaps "add torus baselines (LSTM, Mamba-like, CoPE, Selective RoPE), factorial n=8, wrap existence" | all since run: RNN_TORUS, GENERALITY, SROPE_TORUS, FACTORIAL_N8, WRAP_EXISTENCE. The framing is superseded by draft 2 / BIG_PICTURE |
| `project_sign_axis.md` (2026-10-13) | body still states "cannot represent a -1 action" (l.48, l.66; corrected only in the description and at the end); nothing after WRAP_EXISTENCE | CLAUDE.md sign row: MECHANISM, GENERALITY, GAUGE, ROBUST, LRSWEEP, SIGNROUTE, TORUS_START, EXPTORUS, G3_SSM |
| `project_rank_and_selective_rope.md` (2026-10-11) | no G-1 RANKCLOCK (the rank-2 obstacle is the clock), no RANKWD (rank 4 needs decay 0.05), no faithful SROPE_TORUS | CLAUDE.md rank section |
| `project_clock_vs_map.md` (2026-09-24) | description "Explains sign and rank" predates sign = learnability (2026-10-12) and RANKCLOCK; no banner | CLAUDE.md sign row, rank section |
| `project_what_where_and_language.md` (2026-10-04) | no TW_STATECHANGE (and the 10-11 withdrawal of "the model tracks state changes"), GAIN_PHASE, or dropout re-score (0.969 -> 0.993) | CLAUDE.md rows |
| `project_mappope_asymmetry.md` (2026-10-05) | predates the 2026-10-11 CORRECTED blocks on MAPPOPE_PAIR / GAIN_GRAIN ("no headroom") and DELTA_FREE peak pinning | CLAUDE.md MapPoPE rows |
| `project_loop_and_correction.md` | description "Looping substitutes for depth and beats the Kalman correction"; l.146 "matches or beats 3x its parameters" | CLAUDE.md: the loop *matches* 3 real layers; "beats" withdrawn (minor) |

Current and consistent: `feedback_*` (except baselines_backfill), `feedback_compute_kalman_only.md`, `user_style.md`,
`reference_shared_memory.md`, `project_robustness_vs_capability.md`, `project_state.md` (except l.287).

---

## 6. Risks

1. **Registered preregs cite superseded files** (MAIN_DRAFT1, DRAFT5_workshop, READ_draft4, MOCK_REVIEW_draft3, both
   plans). Editing a prereg after registration breaks rule 7, so any move needs a redirect file or must leave the file in place.
2. **Runtime path dependencies.** Registered analyses import or read files under `docs/` (`analyze_delta_free.py`,
   `analyze_torus_lrsweep.py`, `analyze_mm.py`), and about 70 scripts in `docs/audits` and `docs/paper/expansion` hard-code
   `/home/prashr/mapformer/docs/...`. A move silently breaks re-derivation of committed numbers. Run a grep for the old path
   over `*.py *.sh` after any move (rule 18 style).
3. **The PDF builder is stale.** `python3 docs/paper/tools/build_pdfs.py docs/paper/pdf` (the RUNBOOK's instruction)
   rebuilds draft-1 PDFs and leaves the draft-2 PDFs untouched. The draft-2 build recipe is not committed.
4. **The shared artifact source** `report/language_summary.html` must not move or be republished without `url=`; memory
   files disagree about its version (v10 / v17 / v22).
5. **Rule 28**: WHERE_THINGS_STAND, reference_review_documents and the CLAUDE.md "documents" row predate draft 2.
   A fresh session follows "read second" into a 2026-10-03 picture.
6. Deleting from the tree does not shrink the repo. History keeps the PDFs and npz. Only untracking stops future growth
   (each draft-2 rebuild adds about 5.8 MB).
7. `docs/tools/catalog_results_index.py` globs `docs/*.md`, so moving top-level docs notes changes RESULTS_INDEX's catalogue.
   Regenerate and paste after any move.
8. `archive_stale/` vs top-level: removing the wrong twin loses the copy CLAUDE.md cites. Two of the CLAUDE.md-cited
   archive_stale files have no top-level twin (DOG_RESULTS_FIXED, V4_CONTROL differs), so remove only the 20 byte-identical
   top-level copies, and only after checking that no code reads them.
9. Ignoring `papers/new_*` PDFs without a fetch record for `new_2026-10-14/` would make 5 papers unrecoverable from a fresh
   clone (only their txt would remain).
10. `Prashant_Rangarajan_Generals_Document.pdf` is personal. Ask the user before any action.
