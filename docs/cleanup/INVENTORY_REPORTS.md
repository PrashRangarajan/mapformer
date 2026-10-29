# Inventory of top-level reports, results and outputs (2026-10-28)

Read-only inventory for a repo cleanup. Nothing was moved, deleted, edited or committed. Scope: every top-level `*.md` (673), `*.txt` (61) and `*.json` (368) in `/home/prashr/mapformer` -- 1,102 files, all git-tracked. Built by scripts (kept in the session scratchpad, not the repo): a token scan of 2,991 corpus files for each file name (full `NAME.ext` and, for `.md`, the bare stem), first-12-line banners, `git log` last-commit dates, md5 and line-overlap duplicate checks, and the driver / analysis code that writes or reads top-level paths (rule 18).

**How to read the classes.** Applied in this order; a file gets the first that fits.

| class | rule |
|---|---|
| META | `README`, `CLAUDE`, `RESULTS_INDEX`, `GUARDS`, `KNOWN_BUGS`, `HOURGLASS_README`, `requirements.txt` |
| CITED | named in CLAUDE.md between `## Citable results` and `## Withdrawn` (table, rank block, live negatives), in `docs/paper/MAIN_DRAFT2.md`, `MAIN_APPENDIX2.md`, `DRAFT6_workshop.md`, or `docs/paper/CLAIM_MAP_clock_map.md` |
| WITHDRAWN/VOID | a VOID / RETRACTED / PARTIALLY INVALID (lm200) / CONTAINS RETRACTED CLAIMS banner; a registered family whose results file is titled VOID or RETRACTED (RECENCY_T3, CODE); named only under CLAUDE.md "Withdrawn"; or an identical copy already in `archive_stale/` |
| SUPERSEDED | a SUPERSEDED banner, or an older version of a named newer file (`X` vs `X_n8`, gate v1/v2 vs v3, pre-rescore vs RESCORED, pilot vs later file) |
| REGISTERED-LIVE | member of a registered family: stem shares the prefix of a `*_PREREG.md` (prereg, results, analysis, verdicts, gates, per-arm / per-seed outputs, raw dumps). 103 pre-registrations, one family each. A registered record is never split |
| GENERATED-OUTPUT | a `.json`/`.txt`, a `_RAW`/`_STRATA`/`_OWN_s*`/`_GEOMETRY`/`_RESCORED` file, or a table a driver writes to `$REPO/...`, not in a registered family |
| SUPPORTING | (added class: 98 files fit none of the requested ones) not cited, not registered, but referenced as a record by a "strong" source: CLAUDE.md elsewhere, RESULTS_INDEX hand rows, WHERE_THINGS_STAND, memory, another top-level results file, `docs/**`, or a report/review `.tex`/`.html` source |
| ORPHAN | none of the above: referenced only by the RESULTS_INDEX catalogue (which lists every file by construction), `report/inventory/*` / `report/VERIFY*.md` (inventories), `archive*`, or code |

Reference counts below are distinct referencing files. Abbreviations: C-cit / C-wd / C-oth = CLAUDE.md citable region / Withdrawn / elsewhere; RI = RESULTS_INDEX hand rows; RIcat = its catalogue; DR = the three current drafts; CM = claim map; WTS = docs/WHERE_THINGS_STAND.md; mem = .claude-memory; top = other top-level files; docs = docs/** (LOG, audits, lit, sessions, plans); paper = other docs/paper files; tex = report and review .tex/.html sources; repinv = report/*.md inventories and verify notes; code / dcode = .py/.sh at top level, hyak, paper* / under docs; arch = archive*.

## 1. Summary

| class | files | .md | .txt | .json |
|---|---:|---:|---:|---:|
| CITED | 144 | 127 | 12 | 5 |
| REGISTERED-LIVE | 552 | 263 | 48 | 241 |
| SUPPORTING | 98 | 98 | 0 | 0 |
| GENERATED-OUTPUT | 191 | 75 | 0 | 116 |
| WITHDRAWN/VOID | 60 | 54 | 0 | 6 |
| SUPERSEDED | 14 | 14 | 0 | 0 |
| ORPHAN | 36 | 36 | 0 | 0 |
| META | 7 | 6 | 1 | 0 |
| total | 1102 | 673 | 61 | 368 |

Proposed actions:

| action | files |
|---|---:|
| move (keep) | 896 |
| archive | 179 |
| delete (identical copy already archived) | 20 |
| keep at root | 7 |

Registered families: 103. Files with a CORRECTED / AUDIT banner that stay live: 36. Identical copies already in `archive_stale/`: 20 (5 more `archive_stale/` name-matches differ: the top-level copy carries the 2026-08-09 lm200 banner).

Findings worth acting on before any move:

- **Guards read top-level files.** `test_guards.py` / `stats_guard.py` load `_MAGONLY.json`, `_D5_N24.json`, `_REWIND_PROBE.json` (and name `D5_RESULTS.md`, `MAGONLY_RESULTS.md`) relative to `REPO`. Moving them breaks `python3 -m mapformer.test_guards` unless the loader paths change in the same commit.
- **363 code files hard-code `$REPO/<NAME>` paths** (drivers write results there; analysis scripts read them back, e.g. `analyze_factorial_n8.py` reads `FACTORIAL_L1_STRATA.json` and `FACTORIAL_N8_*_OWN_s*.json`). 704 code files mention at least one top-level name literally. See 4.3 (B).
- **`docs/tools/catalog_results_index.py` globs `*.md` at the root**; after a move it would list nothing and exit 0. It is also the source of the RESULTS_INDEX catalogue, which is stale (last regenerated 2026-10-09, 520 `.md` listed; 673 exist now, so 153 are missing from it).
- **`README.md` carries a `SUPERSEDED` banner** ("What is citable now is in RESULTS_INDEX.md") although CLAUDE.md calls README "the primary documentation". Decide which is true before the cleanup.
- **Exact duplicate:** `MATCH_GATES_64_16.json` == `MATCH_QUERY_GATES.json` (md5).
- **Unrun pre-registrations at the root** (live plans, not results): `RANK_NOWRAP_PREREG.md` (postponed by the user), `TW_AMBIG_PREREG.md` (stopped 2026-10-11), `FORGET_CLOCK_PREREG.md`, `ENWIK8_COMPOSITION_PREREG.md`, `CTXSTEP_PREREG.md` (+5 pilots). `TW_LANDMARK_PREREG.md` is stopped but cited (live negatives).

## 2. Every file

Sorted by name. "refs" = distinct referencing files per source (see abbreviations). "last commit" = last commit touching the file. Destination paths follow the layout in section 4.

| file | class | refs | last commit | proposed action | reason |
|---|---|---|---|---|---|
| `_ACHIEVABLE.json` | GENERATED-OUTPUT | repinv code | 2026-09-12 | move with SEARCH_RESULTS.md -> `results/em_wm/SEARCH/_ACHIEVABLE.json` | script output |
| `_ANATOMY.json` | GENERATED-OUTPUT | repinv code | 2026-09-11 | move with SEARCH_RESULTS.md -> `results/em_wm/SEARCH/_ANATOMY.json` | script output |
| `_ANATOMY_EXT.json` | GENERATED-OUTPUT | repinv code2 | 2026-09-11 | move with SEARCH_RESULTS.md -> `results/em_wm/SEARCH/_ANATOMY_EXT.json` | script output |
| `_COLLISIONS.json` | GENERATED-OUTPUT | repinv code | 2026-09-12 | move with SEARCH_RESULTS.md -> `results/em_wm/SEARCH/_COLLISIONS.json` | script output |
| `_D5_N24.json` | REGISTERED-LIVE | top code2 | 2026-09-10 | move to results/em_wm/D5/ | registered D5 |
| `_DELTA_FREE_RAW.json` | REGISTERED-LIVE | top code | 2026-10-20 | move to results/pope/DELTA_FREE/ | registered DELTA_FREE |
| `_DELTA_FREE_RAW.md` | REGISTERED-LIVE | code | 2026-10-20 | move to results/pope/DELTA_FREE/ | registered DELTA_FREE |
| `_DOF_RECENCY.json` | REGISTERED-LIVE | -- | 2026-09-10 | move to results/em_wm/DOF/ | registered DOF |
| `_DOF_TORUS_LOSS.json` | REGISTERED-LIVE | -- | 2026-09-10 | move to results/em_wm/DOF/ | registered DOF |
| `_DOF_TORUS_RAW.json` | REGISTERED-LIVE | -- | 2026-09-10 | move to results/em_wm/DOF/ | registered DOF |
| `_DOF_TORUS_RAW.md` | REGISTERED-LIVE | RIcat tex repinv | 2026-09-10 | move to results/em_wm/DOF/ | registered DOF |
| `_gate_n16.json` | GENERATED-OUTPUT | -- | 2026-09-19 | move with ALIASING_GATES.md -> `results/navigation/environments/_gate_n16.json` | script output |
| `_gate_n256.json` | GENERATED-OUTPUT | -- | 2026-09-19 | move with ALIASING_GATES.md -> `results/navigation/environments/_gate_n256.json` | script output |
| `_gate_n64.json` | GENERATED-OUTPUT | -- | 2026-09-19 | move with ALIASING_GATES.md -> `results/navigation/environments/_gate_n64.json` | script output |
| `_INIT_GRAD.json` | GENERATED-OUTPUT | repinv code | 2026-09-11 | archive to archive/generated/em_wm/ (regenerable: probe_init_grad.py) | script output |
| `_L15_LOOP_RAW.json` | GENERATED-OUTPUT | code2 | 2026-09-01 | move with L15_LOOP_2X2.md -> `results/recurrence/_L15_LOOP_RAW.json` | script output (run_l15_loop_2x2.sh, run_l15_loop_supp.sh) |
| `_L15_LOOP_RAW.md` | GENERATED-OUTPUT | RIcat repinv code2 | 2026-09-01 | move with L15_LOOP_2X2.md -> `results/recurrence/_L15_LOOP_RAW.md` | script output (run_l15_loop_2x2.sh, run_l15_loop_supp.sh) |
| `_MAGONLY.json` | REGISTERED-LIVE | repinv code2 | 2026-09-10 | move to results/em_wm/MAGONLY/ | registered MAGONLY |
| `_MONOTONE_TORUS.json` | REGISTERED-LIVE | code2 | 2026-09-13 | move to results/sign/MONOTONE/ | registered MONOTONE |
| `_MONOTONE_TORUS.md` | REGISTERED-LIVE | RIcat code | 2026-09-13 | move to results/sign/MONOTONE/ | registered MONOTONE |
| `_N5_RECENCY.json` | REGISTERED-LIVE | -- | 2026-09-10 | move to results/em_wm/N5/ | registered N5 |
| `_N5_TORUS_RAW.json` | REGISTERED-LIVE | -- | 2026-09-10 | move to results/em_wm/N5/ | registered N5 |
| `_N5_TORUS_RAW.md` | REGISTERED-LIVE | RIcat repinv | 2026-09-10 | move to results/em_wm/N5/ | registered N5 |
| `_NOLEAK.json` | REGISTERED-LIVE | repinv code | 2026-09-11 | move to results/em_wm/NOLEAK/ | registered NOLEAK |
| `_PAIRORIGIN.json` | REGISTERED-LIVE | repinv code | 2026-09-12 | move to results/em_wm/PAIRORIGIN/ | registered PAIRORIGIN |
| `_PAPER2X2_RAW.json` | REGISTERED-LIVE | top docs paper tex repinv code2 dcode4 | 2026-09-13 | move to results/navigation/torus/PAPER2X2/ | registered PAPER2X2 |
| `_PAPER2X2_RAW.md` | REGISTERED-LIVE | RIcat code | 2026-09-13 | move to results/navigation/torus/PAPER2X2/ | registered PAPER2X2 |
| `_PHASE_SPREAD.json` | GENERATED-OUTPUT | repinv code | 2026-09-11 | move with SPREAD_RESULTS.md -> `results/em_wm/SPREAD/_PHASE_SPREAD.json` | script output |
| `_RECIPE_C0.json` | GENERATED-OUTPUT | -- | 2026-09-02 | move with RECIPE_POWER.md -> `results/navigation/torus/_RECIPE_C0.json` | script output |
| `_RECIPE_C0.md` | GENERATED-OUTPUT | RIcat repinv | 2026-09-02 | move with RECIPE_POWER.md -> `results/navigation/torus/_RECIPE_C0.md` | script output (run_recipe_power.sh) |
| `_RECIPE_C1.json` | GENERATED-OUTPUT | -- | 2026-09-02 | move with RECIPE_POWER.md -> `results/navigation/torus/_RECIPE_C1.json` | script output |
| `_RECIPE_C1.md` | GENERATED-OUTPUT | RIcat repinv | 2026-09-02 | move with RECIPE_POWER.md -> `results/navigation/torus/_RECIPE_C1.md` | script output (run_recipe_power.sh) |
| `_RECIPE_C2.json` | GENERATED-OUTPUT | -- | 2026-09-02 | move with RECIPE_POWER.md -> `results/navigation/torus/_RECIPE_C2.json` | script output |
| `_RECIPE_C2.md` | GENERATED-OUTPUT | RIcat repinv | 2026-09-02 | move with RECIPE_POWER.md -> `results/navigation/torus/_RECIPE_C2.md` | script output (run_recipe_power.sh) |
| `_REWIND_PROBE.json` | GENERATED-OUTPUT | top5 repinv code | 2026-09-10 | move to results/em_wm/ | script output |
| `_SEARCH_S3.json` | REGISTERED-LIVE | code | 2026-09-11 | move to results/em_wm/SEARCH/ | registered SEARCH |
| `_SELECTIVE_TORUS.json` | GENERATED-OUTPUT | code | 2026-09-03 | move with SELECTIVE_ROPE.md -> `results/pope/_SELECTIVE_TORUS.json` | script output |
| `_SELECTIVE_TORUS.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-09-03 | move with SELECTIVE_ROPE.md -> `results/pope/_SELECTIVE_TORUS.md` | script output (run_selective.sh) |
| `_SIGN_RAW.json` | REGISTERED-LIVE | repinv code2 | 2026-09-19 | move to results/sign/SIGN_ABLATION/ | registered SIGN_ABLATION |
| `_SIGN_RAW.md` | REGISTERED-LIVE | RIcat repinv code | 2026-09-19 | move to results/sign/SIGN_ABLATION/ | registered SIGN_ABLATION |
| `_SPREAD.json` | REGISTERED-LIVE | code | 2026-09-11 | move to results/em_wm/SPREAD/ | registered SPREAD |
| `_SPREAD2.json` | REGISTERED-LIVE | code | 2026-09-12 | move to results/em_wm/SPREAD2/ | registered SPREAD2 |
| `_UNFREEZE.json` | REGISTERED-LIVE | -- | 2026-09-11 | move to results/em_wm/UNFREEZE/ | registered UNFREEZE |
| `_WARM.json` | REGISTERED-LIVE | -- | 2026-09-10 | move to results/em_wm/WARM/ | registered WARM |
| `ABLATE_COMPOSITIONAL.json` | REGISTERED-LIVE | -- | 2026-08-17 | move to results/pope/ABLATE/ | registered ABLATE |
| `ABLATE_COMPOSITIONAL.md` | SUPPORTING | RI RIcat tex repinv2 code arch | 2026-08-17 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `ABLATE_FAMILY_TREE.json` | REGISTERED-LIVE | -- | 2026-08-17 | move to results/pope/ABLATE/ | registered ABLATE |
| `ABLATE_FAMILY_TREE.md` | SUPPORTING | RIcat tex repinv3 code arch | 2026-08-17 | move to results/algorithmic/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `ABLATE_PREREG.md` | REGISTERED-LIVE | RIcat top code | 2026-09-22 | move to results/pope/ABLATE/ | registered ABLATE |
| `ABLATE_RESULTS.md` | CITED | C-cit C-wd RI RIcat top2 docs7 paper | 2026-09-22 | move to results/pope/ABLATE/ | cited; registered ABLATE; banner: CORRECTED/AUDIT block (live); also named in CLAUDE.md Withdrawn (carries the correction) |
| `ACCOUNTS_ANALYSIS.txt` | REGISTERED-LIVE | top paper4 code | 2026-10-15 | move to results/sign/ACCOUNTS/ | registered ACCOUNTS |
| `ACCOUNTS_PREREG.md` | REGISTERED-LIVE | top docs2 paper code2 | 2026-10-14 | move to results/sign/ACCOUNTS/ | registered ACCOUNTS |
| `ACCOUNTS_RESULTS.md` | CITED | C-cit DR CM mem top paper12 dcode | 2026-10-15 | move to results/sign/ACCOUNTS/ | cited; registered ACCOUNTS |
| `ACCOUNTS_VERDICTS.json` | REGISTERED-LIVE | top paper code2 | 2026-10-15 | move to results/sign/ACCOUNTS/ | registered ACCOUNTS |
| `ACCUMULATOR.json` | GENERATED-OUTPUT | -- | 2026-09-06 | archive with ACCUMULATOR.md -> `archive/withdrawn/rank/ACCUMULATOR.json` | script output |
| `ACCUMULATOR.md` | WITHDRAWN/VOID | C-wd RIcat mem top4 docs4 tex repinv2 code | 2026-09-06 | archive to archive/withdrawn/rank/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `ACTION_GEOMETRY.md` | GENERATED-OUTPUT | RIcat mem top7 docs3 tex repinv3 code3 oth2 | 2026-09-25 | move to results/rank/ | script output; banner: CORRECTED/AUDIT block (live) |
| `ADDITION_CHO_REPRO.md` | SUPPORTING | RIcat top | 2026-09-14 | move to results/algorithmic/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `ADDITION_DESIGN.md` | SUPPORTING | RIcat top2 code5 dcode | 2026-09-14 | move to results/algorithmic/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `ADDITION_GATES.json` | GENERATED-OUTPUT | code | 2026-09-14 | move with ADDITION_GATES.md -> `results/algorithmic/ADDITION_GATES.json` | script output (validate_addition.py) |
| `ADDITION_GATES.md` | SUPPORTING | RIcat top | 2026-09-14 | move to results/algorithmic/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `ADDITION_PILOT.md` | SUPERSEDED | RIcat top2 | 2026-09-14 | archive to archive/superseded/algorithmic/ | superseded; superseded by ADDITION_PILOT2.md |
| `ADDITION_PILOT2.md` | SUPPORTING | RIcat top2 code | 2026-09-14 | move to results/algorithmic/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `AGGREGATE_EXTRAS.md` | SUPPORTING | RIcat mem repinv arch | 2026-07-18 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `AGGREGATE_MULTISEED.md` | ORPHAN | RIcat repinv arch | 2026-07-18 | archive to archive/orphan/hierarchy | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `AGGREGATE_TASK_RESULTS.md` | ORPHAN | RIcat repinv arch | 2026-07-17 | archive to archive/orphan/hierarchy | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `ALGORITHMIC_GATES.md` | SUPPORTING | RIcat top tex repinv2 code3 | 2026-09-03 | move to results/algorithmic/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `ALGORITHMIC_RESULTS.md` | SUPPORTING | RIcat mem top4 docs2 tex2 repinv5 code3 | 2026-09-03 | move to results/algorithmic/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `ALIASING_CONTROLLED.md` | CITED | C-cit C-wd RI RIcat WTS mem top3 docs7 paper tex repinv4 code3 | 2026-08-30 | move to results/navigation/environments/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; also named in CLAUDE.md Withdrawn (carries the correction) |
| `ALIASING_COVARIATE.json` | GENERATED-OUTPUT | -- | 2026-08-19 | move with ALIASING_COVARIATE.md -> `results/navigation/environments/ALIASING_COVARIATE.json` | script output |
| `ALIASING_COVARIATE.md` | SUPPORTING | RIcat top repinv2 code | 2026-08-19 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `ALIASING_GATES.md` | GENERATED-OUTPUT | RIcat tex repinv2 code2 | 2026-08-29 | move to results/navigation/environments/ | script output (run_alias_gates.sh) |
| `ALLOCENTRIC.json` | GENERATED-OUTPUT | -- | 2026-08-22 | archive to archive/generated/navigation/torus/ | script output |
| `ALLOCENTRIC_RECODING.md` | CITED | C-cit RI RIcat top3 docs2 tex repinv3 code2 dcode2 | 2026-09-07 | move to results/navigation/torus/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `AP_COHERENCE.json` | GENERATED-OUTPUT | -- | 2026-09-09 | archive to archive/generated/em_wm/ | script output |
| `AP_COHERENCE_MINIGRID.json` | GENERATED-OUTPUT | -- | 2026-09-09 | archive to archive/generated/em_wm/ | script output |
| `AP_KERNEL_DIAGNOSTIC.json` | GENERATED-OUTPUT | -- | 2026-08-09 | move with AP_KERNEL_DIAGNOSTIC.md -> `results/em_wm/AP_KERNEL_DIAGNOSTIC.json` | script output |
| `AP_KERNEL_DIAGNOSTIC.md` | SUPPORTING | RIcat top7 tex repinv3 code arch | 2026-08-09 | move to results/em_wm/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `AUDIT_2026-09-10.md` | SUPPORTING | C-wd C-oth RIcat mem top17 docs5 tex repinv6 code5 | 2026-09-11 | move to results/em_wm/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources); named in CLAUDE.md Where-things-live AND Withdrawn: the audit that records withdrawals |
| `AUDIT_HEADLINE.md` | SUPPORTING | RIcat docs tex repinv code2 | 2026-08-23 | archive to archive/reports_old/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `AUG_PREREG.md` | REGISTERED-LIVE | RIcat top | 2026-09-20 | move to results/pope/AUG/ | registered AUG |
| `AUG_RESULTS.md` | CITED | C-cit RI RIcat top2 docs4 tex code | 2026-09-20 | move to results/pope/AUG/ | cited; registered AUG |
| `BASELINE_TABLE.md` | CITED | C-cit RIcat top4 docs4 tex2 repinv7 code3 | 2026-09-25 | move to results/navigation/torus/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; banner: CORRECTED/AUDIT block (live) |
| `BF16_CHECK.json` | GENERATED-OUTPUT | top code | 2026-09-23 | move to results/code_enwik8/ | script output |
| `BF16_RESULTS.md` | CITED | C-cit RIcat top docs2 | 2026-09-23 | move to results/code_enwik8/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `BOUNDED_MEMORY.json` | GENERATED-OUTPUT | code | 2026-08-09 | move with BOUNDED_MEMORY_RESULTS.md -> `results/hierarchy/BOUNDED_MEMORY.json` | script output |
| `BOUNDED_MEMORY.md` | GENERATED-OUTPUT | RIcat repinv code2 arch | 2026-08-09 | move with BOUNDED_MEMORY_RESULTS.md -> `results/hierarchy/BOUNDED_MEMORY.md` | script output (run_bounded_mem.sh) |
| `BOUNDED_MEMORY_RESULTS.md` | SUPPORTING | RIcat mem repinv arch | 2026-07-20 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `BUMP_TOKEN_RESULTS.md` | SUPPORTING | RIcat docs repinv arch | 2026-07-22 | archive to archive/legacy_level15/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `CANCEL_ANALYSIS.txt` | REGISTERED-LIVE | top paper code | 2026-09-28 | move to results/dyck/CANCEL/ | registered CANCEL |
| `CANCEL_PREREG.md` | CITED | RI RIcat DR top2 docs paper4 code4 dcode2 | 2026-09-30 | move to results/dyck/CANCEL/ | cited; registered CANCEL |
| `CANCEL_RESULTS.md` | CITED | C-cit RI RIcat DR CM WTS mem top2 docs5 paper11 code2 | 2026-09-30 | move to results/dyck/CANCEL/ | cited; registered CANCEL; banner: CORRECTED/AUDIT block (live) |
| `CAPACITY_CONTROL.md` | SUPERSEDED | RIcat top2 repinv code3 arch | 2026-05-22 | archive to archive/legacy_level15/ | superseded; banner: SUPERSEDED |
| `CAPACITY_PERREGIME.md` | WITHDRAWN/VOID | RIcat top2 repinv2 code arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows) |
| `CASCADE_MULTISEED_RESULTS.md` | ORPHAN | RIcat repinv arch | 2026-07-15 | archive to archive/legacy_level15/ | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `CASCADE_REPRO_TEST.md` | ORPHAN | RIcat repinv arch | 2026-07-15 | archive to archive/legacy_level15/ | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `CASCADE_ZEROSHOT_S0.md` | ORPHAN | RIcat repinv arch | 2026-07-15 | archive to archive/legacy_level15/ | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `CLAUDE.md` | META | C-oth RI RIcat CM WTS mem13 top50 docs34 paper15 tex repinv9 code16 dcode2 arch33 oth | 2026-10-28 | keep at root | project infrastructure |
| `CLOCK_SCAN.json` | GENERATED-OUTPUT | code | 2026-08-01 | move with CLOCK_SCAN.md -> `results/sign/CLOCK_SCAN.json` | script output |
| `CLOCK_SCAN.md` | GENERATED-OUTPUT | RIcat top repinv2 code4 arch2 | 2026-08-09 | move to results/sign/ | script output (run_bestofboth.sh, run_clock_multiseed.sh) |
| `CLOCK_SIDE_ANALYSIS.txt` | REGISTERED-LIVE | top docs code | 2026-10-12 | move to results/sign/CLOCK_SIDE/ | registered CLOCK_SIDE |
| `CLOCK_SIDE_EVAL.json` | REGISTERED-LIVE | top code3 | 2026-10-12 | move to results/sign/CLOCK_SIDE/ | registered CLOCK_SIDE |
| `CLOCK_SIDE_GATES.json` | REGISTERED-LIVE | docs | 2026-10-14 | move to results/sign/CLOCK_SIDE/ | registered CLOCK_SIDE; older version of CLOCK_SIDE_GATES_8000.json |
| `CLOCK_SIDE_GATES.md` | REGISTERED-LIVE | top docs | 2026-10-12 | move to results/sign/CLOCK_SIDE/ | registered CLOCK_SIDE; older version of CLOCK_SIDE_GATES_8000.md |
| `CLOCK_SIDE_GATES_8000.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/CLOCK_SIDE/ | registered CLOCK_SIDE |
| `CLOCK_SIDE_GATES_8000.md` | REGISTERED-LIVE | top docs | 2026-10-12 | move to results/sign/CLOCK_SIDE/ | registered CLOCK_SIDE |
| `CLOCK_SIDE_PREREG.md` | REGISTERED-LIVE | top docs code3 dcode | 2026-10-12 | move to results/sign/CLOCK_SIDE/ | registered CLOCK_SIDE |
| `CLOCK_SIDE_RESULTS.md` | CITED | CM docs paper3 | 2026-10-12 | move to results/sign/CLOCK_SIDE/ | cited; registered CLOCK_SIDE |
| `CLOCK_SIDE_VERDICTS.json` | REGISTERED-LIVE | top docs code2 | 2026-10-12 | move to results/sign/CLOCK_SIDE/ | registered CLOCK_SIDE |
| `CLOCK_SPEED_ANALYSIS.txt` | REGISTERED-LIVE | top docs code | 2026-10-12 | move to results/sign/CLOCK_SPEED/ | registered CLOCK_SPEED |
| `CLOCK_SPEED_PREREG.md` | REGISTERED-LIVE | top docs code2 | 2026-10-12 | move to results/sign/CLOCK_SPEED/ | registered CLOCK_SPEED |
| `CLOCK_SPEED_REP_ANALYSIS.txt` | REGISTERED-LIVE | top docs code | 2026-10-13 | move to results/sign/CLOCK_SPEED_REP/ | registered CLOCK_SPEED_REP |
| `CLOCK_SPEED_REP_PREREG.md` | REGISTERED-LIVE | top docs code2 | 2026-10-13 | move to results/sign/CLOCK_SPEED_REP/ | registered CLOCK_SPEED_REP |
| `CLOCK_SPEED_REP_RESULTS.md` | CITED | CM docs paper4 | 2026-10-13 | move to results/sign/CLOCK_SPEED_REP/ | cited; registered CLOCK_SPEED_REP |
| `CLOCK_SPEED_REP_VERDICTS.json` | REGISTERED-LIVE | top docs code2 | 2026-10-13 | move to results/sign/CLOCK_SPEED_REP/ | registered CLOCK_SPEED_REP |
| `CLOCK_SPEED_RESULTS.md` | CITED | CM top docs paper4 | 2026-10-12 | move to results/sign/CLOCK_SPEED/ | cited; registered CLOCK_SPEED |
| `CLOCK_SPEED_VERDICTS.json` | REGISTERED-LIVE | top docs2 code3 | 2026-10-12 | move to results/sign/CLOCK_SPEED/ | registered CLOCK_SPEED |
| `CLONE_ANALYSIS_LEVEL15PC.md` | WITHDRAWN/VOID | RIcat docs repinv code arch | 2026-04-26 | delete (identical copy already in archive_stale/CLONE_ANALYSIS_LEVEL15PC.md) | withdrawn or void; copy in archive_stale/CLONE_ANALYSIS_LEVEL15PC.md (identical) |
| `CLONE_TRANSFER_NOBYPASS.md` | WITHDRAWN/VOID | RIcat docs repinv code arch | 2026-04-27 | delete (identical copy already in archive_stale/CLONE_TRANSFER_NOBYPASS.md) | withdrawn or void; copy in archive_stale/CLONE_TRANSFER_NOBYPASS.md (identical) |
| `CNAV_HEX_Level15.md` | WITHDRAWN/VOID | RIcat repinv code2 arch | 2026-05-03 | delete (identical copy already in archive_stale/CNAV_HEX_Level15.md) | withdrawn or void; copy in archive_stale/CNAV_HEX_Level15.md (identical) |
| `CNAV_HEX_Level15EM.md` | WITHDRAWN/VOID | RIcat repinv code arch | 2026-05-03 | delete (identical copy already in archive_stale/CNAV_HEX_Level15EM.md) | withdrawn or void; copy in archive_stale/CNAV_HEX_Level15EM.md (identical) |
| `CNAV_HEX_Vanilla.md` | WITHDRAWN/VOID | RIcat repinv code2 arch | 2026-05-03 | delete (identical copy already in archive_stale/CNAV_HEX_Vanilla.md) | withdrawn or void; copy in archive_stale/CNAV_HEX_Vanilla.md (identical) |
| `CNAV_HEX_VanillaEM.md` | WITHDRAWN/VOID | RIcat repinv code arch | 2026-05-03 | delete (identical copy already in archive_stale/CNAV_HEX_VanillaEM.md) | withdrawn or void; copy in archive_stale/CNAV_HEX_VanillaEM.md (identical) |
| `CNAV_RESULTS.md` | WITHDRAWN/VOID | RIcat top repinv code3 arch | 2026-05-03 | delete (identical copy already in archive_stale/CNAV_RESULTS.md) | withdrawn or void; copy in archive_stale/CNAV_RESULTS.md (identical) |
| `coarse_angles.json` | GENERATED-OUTPUT | code arch | 2026-08-02 | archive to archive/generated/rank/ (regenerable: probe_coarse_angles.py) | script output |
| `CODE_ABLATE_INDIST.json` | WITHDRAWN/VOID | -- | 2026-09-22 | archive to archive/withdrawn/CODE/ | withdrawn or void; registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_ABLATE_LONG.json` | WITHDRAWN/VOID | -- | 2026-09-22 | archive to archive/withdrawn/CODE/ | withdrawn or void; registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_ABLATE_LONG_FINAL.json` | WITHDRAWN/VOID | -- | 2026-09-22 | archive to archive/withdrawn/CODE/ | withdrawn or void; registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_DECAY_RESULTS.md` | WITHDRAWN/VOID | C-wd RIcat mem top3 docs4 | 2026-09-28 | archive to archive/withdrawn/CODE/ | withdrawn or void; banner: CORRECTED/AUDIT block (live); registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_DELIMS.json` | WITHDRAWN/VOID | code2 | 2026-09-21 | archive to archive/withdrawn/CODE/ | withdrawn or void; registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_FULLVAL_ANALYSIS.txt` | REGISTERED-LIVE | top code | 2026-09-27 | move to results/code_enwik8/CODE_FULLVAL/ | registered CODE_FULLVAL |
| `CODE_FULLVAL_PREREG.md` | REGISTERED-LIVE | RI RIcat top code2 | 2026-09-27 | move to results/code_enwik8/CODE_FULLVAL/ | registered CODE_FULLVAL |
| `CODE_FULLVAL_RESULTS.md` | CITED | C-cit RI RIcat CM mem3 top2 docs3 paper4 tex | 2026-09-27 | move to results/code_enwik8/CODE_FULLVAL/ | cited; registered CODE_FULLVAL |
| `CODE_GATES.json` | WITHDRAWN/VOID | code3 | 2026-09-21 | archive to archive/withdrawn/CODE/ | withdrawn or void; registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_GATES.md` | WITHDRAWN/VOID | RI RIcat top2 docs2 code3 | 2026-09-21 | archive to archive/withdrawn/CODE/ | withdrawn or void; registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_LONG.json` | WITHDRAWN/VOID | code2 | 2026-09-21 | archive to archive/withdrawn/CODE/ | withdrawn or void; registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_PREREG.md` | WITHDRAWN/VOID | RIcat top3 docs code5 | 2026-09-22 | archive to archive/withdrawn/CODE/ | withdrawn or void; registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_RESULTS.md` | WITHDRAWN/VOID | C-wd RI RIcat mem top3 docs4 tex code dcode | 2026-09-28 | archive to archive/withdrawn/CODE/ | withdrawn or void; banner: CORRECTED/AUDIT block (live); registered family whose results file is withdrawn/void (CODE_DECAY_RESULTS.md, CODE_RESULTS.md) |
| `CODE_RESULTS_OOD.md` | SUPPORTING | RIcat top2 docs2 code2 | 2026-09-21 | move to results/code_enwik8/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `COMP_HEADROOM.json` | REGISTERED-LIVE | repinv | 2026-09-08 | move to results/hierarchy/COMP_HEADROOM/ | registered COMP_HEADROOM |
| `COMP_HEADROOM.md` | CITED | C-cit RI RIcat mem top4 docs4 tex repinv4 code | 2026-09-08 | move to results/hierarchy/COMP_HEADROOM/ | cited; registered COMP_HEADROOM |
| `COMP_HEADROOM_PREREG.md` | REGISTERED-LIVE | RIcat top repinv code2 | 2026-09-08 | move to results/hierarchy/COMP_HEADROOM/ | registered COMP_HEADROOM |
| `COMPOSITIONAL_EXPERIMENT.md` | CITED | C-cit RIcat top2 docs tex repinv3 arch | 2026-07-26 | move to results/hierarchy/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `COMPOSITIONAL_MATCH_QUERY.md` | SUPPORTING | RIcat tex repinv | 2026-08-24 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `COMPOSITIONAL_MATCH_QUERY_CURRIC.md` | GENERATED-OUTPUT | RIcat top repinv code3 | 2026-08-24 | move to results/hierarchy/ | script output (run_cmq_curric.sh, run_cmq_curric_full.sh) |
| `COMPOSITIONAL_MATCH_QUERY_GATES.json` | GENERATED-OUTPUT | -- | 2026-08-23 | archive with COMPOSITIONAL_MATCH_QUERY_GATES.md -> `archive/orphan/hierarchy/COMPOSITIONAL_MATCH_QUERY_GATES.json` | script output |
| `COMPOSITIONAL_MATCH_QUERY_GATES.md` | ORPHAN | RIcat repinv code | 2026-08-23 | archive to archive/orphan/hierarchy (regenerable: validate_compositional_match_query.py) | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `COMPOSITIONAL_MATCH_QUERY_RESULTS.md` | ORPHAN | RIcat repinv code3 | 2026-08-23 | archive to archive/orphan/hierarchy (regenerable: run_cmq_sweep.sh, run_cmq_topup.sh) | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `COMPOSITIONAL_MATCH_QUERY_STAB.md` | GENERATED-OUTPUT | RIcat top repinv code2 | 2026-08-23 | move to results/hierarchy/ | script output (run_cmq_stab.sh, run_cmq_stab_topup.sh) |
| `COMPOSITIONAL_MULTISEED.json` | GENERATED-OUTPUT | code2 | 2026-08-01 | move with COMPOSITIONAL_MULTISEED.md -> `results/hierarchy/COMPOSITIONAL_MULTISEED.json` | script output |
| `COMPOSITIONAL_MULTISEED.md` | GENERATED-OUTPUT | RIcat top7 docs tex2 repinv2 code13 arch | 2026-08-01 | move to results/hierarchy/ | script output (agg_comp_multiseed.py, autocommit_bob.sh) |
| `COMPOSITIONAL_PLAIN_RESULTS.md` | ORPHAN | RIcat repinv | 2026-09-19 | archive to archive/orphan/hierarchy | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `COMPOSITIONAL_RESULTS.md` | SUPERSEDED | RIcat top docs repinv code2 arch | 2026-07-25 | archive to archive/superseded/hierarchy/ | superseded; banner: SUPERSEDED |
| `CONTEXT_STEP_DESIGN.md` | SUPPORTING | RI RIcat top3 docs2 code7 dcode | 2026-09-30 | move to results/language/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources); banner: CORRECTED/AUDIT block (live) |
| `CONTINUOUS_ALLOC.json` | GENERATED-OUTPUT | code | 2026-08-23 | move with CONTINUOUS_ALLOC.md -> `results/navigation/environments/CONTINUOUS_ALLOC.json` | script output |
| `CONTINUOUS_ALLOC.md` | SUPPORTING | RIcat top tex repinv4 code3 | 2026-08-23 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `CONV_KERNEL_PROBE.md` | ORPHAN | RIcat repinv code2 | 2026-09-04 | archive to archive/orphan/rank (regenerable: probe_conv_kernel.py) | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `CORRECTED_LM200_LEADERBOARD.md` | WITHDRAWN/VOID | C-wd RIcat top16 docs repinv2 arch35 oth | 2026-07-15 | archive to archive/legacy_level15/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `CORRECTION_COMPOSITIONAL.md` | SUPPORTING | RIcat tex repinv3 code | 2026-08-21 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `COUNTER_BATCH.md` | SUPPORTING | RIcat top code2 dcode | 2026-09-14 | move to results/sign/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `COUNTER_RESULTS.md` | SUPPORTING | RI RIcat mem | 2026-09-14 | move to results/sign/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `CROSS_PREREG.md` | REGISTERED-LIVE | RIcat top | 2026-09-20 | move to results/pope/CROSS/ | registered CROSS |
| `CROSS_RESULTS.md` | CITED | C-cit RI RIcat CM mem2 docs5 tex code | 2026-09-20 | move to results/pope/CROSS/ | cited; registered CROSS; banner: CORRECTED/AUDIT block (live) |
| `CROSSOVER_CONVERGED.md` | GENERATED-OUTPUT | RIcat docs2 tex repinv2 code | 2026-08-28 | move to results/navigation/environments/ | script output (run_g8_converge.sh) |
| `CSCG_TASK_GATES.json` | GENERATED-OUTPUT | -- | 2026-08-16 | move with CSCG_TASK_GATES.md -> `results/hierarchy/CSCG_TASK_GATES.json` | script output |
| `CSCG_TASK_GATES.md` | SUPPORTING | RIcat tex repinv3 code | 2026-08-16 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `CTXSTEP_HS_RECIPE.md` | REGISTERED-LIVE | RI RIcat WTS top docs4 code dcode | 2026-09-30 | move to results/language/CTXSTEP/ | registered CTXSTEP; registration never run; pilots only |
| `CTXSTEP_HSR_PILOT.md` | REGISTERED-LIVE | RI RIcat WTS mem top2 docs3 | 2026-09-30 | move to results/language/CTXSTEP/ | registered CTXSTEP; registration never run; pilots only |
| `CTXSTEP_PILOT1.md` | REGISTERED-LIVE | RI RIcat | 2026-09-30 | move to results/language/CTXSTEP/ | registered CTXSTEP; registration never run; pilots only; older version of CTXSTEP_PREREG.md |
| `CTXSTEP_PILOT2.md` | REGISTERED-LIVE | RI RIcat top | 2026-09-30 | move to results/language/CTXSTEP/ | registered CTXSTEP; registration never run; pilots only; older version of CTXSTEP_PREREG.md |
| `CTXSTEP_PILOT3.md` | REGISTERED-LIVE | RI RIcat top docs | 2026-09-30 | move to results/language/CTXSTEP/ | registered CTXSTEP; registration never run; pilots only; older version of CTXSTEP_PREREG.md |
| `CTXSTEP_PREREG.md` | REGISTERED-LIVE | RI RIcat WTS mem top docs2 code2 | 2026-09-30 | move to results/language/CTXSTEP/ | registered CTXSTEP; registration never run; pilots only |
| `D5_PREREG.md` | REGISTERED-LIVE | RIcat top2 repinv code | 2026-09-10 | move to results/em_wm/D5/ | registered D5 |
| `D5_RESULTS.md` | REGISTERED-LIVE | C-wd RI RIcat top4 docs2 tex repinv3 code2 | 2026-09-10 | move to results/em_wm/D5/ | registered D5; banner: CORRECTED/AUDIT block (live); named in CLAUDE.md Withdrawn: a claim from it is withdrawn, the record stays |
| `DAGGER_DK6_RESULTS.md` | SUPPORTING | RIcat top repinv code arch | 2026-08-09 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DAGGER_EMPTY_RESULTS.md` | SUPPORTING | RIcat top repinv code arch | 2026-08-09 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DAGGER_RESULTS.md` | SUPPORTING | RIcat mem top3 docs repinv code arch | 2026-08-09 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DECAY_PREREG.md` | REGISTERED-LIVE | RIcat top code | 2026-09-19 | move to results/pope/DECAY/ | registered DECAY |
| `DECAY_RESULTS.md` | REGISTERED-LIVE | RI RIcat mem top2 docs3 tex | 2026-09-20 | move to results/pope/DECAY/ | registered DECAY |
| `DELTA_FREE_ANALYSIS.txt` | REGISTERED-LIVE | top2 code | 2026-10-20 | move to results/pope/DELTA_FREE/ | registered DELTA_FREE |
| `DELTA_FREE_PREREG.md` | REGISTERED-LIVE | top2 docs2 code4 | 2026-10-20 | move to results/pope/DELTA_FREE/ | registered DELTA_FREE |
| `DELTA_FREE_RESULTS.md` | CITED | C-cit DR CM mem docs2 paper | 2026-10-20 | move to results/pope/DELTA_FREE/ | cited; registered DELTA_FREE |
| `DELTA_FREE_VERDICTS.json` | REGISTERED-LIVE | top2 code2 | 2026-10-20 | move to results/pope/DELTA_FREE/ | registered DELTA_FREE |
| `DETAILED_RESULTS.md` | SUPPORTING | RIcat top docs repinv arch | 2026-05-13 | archive to archive/reports_old/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DISSOCIATION_SWEEP.json` | GENERATED-OUTPUT | -- | 2026-08-19 | move with DISSOCIATION_SWEEP.md -> `results/hierarchy/DISSOCIATION_SWEEP.json` | script output |
| `DISSOCIATION_SWEEP.md` | SUPPORTING | RIcat tex repinv3 code3 | 2026-08-19 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DOF_PREREG.md` | REGISTERED-LIVE | RIcat top3 repinv code3 dcode | 2026-09-10 | move to results/em_wm/DOF/ | registered DOF |
| `DOF_RESULTS.md` | REGISTERED-LIVE | RIcat top5 docs2 tex repinv2 | 2026-09-10 | move to results/em_wm/DOF/ | registered DOF; banner: CORRECTED/AUDIT block (live) |
| `DOG_RESULTS.md` | WITHDRAWN/VOID | RIcat top3 docs3 repinv code3 arch | 2026-04-30 | delete (identical copy already in archive_stale/DOG_RESULTS.md) | withdrawn or void; copy in archive_stale/DOG_RESULTS.md (identical) |
| `DOORKEY_BC_RESULTS.md` | SUPPORTING | RIcat mem top4 docs repinv code arch | 2026-08-09 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DRIFT_PROBE.md` | SUPPORTING | RIcat tex repinv2 code | 2026-09-04 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DXR_PRELIM.json` | GENERATED-OUTPUT | -- | 2026-09-05 | archive with DXR_PRELIM.md -> `archive/superseded/rank/DXR_PRELIM.json` | script output |
| `DXR_PRELIM.md` | SUPERSEDED | RIcat repinv code | 2026-09-04 | archive to archive/superseded/rank/ | superseded; superseded by DXR_RANK_THRESHOLD.md |
| `DXR_RANK_THRESHOLD.json` | GENERATED-OUTPUT | -- | 2026-09-05 | archive with DXR_RANK_THRESHOLD.md -> `archive/withdrawn/rank/DXR_RANK_THRESHOLD.json` | script output |
| `DXR_RANK_THRESHOLD.md` | WITHDRAWN/VOID | C-wd RIcat top docs paper tex repinv4 code | 2026-09-05 | archive to archive/withdrawn/rank/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `DYCK_DECAY_PREREG.md` | REGISTERED-LIVE | RIcat top | 2026-09-19 | move to results/dyck/DYCK_DECAY/ | registered DYCK_DECAY |
| `DYCK_DECAY_PROBE.md` | REGISTERED-LIVE | RIcat docs code | 2026-09-20 | move to results/dyck/DYCK_DECAY/ | registered DYCK_DECAY |
| `DYCK_DECAY_RESULTS.md` | REGISTERED-LIVE | RIcat mem top2 docs tex code2 | 2026-09-19 | move to results/dyck/DYCK_DECAY/ | registered DYCK_DECAY |
| `DYCK_DEPTH_PREREG.md` | REGISTERED-LIVE | RIcat top code2 | 2026-09-21 | move to results/dyck/DYCK_DEPTH/ | registered DYCK_DEPTH |
| `DYCK_DEPTH_RESULTS.json` | REGISTERED-LIVE | top docs | 2026-09-21 | move to results/dyck/DYCK_DEPTH/ | registered DYCK_DEPTH |
| `DYCK_DEPTH_RESULTS.md` | REGISTERED-LIVE | C-wd RIcat top docs4 code | 2026-09-22 | move to results/dyck/DYCK_DEPTH/ | registered DYCK_DEPTH; named in CLAUDE.md Withdrawn: a claim from it is withdrawn, the record stays |
| `DYCK_FAR_PROBE.md` | SUPPORTING | RIcat docs code | 2026-09-19 | move to results/dyck/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DYCK_GATES.json` | REGISTERED-LIVE | code2 | 2026-09-15 | move to results/dyck/DYCK/ | registered DYCK |
| `DYCK_GATES.md` | REGISTERED-LIVE | RIcat top4 docs code2 | 2026-09-15 | move to results/dyck/DYCK/ | registered DYCK |
| `DYCK_LADDER_PREREG.md` | REGISTERED-LIVE | RIcat top code2 | 2026-09-22 | move to results/dyck/DYCK_LADDER/ | registered DYCK_LADDER |
| `DYCK_LADDER_RESULTS.json` | REGISTERED-LIVE | mem docs3 | 2026-09-23 | move to results/dyck/DYCK_LADDER/ | registered DYCK_LADDER |
| `DYCK_LADDER_RESULTS.md` | CITED | C-cit C-wd RI RIcat mem top docs7 tex code dcode3 | 2026-09-25 | move to results/dyck/DYCK_LADDER/ | cited; registered DYCK_LADDER; banner: CORRECTED/AUDIT block (live); also named in CLAUDE.md Withdrawn (carries the correction) |
| `DYCK_LITERATURE_METRICS.json` | REGISTERED-LIVE | -- | 2026-09-19 | move to results/dyck/DYCK/ | registered DYCK |
| `DYCK_LITERATURE_METRICS.md` | CITED | C-cit C-wd RI RIcat docs2 tex code | 2026-09-19 | move to results/dyck/DYCK/ | cited; registered DYCK; also named in CLAUDE.md Withdrawn (carries the correction) |
| `DYCK_MDEPTH_GATES.json` | REGISTERED-LIVE | code2 dcode2 | 2026-09-26 | move to results/dyck/DYCK_MDEPTH/ | registered DYCK_MDEPTH |
| `DYCK_MDEPTH_GATES.md` | REGISTERED-LIVE | RIcat top2 docs2 code2 dcode2 | 2026-09-26 | move to results/dyck/DYCK_MDEPTH/ | registered DYCK_MDEPTH |
| `DYCK_MDEPTH_PREREG.md` | REGISTERED-LIVE | RI RIcat top docs2 paper3 tex code5 dcode7 | 2026-09-25 | move to results/dyck/DYCK_MDEPTH/ | registered DYCK_MDEPTH |
| `DYCK_MDEPTH_RESULTS.json` | CITED | DR paper3 code dcode2 | 2026-09-26 | move to results/dyck/DYCK_MDEPTH/ | cited; registered DYCK_MDEPTH |
| `DYCK_MDEPTH_RESULTS.md` | CITED | C-cit C-wd RI RIcat DR CM WTS mem2 top4 docs9 paper12 tex code dcode4 | 2026-09-26 | move to results/dyck/DYCK_MDEPTH/ | cited; registered DYCK_MDEPTH; also named in CLAUDE.md Withdrawn (carries the correction) |
| `DYCK_PREREG.md` | REGISTERED-LIVE | RIcat top6 docs code4 dcode | 2026-09-15 | move to results/dyck/DYCK/ | registered DYCK |
| `DYCK_RESULTS_bs128.md` | CITED | C-cit RI RIcat docs3 tex dcode | 2026-09-15 | move to results/dyck/DYCK/ | cited; registered DYCK |
| `DYCK_RESULTS_POPE.md` | SUPPORTING | RIcat docs2 tex | 2026-09-15 | move to results/dyck/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DYCK_SHAREABLE.md` | ORPHAN | RIcat | 2026-09-20 | archive to archive/orphan/dyck | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `DYCK_STACK_PROBE.md` | SUPPORTING | RIcat docs tex code dcode | 2026-09-20 | move to results/dyck/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `DYCK_STANDARD_METRICS.json` | REGISTERED-LIVE | code | 2026-09-15 | move to results/dyck/DYCK/ | registered DYCK |
| `DYCK_T3_RESULTS.md` | WITHDRAWN/VOID | C-oth RIcat docs3 code | 2026-09-20 | archive to archive/withdrawn/DYCK/ | withdrawn or void; title: EMPTY ARTEFACT (CLAUDE.md: an empty artefact) |
| `DYCK_TWO_METRICS.json` | REGISTERED-LIVE | code | 2026-09-15 | move to results/dyck/DYCK/ | registered DYCK |
| `EM_COMP_SAMEBATCH.json` | GENERATED-OUTPUT | code | 2026-08-09 | move with EM_COMP_SAMEBATCH.md -> `results/em_wm/EM_COMP_SAMEBATCH.json` | script output |
| `EM_COMP_SAMEBATCH.md` | GENERATED-OUTPUT | RIcat tex repinv3 code arch | 2026-08-09 | move to results/em_wm/ | script output (run_em_comp_samebatch.sh) |
| `EM_FIX_COMP.json` | GENERATED-OUTPUT | code2 | 2026-08-08 | archive with EM_FIX_COMP.md -> `archive/generated/em_wm/EM_FIX_COMP.json` | script output |
| `EM_FIX_COMP.md` | GENERATED-OUTPUT | RIcat repinv2 code2 arch | 2026-08-08 | archive to archive/generated/em_wm/ (regenerable: run_em_fix.sh) | script output (run_em_fix.sh) |
| `EM_HOPFIELD_CROSSSCALE.md` | GENERATED-OUTPUT | RIcat top repinv code arch | 2026-05-18 | move to results/em_wm/ | script output (run_em_perscale_and_hopfield.sh) |
| `EM_P0_COMP.json` | GENERATED-OUTPUT | code | 2026-08-09 | archive with EM_P0_COMP.md -> `archive/generated/em_wm/EM_P0_COMP.json` | script output |
| `EM_P0_COMP.md` | GENERATED-OUTPUT | RIcat repinv2 code2 arch | 2026-08-09 | archive to archive/generated/em_wm/ (regenerable: run_em_p0.sh) | script output (run_em_p0.sh) |
| `EM_P0_PAPER.md` | GENERATED-OUTPUT | RIcat repinv2 code arch | 2026-08-09 | archive to archive/generated/em_wm/ (regenerable: run_em_p0.sh) | script output (run_em_p0.sh) |
| `EM_WM_STATE.md` | CITED | C-cit C-oth RI RIcat WTS mem3 top5 docs7 paper2 tex repinv5 code | 2026-09-25 | move to results/em_wm/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; banner: CORRECTED/AUDIT block (live) |
| `EM_WM_THEORY.md` | SUPPORTING | RI RIcat mem top7 docs2 tex repinv3 code2 | 2026-09-12 | move to results/em_wm/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `ENWIK8_2X2.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/code_enwik8/ (regenerable: run_enwik8_2x2.sh) | script output (run_enwik8_2x2.sh) |
| `ENWIK8_COMPOSITION_PREREG.md` | REGISTERED-LIVE | RIcat code | 2026-09-22 | move to results/code_enwik8/ENWIK8_COMPOSITION/ | registered ENWIK8_COMPOSITION; pre-registration with no results file (unrun / stopped / postponed) |
| `ENWIK8_HIER.md` | WITHDRAWN/VOID | C-wd RIcat repinv code | 2026-08-30 | archive to archive/withdrawn/code_enwik8/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `ENWIK8_HIERARCHY.md` | CITED | C-cit RI RIcat docs4 tex repinv2 | 2026-09-07 | move to results/code_enwik8/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `ENWIK8_LONG.md` | GENERATED-OUTPUT | RIcat repinv2 code | 2026-09-19 | archive to archive/generated/code_enwik8/ (regenerable: run_enwik8_long.sh) | script output (run_enwik8_long.sh) |
| `ENWIK8_SEEDS.md` | GENERATED-OUTPUT | RIcat top2 docs repinv code | 2026-08-28 | move to results/code_enwik8/ | script output (run_enwik8_seeds.sh) |
| `EXPTORUS_ANALYSIS.txt` | REGISTERED-LIVE | RI top2 code | 2026-10-26 | move to results/sign/EXPTORUS/ | registered EXPTORUS |
| `EXPTORUS_PREREG.md` | CITED | RI DR top2 docs2 code6 | 2026-10-26 | move to results/sign/EXPTORUS/ | cited; registered EXPTORUS |
| `EXPTORUS_RESULTS.md` | CITED | C-cit RI DR2 CM mem paper | 2026-10-26 | move to results/sign/EXPTORUS/ | cited; registered EXPTORUS |
| `EXPTORUS_VERDICTS.json` | REGISTERED-LIVE | RI top2 docs code2 | 2026-10-26 | move to results/sign/EXPTORUS/ | registered EXPTORUS |
| `EXTRAHEAD_CONTROL.md` | CITED | C-cit RI RIcat top4 docs repinv2 code3 arch | 2026-05-21 | move to archive/legacy_level15/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `FACTORIAL_ANALYSIS.txt` | REGISTERED-LIVE | top code2 | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1.json` | REGISTERED-LIVE | code | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1_OWN_s500.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1_OWN_s500.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1_OWN_s501.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1_OWN_s501.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1_OWN_s502.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1_OWN_s502.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1_OWN_s503.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1_OWN_s503.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L1_STRATA.json` | REGISTERED-LIVE | docs paper3 code2 dcode2 | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4.json` | REGISTERED-LIVE | code | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4_OWN_s500.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4_OWN_s500.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4_OWN_s501.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4_OWN_s501.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4_OWN_s502.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4_OWN_s502.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4_OWN_s503.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4_OWN_s503.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_L4_STRATA.json` | REGISTERED-LIVE | paper2 code dcode2 | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP.json` | REGISTERED-LIVE | code | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP_OWN_s500.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP_OWN_s500.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP_OWN_s501.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP_OWN_s501.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP_OWN_s502.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP_OWN_s502.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP_OWN_s503.json` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP_OWN_s503.md` | REGISTERED-LIVE | -- | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_LOOP_STRATA.json` | REGISTERED-LIVE | paper code dcode2 | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_N8_ANALYSIS.txt` | REGISTERED-LIVE | RI top paper code | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1.json` | REGISTERED-LIVE | code | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1_OWN_s2800.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1_OWN_s2800.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1_OWN_s2801.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1_OWN_s2801.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1_OWN_s2802.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1_OWN_s2802.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1_OWN_s2803.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1_OWN_s2803.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L1_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4.json` | REGISTERED-LIVE | code | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4_OWN_s2800.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4_OWN_s2800.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4_OWN_s2801.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4_OWN_s2801.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4_OWN_s2802.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4_OWN_s2802.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4_OWN_s2803.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4_OWN_s2803.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_L4_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP.json` | REGISTERED-LIVE | code | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP_OWN_s2800.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP_OWN_s2800.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP_OWN_s2801.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP_OWN_s2801.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP_OWN_s2802.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP_OWN_s2802.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP_OWN_s2803.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP_OWN_s2803.md` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_LOOP_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_N8_PREREG.md` | CITED | RI DR top docs code4 | 2026-10-24 | move to results/recurrence/FACTORIAL_N8/ | cited; registered FACTORIAL_N8 |
| `FACTORIAL_N8_RESULTS.md` | CITED | C-cit RI DR CM mem docs2 paper | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | cited; registered FACTORIAL_N8 |
| `FACTORIAL_N8_VERDICTS.json` | REGISTERED-LIVE | top docs paper code | 2026-10-25 | move to results/recurrence/FACTORIAL_N8/ | registered FACTORIAL_N8 |
| `FACTORIAL_PREREG.md` | REGISTERED-LIVE | top docs code2 | 2026-10-12 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FACTORIAL_RESULTS.md` | CITED | C-cit RI DR CM top2 docs paper16 dcode | 2026-10-13 | move to results/recurrence/FACTORIAL/ | cited; registered FACTORIAL |
| `FACTORIAL_VERDICTS.json` | REGISTERED-LIVE | top docs code2 | 2026-10-13 | move to results/recurrence/FACTORIAL/ | registered FACTORIAL |
| `FAMILY_TREE_D7_GATES.json` | GENERATED-OUTPUT | repinv | 2026-08-16 | archive with FAMILY_TREE_D7_GATES.md -> `archive/generated/algorithmic/FAMILY_TREE_D7_GATES.json` | script output |
| `FAMILY_TREE_D7_GATES.md` | GENERATED-OUTPUT | RIcat repinv code arch | 2026-08-16 | archive to archive/generated/algorithmic/ (regenerable: run_family_tree_d7.sh) | script output (run_family_tree_d7.sh) |
| `FAMILY_TREE_D7_RESULTS.md` | SUPPORTING | RIcat tex repinv3 code | 2026-08-16 | move to results/algorithmic/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `FAMILY_TREE_GATES.json` | GENERATED-OUTPUT | repinv code3 | 2026-08-15 | move with FAMILY_TREE_GATES.md -> `results/algorithmic/FAMILY_TREE_GATES.json` | script output |
| `FAMILY_TREE_GATES.md` | GENERATED-OUTPUT | RIcat tex repinv code2 arch | 2026-08-16 | move to results/algorithmic/ | script output (run_family_tree.sh, validate_family_tree.py); banner: CORRECTED/AUDIT block (live) |
| `FAMILY_TREE_RESULTS.md` | CITED | C-cit RI RIcat top4 docs3 repinv3 code2 arch | 2026-08-15 | move to results/algorithmic/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `FAMILY_TREE_WM_GAP.md` | SUPPORTING | RIcat top2 docs tex repinv3 | 2026-09-07 | move to results/algorithmic/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `FAST_ATTN_RANK.md` | SUPPORTING | C-oth RIcat top docs code | 2026-09-24 | move to results/rank/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `FLIPFLOP_GATES.json` | GENERATED-OUTPUT | -- | 2026-09-07 | move with FLIPFLOP_GATES.md -> `results/sign/FLIPFLOP_GATES.json` | script output |
| `FLIPFLOP_GATES.md` | SUPPORTING | RIcat top tex repinv code3 | 2026-09-07 | move to results/sign/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `FLIPFLOP_RESULTS.md` | SUPPORTING | RIcat docs tex repinv3 | 2026-09-07 | move to results/sign/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `FORGET_CLOCK_PREREG.md` | REGISTERED-LIVE | RI RIcat top docs2 repinv2 code | 2026-09-08 | move to results/sign/FORGET_CLOCK/ | registered FORGET_CLOCK; pre-registration with no results file (unrun / stopped / postponed) |
| `FORGET_CONTROL.json` | GENERATED-OUTPUT | code | 2026-09-05 | move with FORGET_CONTROL.md -> `results/sign/FORGET_CONTROL.json` | script output (run_lambda_trace.sh) |
| `FORGET_CONTROL.md` | CITED | C-cit RI RIcat top docs4 tex repinv3 code2 | 2026-09-05 | move to results/sign/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `FORGET_GATE.json` | GENERATED-OUTPUT | -- | 2026-09-05 | move with FORGET_GATE.md -> `results/sign/FORGET_GATE.json` | script output |
| `FORGET_GATE.md` | GENERATED-OUTPUT | RI RIcat top2 docs2 tex repinv3 code3 | 2026-09-05 | move to results/sign/ | script output (run_forget.sh) |
| `FREQ_CONTROL.json` | GENERATED-OUTPUT | -- | 2026-08-21 | move with FREQ_CONTROL.md -> `results/navigation/torus/FREQ_CONTROL.json` | script output |
| `FREQ_CONTROL.md` | SUPPORTING | RIcat top docs tex2 repinv3 code4 dcode2 | 2026-08-21 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `FRONTIER_ALGORITHMIC.md` | GENERATED-OUTPUT | RIcat mem2 docs2 tex2 repinv2 code | 2026-09-03 | move to results/algorithmic/ | script output (run_frontier.sh) |
| `G3_SSM_A2_ANALYSIS.txt` | REGISTERED-LIVE | RI top2 paper code | 2026-10-25 | move to results/recurrence/G3_SSM/ | registered G3_SSM |
| `G3_SSM_A2_RESULTS.md` | CITED | C-cit RI DR2 CM mem top3 docs2 paper2 | 2026-10-25 | move to results/recurrence/G3_SSM/ | cited; registered G3_SSM |
| `G3_SSM_A2_VERDICTS.json` | REGISTERED-LIVE | top2 docs paper code | 2026-10-25 | move to results/recurrence/G3_SSM/ | registered G3_SSM |
| `G3_SSM_ANALYSIS.txt` | REGISTERED-LIVE | RI top docs paper code | 2026-10-23 | move to results/recurrence/G3_SSM/ | registered G3_SSM |
| `G3_SSM_PREREG.md` | CITED | RI DR top3 docs3 code6 | 2026-10-24 | move to results/recurrence/G3_SSM/ | cited; registered G3_SSM |
| `G3_SSM_RESULTS.md` | CITED | C-cit RI DR2 CM mem top3 docs3 paper | 2026-10-25 | move to results/recurrence/G3_SSM/ | cited; registered G3_SSM |
| `G3_SSM_VERDICTS.json` | REGISTERED-LIVE | top2 docs3 paper code2 | 2026-10-23 | move to results/recurrence/G3_SSM/ | registered G3_SSM |
| `GAIN_GRAIN_ANALYSIS.txt` | REGISTERED-LIVE | top docs code | 2026-10-05 | move to results/pope/GAIN_GRAIN/ | registered GAIN_GRAIN |
| `GAIN_GRAIN_EVAL.json` | REGISTERED-LIVE | code2 dcode | 2026-10-05 | move to results/pope/GAIN_GRAIN/ | registered GAIN_GRAIN |
| `GAIN_GRAIN_EVAL.md` | REGISTERED-LIVE | RIcat code | 2026-10-05 | move to results/pope/GAIN_GRAIN/ | registered GAIN_GRAIN |
| `GAIN_GRAIN_PREREG.md` | REGISTERED-LIVE | RI RIcat top code4 dcode4 | 2026-10-05 | move to results/pope/GAIN_GRAIN/ | registered GAIN_GRAIN |
| `GAIN_GRAIN_RESCORE.json` | REGISTERED-LIVE | dcode | 2026-10-05 | move to results/pope/GAIN_GRAIN/ | registered GAIN_GRAIN |
| `GAIN_GRAIN_RESCORE.md` | REGISTERED-LIVE | RIcat code | 2026-10-05 | move to results/pope/GAIN_GRAIN/ | registered GAIN_GRAIN |
| `GAIN_GRAIN_RESULTS.md` | CITED | C-cit RI RIcat DR mem2 top2 paper | 2026-10-11 | move to results/pope/GAIN_GRAIN/ | cited; registered GAIN_GRAIN; banner: CORRECTED/AUDIT block (live) |
| `GAIN_PHASE_ANALYSIS.txt` | REGISTERED-LIVE | top docs code | 2026-10-09 | move to results/language/GAIN_PHASE/ | registered GAIN_PHASE |
| `GAIN_PHASE_EVAL.json` | REGISTERED-LIVE | top docs code3 dcode2 | 2026-10-09 | move to results/language/GAIN_PHASE/ | registered GAIN_PHASE |
| `GAIN_PHASE_PREREG.md` | REGISTERED-LIVE | RI RIcat top docs code6 dcode3 | 2026-10-06 | move to results/language/GAIN_PHASE/ | registered GAIN_PHASE |
| `GAIN_PHASE_RESULTS.md` | CITED | C-cit C-wd RI RIcat DR mem docs paper7 | 2026-10-11 | move to results/language/GAIN_PHASE/ | cited; registered GAIN_PHASE; banner: CORRECTED/AUDIT block (live); also named in CLAUDE.md Withdrawn (carries the correction) |
| `GAIN_PHASE_VERDICTS.json` | REGISTERED-LIVE | top code2 | 2026-10-09 | move to results/language/GAIN_PHASE/ | registered GAIN_PHASE |
| `GATE_PROBE.md` | SUPPORTING | RIcat mem top3 docs tex repinv2 code4 oth2 | 2026-09-03 | move to results/rank/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `GATED_PREREG.md` | REGISTERED-LIVE | RIcat top4 repinv code2 | 2026-09-07 | move to results/sign/GATED/ | registered GATED |
| `GATED_RESULTS.md` | CITED | C-cit RIcat top docs3 tex repinv3 | 2026-09-07 | move to results/sign/GATED/ | cited; registered GATED |
| `GATED_SEPARATION.json` | REGISTERED-LIVE | -- | 2026-09-07 | move to results/sign/GATED/ | registered GATED |
| `GATED_SEPARATION.md` | REGISTERED-LIVE | RIcat tex repinv2 code | 2026-09-07 | move to results/sign/GATED/ | registered GATED |
| `GATED_TORUS.json` | REGISTERED-LIVE | docs | 2026-09-07 | move to results/sign/GATED/ | registered GATED |
| `GATED_TORUS.md` | REGISTERED-LIVE | RIcat tex repinv2 | 2026-09-07 | move to results/sign/GATED/ | registered GATED |
| `GAUGE_ANALYSIS.txt` | REGISTERED-LIVE | top code | 2026-10-15 | move to results/sign/GAUGE/ | registered GAUGE |
| `GAUGE_PREREG.md` | REGISTERED-LIVE | top docs2 paper code4 | 2026-10-15 | move to results/sign/GAUGE/ | registered GAUGE |
| `GAUGE_RESULTS.md` | CITED | C-cit DR CM mem paper10 | 2026-10-16 | move to results/sign/GAUGE/ | cited; registered GAUGE; banner: CORRECTED/AUDIT block (live) |
| `GAUGE_VERDICTS.json` | REGISTERED-LIVE | top code2 | 2026-10-15 | move to results/sign/GAUGE/ | registered GAUGE |
| `GENERALITY_ANALYSIS.txt` | CITED | DR top paper5 code dcode | 2026-10-14 | move to results/sign/GENERALITY/ | cited; registered GENERALITY |
| `GENERALITY_PREREG.md` | REGISTERED-LIVE | top docs paper4 code4 | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_RESULTS.md` | CITED | C-cit CM mem top docs paper11 dcode2 | 2026-10-14 | move to results/sign/GENERALITY/ | cited; registered GENERALITY |
| `GENERALITY_TORUS_AO4.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_AO4.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY; older version of GENERALITY_TORUS_AO4_RS.md |
| `GENERALITY_TORUS_AO4_RS.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_AO4_RS.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_AO4_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_AO_bal.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_AO_bal.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_AO_bal_hi.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_AO_bal_hi.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_AO_bal_hi_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_AO_bal_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_CARoPEF.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_CARoPEF.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY; older version of GENERALITY_TORUS_CARoPEF_RS.md |
| `GENERALITY_TORUS_CARoPEF_RS.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_CARoPEF_RS.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_CARoPEF_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_RoPE.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_RoPE.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY; older version of GENERALITY_TORUS_RoPE_RS.md |
| `GENERALITY_TORUS_RoPE_RS.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_RoPE_RS.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_RoPE_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_Signed_r4.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_Signed_r4.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY; older version of GENERALITY_TORUS_Signed_r4_RS.md |
| `GENERALITY_TORUS_Signed_r4_RS.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_Signed_r4_RS.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_Signed_r4_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO4.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO4.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY; older version of GENERALITY_TORUS_SO4_RS.md |
| `GENERALITY_TORUS_SO4_RS.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO4_RS.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO4_STRATA.json` | REGISTERED-LIVE | paper2 | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO_bal.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO_bal.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO_bal_hi.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO_bal_hi.md` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO_bal_hi_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_TORUS_SO_bal_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALITY_VERDICTS.json` | REGISTERED-LIVE | top docs code2 | 2026-10-14 | move to results/sign/GENERALITY/ | registered GENERALITY |
| `GENERALIZATION_REPORT.md` | ORPHAN | RIcat repinv arch | 2026-05-18 | archive to archive/reports_old/ | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `GSF_FULL_RESULTS.md` | WITHDRAWN/VOID | RIcat repinv code arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows) |
| `GUARDS.md` | META | C-oth RIcat docs repinv code | 2026-09-24 | keep at root | project infrastructure |
| `H12_BUDGET_CURVE.md` | CITED | C-cit C-wd RI RIcat WTS top5 docs3 tex repinv4 code2 | 2026-08-23 | move to results/navigation/torus/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; also named in CLAUDE.md Withdrawn (carries the correction) |
| `HABITAT_BUILD.md` | CITED | C-cit RIcat top2 docs3 tex repinv2 code | 2026-08-23 | move to results/navigation/environments/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `HIER_ATTN_LONGT.md` | SUPPORTING | RIcat top docs repinv code2 arch | 2026-07-16 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `HIER_PARITY.md` | GENERATED-OUTPUT | RIcat mem top docs2 tex2 repinv3 code4 | 2026-09-03 | move to results/hierarchy/ | script output (run_hier_parity.sh); banner: SUPERSEDED IN PART (live) |
| `HIER_RECHECK.json` | GENERATED-OUTPUT | repinv | 2026-09-08 | move with HIER_RECHECK.md -> `results/hierarchy/HIER_RECHECK.json` | script output |
| `HIER_RECHECK.md` | CITED | C-cit RI RIcat top docs5 tex repinv3 | 2026-09-08 | move to results/hierarchy/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `HIERGOAL_ABLATION.json` | GENERATED-OUTPUT | -- | 2026-08-09 | archive with HIERGOAL_ABLATION.md -> `archive/withdrawn/hierarchy/HIERGOAL_ABLATION.json` | script output |
| `HIERGOAL_ABLATION.md` | WITHDRAWN/VOID | C-wd RIcat top4 docs tex repinv2 code arch10 | 2026-08-10 | archive to archive/withdrawn/hierarchy/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `HIERGOAL_CLOSEDLOOP.md` | GENERATED-OUTPUT | RIcat tex repinv2 code arch5 | 2026-08-09 | move to results/hierarchy/ | script output (eval_hier_goal_closedloop.py) |
| `HIERGOAL_LONGT.json` | GENERATED-OUTPUT | code | 2026-08-08 | archive with HIERGOAL_LONGT.md -> `archive/orphan/hierarchy/HIERGOAL_LONGT.json` | script output |
| `HIERGOAL_LONGT.md` | ORPHAN | RIcat repinv code2 arch | 2026-08-08 | archive to archive/orphan/hierarchy (regenerable: autocommit_longt.sh, eval_hiergoal_longT.py) | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `HIPPOCAMPAL_ANALYSIS.md` | WITHDRAWN/VOID | RIcat top2 docs2 repinv code arch | 2026-04-24 | delete (identical copy already in archive_stale/HIPPOCAMPAL_ANALYSIS.md) | withdrawn or void; copy in archive_stale/HIPPOCAMPAL_ANALYSIS.md (identical) |
| `HIPPOCAMPAL_GRID.md` | WITHDRAWN/VOID | RIcat repinv arch | 2026-04-26 | delete (identical copy already in archive_stale/HIPPOCAMPAL_GRID.md) | withdrawn or void; copy in archive_stale/HIPPOCAMPAL_GRID.md (identical) |
| `HIPPOCAMPAL_GRID_FREE.md` | WITHDRAWN/VOID | RIcat docs repinv arch | 2026-04-26 | delete (identical copy already in archive_stale/HIPPOCAMPAL_GRID_FREE.md) | withdrawn or void; copy in archive_stale/HIPPOCAMPAL_GRID_FREE.md (identical) |
| `HIPPOCAMPAL_GRIDL15PC.md` | WITHDRAWN/VOID | RIcat repinv arch | 2026-04-26 | delete (identical copy already in archive_stale/HIPPOCAMPAL_GRIDL15PC.md) | withdrawn or void; copy in archive_stale/HIPPOCAMPAL_GRIDL15PC.md (identical) |
| `HIPPOCAMPAL_HIDDEN.md` | WITHDRAWN/VOID | RIcat top repinv code arch | 2026-04-24 | delete (identical copy already in archive_stale/HIPPOCAMPAL_HIDDEN.md) | withdrawn or void; copy in archive_stale/HIPPOCAMPAL_HIDDEN.md (identical) |
| `HIPPOCAMPAL_HIDDEN_GRIDFREE.md` | WITHDRAWN/VOID | RIcat repinv arch | 2026-04-26 | delete (identical copy already in archive_stale/HIPPOCAMPAL_HIDDEN_GRIDFREE.md) | withdrawn or void; copy in archive_stale/HIPPOCAMPAL_HIDDEN_GRIDFREE.md (identical) |
| `HIPPOCAMPAL_LEVEL15PC.md` | WITHDRAWN/VOID | RIcat docs repinv code arch | 2026-04-27 | delete (identical copy already in archive_stale/HIPPOCAMPAL_LEVEL15PC.md) | withdrawn or void; copy in archive_stale/HIPPOCAMPAL_LEVEL15PC.md (identical) |
| `HOPFIELD_NOMAINAP_RESULTS.md` | SUPPORTING | RIcat top repinv code arch | 2026-05-21 | move to results/em_wm/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `HORIZON_L1d128e16.json` | GENERATED-OUTPUT | -- | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L1d128e16.json` | script output |
| `HORIZON_L1d128e16.md` | ORPHAN | RIcat repinv | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L1d128e16.md` | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `HORIZON_L1d128e50.json` | GENERATED-OUTPUT | -- | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L1d128e50.json` | script output |
| `HORIZON_L1d128e50.md` | ORPHAN | RIcat repinv | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L1d128e50.md` | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `HORIZON_L2d128e16.json` | GENERATED-OUTPUT | -- | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L2d128e16.json` | script output |
| `HORIZON_L2d128e16.md` | ORPHAN | RIcat repinv | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L2d128e16.md` | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `HORIZON_L2d256e16.json` | GENERATED-OUTPUT | -- | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L2d256e16.json` | script output |
| `HORIZON_L2d256e16.md` | ORPHAN | RIcat repinv | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L2d256e16.md` | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `HORIZON_L4d128e16.json` | GENERATED-OUTPUT | -- | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L4d128e16.json` | script output |
| `HORIZON_L4d128e16.md` | ORPHAN | RIcat repinv | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L4d128e16.md` | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `HORIZON_L4d256e16.json` | GENERATED-OUTPUT | -- | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L4d256e16.json` | script output |
| `HORIZON_L4d256e16.md` | ORPHAN | RIcat repinv | 2026-08-20 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_L4d256e16.md` | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `HORIZON_RESULTS.md` | WITHDRAWN/VOID | C-wd RIcat top2 docs2 tex repinv code3 | 2026-08-30 | archive to archive/withdrawn/navigation/torus/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `HORIZON_TASK_DISTANCES.json` | GENERATED-OUTPUT | -- | 2026-08-19 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_TASK_DISTANCES.json` | script output |
| `HORIZON_TASK_DISTANCES.md` | ORPHAN | RIcat repinv code2 | 2026-08-19 | archive with HORIZON_RESULTS.md -> `archive/withdrawn/navigation/torus/HORIZON_TASK_DISTANCES.md` | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `HOURGLASS_README.md` | META | C-oth RIcat docs2 repinv arch | 2026-07-25 | keep at root | project infrastructure |
| `IMAGE_TORUS_ANALYSIS.txt` | REGISTERED-LIVE | RI top docs paper code | 2026-10-26 | move to results/images/IMAGE_TORUS/ | registered IMAGE_TORUS |
| `IMAGE_TORUS_GATE.json` | REGISTERED-LIVE | code3 | 2026-10-25 | move to results/images/IMAGE_TORUS/ | registered IMAGE_TORUS |
| `IMAGE_TORUS_GATE.md` | REGISTERED-LIVE | RI top code | 2026-10-25 | move to results/images/IMAGE_TORUS/ | registered IMAGE_TORUS |
| `IMAGE_TORUS_PILOT.txt` | REGISTERED-LIVE | RI top2 docs code | 2026-10-25 | move to results/images/IMAGE_TORUS/ | registered IMAGE_TORUS; older version of IMAGE_TORUS_PILOT2.txt |
| `IMAGE_TORUS_PILOT2.txt` | REGISTERED-LIVE | RI top2 docs code | 2026-10-25 | move to results/images/IMAGE_TORUS/ | registered IMAGE_TORUS |
| `IMAGE_TORUS_PREREG.md` | CITED | RI DR top docs4 code10 | 2026-10-25 | move to results/images/IMAGE_TORUS/ | cited; registered IMAGE_TORUS |
| `IMAGE_TORUS_RESULTS.md` | CITED | C-cit RI DR CM mem docs3 paper | 2026-10-26 | move to results/images/IMAGE_TORUS/ | cited; registered IMAGE_TORUS |
| `IMAGE_TORUS_VERDICTS.json` | REGISTERED-LIVE | RI top2 docs2 paper code2 | 2026-10-26 | move to results/images/IMAGE_TORUS/ | registered IMAGE_TORUS |
| `INDEX_BASELINE_PAPER_TASK.json` | GENERATED-OUTPUT | -- | 2026-08-17 | archive with INDEX_BASELINE_PAPER_TASK.md -> `archive/superseded/navigation/torus/INDEX_BASELINE_PAPER_TASK.json` | script output |
| `INDEX_BASELINE_PAPER_TASK.md` | SUPERSEDED | RIcat top docs repinv2 code4 arch | 2026-08-17 | archive to archive/superseded/navigation/torus/ | superseded; superseded by INDEX_BASELINE_PAPER_TASK_n8.md |
| `INDEX_BASELINE_PAPER_TASK_n8.json` | GENERATED-OUTPUT | top docs repinv code | 2026-08-20 | move with INDEX_BASELINE_PAPER_TASK_n8.md -> `results/navigation/torus/INDEX_BASELINE_PAPER_TASK_n8.json` | script output |
| `INDEX_BASELINE_PAPER_TASK_n8.md` | SUPPORTING | RIcat top9 docs tex2 repinv6 code5 | 2026-08-20 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `INDIRECT_ARITHMETIC.md` | REGISTERED-LIVE | RIcat docs2 | 2026-09-25 | move to results/algorithmic/INDIRECT/ | registered INDIRECT; banner: CORRECTED/AUDIT block (live) |
| `INDIRECT_OOD.json` | REGISTERED-LIVE | -- | 2026-09-17 | move to results/algorithmic/INDIRECT/ | registered INDIRECT |
| `INDIRECT_OOD.md` | REGISTERED-LIVE | RI RIcat top docs tex code2 | 2026-09-20 | move to results/algorithmic/INDIRECT/ | registered INDIRECT |
| `INDIRECT_PREREG.md` | REGISTERED-LIVE | RIcat code3 | 2026-09-16 | move to results/algorithmic/INDIRECT/ | registered INDIRECT |
| `INDIRECT_RESULTS.md` | REGISTERED-LIVE | RIcat tex code2 | 2026-09-20 | move to results/algorithmic/INDIRECT/ | registered INDIRECT; banner: SUPERSEDED IN PART (live); older version of INDIRECT_RESULTS_200k.md |
| `INDIRECT_RESULTS_200k.md` | CITED | C-cit RI RIcat top2 docs3 paper tex code | 2026-09-17 | move to results/algorithmic/INDIRECT/ | cited; registered INDIRECT |
| `JSB_LENGTH_RESULTS.md` | REGISTERED-LIVE | RI RIcat top docs3 code2 | 2026-09-17 | move to results/pope/JSBLEN/ | registered JSBLEN |
| `JSB_LENGTH_RESULTS_BASE.md` | REGISTERED-LIVE | RIcat top tex code | 2026-09-19 | move to results/pope/JSBLEN/ | registered JSBLEN |
| `JSB_LENGTH_RESULTS_RANK.md` | REGISTERED-LIVE | RI RIcat mem top tex code | 2026-09-19 | move to results/pope/JSBLEN/ | registered JSBLEN |
| `JSB_PREREG.md` | REGISTERED-LIVE | RIcat top2 code3 | 2026-09-15 | move to results/pope/JSB/ | registered JSB |
| `JSB_RESULTS.md` | CITED | C-cit RI RIcat mem top docs4 paper tex code | 2026-09-20 | move to results/pope/JSB/ | cited; registered JSB |
| `JSBLEN_PREREG.md` | REGISTERED-LIVE | RIcat top code3 | 2026-09-17 | move to results/pope/JSBLEN/ | registered JSBLEN |
| `KNOB_SWEEP.json` | GENERATED-OUTPUT | -- | 2026-08-21 | move with KNOB_SWEEP.md -> `results/navigation/torus/KNOB_SWEEP.json` | script output |
| `KNOB_SWEEP.md` | CITED | RIcat DR top4 docs2 paper tex2 repinv3 code5 dcode2 | 2026-08-21 | move to results/navigation/torus/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; older version of KNOB_SWEEP_n8.md |
| `KNOB_SWEEP_n8.json` | GENERATED-OUTPUT | docs code2 | 2026-08-22 | move with KNOB_SWEEP_n8.md -> `results/navigation/torus/KNOB_SWEEP_n8.json` | script output |
| `KNOB_SWEEP_n8.md` | CITED | C-cit RI RIcat DR WTS top2 docs5 paper2 tex repinv4 code2 | 2026-08-22 | move to results/navigation/torus/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `KNOWN_BUGS.md` | META | C-oth RIcat top repinv7 | 2026-08-28 | keep at root | project infrastructure |
| `L15_ABLATION.json` | GENERATED-OUTPUT | docs3 | 2026-09-01 | move with L15_ABLATION.md -> `archive/legacy_level15/L15_ABLATION.json` | script output |
| `L15_ABLATION.md` | CITED | C-cit C-wd RI RIcat mem top2 docs5 tex2 repinv4 code3 oth | 2026-09-01 | move to archive/legacy_level15/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; also named in CLAUDE.md Withdrawn (carries the correction) |
| `L15_LOOP_2X2.md` | CITED | C-cit RI RIcat mem top2 docs3 paper tex repinv4 code2 dcode | 2026-09-01 | move to results/recurrence/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `LAMBDA_TRACE.md` | GENERATED-OUTPUT | RIcat top docs2 tex repinv2 code | 2026-09-05 | move to results/sign/ | script output (run_lambda_trace.sh) |
| `LANGUAGE_LANDSCAPE.md` | SUPPORTING | RIcat docs2 tex repinv3 code2 | 2026-08-28 | move to results/language/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `LAP_GATES.json` | GENERATED-OUTPUT | -- | 2026-08-14 | move with LAP_GATES.md -> `results/hierarchy/LAP_GATES.json` | script output |
| `LAP_GATES.md` | SUPPORTING | RIcat top repinv code arch | 2026-08-15 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `LAP_TRANSFER.json` | GENERATED-OUTPUT | -- | 2026-08-14 | move with LAP_TRANSFER.md -> `results/hierarchy/LAP_TRANSFER.json` | script output |
| `LAP_TRANSFER.md` | SUPPORTING | RIcat top repinv code arch | 2026-08-14 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `LAP_TRANSFER_NOREWARD.json` | GENERATED-OUTPUT | -- | 2026-08-15 | archive with LAP_TRANSFER_NOREWARD.md -> `archive/withdrawn/hierarchy/LAP_TRANSFER_NOREWARD.json` | script output |
| `LAP_TRANSFER_NOREWARD.md` | WITHDRAWN/VOID | C-wd RIcat top2 docs repinv arch | 2026-08-15 | archive to archive/withdrawn/hierarchy/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `LEAK.json` | REGISTERED-LIVE | top2 docs2 paper code3 dcode2 | 2026-10-03 | move to results/language/LEAK/ | registered LEAK |
| `LEAK_ANALYSIS.txt` | REGISTERED-LIVE | top2 code | 2026-10-03 | move to results/language/LEAK/ | registered LEAK |
| `LEAK_PREREG.md` | REGISTERED-LIVE | RI RIcat top code5 | 2026-10-03 | move to results/language/LEAK/ | registered LEAK |
| `LEAK_RESULTS.md` | CITED | C-cit RI RIcat DR WTS mem2 top5 docs10 paper6 code | 2026-10-03 | move to results/language/LEAK/ | cited; registered LEAK |
| `LEAK_SECONDARY.txt` | REGISTERED-LIVE | top paper | 2026-10-03 | move to results/language/LEAK/ | registered LEAK |
| `LEARNED_RANK.md` | ORPHAN | RIcat repinv code2 | 2026-09-04 | archive to archive/orphan/rank (regenerable: probe_rank_learned.py) | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `LEVEL15_MEETS_GATED_matchq.md` | SUPPORTING | RIcat top tex repinv3 code arch | 2026-08-17 | archive to archive/legacy_level15/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `LEVEL15_MEETS_GATED_paper.json` | GENERATED-OUTPUT | -- | 2026-08-23 | archive with LEVEL15_MEETS_GATED_paper.md -> `archive/legacy_level15/LEVEL15_MEETS_GATED_paper.json` | script output |
| `LEVEL15_MEETS_GATED_paper.md` | ORPHAN | RIcat repinv code | 2026-08-17 | archive to archive/legacy_level15/ | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `LEVEL15_MEETS_GATED_paper50.json` | GENERATED-OUTPUT | -- | 2026-08-18 | archive with LEVEL15_MEETS_GATED_paper50.md -> `archive/legacy_level15/LEVEL15_MEETS_GATED_paper50.json` | script output |
| `LEVEL15_MEETS_GATED_paper50.md` | SUPPORTING | RIcat tex repinv code arch | 2026-08-18 | archive to archive/legacy_level15/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `LEVEL15BETA_RESULTS.md` | WITHDRAWN/VOID | RIcat mem top2 repinv code arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows) |
| `LEVEL15EM_CROSSSCALE.md` | GENERATED-OUTPUT | RIcat top repinv code arch | 2026-05-18 | archive to archive/legacy_level15/ | script output (run_level15em_crossscale.sh) |
| `LM200_ABLATION.json` | GENERATED-OUTPUT | -- | 2026-08-21 | archive with LM200_ABLATION.md -> `archive/legacy_level15/LM200_ABLATION.json` | script output |
| `LM200_ABLATION.md` | SUPPORTING | RIcat top docs2 tex repinv3 code | 2026-08-21 | archive to archive/legacy_level15/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `LM200_CORRECTED_MULTISEED.md` | SUPPORTING | RI RIcat top4 docs tex repinv code2 arch | 2026-07-22 | archive to archive/legacy_level15/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources); banner: CORRECTED/AUDIT block (live) |
| `LOCALISATION.json` | REGISTERED-LIVE | -- | 2026-09-06 | move to results/rank/LOCALISATION/ | registered LOCALISATION |
| `LOCALISATION.md` | CITED | C-wd RIcat CM mem top2 docs4 tex repinv4 code dcode | 2026-09-06 | move to results/rank/LOCALISATION/ | cited; registered LOCALISATION; also named in CLAUDE.md Withdrawn (carries the correction) |
| `LOCALISATION_PREREG.md` | REGISTERED-LIVE | RIcat top2 repinv code2 | 2026-09-06 | move to results/rank/LOCALISATION/ | registered LOCALISATION |
| `LOCALISATION_RANK.json` | REGISTERED-LIVE | -- | 2026-09-06 | move to results/rank/LOCALISATION/ | registered LOCALISATION |
| `LOCALISATION_RANK.md` | REGISTERED-LIVE | RIcat tex repinv code | 2026-09-06 | move to results/rank/LOCALISATION/ | registered LOCALISATION |
| `LONG_SEQ_clean.md` | GENERATED-OUTPUT | RIcat tex repinv4 code arch | 2026-04-26 | move to results/navigation/torus/ | script output (master_level15pc.sh) |
| `LOOP_DEPTH_STRATA.json` | GENERATED-OUTPUT | -- | 2026-09-02 | move with LOOP_DEPTH_STRATA.md -> `results/recurrence/LOOP_DEPTH_STRATA.json` | script output |
| `LOOP_DEPTH_STRATA.md` | CITED | C-cit RIcat docs3 paper tex repinv3 code | 2026-09-02 | move to results/recurrence/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `LOOP_HEADROOM.md` | CITED | C-cit C-wd RI RIcat WTS mem top6 docs3 paper2 tex repinv4 code4 | 2026-08-31 | move to results/recurrence/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; also named in CLAUDE.md Withdrawn (carries the correction) |
| `LOOP_HIER_COMPUTE.json` | GENERATED-OUTPUT | -- | 2026-09-03 | move with LOOP_HIER_COMPUTE.md -> `results/hierarchy/LOOP_HIER_COMPUTE.json` | script output |
| `LOOP_HIER_COMPUTE.md` | SUPPORTING | RIcat top docs2 tex2 repinv2 code3 | 2026-09-03 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `LOOP_HIER_PARITY.md` | GENERATED-OUTPUT | RIcat mem top docs2 tex2 repinv3 code | 2026-09-03 | move to results/hierarchy/ | script output (run_loop_hier.sh) |
| `LOOP_RANK.json` | REGISTERED-LIVE | code dcode2 | 2026-09-27 | move to results/rank/LOOP_RANK/ | registered LOOP_RANK |
| `LOOP_RANK.md` | CITED | RIcat DR docs5 paper7 code | 2026-09-27 | move to results/rank/LOOP_RANK/ | cited; registered LOOP_RANK |
| `LOOP_RANK_ANALYSIS.txt` | REGISTERED-LIVE | top paper code | 2026-09-27 | move to results/rank/LOOP_RANK/ | registered LOOP_RANK |
| `LOOP_RANK_E1800_P1.json` | REGISTERED-LIVE | code2 dcode | 2026-09-29 | move to results/rank/LOOP_RANK_E1800/ | registered LOOP_RANK_E1800 |
| `LOOP_RANK_E1800_P1.md` | REGISTERED-LIVE | RIcat top docs2 paper3 code | 2026-09-29 | move to results/rank/LOOP_RANK_E1800/ | registered LOOP_RANK_E1800 |
| `LOOP_RANK_E1800_P1_ANALYSIS.txt` | CITED | DR top paper code | 2026-09-29 | move to results/rank/LOOP_RANK_E1800/ | cited; registered LOOP_RANK_E1800 |
| `LOOP_RANK_E1800_P1_RESULTS.md` | CITED | C-cit RI RIcat WTS mem2 top3 docs3 paper6 | 2026-09-30 | move to results/rank/LOOP_RANK_E1800/ | cited; registered LOOP_RANK_E1800 |
| `LOOP_RANK_E1800_PREREG.md` | REGISTERED-LIVE | RI RIcat top paper code3 | 2026-09-28 | move to results/rank/LOOP_RANK_E1800/ | registered LOOP_RANK_E1800 |
| `LOOP_RANK_GEOMETRY.md` | REGISTERED-LIVE | RIcat paper code | 2026-09-27 | move to results/rank/LOOP_RANK/ | registered LOOP_RANK |
| `LOOP_RANK_PREREG.md` | REGISTERED-LIVE | RI RIcat top paper tex code3 dcode3 | 2026-09-26 | move to results/rank/LOOP_RANK/ | registered LOOP_RANK |
| `LOOP_RANK_RESULTS.md` | CITED | C-cit RI RIcat DR WTS mem2 top docs3 paper4 tex dcode2 | 2026-09-27 | move to results/rank/LOOP_RANK/ | cited; registered LOOP_RANK |
| `LOOP_RANK_STRATA.json` | REGISTERED-LIVE | paper2 code | 2026-09-27 | move to results/rank/LOOP_RANK/ | registered LOOP_RANK |
| `LOOP_SAMPLED.json` | GENERATED-OUTPUT | repinv | 2026-09-19 | move with LOOP_SAMPLED.md -> `results/recurrence/LOOP_SAMPLED.json` | script output |
| `LOOP_SAMPLED.md` | GENERATED-OUTPUT | RI RIcat docs2 paper tex repinv2 code | 2026-09-01 | move to results/recurrence/ | script output (run_loop_sampled.sh) |
| `LOOPED_L1.json` | GENERATED-OUTPUT | code | 2026-09-19 | archive with LOOPED_PILOT.md -> `archive/withdrawn/recurrence/LOOPED_L1.json` | script output |
| `LOOPED_L1.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-09-19 | archive with LOOPED_PILOT.md -> `archive/withdrawn/recurrence/LOOPED_L1.md` | script output (run_looped_pilot.sh) |
| `LOOPED_L4.json` | GENERATED-OUTPUT | code | 2026-09-19 | archive with LOOPED_PILOT.md -> `archive/withdrawn/recurrence/LOOPED_L4.json` | script output |
| `LOOPED_L4.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-09-19 | archive with LOOPED_PILOT.md -> `archive/withdrawn/recurrence/LOOPED_L4.md` | script output (run_looped_pilot.sh) |
| `LOOPED_Loop4.json` | GENERATED-OUTPUT | code | 2026-09-19 | archive with LOOPED_PILOT.md -> `archive/withdrawn/recurrence/LOOPED_Loop4.json` | script output |
| `LOOPED_Loop4.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-09-19 | archive with LOOPED_PILOT.md -> `archive/withdrawn/recurrence/LOOPED_Loop4.md` | script output (run_looped_pilot.sh) |
| `LOOPED_PILOT.md` | WITHDRAWN/VOID | C-wd RIcat top3 docs2 paper tex repinv3 code | 2026-09-19 | archive to archive/withdrawn/recurrence/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `LRSWEEP_ANALYSIS.txt` | REGISTERED-LIVE | top2 code | 2026-10-17 | move to results/sign/LRSWEEP/ | registered LRSWEEP |
| `LRSWEEP_PREREG.md` | REGISTERED-LIVE | top2 docs code2 dcode | 2026-10-16 | move to results/sign/LRSWEEP/ | registered LRSWEEP |
| `LRSWEEP_RESULTS.md` | CITED | C-cit DR CM mem top2 paper4 code3 | 2026-10-17 | move to results/sign/LRSWEEP/ | cited; registered LRSWEEP |
| `LRSWEEP_VERDICTS.json` | REGISTERED-LIVE | top2 code2 | 2026-10-17 | move to results/sign/LRSWEEP/ | registered LRSWEEP |
| `MAESTRO_PLAN.md` | SUPPORTING | RIcat top docs | 2026-09-23 | archive to archive/reports_old/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MAGONLY_PREREG.md` | REGISTERED-LIVE | RIcat top repinv code | 2026-09-10 | move to results/em_wm/MAGONLY/ | registered MAGONLY |
| `MAGONLY_RESULTS.md` | REGISTERED-LIVE | RI RIcat top6 docs3 tex repinv2 code | 2026-09-10 | move to results/em_wm/MAGONLY/ | registered MAGONLY |
| `MAP_QUERY_GATES.json` | GENERATED-OUTPUT | code | 2026-08-09 | move with MAP_QUERY_GATES.md -> `results/hierarchy/MAP_QUERY_GATES.json` | script output |
| `MAP_QUERY_GATES.md` | GENERATED-OUTPUT | RI RIcat top repinv code2 arch | 2026-08-09 | move to results/hierarchy/ | script output (run_map_query.sh, validate_map_query.py) |
| `MAP_QUERY_RESULTS.md` | SUPPORTING | RI RIcat top repinv code2 arch | 2026-08-09 | move to results/hierarchy/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MAPPOPE_PAIR_ANALYSIS.txt` | REGISTERED-LIVE | top code | 2026-10-04 | move to results/pope/MAPPOPE_PAIR/ | registered MAPPOPE_PAIR |
| `MAPPOPE_PAIR_PREREG.md` | REGISTERED-LIVE | RI RIcat top2 code6 | 2026-10-04 | move to results/pope/MAPPOPE_PAIR/ | registered MAPPOPE_PAIR |
| `MAPPOPE_PAIR_R2.json` | REGISTERED-LIVE | top code2 dcode3 | 2026-10-04 | move to results/pope/MAPPOPE_PAIR/ | registered MAPPOPE_PAIR |
| `MAPPOPE_PAIR_R2.md` | REGISTERED-LIVE | RIcat code | 2026-10-04 | move to results/pope/MAPPOPE_PAIR/ | registered MAPPOPE_PAIR |
| `MAPPOPE_PAIR_R4.json` | REGISTERED-LIVE | code2 dcode | 2026-10-04 | move to results/pope/MAPPOPE_PAIR/ | registered MAPPOPE_PAIR |
| `MAPPOPE_PAIR_R4.md` | REGISTERED-LIVE | RIcat code | 2026-10-04 | move to results/pope/MAPPOPE_PAIR/ | registered MAPPOPE_PAIR |
| `MAPPOPE_PAIR_RESULTS.md` | CITED | C-cit RI RIcat DR mem2 top3 docs3 paper4 | 2026-10-11 | move to results/pope/MAPPOPE_PAIR/ | cited; registered MAPPOPE_PAIR; banner: CORRECTED/AUDIT block (live) |
| `MAPPOPE_R4.json` | REGISTERED-LIVE | -- | 2026-09-07 | move to results/pope/MAPPOPE_R4/ | registered MAPPOPE_R4 |
| `MAPPOPE_R4.md` | REGISTERED-LIVE | RIcat top docs repinv | 2026-09-07 | move to results/pope/MAPPOPE_R4/ | registered MAPPOPE_R4 |
| `MAPPOPE_R4_PREREG.md` | REGISTERED-LIVE | RIcat top repinv code | 2026-09-07 | move to results/pope/MAPPOPE_R4/ | registered MAPPOPE_R4 |
| `MAPPOPE_R4_RESULTS.md` | REGISTERED-LIVE | RIcat mem top5 docs4 tex repinv4 | 2026-09-07 | move to results/pope/MAPPOPE_R4/ | registered MAPPOPE_R4 |
| `MAPPOPE_VS_POPE.md` | ORPHAN | RIcat repinv | 2026-09-09 | archive to archive/orphan/pope | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `MATCH_GATES_128_16.json` | GENERATED-OUTPUT | -- | 2026-08-15 | move with MATCH_GATES_128_16.md -> `results/navigation/match_query/MATCH_GATES_128_16.json` | script output |
| `MATCH_GATES_128_16.md` | SUPPORTING | RIcat tex repinv3 arch | 2026-08-15 | move to results/navigation/match_query/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MATCH_GATES_64_16.json` | GENERATED-OUTPUT | -- | 2026-08-15 | archive with MATCH_GATES_64_16.md -> `archive/orphan/navigation/match_query/MATCH_GATES_64_16.json` | script output |
| `MATCH_GATES_64_16.md` | ORPHAN | RIcat repinv arch | 2026-08-15 | archive to archive/orphan/navigation/match_query | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `MATCH_GATES_64_4.json` | GENERATED-OUTPUT | -- | 2026-08-15 | move with MATCH_GATES_64_4.md -> `results/navigation/match_query/MATCH_GATES_64_4.json` | script output |
| `MATCH_GATES_64_4.md` | SUPPORTING | RIcat tex repinv2 arch | 2026-08-15 | move to results/navigation/match_query/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MATCH_QUERY_EM.md` | GENERATED-OUTPUT | RIcat top docs2 tex2 repinv4 code2 arch | 2026-08-15 | move to results/em_wm/ | script output (run_match_query_em.sh) |
| `MATCH_QUERY_GATES.json` | GENERATED-OUTPUT | code | 2026-08-09 | move with MATCH_QUERY_GATES.md -> `results/navigation/match_query/MATCH_QUERY_GATES.json` | script output |
| `MATCH_QUERY_GATES.md` | GENERATED-OUTPUT | RIcat top2 tex repinv5 code3 arch | 2026-08-16 | move to results/navigation/match_query/ | script output (run_match_query.sh, validate_match_query.py); banner: CORRECTED/AUDIT block (live) |
| `MATCH_QUERY_GATES_P010.json` | GENERATED-OUTPUT | -- | 2026-09-19 | move with MATCH_QUERY_GATES_P010.md -> `results/navigation/match_query/MATCH_QUERY_GATES_P010.json` | script output |
| `MATCH_QUERY_GATES_P010.md` | SUPPORTING | RIcat tex repinv4 | 2026-09-03 | move to results/navigation/match_query/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MATCH_QUERY_LONGQ.json` | GENERATED-OUTPUT | paper code2 | 2026-08-15 | move with MATCH_QUERY_LONGQ.md -> `results/navigation/match_query/MATCH_QUERY_LONGQ.json` | script output |
| `MATCH_QUERY_LONGQ.md` | GENERATED-OUTPUT | RIcat top docs tex repinv2 code2 arch | 2026-08-15 | move to results/navigation/match_query/ | script output (eval_match_longq.py, run_match_scale.sh) |
| `MATCH_QUERY_NOISE_ABLATION.json` | GENERATED-OUTPUT | -- | 2026-09-03 | move with MATCH_QUERY_NOISE_ABLATION.md -> `results/navigation/match_query/MATCH_QUERY_NOISE_ABLATION.json` | script output |
| `MATCH_QUERY_NOISE_ABLATION.md` | SUPPORTING | RIcat tex repinv2 code | 2026-09-03 | move to results/navigation/match_query/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MATCH_QUERY_RESULTS.md` | CITED | C-cit RI RIcat WTS top3 docs paper tex repinv5 code arch | 2026-09-07 | move to results/navigation/match_query/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `MATCH_QUERY_SCALE.md` | CITED | C-cit RI RIcat CM WTS top2 docs7 paper2 tex2 repinv4 code arch | 2026-08-16 | move to results/navigation/match_query/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `MECHANISM_ANALYSIS.txt` | REGISTERED-LIVE | top paper4 code | 2026-10-14 | move to results/sign/MECHANISM/ | registered MECHANISM |
| `MECHANISM_PREREG.md` | REGISTERED-LIVE | top docs paper4 code3 | 2026-10-14 | move to results/sign/MECHANISM/ | registered MECHANISM |
| `MECHANISM_RESULTS.md` | CITED | C-cit DR CM mem top3 docs paper17 code dcode | 2026-10-16 | move to results/sign/MECHANISM/ | cited; registered MECHANISM; banner: CORRECTED/AUDIT block (live) |
| `MECHANISM_VERDICTS.json` | REGISTERED-LIVE | top docs code2 | 2026-10-14 | move to results/sign/MECHANISM/ | registered MECHANISM |
| `MEMORY_MAZE_ANALYSIS.txt` | CITED | RI DR top docs paper dcode | 2026-10-28 | move to results/memory_maze/MEMORY_MAZE/ | cited; registered MEMORY_MAZE |
| `MEMORY_MAZE_GATE.md` | REGISTERED-LIVE | top3 code | 2026-10-26 | move to results/memory_maze/MEMORY_MAZE/ | registered MEMORY_MAZE; older version of MEMORY_MAZE_GATE_V3.md |
| `MEMORY_MAZE_GATE_V2.md` | REGISTERED-LIVE | top2 code | 2026-10-27 | move to results/memory_maze/MEMORY_MAZE/ | registered MEMORY_MAZE; older version of MEMORY_MAZE_GATE_V3.md |
| `MEMORY_MAZE_GATE_V3.md` | REGISTERED-LIVE | RI top docs2 code2 | 2026-10-27 | move to results/memory_maze/MEMORY_MAZE/ | registered MEMORY_MAZE |
| `MEMORY_MAZE_PREREG.md` | CITED | RI DR top docs code10 | 2026-10-27 | move to results/memory_maze/MEMORY_MAZE/ | cited; registered MEMORY_MAZE |
| `MEMORY_MAZE_RESULTS.md` | CITED | C-cit RI DR CM mem docs dcode | 2026-10-28 | move to results/memory_maze/MEMORY_MAZE/ | cited; registered MEMORY_MAZE |
| `MEMORY_MAZE_VERDICTS.json` | REGISTERED-LIVE | RI top docs | 2026-10-28 | move to results/memory_maze/MEMORY_MAZE/ | registered MEMORY_MAZE |
| `MINIGRID_2X2.json` | GENERATED-OUTPUT | -- | 2026-08-21 | archive with MINIGRID_2X2.md -> `archive/orphan/navigation/environments/MINIGRID_2X2.json` | script output |
| `MINIGRID_2X2.md` | ORPHAN | RIcat repinv code2 | 2026-08-21 | archive to archive/orphan/navigation/environments (regenerable: eval_minigrid_2x2.py, model_fixed_omega.py) | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `MINIGRID_2X2X2.json` | GENERATED-OUTPUT | -- | 2026-08-21 | archive with MINIGRID_2X2X2.md -> `archive/superseded/navigation/environments/MINIGRID_2X2X2.json` | script output |
| `MINIGRID_2X2X2.md` | SUPERSEDED | RIcat top repinv code3 dcode2 | 2026-09-06 | archive to archive/superseded/navigation/environments/ | superseded; superseded by MINIGRID_2X2X2_n8.md |
| `MINIGRID_2X2X2_n8.json` | GENERATED-OUTPUT | -- | 2026-08-22 | move with MINIGRID_2X2X2_n8.md -> `results/navigation/environments/MINIGRID_2X2X2_n8.json` | script output |
| `MINIGRID_2X2X2_n8.md` | SUPPORTING | RIcat tex repinv code2 | 2026-08-22 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIGRID_ALLO_8THCELL.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-09-19 | archive to archive/generated/navigation/environments/ (regenerable: run_pope_hier_allo.sh) | script output (run_pope_hier_allo.sh) |
| `MINIGRID_ALLOCENTRIC_2X2X2.md` | GENERATED-OUTPUT | RIcat docs tex2 repinv2 code3 | 2026-08-28 | move to results/navigation/environments/ | script output (run_minigrid_allo.sh) |
| `MINIGRID_ALLOCENTRIC_8CELL.json` | GENERATED-OUTPUT | code | 2026-09-19 | move with MINIGRID_ALLOCENTRIC_8CELL.md -> `results/navigation/environments/MINIGRID_ALLOCENTRIC_8CELL.json` | script output |
| `MINIGRID_ALLOCENTRIC_8CELL.md` | GENERATED-OUTPUT | RIcat top repinv code | 2026-08-28 | move to results/navigation/environments/ | script output (run_minigrid_allo8.sh) |
| `MINIGRID_DK16_RESULTS.md` | SUPPORTING | RIcat top4 repinv code2 arch | 2026-05-01 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIGRID_DOORKEY_CACHED.md` | GENERATED-OUTPUT | RIcat repinv code arch | 2026-05-01 | archive to archive/generated/navigation/environments/ (regenerable: run_minigrid_cached_full.sh) | script output (run_minigrid_cached_full.sh) |
| `MINIGRID_DOORKEY_LONGT.md` | GENERATED-OUTPUT | RIcat top2 docs repinv code arch | 2026-05-01 | move to results/navigation/environments/ | script output (run_minigrid_longt.sh) |
| `MINIGRID_DOORKEY_RESULTS.md` | SUPPORTING | RIcat top repinv code2 arch | 2026-05-01 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIGRID_DOORKEY_ROPE_DIAG.md` | GENERATED-OUTPUT | RIcat docs repinv code arch | 2026-05-01 | move to results/navigation/environments/ | script output (run_minigrid_rope_diag.sh) |
| `MINIGRID_EM.json` | REGISTERED-LIVE | -- | 2026-09-08 | move to results/em_wm/MINIGRID_EM/ | registered MINIGRID_EM |
| `MINIGRID_EM.md` | REGISTERED-LIVE | RIcat top2 docs2 tex repinv5 code2 dcode | 2026-09-08 | move to results/em_wm/MINIGRID_EM/ | registered MINIGRID_EM |
| `MINIGRID_EM_FIX.json` | REGISTERED-LIVE | -- | 2026-09-09 | move to results/em_wm/MINIGRID_EM_FIX/ | registered MINIGRID_EM_FIX |
| `MINIGRID_EM_FIX.md` | REGISTERED-LIVE | RIcat top docs tex repinv4 | 2026-09-11 | move to results/em_wm/MINIGRID_EM_FIX/ | registered MINIGRID_EM_FIX |
| `MINIGRID_EM_FIX_PREREG.md` | REGISTERED-LIVE | RIcat top repinv2 code | 2026-09-08 | move to results/em_wm/MINIGRID_EM_FIX/ | registered MINIGRID_EM_FIX |
| `MINIGRID_EM_PREREG.md` | REGISTERED-LIVE | RIcat top repinv3 code | 2026-09-08 | move to results/em_wm/MINIGRID_EM/ | registered MINIGRID_EM |
| `MINIGRID_FULL_2X2X2.json` | GENERATED-OUTPUT | docs code | 2026-08-23 | move with MINIGRID_FULL_2X2X2.md -> `results/navigation/environments/MINIGRID_FULL_2X2X2.json` | script output |
| `MINIGRID_FULL_2X2X2.md` | SUPPORTING | RI RIcat top docs3 tex repinv3 code2 | 2026-08-23 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIGRID_MEMORY_RESULTS.md` | SUPPORTING | RIcat top3 repinv code2 arch | 2026-05-02 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIGRID_REPRO_CONTROL.json` | GENERATED-OUTPUT | -- | 2026-09-19 | move with MINIGRID_REPRO_CONTROL.md -> `results/navigation/environments/MINIGRID_REPRO_CONTROL.json` | script output |
| `MINIGRID_REPRO_CONTROL.md` | SUPPORTING | RIcat tex repinv2 | 2026-08-28 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIWORLD_ENDPOINTS.md` | SUPPORTING | RIcat docs2 tex2 repinv2 | 2026-09-13 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIWORLD_FIXED_FINDINGS.md` | SUPPORTING | RIcat tex repinv2 | 2026-08-28 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIWORLD_FIXED_RESULTS.md` | SUPERSEDED | RIcat top2 tex2 repinv3 code3 | 2026-08-28 | archive to archive/superseded/navigation/environments/ | superseded; superseded by MINIWORLD_FIXED_RESULTS_T1024.md |
| `MINIWORLD_FIXED_RESULTS_T1024.md` | SUPPORTING | RIcat top repinv | 2026-08-28 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIWORLD_FRESH_ABLATION.md` | SUPPORTING | RIcat top repinv code | 2026-08-28 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIWORLD_FRESH_FINDINGS.md` | ORPHAN | RIcat repinv | 2026-08-28 | archive to archive/orphan/navigation/environments | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `MINIWORLD_FRESH_GATES_ALLO.json` | GENERATED-OUTPUT | -- | 2026-09-19 | archive with MINIWORLD_FRESH_GATES_ALLO.md -> `archive/generated/navigation/environments/MINIWORLD_FRESH_GATES_ALLO.json` | script output |
| `MINIWORLD_FRESH_GATES_ALLO.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_fresh_factorial.sh) | script output (run_mw_fresh_factorial.sh) |
| `MINIWORLD_FRESH_GATES_RAW.json` | GENERATED-OUTPUT | -- | 2026-09-19 | archive with MINIWORLD_FRESH_GATES_RAW.md -> `archive/generated/navigation/environments/MINIWORLD_FRESH_GATES_RAW.json` | script output |
| `MINIWORLD_FRESH_GATES_RAW.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_fresh_factorial.sh) | script output (run_mw_fresh_factorial.sh) |
| `MINIWORLD_FRESH_RESULTS.md` | SUPERSEDED | RIcat tex repinv code | 2026-08-28 | archive to archive/superseded/navigation/environments/ | superseded; superseded by MINIWORLD_FRESH_RESULTS_T1024.md |
| `MINIWORLD_FRESH_RESULTS_T1024.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_fresh_factorial.sh) | script output (run_mw_fresh_factorial.sh) |
| `MINIWORLD_GATE_CONTROL.md` | GENERATED-OUTPUT | RIcat tex repinv2 code | 2026-08-28 | move to results/navigation/environments/ | script output (run_mw_gatectl.sh) |
| `MINIWORLD_GATES_ALLO.json` | GENERATED-OUTPUT | -- | 2026-08-28 | move with MINIWORLD_GATES_ALLO.md -> `results/navigation/environments/MINIWORLD_GATES_ALLO.json` | script output |
| `MINIWORLD_GATES_ALLO.md` | SUPPORTING | RIcat top2 repinv | 2026-08-28 | move to results/navigation/environments/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MINIWORLD_GATES_FIXED.json` | GENERATED-OUTPUT | -- | 2026-08-28 | archive with MINIWORLD_GATES_FIXED.md -> `archive/orphan/navigation/environments/MINIWORLD_GATES_FIXED.json` | script output |
| `MINIWORLD_GATES_FIXED.md` | ORPHAN | RIcat repinv | 2026-08-28 | archive to archive/orphan/navigation/environments | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `MINIWORLD_GRID16_GATES.json` | GENERATED-OUTPUT | -- | 2026-09-19 | archive with MINIWORLD_GRID16_GATES.md -> `archive/generated/navigation/environments/MINIWORLD_GRID16_GATES.json` | script output |
| `MINIWORLD_GRID16_GATES.md` | GENERATED-OUTPUT | RIcat repinv | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_grid_sweep.sh) | script output (run_mw_grid_sweep.sh) |
| `MINIWORLD_GRID24_GATES.json` | GENERATED-OUTPUT | -- | 2026-09-19 | archive with MINIWORLD_GRID24_GATES.md -> `archive/generated/navigation/environments/MINIWORLD_GRID24_GATES.json` | script output |
| `MINIWORLD_GRID24_GATES.md` | GENERATED-OUTPUT | RIcat repinv | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_grid_sweep.sh) | script output (run_mw_grid_sweep.sh) |
| `MINIWORLD_GRID32_GATES.json` | GENERATED-OUTPUT | -- | 2026-09-19 | archive with MINIWORLD_GRID32_GATES.md -> `archive/generated/navigation/environments/MINIWORLD_GRID32_GATES.json` | script output |
| `MINIWORLD_GRID32_GATES.md` | GENERATED-OUTPUT | RIcat repinv | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_grid_sweep.sh) | script output (run_mw_grid_sweep.sh) |
| `MINIWORLD_GRID_SWEEP.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_grid_sweep.sh) | script output (run_mw_grid_sweep.sh) |
| `MINIWORLD_GRID_SWEEP_HIER.md` | GENERATED-OUTPUT | RIcat top repinv code | 2026-08-28 | move to results/navigation/environments/ | script output (run_mw_grid_sweep_hier.sh) |
| `MINIWORLD_HIER_ABLATION.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_hier_ablation.sh) | script output (run_mw_hier_ablation.sh) |
| `MINIWORLD_ORACLE_ABLATION.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_oracle.sh) | script output (run_mw_oracle.sh) |
| `MINIWORLD_ORACLE_GATES.json` | GENERATED-OUTPUT | -- | 2026-09-19 | archive with MINIWORLD_ORACLE_GATES.md -> `archive/generated/navigation/environments/MINIWORLD_ORACLE_GATES.json` | script output |
| `MINIWORLD_ORACLE_GATES.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_oracle.sh) | script output (run_mw_oracle.sh) |
| `MINIWORLD_ORACLE_RESULTS_T1024.md` | GENERATED-OUTPUT | RIcat repinv | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_oracle.sh) | script output (run_mw_oracle.sh) |
| `MINIWORLD_ORACLE_RESULTS_T512.md` | SUPERSEDED | RIcat tex repinv | 2026-08-28 | archive to archive/superseded/navigation/environments/ | superseded; superseded by MINIWORLD_ORACLE_RESULTS_T1024.md |
| `MINIWORLD_PROBE3.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_probe3.sh) | script output (run_mw_probe3.sh) |
| `MINIWORLD_RESULTS.md` | ORPHAN | RIcat repinv | 2026-08-24 | archive to archive/orphan/navigation/environments | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `MINIWORLD_SROPE_COMPONENTS.md` | GENERATED-OUTPUT | RIcat repinv code2 | 2026-08-28 | archive to archive/generated/navigation/environments/ (regenerable: run_mw_probe3_seeds.sh, run_mw_srope_components.sh) | script output (run_mw_probe3_seeds.sh, run_mw_srope_components.sh) |
| `MINIWORLD_TODO.md` | ORPHAN | RIcat repinv | 2026-08-28 | archive to archive/reports_old/ | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `MONOTONE_PREREG.md` | REGISTERED-LIVE | RIcat top2 repinv2 code4 dcode | 2026-09-13 | move to results/sign/MONOTONE/ | registered MONOTONE |
| `MONOTONE_RAW.json` | REGISTERED-LIVE | top docs code2 | 2026-09-13 | move to results/sign/MONOTONE/ | registered MONOTONE |
| `MONOTONE_RAW.md` | REGISTERED-LIVE | RIcat top code | 2026-09-13 | move to results/sign/MONOTONE/ | registered MONOTONE |
| `MONOTONE_RESULTS.md` | REGISTERED-LIVE | RIcat top3 docs tex repinv2 | 2026-09-13 | move to results/sign/MONOTONE/ | registered MONOTONE |
| `MQ_NOISE_2X2.json` | GENERATED-OUTPUT | repinv | 2026-09-02 | move with MQ_NOISE_2X2.md -> `results/recurrence/MQ_NOISE_2X2.json` | script output |
| `MQ_NOISE_2X2.md` | CITED | C-cit RIcat mem2 top2 docs6 paper tex2 repinv5 code oth | 2026-09-02 | move to results/recurrence/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; older version of MQ_NOISE_2X2_C2.md |
| `MQ_NOISE_2X2_C2.json` | GENERATED-OUTPUT | repinv | 2026-09-03 | move with MQ_NOISE_2X2_C2.md -> `results/recurrence/MQ_NOISE_2X2_C2.json` | script output |
| `MQ_NOISE_2X2_C2.md` | GENERATED-OUTPUT | RIcat mem top2 docs3 tex2 repinv4 code oth | 2026-09-03 | move to results/recurrence/ | script output (run_mq_noise_c2.sh) |
| `MQ_RANK_2X2.md` | CITED | C-cit RI RIcat top docs3 paper2 tex repinv4 code | 2026-09-05 | move to results/rank/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `MQAR_PREREG.md` | REGISTERED-LIVE | RIcat top repinv code2 | 2026-09-08 | move to results/sign/MQAR/ | registered MQAR |
| `MQAR_RESULTS.md` | REGISTERED-LIVE | RIcat docs tex repinv2 | 2026-09-08 | move to results/sign/MQAR/ | registered MQAR |
| `MULTICLASS_MULTISEED_RESULTS.md` | SUPPORTING | RIcat top3 repinv2 code arch | 2026-05-15 | archive to archive/legacy_level15/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `MULTICLASS_RESULTS.md` | WITHDRAWN/VOID | RIcat top3 repinv code arch | 2026-05-13 | delete (identical copy already in archive_stale/MULTICLASS_RESULTS.md) | withdrawn or void; copy in archive_stale/MULTICLASS_RESULTS.md (identical) |
| `MULTISEED_FOLLOWUP.md` | WITHDRAWN/VOID | RIcat top repinv code arch | 2026-05-08 | delete (identical copy already in archive_stale/MULTISEED_FOLLOWUP.md) | withdrawn or void; copy in archive_stale/MULTISEED_FOLLOWUP.md (identical) |
| `MULTISEED_FOLLOWUP_RESULTS.md` | SUPPORTING | RIcat top repinv code arch | 2026-05-14 | archive to archive/legacy_level15/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `N3_AUDIT.md` | WITHDRAWN/VOID | C-wd RI RIcat top4 docs3 tex repinv8 | 2026-09-07 | archive to archive/withdrawn/navigation/torus/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `N5_PREREG.md` | REGISTERED-LIVE | RIcat top repinv code | 2026-09-09 | move to results/em_wm/N5/ | registered N5 |
| `N5_RESULTS.md` | REGISTERED-LIVE | RIcat top4 tex repinv3 code | 2026-09-10 | move to results/em_wm/N5/ | registered N5; banner: CORRECTED/AUDIT block (live) |
| `ND_GATES.md` | SUPPORTING | RIcat tex repinv3 code oth | 2026-09-04 | move to results/rank/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `NOBYPASS_RESULTS.md` | WITHDRAWN/VOID | RIcat top docs repinv code arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows); copy in archive_stale/NOBYPASS_RESULTS.md (differs) |
| `NODROP_PARETO_RESULTS.md` | WITHDRAWN/VOID | RIcat mem top4 docs tex repinv code2 arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows) |
| `NOISE_CLEAN_REVALIDATION.md` | SUPPORTING | RIcat top16 docs tex repinv3 arch32 | 2026-07-16 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `NOISE_REFINE.json` | GENERATED-OUTPUT | docs repinv | 2026-09-19 | move with NOISE_REFINE.md -> `archive/legacy_level15/NOISE_REFINE.json` | script output |
| `NOISE_REFINE.md` | CITED | C-cit RIcat docs5 tex repinv4 code | 2026-08-31 | move to archive/legacy_level15/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `NOLEAK_PREREG.md` | REGISTERED-LIVE | RIcat top2 repinv code3 dcode | 2026-09-11 | move to results/em_wm/NOLEAK/ | registered NOLEAK |
| `NOLEAK_RESULTS.md` | CITED | RIcat DR mem top3 docs3 paper tex repinv3 | 2026-09-11 | move to results/em_wm/NOLEAK/ | cited; registered NOLEAK |
| `NUMBERLINE_RESULTS.md` | SUPPORTING | RIcat top repinv2 code arch | 2026-05-22 | archive to archive/legacy_level15/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `OMEGA_RESCALE_clean.md` | ORPHAN | RIcat repinv3 arch | 2026-04-24 | archive to archive/orphan/navigation/torus | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `OOD_GRID_RESULTS.md` | WITHDRAWN/VOID | RIcat top repinv code2 arch | 2026-05-11 | delete (identical copy already in archive_stale/OOD_GRID_RESULTS.md) | withdrawn or void; copy in archive_stale/OOD_GRID_RESULTS.md (identical) |
| `PAIRCONST_PREREG.md` | REGISTERED-LIVE | RIcat top3 repinv code2 | 2026-09-12 | move to results/em_wm/PAIRCONST/ | registered PAIRCONST |
| `PAIRCONST_RESULTS.md` | REGISTERED-LIVE | RIcat top5 docs tex repinv3 | 2026-09-12 | move to results/em_wm/PAIRCONST/ | registered PAIRCONST |
| `PAIRORIGIN_PREREG.md` | REGISTERED-LIVE | RIcat top repinv code3 dcode | 2026-09-11 | move to results/em_wm/PAIRORIGIN/ | registered PAIRORIGIN |
| `PAIRORIGIN_RESULTS.md` | REGISTERED-LIVE | RIcat top5 docs tex repinv3 code | 2026-09-12 | move to results/em_wm/PAIRORIGIN/ | registered PAIRORIGIN |
| `PAIRSPLIT_PREREG.md` | REGISTERED-LIVE | RIcat top repinv code2 | 2026-09-12 | move to results/em_wm/PAIRSPLIT/ | registered PAIRSPLIT |
| `PAIRSPLIT_RESULTS.md` | CITED | C-cit RI RIcat top3 docs5 paper tex repinv4 dcode | 2026-09-12 | move to results/em_wm/PAIRSPLIT/ | cited; registered PAIRSPLIT |
| `PAPER2X2_PREREG.md` | REGISTERED-LIVE | RIcat top docs repinv2 code2 | 2026-09-13 | move to results/navigation/torus/PAPER2X2/ | registered PAPER2X2 |
| `PAPER2X2_RAW_CONTRASTS.json` | REGISTERED-LIVE | docs code | 2026-09-13 | move to results/navigation/torus/PAPER2X2/ | registered PAPER2X2 |
| `PAPER2X2_RESULTS.md` | CITED | C-cit C-wd RI RIcat DR CM WTS mem2 top3 docs11 paper6 tex2 repinv2 code2 | 2026-09-13 | move to results/navigation/torus/PAPER2X2/ | cited; registered PAPER2X2; also named in CLAUDE.md Withdrawn (carries the correction) |
| `PAPER_FIG4_EM.md` | GENERATED-OUTPUT | RIcat tex repinv2 code | 2026-09-04 | move to results/em_wm/ | script output (run_em_fig4.sh) |
| `PAPER_FIG4_REPRO.md` | SUPPORTING | RIcat mem top docs3 tex repinv3 code2 oth2 | 2026-09-04 | move to results/rank/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `PAPER_OOD_EXTENDED.json` | GENERATED-OUTPUT | -- | 2026-08-19 | archive with PAPER_OOD_EXTENDED.md -> `archive/superseded/navigation/torus/PAPER_OOD_EXTENDED.json` | script output |
| `PAPER_OOD_EXTENDED.md` | SUPERSEDED | RIcat top repinv | 2026-08-19 | archive to archive/superseded/navigation/torus/ | superseded; superseded by PAPER_OOD_EXTENDED_n8.md |
| `PAPER_OOD_EXTENDED_n8.json` | GENERATED-OUTPUT | top repinv | 2026-08-20 | move with PAPER_OOD_EXTENDED_n8.md -> `results/navigation/torus/PAPER_OOD_EXTENDED_n8.json` | script output |
| `PAPER_OOD_EXTENDED_n8.md` | SUPPORTING | RIcat docs tex2 repinv3 code | 2026-08-20 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `PAPER_OOD_PROTOCOL.json` | GENERATED-OUTPUT | -- | 2026-08-09 | move with PAPER_OOD_PROTOCOL.md -> `results/navigation/torus/PAPER_OOD_PROTOCOL.json` | script output |
| `PAPER_OOD_PROTOCOL.md` | SUPPORTING | RIcat top5 repinv code arch | 2026-08-09 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `PAPER_OOD_RERUN.json` | GENERATED-OUTPUT | code3 | 2026-09-12 | move with PAPER_OOD_RERUN.md -> `results/navigation/torus/PAPER_OOD_RERUN.json` | script output (run_papertask_eval.sh, run_papertask_rerun.sh) |
| `PAPER_OOD_RERUN.md` | GENERATED-OUTPUT | RIcat docs tex repinv4 code2 | 2026-09-12 | move to results/navigation/torus/ | script output (run_papertask_eval.sh, run_papertask_rerun.sh) |
| `PAPER_OOD_WITH_POPE.json` | GENERATED-OUTPUT | -- | 2026-08-18 | move with PAPER_OOD_WITH_POPE.md -> `results/navigation/torus/PAPER_OOD_WITH_POPE.json` | script output |
| `PAPER_OOD_WITH_POPE.md` | SUPPORTING | RIcat tex repinv2 code arch | 2026-08-18 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `PAPER_TASK_ABLATION.json` | GENERATED-OUTPUT | -- | 2026-08-17 | move with PAPER_TASK_ABLATION.md -> `results/navigation/torus/PAPER_TASK_ABLATION.json` | script output |
| `PAPER_TASK_ABLATION.md` | SUPPORTING | RIcat docs tex repinv3 code2 arch | 2026-08-17 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `PAPER_TASK_ACCURACY.json` | GENERATED-OUTPUT | -- | 2026-08-09 | move with PAPER_TASK_ACCURACY.md -> `results/navigation/torus/PAPER_TASK_ACCURACY.json` | script output |
| `PAPER_TASK_ACCURACY.md` | CITED | RIcat DR docs2 paper3 tex repinv4 code arch | 2026-09-11 | move to results/navigation/torus/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; banner: CORRECTED/AUDIT block (live) |
| `PAPER_TASK_FLOORS.json` | GENERATED-OUTPUT | code2 | 2026-09-12 | move with PAPER_TASK_FLOORS.md -> `results/navigation/torus/PAPER_TASK_FLOORS.json` | script output (measure_paper_floor.py) |
| `PAPER_TASK_FLOORS.md` | SUPPORTING | RIcat top4 tex repinv6 code2 | 2026-09-12 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `PAPER_VALIDATION.md` | GENERATED-OUTPUT | RIcat top tex repinv3 code2 arch | 2026-08-09 | move to results/navigation/torus/ | script output (run_paper_validation.sh) |
| `PAPERTASK_PREREG.md` | REGISTERED-LIVE | RIcat top repinv2 code2 | 2026-09-12 | move to results/navigation/torus/PAPERTASK/ | registered PAPERTASK |
| `PAPERTASK_RESULTS.md` | REGISTERED-LIVE | RIcat top3 docs tex repinv5 | 2026-09-12 | move to results/navigation/torus/PAPERTASK/ | registered PAPERTASK |
| `PER_VISIT_clean.md` | GENERATED-OUTPUT | RIcat repinv2 code2 arch | 2026-04-26 | archive to archive/generated/navigation/torus/ (regenerable: make_per_visit_figure.py, master_level15pc.sh) | script output (make_per_visit_figure.py, master_level15pc.sh) |
| `PERCEPTION_EXPERIMENT_PLAN.md` | SUPPORTING | RIcat top docs repinv code | 2026-08-28 | archive to archive/reports_old/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `PERSCALE_OMEGA_RESULTS.md` | SUPPORTING | RIcat top3 repinv3 code arch | 2026-05-17 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `PLANNER_TASK_AUDIT.json` | GENERATED-OUTPUT | -- | 2026-08-15 | archive with PLANNER_TASK_AUDIT.md -> `archive/withdrawn/hierarchy/PLANNER_TASK_AUDIT.json` | script output |
| `PLANNER_TASK_AUDIT.md` | WITHDRAWN/VOID | C-wd RIcat top2 docs tex repinv4 code2 arch15 | 2026-08-15 | archive to archive/withdrawn/hierarchy/ | withdrawn or void; named in CLAUDE.md Withdrawn (the record of a withdrawn claim) |
| `POPE_WRAPPING.md` | GENERATED-OUTPUT | RI RIcat docs2 tex repinv3 code | 2026-09-05 | move to results/pope/ | script output (run_popewrap.sh) |
| `POSITION_EFFECT_CONVERGED.md` | GENERATED-OUTPUT | RIcat docs2 tex repinv2 code | 2026-08-28 | move to results/navigation/environments/ | script output (run_vanilla_converge.sh) |
| `PUBLICATION_VENUES.md` | ORPHAN | RIcat repinv arch | 2026-08-10 | archive to archive/reports_old/ | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `R_T_DISTRIBUTION_3WAY.md` | WITHDRAWN/VOID | RIcat top docs repinv code arch | 2026-04-27 | delete (identical copy already in archive_stale/R_T_DISTRIBUTION_3WAY.md) | withdrawn or void; copy in archive_stale/R_T_DISTRIBUTION_3WAY.md (identical) |
| `RANK3.json` | CITED | DR paper2 code dcode | 2026-09-28 | move to results/rank/RANK3/ | cited; registered RANK3 |
| `RANK3.md` | CITED | RIcat DR top2 docs4 paper9 code | 2026-09-28 | move to results/rank/RANK3/ | cited; registered RANK3 |
| `RANK3_ANALYSIS.txt` | CITED | DR top paper4 code dcode | 2026-09-28 | move to results/rank/RANK3/ | cited; registered RANK3 |
| `RANK3_GEOMETRY.md` | REGISTERED-LIVE | RI RIcat top docs2 code | 2026-09-28 | move to results/rank/RANK3/ | registered RANK3 |
| `RANK3_PREREG.md` | REGISTERED-LIVE | RI RIcat top code3 dcode | 2026-09-27 | move to results/rank/RANK3/ | registered RANK3 |
| `RANK3_RESULTS.md` | CITED | C-cit RI RIcat WTS mem2 top2 docs5 paper5 | 2026-09-28 | move to results/rank/RANK3/ | cited; registered RANK3 |
| `RANK3_STRATA.json` | REGISTERED-LIVE | top paper2 code dcode | 2026-09-28 | move to results/rank/RANK3/ | registered RANK3 |
| `RANK_MATCHED.json` | REGISTERED-LIVE | top docs2 code dcode4 | 2026-09-23 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED.md` | REGISTERED-LIVE | RIcat docs5 paper2 | 2026-09-23 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900.json` | REGISTERED-LIVE | docs5 code2 dcode4 | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900.md` | REGISTERED-LIVE | RIcat docs4 dcode | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900_ANALYSIS.txt` | REGISTERED-LIVE | top docs3 dcode | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900_GEOMETRY.md` | REGISTERED-LIVE | RIcat docs4 dcode | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900_STRATA.json` | REGISTERED-LIVE | docs3 code dcode | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900c.json` | REGISTERED-LIVE | docs code dcode | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900c.md` | REGISTERED-LIVE | RIcat | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900c_ANALYSIS.txt` | REGISTERED-LIVE | top docs3 dcode | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900c_GEOMETRY.md` | REGISTERED-LIVE | RIcat | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_e900c_STRATA.json` | REGISTERED-LIVE | docs | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_GEOMETRY.md` | REGISTERED-LIVE | RIcat top docs2 | 2026-09-23 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_PREREG.md` | REGISTERED-LIVE | RIcat mem top2 docs3 paper tex code2 dcode2 | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MATCHED_RESULTS.md` | REGISTERED-LIVE | C-wd RI RIcat mem3 top5 docs7 paper tex dcode2 | 2026-09-24 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED; named in CLAUDE.md Withdrawn: a claim from it is withdrawn, the record stays |
| `RANK_MATCHED_STRATA.json` | REGISTERED-LIVE | top docs2 dcode | 2026-09-23 | move to results/rank/RANK_MATCHED/ | registered RANK_MATCHED |
| `RANK_MI.json` | REGISTERED-LIVE | code3 dcode3 | 2026-09-24 | move to results/rank/RANK_MI/ | registered RANK_MI |
| `RANK_MI.md` | CITED | RIcat DR top22 docs6 paper14 code | 2026-09-24 | move to results/rank/RANK_MI/ | cited; registered RANK_MI |
| `RANK_MI_ANALYSIS.txt` | CITED | DR top paper2 tex code dcode | 2026-09-24 | move to results/rank/RANK_MI/ | cited; registered RANK_MI |
| `RANK_MI_GEOMETRY.md` | REGISTERED-LIVE | RIcat top code | 2026-09-24 | move to results/rank/RANK_MI/ | registered RANK_MI |
| `RANK_MI_PREREG.md` | REGISTERED-LIVE | RIcat top code3 | 2026-09-24 | move to results/rank/RANK_MI/ | registered RANK_MI |
| `RANK_MI_RESULTS.md` | CITED | C-cit RI RIcat mem top7 docs paper3 tex repinv code dcode2 | 2026-09-25 | move to results/rank/RANK_MI/ | cited; registered RANK_MI |
| `RANK_MI_STRATA.json` | REGISTERED-LIVE | paper3 code2 dcode | 2026-09-24 | move to results/rank/RANK_MI/ | registered RANK_MI |
| `RANK_ND.json` | REGISTERED-LIVE | docs code2 dcode2 | 2026-10-01 | move to results/rank/RANK_ND/ | registered RANK_ND |
| `RANK_ND.md` | CITED | RI RIcat DR WTS top6 docs3 paper6 code | 2026-10-01 | move to results/rank/RANK_ND/ | cited; registered RANK_ND |
| `RANK_ND_ANALYSIS.txt` | REGISTERED-LIVE | top paper code dcode | 2026-10-01 | move to results/rank/RANK_ND/ | registered RANK_ND |
| `RANK_ND_PREREG.md` | REGISTERED-LIVE | RI RIcat top code3 | 2026-09-30 | move to results/rank/RANK_ND/ | registered RANK_ND |
| `RANK_ND_RESULTS.md` | CITED | C-cit RI RIcat DR WTS mem2 top4 docs4 paper5 | 2026-10-01 | move to results/rank/RANK_ND/ | cited; registered RANK_ND |
| `RANK_ND_SECONDARY.json` | REGISTERED-LIVE | code dcode | 2026-10-01 | move to results/rank/RANK_ND/ | registered RANK_ND |
| `RANK_ND_SECONDARY.txt` | REGISTERED-LIVE | top | 2026-10-01 | move to results/rank/RANK_ND/ | registered RANK_ND |
| `RANK_NOWRAP_PREREG.md` | REGISTERED-LIVE | RIcat mem docs paper code3 dcode3 | 2026-10-08 | move to results/rank/RANK_NOWRAP/ | registered RANK_NOWRAP; pre-registration with no results file (unrun / stopped / postponed) |
| `RANK_PERHEAD_PILOT.json` | REGISTERED-LIVE | code2 | 2026-09-24 | move to results/rank/RANK_PERHEAD/ | registered RANK_PERHEAD |
| `RANK_PERHEAD_PILOT.md` | REGISTERED-LIVE | RIcat docs code | 2026-09-24 | move to results/rank/RANK_PERHEAD/ | registered RANK_PERHEAD |
| `RANK_PERHEAD_PILOT_ANALYSIS.txt` | REGISTERED-LIVE | top docs3 code dcode | 2026-09-24 | move to results/rank/RANK_PERHEAD/ | registered RANK_PERHEAD |
| `RANK_PERHEAD_PILOT_GEOMETRY.md` | REGISTERED-LIVE | RIcat docs5 code dcode | 2026-09-24 | move to results/rank/RANK_PERHEAD/ | registered RANK_PERHEAD |
| `RANK_PERHEAD_PILOT_RESULTS.md` | REGISTERED-LIVE | RIcat top | 2026-09-24 | move to results/rank/RANK_PERHEAD/ | registered RANK_PERHEAD |
| `RANK_PERHEAD_PILOT_STRATA.json` | REGISTERED-LIVE | docs code2 | 2026-09-24 | move to results/rank/RANK_PERHEAD/ | registered RANK_PERHEAD |
| `RANK_PERHEAD_PREREG.md` | REGISTERED-LIVE | RIcat top code3 | 2026-09-24 | move to results/rank/RANK_PERHEAD/ | registered RANK_PERHEAD |
| `RANK_PROJ_ANALYSIS.txt` | CITED | DR top docs3 paper3 code2 dcode2 | 2026-09-24 | move to results/rank/RANK_PROJ/ | cited; registered RANK_PROJ |
| `RANK_PROJ_FROZEN.json` | REGISTERED-LIVE | code dcode | 2026-09-24 | move to results/rank/RANK_PROJ/ | registered RANK_PROJ |
| `RANK_PROJ_FROZEN.md` | REGISTERED-LIVE | C-wd RIcat mem2 docs code | 2026-09-24 | move to results/rank/RANK_PROJ/ | registered RANK_PROJ; named in CLAUDE.md Withdrawn: a claim from it is withdrawn, the record stays |
| `RANK_PROJ_PREREG.md` | REGISTERED-LIVE | RIcat top docs code3 | 2026-09-24 | move to results/rank/RANK_PROJ/ | registered RANK_PROJ |
| `RANK_PROJ_RESULTS.md` | CITED | C-cit RI RIcat DR WTS mem top10 docs6 paper5 tex repinv dcode3 | 2026-09-24 | move to results/rank/RANK_PROJ/ | cited; registered RANK_PROJ |
| `RANK_PROJ_TRAIN.json` | REGISTERED-LIVE | docs3 code2 dcode2 | 2026-09-24 | move to results/rank/RANK_PROJ/ | registered RANK_PROJ |
| `RANK_PROJ_TRAIN.md` | REGISTERED-LIVE | RIcat docs3 paper code dcode | 2026-09-24 | move to results/rank/RANK_PROJ/ | registered RANK_PROJ |
| `RANK_PROJ_TRAIN_GEOMETRY.md` | REGISTERED-LIVE | RIcat code | 2026-09-24 | move to results/rank/RANK_PROJ/ | registered RANK_PROJ |
| `RANK_PROJ_TRAIN_STRATA.json` | REGISTERED-LIVE | docs3 code dcode | 2026-09-24 | move to results/rank/RANK_PROJ/ | registered RANK_PROJ |
| `RANK_SEP.json` | CITED | DR paper2 code dcode | 2026-09-26 | move to results/rank/RANK_SEP/ | cited; registered RANK_SEP |
| `RANK_SEP.md` | CITED | RIcat DR top2 docs5 paper10 code | 2026-09-26 | move to results/rank/RANK_SEP/ | cited; registered RANK_SEP |
| `RANK_SEP_ANALYSIS.txt` | CITED | DR top paper4 code dcode2 | 2026-09-26 | move to results/rank/RANK_SEP/ | cited; registered RANK_SEP |
| `RANK_SEP_GEOMETRY.md` | REGISTERED-LIVE | RIcat code | 2026-09-26 | move to results/rank/RANK_SEP/ | registered RANK_SEP |
| `RANK_SEP_PREREG.md` | REGISTERED-LIVE | RI RIcat top tex code3 dcode2 | 2026-09-25 | move to results/rank/RANK_SEP/ | registered RANK_SEP |
| `RANK_SEP_RESULTS.md` | CITED | C-cit C-wd RI RIcat WTS mem2 top6 docs9 paper6 tex dcode2 | 2026-09-26 | move to results/rank/RANK_SEP/ | cited; registered RANK_SEP; also named in CLAUDE.md Withdrawn (carries the correction) |
| `RANK_SEP_STRATA.json` | REGISTERED-LIVE | code | 2026-09-26 | move to results/rank/RANK_SEP/ | registered RANK_SEP |
| `RANK_SWEEP.json` | GENERATED-OUTPUT | top docs3 repinv2 code dcode2 | 2026-09-04 | move with RANK_SWEEP.md -> `results/rank/RANK_SWEEP.json` | script output (run_rank_matched.sh) |
| `RANK_SWEEP.md` | CITED | C-cit RI RIcat mem top5 docs5 paper tex repinv7 code oth2 | 2026-09-25 | move to results/rank/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; banner: CORRECTED/AUDIT block (live) |
| `RANK_SWEEP_STRATA.json` | GENERATED-OUTPUT | top docs3 code | 2026-09-23 | move with RANK_SWEEP.md -> `results/rank/RANK_SWEEP_STRATA.json` | script output (run_rank_matched.sh) |
| `RANK_TESTS_ANALYSIS.txt` | REGISTERED-LIVE | top paper code dcode | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_C.json` | REGISTERED-LIVE | -- | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_C.md` | REGISTERED-LIVE | RIcat | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS; older version of RANK_TESTS_ESCAPE_C_RESCORED.md |
| `RANK_TESTS_ESCAPE_C_RESCORED.json` | REGISTERED-LIVE | -- | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_C_RESCORED.md` | REGISTERED-LIVE | RIcat | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_C_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_E.json` | REGISTERED-LIVE | -- | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_E.md` | REGISTERED-LIVE | RIcat | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS; older version of RANK_TESTS_ESCAPE_E_RESCORED.md |
| `RANK_TESTS_ESCAPE_E_RESCORED.json` | REGISTERED-LIVE | -- | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_E_RESCORED.md` | REGISTERED-LIVE | RIcat | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_E_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_F.json` | REGISTERED-LIVE | -- | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_F.md` | REGISTERED-LIVE | RIcat | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS; older version of RANK_TESTS_ESCAPE_F_RESCORED.md |
| `RANK_TESTS_ESCAPE_F_RESCORED.json` | REGISTERED-LIVE | -- | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_F_RESCORED.md` | REGISTERED-LIVE | RIcat | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_ESCAPE_F_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_PREREG.md` | REGISTERED-LIVE | RI RIcat top code6 dcode3 | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TESTS_RESULTS.md` | CITED | C-cit RI RIcat DR CM mem paper6 | 2026-10-11 | move to results/rank/RANK_TESTS/ | cited; registered RANK_TESTS |
| `RANK_TESTS_VERDICTS.json` | REGISTERED-LIVE | top code2 | 2026-10-11 | move to results/rank/RANK_TESTS/ | registered RANK_TESTS |
| `RANK_TRUNCATION.md` | SUPPORTING | RIcat top tex repinv2 code | 2026-09-04 | move to results/rank/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `RANK_WRAP.json` | REGISTERED-LIVE | top paper code3 dcode | 2026-10-02 | move to results/rank/RANK_WRAP/ | registered RANK_WRAP |
| `RANK_WRAP_AMENDED.txt` | REGISTERED-LIVE | top | 2026-10-02 | move to results/rank/RANK_WRAP/ | registered RANK_WRAP |
| `RANK_WRAP_ANALYSIS.txt` | REGISTERED-LIVE | top paper code dcode | 2026-10-02 | move to results/rank/RANK_WRAP/ | registered RANK_WRAP |
| `RANK_WRAP_PREREG.md` | REGISTERED-LIVE | RI RIcat top code3 | 2026-10-01 | move to results/rank/RANK_WRAP/ | registered RANK_WRAP |
| `RANK_WRAP_RESULTS.md` | CITED | C-cit RI RIcat DR WTS mem2 top docs4 paper5 | 2026-10-02 | move to results/rank/RANK_WRAP/ | cited; registered RANK_WRAP |
| `RANKCLOCK.json` | REGISTERED-LIVE | code2 | 2026-10-23 | move to results/rank/RANKCLOCK/ | registered RANKCLOCK |
| `RANKCLOCK.md` | CITED | DR CM mem top9 docs paper2 code | 2026-10-23 | move to results/rank/RANKCLOCK/ | cited; registered RANKCLOCK |
| `RANKCLOCK_ANALYSIS.txt` | CITED | DR top paper code | 2026-10-23 | move to results/rank/RANKCLOCK/ | cited; registered RANKCLOCK |
| `RANKCLOCK_PREREG.md` | CITED | DR top2 paper code3 | 2026-10-23 | move to results/rank/RANKCLOCK/ | cited; registered RANKCLOCK |
| `RANKCLOCK_RESULTS.md` | CITED | C-cit DR CM mem top docs paper | 2026-10-23 | move to results/rank/RANKCLOCK/ | cited; registered RANKCLOCK |
| `RANKCLOCK_VERDICTS.json` | CITED | DR top docs paper code2 | 2026-10-23 | move to results/rank/RANKCLOCK/ | cited; registered RANKCLOCK |
| `RANKCLOCK_WD0.json` | REGISTERED-LIVE | code2 | 2026-10-23 | move to results/rank/RANKCLOCK/ | registered RANKCLOCK |
| `RANKCLOCK_WD0.md` | REGISTERED-LIVE | top code | 2026-10-23 | move to results/rank/RANKCLOCK/ | registered RANKCLOCK |
| `RANKRECIPE_ANALYSIS.txt` | REGISTERED-LIVE | RI top paper code | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKRECIPE_PREREG.md` | CITED | RI DR top code2 | 2026-10-24 | move to results/rank/RANKRECIPE/ | cited; registered RANKRECIPE |
| `RANKRECIPE_r2_R0.json` | REGISTERED-LIVE | code | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKRECIPE_r2_R0.md` | REGISTERED-LIVE | -- | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKRECIPE_r2_R1.json` | REGISTERED-LIVE | code | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKRECIPE_r2_R1.md` | REGISTERED-LIVE | -- | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKRECIPE_r2_R2.json` | REGISTERED-LIVE | code | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKRECIPE_r2_R2.md` | REGISTERED-LIVE | -- | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKRECIPE_r4_R0.json` | REGISTERED-LIVE | code | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKRECIPE_r4_R0.md` | REGISTERED-LIVE | -- | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKRECIPE_RESULTS.md` | CITED | C-cit RI DR CM mem docs2 paper | 2026-10-24 | move to results/rank/RANKRECIPE/ | cited; registered RANKRECIPE |
| `RANKRECIPE_VERDICTS.json` | REGISTERED-LIVE | top docs paper code2 | 2026-10-24 | move to results/rank/RANKRECIPE/ | registered RANKRECIPE |
| `RANKWD.json` | REGISTERED-LIVE | code2 | 2026-10-23 | move to results/rank/RANKWD/ | registered RANKWD |
| `RANKWD.md` | CITED | C-cit C-oth DR mem top13 docs9 paper2 code | 2026-10-23 | move to results/rank/RANKWD/ | cited; registered RANKWD |
| `RANKWD_ANALYSIS.txt` | REGISTERED-LIVE | top code | 2026-10-23 | move to results/rank/RANKWD/ | registered RANKWD |
| `RANKWD_PREREG.md` | CITED | DR top code2 | 2026-10-23 | move to results/rank/RANKWD/ | cited; registered RANKWD |
| `RANKWD_RESULTS.md` | CITED | C-cit DR mem top docs2 paper | 2026-10-24 | move to results/rank/RANKWD/ | cited; registered RANKWD |
| `RANKWD_VERDICTS.json` | REGISTERED-LIVE | top docs code2 | 2026-10-23 | move to results/rank/RANKWD/ | registered RANKWD |
| `RANKWD_WD0.json` | REGISTERED-LIVE | code2 | 2026-10-23 | move to results/rank/RANKWD/ | registered RANKWD |
| `RANKWD_WD0.md` | REGISTERED-LIVE | top code | 2026-10-23 | move to results/rank/RANKWD/ | registered RANKWD |
| `README.md` | META | C-oth RIcat WTS mem top2 docs5 repinv2 arch | 2026-09-25 | keep at root | project infrastructure; banner: SUPERSEDED |
| `REC_EM_PREREG.md` | REGISTERED-LIVE | RIcat top2 repinv code | 2026-09-09 | move to results/em_wm/REC_EM/ | registered REC_EM |
| `RECENCY_EM_RESULTS.md` | CITED | RI RIcat DR top9 docs2 paper tex repinv6 code dcode | 2026-09-10 | move to results/em_wm/REC_EM/ | cited; registered REC_EM; banner: CORRECTED/AUDIT block (live) |
| `RECENCY_GATE_ABLATION.json` | REGISTERED-LIVE | -- | 2026-09-07 | move to results/sign/RECENCY/ | registered RECENCY |
| `RECENCY_GATE_ABLATION.md` | CITED | C-cit RI RIcat CM top3 docs3 tex repinv3 code2 | 2026-09-07 | move to results/sign/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `RECENCY_GATES.json` | REGISTERED-LIVE | -- | 2026-09-07 | move to results/sign/RECENCY/ | registered RECENCY |
| `RECENCY_GATES.md` | REGISTERED-LIVE | RIcat top repinv code3 | 2026-09-07 | move to results/sign/RECENCY/ | registered RECENCY |
| `RECENCY_GATES_K16SET.md` | SUPPORTING | RIcat top2 repinv2 | 2026-09-11 | move to results/sign/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `RECENCY_GATES_K4SET.md` | SUPPORTING | RIcat top2 tex repinv2 | 2026-09-11 | move to results/sign/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `RECENCY_GATES_K64.json` | REGISTERED-LIVE | -- | 2026-09-19 | move to results/sign/RECENCY/ | registered RECENCY |
| `RECENCY_GATES_K64.md` | SUPPORTING | RIcat top4 tex repinv3 code | 2026-09-07 | move to results/sign/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `RECENCY_H2.json` | REGISTERED-LIVE | -- | 2026-09-07 | move to results/sign/RECENCY/ | registered RECENCY |
| `RECENCY_H2.md` | SUPPORTING | RIcat docs tex repinv3 code | 2026-09-07 | move to results/sign/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `RECENCY_PREREG.md` | REGISTERED-LIVE | RIcat top docs repinv code | 2026-09-07 | move to results/sign/RECENCY/ | registered RECENCY |
| `RECENCY_RESULTS.md` | CITED | C-cit RI RIcat CM mem top10 docs9 tex repinv5 dcode | 2026-09-07 | move to results/sign/RECENCY/ | cited; registered RECENCY |
| `RECENCY_T3_PREREG.md` | WITHDRAWN/VOID | RIcat top code | 2026-09-19 | archive to archive/withdrawn/RECENCY_T3/ | withdrawn or void; registered family whose results file is withdrawn/void (RECENCY_T3_RESULTS.md) |
| `RECENCY_T3_RESULTS.md` | WITHDRAWN/VOID | RIcat docs | 2026-09-20 | archive to archive/withdrawn/RECENCY_T3/ | withdrawn or void; banner: VOID; registered family whose results file is withdrawn/void (RECENCY_T3_RESULTS.md) |
| `RECIPE_CHOICE.json` | GENERATED-OUTPUT | repinv code2 | 2026-09-02 | archive to archive/generated/navigation/torus/ (regenerable: run_mq_noise_2x2.sh) | script output (run_mq_noise_2x2.sh) |
| `RECIPE_POWER.md` | GENERATED-OUTPUT | RIcat top3 docs4 tex repinv3 code dcode | 2026-09-02 | move to results/navigation/torus/ | script output (run_recipe_power.sh) |
| `RECURSIVE_RESULTS.md` | ORPHAN | RIcat repinv arch | 2026-07-22 | archive to archive/orphan/algorithmic | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `REFINE_RESULTS.md` | CITED | C-cit RI RIcat WTS mem2 top2 docs4 paper tex repinv4 code | 2026-08-31 | move to results/recurrence/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `REPORT.md` | SUPERSEDED | RIcat WTS top2 docs5 repinv code3 arch | 2026-05-15 | archive to archive/reports_old/ | superseded; superseded by REPORT_v2.md |
| `REPORT_ADDENDUM.md` | WITHDRAWN/VOID | RIcat top repinv arch | 2026-08-09 | archive to archive/reports_old/ | withdrawn or void; banner: CONTAINS RETRACTED CLAIMS |
| `REPORT_v2.md` | WITHDRAWN/VOID | RIcat repinv arch | 2026-08-09 | archive to archive/reports_old/ | withdrawn or void; banner: CONTAINS RETRACTED CLAIMS |
| `requirements.txt` | META | C-oth top3 docs2 | 2026-04-09 | keep at root | project infrastructure |
| `RESULTS_INDEX.md` | META | C-oth RIcat WTS mem top6 docs10 repinv10 dcode2 | 2026-10-28 | keep at root | project infrastructure |
| `RESULTS_PAPER.md` | WITHDRAWN/VOID | RIcat top3 docs2 tex repinv3 code6 arch | 2026-08-09 | archive to archive/reports_old/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows) |
| `RESULTS_SUMMARY_2026-05-10.md` | WITHDRAWN/VOID | RIcat top2 repinv arch | 2026-08-09 | archive to archive/reports_old/ | withdrawn or void; banner: CONTAINS RETRACTED CLAIMS |
| `REVISIT_2X2.json` | GENERATED-OUTPUT | code | 2026-09-14 | move with REVISIT_2X2.md -> `results/navigation/torus/REVISIT_2X2.json` | script output |
| `REVISIT_2X2.md` | SUPPORTING | RI RIcat docs2 code | 2026-09-14 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `REVISIT_DISTANCE.json` | GENERATED-OUTPUT | -- | 2026-08-17 | move with REVISIT_DISTANCE.md -> `results/navigation/torus/REVISIT_DISTANCE.json` | script output |
| `REVISIT_DISTANCE.md` | SUPPORTING | RI RIcat top3 docs2 tex2 repinv4 code3 arch | 2026-08-17 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `RING_BUDGET_ANALYSIS.txt` | REGISTERED-LIVE | top paper2 code | 2026-10-12 | move to results/sign/RING_BUDGET/ | registered RING_BUDGET |
| `RING_BUDGET_PREREG.md` | REGISTERED-LIVE | top docs paper code2 | 2026-10-12 | move to results/sign/RING_BUDGET/ | registered RING_BUDGET |
| `RING_BUDGET_RESULTS.md` | CITED | DR CM top2 docs paper7 dcode | 2026-10-12 | move to results/sign/RING_BUDGET/ | cited; registered RING_BUDGET |
| `RING_BUDGET_VERDICTS.json` | REGISTERED-LIVE | top code2 | 2026-10-12 | move to results/sign/RING_BUDGET/ | registered RING_BUDGET |
| `RING_DIAL_ANALYSIS.txt` | REGISTERED-LIVE | top docs paper2 code | 2026-10-12 | move to results/sign/RING_DIAL/ | registered RING_DIAL |
| `RING_DIAL_PREREG.md` | REGISTERED-LIVE | top2 paper code3 | 2026-10-12 | move to results/sign/RING_DIAL/ | registered RING_DIAL |
| `RING_DIAL_RESULTS.md` | CITED | CM top4 docs paper7 dcode | 2026-10-12 | move to results/sign/RING_DIAL/ | cited; registered RING_DIAL |
| `RING_DIAL_VERDICTS.json` | CITED | DR top paper5 code2 | 2026-10-12 | move to results/sign/RING_DIAL/ | cited; registered RING_DIAL |
| `RNN_TORUS_ANALYSIS.txt` | REGISTERED-LIVE | RI top2 code dcode | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor.json` | REGISTERED-LIVE | docs | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3300.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3300.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3301.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3301.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3302.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3302.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3303.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3303.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3304.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3304.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3305.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3305.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3306.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3306.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3307.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_OWN_s3307.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_anchor_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_CD.json` | REGISTERED-LIVE | top2 code | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3300.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3300.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3301.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3301.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3302.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3302.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3303.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3303.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3304.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3304.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3305.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3305.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3306.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3306.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3307.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_OWN_s3307.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr1e-03_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3300.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3300.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3301.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3301.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3302.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3302.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3303.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3303.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3304.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3304.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3305.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3305.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3306.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3306.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3307.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_OWN_s3307.md` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_lr3e-03_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `RNN_TORUS_PREREG.md` | CITED | RI DR top docs code7 | 2026-10-26 | move to results/recurrence/RNN_TORUS/ | cited; registered RNN_TORUS |
| `RNN_TORUS_RESULTS.md` | CITED | C-cit RI DR CM mem top2 docs | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | cited; registered RNN_TORUS |
| `RNN_TORUS_VERDICTS.json` | REGISTERED-LIVE | top2 docs code | 2026-10-27 | move to results/recurrence/RNN_TORUS/ | registered RNN_TORUS |
| `ROBUST_ANALYSIS.txt` | CITED | DR top2 paper3 code | 2026-10-16 | move to results/sign/ROBUST/ | cited; registered ROBUST |
| `ROBUST_PREREG.md` | REGISTERED-LIVE | top docs3 paper2 code3 | 2026-10-16 | move to results/sign/ROBUST/ | registered ROBUST |
| `ROBUST_RESULTS.md` | CITED | C-cit DR CM mem top docs paper10 | 2026-10-16 | move to results/sign/ROBUST/ | cited; registered ROBUST |
| `ROBUST_TORUS_AO.json` | REGISTERED-LIVE | -- | 2026-10-16 | move to results/sign/ROBUST/ | registered ROBUST |
| `ROBUST_TORUS_AO.md` | REGISTERED-LIVE | -- | 2026-10-16 | move to results/sign/ROBUST/ | registered ROBUST |
| `ROBUST_TORUS_AO_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-16 | move to results/sign/ROBUST/ | registered ROBUST |
| `ROBUST_TORUS_SO.json` | REGISTERED-LIVE | -- | 2026-10-16 | move to results/sign/ROBUST/ | registered ROBUST |
| `ROBUST_TORUS_SO.md` | REGISTERED-LIVE | -- | 2026-10-16 | move to results/sign/ROBUST/ | registered ROBUST |
| `ROBUST_TORUS_SO_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-16 | move to results/sign/ROBUST/ | registered ROBUST |
| `ROBUST_VERDICTS.json` | REGISTERED-LIVE | top2 code2 | 2026-10-16 | move to results/sign/ROBUST/ | registered ROBUST |
| `ROOMS_GOAL_RESULTS.md` | ORPHAN | RIcat repinv | 2026-09-19 | archive to archive/orphan/hierarchy | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `ROPE_CANONICAL.md` | GENERATED-OUTPUT | RIcat top docs tex repinv2 code2 | 2026-09-04 | move to results/navigation/torus/ | script output (run_rope_canonical.sh) |
| `ROPE_CONVERGE.md` | GENERATED-OUTPUT | RIcat repinv code | 2026-08-28 | archive to archive/generated/navigation/torus/ (regenerable: run_rope_converge.sh) | script output (run_rope_converge.sh) |
| `ROTATE_REDO.json` | GENERATED-OUTPUT | -- | 2026-08-21 | archive to archive/generated/navigation/torus/ | script output |
| `ROUTE_ATTN_RESULTS.md` | ORPHAN | RIcat repinv arch | 2026-07-22 | archive to archive/orphan/hierarchy | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `SAMEBLOCK_COMPILE_CHECK.md` | REGISTERED-LIVE | RIcat top | 2026-09-15 | move to results/recurrence/SAMEBLOCK/ | registered SAMEBLOCK |
| `SAMEBLOCK_PREREG.md` | REGISTERED-LIVE | RIcat top2 code4 dcode | 2026-09-15 | move to results/recurrence/SAMEBLOCK/ | registered SAMEBLOCK |
| `SAMEBLOCK_RAW.md` | REGISTERED-LIVE | RIcat top code | 2026-09-15 | move to results/recurrence/SAMEBLOCK/ | registered SAMEBLOCK |
| `SAMEBLOCK_RESULTS.md` | REGISTERED-LIVE | RI RIcat | 2026-09-15 | move to results/recurrence/SAMEBLOCK/ | registered SAMEBLOCK |
| `SCORE_RANK_ANALYSIS.txt` | REGISTERED-LIVE | top docs code | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SCORE_RANK_PREREG.md` | REGISTERED-LIVE | RI RIcat top paper2 code4 dcode2 | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SCORE_RANK_R2.json` | REGISTERED-LIVE | code2 | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SCORE_RANK_R2.md` | REGISTERED-LIVE | RIcat | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK; older version of SCORE_RANK_RESCORE_R2.md |
| `SCORE_RANK_R4.json` | REGISTERED-LIVE | code2 | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SCORE_RANK_R4.md` | REGISTERED-LIVE | RIcat | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK; older version of SCORE_RANK_RESCORE_R4.md |
| `SCORE_RANK_RESCORE_R2.json` | REGISTERED-LIVE | code2 | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SCORE_RANK_RESCORE_R2.md` | REGISTERED-LIVE | RIcat | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SCORE_RANK_RESCORE_R4.json` | REGISTERED-LIVE | code2 | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SCORE_RANK_RESCORE_R4.md` | REGISTERED-LIVE | RIcat | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SCORE_RANK_RESULTS.md` | CITED | C-cit C-wd RI RIcat DR CM mem3 top docs paper3 | 2026-10-11 | move to results/pope/SCORE_RANK/ | cited; registered SCORE_RANK; banner: CORRECTED/AUDIT block (live); also named in CLAUDE.md Withdrawn (carries the correction) |
| `SCORE_RANK_STRATA_R2.json` | REGISTERED-LIVE | code2 | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SCORE_RANK_STRATA_R4.json` | REGISTERED-LIVE | code2 | 2026-10-05 | move to results/pope/SCORE_RANK/ | registered SCORE_RANK |
| `SEARCH_PREREG.md` | REGISTERED-LIVE | RIcat top docs repinv code6 | 2026-09-11 | move to results/em_wm/SEARCH/ | registered SEARCH |
| `SEARCH_RESULTS.md` | CITED | C-cit C-wd RI RIcat DR WTS mem2 top12 docs4 paper3 tex repinv4 code dcode | 2026-09-11 | move to results/em_wm/SEARCH/ | cited; registered SEARCH; also named in CLAUDE.md Withdrawn (carries the correction) |
| `SELECTIVE_ROPE.md` | CITED | C-cit RIcat mem top3 docs3 tex repinv3 code2 oth2 | 2026-09-05 | move to results/pope/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `SESSION_2026-05-01.md` | SUPPORTING | RIcat docs repinv arch | 2026-05-01 | archive to archive/reports_old/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `SESSION_HIERARCHICAL_CASCADE.md` | ORPHAN | RIcat repinv arch | 2026-07-12 | archive to archive/reports_old/ | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `SIGN_ABLATION.md` | CITED | C-cit RI RIcat CM mem2 top9 docs4 tex repinv6 code2 | 2026-09-06 | move to results/sign/SIGN_ABLATION/ | cited; registered SIGN_ABLATION |
| `SIGN_ABLATION_PREREG.md` | REGISTERED-LIVE | RIcat tex repinv3 code2 | 2026-09-06 | move to results/sign/SIGN_ABLATION/ | registered SIGN_ABLATION |
| `SIGN_MATCHED.json` | REGISTERED-LIVE | paper5 code2 dcode3 | 2026-09-28 | move to results/sign/SIGN_MATCHED/ | registered SIGN_MATCHED |
| `SIGN_MATCHED.md` | CITED | RIcat DR2 top8 docs15 paper21 code | 2026-09-28 | move to results/sign/SIGN_MATCHED/ | cited; registered SIGN_MATCHED |
| `SIGN_MATCHED_ANALYSIS.txt` | CITED | DR top paper3 code | 2026-09-28 | move to results/sign/SIGN_MATCHED/ | cited; registered SIGN_MATCHED |
| `SIGN_MATCHED_PREREG.md` | REGISTERED-LIVE | RI RIcat top paper3 code2 dcode | 2026-09-27 | move to results/sign/SIGN_MATCHED/ | registered SIGN_MATCHED |
| `SIGN_MATCHED_PROBE.json` | REGISTERED-LIVE | -- | 2026-09-30 | move to results/sign/SIGN_MATCHED/ | registered SIGN_MATCHED |
| `SIGN_MATCHED_PROBE.md` | REGISTERED-LIVE | RI RIcat top code | 2026-09-28 | move to results/sign/SIGN_MATCHED/ | registered SIGN_MATCHED |
| `SIGN_MATCHED_PROBE_SIGNED.md` | REGISTERED-LIVE | RIcat top | 2026-09-28 | move to results/sign/SIGN_MATCHED/ | registered SIGN_MATCHED |
| `SIGN_MATCHED_RESULTS.md` | CITED | C-cit RI RIcat CM WTS mem3 top7 docs10 paper12 tex dcode | 2026-10-16 | move to results/sign/SIGN_MATCHED/ | cited; registered SIGN_MATCHED; banner: CORRECTED/AUDIT block (live) |
| `SIGN_MATCHED_STRATA.json` | REGISTERED-LIVE | top paper7 code dcode2 | 2026-09-28 | move to results/sign/SIGN_MATCHED/ | registered SIGN_MATCHED |
| `SIGN_PROBE.json` | GENERATED-OUTPUT | repinv | 2026-09-19 | move with SIGN_PROBE.md -> `results/sign/SIGN_PROBE.json` | script output |
| `SIGN_PROBE.md` | GENERATED-OUTPUT | RIcat mem top2 docs tex repinv3 code | 2026-09-06 | move to results/sign/ | script output (run_sign.sh) |
| `SIGNROUTE_ANALYSIS.txt` | REGISTERED-LIVE | top code | 2026-10-21 | move to results/sign/SIGNROUTE/ | registered SIGNROUTE |
| `SIGNROUTE_PREREG.md` | REGISTERED-LIVE | top code5 | 2026-10-20 | move to results/sign/SIGNROUTE/ | registered SIGNROUTE |
| `SIGNROUTE_RESULTS.md` | CITED | C-cit DR mem top4 docs paper code | 2026-10-21 | move to results/sign/SIGNROUTE/ | cited; registered SIGNROUTE |
| `SIGNROUTE_VERDICTS.json` | REGISTERED-LIVE | top code2 | 2026-10-21 | move to results/sign/SIGNROUTE/ | registered SIGNROUTE |
| `SIGNSEP_ANALYSIS.txt` | REGISTERED-LIVE | RI top2 paper code | 2026-10-24 | move to results/sign/SIGNSEP/ | registered SIGNSEP |
| `SIGNSEP_PREREG.md` | CITED | RI DR top code7 | 2026-10-23 | move to results/sign/SIGNSEP/ | cited; registered SIGNSEP |
| `SIGNSEP_RESULTS.md` | CITED | C-cit RI DR CM mem docs2 paper | 2026-10-24 | move to results/sign/SIGNSEP/ | cited; registered SIGNSEP |
| `SIGNSEP_VERDICTS.json` | REGISTERED-LIVE | top2 docs paper code2 | 2026-10-24 | move to results/sign/SIGNSEP/ | registered SIGNSEP |
| `SPACETIME_HIER_RESULTS.md` | ORPHAN | RIcat repinv arch | 2026-07-22 | archive to archive/orphan/hierarchy | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `SPREAD2_PREREG.md` | REGISTERED-LIVE | RIcat top repinv code2 | 2026-09-12 | move to results/em_wm/SPREAD2/ | registered SPREAD2 |
| `SPREAD2_RESULTS.md` | REGISTERED-LIVE | RIcat top2 docs2 tex repinv3 | 2026-09-12 | move to results/em_wm/SPREAD2/ | registered SPREAD2 |
| `SPREAD_PREREG.md` | REGISTERED-LIVE | RIcat top2 repinv code4 | 2026-09-11 | move to results/em_wm/SPREAD/ | registered SPREAD |
| `SPREAD_RESULTS.md` | CITED | RIcat DR top4 docs paper2 tex repinv3 code | 2026-09-11 | move to results/em_wm/SPREAD/ | cited; registered SPREAD |
| `SROPE_TORUS_ANALYSIS.txt` | REGISTERED-LIVE | top2 | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_MapWM.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_MapWM.md` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_MapWM_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_PREREG.md` | REGISTERED-LIVE | top code3 | 2026-10-20 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_RESULTS.md` | CITED | C-cit DR CM mem docs paper | 2026-10-21 | move to results/pope/SROPE_TORUS/ | cited; registered SROPE_TORUS |
| `SROPE_TORUS_RoPE.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_RoPE.md` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_RoPE_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPE.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPE.md` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPE_NoSiLU.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPE_NoSiLU.md` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPE_NoSiLU_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPE_ST.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPE_ST.md` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPE_ST_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPE_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPEGen.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPEGen.md` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_SRoPEGen_STRATA.json` | REGISTERED-LIVE | -- | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `SROPE_TORUS_VERDICTS.json` | REGISTERED-LIVE | top2 code | 2026-10-21 | move to results/pope/SROPE_TORUS/ | registered SROPE_TORUS |
| `STALE_MINIGRID_ALLOCENTRIC_2X2X2.json` | GENERATED-OUTPUT | -- | 2026-08-28 | archive to archive/generated/navigation/environments/ | script output |
| `STITCH_ATTENTION.json` | GENERATED-OUTPUT | -- | 2026-08-16 | move with STITCH_ATTENTION.md -> `results/hierarchy/STITCH_ATTENTION.json` | script output |
| `STITCH_ATTENTION.md` | CITED | C-cit RI RIcat docs tex repinv3 code2 arch | 2026-08-16 | move to results/hierarchy/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `T1_PREREG.md` | REGISTERED-LIVE | RIcat docs code2 | 2026-09-18 | move to results/pope/T1/ | registered T1 |
| `T1_RESULTS.md` | REGISTERED-LIVE | RIcat mem top docs tex code | 2026-09-20 | move to results/pope/T1/ | registered T1 |
| `T2_PREREG.md` | REGISTERED-LIVE | RIcat top | 2026-09-20 | move to results/pope/T2/ | registered T2 |
| `T2_RESULTS.md` | CITED | C-wd RIcat CM mem2 top7 docs5 tex code2 | 2026-09-20 | move to results/pope/T2/ | cited; registered T2; also named in CLAUDE.md Withdrawn (carries the correction) |
| `T3_PREREG.md` | REGISTERED-LIVE | RIcat top code | 2026-09-18 | move to results/pope/T3/ | registered T3 |
| `T3_RESULTS.md` | REGISTERED-LIVE | RIcat top2 docs tex | 2026-09-20 | move to results/pope/T3/ | registered T3 |
| `T3GEN_PREREG.md` | REGISTERED-LIVE | RIcat top code | 2026-09-19 | move to results/pope/T3GEN/ | registered T3GEN |
| `T3GEN_RESULTS.md` | REGISTERED-LIVE | RIcat mem top5 docs2 tex code3 | 2026-09-20 | move to results/pope/T3GEN/ | registered T3GEN; banner: SUPERSEDED IN PART (live) |
| `TALE_OF_TWO_ALGORITHMS.md` | SUPPORTING | RIcat mem top6 docs tex repinv code | 2026-09-10 | move to results/em_wm/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources); banner: CORRECTED/AUDIT block (live) |
| `TEM_BACKGROUND_BASELINES.md` | GENERATED-OUTPUT | RIcat top4 tex repinv3 code5 arch | 2026-05-15 | archive to archive/legacy_level15/ | script output (run_fixup.sh, run_post_pipeline_analysis.sh) |
| `TEM_CROSSSCALE_DIAGNOSTIC.md` | GENERATED-OUTPUT | RIcat top3 repinv code3 arch2 | 2026-05-15 | archive to archive/legacy_level15/ | script output (run_post_pipeline_analysis.sh) |
| `TEM_NOISE_FFN_RESULTS.md` | WITHDRAWN/VOID | RIcat top tex repinv2 code arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows) |
| `TEM_RECENCY_DIAG.md` | SUPPORTING | RIcat top | 2026-09-14 | move to results/sign/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `TEM_RECENCY_PILOT.md` | SUPERSEDED | RIcat top code | 2026-09-14 | archive to archive/superseded/sign/ | superseded; superseded by TEM_RECENCY_DIAG.md |
| `TEM_RESULTS.md` | WITHDRAWN/VOID | RIcat top2 repinv code4 arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows) |
| `TEM_T_MULTISEED.md` | WITHDRAWN/VOID | RIcat top tex repinv3 code arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows) |
| `TEM_T_RESULTS.md` | WITHDRAWN/VOID | RIcat top2 repinv code3 arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows) |
| `TEXTWORLD_ANALYSIS.txt` | CITED | DR top paper2 code | 2026-09-28 | move to results/language/TEXTWORLD/ | cited; registered TEXTWORLD |
| `TEXTWORLD_PREREG.md` | REGISTERED-LIVE | RI RIcat top2 paper tex code6 dcode | 2026-09-30 | move to results/language/TEXTWORLD/ | registered TEXTWORLD |
| `TEXTWORLD_PROBE.json` | REGISTERED-LIVE | top docs2 paper code2 | 2026-09-28 | move to results/language/TEXTWORLD/ | registered TEXTWORLD |
| `TEXTWORLD_RESULTS.md` | CITED | C-cit RI RIcat DR CM WTS mem2 top6 docs9 paper14 tex dcode | 2026-10-04 | move to results/language/TEXTWORLD/ | cited; registered TEXTWORLD; banner: CORRECTED/AUDIT block (live) |
| `TEXTWORLD_SECONDARY.json` | REGISTERED-LIVE | top paper3 code dcode2 | 2026-09-28 | move to results/language/TEXTWORLD/ | registered TEXTWORLD |
| `TEXTWORLD_SECONDARY.txt` | REGISTERED-LIVE | top paper | 2026-09-28 | move to results/language/TEXTWORLD/ | registered TEXTWORLD |
| `THEORY_KERNEL.md` | CITED | RIcat CM mem3 top6 docs repinv2 code3 dcode | 2026-09-11 | move to results/em_wm/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; banner: CORRECTED/AUDIT block (live) |
| `THEORY_MAPPOPE.md` | SUPPORTING | RIcat mem top5 docs tex code5 | 2026-09-20 | move to results/pope/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `THEORY_NARRATIVE.md` | ORPHAN | RIcat repinv3 | 2026-09-13 | archive to archive/orphan/rank | no reference outside the RESULTS_INDEX catalogue, report inventories, archives or code |
| `THEORY_NUMBERS.md` | SUPPORTING | RIcat top code | 2026-09-19 | move to results/rank/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `THEORY_SEARCH_AND_LENGTH.md` | SUPPORTING | RIcat top2 docs tex repinv3 code2 | 2026-09-12 | move to results/rank/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `TIMING_BENCHMARK.json` | GENERATED-OUTPUT | code2 | 2026-08-16 | move with TIMING_BENCHMARK.md -> `results/navigation/torus/TIMING_BENCHMARK.json` | script output |
| `TIMING_BENCHMARK.md` | CITED | C-cit RI RIcat top2 docs4 tex2 repinv3 code2 arch | 2026-08-16 | move to results/navigation/torus/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map |
| `TINYSTORIES_PILOT.md` | SUPERSEDED | RI RIcat top code dcode | 2026-10-09 | archive to archive/superseded/language/ | superseded; superseded by TINYSTORIES_PILOT_RESULTS.md |
| `TINYSTORIES_PILOT_RESULTS.md` | CITED | C-wd RI RIcat CM mem docs2 paper6 | 2026-10-11 | move to results/language/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; banner: CORRECTED/AUDIT block (live); also named in CLAUDE.md Withdrawn (carries the correction) |
| `TOPOLOGY_RESULTS.md` | SUPPORTING | RIcat top repinv code arch | 2026-05-13 | move to results/navigation/torus/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `TORUS_LRSWEEP_ANALYSIS.txt` | REGISTERED-LIVE | RI top2 paper code | 2026-10-24 | move to results/sign/TORUS_LRSWEEP/ | registered TORUS_LRSWEEP |
| `TORUS_LRSWEEP_PREREG.md` | CITED | RI DR top docs code2 | 2026-10-23 | move to results/sign/TORUS_LRSWEEP/ | cited; registered TORUS_LRSWEEP |
| `TORUS_LRSWEEP_RESULTS.md` | CITED | C-cit RI DR CM mem top docs paper | 2026-10-24 | move to results/sign/TORUS_LRSWEEP/ | cited; registered TORUS_LRSWEEP |
| `TORUS_LRSWEEP_VERDICTS.json` | REGISTERED-LIVE | top2 docs paper code2 | 2026-10-24 | move to results/sign/TORUS_LRSWEEP/ | registered TORUS_LRSWEEP |
| `TORUS_START_ANALYSIS.txt` | REGISTERED-LIVE | top paper4 code | 2026-10-15 | move to results/sign/TORUS_START/ | registered TORUS_START |
| `TORUS_START_PREREG.md` | REGISTERED-LIVE | top docs paper2 code4 | 2026-10-14 | move to results/sign/TORUS_START/ | registered TORUS_START |
| `TORUS_START_RESULTS.md` | CITED | C-cit DR CM mem top paper6 dcode | 2026-10-15 | move to results/sign/TORUS_START/ | cited; registered TORUS_START |
| `TORUS_START_VERDICTS.json` | REGISTERED-LIVE | top code2 | 2026-10-15 | move to results/sign/TORUS_START/ | registered TORUS_START |
| `TORUS_T3_RESULTS.json` | GENERATED-OUTPUT | -- | 2026-09-19 | move with TORUS_T3_RESULTS.md -> `results/pope/TORUS_T3_RESULTS.json` | script output |
| `TORUS_T3_RESULTS.md` | SUPPORTING | RIcat docs code | 2026-09-19 | move to results/pope/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `TW_AMBIG_PREREG.md` | REGISTERED-LIVE | RIcat mem code6 dcode3 | 2026-10-08 | move to results/language/TW_AMBIG/ | registered TW_AMBIG; pre-registration with no results file (unrun / stopped / postponed) |
| `TW_LANDMARK_PREREG.md` | CITED | C-cit RIcat mem code6 dcode3 | 2026-10-11 | move to results/language/TW_LANDMARK/ | cited; registered TW_LANDMARK |
| `TW_NORMSTEP.json` | REGISTERED-LIVE | top paper code2 dcode2 | 2026-10-04 | move to results/language/TW_NORMSTEP/ | registered TW_NORMSTEP |
| `TW_NORMSTEP_ANALYSIS.txt` | REGISTERED-LIVE | top paper code | 2026-10-04 | move to results/language/TW_NORMSTEP/ | registered TW_NORMSTEP |
| `TW_NORMSTEP_PREREG.md` | REGISTERED-LIVE | RI RIcat top code5 dcode | 2026-10-03 | move to results/language/TW_NORMSTEP/ | registered TW_NORMSTEP |
| `TW_NORMSTEP_RESULTS.md` | CITED | C-cit RI RIcat DR mem2 top2 docs7 paper2 tex dcode2 | 2026-10-04 | move to results/language/TW_NORMSTEP/ | cited; registered TW_NORMSTEP; banner: CORRECTED/AUDIT block (live) |
| `TW_STATECHANGE.json` | REGISTERED-LIVE | top code2 | 2026-10-09 | move to results/language/TW_STATECHANGE/ | registered TW_STATECHANGE |
| `TW_STATECHANGE_ANALYSIS.txt` | REGISTERED-LIVE | top docs code | 2026-10-09 | move to results/language/TW_STATECHANGE/ | registered TW_STATECHANGE |
| `TW_STATECHANGE_PREREG.md` | REGISTERED-LIVE | RI RIcat top docs code5 dcode7 | 2026-10-09 | move to results/language/TW_STATECHANGE/ | registered TW_STATECHANGE |
| `TW_STATECHANGE_READOUTS.txt` | REGISTERED-LIVE | code | 2026-10-09 | move to results/language/TW_STATECHANGE/ | registered TW_STATECHANGE |
| `TW_STATECHANGE_RESULTS.md` | CITED | C-cit C-wd RI RIcat mem docs paper2 | 2026-10-11 | move to results/language/TW_STATECHANGE/ | cited; registered TW_STATECHANGE; banner: CORRECTED/AUDIT block (live); also named in CLAUDE.md Withdrawn (carries the correction) |
| `TW_STATECHANGE_VERDICTS.json` | REGISTERED-LIVE | top code2 | 2026-10-09 | move to results/language/TW_STATECHANGE/ | registered TW_STATECHANGE |
| `UNFREEZE_PREREG.md` | REGISTERED-LIVE | RIcat top repinv code2 dcode | 2026-09-10 | move to results/em_wm/UNFREEZE/ | registered UNFREEZE |
| `UNFREEZE_RESULTS.md` | REGISTERED-LIVE | C-wd RIcat top7 docs2 tex repinv3 code2 | 2026-09-11 | move to results/em_wm/UNFREEZE/ | registered UNFREEZE; banner: CORRECTED/AUDIT block (live); named in CLAUDE.md Withdrawn: a claim from it is withdrawn, the record stays |
| `V3_RESULTS.md` | WITHDRAWN/VOID | RIcat repinv code arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows); copy in archive_stale/V3_RESULTS.md (differs) |
| `V4_CONTROL_RESULTS.md` | WITHDRAWN/VOID | RIcat mem docs repinv code arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows); copy in archive_stale/V4_CONTROL_RESULTS.md (differs) |
| `V4_MULTISEED.md` | WITHDRAWN/VOID | RIcat top2 docs repinv code3 arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows); copy in archive_stale/V4_MULTISEED.md (differs) |
| `V4_RESULTS.md` | WITHDRAWN/VOID | RIcat repinv code arch | 2026-08-09 | archive to archive/legacy_level15/ | withdrawn or void; banner: PARTIALLY INVALID (lm200 rows); copy in archive_stale/V4_RESULTS.md (differs) |
| `VECTOR_NAV_V2_RESULTS.md` | SUPPORTING | RIcat docs repinv code2 arch | 2026-08-09 | archive to archive/legacy_level15/ | not cited, but referenced as a record (RESULTS_INDEX rows, WTS, memory, other results, docs, report sources) |
| `VISITS_PER_CELL.json` | GENERATED-OUTPUT | repinv code | 2026-08-30 | move with VISITS_TEST.md -> `results/navigation/environments/VISITS_PER_CELL.json` | script output (probe_visits_per_cell.py) |
| `VISITS_TEST.md` | CITED | C-cit C-wd RI RIcat WTS top3 docs8 paper tex repinv3 code | 2026-08-30 | move to results/navigation/environments/ | cited in CLAUDE.md Citable/rank block, a current draft or the claim map; also named in CLAUDE.md Withdrawn (carries the correction) |
| `VOCAB_EM.json` | REGISTERED-LIVE | repinv | 2026-09-19 | move to results/em_wm/VOCAB_EM/ | registered VOCAB_EM |
| `VOCAB_EM.md` | REGISTERED-LIVE | RIcat top docs tex repinv4 | 2026-09-11 | move to results/em_wm/VOCAB_EM/ | registered VOCAB_EM |
| `VOCAB_EM_PREREG.md` | REGISTERED-LIVE | RIcat top repinv2 code | 2026-09-09 | move to results/em_wm/VOCAB_EM/ | registered VOCAB_EM |
| `VOCAB_SWEEP_MULTISEED.json` | GENERATED-OUTPUT | code | 2026-08-09 | archive with VOCAB_SWEEP_MULTISEED.md -> `archive/legacy_level15/VOCAB_SWEEP_MULTISEED.json` | script output |
| `VOCAB_SWEEP_MULTISEED.md` | GENERATED-OUTPUT | RIcat top4 docs repinv2 code2 arch | 2026-08-09 | archive to archive/legacy_level15/ | script output (run_vocab_samebatch.sh) |
| `VOCAB_SWEEP_RESULTS.md` | SUPERSEDED | RIcat mem top5 docs repinv code2 arch | 2026-08-09 | archive to archive/legacy_level15/ | superseded; banner: SUPERSEDED |
| `WARM_PREREG.md` | REGISTERED-LIVE | RIcat top2 repinv code2 dcode | 2026-09-10 | move to results/em_wm/WARM/ | registered WARM |
| `WARM_RESULTS.md` | CITED | RI RIcat DR mem top6 docs3 paper2 tex repinv4 code dcode | 2026-09-11 | move to results/em_wm/WARM/ | cited; registered WARM; banner: CORRECTED/AUDIT block (live) |
| `WRAP_COPE_ANALYSIS.txt` | REGISTERED-LIVE | top docs2 code dcode | 2026-10-14 | move to results/sign/WRAP_COPE/ | registered WRAP_COPE |
| `WRAP_COPE_PREREG.md` | REGISTERED-LIVE | top docs code3 | 2026-10-14 | move to results/sign/WRAP_COPE/ | registered WRAP_COPE |
| `WRAP_COPE_RESULTS.md` | CITED | CM mem paper4 dcode | 2026-10-14 | move to results/sign/WRAP_COPE/ | cited; registered WRAP_COPE |
| `WRAP_COPE_VERDICTS.json` | REGISTERED-LIVE | top docs code2 | 2026-10-14 | move to results/sign/WRAP_COPE/ | registered WRAP_COPE |
| `WRAP_EXISTENCE_ANALYSIS.txt` | CITED | DR top paper4 code dcode | 2026-10-13 | move to results/sign/WRAP_EXISTENCE/ | cited; registered WRAP_EXISTENCE |
| `WRAP_EXISTENCE_PREREG.md` | REGISTERED-LIVE | top docs paper3 code4 | 2026-10-13 | move to results/sign/WRAP_EXISTENCE/ | registered WRAP_EXISTENCE |
| `WRAP_EXISTENCE_RESULTS.md` | CITED | C-cit CM mem2 top2 docs paper13 dcode2 | 2026-10-16 | move to results/sign/WRAP_EXISTENCE/ | cited; registered WRAP_EXISTENCE; banner: CORRECTED/AUDIT block (live) |
| `WRAP_EXISTENCE_VERDICTS.json` | REGISTERED-LIVE | top code2 | 2026-10-13 | move to results/sign/WRAP_EXISTENCE/ | registered WRAP_EXISTENCE |
| `ZERO_SHOT_TRANSFER_clean.md` | GENERATED-OUTPUT | RIcat top repinv2 code2 arch | 2026-04-26 | move to results/navigation/torus/ | script output (master_finish_v3.sh, master_level15pc.sh) |
| `ZERO_SHOT_TRANSFER_clean_brokeninit.md` | WITHDRAWN/VOID | RIcat repinv arch | 2026-04-24 | delete (identical copy already in archive_stale/ZERO_SHOT_TRANSFER_clean_brokeninit.md) | withdrawn or void; copy in archive_stale/ZERO_SHOT_TRANSFER_clean_brokeninit.md (identical) |

## 3. Consolidation groups

Exact duplicates (md5): `MATCH_GATES_64_16.json` == `MATCH_QUERY_GATES.json` (keep `MATCH_QUERY_GATES.json`, which `run_match_query.sh` names). 20 top-level files are byte-identical to copies already in `archive_stale/` (marked "delete" in section 2): `CLONE_ANALYSIS_LEVEL15PC`, `CLONE_TRANSFER_NOBYPASS`, `CNAV_HEX_{Level15,Level15EM,Vanilla,VanillaEM}`, `CNAV_RESULTS`, `DOG_RESULTS`, `HIPPOCAMPAL_{ANALYSIS,GRID,GRIDL15PC,GRID_FREE,HIDDEN,HIDDEN_GRIDFREE,LEVEL15PC}`, `MULTICLASS_RESULTS`, `MULTISEED_FOLLOWUP`, `OOD_GRID_RESULTS`, `R_T_DISTRIBUTION_3WAY`, `ZERO_SHOT_TRANSFER_clean_brokeninit`. Five more name-match `archive_stale/` but differ (the top-level copy has the 2026-08-09 lm200 banner): `NOBYPASS_RESULTS`, `V3_RESULTS`, `V4_CONTROL_RESULTS`, `V4_MULTISEED`, `V4_RESULTS` -- keep the bannered top-level version in `archive/legacy_level15/` and drop the stale copy.

Two kinds of grouping. **Co-locate only** = same topic split across files that are a registered record or are read by code at a fixed path: put them in one folder, do not merge text (merging would change the record or break a script). **Merge** = narrative notes on one topic with no script reading them; one file replaces several (the old names then need the rewrite in 4.3).

| # | group | files | kind | proposed target | why |
|---|---|---:|---|---|---|
| 1 | Per-seed own-map evals, torus depth factorial | 48 | co-locate; merge the .md only | `results/recurrence/FACTORIAL{,_N8}/own_map/`; replace the 24 five-line .md with one `FACTORIAL_OWN_MAP.md` / `FACTORIAL_N8_OWN_MAP.md` table | the .md are renderings of the .json; `analyze_factorial*.py` read the .json by path (keep them) |
| 2 | Per-seed own-map evals, RNN torus | 48 | co-locate; merge the .md only | `results/recurrence/RNN_TORUS/own_map/` + one `RNN_TORUS_OWN_MAP.md` | same; 16 of the lr3e-03 .json are content-identical by line |
| 3 | Per-arm eval tables of registered sign/rank batches | 110 | co-locate | each in its family folder, `arms/` subfolder | registered records; analysis scripts read them; the pre-rescore copies (`_RS`/`_RESCORED` newer) stay inside the record, marked superseded |
| 4 | Gate files of registered batches | 13 | co-locate | family folder `gates/` | each gate version was registered before its numbers; keep all three Memory Maze versions |
| 5 | Unregistered task gates | 22 | merge per task | `RECENCY_GATES_ALL.md`, `MATCH_QUERY_GATES_ALL.md`, `MINIWORLD_GATES_ALL.md`, `ALIASING_GATES.md` (+ its json) | one gate script per task, one table per setting; `MINIWORLD_GRID32_GATES.json` and `_gate_n16.json` have identical content lines |
| 6 | Old broad reports and plans | 14 | co-locate | `archive/reports_old/` | four carry "CONTAINS RETRACTED CLAIMS"/lm200 banners; superseded by `docs/paper/*`, `docs/WHERE_THINGS_STAND.md`, `docs/paper/VENUES_2026-10-15.md` |
| 7 | Level 1.5 / InEKF / PC / TEM / grid cells / lm200 era | 60 | co-locate | `archive/legacy_level15/` (live negatives L15_ABLATION, EXTRAHEAD_CONTROL, NOISE_REFINE, MQ_NOISE_2X2* are CITED: keep them under `results/recurrence/` or a `results/legacy_negatives/`) | CLAUDE.md withdraws the lm200 era; 20 already sit identically in `archive_stale/` |
| 8 | MiniWorld line | 40 | merge notes, co-locate tables | `results/navigation/environments/miniworld/`: `MINIWORLD_FIXED.md` (FIXED_FINDINGS + FIXED_RESULTS + _T1024), `MINIWORLD_FRESH.md` (FRESH_FINDINGS + FRESH_RESULTS + _T1024 + FRESH_ABLATION), `MINIWORLD_ORACLE.md` (ORACLE_* T512/T1024/ABLATION), `MINIWORLD_GRID_SWEEP.md` (+_HIER, HIER_ABLATION) | four versions of each table; none cited; the citable map-extent result lives in ALIASING_CONTROLLED / VISITS_TEST |
| 9 | MiniGrid factorial and DoorKey | 18 | merge | `MINIGRID_FACTORIAL.md` (n8 and full 2x2x2 lead; older as appendix) and `MINIGRID_DOORKEY.md` | same factorial reported at n=3, n=8, full, allocentric; DoorKey in six files |
| 10 | Paper-task OOD and index baseline | 13 | merge the OOD set; co-locate the rest | `results/navigation/torus/paper_task/`; `PAPER_OOD.md` (n8 leads) | n=3 and n=8 versions side by side; PAPER_TASK_ACCURACY is CITED (keep its name) |
| 11 | Composition on Match-Query | 8 | merge | `COMPOSITIONAL_MATCH_QUERY.md`, `COMPOSITIONAL_MULTISEED.md` (COMPOSITIONAL_EXPERIMENT is CITED, keep) | single-seed versions bannered SUPERSEDED; three orphans |
| 12 | EM probes | 9 | merge | `results/em_wm/EM_PROBES.md` | 67-83% shared lines (same table layout, same arms) |
| 13 | Sign probes | 4 | co-locate | `results/sign/SIGN_MATCHED/probes/` | SIGN_MATCHED_PROBE* are in the registered SIGN_MATCHED record |
| 14 | Hierarchy parity / loop | 4 | merge | `results/hierarchy/HIER_PARITY.md` | HIER_PARITY is bannered "superseded in part" by LOOP_HIER_PARITY |
| 15 | Level 1.5 meets gated | 5 | merge | `archive/legacy_level15/LEVEL15_MEETS_GATED.md` | one experiment at three settings |
| 16 | enwik8 runs | 5 | merge the three run tables | `results/code_enwik8/ENWIK8_RUNS.md`; archive ENWIK8_HIER | ENWIK8_HIER is the withdrawn reading of the same runs |
| 17 | Horizon and looped pilots | 21 | co-locate | `archive/withdrawn/horizon/`, `archive/withdrawn/looped_pilot/` | both summary files are withdrawn in CLAUDE.md; the per-arm tables have no other reader |
| 18 | Context-step pilots | 7 | merge pilots 1-3 | `results/language/CTXSTEP/CTXSTEP_PILOTS.md` beside the prereg and design | registration never run; RESULTS_INDEX names each pilot |
| 19 | Addition line | 6 | merge | `results/algorithmic/ADDITION.md` | one unregistered line, five notes, cross-referenced only by each other |
| 20 | Theory notes at the root | 7 | co-locate | `docs/theory/2026-09/` rather than `results/` | they are arguments, not measurements; the later theory lives in `docs/theory/` |
| 21 | Dyck result variants | 4 | co-locate; archive SHAREABLE and T3 | `results/dyck/DYCK/` | 58% shared lines between bs128 and POPE; T3 is empty |

Not consolidated on purpose: the two G3_SSM blocks (`G3_SSM_*` vs `G3_SSM_A2_*`, 65-72% shared lines) are separate registered blocks; `CLOCK_SIDE_GATES*` and `RECENCY_GATES_*` share a gate script but belong to different registrations; `KNOB_SWEEP` and `KNOB_SWEEP_n8` are both CITED.

## 4. Proposed folder layout and rewrite map

### 4.1 Layout

Topic folders follow the RESULTS_INDEX catalogue lines (`docs/tools/catalog_results_index.py` groups), extended for the post-10-09 lines. Each registered family gets its own subfolder (prereg, results, analysis, verdicts and per-arm outputs together); cited and supporting files without a family sit at the topic level. Counts are files that would land there under section 2.

```
/  (root)            README.md CLAUDE.md RESULTS_INDEX.md GUARDS.md KNOWN_BUGS.md HOURGLASS_README.md requirements.txt + code
archive/generated/                     40 files -- algorithmic, code_enwik8, em_wm, navigation, rank
archive/legacy_level15/                42 files
archive/orphan/                        26 files -- algorithmic, dyck, hierarchy, navigation, pope, rank
archive/reports_old/                   14 files
archive/superseded/                    15 files -- algorithmic, hierarchy, language, navigation, rank, sign
archive/withdrawn/                     47 files -- CODE, DYCK, RECENCY_T3, code_enwik8, hierarchy, navigation, rank, recurrence
results/algorithmic/                   20 files -- INDIRECT
results/code_enwik8/                    9 files -- CODE_FULLVAL, ENWIK8_COMPOSITION
results/dyck/                          28 files -- CANCEL, DYCK, DYCK_DECAY, DYCK_DEPTH, DYCK_LADDER, DYCK_MDEPTH
results/em_wm/                         72 files -- D5, DOF, MAGONLY, MINIGRID_EM, MINIGRID_EM_FIX, N5, NOLEAK, PAIRCONST, PAIRORIGIN, PAIRSPLIT, REC_EM, SEARCH, SPREAD, SPREAD2, UNFREEZE, VOCAB_EM, WARM
results/hierarchy/                     36 files -- COMP_HEADROOM
results/images/                         8 files -- IMAGE_TORUS
results/language/                      37 files -- CTXSTEP, GAIN_PHASE, LEAK, TEXTWORLD, TW_AMBIG, TW_LANDMARK, TW_NORMSTEP, TW_STATECHANGE
results/memory_maze/                    7 files -- MEMORY_MAZE
results/navigation/environments/       40 files
results/navigation/match_query/        14 files
results/navigation/torus/              53 files -- PAPER2X2, PAPERTASK
results/pope/                          90 files -- ABLATE, AUG, CROSS, DECAY, DELTA_FREE, GAIN_GRAIN, JSB, JSBLEN, MAPPOPE_PAIR, MAPPOPE_R4, SCORE_RANK, SROPE_TORUS, T1, T2, T3, T3GEN
results/rank/                         142 files -- LOCALISATION, LOOP_RANK, LOOP_RANK_E1800, RANK3, RANKCLOCK, RANKRECIPE, RANKWD, RANK_MATCHED, RANK_MI, RANK_ND, RANK_NOWRAP, RANK_PERHEAD, RANK_PROJ, RANK_SEP, RANK_TESTS, RANK_WRAP
results/recurrence/                   160 files -- FACTORIAL, FACTORIAL_N8, G3_SSM, RNN_TORUS, SAMEBLOCK
results/sign/                         175 files -- ACCOUNTS, CLOCK_SIDE, CLOCK_SPEED, CLOCK_SPEED_REP, EXPTORUS, FORGET_CLOCK, GATED, GAUGE, GENERALITY, LRSWEEP, MECHANISM, MONOTONE, MQAR, RECENCY, RING_BUDGET, RING_DIAL, ROBUST, SIGNROUTE, SIGNSEP, SIGN_ABLATION, SIGN_MATCHED, TORUS_LRSWEEP, TORUS_START, WRAP_COPE, WRAP_EXISTENCE
```

Topic assignment (regex on the stem, first match): `memory_maze` MEMORY_MAZE; `images` IMAGE_TORUS; `recurrence` RNN_TORUS, FACTORIAL*, LOOP_*/LOOPED_*, REFINE_, MQ_* (not MQ_RANK), L15_LOOP, SAMEBLOCK, G3_SSM; `rank` RANK*, LOOP_RANK*, MQ_RANK, FAST_ATTN_RANK, PAPER_FIG4, ACTION_GEOMETRY, LEARNED_RANK, DXR_, ND_GATES, GATE_PROBE, CONV_KERNEL, LOCALISATION, ACCUMULATOR, THEORY_{NARRATIVE,NUMBERS,SEARCH}; `pope` MAPPOPE*, SCORE_RANK, GAIN_GRAIN, DELTA_FREE, ABLATE_{PREREG,RESULTS}, POPE_*, THEORY_MAPPOPE, JSB*, AUG_, DECAY_, T1/T2/T3/T3GEN, TORUS_T3, RECENCY_T3, CROSS_, SELECTIVE_ROPE, SROPE_TORUS; `sign` SIGN*, MONOTONE, ACCOUNTS, CLOCK_SIDE/SPEED/SCAN, EXPTORUS, GAUGE, GENERALITY, (TORUS_)LRSWEEP, MECHANISM, RING_*, ROBUST, TORUS_START, WRAP_*, RECENCY_* (not EM), GATED_, FLIPFLOP, MQAR, FORGET_, LAMBDA_TRACE, COUNTER_, TEM_RECENCY; `language` TEXTWORLD, CTXSTEP, CONTEXT_STEP, TW_, LEAK, GAIN_PHASE, TINYSTORIES, LANGUAGE_LANDSCAPE; `code_enwik8` CODE_, ENWIK8, BF16; `dyck` DYCK_, CANCEL_; `em_wm` EM_, AUDIT_2026, THEORY_KERNEL, TALE_OF, REC_EM, RECENCY_EM, N5, DOF, D5, MAGONLY, WARM, UNFREEZE, NOLEAK, SEARCH, SPREAD*, PAIR*, MATCH_QUERY_EM, VOCAB_EM, MINIGRID_EM*, AP_KERNEL/COHERENCE, HOPFIELD, PAPER_FIG4_EM, _REWIND; `hierarchy` COMPOSITIONAL*, COMP_HEADROOM, HIER_*, HIERGOAL, AGGREGATE, BOUNDED_MEMORY, ROUTE_ATTN, SPACETIME, ABLATE_COMPOSITIONAL, CORRECTION_COMPOSITIONAL, PLANNER, ROOMS_GOAL, DISSOCIATION, CSCG, STITCH, MAP_QUERY, LAP_, LOOP_HIER; `algorithmic` FAMILY_TREE, ABLATE_FAMILY, FRONTIER, ALGORITHMIC, RECURSIVE, ADDITION_, INDIRECT_; `navigation/match_query` MATCH_QUERY*, MATCH_GATES; `navigation/environments` MINIGRID, MINIWORLD, ALIASING, VISITS_, POSITION_EFFECT, CROSSOVER_CONVERGED, CONTINUOUS_ALLOC, DAGGER, DOORKEY, HABITAT; `navigation/torus` PAPER*, INDEX_BASELINE, BASELINE_TABLE, REVISIT_, N3_AUDIT, NOISE_CLEAN, HORIZON_, FREQ_CONTROL, ROPE_*, RECIPE_*, OOD_GRID, PERSCALE/OMEGA_RESCALE, LONG_SEQ, PER_VISIT, ZERO_SHOT, DRIFT_PROBE, TOPOLOGY, KNOB_SWEEP, ALLOCENTRIC, ROTATE, H12_, TIMING; `archive/legacy_level15` the April-August Level 1.5 / InEKF / PC / TEM / grid-cell / lm200 files.

Alternatives worth weighing: (a) a flatter `results/<topic>/` with no family subfolders (fewer path segments in CLAUDE.md, but RNN_TORUS alone puts 62 files in `recurrence/`); (b) leave the 144 CITED files at the root and move only the rest (no CLAUDE.md / draft rewrites at all; root drops from 1,102 to ~150 files); (c) move only the 191 generated outputs and 553 registered per-arm/raw files into `results/` and keep every narrative `.md` at the root. (b) gets most of the tidiness for a fraction of the rewrite risk.

### 4.2 Old path -> new path

Section 2 gives every destination. The mapping is mechanical: the name never changes, only the folder, so one `old -> new` table (one row per file, generated from section 2) drives `git mv` and the text rewrite. Names are unique across the repo root, so after the move a bare name still identifies one file (`find results archive -name NAME`).

### 4.3 Rewrite map (every file that names a file that moves)

A full-name reference (`NAME.md`, `NAME.json`, `NAME.txt`, or `$REPO/NAME...`) to a file that moves or is deleted appears in 1218 files, 5156 references in all. Grouped by who must edit:

| group | files | references | treatment |
|---|---:|---:|---|
| A. live control documents (CLAUDE.md, RESULTS_INDEX.md, README, GUARDS, KNOWN_BUGS, HOURGLASS_README, WHERE_THINGS_STAND, memory, current drafts, claim map) | 25 | 577 | rewrite to the new path in the same commit as the move |
| A2. document sources (`*.tex`, `report/*.tex`, `report/*.html` incl. the shared report `language_summary.html`) | 6 | 219 | rewrite if rebuilt; the shared artifact must be republished WITH `url=` |
| B. code (rule 18) | 696 | 1323 | rewrite the ones still used (guards, analysis, figures, live drivers); historical drivers can keep the old path if a note says they predate the move |
| C. other top-level files (they move too) | 311 | 932 | bare names stay valid as names; rewrite only if links must resolve |
| D. history and inventories (`docs/LOG.md`, `docs/audits/**`, `docs/SESSION_*`, `docs/lit`, `docs/paper` old drafts, `report/*.md`, `archive*`) | 180 | 2105 | do not rewrite history; add one `docs/MOVED_2026-10.md` (old name -> new path) instead |

Bare-stem mentions (e.g. `` `RANK_SEP` `` without `.md`) are not paths and need no rewrite; in group A they number 430 (mostly the RESULTS_INDEX catalogue, which should be regenerated, not edited).

#### A. Live control documents

- **CLAUDE.md** (113): `ABLATE_RESULTS.md` `ACCOUNTS_RESULTS.md` `ACCUMULATOR.md` `ALIASING_CONTROLLED.md` `ALLOCENTRIC_RECODING.md` `AUDIT_2026-09-10.md` `AUG_RESULTS.md` `BASELINE_TABLE.md` `BF16_RESULTS.md` `CANCEL_RESULTS.md` `CODE_DECAY_RESULTS.md` `CODE_FULLVAL_RESULTS.md` `CODE_RESULTS.md` `COMPOSITIONAL_EXPERIMENT.md` `COMP_HEADROOM.md` `CORRECTED_LM200_LEADERBOARD.md` `CROSS_RESULTS.md` `D5_RESULTS.md` `DELTA_FREE_RESULTS.md` `DXR_RANK_THRESHOLD.md` `DYCK_DEPTH_RESULTS.md` `DYCK_LADDER_RESULTS.md` `DYCK_LITERATURE_METRICS.md` `DYCK_MDEPTH_RESULTS.md` `DYCK_RESULTS_bs128.md` `DYCK_T3_RESULTS.md` `EM_WM_STATE.md` `ENWIK8_HIER.md` `ENWIK8_HIERARCHY.md` `EXPTORUS_RESULTS.md` `EXTRAHEAD_CONTROL.md` `FACTORIAL_N8_RESULTS.md` `FACTORIAL_RESULTS.md` `FAMILY_TREE_RESULTS.md` `FAST_ATTN_RANK.md` `FORGET_CONTROL.md` `G3_SSM_A2_RESULTS.md` `G3_SSM_RESULTS.md` `GAIN_GRAIN_RESULTS.md` `GAIN_PHASE_RESULTS.md` `GATED_RESULTS.md` `GAUGE_RESULTS.md` `GENERALITY_RESULTS.md` `H12_BUDGET_CURVE.md` `HABITAT_BUILD.md` `HIERGOAL_ABLATION.md` `HIER_RECHECK.md` `HORIZON_RESULTS.md` `IMAGE_TORUS_RESULTS.md` `INDIRECT_RESULTS_200k.md` `JSB_RESULTS.md` `KNOB_SWEEP_n8.md` `L15_ABLATION.md` `L15_LOOP_2X2.md` `LAP_TRANSFER_NOREWARD.md` `LEAK_RESULTS.md` `LOCALISATION.md` `LOOPED_PILOT.md` `LOOP_DEPTH_STRATA.md` `LOOP_HEADROOM.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LOOP_RANK_RESULTS.md` `LRSWEEP_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `MECHANISM_RESULTS.md` `MEMORY_MAZE_RESULTS.md` `MQ_RANK_2X2.md` `N3_AUDIT.md` `NOISE_REFINE.md` `PAIRSPLIT_RESULTS.md` `PAPER2X2_RESULTS.md` `PLANNER_TASK_AUDIT.md` `RANK3_RESULTS.md` `RANKCLOCK_RESULTS.md` `RANKRECIPE_RESULTS.md` `RANKWD_RESULTS.md` `RANK_MATCHED_RESULTS.md` `RANK_MI_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_PROJ_FROZEN.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md` `RANK_SWEEP.md` `RANK_TESTS_RESULTS.md` `RANK_WRAP_RESULTS.md` `RECENCY_GATE_ABLATION.md` `RECENCY_RESULTS.md` `REFINE_RESULTS.md` `RNN_TORUS_RESULTS.md` `ROBUST_RESULTS.md` `SCORE_RANK_RESULTS.md` `SEARCH_RESULTS.md` `SELECTIVE_ROPE.md` `SIGNROUTE_RESULTS.md` `SIGNSEP_RESULTS.md` `SIGN_ABLATION.md` `SIGN_MATCHED_RESULTS.md` `SROPE_TORUS_RESULTS.md` `STITCH_ATTENTION.md` `T2_RESULTS.md` `TEXTWORLD_RESULTS.md` `TIMING_BENCHMARK.md` `TINYSTORIES_PILOT_RESULTS.md` `TORUS_LRSWEEP_RESULTS.md` `TORUS_START_RESULTS.md` `TW_LANDMARK_PREREG.md` `TW_NORMSTEP_RESULTS.md` `TW_STATECHANGE_RESULTS.md` `UNFREEZE_RESULTS.md` `VISITS_TEST.md` `WRAP_EXISTENCE_RESULTS.md`
- **RESULTS_INDEX.md** (150): `ABLATE_COMPOSITIONAL.md` `ABLATE_RESULTS.md` `ALIASING_CONTROLLED.md` `ALLOCENTRIC_RECODING.md` `AUG_RESULTS.md` `CANCEL_PREREG.md` `CANCEL_RESULTS.md` `CODE_FULLVAL_PREREG.md` `CODE_FULLVAL_RESULTS.md` `CODE_GATES.md` `CODE_RESULTS.md` `COMP_HEADROOM.md` `CONTEXT_STEP_DESIGN.md` `COUNTER_RESULTS.md` `CROSS_RESULTS.md` `CTXSTEP_HSR_PILOT.md` `CTXSTEP_HS_RECIPE.md` `CTXSTEP_PILOT1.md` `CTXSTEP_PILOT2.md` `CTXSTEP_PILOT3.md` `CTXSTEP_PREREG.md` `D5_RESULTS.md` `DECAY_RESULTS.md` `DYCK_LADDER_RESULTS.md` `DYCK_LITERATURE_METRICS.md` `DYCK_MDEPTH_PREREG.md` `DYCK_MDEPTH_RESULTS.md` `DYCK_RESULTS_bs128.md` `EM_WM_STATE.md` `EM_WM_THEORY.md` `ENWIK8_HIERARCHY.md` `EXPTORUS_ANALYSIS.txt` `EXPTORUS_PREREG.md` `EXPTORUS_RESULTS.md` `EXPTORUS_VERDICTS.json` `EXTRAHEAD_CONTROL.md` `FACTORIAL_N8_ANALYSIS.txt` `FACTORIAL_N8_PREREG.md` `FACTORIAL_N8_RESULTS.md` `FACTORIAL_RESULTS.md` `FAMILY_TREE_RESULTS.md` `FORGET_CLOCK_PREREG.md` `FORGET_CONTROL.md` `FORGET_GATE.md` `G3_SSM_A2_ANALYSIS.txt` `G3_SSM_A2_RESULTS.md` `G3_SSM_ANALYSIS.txt` `G3_SSM_PREREG.md` `G3_SSM_RESULTS.md` `GAIN_GRAIN_PREREG.md` `GAIN_GRAIN_RESULTS.md` `GAIN_PHASE_PREREG.md` `GAIN_PHASE_RESULTS.md` `H12_BUDGET_CURVE.md` `HIER_RECHECK.md` `IMAGE_TORUS_ANALYSIS.txt` `IMAGE_TORUS_GATE.md` `IMAGE_TORUS_PILOT.txt` `IMAGE_TORUS_PILOT2.txt` `IMAGE_TORUS_PREREG.md` `IMAGE_TORUS_RESULTS.md` `IMAGE_TORUS_VERDICTS.json` `INDIRECT_OOD.md` `INDIRECT_RESULTS_200k.md` `JSB_LENGTH_RESULTS.md` `JSB_LENGTH_RESULTS_RANK.md` `JSB_RESULTS.md` `KNOB_SWEEP_n8.md` `L15_ABLATION.md` `L15_LOOP_2X2.md` `LEAK_PREREG.md` `LEAK_RESULTS.md` `LM200_CORRECTED_MULTISEED.md` `LOOP_HEADROOM.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LOOP_RANK_E1800_PREREG.md` `LOOP_RANK_PREREG.md` `LOOP_RANK_RESULTS.md` `LOOP_SAMPLED.md` `MAGONLY_RESULTS.md` `MAPPOPE_PAIR_PREREG.md` `MAPPOPE_PAIR_RESULTS.md` `MAP_QUERY_GATES.md` `MAP_QUERY_RESULTS.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `MEMORY_MAZE_ANALYSIS.txt` `MEMORY_MAZE_GATE_V3.md` `MEMORY_MAZE_PREREG.md` `MEMORY_MAZE_RESULTS.md` `MEMORY_MAZE_VERDICTS.json` `MINIGRID_FULL_2X2X2.md` `MQ_RANK_2X2.md` `N3_AUDIT.md` `PAIRSPLIT_RESULTS.md` `PAPER2X2_RESULTS.md` `POPE_WRAPPING.md` `RANK3_GEOMETRY.md` `RANK3_PREREG.md` `RANK3_RESULTS.md` `RANKRECIPE_ANALYSIS.txt` `RANKRECIPE_PREREG.md` `RANKRECIPE_RESULTS.md` `RANK_MATCHED_RESULTS.md` `RANK_MI_RESULTS.md` `RANK_ND_PREREG.md` `RANK_ND_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_PREREG.md` `RANK_SEP_RESULTS.md` `RANK_SWEEP.md` `RANK_TESTS_PREREG.md` `RANK_TESTS_RESULTS.md` `RANK_WRAP_PREREG.md` `RANK_WRAP_RESULTS.md` `RECENCY_EM_RESULTS.md` `RECENCY_GATE_ABLATION.md` `RECENCY_RESULTS.md` `REFINE_RESULTS.md` `REVISIT_2X2.md` `REVISIT_DISTANCE.md` `RNN_TORUS_ANALYSIS.txt` `RNN_TORUS_PREREG.md` `RNN_TORUS_RESULTS.md` `SAMEBLOCK_RESULTS.md` `SCORE_RANK_PREREG.md` `SCORE_RANK_RESULTS.md` `SEARCH_RESULTS.md` `SIGNSEP_ANALYSIS.txt` `SIGNSEP_PREREG.md` `SIGNSEP_RESULTS.md` `SIGN_ABLATION.md` `SIGN_MATCHED_PREREG.md` `SIGN_MATCHED_PROBE.md` `SIGN_MATCHED_RESULTS.md` `STITCH_ATTENTION.md` `TEXTWORLD_PREREG.md` `TEXTWORLD_RESULTS.md` `TIMING_BENCHMARK.md` `TINYSTORIES_PILOT.md` `TINYSTORIES_PILOT_RESULTS.md` `TORUS_LRSWEEP_ANALYSIS.txt` `TORUS_LRSWEEP_PREREG.md` `TORUS_LRSWEEP_RESULTS.md` `TW_NORMSTEP_PREREG.md` `TW_NORMSTEP_RESULTS.md` `TW_STATECHANGE_PREREG.md` `TW_STATECHANGE_RESULTS.md` `VISITS_TEST.md` `WARM_RESULTS.md`
- **README.md** (19): `ACCUMULATOR.md` `CORRECTED_LM200_LEADERBOARD.md` `DETAILED_RESULTS.md` `EM_WM_STATE.md` `EXTRAHEAD_CONTROL.md` `FAMILY_TREE_RESULTS.md` `L15_ABLATION.md` `LM200_CORRECTED_MULTISEED.md` `MATCH_QUERY_EM.md` `MULTICLASS_RESULTS.md`(del) `NOISE_CLEAN_REVALIDATION.md` `PAPER2X2_RESULTS.md` `PAPER_OOD_PROTOCOL.md` `PLANNER_TASK_AUDIT.md` `RANK_MI_RESULTS.md` `RESULTS_PAPER.md` `RESULTS_SUMMARY_2026-05-10.md` `TIMING_BENCHMARK.md` `VOCAB_SWEEP_MULTISEED.md`
- **GUARDS.md** (2): `_D5_N24.json` `_REWIND_PROBE.json`
- **KNOWN_BUGS.md** (1): `MINIWORLD_FIXED_RESULTS.md`
- **HOURGLASS_README.md** (3): `COMPOSITIONAL_EXPERIMENT.md` `COMPOSITIONAL_MULTISEED.md` `COMPOSITIONAL_RESULTS.md`
- **docs/WHERE_THINGS_STAND.md** (26): `ALIASING_CONTROLLED.md` `CANCEL_RESULTS.md` `CTXSTEP_HSR_PILOT.md` `CTXSTEP_HS_RECIPE.md` `CTXSTEP_PREREG.md` `DYCK_MDEPTH_RESULTS.md` `EM_WM_STATE.md` `H12_BUDGET_CURVE.md` `KNOB_SWEEP_n8.md` `LEAK_RESULTS.md` `LOOP_HEADROOM.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LOOP_RANK_RESULTS.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `PAPER2X2_RESULTS.md` `RANK3_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md` `RANK_WRAP_RESULTS.md` `REFINE_RESULTS.md` `SEARCH_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `VISITS_TEST.md`
- **docs/paper/MAIN_APPENDIX2.md** (75): `ACCOUNTS_RESULTS.md` `CANCEL_PREREG.md` `CANCEL_RESULTS.md` `DELTA_FREE_RESULTS.md` `DYCK_MDEPTH_RESULTS.json` `DYCK_MDEPTH_RESULTS.md` `EXPTORUS_PREREG.md` `EXPTORUS_RESULTS.md` `FACTORIAL_N8_PREREG.md` `FACTORIAL_N8_RESULTS.md` `FACTORIAL_RESULTS.md` `G3_SSM_A2_RESULTS.md` `G3_SSM_PREREG.md` `G3_SSM_RESULTS.md` `GAIN_GRAIN_RESULTS.md` `GAIN_PHASE_RESULTS.md` `GAUGE_RESULTS.md` `GENERALITY_ANALYSIS.txt` `IMAGE_TORUS_PREREG.md` `IMAGE_TORUS_RESULTS.md` `KNOB_SWEEP_n8.md` `LEAK_RESULTS.md` `LOOP_RANK_E1800_P1_ANALYSIS.txt` `LOOP_RANK_RESULTS.md` `LRSWEEP_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `MECHANISM_RESULTS.md` `MEMORY_MAZE_ANALYSIS.txt` `MEMORY_MAZE_PREREG.md` `MEMORY_MAZE_RESULTS.md` `NOLEAK_RESULTS.md` `PAPER2X2_RESULTS.md` `PAPER_TASK_ACCURACY.md` `RANK3.json` `RANK3_ANALYSIS.txt` `RANKCLOCK.md` `RANKCLOCK_ANALYSIS.txt` `RANKCLOCK_PREREG.md` `RANKCLOCK_RESULTS.md` `RANKCLOCK_VERDICTS.json` `RANKRECIPE_PREREG.md` `RANKRECIPE_RESULTS.md` `RANKWD_PREREG.md` `RANKWD_RESULTS.md` `RANK_MI_ANALYSIS.txt` `RANK_ND_RESULTS.md` `RANK_PROJ_ANALYSIS.txt` `RANK_PROJ_RESULTS.md` `RANK_SEP.json` `RANK_SEP_ANALYSIS.txt` `RANK_TESTS_RESULTS.md` `RANK_WRAP_RESULTS.md` `RECENCY_EM_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_VERDICTS.json` `RNN_TORUS_PREREG.md` `RNN_TORUS_RESULTS.md` `ROBUST_ANALYSIS.txt` `ROBUST_RESULTS.md` `SCORE_RANK_RESULTS.md` `SEARCH_RESULTS.md` `SIGNROUTE_RESULTS.md` `SIGNSEP_PREREG.md` `SIGNSEP_RESULTS.md` `SIGN_MATCHED_ANALYSIS.txt` `SPREAD_RESULTS.md` `SROPE_TORUS_RESULTS.md` `TEXTWORLD_ANALYSIS.txt` `TEXTWORLD_RESULTS.md` `TORUS_LRSWEEP_PREREG.md` `TORUS_LRSWEEP_RESULTS.md` `TORUS_START_RESULTS.md` `TW_NORMSTEP_RESULTS.md` `WARM_RESULTS.md` `WRAP_EXISTENCE_ANALYSIS.txt`
- **docs/paper/DRAFT6_workshop.md** (3): `EXPTORUS_RESULTS.md` `G3_SSM_A2_RESULTS.md` `G3_SSM_RESULTS.md`
- **docs/paper/CLAIM_MAP_clock_map.md** (45): `ACCOUNTS_RESULTS.md` `CANCEL_RESULTS.md` `CLOCK_SIDE_RESULTS.md` `CLOCK_SPEED_REP_RESULTS.md` `CLOCK_SPEED_RESULTS.md` `CODE_FULLVAL_RESULTS.md` `CROSS_RESULTS.md` `DELTA_FREE_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `EXPTORUS_RESULTS.md` `FACTORIAL_N8_RESULTS.md` `FACTORIAL_RESULTS.md` `G3_SSM_A2_RESULTS.md` `G3_SSM_RESULTS.md` `GAUGE_RESULTS.md` `GENERALITY_RESULTS.md` `IMAGE_TORUS_RESULTS.md` `LOCALISATION.md` `LRSWEEP_RESULTS.md` `MATCH_QUERY_SCALE.md` `MECHANISM_RESULTS.md` `MEMORY_MAZE_RESULTS.md` `PAPER2X2_RESULTS.md` `RANKCLOCK_RESULTS.md` `RANKRECIPE_RESULTS.md` `RANK_TESTS_RESULTS.md` `RECENCY_GATE_ABLATION.md` `RECENCY_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_RESULTS.md` `RNN_TORUS_RESULTS.md` `ROBUST_RESULTS.md` `SCORE_RANK_RESULTS.md` `SIGNSEP_RESULTS.md` `SIGN_ABLATION.md` `SIGN_MATCHED_RESULTS.md` `SROPE_TORUS_RESULTS.md` `T2_RESULTS.md` `TEXTWORLD_RESULTS.md` `THEORY_KERNEL.md` `TINYSTORIES_PILOT_RESULTS.md` `TORUS_LRSWEEP_RESULTS.md` `TORUS_START_RESULTS.md` `WRAP_COPE_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- **.claude-memory/feedback_baselines_backfill.md** (5): `DAGGER_RESULTS.md` `DOORKEY_BC_RESULTS.md` `LEVEL15BETA_RESULTS.md` `NODROP_PARETO_RESULTS.md` `VOCAB_SWEEP_RESULTS.md`
- **.claude-memory/feedback_convergence_first.md** (4): `COMP_HEADROOM.md` `CROSS_RESULTS.md` `RANK_MATCHED_PREREG.md` `RANK_MATCHED_RESULTS.md`
- **.claude-memory/feedback_em_vs_wm_mechanism.md** (8): `AUDIT_2026-09-10.md` `COUNTER_RESULTS.md` `EM_WM_STATE.md` `EM_WM_THEORY.md` `NOLEAK_RESULTS.md` `SEARCH_RESULTS.md` `TALE_OF_TWO_ALGORITHMS.md` `THEORY_KERNEL.md`
- **.claude-memory/feedback_existence_before_mechanism.md** (2): `EM_WM_STATE.md` `THEORY_KERNEL.md`
- **.claude-memory/project_clock_vs_map.md** (13): `ACCUMULATOR.md` `CROSS_RESULTS.md` `DYCK_DECAY_RESULTS.md` `EM_WM_STATE.md` `LOCALISATION.md` `RECENCY_RESULTS.md` `SEARCH_RESULTS.md` `T1_RESULTS.md` `T2_RESULTS.md` `T3GEN_RESULTS.md` `THEORY_KERNEL.md` `THEORY_MAPPOPE.md` `WARM_RESULTS.md`
- **.claude-memory/project_hierarchy_negative.md** (5): `AGGREGATE_EXTRAS.md` `BOUNDED_MEMORY_RESULTS.md` `HIER_PARITY.md` `LOOP_HIER_PARITY.md` `REFINE_RESULTS.md`
- **.claude-memory/project_loop_and_correction.md** (7): `ALGORITHMIC_RESULTS.md` `FRONTIER_ALGORITHMIC.md` `L15_ABLATION.md` `L15_LOOP_2X2.md` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md` `V4_CONTROL_RESULTS.md`
- **.claude-memory/project_mappope_asymmetry.md** (7): `CODE_FULLVAL_RESULTS.md` `DYCK_LADDER_RESULTS.json` `DYCK_MDEPTH_RESULTS.md` `GAIN_GRAIN_RESULTS.md` `JSB_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `SCORE_RANK_RESULTS.md`
- **.claude-memory/project_miniworld_flip_negative.md** (2): `ALIASING_CONTROLLED.md` `PAPER2X2_RESULTS.md`
- **.claude-memory/project_rank_and_selective_rope.md** (18): `ACTION_GEOMETRY.md` `GATE_PROBE.md` `JSB_LENGTH_RESULTS_RANK.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LOOP_RANK_RESULTS.md` `MAPPOPE_R4_RESULTS.md` `PAPER_FIG4_REPRO.md` `RANK3_RESULTS.md` `RANK_MATCHED_RESULTS.md` `RANK_MI_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_PROJ_FROZEN.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md` `RANK_SWEEP.md` `RANK_WRAP_RESULTS.md` `SCORE_RANK_RESULTS.md` `SELECTIVE_ROPE.md`
- **.claude-memory/project_robustness_vs_capability.md** (13): `CODE_DECAY_RESULTS.md` `CODE_FULLVAL_RESULTS.md` `CODE_RESULTS.md` `DECAY_RESULTS.md` `DYCK_LADDER_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `LOOP_RANK_RESULTS.md` `PAPER2X2_RESULTS.md` `RANK_MATCHED_RESULTS.md` `RANK_PROJ_FROZEN.md` `RANK_SEP_RESULTS.md` `SIGN_ABLATION.md` `SIGN_MATCHED_RESULTS.md`
- **.claude-memory/project_sign_axis.md** (5): `SIGN_ABLATION.md` `SIGN_MATCHED_RESULTS.md` `SIGN_PROBE.md` `T2_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- **.claude-memory/project_state.md** (46): `ACCOUNTS_RESULTS.md` `CANCEL_RESULTS.md` `CODE_FULLVAL_RESULTS.md` `CTXSTEP_HSR_PILOT.md` `CTXSTEP_PREREG.md` `DELTA_FREE_RESULTS.md` `EXPTORUS_RESULTS.md` `FACTORIAL_N8_RESULTS.md` `G3_SSM_A2_RESULTS.md` `G3_SSM_RESULTS.md` `GAIN_GRAIN_RESULTS.md` `GAIN_PHASE_RESULTS.md` `GAUGE_RESULTS.md` `GENERALITY_RESULTS.md` `IMAGE_TORUS_RESULTS.md` `LEAK_RESULTS.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LRSWEEP_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `MECHANISM_RESULTS.md` `MEMORY_MAZE_RESULTS.md` `RANK3_RESULTS.md` `RANKCLOCK_RESULTS.md` `RANKRECIPE_RESULTS.md` `RANKWD_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_NOWRAP_PREREG.md` `RANK_TESTS_RESULTS.md` `RANK_WRAP_RESULTS.md` `RNN_TORUS_RESULTS.md` `ROBUST_RESULTS.md` `SCORE_RANK_RESULTS.md` `SIGNROUTE_RESULTS.md` `SIGNSEP_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `SROPE_TORUS_RESULTS.md` `TEXTWORLD_RESULTS.md` `TINYSTORIES_PILOT_RESULTS.md` `TORUS_LRSWEEP_RESULTS.md` `TORUS_START_RESULTS.md` `TW_AMBIG_PREREG.md` `TW_LANDMARK_PREREG.md` `TW_NORMSTEP_RESULTS.md` `TW_STATECHANGE_RESULTS.md` `WRAP_COPE_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- **.claude-memory/project_what_where_and_language.md** (3): `LEAK_RESULTS.md` `TEXTWORLD_RESULTS.md` `TW_NORMSTEP_RESULTS.md`
- **.claude-memory/reference_looped_transformer_lit.md** (2): `FRONTIER_ALGORITHMIC.md` `REFINE_RESULTS.md`

#### A2. Document sources

- **artifacts_results_brief.html** (28): `ALGORITHMIC_RESULTS.md` `BASELINE_TABLE.md` `COMPOSITIONAL_MATCH_QUERY.md` `COMPOSITIONAL_MULTISEED.md` `EM_COMP_SAMEBATCH.md` `FREQ_CONTROL.md` `FRONTIER_ALGORITHMIC.md` `HIER_PARITY.md` `INDEX_BASELINE_PAPER_TASK_n8.md` `KNOB_SWEEP.md` `L15_ABLATION.md` `LM200_CORRECTED_MULTISEED.md` `LOOP_HIER_COMPUTE.md` `LOOP_HIER_PARITY.md` `MATCH_QUERY_EM.md` `MATCH_QUERY_SCALE.md` `MINIGRID_2X2X2_n8.md` `MINIGRID_ALLOCENTRIC_2X2X2.md` `MINIWORLD_FIXED_RESULTS.md` `MINIWORLD_FRESH_RESULTS.md` `MINIWORLD_ORACLE_RESULTS_T512.md` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md` `NODROP_PARETO_RESULTS.md` `PAPER_OOD_EXTENDED_n8.md` `PAPER_OOD_WITH_POPE.md` `REVISIT_DISTANCE.md` `TIMING_BENCHMARK.md`
- **paper_rank/sections/04_experiments_2d.tex** (3): `ACTION_GEOMETRY.md` `PAPER_FIG4_REPRO.md` `RANK_SWEEP.md`
- **paper_rank/sections/05_experiments_generator.tex** (2): `GATE_PROBE.md` `SELECTIVE_ROPE.md`
- **report/navigation_in_words.html** (3): `TEXTWORLD_PREREG.md` `TEXTWORLD_RESULTS.md` `TW_NORMSTEP_RESULTS.md`
- **report/pending.tex** (2): `MINIWORLD_ENDPOINTS.md` `PAPER2X2_RESULTS.md`
- **report/report.tex** (181): `ABLATE_COMPOSITIONAL.md` `ABLATE_FAMILY_TREE.md` `ACCUMULATOR.md` `ACTION_GEOMETRY.md` `ALGORITHMIC_GATES.md` `ALGORITHMIC_RESULTS.md` `ALIASING_CONTROLLED.md` `ALIASING_GATES.md` `ALLOCENTRIC_RECODING.md` `AP_KERNEL_DIAGNOSTIC.md` `AUDIT_2026-09-10.md` `AUDIT_HEADLINE.md` `AUG_RESULTS.md` `BASELINE_TABLE.md` `CODE_FULLVAL_RESULTS.md` `CODE_RESULTS.md` `COMPOSITIONAL_EXPERIMENT.md` `COMPOSITIONAL_MULTISEED.md` `COMP_HEADROOM.md` `CONTINUOUS_ALLOC.md` `CORRECTION_COMPOSITIONAL.md` `CROSSOVER_CONVERGED.md` `CROSS_RESULTS.md` `CSCG_TASK_GATES.md` `D5_RESULTS.md` `DECAY_RESULTS.md` `DISSOCIATION_SWEEP.md` `DOF_RESULTS.md` `DRIFT_PROBE.md` `DXR_RANK_THRESHOLD.md` `DYCK_DECAY_RESULTS.md` `DYCK_LADDER_RESULTS.md` `DYCK_LITERATURE_METRICS.md` `DYCK_MDEPTH_PREREG.md` `DYCK_MDEPTH_RESULTS.md` `DYCK_RESULTS_POPE.md` `DYCK_RESULTS_bs128.md` `DYCK_STACK_PROBE.md` `EM_WM_STATE.md` `EM_WM_THEORY.md` `ENWIK8_HIERARCHY.md` `FAMILY_TREE_D7_RESULTS.md` `FAMILY_TREE_GATES.md` `FAMILY_TREE_WM_GAP.md` `FLIPFLOP_GATES.md` `FLIPFLOP_RESULTS.md` `FORGET_CONTROL.md` `FORGET_GATE.md` `FREQ_CONTROL.md` `FRONTIER_ALGORITHMIC.md` `GATED_RESULTS.md` `GATED_SEPARATION.md` `GATED_TORUS.md` `GATE_PROBE.md` `H12_BUDGET_CURVE.md` `HABITAT_BUILD.md` `HIERGOAL_ABLATION.md` `HIERGOAL_CLOSEDLOOP.md` `HIER_PARITY.md` `HIER_RECHECK.md` `HORIZON_RESULTS.md` `INDEX_BASELINE_PAPER_TASK_n8.md` `INDIRECT_OOD.md` `INDIRECT_RESULTS.md` `INDIRECT_RESULTS_200k.md` `JSB_LENGTH_RESULTS_BASE.md` `JSB_LENGTH_RESULTS_RANK.md` `JSB_RESULTS.md` `KNOB_SWEEP.md` `KNOB_SWEEP_n8.md` `L15_ABLATION.md` `L15_LOOP_2X2.md` `LAMBDA_TRACE.md` `LANGUAGE_LANDSCAPE.md` `LEVEL15_MEETS_GATED_matchq.md` `LEVEL15_MEETS_GATED_paper50.md` `LM200_ABLATION.md` `LOCALISATION.md` `LOCALISATION_RANK.md` `LONG_SEQ_clean.md` `LOOPED_PILOT.md` `LOOP_DEPTH_STRATA.md` `LOOP_HEADROOM.md` `LOOP_HIER_COMPUTE.md` `LOOP_HIER_PARITY.md` `LOOP_RANK_PREREG.md` `LOOP_RANK_RESULTS.md` `LOOP_SAMPLED.md` `MAGONLY_RESULTS.md` `MAPPOPE_R4_RESULTS.md` `MATCH_GATES_128_16.md` `MATCH_GATES_64_4.md` `MATCH_QUERY_EM.md` `MATCH_QUERY_GATES.md` `MATCH_QUERY_GATES_P010.md` `MATCH_QUERY_LONGQ.md` `MATCH_QUERY_NOISE_ABLATION.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `MINIGRID_ALLOCENTRIC_2X2X2.md` `MINIGRID_EM.md` `MINIGRID_EM_FIX.md` `MINIGRID_FULL_2X2X2.md` `MINIGRID_REPRO_CONTROL.md` `MINIWORLD_ENDPOINTS.md` `MINIWORLD_FIXED_FINDINGS.md` `MINIWORLD_FIXED_RESULTS.md` `MINIWORLD_GATE_CONTROL.md` `MONOTONE_RESULTS.md` `MQAR_RESULTS.md` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md` `MQ_RANK_2X2.md` `N3_AUDIT.md` `N5_RESULTS.md` `ND_GATES.md` `NOISE_CLEAN_REVALIDATION.md` `NOISE_REFINE.md` `NOLEAK_RESULTS.md` `PAIRCONST_RESULTS.md` `PAIRORIGIN_RESULTS.md` `PAIRSPLIT_RESULTS.md` `PAPER2X2_RESULTS.md` `PAPERTASK_RESULTS.md` `PAPER_FIG4_EM.md` `PAPER_FIG4_REPRO.md` `PAPER_OOD_EXTENDED_n8.md` `PAPER_OOD_RERUN.md` `PAPER_TASK_ABLATION.md` `PAPER_TASK_ACCURACY.md` `PAPER_TASK_FLOORS.md` `PAPER_VALIDATION.md` `PLANNER_TASK_AUDIT.md` `POPE_WRAPPING.md` `POSITION_EFFECT_CONVERGED.md` `RANK_MATCHED_PREREG.md` `RANK_MATCHED_RESULTS.md` `RANK_MI_ANALYSIS.txt` `RANK_MI_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_PREREG.md` `RANK_SEP_RESULTS.md` `RANK_SWEEP.md` `RANK_TRUNCATION.md` `RECENCY_EM_RESULTS.md` `RECENCY_GATES_K4SET.md` `RECENCY_GATES_K64.md` `RECENCY_GATE_ABLATION.md` `RECENCY_H2.md` `RECENCY_RESULTS.md` `RECIPE_POWER.md` `REFINE_RESULTS.md` `RESULTS_PAPER.md` `REVISIT_DISTANCE.md` `ROPE_CANONICAL.md` `SEARCH_RESULTS.md` `SELECTIVE_ROPE.md` `SIGN_ABLATION.md` `SIGN_ABLATION_PREREG.md` `SIGN_MATCHED_RESULTS.md` `SIGN_PROBE.md` `SPREAD2_RESULTS.md` `SPREAD_RESULTS.md` `STITCH_ATTENTION.md` `T1_RESULTS.md` `T2_RESULTS.md` `T3GEN_RESULTS.md` `T3_RESULTS.md` `TALE_OF_TWO_ALGORITHMS.md` `TEM_BACKGROUND_BASELINES.md` `TEM_NOISE_FFN_RESULTS.md` `TEM_T_MULTISEED.md` `THEORY_MAPPOPE.md` `THEORY_SEARCH_AND_LENGTH.md` `TIMING_BENCHMARK.md` `UNFREEZE_RESULTS.md` `VISITS_TEST.md` `VOCAB_EM.md` `WARM_RESULTS.md` `_DOF_TORUS_RAW.md` `_PAPER2X2_RAW.json`

#### B. Code (rule 18)

Literal names are listed per file with the file's last commit date. Files touched since 2026-10-01 or used by every session (guards, stats, figures, catalogue) come first. `(del)` marks a name whose file is proposed for deletion.

<details><summary>696 code files</summary>

- `make_figures.py` (2026-08-16; 4): `FAMILY_TREE_GATES.json` `MATCH_QUERY_EM.md` `MATCH_QUERY_LONGQ.json` `TIMING_BENCHMARK.json`
- `stats_core.py` (2026-09-24; 1): `RANK_PROJ_ANALYSIS.txt`
- `stats_guard.py` (2026-09-24; 4): `AUDIT_2026-09-10.md` `D5_RESULTS.md` `_D5_N24.json` `_MAGONLY.json`
- `test_guards.py` (2026-09-11; 5): `D5_RESULTS.md` `MAGONLY_RESULTS.md` `_D5_N24.json` `_MAGONLY.json` `_REWIND_PROBE.json`
- `analyze_accounts.py` (2026-10-14; 2): `ACCOUNTS_PREREG.md` `ACCOUNTS_VERDICTS.json`
- `analyze_clock_side.py` (2026-10-12; 3): `CLOCK_SIDE_EVAL.json` `CLOCK_SIDE_PREREG.md` `CLOCK_SIDE_VERDICTS.json`
- `analyze_clock_speed.py` (2026-10-12; 2): `CLOCK_SPEED_PREREG.md` `CLOCK_SPEED_VERDICTS.json`
- `analyze_clock_speed_rep.py` (2026-10-13; 3): `CLOCK_SPEED_REP_PREREG.md` `CLOCK_SPEED_REP_VERDICTS.json` `CLOCK_SPEED_VERDICTS.json`
- `analyze_delta_free.py` (2026-10-20; 3): `DELTA_FREE_PREREG.md` `DELTA_FREE_VERDICTS.json` `_DELTA_FREE_RAW.json`
- `analyze_exptorus.py` (2026-10-26; 2): `EXPTORUS_PREREG.md` `EXPTORUS_VERDICTS.json`
- `analyze_factorial.py` (2026-10-12; 2): `FACTORIAL_PREREG.md` `FACTORIAL_VERDICTS.json`
- `analyze_factorial_n8.py` (2026-10-24; 3): `FACTORIAL_ANALYSIS.txt` `FACTORIAL_L1_STRATA.json` `FACTORIAL_N8_PREREG.md`
- `analyze_gain_grain.py` (2026-10-05; 2): `GAIN_GRAIN_EVAL.json` `GAIN_GRAIN_PREREG.md`
- `analyze_gain_phase.py` (2026-10-06; 3): `GAIN_PHASE_EVAL.json` `GAIN_PHASE_PREREG.md` `GAIN_PHASE_VERDICTS.json`
- `analyze_gauge.py` (2026-10-15; 2): `GAUGE_PREREG.md` `GAUGE_VERDICTS.json`
- `analyze_generality.py` (2026-10-14; 2): `GENERALITY_PREREG.md` `GENERALITY_VERDICTS.json`
- `analyze_image_torus.py` (2026-10-25; 3): `IMAGE_TORUS_GATE.json` `IMAGE_TORUS_PREREG.md` `IMAGE_TORUS_VERDICTS.json`
- `analyze_leak.py` (2026-10-03; 2): `LEAK.json` `LEAK_PREREG.md`
- `analyze_leak_secondary.py` (2026-10-03; 2): `LEAK.json` `LEAK_PREREG.md`
- `analyze_lrsweep.py` (2026-10-16; 2): `LRSWEEP_PREREG.md` `LRSWEEP_VERDICTS.json`
- `analyze_mappope_pair.py` (2026-10-04; 3): `MAPPOPE_PAIR_PREREG.md` `MAPPOPE_PAIR_R2.json` `MAPPOPE_PAIR_R4.json`
- `analyze_mm.py` (2026-10-27; 1): `MEMORY_MAZE_PREREG.md`
- `analyze_mm_pilot.py` (2026-10-27; 1): `MEMORY_MAZE_PREREG.md`
- `analyze_rank_nowrap.py` (2026-10-08; 1): `RANK_NOWRAP_PREREG.md`
- `analyze_rank_tests.py` (2026-10-11; 2): `RANK_TESTS_PREREG.md` `RANK_TESTS_VERDICTS.json`
- `analyze_rank_wrap.py` (2026-10-01; 2): `RANK_WRAP.json` `RANK_WRAP_PREREG.md`
- `analyze_rank_wrap_secondary.py` (2026-10-01; 2): `RANK_WRAP.json` `RANK_WRAP_PREREG.md`
- `analyze_rankclock.py` (2026-10-23; 4): `RANKCLOCK.json` `RANKCLOCK_PREREG.md` `RANKCLOCK_VERDICTS.json` `RANKCLOCK_WD0.json`
- `analyze_rankrecipe.py` (2026-10-24; 2): `RANKRECIPE_PREREG.md` `RANKRECIPE_VERDICTS.json`
- `analyze_rankwd.py` (2026-10-23; 4): `RANKWD.json` `RANKWD_PREREG.md` `RANKWD_VERDICTS.json` `RANKWD_WD0.json`
- `analyze_ring_budget.py` (2026-10-12; 2): `RING_BUDGET_PREREG.md` `RING_BUDGET_VERDICTS.json`
- `analyze_ring_dial.py` (2026-10-12; 3): `CANCEL_RESULTS.md` `RING_DIAL_PREREG.md` `RING_DIAL_VERDICTS.json`
- `analyze_rnn_torus.py` (2026-10-26; 2): `RNN_TORUS_CD.json` `RNN_TORUS_PREREG.md`
- `analyze_robust.py` (2026-10-16; 2): `ROBUST_PREREG.md` `ROBUST_VERDICTS.json`
- `analyze_score_rank.py` (2026-10-05; 7): `SCORE_RANK_PREREG.md` `SCORE_RANK_R2.json` `SCORE_RANK_R4.json` `SCORE_RANK_RESCORE_R2.json` `SCORE_RANK_RESCORE_R4.json` `SCORE_RANK_STRATA_R2.json` `SCORE_RANK_STRATA_R4.json`
- `analyze_signroute.py` (2026-10-21; 4): `LRSWEEP_RESULTS.md` `SIGNROUTE_PREREG.md` `SIGNROUTE_RESULTS.md` `SIGNROUTE_VERDICTS.json`
- `analyze_signsep.py` (2026-10-23; 3): `LRSWEEP_RESULTS.md` `SIGNSEP_PREREG.md` `SIGNSEP_VERDICTS.json`
- `analyze_srope_torus.py` (2026-10-20; 2): `SROPE_TORUS_PREREG.md` `SROPE_TORUS_VERDICTS.json`
- `analyze_ssm_sign.py` (2026-10-24; 2): `G3_SSM_PREREG.md` `G3_SSM_VERDICTS.json`
- `analyze_torus_lrsweep.py` (2026-10-23; 2): `TORUS_LRSWEEP_PREREG.md` `TORUS_LRSWEEP_VERDICTS.json`
- `analyze_torus_start.py` (2026-10-14; 2): `TORUS_START_PREREG.md` `TORUS_START_VERDICTS.json`
- `analyze_tw_ambig.py` (2026-10-08; 1): `TW_AMBIG_PREREG.md`
- `analyze_tw_landmark.py` (2026-10-08; 1): `TW_LANDMARK_PREREG.md`
- `analyze_tw_normstep.py` (2026-10-03; 2): `TW_NORMSTEP.json` `TW_NORMSTEP_PREREG.md`
- `analyze_tw_statechange.py` (2026-10-08; 3): `TW_STATECHANGE.json` `TW_STATECHANGE_PREREG.md` `TW_STATECHANGE_VERDICTS.json`
- `analyze_wrap_cope.py` (2026-10-14; 2): `WRAP_COPE_PREREG.md` `WRAP_COPE_VERDICTS.json`
- `analyze_wrap_existence.py` (2026-10-13; 3): `CANCEL_RESULTS.md` `WRAP_EXISTENCE_PREREG.md` `WRAP_EXISTENCE_VERDICTS.json`
- `analyze_wrap_mech.py` (2026-10-14; 2): `MECHANISM_PREREG.md` `MECHANISM_VERDICTS.json`
- `build_rank_lift.py` (2026-10-11; 1): `RANK_TESTS_PREREG.md`
- `construct_signroute.py` (2026-10-20; 2): `LRSWEEP_RESULTS.md` `SIGNROUTE_PREREG.md`
- `construct_ssm_sign.py` (2026-10-23; 1): `G3_SSM_PREREG.md`
- `docs/audits/2026-10-03/causal_whatwhere.py` (2026-10-03; 1): `_PAPER2X2_RAW.json`
- `docs/audits/2026-10-03/probe_whatwhere_nd.py` (2026-10-03; 3): `RANK_ND.json` `RANK_ND_SECONDARY.json` `RANK_WRAP.json`
- `docs/audits/2026-10-03/tw_normstep_floors.py` (2026-10-03; 1): `TW_NORMSTEP_PREREG.md`
- `docs/audits/2026-10-04/dironly_aside_errors.py` (2026-10-04; 1): `TW_NORMSTEP_RESULTS.md`
- `docs/audits/2026-10-04/mappope_pair_power.py` (2026-10-04; 1): `_PAPER2X2_RAW.json`
- `docs/audits/2026-10-04/rescore_other_compare.py` (2026-10-04; 2): `LEAK.json` `RANK_ND.json`
- `docs/audits/2026-10-04/rescore_torus.sh` (2026-10-04; 1): `_PAPER2X2_RAW.json`
- `docs/audits/2026-10-04/rescore_torus_compare.py` (2026-10-04; 7): `LOOP_RANK.json` `LOOP_RANK_E1800_P1.json` `RANK3.json` `RANK_MI.json` `RANK_SEP.json` `SIGN_MATCHED.json` `_PAPER2X2_RAW.json`
- `docs/audits/2026-10-04/tw_normstep_wordsteps.py` (2026-10-04; 2): `TW_NORMSTEP.json` `TW_NORMSTEP_RESULTS.md`
- `docs/audits/2026-10-05/gain_grain_equiv.py` (2026-10-05; 1): `GAIN_GRAIN_PREREG.md`
- `docs/audits/2026-10-05/gain_grain_power.py` (2026-10-05; 2): `GAIN_GRAIN_PREREG.md` `MAPPOPE_PAIR_R2.json`
- `docs/audits/2026-10-05/gain_grain_remap.py` (2026-10-05; 1): `GAIN_GRAIN_PREREG.md`
- `docs/audits/2026-10-05/gain_grain_rescore.py` (2026-10-05; 2): `GAIN_GRAIN_EVAL.json` `GAIN_GRAIN_RESCORE.json`
- `docs/audits/2026-10-05/gain_grain_smoke.py` (2026-10-05; 1): `GAIN_GRAIN_PREREG.md`
- `docs/audits/2026-10-05/mappope_pair_ci.py` (2026-10-04; 1): `MAPPOPE_PAIR_R2.json`
- `docs/audits/2026-10-05/neuro_rank2_mech.py` (2026-10-05; 2): `MAPPOPE_PAIR_R2.json` `MAPPOPE_PAIR_R4.json`
- `docs/audits/2026-10-05/score_rank_init_check.py` (2026-10-05; 1): `SCORE_RANK_PREREG.md`
- `docs/audits/2026-10-05/score_rank_power.py` (2026-10-05; 2): `RANK_MI.json` `SCORE_RANK_PREREG.md`
- `docs/audits/2026-10-06/gain_phase_equiv.py` (2026-10-06; 1): `GAIN_PHASE_PREREG.md`
- `docs/audits/2026-10-06/gain_phase_leak_validate.py` (2026-10-06; 1): `LEAK.json`
- `docs/audits/2026-10-06/gain_phase_repro.py` (2026-10-06; 1): `GAIN_PHASE_PREREG.md`
- `docs/audits/2026-10-06/gain_phase_secondary.py` (2026-10-06; 2): `GAIN_PHASE_EVAL.json` `GAIN_PHASE_PREREG.md`
- `docs/audits/2026-10-06/gain_phase_smoke.py` (2026-10-06; 1): `GAIN_PHASE_EVAL.json`
- `docs/audits/2026-10-06/rank_nowrap_gate.py` (2026-10-06; 1): `RANK_NOWRAP_PREREG.md`
- `docs/audits/2026-10-06/rank_nowrap_power.py` (2026-10-06; 1): `RANK_NOWRAP_PREREG.md`
- `docs/audits/2026-10-06/rank_nowrap_repro.py` (2026-10-06; 1): `RANK_NOWRAP_PREREG.md`
- `docs/audits/2026-10-08/gate_tw_ambig.py` (2026-10-08; 1): `TW_AMBIG_PREREG.md`
- `docs/audits/2026-10-08/gate_tw_statechange.py` (2026-10-08; 1): `TW_STATECHANGE_PREREG.md`
- `docs/audits/2026-10-08/tw_ambig_checks.py` (2026-10-08; 1): `TW_AMBIG_PREREG.md`
- `docs/audits/2026-10-08/tw_ambig_power.py` (2026-10-08; 1): `TW_NORMSTEP.json`
- `docs/audits/2026-10-08/tw_ambig_repro.py` (2026-10-08; 1): `TW_AMBIG_PREREG.md`
- `docs/audits/2026-10-08/tw_landmark/gate_tw_landmark.py` (2026-10-08; 1): `TW_LANDMARK_PREREG.md`
- `docs/audits/2026-10-08/tw_landmark/tw_landmark_checks.py` (2026-10-08; 1): `TW_LANDMARK_PREREG.md`
- `docs/audits/2026-10-08/tw_landmark/tw_landmark_power.py` (2026-10-08; 1): `TW_LANDMARK_PREREG.md`
- `docs/audits/2026-10-08/tw_statechange_calib.py` (2026-10-08; 1): `TW_STATECHANGE_PREREG.md`
- `docs/audits/2026-10-08/tw_statechange_checks.py` (2026-10-08; 1): `TW_STATECHANGE_PREREG.md`
- `docs/audits/2026-10-08/tw_statechange_power.py` (2026-10-08; 1): `TW_STATECHANGE_PREREG.md`
- `docs/audits/2026-10-08/tw_statechange_repro.py` (2026-10-08; 1): `TW_STATECHANGE_PREREG.md`
- `docs/audits/2026-10-08/tw_statechange_smoke.py` (2026-10-08; 1): `TW_STATECHANGE_PREREG.md`
- `docs/audits/2026-10-09/queue_tw_statechange_pilot.sh` (2026-10-09; 1): `TW_STATECHANGE_PREREG.md`
- `docs/audits/2026-10-09/tinystories_probe.py` (2026-10-09; 1): `TINYSTORIES_PILOT.md`
- `docs/audits/2026-10-11/rank_tests_checks.py` (2026-10-11; 2): `CANCEL_PREREG.md` `RANK_TESTS_PREREG.md`
- `docs/audits/2026-10-11/rank_tests_pilot.sh` (2026-10-11; 1): `RANK_TESTS_PREREG.md`
- `docs/audits/2026-10-11/rank_tests_secondary.py` (2026-10-11; 1): `RANK_TESTS_PREREG.md`
- `docs/audits/2026-10-12/alpha_language.py` (2026-10-12; 2): `LOCALISATION.md` `RECENCY_RESULTS.md`
- `docs/audits/2026-10-12/clock_side_calib.sh` (2026-10-12; 1): `CLOCK_SIDE_PREREG.md`
- `docs/audits/2026-10-14/cope_diagnosis/m1_measure.py` (2026-10-14; 1): `WRAP_COPE_ANALYSIS.txt`
- `docs/audits/2026-10-15/reanalyses/solved_thresholds.py` (2026-10-15; 10): `ACCOUNTS_RESULTS.md` `GENERALITY_RESULTS.md` `MECHANISM_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_RESULTS.md` `SIGN_MATCHED.json` `SIGN_MATCHED_RESULTS.md` `TORUS_START_RESULTS.md` `WRAP_COPE_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/audits/2026-10-16/lrsweep_check/indep.py` (2026-10-16; 1): `LRSWEEP_PREREG.md`
- `docs/audits/2026-10-20/causal_whatwhere_T1024.py` (2026-10-20; 1): `SIGN_MATCHED.json`
- `docs/audits/2026-10-27/rnn_torus/verify_numbers.py` (2026-10-27; 1): `RNN_TORUS_ANALYSIS.txt`
- `docs/audits/2026-10-28/memory_maze/pose_only_probe.py` (2026-10-28; 1): `MEMORY_MAZE_ANALYSIS.txt`
- `docs/audits/2026-10-28/memory_maze/verify_numbers.py` (2026-10-28; 1): `MEMORY_MAZE_RESULTS.md`
- `docs/paper/expansion/scripts/depth/dyck_distance.py` (2026-10-20; 1): `DYCK_MDEPTH_RESULTS.md`
- `docs/paper/expansion/scripts/depth/torus_depth.py` (2026-10-20; 3): `FACTORIAL_L1_STRATA.json` `FACTORIAL_L4_STRATA.json` `FACTORIAL_LOOP_STRATA.json`
- `docs/paper/expansion/scripts/language/tw_common_used.py` (2026-10-20; 1): `TEXTWORLD_SECONDARY.json`
- `docs/paper/expansion/scripts/recurrence/common.py` (2026-10-20; 1): `LOOP_RANK_PREREG.md`
- `docs/paper/expansion/scripts/recurrence/summarize.py` (2026-10-20; 2): `LOOP_RANK.json` `RANK_MI.json`
- `docs/paper/figs/fig_exchange_rate.py` (2026-10-16; 1): `DYCK_MDEPTH_RESULTS.json`
- `docs/paper/figs/fig_existence.py` (2026-10-16; 4): `GENERALITY_ANALYSIS.txt` `RANK_PROJ_ANALYSIS.txt` `RANK_SEP_ANALYSIS.txt` `WRAP_EXISTENCE_ANALYSIS.txt`
- `docs/paper/figs/fig_rank_solved.py` (2026-10-16; 5): `RANK3_ANALYSIS.txt` `RANK_ND_ANALYSIS.txt` `RANK_SEP_ANALYSIS.txt` `RANK_TESTS_ANALYSIS.txt` `RANK_WRAP_ANALYSIS.txt`
- `docs/paper/figs/fig_strata_modes.py` (2026-10-16; 6): `FACTORIAL_L1_STRATA.json` `FACTORIAL_L4_STRATA.json` `FACTORIAL_LOOP_STRATA.json` `RANK3_STRATA.json` `RANK_MI_STRATA.json` `SIGN_MATCHED_STRATA.json`
- `docs/paper/figs/fig_summary.py` (2026-10-21; 4): `FACTORIAL_RESULTS.md` `GENERALITY_RESULTS.md` `RANK_PROJ_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/figs/fig_textworld.py` (2026-10-16; 2): `TEXTWORLD_RESULTS.md` `TEXTWORLD_SECONDARY.json`
- `docs/paper/figs/torus_strata_floors.py` (2026-10-16; 1): `SIGN_MATCHED_STRATA.json`
- `environment_steps.py` (2026-10-15; 1): `GAUGE_PREREG.md`
- `environment_tw_ambig.py` (2026-10-08; 1): `TW_AMBIG_PREREG.md`
- `environment_tw_landmark.py` (2026-10-08; 1): `TW_LANDMARK_PREREG.md`
- `environment_tw_statechange.py` (2026-10-08; 1): `TW_STATECHANGE_PREREG.md`
- `eval_clock_side.py` (2026-10-12; 2): `CLOCK_SIDE_EVAL.json` `CLOCK_SIDE_PREREG.md`
- `eval_exptorus.py` (2026-10-26; 1): `EXPTORUS_PREREG.md`
- `eval_gain_phase.py` (2026-10-06; 2): `GAIN_PHASE_EVAL.json` `GAIN_PHASE_PREREG.md`
- `eval_image_torus.py` (2026-10-25; 1): `IMAGE_TORUS_PREREG.md`
- `eval_pair.py` (2026-10-04; 1): `MAPPOPE_PAIR_PREREG.md`
- `gain_phase_eval.py` (2026-10-06; 1): `GAIN_PHASE_PREREG.md`
- `gate_image_torus.py` (2026-10-25; 2): `IMAGE_TORUS_GATE.json` `IMAGE_TORUS_PREREG.md`
- `gate_ssm_sign.py` (2026-10-23; 1): `G3_SSM_PREREG.md`
- `image_bank.py` (2026-10-25; 1): `IMAGE_TORUS_PREREG.md`
- `leak_eval.py` (2026-10-03; 1): `LEAK_PREREG.md`
- `mm_data.py` (2026-10-27; 2): `MEMORY_MAZE_GATE_V3.md` `MEMORY_MAZE_PREREG.md`
- `mm_encoder.py` (2026-10-27; 1): `MEMORY_MAZE_PREREG.md`
- `mm_floors.py` (2026-10-27; 1): `MEMORY_MAZE_PREREG.md`
- `mm_gate.py` (2026-10-26; 1): `MEMORY_MAZE_GATE.md`
- `mm_gate_v2.py` (2026-10-27; 1): `MEMORY_MAZE_GATE_V2.md`
- `mm_gate_v3.py` (2026-10-27; 1): `MEMORY_MAZE_GATE_V3.md`
- `model_codes.py` (2026-10-03; 1): `LEAK_PREREG.md`
- `model_em_pope.py` (2026-10-05; 1): `GAIN_GRAIN_PREREG.md`
- `model_exptorus.py` (2026-10-26; 1): `EXPTORUS_PREREG.md`
- `model_gain_phase.py` (2026-10-06; 2): `GAIN_PHASE_PREREG.md` `LEAK_RESULTS.md`
- `model_image.py` (2026-10-25; 1): `IMAGE_TORUS_PREREG.md`
- `model_mm.py` (2026-10-27; 1): `MEMORY_MAZE_PREREG.md`
- `model_mono_baselines.py` (2026-10-14; 3): `GENERALITY_PREREG.md` `MECHANISM_RESULTS.md` `WRAP_COPE_PREREG.md`
- `model_pope_delta.py` (2026-10-20; 1): `DELTA_FREE_PREREG.md`
- `model_pope_pair.py` (2026-10-04; 1): `MAPPOPE_PAIR_PREREG.md`
- `model_pope_pair_mi.py` (2026-10-05; 1): `SCORE_RANK_PREREG.md`
- `model_rankclock.py` (2026-10-23; 1): `RANK_MI_RESULTS.md`
- `model_ring_rank.py` (2026-10-11; 1): `RANK_TESTS_PREREG.md`
- `model_rnn.py` (2026-10-26; 2): `FACTORIAL_N8_PREREG.md` `RNN_TORUS_PREREG.md`
- `model_signroute.py` (2026-10-20; 1): `SIGNROUTE_PREREG.md`
- `model_signsep.py` (2026-10-23; 1): `SIGNSEP_PREREG.md`
- `model_srope.py` (2026-10-20; 1): `SROPE_TORUS_PREREG.md`
- `model_ssm_sign.py` (2026-10-23; 1): `G3_SSM_PREREG.md`
- `model_textstep.py` (2026-10-03; 1): `TW_NORMSTEP_PREREG.md`
- `model_torus_start.py` (2026-10-14; 1): `TORUS_START_PREREG.md`
- `model_tw_ambig.py` (2026-10-08; 1): `TW_AMBIG_PREREG.md`
- `model_wrap_exist.py` (2026-10-14; 1): `WRAP_EXISTENCE_PREREG.md`
- `ring_rank_readouts.py` (2026-10-11; 1): `RANK_TESTS_PREREG.md`
- `run_accounts.sh` (2026-10-14; 3): `ACCOUNTS_ANALYSIS.txt` `ACCOUNTS_PREREG.md` `ACCOUNTS_VERDICTS.json`
- `run_clock_side.sh` (2026-10-12; 4): `CLOCK_SIDE_ANALYSIS.txt` `CLOCK_SIDE_EVAL.json` `CLOCK_SIDE_PREREG.md` `CLOCK_SIDE_VERDICTS.json`
- `run_clock_speed.sh` (2026-10-12; 3): `CLOCK_SPEED_ANALYSIS.txt` `CLOCK_SPEED_PREREG.md` `CLOCK_SPEED_VERDICTS.json`
- `run_clock_speed_rep.sh` (2026-10-12; 3): `CLOCK_SPEED_REP_ANALYSIS.txt` `CLOCK_SPEED_REP_PREREG.md` `CLOCK_SPEED_REP_VERDICTS.json`
- `run_delta_free.sh` (2026-10-20; 4): `DELTA_FREE_ANALYSIS.txt` `DELTA_FREE_PREREG.md` `DELTA_FREE_VERDICTS.json` `_DELTA_FREE_RAW.md`
- `run_exptorus.sh` (2026-10-26; 3): `EXPTORUS_ANALYSIS.txt` `EXPTORUS_PREREG.md` `EXPTORUS_VERDICTS.json`
- `run_factorial.sh` (2026-10-12; 3): `FACTORIAL_ANALYSIS.txt` `FACTORIAL_PREREG.md` `FACTORIAL_VERDICTS.json`
- `run_factorial_n8.sh` (2026-10-24; 12): `FACTORIAL_L1.json` `FACTORIAL_L1_STRATA.json` `FACTORIAL_L4.json` `FACTORIAL_L4_STRATA.json` `FACTORIAL_LOOP.json` `FACTORIAL_LOOP_STRATA.json` `FACTORIAL_N8_ANALYSIS.txt` `FACTORIAL_N8_L1.json` `FACTORIAL_N8_L4.json` `FACTORIAL_N8_LOOP.json` `FACTORIAL_N8_PREREG.md` `FACTORIAL_N8_VERDICTS.json`
- `run_factorial_n8_pilot.sh` (2026-10-24; 1): `FACTORIAL_N8_PREREG.md`
- `run_gain_grain.sh` (2026-10-05; 5): `GAIN_GRAIN_ANALYSIS.txt` `GAIN_GRAIN_EVAL.json` `GAIN_GRAIN_EVAL.md` `GAIN_GRAIN_PREREG.md` `GAIN_GRAIN_RESCORE.md`
- `run_gain_phase.sh` (2026-10-06; 4): `GAIN_PHASE_ANALYSIS.txt` `GAIN_PHASE_EVAL.json` `GAIN_PHASE_PREREG.md` `GAIN_PHASE_VERDICTS.json`
- `run_gauge.sh` (2026-10-15; 3): `GAUGE_ANALYSIS.txt` `GAUGE_PREREG.md` `GAUGE_VERDICTS.json`
- `run_generality.sh` (2026-10-14; 3): `GENERALITY_ANALYSIS.txt` `GENERALITY_PREREG.md` `GENERALITY_VERDICTS.json`
- `run_image_torus.sh` (2026-10-25; 5): `IMAGE_TORUS_ANALYSIS.txt` `IMAGE_TORUS_GATE.json` `IMAGE_TORUS_GATE.md` `IMAGE_TORUS_PREREG.md` `IMAGE_TORUS_VERDICTS.json`
- `run_image_torus_pilot.sh` (2026-10-25; 2): `IMAGE_TORUS_PILOT.txt` `IMAGE_TORUS_PREREG.md`
- `run_image_torus_pilot2.sh` (2026-10-25; 2): `IMAGE_TORUS_PILOT2.txt` `IMAGE_TORUS_PREREG.md`
- `run_leak.sh` (2026-10-03; 3): `LEAK.json` `LEAK_ANALYSIS.txt` `LEAK_PREREG.md`
- `run_lrsweep.sh` (2026-10-16; 3): `LRSWEEP_ANALYSIS.txt` `LRSWEEP_PREREG.md` `LRSWEEP_VERDICTS.json`
- `run_mappope_pair.sh` (2026-10-04; 6): `MAPPOPE_PAIR_ANALYSIS.txt` `MAPPOPE_PAIR_PREREG.md` `MAPPOPE_PAIR_R2.json` `MAPPOPE_PAIR_R2.md` `MAPPOPE_PAIR_R4.json` `MAPPOPE_PAIR_R4.md`
- `run_mm_pilot.sh` (2026-10-27; 1): `MEMORY_MAZE_PREREG.md`
- `run_mm_pilot3.sh` (2026-10-27; 1): `MEMORY_MAZE_PREREG.md`
- `run_rank_nowrap.sh` (2026-10-06; 1): `RANK_NOWRAP_PREREG.md`
- `run_rank_tests.sh` (2026-10-11; 3): `RANK_TESTS_ANALYSIS.txt` `RANK_TESTS_PREREG.md` `RANK_TESTS_VERDICTS.json`
- `run_rank_wrap.sh` (2026-10-01; 3): `RANK_WRAP.json` `RANK_WRAP_ANALYSIS.txt` `RANK_WRAP_PREREG.md`
- `run_rankclock.sh` (2026-10-23; 7): `RANKCLOCK.json` `RANKCLOCK.md` `RANKCLOCK_ANALYSIS.txt` `RANKCLOCK_PREREG.md` `RANKCLOCK_VERDICTS.json` `RANKCLOCK_WD0.json` `RANKCLOCK_WD0.md`
- `run_rankclock_existence.sh` (2026-10-23; 1): `RANKCLOCK_PREREG.md`
- `run_rankrecipe.sh` (2026-10-24; 7): `RANKRECIPE_ANALYSIS.txt` `RANKRECIPE_PREREG.md` `RANKRECIPE_VERDICTS.json` `RANKRECIPE_r2_R0.json` `RANKRECIPE_r2_R1.json` `RANKRECIPE_r2_R2.json` `RANKRECIPE_r4_R0.json`
- `run_rankwd.sh` (2026-10-23; 7): `RANKWD.json` `RANKWD.md` `RANKWD_ANALYSIS.txt` `RANKWD_PREREG.md` `RANKWD_VERDICTS.json` `RANKWD_WD0.json` `RANKWD_WD0.md`
- `run_ring_budget.sh` (2026-10-12; 3): `RING_BUDGET_ANALYSIS.txt` `RING_BUDGET_PREREG.md` `RING_BUDGET_VERDICTS.json`
- `run_ring_dial.sh` (2026-10-12; 3): `RING_DIAL_ANALYSIS.txt` `RING_DIAL_PREREG.md` `RING_DIAL_VERDICTS.json`
- `run_rnn_torus.sh` (2026-10-26; 3): `RNN_TORUS_ANALYSIS.txt` `RNN_TORUS_PREREG.md` `RNN_TORUS_VERDICTS.json`
- `run_rnn_torus_pilot.sh` (2026-10-26; 1): `RNN_TORUS_PREREG.md`
- `run_rnn_torus_pilot2.sh` (2026-10-26; 1): `RNN_TORUS_PREREG.md`
- `run_robust.sh` (2026-10-16; 3): `ROBUST_ANALYSIS.txt` `ROBUST_PREREG.md` `ROBUST_VERDICTS.json`
- `run_score_rank.sh` (2026-10-05; 8): `SCORE_RANK_ANALYSIS.txt` `SCORE_RANK_PREREG.md` `SCORE_RANK_R2.json` `SCORE_RANK_R4.json` `SCORE_RANK_RESCORE_R2.json` `SCORE_RANK_RESCORE_R4.json` `SCORE_RANK_STRATA_R2.json` `SCORE_RANK_STRATA_R4.json`
- `run_signroute.sh` (2026-10-20; 3): `SIGNROUTE_ANALYSIS.txt` `SIGNROUTE_PREREG.md` `SIGNROUTE_VERDICTS.json`
- `run_signsep.sh` (2026-10-23; 3): `SIGNSEP_ANALYSIS.txt` `SIGNSEP_PREREG.md` `SIGNSEP_VERDICTS.json`
- `run_signsep_pilot.sh` (2026-10-23; 1): `SIGNSEP_PREREG.md`
- `run_signsep_pilot2.sh` (2026-10-23; 1): `SIGNSEP_PREREG.md`
- `run_srope_torus.sh` (2026-10-20; 1): `SROPE_TORUS_PREREG.md`
- `run_ssm_sign.sh` (2026-10-24; 5): `G3_SSM_A2_ANALYSIS.txt` `G3_SSM_A2_VERDICTS.json` `G3_SSM_ANALYSIS.txt` `G3_SSM_PREREG.md` `G3_SSM_VERDICTS.json`
- `run_tinystories.sh` (2026-10-09; 1): `TINYSTORIES_PILOT.md`
- `run_torus_lrsweep.sh` (2026-10-23; 3): `TORUS_LRSWEEP_ANALYSIS.txt` `TORUS_LRSWEEP_PREREG.md` `TORUS_LRSWEEP_VERDICTS.json`
- `run_torus_start.sh` (2026-10-14; 3): `TORUS_START_ANALYSIS.txt` `TORUS_START_PREREG.md` `TORUS_START_VERDICTS.json`
- `run_tw_ambig.sh` (2026-10-08; 1): `TW_AMBIG_PREREG.md`
- `run_tw_landmark.sh` (2026-10-08; 1): `TW_LANDMARK_PREREG.md`
- `run_tw_landmark_pilot.sh` (2026-10-08; 1): `TW_LANDMARK_PREREG.md`
- `run_tw_normstep.sh` (2026-10-03; 3): `TW_NORMSTEP.json` `TW_NORMSTEP_ANALYSIS.txt` `TW_NORMSTEP_PREREG.md`
- `run_tw_statechange.sh` (2026-10-08; 5): `TW_STATECHANGE.json` `TW_STATECHANGE_ANALYSIS.txt` `TW_STATECHANGE_PREREG.md` `TW_STATECHANGE_READOUTS.txt` `TW_STATECHANGE_VERDICTS.json`
- `run_wrap_cope.sh` (2026-10-14; 3): `WRAP_COPE_ANALYSIS.txt` `WRAP_COPE_PREREG.md` `WRAP_COPE_VERDICTS.json`
- `run_wrap_existence.sh` (2026-10-13; 3): `WRAP_EXISTENCE_ANALYSIS.txt` `WRAP_EXISTENCE_PREREG.md` `WRAP_EXISTENCE_VERDICTS.json`
- `run_wrap_mech.sh` (2026-10-14; 3): `MECHANISM_ANALYSIS.txt` `MECHANISM_PREREG.md` `MECHANISM_VERDICTS.json`
- `test_exptorus.py` (2026-10-26; 1): `EXPTORUS_PREREG.md`
- `test_image_torus.py` (2026-10-25; 1): `IMAGE_TORUS_PREREG.md`
- `test_mm.py` (2026-10-27; 1): `MEMORY_MAZE_PREREG.md`
- `test_rnn_torus.py` (2026-10-26; 1): `RNN_TORUS_PREREG.md`
- `test_signsep.py` (2026-10-23; 1): `SIGNSEP_PREREG.md`
- `train_exptorus.py` (2026-10-26; 1): `EXPTORUS_PREREG.md`
- `train_gain_grain.py` (2026-10-05; 1): `GAIN_GRAIN_PREREG.md`
- `train_gain_phase.py` (2026-10-06; 1): `GAIN_PHASE_PREREG.md`
- `train_gauge.py` (2026-10-16; 2): `GAUGE_PREREG.md` `ROBUST_PREREG.md`
- `train_image.py` (2026-10-25; 1): `IMAGE_TORUS_PREREG.md`
- `train_mm.py` (2026-10-27; 1): `MEMORY_MAZE_PREREG.md`
- `train_pair.py` (2026-10-04; 1): `MAPPOPE_PAIR_PREREG.md`
- `train_rank_nowrap.py` (2026-10-06; 1): `RANK_NOWRAP_PREREG.md`
- `train_ring_dial.py` (2026-10-12; 1): `RING_DIAL_PREREG.md`
- `train_ring_rank.py` (2026-10-11; 1): `RANK_TESTS_PREREG.md`
- `train_ring_variant.py` (2026-10-14; 1): `GENERALITY_PREREG.md`
- `train_rnn.py` (2026-10-26; 1): `RNN_TORUS_PREREG.md`
- `train_score_rank.py` (2026-10-05; 2): `MAPPOPE_PAIR_PREREG.md` `SCORE_RANK_PREREG.md`
- `train_signroute.py` (2026-10-20; 1): `SIGNROUTE_PREREG.md`
- `train_signsep.py` (2026-10-23; 1): `SIGNSEP_PREREG.md`
- `train_ssm_sign.py` (2026-10-23; 1): `G3_SSM_PREREG.md`
- `train_torus_start.py` (2026-10-14; 1): `TORUS_START_PREREG.md`
- `train_tw_ambig.py` (2026-10-08; 1): `TW_AMBIG_PREREG.md`
- `train_tw_landmark.py` (2026-10-08; 1): `TW_LANDMARK_PREREG.md`
- `train_tw_normstep.py` (2026-10-03; 1): `TW_NORMSTEP_PREREG.md`
- `train_tw_statechange.py` (2026-10-08; 1): `TW_STATECHANGE_PREREG.md`
- `train_variant.py` (2026-10-23; 13): `ADDITION_DESIGN.md` `COUNTER_BATCH.md` `DELTA_FREE_PREREG.md` `DOF_PREREG.md` `FAST_ATTN_RANK.md` `MINIGRID_EM.md` `MONOTONE_PREREG.md` `NOLEAK_PREREG.md` `PAIRORIGIN_PREREG.md` `SAMEBLOCK_PREREG.md` `THEORY_KERNEL.md` `UNFREEZE_PREREG.md` `WARM_PREREG.md`
- `train_wrap_exist.py` (2026-10-14; 1): `WRAP_EXISTENCE_PREREG.md`
- `train_wrap_mech.py` (2026-10-14; 1): `MECHANISM_PREREG.md`
- `tw_ambig_readouts.py` (2026-10-08; 1): `TW_AMBIG_PREREG.md`
- `tw_landmark_eval.py` (2026-10-08; 1): `TW_LANDMARK_PREREG.md`
- `tw_normstep_readouts.py` (2026-10-03; 1): `TW_NORMSTEP_PREREG.md`
- `tw_statechange_readouts.py` (2026-10-08; 1): `TW_STATECHANGE_PREREG.md`
- `ablate_context.py` (2026-08-17; 2): `ABLATE_COMPOSITIONAL.md` `ABLATE_FAMILY_TREE.md`
- `ablate_hiergoal.py` (2026-08-09; 1): `HIERGOAL_ABLATION.md`
- `ablate_match_noise.py` (2026-09-24; 1): `MATCH_QUERY_NOISE_ABLATION.md`
- `ablate_paper_task.py` (2026-08-17; 1): `PAPER_TASK_ABLATION.md`
- `ablate_recency_gate.py` (2026-09-07; 1): `RECENCY_GATE_ABLATION.md`
- `agg_algorithmic.py` (2026-09-03; 1): `ALGORITHMIC_GATES.md`
- `agg_cmq.py` (2026-08-23; 1): `COMPOSITIONAL_MATCH_QUERY_RESULTS.md`
- `agg_comp_headroom.py` (2026-09-24; 2): `COMP_HEADROOM.md` `COMP_HEADROOM_PREREG.md`
- `agg_comp_multiseed.py` (2026-09-24; 1): `COMPOSITIONAL_MULTISEED.md`
- `agg_hier_parity.py` (2026-09-03; 1): `ALGORITHMIC_RESULTS.md`
- `agg_loop_hier.py` (2026-09-03; 3): `ALGORITHMIC_RESULTS.md` `HIER_PARITY.md` `LOOP_HIER_COMPUTE.md`
- `agg_looped.py` (2026-08-29; 4): `HORIZON_RESULTS.md` `LOOPED_L1.json` `LOOPED_L4.json` `LOOPED_Loop4.json`
- `agg_miniworld.py` (2026-08-28; 1): `MINIWORLD_FIXED_RESULTS.md`
- `agg_popewrap.py` (2026-09-05; 1): `ALIASING_CONTROLLED.md`
- `agg_recipe.py` (2026-09-02; 1): `RECIPE_CHOICE.json`
- `agg_selective.py` (2026-09-03; 1): `_SELECTIVE_TORUS.json`
- `agg_sign.py` (2026-09-06; 1): `SIGN_ABLATION_PREREG.md`
- `aliasing_covariate.py` (2026-08-19; 2): `ALIASING_COVARIATE.md` `DISSOCIATION_SWEEP.md`
- `analyze_aug.py` (2026-09-20; 1): `AUG_RESULTS.md`
- `analyze_cancel.py` (2026-09-27; 1): `CANCEL_PREREG.md`
- `analyze_code.py` (2026-09-21; 5): `CODE_DELIMS.json` `CODE_GATES.json` `CODE_GATES.md` `CODE_PREREG.md` `CODE_RESULTS.md`
- `analyze_code_fullval.py` (2026-09-27; 1): `CODE_FULLVAL_PREREG.md`
- `analyze_code_long.py` (2026-09-21; 3): `CODE_LONG.json` `CODE_PREREG.md` `CODE_RESULTS_OOD.md`
- `analyze_cross.py` (2026-09-20; 1): `CROSS_RESULTS.md`
- `analyze_ctxstep.py` (2026-09-30; 1): `CTXSTEP_PREREG.md`
- `analyze_dyck.py` (2026-09-15; 3): `DYCK_GATES.json` `DYCK_GATES.md` `DYCK_PREREG.md`
- `analyze_dyck_depth.py` (2026-09-21; 2): `DYCK_DEPTH_PREREG.md` `DYCK_DEPTH_RESULTS.md`
- `analyze_dyck_ladder.py` (2026-09-24; 1): `DYCK_LADDER_PREREG.md`
- `analyze_dyck_mdepth.py` (2026-09-25; 3): `DYCK_MDEPTH_GATES.json` `DYCK_MDEPTH_GATES.md` `DYCK_MDEPTH_PREREG.md`
- `analyze_indirect.py` (2026-09-16; 1): `INDIRECT_PREREG.md`
- `analyze_jsb.py` (2026-09-15; 1): `JSB_PREREG.md`
- `analyze_jsb_base.py` (2026-09-17; 1): `JSBLEN_PREREG.md`
- `analyze_jsb_len.py` (2026-09-17; 1): `JSBLEN_PREREG.md`
- `analyze_loop_rank.py` (2026-09-26; 1): `LOOP_RANK_PREREG.md`
- `analyze_loop_rank_e1800.py` (2026-09-27; 1): `LOOP_RANK_E1800_PREREG.md`
- `analyze_loop_rank_e1800_budget.py` (2026-09-28; 3): `LOOP_RANK_E1800_P1.json` `LOOP_RANK_E1800_PREREG.md` `RANK_MI.json`
- `analyze_monotone.py` (2026-09-13; 5): `MONOTONE_PREREG.md` `MONOTONE_RAW.json` `MONOTONE_RAW.md` `_MONOTONE_TORUS.json` `_SIGN_RAW.json`
- `analyze_noleak.py` (2026-09-11; 2): `NOLEAK_PREREG.md` `_NOLEAK.json`
- `analyze_pairconst.py` (2026-09-12; 1): `PAIRCONST_PREREG.md`
- `analyze_pairorigin.py` (2026-09-11; 3): `PAIRORIGIN_PREREG.md` `REPORT.md` `_PAIRORIGIN.json`
- `analyze_pairsplit.py` (2026-09-12; 1): `PAIRSPLIT_PREREG.md`
- `analyze_paper2x2.py` (2026-09-13; 4): `PAPER2X2_PREREG.md` `PAPER2X2_RAW_CONTRASTS.json` `PAPER2X2_RESULTS.md` `_PAPER2X2_RAW.json`
- `analyze_papertask.py` (2026-09-12; 4): `PAPERTASK_PREREG.md` `PAPER_OOD_RERUN.json` `PAPER_TASK_FLOORS.json` `REPORT.md`
- `analyze_rank3.py` (2026-09-27; 1): `RANK3_PREREG.md`
- `analyze_rank_matched.py` (2026-09-24; 1): `RANK_MATCHED_PREREG.md`
- `analyze_rank_mi.py` (2026-09-24; 3): `RANK_MI.json` `RANK_MI_PREREG.md` `RANK_MI_STRATA.json`
- `analyze_rank_nd.py` (2026-09-30; 2): `RANK_ND.json` `RANK_ND_PREREG.md`
- `analyze_rank_nd_secondary.py` (2026-09-30; 2): `RANK_ND_PREREG.md` `RANK_ND_SECONDARY.json`
- `analyze_rank_perhead.py` (2026-09-24; 5): `RANK_MATCHED_e900.json` `RANK_MATCHED_e900_STRATA.json` `RANK_PERHEAD_PILOT.json` `RANK_PERHEAD_PILOT_STRATA.json` `RANK_PERHEAD_PREREG.md`
- `analyze_rank_proj.py` (2026-09-24; 5): `RANK_MATCHED_e900.json` `RANK_MATCHED_e900c.json` `RANK_PROJ_FROZEN.json` `RANK_PROJ_PREREG.md` `RANK_PROJ_TRAIN.json`
- `analyze_rank_sep.py` (2026-09-25; 1): `RANK_SEP_PREREG.md`
- `analyze_sameblock.py` (2026-09-15; 2): `SAMEBLOCK_PREREG.md` `SAMEBLOCK_RAW.md`
- `analyze_search.py` (2026-09-11; 3): `SEARCH_PREREG.md` `_ANATOMY_EXT.json` `_SEARCH_S3.json`
- `analyze_sign_matched.py` (2026-09-27; 2): `SIGN_MATCHED.json` `SIGN_MATCHED_PREREG.md`
- `analyze_spread.py` (2026-09-11; 2): `SPREAD_PREREG.md` `_SPREAD.json`
- `analyze_spread2.py` (2026-09-12; 3): `REPORT.md` `SPREAD2_PREREG.md` `_SPREAD2.json`
- `analyze_t1.py` (2026-09-18; 1): `T1_PREREG.md`
- `analyze_t2.py` (2026-09-20; 1): `T2_RESULTS.md`
- `analyze_textworld.py` (2026-09-28; 2): `TEXTWORLD_PREREG.md` `TEXTWORLD_PROBE.json`
- `analyze_textworld_secondary.py` (2026-09-28; 2): `TEXTWORLD_PREREG.md` `TEXTWORLD_SECONDARY.json`
- `ap_kernel_diagnostic.py` (2026-08-09; 1): `AP_KERNEL_DIAGNOSTIC.md`
- `audit_headline_numbers.py` (2026-08-23; 8): `AUDIT_HEADLINE.md` `BASELINE_TABLE.md` `INDEX_BASELINE_PAPER_TASK_n8.json` `INDEX_BASELINE_PAPER_TASK_n8.md` `KNOB_SWEEP_n8.json` `KNOB_SWEEP_n8.md` `MINIGRID_FULL_2X2X2.json` `MINIGRID_FULL_2X2X2.md`
- `audit_planner_tasks.py` (2026-09-24; 1): `PLANNER_TASK_AUDIT.md`
- `autocommit_bob.sh` (2026-07-29; 2): `COMPOSITIONAL_MULTISEED.json` `COMPOSITIONAL_MULTISEED.md`
- `autocommit_longt.sh` (2026-08-08; 4): `EM_FIX_COMP.json` `EM_FIX_COMP.md` `HIERGOAL_LONGT.json` `HIERGOAL_LONGT.md`
- `bench_compute.py` (2026-09-24; 1): `LOOP_HIER_COMPUTE.md`
- `benchmark_timing.py` (2026-08-16; 1): `TIMING_BENCHMARK.md`
- `build_rank_proj.py` (2026-09-24; 1): `RANK_PROJ_PREREG.md`
- `check_bf16.py` (2026-09-23; 1): `BF16_CHECK.json`
- `ckpt_guard.py` (2026-09-24; 1): `AUDIT_2026-09-10.md`
- `docs/audits/2026-09-24/applied/cmp_evals.sh` (2026-09-24; 12): `RANK_MATCHED_e900.json` `RANK_MATCHED_e900.md` `RANK_MATCHED_e900_ANALYSIS.txt` `RANK_MATCHED_e900_GEOMETRY.md` `RANK_MATCHED_e900_STRATA.json` `RANK_MATCHED_e900c_ANALYSIS.txt` `RANK_PERHEAD_PILOT_ANALYSIS.txt` `RANK_PERHEAD_PILOT_GEOMETRY.md` `RANK_PROJ_ANALYSIS.txt` `RANK_PROJ_TRAIN.json` `RANK_PROJ_TRAIN.md` `RANK_PROJ_TRAIN_STRATA.json`
- `docs/audits/2026-09-24/applied/verify_perm_repo.py` (2026-09-24; 6): `RANK_MATCHED.json` `RANK_MATCHED_e900.json` `RANK_MATCHED_e900c.json` `RANK_PROJ_FROZEN.json` `RANK_PROJ_TRAIN.json` `RANK_SWEEP.json`
- `docs/audits/2026-09-24/efficiency/mfprop/analyze_rank_matched.py` (2026-09-24; 1): `RANK_MATCHED_PREREG.md`
- `docs/audits/2026-09-24/efficiency/mfprop/environment.py` (2026-09-24; 4): `ALLOCENTRIC_RECODING.md` `FREQ_CONTROL.md` `KNOB_SWEEP.md` `MINIGRID_2X2X2.md`
- `docs/audits/2026-09-24/efficiency/proposed/analyze_rank_matched.py` (2026-09-24; 1): `RANK_MATCHED_PREREG.md`
- `docs/audits/2026-09-24/efficiency/proposed/environment.py` (2026-09-24; 4): `ALLOCENTRIC_RECODING.md` `FREQ_CONTROL.md` `KNOB_SWEEP.md` `MINIGRID_2X2X2.md`
- `docs/audits/2026-09-24/efficiency/proposed/train_variant.py` (2026-09-24; 11): `ADDITION_DESIGN.md` `COUNTER_BATCH.md` `DOF_PREREG.md` `MINIGRID_EM.md` `MONOTONE_PREREG.md` `NOLEAK_PREREG.md` `PAIRORIGIN_PREREG.md` `SAMEBLOCK_PREREG.md` `THEORY_KERNEL.md` `UNFREEZE_PREREG.md` `WARM_PREREG.md`
- `docs/audits/2026-09-24/efficiency/verify_perm.py` (2026-09-24; 1): `RANK_MATCHED_e900.json`
- `docs/audits/2026-09-24/hygiene/verify_stats_core.py` (2026-09-24; 1): `RANK_MATCHED_e900.json`
- `docs/audits/2026-09-24/rank_review/project_r2.py` (2026-09-27; 1): `RANK_MATCHED.json`
- `docs/audits/2026-09-24/rank_review/stats.py` (2026-09-27; 2): `RANK_MATCHED.json` `RANK_SWEEP.json`
- `docs/audits/2026-09-24/rank_review/strata_lossmatched.py` (2026-09-27; 1): `RANK_MATCHED_STRATA.json`
- `docs/audits/2026-09-24/rank_review/verify_proj/p.py` (2026-09-27; 1): `RANK_MATCHED.json`
- `docs/audits/2026-09-27/gate_cancel.py` (2026-09-27; 1): `CANCEL_PREREG.md`
- `docs/audits/2026-09-27/gate_ctxstep.py` (2026-09-28; 1): `CONTEXT_STEP_DESIGN.md`
- `docs/audits/2026-09-27/gate_rank3.py` (2026-09-27; 1): `RANK3_PREREG.md`
- `docs/audits/2026-09-27/gate_sign_matched.py` (2026-09-27; 1): `SIGN_MATCHED_PREREG.md`
- `docs/audits/2026-09-27/gate_textworld.py` (2026-09-28; 1): `TEXTWORLD_PREREG.md`
- `docs/audits/2026-09-27/run_swap_all.sh` (2026-09-30; 1): `CTXSTEP_HS_RECIPE.md`
- `docs/prepared/dyck_mdepth/analyze_dyck_mdepth.py` (2026-09-25; 3): `DYCK_MDEPTH_GATES.json` `DYCK_MDEPTH_GATES.md` `DYCK_MDEPTH_PREREG.md`
- `docs/prepared/dyck_mdepth/dyck_mdepth_common.py` (2026-09-25; 1): `DYCK_MDEPTH_PREREG.md`
- `docs/prepared/dyck_mdepth/run_dyck_mdepth.sh` (2026-09-25; 5): `DYCK_MDEPTH_GATES.json` `DYCK_MDEPTH_GATES.md` `DYCK_MDEPTH_PREREG.md` `DYCK_MDEPTH_RESULTS.json` `DYCK_MDEPTH_RESULTS.md`
- `docs/prepared/dyck_mdepth/train_dyck.py` (2026-09-25; 2): `DYCK_MDEPTH_PREREG.md` `DYCK_PREREG.md`
- `docs/prepared/dyck_mdepth/validate_dyck_mdepth.py` (2026-09-25; 2): `DYCK_LADDER_RESULTS.md` `DYCK_MDEPTH_PREREG.md`
- `docs/prepared/tex_edits_2026-09-27/ed_index.py` (2026-09-27; 11): `DYCK_LADDER_RESULTS.md` `DYCK_MDEPTH_PREREG.md` `DYCK_MDEPTH_RESULTS.md` `DYCK_RESULTS_bs128.md` `LOOP_RANK_PREREG.md` `LOOP_RANK_RESULTS.md` `RANK_MATCHED_RESULTS.md` `RANK_MI_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_PREREG.md` `RANK_SEP_RESULTS.md`
- `docs/prepared/tex_edits_2026-09-27/ed_rep.py` (2026-09-27; 19): `CODE_RESULTS.md` `DYCK_LADDER_RESULTS.md` `DYCK_MDEPTH_PREREG.md` `DYCK_MDEPTH_RESULTS.md` `DYCK_STACK_PROBE.md` `L15_LOOP_2X2.md` `LOOP_RANK_PREREG.md` `LOOP_RANK_RESULTS.md` `PAIRSPLIT_RESULTS.md` `RANK_MATCHED_RESULTS.md` `RANK_MI_ANALYSIS.txt` `RANK_MI_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_PREREG.md` `RANK_SEP_RESULTS.md` `RECENCY_EM_RESULTS.md` `RECIPE_POWER.md` `SEARCH_RESULTS.md` `WARM_RESULTS.md`
- `dyck_mdepth_common.py` (2026-09-25; 1): `DYCK_MDEPTH_PREREG.md`
- `dyck_standard_metrics.py` (2026-09-15; 1): `DYCK_STANDARD_METRICS.json`
- `environment.py` (2026-09-24; 4): `ALLOCENTRIC_RECODING.md` `FREQ_CONTROL.md` `KNOB_SWEEP.md` `MINIGRID_2X2X2.md`
- `environment_addition.py` (2026-09-15; 1): `ADDITION_DESIGN.md`
- `environment_cancel.py` (2026-09-27; 1): `CANCEL_PREREG.md`
- `environment_dyck.py` (2026-09-15; 1): `DYCK_PREREG.md`
- `environment_mqar.py` (2026-09-08; 1): `MQAR_PREREG.md`
- `environment_recency.py` (2026-09-11; 3): `RECENCY_GATES.md` `SEARCH_PREREG.md` `SPREAD_PREREG.md`
- `environment_rooms.py` (2026-07-18; 1): `HIER_ATTN_LONGT.md`
- `environment_schema.py` (2026-08-16; 1): `PLANNER_TASK_AUDIT.md`
- `environment_textworld.py` (2026-09-28; 1): `TEXTWORLD_PREREG.md`
- `environment_textworld_ctx.py` (2026-09-28; 1): `CONTEXT_STEP_DESIGN.md`
- `environment_textworld_ctx2.py` (2026-09-28; 1): `CONTEXT_STEP_DESIGN.md`
- `environment_textworld_ctx3.py` (2026-09-30; 1): `CONTEXT_STEP_DESIGN.md`
- `eval_ablation_miniworld.py` (2026-08-28; 1): `MINIWORLD_FRESH_ABLATION.md`
- `eval_bounded_memory.py` (2026-08-09; 1): `BOUNDED_MEMORY.md`
- `eval_code_delims.py` (2026-09-24; 3): `CODE_DELIMS.json` `CODE_GATES.json` `CODE_PREREG.md`
- `eval_code_long.py` (2026-09-22; 3): `CODE_LONG.json` `CODE_PREREG.md` `JSB_LENGTH_RESULTS.md`
- `eval_compositional.py` (2026-08-21; 1): `COMPOSITIONAL_RESULTS.md`
- `eval_dyck_literature.py` (2026-09-19; 1): `DYCK_LITERATURE_METRICS.md`
- `eval_h12_budget.py` (2026-08-23; 2): `CONTINUOUS_ALLOC.md` `H12_BUDGET_CURVE.md`
- `eval_h12_perseed.py` (2026-08-23; 1): `H12_BUDGET_CURVE.md`
- `eval_hier_goal_closedloop.py` (2026-09-24; 1): `HIERGOAL_CLOSEDLOOP.md`
- `eval_hiergoal_longT.py` (2026-08-09; 1): `HIERGOAL_LONGT.md`
- `eval_indirect_ood.py` (2026-09-17; 1): `INDIRECT_OOD.md`
- `eval_knob_sweep.py` (2026-08-21; 1): `KNOB_SWEEP.md`
- `eval_match_longq.py` (2026-09-24; 1): `MATCH_QUERY_LONGQ.md`
- `eval_minigrid_2x2.py` (2026-08-24; 2): `INDEX_BASELINE_PAPER_TASK_n8.md` `MINIGRID_2X2.md`
- `eval_ood_grid.py` (2026-09-24; 1): `OOD_GRID_RESULTS.md`(del)
- `eval_paper_ood.py` (2026-08-19; 1): `PAPER_OOD_PROTOCOL.md`
- `eval_paper_task.py` (2026-08-17; 2): `PAPER_TASK_ACCURACY.md` `PAPER_VALIDATION.md`
- `eval_single_env.py` (2026-05-21; 1): `TEM_BACKGROUND_BASELINES.md`
- `eval_vocab_sweep.py` (2026-08-09; 2): `VOCAB_SWEEP_MULTISEED.md` `VOCAB_SWEEP_RESULTS.md`
- `hand_indirect.py` (2026-09-23; 1): `INDIRECT_OOD.md`
- `hippocampal_analysis.py` (2026-09-24; 1): `HIPPOCAMPAL_ANALYSIS.md`(del)
- `hippocampal_hidden_eval.py` (2026-09-24; 1): `HIPPOCAMPAL_HIDDEN.md`(del)
- `horizon_task_distances.py` (2026-08-19; 2): `HORIZON_TASK_DISTANCES.md` `REVISIT_DISTANCE.md`
- `loop_depth_strata.py` (2026-09-24; 1): `LOOP_DEPTH_STRATA.md`
- `make_dyck_table.py` (2026-09-15; 1): `DYCK_TWO_METRICS.json`
- `make_per_visit_figure.py` (2026-09-24; 1): `PER_VISIT_clean.md`
- `master_finish_v3.sh` (2026-04-24; 2): `RESULTS_PAPER.md` `ZERO_SHOT_TRANSFER_clean.md`
- `master_level15pc.sh` (2026-04-26; 6): `CLONE_ANALYSIS_LEVEL15PC.md`(del) `HIPPOCAMPAL_LEVEL15PC.md`(del) `LONG_SEQ_clean.md` `PER_VISIT_clean.md` `RESULTS_PAPER.md` `ZERO_SHOT_TRANSFER_clean.md`
- `measure_paper_floor.py` (2026-09-12; 4): `BASELINE_TABLE.md` `EM_WM_THEORY.md` `PAPER_TASK_FLOORS.json` `PAPER_TASK_FLOORS.md`
- `miniworld_env.py` (2026-08-28; 2): `HABITAT_BUILD.md` `PERCEPTION_EXPERIMENT_PLAN.md`
- `model_baseline_rope.py` (2026-09-04; 1): `ROPE_CANONICAL.md`
- `model_centered.py` (2026-09-18; 1): `THEORY_MAPPOPE.md`
- `model_cho_positions.py` (2026-09-15; 1): `SAMEBLOCK_PREREG.md`
- `model_code_decay.py` (2026-09-21; 1): `CODE_RESULTS_OOD.md`
- `model_context_step.py` (2026-09-30; 2): `CONTEXT_STEP_DESIGN.md` `CTXSTEP_HS_RECIPE.md`
- `model_counter_installed.py` (2026-09-14; 1): `COUNTER_BATCH.md`
- `model_dyck_monotone.py` (2026-09-20; 2): `T2_RESULTS.md` `THEORY_MAPPOPE.md`
- `model_em_dof.py` (2026-09-10; 2): `N5_RESULTS.md` `RECENCY_EM_RESULTS.md`
- `model_em_fixed.py` (2026-09-08; 1): `MINIGRID_EM.md`
- `model_em_magonly.py` (2026-09-10; 2): `AUDIT_2026-09-10.md` `DOF_PREREG.md`
- `model_em_noleak.py` (2026-09-11; 1): `UNFREEZE_RESULTS.md`
- `model_em_pairconst.py` (2026-09-12; 1): `PAIRORIGIN_RESULTS.md`
- `model_em_pairorigin.py` (2026-09-11; 1): `EM_WM_THEORY.md`
- `model_em_phase.py` (2026-09-09; 1): `THEORY_KERNEL.md`
- `model_em_unfreeze.py` (2026-09-10; 1): `WARM_RESULTS.md`
- `model_em_warm.py` (2026-09-10; 1): `AUDIT_2026-09-10.md`
- `model_fixed_omega.py` (2026-08-21; 3): `INDEX_BASELINE_PAPER_TASK_n8.md` `MINIGRID_2X2.md` `MINIGRID_2X2X2.md`
- `model_forget.py` (2026-09-06; 2): `FORGET_GATE.md` `V4_MULTISEED.md`
- `model_gated.py` (2026-09-07; 1): `RECENCY_GATE_ABLATION.md`
- `model_hourglass.py` (2026-09-03; 1): `HIER_PARITY.md`
- `model_inekf_level15_perscale.py` (2026-05-17; 1): `TEM_CROSSSCALE_DIAGNOSTIC.md`
- `model_level15_dog.py` (2026-05-01; 1): `DOG_RESULTS.md`(del)
- `model_looped.py` (2026-09-26; 3): `HORIZON_RESULTS.md` `L15_ABLATION.md` `LOOP_RANK_PREREG.md`
- `model_monotone.py` (2026-09-13; 3): `MONOTONE_PREREG.md` `SEARCH_RESULTS.md` `SIGN_ABLATION.md`
- `model_pope_decay.py` (2026-09-20; 3): `DYCK_DECAY_RESULTS.md` `T3GEN_RESULTS.md` `THEORY_MAPPOPE.md`
- `model_pope_index_hier.py` (2026-08-28; 2): `FREQ_CONTROL.md` `MINIGRID_2X2X2_n8.md`
- `model_pope_t3.py` (2026-09-19; 2): `T3GEN_RESULTS.md` `THEORY_MAPPOPE.md`
- `model_rank.py` (2026-09-04; 1): `GATE_PROBE.md`
- `model_rank_perhead.py` (2026-09-27; 4): `RANK3_PREREG.md` `RANK_MI_PREREG.md` `RANK_PERHEAD_PREREG.md` `RANK_SEP_PREREG.md`
- `model_selective.py` (2026-09-04; 1): `CONV_KERNEL_PROBE.md`
- `orchestrator_finalize.sh` (2026-04-26; 1): `RESULTS_PAPER.md`
- `probe_achievable.py` (2026-09-12; 3): `THEORY_SEARCH_AND_LENGTH.md` `_ACHIEVABLE.json` `_ANATOMY_EXT.json`
- `probe_action_geometry.py` (2026-09-24; 1): `ACTION_GEOMETRY.md`
- `probe_anatomy.py` (2026-09-12; 2): `SEARCH_PREREG.md` `_ANATOMY.json`
- `probe_ap_coherence.py` (2026-09-09; 1): `THEORY_KERNEL.md`
- `probe_coarse_angles.py` (2026-08-02; 1): `coarse_angles.json`
- `probe_collisions.py` (2026-09-12; 2): `THEORY_SEARCH_AND_LENGTH.md` `_COLLISIONS.json`
- `probe_conv_kernel.py` (2026-09-04; 2): `CONV_KERNEL_PROBE.md` `SELECTIVE_ROPE.md`
- `probe_drift.py` (2026-09-04; 2): `ACTION_GEOMETRY.md` `DRIFT_PROBE.md`
- `probe_dyck_far.py` (2026-09-20; 1): `DYCK_FAR_PROBE.md`
- `probe_dyck_metric.py` (2026-09-20; 1): `DYCK_DECAY_RESULTS.md`
- `probe_dyck_stack.py` (2026-09-21; 1): `DYCK_STACK_PROBE.md`
- `probe_gate.py` (2026-09-24; 1): `GATE_PROBE.md`
- `probe_gate_separation.py` (2026-09-07; 3): `GATED_PREREG.md` `GATED_SEPARATION.md` `GATE_PROBE.md`
- `probe_init_grad.py` (2026-09-11; 2): `SEARCH_PREREG.md` `_INIT_GRAD.json`
- `probe_localisation.py` (2026-09-06; 1): `LOCALISATION_PREREG.md`
- `probe_paper_fig4.py` (2026-09-24; 1): `PAPER_FIG4_REPRO.md`
- `probe_phase_spread.py` (2026-09-12; 2): `EM_WM_STATE.md` `_PHASE_SPREAD.json`
- `probe_rank_learned.py` (2026-09-24; 1): `LEARNED_RANK.md`
- `probe_rank_projection.py` (2026-09-23; 1): `RANK_MATCHED.json`
- `probe_rank_truncate.py` (2026-09-24; 2): `LEARNED_RANK.md` `RANK_TRUNCATION.md`
- `probe_recency_accumulator.py` (2026-09-24; 1): `RECENCY_H2.md`
- `probe_revisit_2x2.py` (2026-09-14; 2): `REVISIT_2X2.json` `REVISIT_2X2.md`
- `probe_revisit_distance.py` (2026-08-17; 2): `INDEX_BASELINE_PAPER_TASK.md` `REVISIT_DISTANCE.md`
- `probe_rewind.py` (2026-09-11; 2): `AUDIT_2026-09-10.md` `UNFREEZE_RESULTS.md`
- `probe_sign.py` (2026-09-06; 1): `ACTION_GEOMETRY.md`
- `probe_stitch_attention.py` (2026-08-16; 1): `STITCH_ATTENTION.md`
- `probe_textworld.py` (2026-09-28; 1): `TEXTWORLD_PREREG.md`
- `probe_theory_numbers.py` (2026-09-19; 3): `T3GEN_RESULTS.md` `THEORY_MAPPOPE.md` `THEORY_NUMBERS.md`
- `probe_vector_nav_v2.py` (2026-09-24; 1): `VECTOR_NAV_V2_RESULTS.md`
- `probe_visits_per_cell.py` (2026-09-24; 1): `VISITS_PER_CELL.json`
- `retrain_level15em.sh` (2026-04-24; 1): `RESULTS_PAPER.md`
- `run_ablate.sh` (2026-09-22; 1): `ABLATE_PREREG.md`
- `run_accumulator.sh` (2026-09-06; 1): `ACCUMULATOR.md`
- `run_addition_pilot.sh` (2026-09-14; 1): `ADDITION_DESIGN.md`
- `run_addition_pilot2.sh` (2026-09-14; 1): `ADDITION_PILOT2.md`
- `run_algorithmic.sh` (2026-09-24; 2): `ALGORITHMIC_GATES.md` `ALGORITHMIC_RESULTS.md`
- `run_alias_finish.sh` (2026-09-24; 1): `ALIASING_CONTROLLED.md`
- `run_alias_gates.sh` (2026-08-28; 1): `ALIASING_GATES.md`
- `run_alias_sweep.sh` (2026-08-28; 2): `ALIASING_CONTROLLED.md` `ALIASING_GATES.md`
- `run_allocentric.sh` (2026-08-22; 1): `KNOB_SWEEP.md`
- `run_bestofboth.sh` (2026-07-28; 2): `CLOCK_SCAN.md` `COMPOSITIONAL_MULTISEED.md`
- `run_bestofboth_seeds.sh` (2026-07-29; 1): `COMPOSITIONAL_MULTISEED.md`
- `run_bounded_mem.sh` (2026-08-09; 2): `BOUNDED_MEMORY.json` `BOUNDED_MEMORY.md`
- `run_cancel.sh` (2026-09-27; 2): `CANCEL_ANALYSIS.txt` `CANCEL_PREREG.md`
- `run_capacity_control.sh` (2026-05-21; 1): `CAPACITY_CONTROL.md`
- `run_capacity_perregime.sh` (2026-05-22; 2): `CAPACITY_CONTROL.md` `CAPACITY_PERREGIME.md`
- `run_capacity_rerun.sh` (2026-05-22; 1): `CAPACITY_CONTROL.md`
- `run_clock_multiseed.sh` (2026-07-28; 1): `CLOCK_SCAN.md`
- `run_cmq_curric.sh` (2026-08-23; 1): `COMPOSITIONAL_MATCH_QUERY_CURRIC.md`
- `run_cmq_curric_full.sh` (2026-08-24; 1): `COMPOSITIONAL_MATCH_QUERY_CURRIC.md`
- `run_cmq_curric_topup.sh` (2026-08-24; 1): `COMPOSITIONAL_MATCH_QUERY_CURRIC.md`
- `run_cmq_stab.sh` (2026-08-23; 1): `COMPOSITIONAL_MATCH_QUERY_STAB.md`
- `run_cmq_stab_topup.sh` (2026-08-23; 1): `COMPOSITIONAL_MATCH_QUERY_STAB.md`
- `run_cmq_sweep.sh` (2026-08-23; 1): `COMPOSITIONAL_MATCH_QUERY_RESULTS.md`
- `run_cmq_topup.sh` (2026-08-23; 1): `COMPOSITIONAL_MATCH_QUERY_RESULTS.md`
- `run_cnav_redo.sh` (2026-05-03; 5): `CNAV_HEX_Level15.md`(del) `CNAV_HEX_Level15EM.md`(del) `CNAV_HEX_Vanilla.md`(del) `CNAV_HEX_VanillaEM.md`(del) `CNAV_RESULTS.md`(del)
- `run_coarseidx.sh` (2026-07-27; 1): `COMPOSITIONAL_MULTISEED.md`
- `run_coarsepi.sh` (2026-07-27; 1): `COMPOSITIONAL_MULTISEED.md`
- `run_code.sh` (2026-09-21; 2): `CODE_GATES.md` `CODE_PREREG.md`
- `run_code_fullval.sh` (2026-09-27; 2): `CODE_FULLVAL_ANALYSIS.txt` `CODE_FULLVAL_PREREG.md`
- `run_comp_headroom.sh` (2026-09-08; 1): `COMP_HEADROOM_PREREG.md`
- `run_comp_multiseed.sh` (2026-07-25; 1): `COMPOSITIONAL_MULTISEED.md`
- `run_comprehensive_followup.sh` (2026-05-07; 2): `MULTISEED_FOLLOWUP.md`(del) `TEM_RESULTS.md`
- `run_continuous_alloc.sh` (2026-08-23; 2): `ALLOCENTRIC_RECODING.md` `KNOB_SWEEP.md`
- `run_correction_gaps.sh` (2026-08-21; 4): `BASELINE_TABLE.md` `CORRECTION_COMPOSITIONAL.md` `LM200_ABLATION.md` `LM200_CORRECTED_MULTISEED.md`
- `run_ctxstep.sh` (2026-09-30; 1): `CTXSTEP_PREREG.md`
- `run_d5_seeds.sh` (2026-09-10; 1): `D5_PREREG.md`
- `run_dagger.sh` (2026-05-10; 1): `DAGGER_RESULTS.md`
- `run_dagger_dk6.sh` (2026-05-10; 1): `DAGGER_DK6_RESULTS.md`
- `run_dagger_empty.sh` (2026-05-10; 1): `DAGGER_EMPTY_RESULTS.md`
- `run_decay.sh` (2026-09-19; 1): `DECAY_PREREG.md`
- `run_diagnostic_and_v3.sh` (2026-04-28; 1): `V3_RESULTS.md`
- `run_dissociation.sh` (2026-08-19; 1): `DISSOCIATION_SWEEP.md`
- `run_dof.sh` (2026-09-10; 1): `DOF_PREREG.md`
- `run_dog_fix_and_continuous.sh` (2026-05-01; 6): `CNAV_HEX_Level15.md`(del) `CNAV_HEX_Vanilla.md`(del) `CNAV_RESULTS.md`(del) `DOG_RESULTS.md`(del) `MINIGRID_MEMORY_RESULTS.md` `TEM_RESULTS.md`
- `run_dog_test.sh` (2026-04-30; 1): `DOG_RESULTS.md`(del)
- `run_doorkey_bc.sh` (2026-05-10; 1): `DOORKEY_BC_RESULTS.md`
- `run_dxr.sh` (2026-09-04; 2): `DXR_RANK_THRESHOLD.md` `ND_GATES.md`
- `run_dyck.sh` (2026-09-15; 1): `DYCK_PREREG.md`
- `run_dyck_decay.sh` (2026-09-19; 1): `DYCK_DECAY_PROBE.md`
- `run_dyck_depth.sh` (2026-09-22; 1): `DYCK_DEPTH_PREREG.md`
- `run_dyck_ladder.sh` (2026-09-22; 1): `DYCK_LADDER_PREREG.md`
- `run_dyck_mdepth.sh` (2026-09-25; 5): `DYCK_MDEPTH_GATES.json` `DYCK_MDEPTH_GATES.md` `DYCK_MDEPTH_PREREG.md` `DYCK_MDEPTH_RESULTS.json` `DYCK_MDEPTH_RESULTS.md`
- `run_em_comp_samebatch.sh` (2026-08-09; 3): `EM_COMP_SAMEBATCH.json` `EM_COMP_SAMEBATCH.md` `EM_P0_COMP.md`
- `run_em_fig4.sh` (2026-09-24; 2): `PAPER_FIG4_EM.md` `PAPER_FIG4_REPRO.md`
- `run_em_fix.sh` (2026-08-08; 2): `EM_FIX_COMP.json` `EM_FIX_COMP.md`
- `run_em_p0.sh` (2026-08-09; 3): `EM_P0_COMP.json` `EM_P0_COMP.md` `EM_P0_PAPER.md`
- `run_em_perscale_and_hopfield.sh` (2026-05-18; 1): `EM_HOPFIELD_CROSSSCALE.md`
- `run_enwik8_2x2.sh` (2026-08-28; 2): `ENWIK8_2X2.md` `LANGUAGE_LANDSCAPE.md`
- `run_enwik8_comp.sh` (2026-09-22; 1): `ENWIK8_COMPOSITION_PREREG.md`
- `run_enwik8_hier.sh` (2026-08-28; 1): `ENWIK8_HIER.md`
- `run_enwik8_long.sh` (2026-08-28; 1): `ENWIK8_LONG.md`
- `run_enwik8_seeds.sh` (2026-08-28; 1): `ENWIK8_SEEDS.md`
- `run_extrahead_control.sh` (2026-05-21; 1): `EXTRAHEAD_CONTROL.md`
- `run_family_tree.sh` (2026-08-15; 3): `FAMILY_TREE_GATES.json` `FAMILY_TREE_GATES.md` `FAMILY_TREE_RESULTS.md`
- `run_family_tree_d7.sh` (2026-08-16; 3): `FAMILY_TREE_D7_GATES.md` `FAMILY_TREE_D7_RESULTS.md` `FAMILY_TREE_GATES.json`
- `run_final_evals.sh` (2026-08-23; 3): `CONTINUOUS_ALLOC.json` `CONTINUOUS_ALLOC.md` `MINIGRID_FULL_2X2X2.md`
- `run_fixup.sh` (2026-05-15; 1): `TEM_BACKGROUND_BASELINES.md`
- `run_flipflop.sh` (2026-09-07; 1): `FLIPFLOP_GATES.md`
- `run_forget.sh` (2026-09-05; 1): `FORGET_GATE.md`
- `run_forget_clock.sh` (2026-09-08; 1): `FORGET_CLOCK_PREREG.md`
- `run_forget_control.sh` (2026-09-05; 3): `FORGET_CONTROL.md` `FORGET_GATE.md` `V4_MULTISEED.md`
- `run_fr.sh` (2026-07-26; 1): `COMPOSITIONAL_MULTISEED.md`
- `run_freq_control.sh` (2026-08-21; 1): `FREQ_CONTROL.md`
- `run_frontier.sh` (2026-09-24; 1): `FRONTIER_ALGORITHMIC.md`
- `run_ft_extra.sh` (2026-08-21; 1): `FAMILY_TREE_RESULTS.md`
- `run_g8_converge.sh` (2026-08-28; 1): `CROSSOVER_CONVERGED.md`
- `run_gated.sh` (2026-09-07; 1): `GATED_PREREG.md`
- `run_gsf_clean_noise.sh` (2026-05-11; 1): `GSF_FULL_RESULTS.md`
- `run_h12_budget.sh` (2026-08-23; 1): `CONTINUOUS_ALLOC.md`
- `run_hier_parity.sh` (2026-09-24; 1): `HIER_PARITY.md`
- `run_hopfield_nomainap.sh` (2026-05-21; 1): `HOPFIELD_NOMAINAP_RESULTS.md`
- `run_horizon.sh` (2026-08-19; 2): `HORIZON_TASK_DISTANCES.md` `REVISIT_DISTANCE.md`
- `run_hourglass_experiments.sh` (2026-07-23; 1): `COMPOSITIONAL_RESULTS.md`
- `run_index_baseline_paper.sh` (2026-08-17; 2): `INDEX_BASELINE_PAPER_TASK.md` `PAPER_TASK_ABLATION.md`
- `run_indirect.sh` (2026-09-15; 2): `INDIRECT_PREREG.md` `INDIRECT_RESULTS.md`
- `run_indirect_200k.sh` (2026-09-16; 1): `INDIRECT_RESULTS_200k.md`
- `run_indirect_more.sh` (2026-09-15; 1): `INDIRECT_RESULTS.md`
- `run_jsb.sh` (2026-09-15; 2): `JSB_PREREG.md` `JSB_RESULTS.md`
- `run_jsb_base.sh` (2026-09-17; 1): `JSB_LENGTH_RESULTS_BASE.md`
- `run_jsb_len.sh` (2026-09-17; 2): `JSBLEN_PREREG.md` `JSB_LENGTH_RESULTS.md`
- `run_jsb_rank.sh` (2026-09-17; 1): `JSB_LENGTH_RESULTS_RANK.md`
- `run_knob_sweep.sh` (2026-08-21; 2): `FREQ_CONTROL.md` `INDEX_BASELINE_PAPER_TASK_n8.md`
- `run_l15_ablation.sh` (2026-09-24; 2): `L15_ABLATION.md` `RESULTS_PAPER.md`
- `run_l15_loop_2x2.sh` (2026-09-24; 4): `L15_ABLATION.md` `L15_LOOP_2X2.md` `_L15_LOOP_RAW.json` `_L15_LOOP_RAW.md`
- `run_l15_loop_supp.sh` (2026-09-24; 3): `L15_LOOP_2X2.md` `_L15_LOOP_RAW.json` `_L15_LOOP_RAW.md`
- `run_lambda_trace.sh` (2026-09-05; 3): `FORGET_CONTROL.json` `FORGET_CONTROL.md` `LAMBDA_TRACE.md`
- `run_lap_transfer.py` (2026-08-15; 1): `LAP_TRANSFER.md`
- `run_level15_beta.sh` (2026-05-10; 1): `LEVEL15BETA_RESULTS.md`
- `run_level15_meets_gated.sh` (2026-08-17; 4): `EXTRAHEAD_CONTROL.md` `LEVEL15_MEETS_GATED_matchq.md` `LEVEL15_MEETS_GATED_paper.md` `LM200_CORRECTED_MULTISEED.md`
- `run_level15_paper_50ep.sh` (2026-08-17; 2): `LEVEL15_MEETS_GATED_paper50.md` `MAP_QUERY_RESULTS.md`
- `run_level15em_crossscale.sh` (2026-05-18; 1): `LEVEL15EM_CROSSSCALE.md`
- `run_localisation.sh` (2026-09-06; 3): `LOCALISATION.md` `LOCALISATION_PREREG.md` `LOCALISATION_RANK.md`
- `run_loop_headroom.sh` (2026-09-24; 1): `LOOP_HEADROOM.md`
- `run_loop_hier.sh` (2026-09-24; 3): `HIER_PARITY.md` `LOOP_HIER_COMPUTE.md` `LOOP_HIER_PARITY.md`
- `run_loop_rank.sh` (2026-09-26; 6): `LOOP_RANK.json` `LOOP_RANK.md` `LOOP_RANK_ANALYSIS.txt` `LOOP_RANK_GEOMETRY.md` `LOOP_RANK_PREREG.md` `LOOP_RANK_STRATA.json`
- `run_loop_rank_e1800_budget.sh` (2026-09-28; 4): `LOOP_RANK_E1800_P1.json` `LOOP_RANK_E1800_P1.md` `LOOP_RANK_E1800_P1_ANALYSIS.txt` `LOOP_RANK_E1800_PREREG.md`
- `run_loop_sampled.sh` (2026-09-24; 1): `LOOP_SAMPLED.md`
- `run_loop_topup.sh` (2026-09-24; 1): `LOOP_HEADROOM.md`
- `run_loop_topup2.sh` (2026-09-24; 1): `LOOP_HEADROOM.md`
- `run_looped_pilot.sh` (2026-09-24; 5): `HORIZON_RESULTS.md` `LOOPED_L1.md` `LOOPED_L4.md` `LOOPED_Loop4.md` `LOOPED_PILOT.md`
- `run_magonly.sh` (2026-09-10; 1): `MAGONLY_PREREG.md`
- `run_map_query.sh` (2026-08-09; 3): `MAP_QUERY_GATES.json` `MAP_QUERY_GATES.md` `MAP_QUERY_RESULTS.md`
- `run_mappope_r4.sh` (2026-09-07; 1): `MAPPOPE_R4_PREREG.md`
- `run_match_query.sh` (2026-08-09; 3): `MATCH_QUERY_GATES.json` `MATCH_QUERY_GATES.md` `MATCH_QUERY_RESULTS.md`
- `run_match_query_em.sh` (2026-08-15; 2): `MATCH_QUERY_EM.md` `MATCH_QUERY_GATES.md`
- `run_match_scale.sh` (2026-08-16; 3): `MATCH_QUERY_LONGQ.json` `MATCH_QUERY_LONGQ.md` `MATCH_QUERY_SCALE.md`
- `run_minigrid_2x2.sh` (2026-08-21; 1): `MINIGRID_DK16_RESULTS.md`
- `run_minigrid_2x2x2.sh` (2026-08-21; 2): `EXTRAHEAD_CONTROL.md` `MINIGRID_2X2X2.md`
- `run_minigrid_allo.sh` (2026-08-24; 1): `MINIGRID_ALLOCENTRIC_2X2X2.md`
- `run_minigrid_allo8.sh` (2026-08-26; 4): `AUDIT_HEADLINE.md` `MINIGRID_ALLOCENTRIC_2X2X2.md` `MINIGRID_ALLOCENTRIC_8CELL.json` `MINIGRID_ALLOCENTRIC_8CELL.md`
- `run_minigrid_cached_full.sh` (2026-05-01; 1): `MINIGRID_DOORKEY_CACHED.md`
- `run_minigrid_dk16.sh` (2026-05-01; 1): `MINIGRID_DK16_RESULTS.md`
- `run_minigrid_doorkey.sh` (2026-05-01; 1): `MINIGRID_DOORKEY_RESULTS.md`
- `run_minigrid_em.sh` (2026-09-08; 1): `MINIGRID_EM_PREREG.md`
- `run_minigrid_em_fix.sh` (2026-09-08; 1): `MINIGRID_EM_FIX_PREREG.md`
- `run_minigrid_longt.sh` (2026-05-01; 1): `MINIGRID_DOORKEY_LONGT.md`
- `run_minigrid_rope_diag.sh` (2026-05-01; 1): `MINIGRID_DOORKEY_ROPE_DIAG.md`
- `run_miniworld.sh` (2026-08-28; 1): `MINIWORLD_FIXED_RESULTS.md`
- `run_monotone.sh` (2026-09-13; 4): `MONOTONE_PREREG.md` `MONOTONE_RAW.json` `_MONOTONE_TORUS.json` `_MONOTONE_TORUS.md`
- `run_motifseg.sh` (2026-07-25; 1): `COMPOSITIONAL_MULTISEED.md`
- `run_mq_noise_2x2.sh` (2026-09-24; 2): `MQ_NOISE_2X2.md` `RECIPE_CHOICE.json`
- `run_mq_noise_c2.sh` (2026-09-24; 1): `MQ_NOISE_2X2_C2.md`
- `run_mq_rank.sh` (2026-09-04; 3): `DXR_PRELIM.md` `LOOP_HEADROOM.md` `MQ_RANK_2X2.md`
- `run_multiclass.sh` (2026-05-13; 1): `MULTICLASS_RESULTS.md`(del)
- `run_multiclass_multiseed.sh` (2026-05-15; 1): `MULTICLASS_MULTISEED_RESULTS.md`
- `run_multienv_clean_baselines.sh` (2026-05-15; 1): `TEM_BACKGROUND_BASELINES.md`
- `run_multiseed_followup.sh` (2026-05-14; 1): `MULTISEED_FOLLOWUP_RESULTS.md`
- `run_mw_fresh_factorial.sh` (2026-08-28; 4): `MINIWORLD_FRESH_GATES_ALLO.md` `MINIWORLD_FRESH_GATES_RAW.md` `MINIWORLD_FRESH_RESULTS.md` `MINIWORLD_FRESH_RESULTS_T1024.md`
- `run_mw_gatectl.sh` (2026-08-28; 1): `MINIWORLD_GATE_CONTROL.md`
- `run_mw_grid_sweep.sh` (2026-08-28; 1): `MINIWORLD_GRID_SWEEP.md`
- `run_mw_grid_sweep_hier.sh` (2026-08-28; 1): `MINIWORLD_GRID_SWEEP_HIER.md`
- `run_mw_hier_ablation.sh` (2026-08-28; 1): `MINIWORLD_HIER_ABLATION.md`
- `run_mw_missing.sh` (2026-08-28; 1): `MINIWORLD_FIXED_RESULTS.md`
- `run_mw_oracle.sh` (2026-08-28; 2): `MINIWORLD_ORACLE_ABLATION.md` `MINIWORLD_ORACLE_GATES.md`
- `run_mw_probe3.sh` (2026-08-28; 1): `MINIWORLD_PROBE3.md`
- `run_mw_probe3_seeds.sh` (2026-08-28; 1): `MINIWORLD_SROPE_COMPONENTS.md`
- `run_mw_srope_components.sh` (2026-08-28; 1): `MINIWORLD_SROPE_COMPONENTS.md`
- `run_n5_phase.sh` (2026-09-09; 1): `N5_PREREG.md`
- `run_n8_evals.sh` (2026-08-22; 3): `KNOB_SWEEP_n8.json` `KNOB_SWEEP_n8.md` `MINIGRID_2X2X2_n8.md`
- `run_nobypass_test.sh` (2026-04-27; 3): `CLONE_TRANSFER_NOBYPASS.md`(del) `NOBYPASS_RESULTS.md` `R_T_DISTRIBUTION_3WAY.md`(del)
- `run_nodrop_multiseed.sh` (2026-05-10; 1): `NODROP_PARETO_RESULTS.md`
- `run_noise_refine.sh` (2026-09-24; 1): `NOISE_REFINE.md`
- `run_noleak.sh` (2026-09-11; 1): `NOLEAK_PREREG.md`
- `run_numberline.sh` (2026-05-21; 1): `NUMBERLINE_RESULTS.md`
- `run_ood_grid.sh` (2026-05-11; 1): `OOD_GRID_RESULTS.md`(del)
- `run_p4_p5_fix.sh` (2026-05-02; 3): `CNAV_RESULTS.md`(del) `MINIGRID_MEMORY_RESULTS.md` `TEM_RESULTS.md`
- `run_pairconst.sh` (2026-09-12; 1): `PAIRCONST_PREREG.md`
- `run_pairorigin.sh` (2026-09-11; 1): `PAIRORIGIN_PREREG.md`
- `run_pairsplit.sh` (2026-09-12; 1): `PAIRSPLIT_PREREG.md`
- `run_paper2x2.sh` (2026-09-13; 4): `PAPER2X2_PREREG.md` `PAPER2X2_RESULTS.md` `_PAPER2X2_RAW.json` `_PAPER2X2_RAW.md`
- `run_paper_validation.sh` (2026-08-09; 1): `PAPER_VALIDATION.md`
- `run_papertask_eval.sh` (2026-09-12; 3): `PAPER_OOD_RERUN.json` `PAPER_OOD_RERUN.md` `PAPER_TASK_FLOORS.md`
- `run_papertask_rerun.sh` (2026-09-12; 3): `PAPERTASK_PREREG.md` `PAPER_OOD_RERUN.json` `PAPER_OOD_RERUN.md`
- `run_perscale_omega.sh` (2026-05-17; 1): `PERSCALE_OMEGA_RESULTS.md`
- `run_planning_and_scale_controls.sh` (2026-05-15; 1): `TEM_CROSSSCALE_DIAGNOSTIC.md`
- `run_pope.sh` (2026-07-28; 1): `COMPOSITIONAL_MULTISEED.md`
- `run_pope_faithful.sh` (2026-07-29; 2): `CLOCK_SCAN.md` `COMPOSITIONAL_MULTISEED.md`
- `run_pope_hier_allo.sh` (2026-08-28; 2): `MINIGRID_ALLOCENTRIC_2X2X2.md` `MINIGRID_ALLO_8THCELL.md`
- `run_pope_ood.sh` (2026-08-17; 2): `INDEX_BASELINE_PAPER_TASK.md` `PAPER_OOD_WITH_POPE.md`
- `run_pope_paper.sh` (2026-08-17; 1): `INDEX_BASELINE_PAPER_TASK.md`
- `run_popewrap.sh` (2026-09-05; 1): `POPE_WRAPPING.md`
- `run_post_pipeline_analysis.sh` (2026-05-15; 2): `TEM_BACKGROUND_BASELINES.md` `TEM_CROSSSCALE_DIAGNOSTIC.md`
- `run_rank3.sh` (2026-09-27; 6): `RANK3.json` `RANK3.md` `RANK3_ANALYSIS.txt` `RANK3_GEOMETRY.md` `RANK3_PREREG.md` `RANK3_STRATA.json`
- `run_rank_matched.sh` (2026-09-24; 3): `RANK_MATCHED_PREREG.md` `RANK_SWEEP.json` `RANK_SWEEP_STRATA.json`
- `run_rank_mi.sh` (2026-09-24; 6): `RANK_MI.json` `RANK_MI.md` `RANK_MI_ANALYSIS.txt` `RANK_MI_GEOMETRY.md` `RANK_MI_PREREG.md` `RANK_MI_STRATA.json`
- `run_rank_nd.sh` (2026-09-30; 4): `RANK_ND.json` `RANK_ND.md` `RANK_ND_ANALYSIS.txt` `RANK_ND_PREREG.md`
- `run_rank_perhead_pilot.sh` (2026-09-24; 6): `RANK_PERHEAD_PILOT.json` `RANK_PERHEAD_PILOT.md` `RANK_PERHEAD_PILOT_ANALYSIS.txt` `RANK_PERHEAD_PILOT_GEOMETRY.md` `RANK_PERHEAD_PILOT_STRATA.json` `RANK_PERHEAD_PREREG.md`
- `run_rank_proj.sh` (2026-09-24; 7): `RANK_PROJ_ANALYSIS.txt` `RANK_PROJ_FROZEN.md` `RANK_PROJ_PREREG.md` `RANK_PROJ_TRAIN.json` `RANK_PROJ_TRAIN.md` `RANK_PROJ_TRAIN_GEOMETRY.md` `RANK_PROJ_TRAIN_STRATA.json`
- `run_rank_sep.sh` (2026-09-25; 6): `RANK_SEP.json` `RANK_SEP.md` `RANK_SEP_ANALYSIS.txt` `RANK_SEP_GEOMETRY.md` `RANK_SEP_PREREG.md` `RANK_SEP_STRATA.json`
- `run_rank_sweep.sh` (2026-09-24; 2): `GATE_PROBE.md` `RANK_SWEEP.md`
- `run_recency.sh` (2026-09-07; 2): `RECENCY_GATES_K64.md` `RECENCY_PREREG.md`
- `run_recency_em.sh` (2026-09-09; 2): `REC_EM_PREREG.md` `TALE_OF_TWO_ALGORITHMS.md`
- `run_recency_t3.sh` (2026-09-19; 1): `RECENCY_T3_PREREG.md`
- `run_recipe_power.sh` (2026-09-24; 1): `RECIPE_POWER.md`
- `run_refine.sh` (2026-09-24; 1): `REFINE_RESULTS.md`
- `run_rope_canonical.sh` (2026-09-24; 1): `ROPE_CANONICAL.md`
- `run_rope_converge.sh` (2026-08-28; 1): `ROPE_CONVERGE.md`
- `run_rotate_redo.sh` (2026-08-21; 1): `KNOB_SWEEP.md`
- `run_sameblock.sh` (2026-09-15; 1): `SAMEBLOCK_PREREG.md`
- `run_search.sh` (2026-09-11; 1): `SEARCH_PREREG.md`
- `run_seed_scaleup.sh` (2026-08-19; 2): `INDEX_BASELINE_PAPER_TASK_n8.md` `PAPER_OOD_EXTENDED_n8.md`
- `run_selective.sh` (2026-09-24; 2): `SELECTIVE_ROPE.md` `_SELECTIVE_TORUS.md`
- `run_sign.sh` (2026-09-06; 5): `SIGN_ABLATION.md` `SIGN_ABLATION_PREREG.md` `SIGN_PROBE.md` `_SIGN_RAW.json` `_SIGN_RAW.md`
- `run_sign_matched.sh` (2026-09-27; 6): `SIGN_MATCHED.json` `SIGN_MATCHED.md` `SIGN_MATCHED_ANALYSIS.txt` `SIGN_MATCHED_PREREG.md` `SIGN_MATCHED_PROBE.md` `SIGN_MATCHED_STRATA.json`
- `run_spread.sh` (2026-09-11; 1): `SPREAD_PREREG.md`
- `run_spread2.sh` (2026-09-12; 1): `SPREAD2_PREREG.md`
- `run_spread_cal.sh` (2026-09-11; 1): `SPREAD_RESULTS.md`
- `run_stitch_probe.sh` (2026-08-16; 1): `STITCH_ATTENTION.md`
- `run_t1_centered.sh` (2026-09-18; 2): `T1_PREREG.md` `T1_RESULTS.md`
- `run_t3.sh` (2026-09-18; 1): `T3_PREREG.md`
- `run_t3gen.sh` (2026-09-19; 2): `DYCK_T3_RESULTS.md` `T3GEN_PREREG.md`
- `run_t3gen2.sh` (2026-09-19; 1): `TORUS_T3_RESULTS.md`
- `run_tem_background_baselines.sh` (2026-05-15; 1): `TEM_BACKGROUND_BASELINES.md`
- `run_tem_noise_and_ffn.sh` (2026-05-21; 4): `NODROP_PARETO_RESULTS.md` `TEM_NOISE_FFN_RESULTS.md` `TEM_RESULTS.md` `TEM_T_RESULTS.md`
- `run_tem_recency_pilot.sh` (2026-09-14; 1): `TEM_RECENCY_PILOT.md`
- `run_tem_t.sh` (2026-05-03; 1): `TEM_T_RESULTS.md`
- `run_tem_t_multiseed.sh` (2026-05-04; 3): `RESULTS_PAPER.md` `TEM_T_MULTISEED.md` `TEM_T_RESULTS.md`
- `run_textworld.sh` (2026-09-28; 3): `TEXTWORLD_ANALYSIS.txt` `TEXTWORLD_PREREG.md` `TEXTWORLD_PROBE.json`
- `run_timing_rerun.sh` (2026-08-16; 2): `TIMING_BENCHMARK.json` `TIMING_BENCHMARK.md`
- `run_topology.sh` (2026-05-13; 1): `TOPOLOGY_RESULTS.md`
- `run_topup_n8.sh` (2026-08-01; 4): `CLOCK_SCAN.json` `CLOCK_SCAN.md` `COMPOSITIONAL_MULTISEED.json` `COMPOSITIONAL_MULTISEED.md`
- `run_unfreeze.sh` (2026-09-10; 1): `UNFREEZE_PREREG.md`
- `run_v4_control.sh` (2026-04-30; 1): `V4_CONTROL_RESULTS.md`
- `run_v4_multiseed.sh` (2026-04-29; 1): `V4_MULTISEED.md`
- `run_v4_test.sh` (2026-04-28; 1): `V4_RESULTS.md`
- `run_vanilla_converge.sh` (2026-08-28; 1): `POSITION_EFFECT_CONVERGED.md`
- `run_vector_nav_v2.sh` (2026-05-13; 1): `VECTOR_NAV_V2_RESULTS.md`
- `run_visits_test.sh` (2026-09-24; 1): `VISITS_TEST.md`
- `run_vocab_em.sh` (2026-09-09; 1): `VOCAB_EM_PREREG.md`
- `run_vocab_samebatch.sh` (2026-08-09; 2): `VOCAB_SWEEP_MULTISEED.json` `VOCAB_SWEEP_MULTISEED.md`
- `run_vocab_sweep.sh` (2026-05-10; 1): `VOCAB_SWEEP_RESULTS.md`
- `run_warm.sh` (2026-09-10; 1): `WARM_PREREG.md`
- `sweep_dissociation.py` (2026-08-19; 2): `COMPOSITIONAL_MULTISEED.md` `DISSOCIATION_SWEEP.md`
- `train_addition.py` (2026-09-15; 1): `ADDITION_DESIGN.md`
- `train_aggregate.py` (2026-07-17; 1): `HIER_ATTN_LONGT.md`
- `train_cancel.py` (2026-09-30; 1): `CANCEL_PREREG.md`
- `train_ctxstep.py` (2026-09-28; 1): `CONTEXT_STEP_DESIGN.md`
- `train_ctxstep2.py` (2026-09-28; 1): `CONTEXT_STEP_DESIGN.md`
- `train_ctxstep3.py` (2026-09-30; 1): `CONTEXT_STEP_DESIGN.md`
- `train_dyck.py` (2026-09-25; 2): `DYCK_MDEPTH_PREREG.md` `DYCK_PREREG.md`
- `train_flipflop.py` (2026-09-07; 1): `FLIPFLOP_GATES.md`
- `train_hourglass_enwik8.py` (2026-09-21; 1): `LANGUAGE_LANDSCAPE.md`
- `train_indirect.py` (2026-09-18; 1): `INDIRECT_PREREG.md`
- `train_jsb.py` (2026-09-20; 1): `JSB_PREREG.md`
- `train_mqar.py` (2026-09-08; 1): `MQAR_PREREG.md`
- `train_recency.py` (2026-09-11; 3): `RECENCY_GATES.md` `SEARCH_PREREG.md` `SPREAD_PREREG.md`
- `train_textworld.py` (2026-09-30; 1): `TEXTWORLD_PREREG.md`
- `validate_addition.py` (2026-09-14; 2): `ADDITION_DESIGN.md` `ADDITION_GATES.json`
- `validate_algorithmic.py` (2026-09-24; 1): `ALGORITHMIC_GATES.md`
- `validate_code.py` (2026-09-21; 2): `CODE_GATES.json` `CODE_GATES.md`
- `validate_compositional_match_query.py` (2026-09-24; 1): `COMPOSITIONAL_MATCH_QUERY_GATES.md`
- `validate_cscg_tasks.py` (2026-09-24; 1): `CSCG_TASK_GATES.md`
- `validate_dyck.py` (2026-09-15; 2): `DYCK_GATES.json` `DYCK_GATES.md`
- `validate_dyck_mdepth.py` (2026-09-25; 2): `DYCK_LADDER_RESULTS.md` `DYCK_MDEPTH_PREREG.md`
- `validate_family_tree.py` (2026-09-24; 1): `FAMILY_TREE_GATES.md`
- `validate_flipflop.py` (2026-09-07; 1): `FLIPFLOP_GATES.md`
- `validate_lap.py` (2026-09-24; 1): `LAP_GATES.md`
- `validate_map_query.py` (2026-09-24; 1): `MAP_QUERY_GATES.md`
- `validate_match_query.py` (2026-09-24; 1): `MATCH_QUERY_GATES.md`
- `validate_recency.py` (2026-09-24; 1): `RECENCY_GATES.md`
- `watch_minigrid_doorkey.sh` (2026-05-01; 1): `MINIGRID_DOORKEY_RESULTS.md`

</details>

Dynamic output paths (a name built from a variable, so a literal grep misses it). Every one writes to the repo root; after a move each must point at its family folder or new batches will refill the root:

- `analyze_factorial.py` (2026-10-12): `FACTORIAL_{root}.json`, `FACTORIAL_{root}_OWN_s{sd}.json`, `FACTORIAL_{root}_STRATA.json`
- `analyze_rank_nowrap.py` (2026-10-08): `RANK_NOWRAP_RESCORE_N{{N}}.json`
- `analyze_rank_tests.py` (2026-10-11): `RANK_TESTS_ESCAPE_{arm}{tag}.json`
- `analyze_score_rank.py` (2026-10-05): `SCORE_RANK_RESCORE_{r}.json`, `SCORE_RANK_STRATA_{r}.json`
- `docs/paper/expansion/scripts/recurrence/index_loop_passes.py` (2026-10-20): `FACTORIAL_{tag}.json`
- `docs/paper/figs/fig_exchange_rate.py` (2026-10-16): `FACTORIAL_{f}.json`
- `docs/paper/figs/fig_factorial_strata.py` (2026-10-16): `FACTORIAL_{f}_STRATA.json`
- `run_factorial.sh` (2026-10-12): `FACTORIAL_${ROOT}.json`, `FACTORIAL_${ROOT}.md`, `FACTORIAL_${ROOT}_OWN_s${S}.md`, `FACTORIAL_${ROOT}_STRATA.json`
- `run_factorial_n8.sh` (2026-10-24): `FACTORIAL_N8_${ROOT}.json`, `FACTORIAL_N8_${ROOT}.md`, `FACTORIAL_N8_${ROOT}_OWN_s${S}.md`, `FACTORIAL_N8_${ROOT}_STRATA.json`
- `run_generality.sh` (2026-10-14): `GENERALITY_TORUS_$A.json`, `GENERALITY_TORUS_$A.md`, `GENERALITY_TORUS_${A}_RS.md`, `GENERALITY_TORUS_${A}_STRATA.json`
- `run_rank_nowrap.sh` (2026-10-06): `RANK_NOWRAP_RESCORE_N$N.md`
- `run_rank_tests.sh` (2026-10-11): `RANK_TESTS_ESCAPE_${ARM}.md`, `RANK_TESTS_ESCAPE_${ARM}_RESCORED.md`, `RANK_TESTS_ESCAPE_${ARM}_STRATA.json`
- `run_rankrecipe.sh` (2026-10-24): `RANKRECIPE_$C.md`
- `run_rnn_torus.sh` (2026-10-26): `RNN_TORUS_${ROOT}.json`, `RNN_TORUS_${ROOT}.md`, `RNN_TORUS_${ROOT}_OWN_s${S}.md`, `RNN_TORUS_${ROOT}_STRATA.json`
- `run_robust.sh` (2026-10-16): `ROBUST_TORUS_$A.json`, `ROBUST_TORUS_$A.md`, `ROBUST_TORUS_${A}_STRATA.json`
- `run_score_rank.sh` (2026-10-05): `SCORE_RANK_$G.json`, `SCORE_RANK_$G.md`, `SCORE_RANK_RESCORE_$G.md`, `SCORE_RANK_STRATA_$G.json`
- `analyze_rank_matched.py` (2026-09-24): `RANK_MATCHED{a.tag}.json`, `RANK_MATCHED{a.tag}_STRATA.json`
- `run_alias_gates.sh` (2026-08-28): `_gate_n${NOBS}.md`
- `run_cnav_redo.sh` (2026-05-03): `CNAV_HEX_${v}.md`
- `run_dimsweep.sh` (2026-08-03): `DIMSWEEP_d$d.md`
- `run_dog_fix_and_continuous.sh` (2026-05-01): `CNAV_HEX_${v}.md`
- `run_dyck.sh` (2026-09-15): `DYCK_RESULTS_bs$BS.md`
- `run_match_scale.sh` (2026-08-16): `MATCH_GATES_${1}_${2}.md`
- `run_mw_grid_sweep.sh` (2026-08-28): `MINIWORLD_GRID${G}_GATES.md`
- `run_mw_oracle.sh` (2026-08-28): `MINIWORLD_ORACLE_RESULTS_T${L}.md`
- `run_p4_p5_fix.sh` (2026-05-02): `CNAV_HEX_${v}.md`
- `run_rank_matched.sh` (2026-09-24): `RANK_MATCHED${TAG}.json`, `RANK_MATCHED${TAG}.md`, `RANK_MATCHED${TAG}_ANALYSIS.txt`, `RANK_MATCHED${TAG}_GEOMETRY.md`, `RANK_MATCHED${TAG}_STRATA.json`
- `run_recipe_power.sh` (2026-09-24): `_RECIPE_${C}.md`

Other rule-18 points: `docs/tools/catalog_results_index.py` globs `*.md` at the root (must glob `results/**` and `archive/**`); `lib_driver.sh` `drv_done` takes artifact paths from each driver (they are `$REPO/NAME`); `docs/paper/figs/fig_exchange_rate.py`, `fig_factorial_strata.py` and `docs/paper/expansion/scripts/recurrence/index_loop_passes.py` read `FACTORIAL_*.json` from the root; `docs/prepared/dyck_mdepth/pkg/mapformer/` is a frozen package copy holding its own `DYCK_MDEPTH_GATES.*` (leave it).

#### C. Other top-level files that name a moving file

<details><summary>311 files</summary>

- `ABLATE_COMPOSITIONAL.md` (1): `COMPOSITIONAL_MULTISEED.md`
- `ABLATE_RESULTS.md` (1): `ABLATE_PREREG.md`
- `ACCOUNTS_RESULTS.md` (3): `ACCOUNTS_ANALYSIS.txt` `ACCOUNTS_PREREG.md` `ACCOUNTS_VERDICTS.json`
- `ACTION_GEOMETRY.md` (2): `RANK_MI_RESULTS.md` `RANK_PROJ_RESULTS.md`
- `ADDITION_CHO_REPRO.md` (1): `ADDITION_PILOT2.md`
- `ADDITION_PILOT.md` (2): `ADDITION_DESIGN.md` `ADDITION_GATES.md`
- `ADDITION_PILOT2.md` (1): `ADDITION_PILOT.md`
- `AGGREGATE_TASK_RESULTS.md` (1): `HIER_ATTN_LONGT.md`
- `ALGORITHMIC_RESULTS.md` (1): `ALGORITHMIC_GATES.md`
- `ALLOCENTRIC_RECODING.md` (3): `KNOB_SWEEP.md` `MINIGRID_2X2X2.md` `N3_AUDIT.md`
- `AUDIT_2026-09-10.md` (13): `D5_PREREG.md` `D5_RESULTS.md` `DOF_PREREG.md` `DOF_RESULTS.md` `MAGONLY_RESULTS.md` `N5_RESULTS.md` `RECENCY_EM_RESULTS.md` `REC_EM_PREREG.md` `SIGN_ABLATION.md` `TALE_OF_TWO_ALGORITHMS.md` `THEORY_KERNEL.md` `UNFREEZE_RESULTS.md` `WARM_RESULTS.md`
- `AUG_PREREG.md` (1): `JSB_PREREG.md`
- `AUG_RESULTS.md` (1): `AUG_PREREG.md`
- `BASELINE_TABLE.md` (13): `ALLOCENTRIC_RECODING.md` `COMPOSITIONAL_MULTISEED.md` `COMP_HEADROOM.md` `FAMILY_TREE_RESULTS.md` `FAMILY_TREE_WM_GAP.md` `FREQ_CONTROL.md` `H12_BUDGET_CURVE.md` `HIER_RECHECK.md` `INDEX_BASELINE_PAPER_TASK_n8.json` `KNOB_SWEEP.md` `LM200_ABLATION.md` `N3_AUDIT.md` `PAPER2X2_RESULTS.md`
- `BF16_RESULTS.md` (2): `BF16_CHECK.json` `MAESTRO_PLAN.md`
- `CANCEL_PREREG.md` (2): `CANCEL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md`
- `CANCEL_RESULTS.md` (3): `CANCEL_ANALYSIS.txt` `CANCEL_PREREG.md` `DYCK_MDEPTH_RESULTS.md`
- `CAPACITY_CONTROL.md` (1): `CAPACITY_PERREGIME.md`
- `CAPACITY_PERREGIME.md` (3): `CAPACITY_CONTROL.md` `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `CLOCK_SIDE_PREREG.md` (4): `CLOCK_SIDE_GATES.md` `CLOCK_SIDE_GATES_8000.md` `RECENCY_RESULTS.md` `SIGN_MATCHED_RESULTS.md`
- `CLOCK_SIDE_RESULTS.md` (4): `CLOCK_SIDE_ANALYSIS.txt` `CLOCK_SIDE_EVAL.json` `CLOCK_SIDE_PREREG.md` `CLOCK_SIDE_VERDICTS.json`
- `CLOCK_SPEED_PREREG.md` (3): `RING_BUDGET_RESULTS.md` `RING_DIAL_PREREG.md` `RING_DIAL_RESULTS.md`
- `CLOCK_SPEED_REP_PREREG.md` (1): `CLOCK_SPEED_RESULTS.md`
- `CLOCK_SPEED_REP_RESULTS.md` (3): `CLOCK_SPEED_REP_ANALYSIS.txt` `CLOCK_SPEED_REP_PREREG.md` `CLOCK_SPEED_REP_VERDICTS.json`
- `CLOCK_SPEED_RESULTS.md` (3): `CLOCK_SPEED_ANALYSIS.txt` `CLOCK_SPEED_PREREG.md` `CLOCK_SPEED_VERDICTS.json`
- `CODE_DECAY_RESULTS.md` (2): `CODE_FULLVAL_RESULTS.md` `CODE_PREREG.md`
- `CODE_FULLVAL_PREREG.md` (3): `ABLATE_RESULTS.md` `CODE_DECAY_RESULTS.md` `CODE_RESULTS.md`
- `CODE_FULLVAL_RESULTS.md` (2): `CODE_FULLVAL_ANALYSIS.txt` `CODE_FULLVAL_PREREG.md`
- `CODE_PREREG.md` (6): `CODE_GATES.md` `CODE_RESULTS_OOD.md` `DECAY_RESULTS.md` `JSB_LENGTH_RESULTS.md` `SIGN_ABLATION.md` `T2_RESULTS.md`
- `CODE_RESULTS.md` (6): `CODE_DECAY_RESULTS.md` `CODE_FULLVAL_RESULTS.md` `CODE_GATES.md` `CODE_PREREG.md` `CODE_RESULTS_OOD.md` `ENWIK8_SEEDS.md`
- `CODE_RESULTS_OOD.md` (1): `CODE_PREREG.md`
- `COMPOSITIONAL_EXPERIMENT.md` (1): `COMPOSITIONAL_MULTISEED.md`
- `COMPOSITIONAL_MATCH_QUERY.md` (2): `COMPOSITIONAL_MATCH_QUERY_CURRIC.md` `COMPOSITIONAL_MATCH_QUERY_STAB.md`
- `COMPOSITIONAL_RESULTS.md` (2): `COMPOSITIONAL_EXPERIMENT.md` `COMPOSITIONAL_MULTISEED.md`
- `COMP_HEADROOM.md` (1): `COMP_HEADROOM_PREREG.md`
- `COMP_HEADROOM_PREREG.md` (1): `RECIPE_POWER.md`
- `CONTEXT_STEP_DESIGN.md` (2): `CTXSTEP_PILOT2.md` `TEXTWORLD_RESULTS.md`
- `CONTINUOUS_ALLOC.md` (2): `H12_BUDGET_CURVE.md` `HABITAT_BUILD.md`
- `CONV_KERNEL_PROBE.md` (1): `SELECTIVE_ROPE.md`
- `CORRECTION_COMPOSITIONAL.md` (1): `BASELINE_TABLE.md`
- `COUNTER_BATCH.md` (1): `TEM_RECENCY_DIAG.md`
- `COUNTER_RESULTS.md` (2): `COUNTER_BATCH.md` `RECENCY_EM_RESULTS.md`
- `CROSS_PREREG.md` (2): `DYCK_DECAY_RESULTS.md` `DYCK_PREREG.md`
- `CROSS_RESULTS.md` (1): `CROSS_PREREG.md`
- `CTXSTEP_PILOT1.md` (1): `CONTEXT_STEP_DESIGN.md`
- `CTXSTEP_PREREG.md` (4): `CONTEXT_STEP_DESIGN.md` `CTXSTEP_HSR_PILOT.md` `CTXSTEP_HS_RECIPE.md` `CTXSTEP_PILOT3.md`
- `D5_PREREG.md` (1): `DOF_RESULTS.md`
- `D5_RESULTS.md` (4): `AUDIT_2026-09-10.md` `D5_PREREG.md` `DOF_RESULTS.md` `RECENCY_EM_RESULTS.md`
- `DECAY_PREREG.md` (2): `JSBLEN_PREREG.md` `THEORY_MAPPOPE.md`
- `DECAY_RESULTS.md` (3): `DECAY_PREREG.md` `DYCK_DECAY_RESULTS.md` `T2_RESULTS.md`
- `DELTA_FREE_ANALYSIS.txt` (1): `DELTA_FREE_PREREG.md`
- `DELTA_FREE_PREREG.md` (4): `DELTA_FREE_ANALYSIS.txt` `DELTA_FREE_VERDICTS.json` `PAPER2X2_RESULTS.md` `_DELTA_FREE_RAW.json`
- `DELTA_FREE_RESULTS.md` (3): `DELTA_FREE_ANALYSIS.txt` `DELTA_FREE_PREREG.md` `DELTA_FREE_VERDICTS.json`
- `DETAILED_RESULTS.md` (10): `DOG_RESULTS.md`(del) `HIPPOCAMPAL_ANALYSIS.md`(del) `HIPPOCAMPAL_HIDDEN.md`(del) `NOBYPASS_RESULTS.md` `RESULTS_PAPER.md` `RESULTS_SUMMARY_2026-05-10.md` `R_T_DISTRIBUTION_3WAY.md`(del) `TEM_RESULTS.md` `V4_MULTISEED.md` `ZERO_SHOT_TRANSFER_clean.md`
- `DISSOCIATION_SWEEP.md` (1): `ALIASING_COVARIATE.md`
- `DOF_PREREG.md` (1): `RECENCY_EM_RESULTS.md`
- `DOF_RESULTS.md` (6): `AUDIT_2026-09-10.md` `D5_RESULTS.md` `DOF_PREREG.md` `L15_LOOP_2X2.md` `RECENCY_EM_RESULTS.md` `THEORY_KERNEL.md`
- `DRIFT_PROBE.md` (2): `ACTION_GEOMETRY.md` `RANK_SWEEP.json`
- `DYCK_DECAY_PREREG.md` (2): `DECAY_RESULTS.md` `DYCK_PREREG.md`
- `DYCK_DECAY_RESULTS.md` (1): `DYCK_DECAY_PREREG.md`
- `DYCK_DEPTH_PREREG.md` (3): `CODE_RESULTS.md` `HORIZON_RESULTS.md` `ROPE_CANONICAL.md`
- `DYCK_DEPTH_RESULTS.md` (2): `DYCK_DEPTH_PREREG.md` `DYCK_DEPTH_RESULTS.json`
- `DYCK_LADDER_PREREG.md` (2): `DYCK_DEPTH_RESULTS.md` `DYCK_GATES.md`
- `DYCK_LADDER_RESULTS.md` (1): `DYCK_LADDER_PREREG.md`
- `DYCK_MDEPTH_PREREG.md` (2): `DYCK_LADDER_RESULTS.md` `DYCK_MDEPTH_GATES.md`
- `DYCK_MDEPTH_RESULTS.md` (2): `DYCK_MDEPTH_GATES.md` `DYCK_MDEPTH_PREREG.md`
- `DYCK_PREREG.md` (1): `DYCK_GATES.md`
- `DYCK_RESULTS_POPE.md` (2): `DYCK_GATES.md` `DYCK_PREREG.md`
- `DYCK_RESULTS_bs128.md` (2): `DYCK_GATES.md` `DYCK_PREREG.md`
- `DYCK_T3_RESULTS.md` (1): `T3GEN_RESULTS.md`
- `EM_WM_STATE.md` (23): `AUDIT_2026-09-10.md` `D5_RESULTS.md` `DOF_RESULTS.md` `EM_WM_THEORY.md` `MAGONLY_RESULTS.md` `N5_RESULTS.md` `NOLEAK_RESULTS.md` `PAIRCONST_PREREG.md` `PAIRCONST_RESULTS.md` `PAIRORIGIN_RESULTS.md` `PAIRSPLIT_RESULTS.md` `PAPERTASK_RESULTS.md` `PAPER_OOD_EXTENDED_n8.json` `RECENCY_EM_RESULTS.md` `RECENCY_RESULTS.md` `SEARCH_RESULTS.md` `SPREAD2_RESULTS.md` `SPREAD_RESULTS.md` `TALE_OF_TWO_ALGORITHMS.md` `THEORY_KERNEL.md` `UNFREEZE_RESULTS.md` `WARM_RESULTS.md` `_REWIND_PROBE.json`
- `EM_WM_THEORY.md` (12): `AP_KERNEL_DIAGNOSTIC.md` `AUDIT_2026-09-10.md` `EM_WM_STATE.md` `PAIRCONST_PREREG.md` `PAIRCONST_RESULTS.md` `PAIRORIGIN_RESULTS.md` `PAIRSPLIT_RESULTS.md` `PAPERTASK_RESULTS.md` `PAPER_TASK_FLOORS.md` `SEARCH_RESULTS.md` `SPREAD_RESULTS.md` `TALE_OF_TWO_ALGORITHMS.md`
- `ENWIK8_COMPOSITION_PREREG.md` (1): `ENWIK8_SEEDS.md`
- `EXPTORUS_PREREG.md` (3): `EXPTORUS_ANALYSIS.txt` `EXPTORUS_VERDICTS.json` `G3_SSM_A2_RESULTS.md`
- `EXPTORUS_RESULTS.md` (8): `EXPTORUS_ANALYSIS.txt` `EXPTORUS_PREREG.md` `EXPTORUS_VERDICTS.json` `G3_SSM_A2_RESULTS.md` `G3_SSM_RESULTS.md` `SIGNROUTE_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TORUS_LRSWEEP_RESULTS.md`
- `FACTORIAL_N8_PREREG.md` (4): `FACTORIAL_RESULTS.md` `FAST_ATTN_RANK.md` `G3_SSM_PREREG.md` `REFINE_RESULTS.md`
- `FACTORIAL_N8_RESULTS.md` (3): `FACTORIAL_N8_ANALYSIS.txt` `FACTORIAL_N8_PREREG.md` `FACTORIAL_N8_VERDICTS.json`
- `FACTORIAL_PREREG.md` (4): `DYCK_MDEPTH_RESULTS.md` `LOOPED_PILOT.md` `LOOP_HEADROOM.md` `SIGN_MATCHED_RESULTS.md`
- `FACTORIAL_RESULTS.md` (5): `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_ANALYSIS.txt` `FACTORIAL_PREREG.md` `FACTORIAL_VERDICTS.json` `LOOPED_PILOT.md`
- `FAMILY_TREE_D7_RESULTS.md` (2): `FAMILY_TREE_RESULTS.md` `TIMING_BENCHMARK.md`
- `FAMILY_TREE_RESULTS.md` (2): `AP_KERNEL_DIAGNOSTIC.md` `LAP_TRANSFER_NOREWARD.md`
- `FAMILY_TREE_WM_GAP.md` (3): `BASELINE_TABLE.md` `FAMILY_TREE_RESULTS.md` `N3_AUDIT.md`
- `FLIPFLOP_RESULTS.md` (2): `FLIPFLOP_GATES.md` `RECENCY_RESULTS.md`
- `FORGET_CONTROL.md` (1): `FORGET_GATE.md`
- `FORGET_GATE.md` (2): `SELECTIVE_ROPE.md` `V4_MULTISEED.md`
- `FREQ_CONTROL.md` (1): `INDEX_BASELINE_PAPER_TASK_n8.md`
- `FRONTIER_ALGORITHMIC.md` (1): `ALGORITHMIC_RESULTS.md`
- `G3_SSM_A2_RESULTS.md` (5): `G3_SSM_A2_ANALYSIS.txt` `G3_SSM_A2_VERDICTS.json` `G3_SSM_PREREG.md` `G3_SSM_RESULTS.md` `SIGNROUTE_RESULTS.md`
- `G3_SSM_PREREG.md` (8): `G3_SSM_A2_ANALYSIS.txt` `G3_SSM_A2_VERDICTS.json` `G3_SSM_RESULTS.md` `G3_SSM_VERDICTS.json` `LRSWEEP_RESULTS.md` `MECHANISM_RESULTS.md` `SIGNROUTE_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `G3_SSM_RESULTS.md` (5): `G3_SSM_A2_RESULTS.md` `G3_SSM_ANALYSIS.txt` `G3_SSM_PREREG.md` `G3_SSM_VERDICTS.json` `SIGNROUTE_RESULTS.md`
- `GAIN_GRAIN_PREREG.md` (1): `MAPPOPE_PAIR_RESULTS.md`
- `GAIN_GRAIN_RESULTS.md` (2): `GAIN_GRAIN_ANALYSIS.txt` `GAIN_GRAIN_PREREG.md`
- `GAIN_PHASE_PREREG.md` (3): `GAIN_GRAIN_RESULTS.md` `LEAK.json` `LEAK_RESULTS.md`
- `GAIN_PHASE_RESULTS.md` (4): `GAIN_PHASE_ANALYSIS.txt` `GAIN_PHASE_EVAL.json` `GAIN_PHASE_PREREG.md` `GAIN_PHASE_VERDICTS.json`
- `GATED_PREREG.md` (1): `RECENCY_GATE_ABLATION.md`
- `GATED_RESULTS.md` (1): `GATED_PREREG.md`
- `GATED_SEPARATION.md` (2): `GATED_PREREG.md` `GATE_PROBE.md`
- `GAUGE_RESULTS.md` (3): `GAUGE_ANALYSIS.txt` `GAUGE_PREREG.md` `GAUGE_VERDICTS.json`
- `GENERALITY_PREREG.md` (2): `SIGN_MATCHED_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `GENERALITY_RESULTS.md` (3): `GENERALITY_ANALYSIS.txt` `GENERALITY_PREREG.md` `GENERALITY_VERDICTS.json`
- `GENERALIZATION_REPORT.md` (12): `DOORKEY_BC_RESULTS.md` `MINIGRID_DK16_RESULTS.md` `MINIGRID_DOORKEY_LONGT.md` `MINIGRID_DOORKEY_RESULTS.md` `MINIGRID_MEMORY_RESULTS.md` `MULTICLASS_MULTISEED_RESULTS.md` `MULTICLASS_RESULTS.md`(del) `OOD_GRID_RESULTS.md`(del) `PERSCALE_OMEGA_RESULTS.md` `TEM_CROSSSCALE_DIAGNOSTIC.md` `TOPOLOGY_RESULTS.md` `VOCAB_SWEEP_RESULTS.md`
- `GSF_FULL_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `HABITAT_BUILD.md` (3): `ALLOCENTRIC_RECODING.md` `CONTINUOUS_ALLOC.md` `KNOB_SWEEP.md`
- `HIER_PARITY.md` (2): `ALGORITHMIC_RESULTS.md` `LOOP_HIER_PARITY.md`
- `HIER_RECHECK.md` (1): `COMP_HEADROOM.md`
- `HORIZON_RESULTS.md` (2): `LOOPED_PILOT.md` `REVISIT_DISTANCE.md`
- `HORIZON_TASK_DISTANCES.md` (1): `REVISIT_DISTANCE.md`
- `IMAGE_TORUS_PREREG.md` (7): `FACTORIAL_RESULTS.md` `IMAGE_TORUS_GATE.md` `IMAGE_TORUS_PILOT.txt` `IMAGE_TORUS_PILOT2.txt` `IMAGE_TORUS_VERDICTS.json` `LEAK_RESULTS.md` `SIGN_MATCHED_RESULTS.md`
- `IMAGE_TORUS_RESULTS.md` (5): `IMAGE_TORUS_ANALYSIS.txt` `IMAGE_TORUS_PILOT.txt` `IMAGE_TORUS_PILOT2.txt` `IMAGE_TORUS_PREREG.md` `IMAGE_TORUS_VERDICTS.json`
- `INDEX_BASELINE_PAPER_TASK.md` (2): `MATCH_QUERY_RESULTS.md` `PAPER_OOD_PROTOCOL.md`
- `INDIRECT_ARITHMETIC.md` (2): `INDIRECT_OOD.md` `INDIRECT_RESULTS_200k.md`
- `INDIRECT_RESULTS.md` (1): `INDIRECT_RESULTS_200k.md`
- `JSBLEN_PREREG.md` (1): `JSB_PREREG.md`
- `JSB_LENGTH_RESULTS_BASE.md` (1): `LOCALISATION.md`
- `JSB_LENGTH_RESULTS_RANK.md` (1): `RANK_SWEEP.md`
- `JSB_RESULTS.md` (1): `AUG_RESULTS.md`
- `LAP_GATES.md` (2): `LAP_TRANSFER.md` `LAP_TRANSFER_NOREWARD.md`
- `LAP_TRANSFER_NOREWARD.md` (1): `LAP_GATES.md`
- `LEAK_PREREG.md` (1): `LEAK_ANALYSIS.txt`
- `LEAK_RESULTS.md` (4): `LEAK.json` `LEAK_ANALYSIS.txt` `LEAK_PREREG.md` `LEAK_SECONDARY.txt`
- `LEVEL15BETA_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `LEVEL15_MEETS_GATED_matchq.md` (3): `LM200_CORRECTED_MULTISEED.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md`
- `LEVEL15_MEETS_GATED_paper50.md` (3): `EXTRAHEAD_CONTROL.md` `LEVEL15_MEETS_GATED_matchq.md` `MAP_QUERY_RESULTS.md`
- `LM200_ABLATION.md` (2): `EXTRAHEAD_CONTROL.md` `LM200_CORRECTED_MULTISEED.md`
- `LOCALISATION.md` (3): `ACTION_GEOMETRY.md` `LOCALISATION_PREREG.md` `SIGN_PROBE.md`
- `LOCALISATION_RANK.md` (1): `LOCALISATION_PREREG.md`
- `LOOP_HIER_PARITY.md` (3): `ALGORITHMIC_RESULTS.md` `HIER_PARITY.md` `LOOP_HIER_COMPUTE.md`
- `LOOP_RANK_E1800_P1_RESULTS.md` (3): `LOOP_RANK_E1800_P1.md` `LOOP_RANK_E1800_P1_ANALYSIS.txt` `LOOP_RANK_E1800_PREREG.md`
- `LOOP_RANK_E1800_PREREG.md` (3): `LOOP_RANK_RESULTS.md` `RANK3_RESULTS.md` `RANK_MATCHED_RESULTS.md`
- `LOOP_RANK_PREREG.md` (7): `L15_LOOP_2X2.md` `LOOP_HEADROOM.md` `MQ_NOISE_2X2_C2.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md` `REFINE_RESULTS.md` `SEARCH_RESULTS.md`
- `LOOP_RANK_RESULTS.md` (3): `LOOP_RANK_ANALYSIS.txt` `LOOP_RANK_PREREG.md` `RANK_PROJ_RESULTS.md`
- `LRSWEEP_PREREG.md` (3): `LRSWEEP_ANALYSIS.txt` `LRSWEEP_VERDICTS.json` `ROBUST_RESULTS.md`
- `LRSWEEP_RESULTS.md` (3): `LRSWEEP_ANALYSIS.txt` `LRSWEEP_PREREG.md` `LRSWEEP_VERDICTS.json`
- `MAESTRO_PLAN.md` (5): `ABLATE_RESULTS.md` `AUG_RESULTS.md` `BF16_RESULTS.md` `CODE_DECAY_RESULTS.md` `JSB_RESULTS.md`
- `MAGONLY_PREREG.md` (1): `AUDIT_2026-09-10.md`
- `MAGONLY_RESULTS.md` (3): `MAGONLY_PREREG.md` `WARM_RESULTS.md` `_REWIND_PROBE.json`
- `MAPPOPE_PAIR_PREREG.md` (3): `JSB_LENGTH_RESULTS_RANK.md` `MAPPOPE_R4_RESULTS.md` `T2_RESULTS.md`
- `MAPPOPE_PAIR_RESULTS.md` (5): `MAPPOPE_PAIR_ANALYSIS.txt` `MAPPOPE_PAIR_PREREG.md` `MAPPOPE_PAIR_R2.json` `MAPPOPE_R4_RESULTS.md` `RANK_SEP_RESULTS.md`
- `MAPPOPE_R4_PREREG.md` (1): `GATED_PREREG.md`
- `MAPPOPE_R4_RESULTS.md` (1): `MAPPOPE_R4_PREREG.md`
- `MAPPOPE_VS_POPE.md` (1): `COMP_HEADROOM.md`
- `MAP_QUERY_RESULTS.md` (1): `MAP_QUERY_GATES.md`
- `MATCH_QUERY_EM.md` (2): `AP_KERNEL_DIAGNOSTIC.md` `MATCH_QUERY_GATES.md`
- `MATCH_QUERY_NOISE_ABLATION.md` (2): `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md`
- `MATCH_QUERY_RESULTS.md` (5): `COMPOSITIONAL_MULTISEED.md` `HIERGOAL_ABLATION.md` `MATCH_QUERY_GATES.md` `MATCH_QUERY_SCALE.md` `N3_AUDIT.md`
- `MATCH_QUERY_SCALE.md` (1): `MATCH_QUERY_LONGQ.md`
- `MECHANISM_RESULTS.md` (4): `MECHANISM_ANALYSIS.txt` `MECHANISM_PREREG.md` `MECHANISM_VERDICTS.json` `TORUS_START_RESULTS.md`
- `MEMORY_MAZE_GATE.md` (2): `ALIASING_CONTROLLED.md` `VISITS_TEST.md`
- `MEMORY_MAZE_GATE_V2.md` (1): `MEMORY_MAZE_GATE.md`
- `MEMORY_MAZE_GATE_V3.md` (3): `MEMORY_MAZE_GATE.md` `MEMORY_MAZE_GATE_V2.md` `RNN_TORUS_RESULTS.md`
- `MEMORY_MAZE_PREREG.md` (5): `KNOB_SWEEP_n8.md` `MEMORY_MAZE_GATE.md` `MEMORY_MAZE_GATE_V2.md` `MEMORY_MAZE_GATE_V3.md` `RNN_TORUS_RESULTS.md`
- `MEMORY_MAZE_RESULTS.md` (5): `ALLOCENTRIC_RECODING.md` `KNOB_SWEEP_n8.md` `MEMORY_MAZE_ANALYSIS.txt` `MEMORY_MAZE_PREREG.md` `MEMORY_MAZE_VERDICTS.json`
- `MINIGRID_2X2.md` (2): `INDEX_BASELINE_PAPER_TASK_n8.md` `MINIGRID_DK16_RESULTS.md`
- `MINIGRID_2X2X2.md` (1): `INDEX_BASELINE_PAPER_TASK_n8.md`
- `MINIGRID_2X2X2_n8.md` (1): `INDEX_BASELINE_PAPER_TASK_n8.md`
- `MINIGRID_ALLOCENTRIC_2X2X2.md` (3): `INDEX_BASELINE_PAPER_TASK_n8.md` `MINIGRID_ALLOCENTRIC_8CELL.md` `MINIGRID_FULL_2X2X2.md`
- `MINIGRID_ALLOCENTRIC_8CELL.md` (1): `INDEX_BASELINE_PAPER_TASK_n8.md`
- `MINIGRID_EM.md` (1): `MINIGRID_EM_PREREG.md`
- `MINIGRID_EM_FIX.md` (3): `EM_WM_STATE.md` `MINIGRID_EM.md` `MINIGRID_EM_FIX_PREREG.md`
- `MINIGRID_EM_FIX_PREREG.md` (1): `MINIGRID_EM.md`
- `MINIGRID_FULL_2X2X2.md` (1): `INDEX_BASELINE_PAPER_TASK_n8.md`
- `MINIGRID_REPRO_CONTROL.md` (1): `INDEX_BASELINE_PAPER_TASK_n8.md`
- `MINIWORLD_ENDPOINTS.md` (1): `VISITS_TEST.md`
- `MINIWORLD_FIXED_FINDINGS.md` (3): `MINIWORLD_FIXED_RESULTS.md` `MINIWORLD_FIXED_RESULTS_T1024.md` `MINIWORLD_GATES_ALLO.md`
- `MINIWORLD_FRESH_FINDINGS.md` (1): `MINIWORLD_FRESH_ABLATION.md`
- `MINIWORLD_HIER_ABLATION.md` (1): `MINIWORLD_GRID_SWEEP_HIER.md`
- `MINIWORLD_TODO.md` (2): `MINIWORLD_GATES_ALLO.md` `PERCEPTION_EXPERIMENT_PLAN.md`
- `MONOTONE_PREREG.md` (5): `MONOTONE_RESULTS.md` `RECENCY_RESULTS.md` `SEARCH_RESULTS.md` `SIGN_ABLATION.md` `THEORY_SEARCH_AND_LENGTH.md`
- `MONOTONE_RAW.md` (2): `AUDIT_2026-09-10.md` `MONOTONE_PREREG.md`
- `MONOTONE_RESULTS.md` (5): `MONOTONE_PREREG.md` `MONOTONE_RAW.json` `MONOTONE_RAW.md` `SEARCH_RESULTS.md` `SIGN_ABLATION.md`
- `MQAR_RESULTS.md` (1): `MQAR_PREREG.md`
- `N3_AUDIT.md` (7): `ALIASING_CONTROLLED.md` `BASELINE_TABLE.md` `COMPOSITIONAL_MULTISEED.md` `FAMILY_TREE_WM_GAP.md` `H12_BUDGET_CURVE.md` `KNOB_SWEEP.md` `MATCH_QUERY_RESULTS.md`
- `N5_PREREG.md` (1): `THEORY_KERNEL.md`
- `N5_RESULTS.md` (2): `AUDIT_2026-09-10.md` `N5_PREREG.md`
- `NOBYPASS_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `NODROP_PARETO_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `NOISE_CLEAN_REVALIDATION.md` (1): `CORRECTED_LM200_LEADERBOARD.md`
- `NOLEAK_PREREG.md` (1): `UNFREEZE_RESULTS.md`
- `NOLEAK_RESULTS.md` (3): `NOLEAK_PREREG.md` `SEARCH_RESULTS.md` `UNFREEZE_RESULTS.md`
- `PAIRCONST_PREREG.md` (2): `AUDIT_2026-09-10.md` `PAIRORIGIN_RESULTS.md`
- `PAIRCONST_RESULTS.md` (2): `PAIRCONST_PREREG.md` `PAIRORIGIN_RESULTS.md`
- `PAIRORIGIN_PREREG.md` (1): `EM_WM_THEORY.md`
- `PAIRORIGIN_RESULTS.md` (3): `AUDIT_2026-09-10.md` `EM_WM_THEORY.md` `PAIRORIGIN_PREREG.md`
- `PAIRSPLIT_PREREG.md` (2): `AUDIT_2026-09-10.md` `PAIRCONST_RESULTS.md`
- `PAIRSPLIT_RESULTS.md` (2): `PAIRCONST_RESULTS.md` `PAIRSPLIT_PREREG.md`
- `PAPER2X2_PREREG.md` (4): `BASELINE_TABLE.md` `INDEX_BASELINE_PAPER_TASK_n8.md` `MAPPOPE_R4_RESULTS.md` `SIGN_ABLATION.md`
- `PAPER2X2_RESULTS.md` (2): `PAPER2X2_PREREG.md` `SIGN_ABLATION.md`
- `PAPERTASK_PREREG.md` (3): `AUDIT_2026-09-10.md` `EM_WM_THEORY.md` `PAPER_TASK_FLOORS.md`
- `PAPERTASK_RESULTS.md` (3): `EM_WM_THEORY.md` `PAPERTASK_PREREG.md` `PAPER_TASK_FLOORS.md`
- `PAPER_OOD_EXTENDED_n8.md` (1): `PAPER_OOD_EXTENDED.md`
- `PAPER_OOD_WITH_POPE.md` (2): `INDEX_BASELINE_PAPER_TASK.md` `REVISIT_DISTANCE.md`
- `PAPER_TASK_ABLATION.md` (2): `AP_KERNEL_DIAGNOSTIC.md` `NOISE_CLEAN_REVALIDATION.md`
- `PERCEPTION_EXPERIMENT_PLAN.md` (1): `HABITAT_BUILD.md`
- `RANK3_PREREG.md` (1): `RANK_SEP_RESULTS.md`
- `RANK3_RESULTS.md` (6): `RANK3.md` `RANK3_ANALYSIS.txt` `RANK3_GEOMETRY.md` `RANK3_PREREG.md` `RANK3_STRATA.json` `RANK_MATCHED_RESULTS.md`
- `RANKCLOCK_PREREG.md` (1): `RANK_MI_RESULTS.md`
- `RANKCLOCK_RESULTS.md` (6): `LOOP_RANK_E1800_P1_RESULTS.md` `RANKCLOCK.md` `RANKCLOCK_ANALYSIS.txt` `RANKCLOCK_PREREG.md` `RANKCLOCK_VERDICTS.json` `RANKCLOCK_WD0.md`
- `RANKRECIPE_PREREG.md` (2): `RANKCLOCK_PREREG.md` `RANK_PROJ_RESULTS.md`
- `RANKRECIPE_RESULTS.md` (4): `LOOP_RANK_E1800_P1_RESULTS.md` `RANKRECIPE_ANALYSIS.txt` `RANKRECIPE_PREREG.md` `RANKRECIPE_VERDICTS.json`
- `RANKWD_PREREG.md` (2): `RANKCLOCK_RESULTS.md` `RANKWD_RESULTS.md`
- `RANKWD_RESULTS.md` (5): `RANKWD.md` `RANKWD_ANALYSIS.txt` `RANKWD_PREREG.md` `RANKWD_VERDICTS.json` `RANKWD_WD0.md`
- `RANK_MATCHED_PREREG.md` (3): `CODE_RESULTS.md` `RANK_MATCHED_RESULTS.md` `RANK_SWEEP.md`
- `RANK_MATCHED_RESULTS.md` (8): `RANK_MATCHED.json` `RANK_MATCHED_GEOMETRY.md` `RANK_MATCHED_PREREG.md` `RANK_MATCHED_STRATA.json` `RANK_MATCHED_e900_ANALYSIS.txt` `RANK_MATCHED_e900c_ANALYSIS.txt` `RANK_PROJ_RESULTS.md` `RANK_SWEEP_STRATA.json`
- `RANK_MI_PREREG.md` (1): `RANK_PERHEAD_PILOT_RESULTS.md`
- `RANK_MI_RESULTS.md` (4): `RANK_MI_ANALYSIS.txt` `RANK_MI_GEOMETRY.md` `RANK_MI_PREREG.md` `RANK_PROJ_RESULTS.md`
- `RANK_ND_PREREG.md` (2): `RANK3_RESULTS.md` `RANK_SEP_RESULTS.md`
- `RANK_ND_RESULTS.md` (5): `RANK_MATCHED_RESULTS.md` `RANK_ND.md` `RANK_ND_ANALYSIS.txt` `RANK_ND_PREREG.md` `RANK_ND_SECONDARY.txt`
- `RANK_NOWRAP_PREREG.md` (2): `RANK_ND_RESULTS.md` `SCORE_RANK_RESULTS.md`
- `RANK_PERHEAD_PILOT_RESULTS.md` (2): `RANK_PERHEAD_PILOT_ANALYSIS.txt` `RANK_PERHEAD_PREREG.md`
- `RANK_PERHEAD_PREREG.md` (1): `RANK_PROJ_RESULTS.md`
- `RANK_PROJ_PREREG.md` (1): `RANK_MATCHED_PREREG.md`
- `RANK_PROJ_RESULTS.md` (3): `RANK_PROJ_ANALYSIS.txt` `RANK_PROJ_PREREG.md` `SEARCH_RESULTS.md`
- `RANK_SEP_PREREG.md` (1): `RANK_MI_RESULTS.md`
- `RANK_SEP_RESULTS.md` (4): `RANK_MI_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_ANALYSIS.txt` `RANK_SEP_PREREG.md`
- `RANK_SWEEP.md` (5): `ACTION_GEOMETRY.md` `GATE_PROBE.md` `RANK_MI_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_TRUNCATION.md`
- `RANK_TESTS_PREREG.md` (1): `LEAK_RESULTS.md`
- `RANK_TESTS_RESULTS.md` (3): `RANK_TESTS_ANALYSIS.txt` `RANK_TESTS_PREREG.md` `RANK_TESTS_VERDICTS.json`
- `RANK_WRAP_PREREG.md` (1): `RANK_ND_RESULTS.md`
- `RANK_WRAP_RESULTS.md` (5): `RANK_ND_RESULTS.md` `RANK_WRAP.json` `RANK_WRAP_AMENDED.txt` `RANK_WRAP_ANALYSIS.txt` `RANK_WRAP_PREREG.md`
- `RECENCY_EM_RESULTS.md` (5): `AUDIT_2026-09-10.md` `RECENCY_RESULTS.md` `REC_EM_PREREG.md` `TALE_OF_TWO_ALGORITHMS.md` `THEORY_KERNEL.md`
- `RECENCY_PREREG.md` (2): `RECENCY_GATES.md` `RECENCY_GATES_K64.md`
- `RECENCY_RESULTS.md` (3): `RECENCY_GATES_K64.md` `RECENCY_GATE_ABLATION.md` `RECENCY_PREREG.md`
- `RECENCY_T3_PREREG.md` (3): `MAPPOPE_R4_RESULTS.md` `RECENCY_RESULTS.md` `T3GEN_RESULTS.md`
- `RECENCY_T3_RESULTS.md` (1): `RECENCY_T3_PREREG.md`
- `REC_EM_PREREG.md` (3): `RECENCY_GATES_K64.md` `RECENCY_RESULTS.md` `TALE_OF_TWO_ALGORITHMS.md`
- `REPORT.md` (13): `DAGGER_RESULTS.md` `DOG_RESULTS.md`(del) `DOORKEY_BC_RESULTS.md` `MINIGRID_DK16_RESULTS.md` `MINIGRID_DOORKEY_LONGT.md` `MINIGRID_MEMORY_RESULTS.md` `MULTICLASS_MULTISEED_RESULTS.md` `MULTICLASS_RESULTS.md`(del) `NODROP_PARETO_RESULTS.md` `TEM_BACKGROUND_BASELINES.md` `TEM_CROSSSCALE_DIAGNOSTIC.md` `TEM_RESULTS.md` `VOCAB_SWEEP_RESULTS.md`
- `REPORT_ADDENDUM.md` (5): `AP_KERNEL_DIAGNOSTIC.md` `HIERGOAL_ABLATION.md` `PAPER_OOD_PROTOCOL.md` `PERSCALE_OMEGA_RESULTS.md` `REPORT.md`
- `REPORT_v2.md` (27): `AP_KERNEL_DIAGNOSTIC.md` `CAPACITY_CONTROL.md` `CAPACITY_PERREGIME.md` `DAGGER_RESULTS.md` `DOORKEY_BC_RESULTS.md` `EM_HOPFIELD_CROSSSCALE.md` `EXTRAHEAD_CONTROL.md` `HIERGOAL_ABLATION.md` `HOPFIELD_NOMAINAP_RESULTS.md` `LEVEL15BETA_RESULTS.md` `LEVEL15EM_CROSSSCALE.md` `MINIGRID_DK16_RESULTS.md` `MINIGRID_MEMORY_RESULTS.md` `MULTICLASS_MULTISEED_RESULTS.md` `MULTISEED_FOLLOWUP_RESULTS.md` `NODROP_PARETO_RESULTS.md` `NUMBERLINE_RESULTS.md` `PAPER_OOD_PROTOCOL.md` `PERSCALE_OMEGA_RESULTS.md` `REPORT.md` `REPORT_ADDENDUM.md` `RESULTS_PAPER.md` `TEM_BACKGROUND_BASELINES.md` `TEM_CROSSSCALE_DIAGNOSTIC.md` `TEM_NOISE_FFN_RESULTS.md` `TEM_T_MULTISEED.md` `VOCAB_SWEEP_RESULTS.md`
- `RESULTS_PAPER.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `LM200_CORRECTED_MULTISEED.md` `NOISE_CLEAN_REVALIDATION.md`
- `RESULTS_SUMMARY_2026-05-10.md` (11): `AP_KERNEL_DIAGNOSTIC.md` `DAGGER_DK6_RESULTS.md` `DAGGER_EMPTY_RESULTS.md` `DAGGER_RESULTS.md` `DOORKEY_BC_RESULTS.md` `HIERGOAL_ABLATION.md` `LEVEL15BETA_RESULTS.md` `NODROP_PARETO_RESULTS.md` `PAPER_OOD_PROTOCOL.md` `PAPER_VALIDATION.md` `VOCAB_SWEEP_RESULTS.md`
- `REVISIT_2X2.md` (1): `_PAPER2X2_RAW.json`
- `RING_BUDGET_PREREG.md` (1): `RING_DIAL_RESULTS.md`
- `RING_BUDGET_RESULTS.md` (4): `RING_BUDGET_ANALYSIS.txt` `RING_BUDGET_PREREG.md` `RING_BUDGET_VERDICTS.json` `RING_DIAL_RESULTS.md`
- `RING_DIAL_RESULTS.md` (4): `RING_DIAL_ANALYSIS.txt` `RING_DIAL_PREREG.md` `RING_DIAL_VERDICTS.json` `SIGN_MATCHED_RESULTS.md`
- `RNN_TORUS_PREREG.md` (4): `EXPTORUS_PREREG.md` `RNN_TORUS_ANALYSIS.txt` `RNN_TORUS_CD.json` `RNN_TORUS_VERDICTS.json`
- `RNN_TORUS_RESULTS.md` (4): `RNN_TORUS_ANALYSIS.txt` `RNN_TORUS_CD.json` `RNN_TORUS_PREREG.md` `RNN_TORUS_VERDICTS.json`
- `ROBUST_PREREG.md` (2): `ROBUST_ANALYSIS.txt` `ROBUST_VERDICTS.json`
- `ROBUST_RESULTS.md` (3): `ROBUST_ANALYSIS.txt` `ROBUST_PREREG.md` `ROBUST_VERDICTS.json`
- `SAMEBLOCK_PREREG.md` (3): `ADDITION_CHO_REPRO.md` `ADDITION_DESIGN.md` `ADDITION_PILOT.md`
- `SAMEBLOCK_RAW.md` (1): `SAMEBLOCK_PREREG.md`
- `SAMEBLOCK_RESULTS.md` (4): `ADDITION_PILOT2.md` `SAMEBLOCK_COMPILE_CHECK.md` `SAMEBLOCK_PREREG.md` `SAMEBLOCK_RAW.md`
- `SCORE_RANK_PREREG.md` (6): `LOOP_RANK_E1800_P1_RESULTS.md` `MAPPOPE_PAIR_PREREG.md` `MAPPOPE_PAIR_RESULTS.md` `RANK_MI_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md`
- `SCORE_RANK_RESULTS.md` (5): `MAPPOPE_PAIR_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_WRAP_RESULTS.md` `SCORE_RANK_ANALYSIS.txt` `SCORE_RANK_PREREG.md`
- `SEARCH_PREREG.md` (3): `EM_WM_STATE.md` `MAGONLY_RESULTS.md` `_REWIND_PROBE.json`
- `SEARCH_RESULTS.md` (2): `AUDIT_2026-09-10.md` `SEARCH_PREREG.md`
- `SELECTIVE_ROPE.md` (1): `RANK_SWEEP.md`
- `SESSION_2026-05-01.md` (2): `CNAV_RESULTS.md`(del) `DOG_RESULTS.md`(del)
- `SESSION_HIERARCHICAL_CASCADE.md` (2): `HIPPOCAMPAL_ANALYSIS.md`(del) `MULTISEED_FOLLOWUP.md`(del)
- `SIGNROUTE_PREREG.md` (1): `LRSWEEP_RESULTS.md`
- `SIGNROUTE_RESULTS.md` (3): `SIGNROUTE_ANALYSIS.txt` `SIGNROUTE_PREREG.md` `SIGNROUTE_VERDICTS.json`
- `SIGNSEP_PREREG.md` (2): `SIGNSEP_ANALYSIS.txt` `SIGNSEP_VERDICTS.json`
- `SIGNSEP_RESULTS.md` (3): `SIGNSEP_ANALYSIS.txt` `SIGNSEP_PREREG.md` `SIGNSEP_VERDICTS.json`
- `SIGN_ABLATION.md` (1): `SIGN_PROBE.md`
- `SIGN_ABLATION_PREREG.md` (1): `ALGORITHMIC_RESULTS.md`
- `SIGN_MATCHED_PREREG.md` (3): `RANK_MATCHED_RESULTS.md` `RANK_SEP_RESULTS.md` `SIGN_ABLATION.md`
- `SIGN_MATCHED_RESULTS.md` (7): `SIGN_ABLATION.md` `SIGN_MATCHED.md` `SIGN_MATCHED_ANALYSIS.txt` `SIGN_MATCHED_PREREG.md` `SIGN_MATCHED_PROBE.md` `SIGN_MATCHED_PROBE_SIGNED.md` `SIGN_MATCHED_STRATA.json`
- `SPREAD2_PREREG.md` (4): `RECENCY_GATES_K16SET.md` `RECENCY_GATES_K4SET.md` `SPREAD_PREREG.md` `SPREAD_RESULTS.md`
- `SPREAD2_RESULTS.md` (1): `SPREAD2_PREREG.md`
- `SPREAD_PREREG.md` (4): `RECENCY_GATES_K16SET.md` `RECENCY_GATES_K4SET.md` `RECENCY_GATES_K64.md` `SEARCH_RESULTS.md`
- `SPREAD_RESULTS.md` (2): `H12_BUDGET_CURVE.md` `SPREAD_PREREG.md`
- `SROPE_TORUS_PREREG.md` (5): `GENERALITY_RESULTS.md` `MONOTONE_RESULTS.md` `SELECTIVE_ROPE.md` `SROPE_TORUS_ANALYSIS.txt` `SROPE_TORUS_VERDICTS.json`
- `SROPE_TORUS_RESULTS.md` (4): `SIGN_MATCHED_RESULTS.md` `SROPE_TORUS_ANALYSIS.txt` `SROPE_TORUS_PREREG.md` `SROPE_TORUS_VERDICTS.json`
- `T1_PREREG.md` (1): `THEORY_MAPPOPE.md`
- `T1_RESULTS.md` (1): `T2_RESULTS.md`
- `T2_PREREG.md` (2): `DYCK_PREREG.md` `THEORY_MAPPOPE.md`
- `T2_RESULTS.md` (1): `T2_PREREG.md`
- `T3GEN_PREREG.md` (3): `ACCUMULATOR.md` `DYCK_PREREG.md` `T3_RESULTS.md`
- `T3GEN_RESULTS.md` (2): `T2_RESULTS.md` `T3GEN_PREREG.md`
- `T3_PREREG.md` (1): `THEORY_MAPPOPE.md`
- `T3_RESULTS.md` (3): `T2_RESULTS.md` `T3GEN_RESULTS.md` `T3_PREREG.md`
- `TALE_OF_TWO_ALGORITHMS.md` (3): `AUDIT_2026-09-10.md` `PLANNER_TASK_AUDIT.md` `VOCAB_EM.md`
- `TEM_NOISE_FFN_RESULTS.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NODROP_PARETO_RESULTS.md` `NOISE_CLEAN_REVALIDATION.md`
- `TEM_RECENCY_DIAG.md` (1): `TEM_RECENCY_PILOT.md`
- `TEM_RECENCY_PILOT.md` (1): `MONOTONE_RESULTS.md`
- `TEM_RESULTS.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md` `TEM_BACKGROUND_BASELINES.md`
- `TEM_T_MULTISEED.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md` `TEM_T_RESULTS.md`
- `TEM_T_RESULTS.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md` `TEM_BACKGROUND_BASELINES.md`
- `TEXTWORLD_PREREG.md` (2): `TEXTWORLD_RESULTS.md` `TEXTWORLD_SECONDARY.json`
- `TEXTWORLD_RESULTS.md` (5): `TEXTWORLD_ANALYSIS.txt` `TEXTWORLD_PREREG.md` `TEXTWORLD_PROBE.json` `TEXTWORLD_SECONDARY.txt` `TW_NORMSTEP_RESULTS.md`
- `THEORY_KERNEL.md` (8): `ACCUMULATOR.md` `AUDIT_2026-09-10.md` `DOF_PREREG.md` `DOF_RESULTS.md` `N5_RESULTS.md` `RECENCY_EM_RESULTS.md` `RECENCY_RESULTS.md` `UNFREEZE_RESULTS.md`
- `THEORY_MAPPOPE.md` (7): `JSB_LENGTH_RESULTS_BASE.md` `MAGONLY_RESULTS.md` `T1_RESULTS.md` `T2_RESULTS.md` `T3GEN_RESULTS.md` `T3_RESULTS.md` `THEORY_NUMBERS.md`
- `THEORY_NARRATIVE.md` (40): `ACCUMULATOR.md` `ACTION_GEOMETRY.md` `ALIASING_CONTROLLED.md` `AUDIT_2026-09-10.md` `D5_RESULTS.md` `DXR_RANK_THRESHOLD.md` `EM_WM_STATE.md` `EM_WM_THEORY.md` `FORGET_CLOCK_PREREG.md` `FORGET_CONTROL.md` `FORGET_GATE.md` `GATED_RESULTS.md` `GATE_PROBE.md` `L15_ABLATION.md` `LAMBDA_TRACE.md` `LOCALISATION.md` `MAGONLY_RESULTS.md` `MAPPOPE_R4_RESULTS.md` `N5_RESULTS.md` `NOLEAK_RESULTS.md` `PAIRCONST_RESULTS.md` `PAIRORIGIN_RESULTS.md` `PAIRSPLIT_RESULTS.md` `PAPERTASK_RESULTS.md` `PAPER_FIG4_REPRO.md` `PAPER_TASK_FLOORS.md` `RANK_SWEEP.md` `RECENCY_EM_RESULTS.md` `RECENCY_GATE_ABLATION.md` `RECENCY_RESULTS.md` `SEARCH_RESULTS.md` `SIGN_ABLATION.md` `SPREAD2_RESULTS.md` `SPREAD_RESULTS.md` `TALE_OF_TWO_ALGORITHMS.md` `THEORY_KERNEL.md` `THEORY_SEARCH_AND_LENGTH.md` `UNFREEZE_RESULTS.md` `VISITS_TEST.md` `WARM_RESULTS.md`
- `THEORY_NUMBERS.md` (2): `T3GEN_RESULTS.md` `THEORY_MAPPOPE.md`
- `THEORY_SEARCH_AND_LENGTH.md` (2): `EM_WM_THEORY.md` `SEARCH_RESULTS.md`
- `TINYSTORIES_PILOT_RESULTS.md` (1): `TINYSTORIES_PILOT.md`
- `TORUS_LRSWEEP_PREREG.md` (3): `LRSWEEP_PREREG.md` `TORUS_LRSWEEP_ANALYSIS.txt` `TORUS_LRSWEEP_VERDICTS.json`
- `TORUS_LRSWEEP_RESULTS.md` (3): `TORUS_LRSWEEP_ANALYSIS.txt` `TORUS_LRSWEEP_PREREG.md` `TORUS_LRSWEEP_VERDICTS.json`
- `TORUS_START_PREREG.md` (1): `MECHANISM_RESULTS.md`
- `TORUS_START_RESULTS.md` (3): `TORUS_START_ANALYSIS.txt` `TORUS_START_PREREG.md` `TORUS_START_VERDICTS.json`
- `TW_AMBIG_PREREG.md` (4): `CONTEXT_STEP_DESIGN.md` `CTXSTEP_HSR_PILOT.md` `CTXSTEP_PREREG.md` `TEXTWORLD_RESULTS.md`
- `TW_LANDMARK_PREREG.md` (3): `GAIN_GRAIN_RESULTS.md` `LEAK_RESULTS.md` `TEXTWORLD_RESULTS.md`
- `TW_NORMSTEP_PREREG.md` (2): `LEAK_RESULTS.md` `TEXTWORLD_PREREG.md`
- `TW_NORMSTEP_RESULTS.md` (3): `TW_NORMSTEP.json` `TW_NORMSTEP_ANALYSIS.txt` `TW_NORMSTEP_PREREG.md`
- `TW_STATECHANGE_PREREG.md` (2): `TEXTWORLD_RESULTS.md` `TW_NORMSTEP_RESULTS.md`
- `TW_STATECHANGE_RESULTS.md` (4): `TW_STATECHANGE.json` `TW_STATECHANGE_ANALYSIS.txt` `TW_STATECHANGE_PREREG.md` `TW_STATECHANGE_VERDICTS.json`
- `UNFREEZE_PREREG.md` (2): `WARM_PREREG.md` `WARM_RESULTS.md`
- `UNFREEZE_RESULTS.md` (5): `MAGONLY_RESULTS.md` `NOLEAK_RESULTS.md` `SEARCH_RESULTS.md` `UNFREEZE_PREREG.md` `WARM_RESULTS.md`
- `V3_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `V4_CONTROL_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `V4_MULTISEED.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `V4_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `VOCAB_EM.md` (3): `RECENCY_EM_RESULTS.md` `VOCAB_EM_PREREG.md` `VOCAB_SWEEP_MULTISEED.md`
- `VOCAB_EM_PREREG.md` (3): `COMP_HEADROOM.md` `MINIGRID_EM_FIX.md` `RECIPE_POWER.md`
- `VOCAB_SWEEP_MULTISEED.md` (1): `VOCAB_SWEEP_RESULTS.md`
- `VOCAB_SWEEP_RESULTS.md` (1): `VOCAB_SWEEP_MULTISEED.md`
- `WARM_PREREG.md` (1): `AUDIT_2026-09-10.md`
- `WARM_RESULTS.md` (4): `SEARCH_RESULTS.md` `UNFREEZE_RESULTS.md` `WARM_PREREG.md` `_REWIND_PROBE.json`
- `WRAP_COPE_PREREG.md` (1): `MECHANISM_RESULTS.md`
- `WRAP_COPE_RESULTS.md` (3): `WRAP_COPE_ANALYSIS.txt` `WRAP_COPE_PREREG.md` `WRAP_COPE_VERDICTS.json`
- `WRAP_EXISTENCE_PREREG.md` (2): `RING_BUDGET_RESULTS.md` `RING_DIAL_RESULTS.md`
- `WRAP_EXISTENCE_RESULTS.md` (4): `ACCOUNTS_RESULTS.md` `WRAP_EXISTENCE_ANALYSIS.txt` `WRAP_EXISTENCE_PREREG.md` `WRAP_EXISTENCE_VERDICTS.json`

</details>

#### D. History and inventories (do not rewrite; covered by a MOVED map)

<details><summary>180 files</summary>

- `archive/RESULTS_INDEX_2026-08-18.md` (130): `ABLATE_COMPOSITIONAL.md` `ABLATE_FAMILY_TREE.md` `AGGREGATE_EXTRAS.md` `AGGREGATE_MULTISEED.md` `AGGREGATE_TASK_RESULTS.md` `AP_KERNEL_DIAGNOSTIC.md` `BOUNDED_MEMORY.md` `BOUNDED_MEMORY_RESULTS.md` `BUMP_TOKEN_RESULTS.md` `CAPACITY_CONTROL.md` `CAPACITY_PERREGIME.md` `CASCADE_MULTISEED_RESULTS.md` `CASCADE_REPRO_TEST.md` `CASCADE_ZEROSHOT_S0.md` `CLOCK_SCAN.md` `CLONE_ANALYSIS_LEVEL15PC.md`(del) `CLONE_TRANSFER_NOBYPASS.md`(del) `CNAV_HEX_Level15.md`(del) `CNAV_HEX_Level15EM.md`(del) `CNAV_HEX_Vanilla.md`(del) `CNAV_HEX_VanillaEM.md`(del) `CNAV_RESULTS.md`(del) `COMPOSITIONAL_EXPERIMENT.md` `COMPOSITIONAL_MULTISEED.md` `COMPOSITIONAL_RESULTS.md` `CORRECTED_LM200_LEADERBOARD.md` `DAGGER_DK6_RESULTS.md` `DAGGER_EMPTY_RESULTS.md` `DAGGER_RESULTS.md` `DETAILED_RESULTS.md` `DOG_RESULTS.md`(del) `DOORKEY_BC_RESULTS.md` `EM_COMP_SAMEBATCH.md` `EM_FIX_COMP.md` `EM_HOPFIELD_CROSSSCALE.md` `EM_P0_COMP.md` `EM_P0_PAPER.md` `EXTRAHEAD_CONTROL.md` `FAMILY_TREE_D7_GATES.md` `FAMILY_TREE_GATES.md` `FAMILY_TREE_RESULTS.md` `GENERALIZATION_REPORT.md` `GSF_FULL_RESULTS.md` `HIERGOAL_ABLATION.md` `HIERGOAL_CLOSEDLOOP.md` `HIERGOAL_LONGT.md` `HIER_ATTN_LONGT.md` `HIPPOCAMPAL_ANALYSIS.md`(del) `HIPPOCAMPAL_GRID.md`(del) `HIPPOCAMPAL_GRIDL15PC.md`(del) `HIPPOCAMPAL_GRID_FREE.md`(del) `HIPPOCAMPAL_HIDDEN.md`(del) `HIPPOCAMPAL_HIDDEN_GRIDFREE.md`(del) `HIPPOCAMPAL_LEVEL15PC.md`(del) `HOPFIELD_NOMAINAP_RESULTS.md` `INDEX_BASELINE_PAPER_TASK.md` `LAP_GATES.md` `LAP_TRANSFER.md` `LAP_TRANSFER_NOREWARD.md` `LEVEL15BETA_RESULTS.md` `LEVEL15EM_CROSSSCALE.md` `LEVEL15_MEETS_GATED_matchq.md` `LEVEL15_MEETS_GATED_paper50.md` `LM200_CORRECTED_MULTISEED.md` `LONG_SEQ_clean.md` `MAP_QUERY_GATES.md` `MAP_QUERY_RESULTS.md` `MATCH_GATES_128_16.md` `MATCH_GATES_64_16.md` `MATCH_GATES_64_4.md` `MATCH_QUERY_EM.md` `MATCH_QUERY_GATES.md` `MATCH_QUERY_LONGQ.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `MINIGRID_DK16_RESULTS.md` `MINIGRID_DOORKEY_CACHED.md` `MINIGRID_DOORKEY_LONGT.md` `MINIGRID_DOORKEY_RESULTS.md` `MINIGRID_DOORKEY_ROPE_DIAG.md` `MINIGRID_MEMORY_RESULTS.md` `MULTICLASS_MULTISEED_RESULTS.md` `MULTICLASS_RESULTS.md`(del) `MULTISEED_FOLLOWUP.md`(del) `MULTISEED_FOLLOWUP_RESULTS.md` `NOBYPASS_RESULTS.md` `NODROP_PARETO_RESULTS.md` `NOISE_CLEAN_REVALIDATION.md` `NUMBERLINE_RESULTS.md` `OMEGA_RESCALE_clean.md` `OOD_GRID_RESULTS.md`(del) `PAPER_OOD_PROTOCOL.md` `PAPER_OOD_WITH_POPE.md` `PAPER_TASK_ABLATION.md` `PAPER_TASK_ACCURACY.md` `PAPER_VALIDATION.md` `PERSCALE_OMEGA_RESULTS.md` `PER_VISIT_clean.md` `PLANNER_TASK_AUDIT.md` `PUBLICATION_VENUES.md` `RECURSIVE_RESULTS.md` `REPORT.md` `REPORT_ADDENDUM.md` `REPORT_v2.md` `RESULTS_PAPER.md` `RESULTS_SUMMARY_2026-05-10.md` `REVISIT_DISTANCE.md` `ROUTE_ATTN_RESULTS.md` `R_T_DISTRIBUTION_3WAY.md`(del) `SESSION_2026-05-01.md` `SESSION_HIERARCHICAL_CASCADE.md` `SPACETIME_HIER_RESULTS.md` `STITCH_ATTENTION.md` `TEM_BACKGROUND_BASELINES.md` `TEM_CROSSSCALE_DIAGNOSTIC.md` `TEM_NOISE_FFN_RESULTS.md` `TEM_RESULTS.md` `TEM_T_MULTISEED.md` `TEM_T_RESULTS.md` `TIMING_BENCHMARK.md` `TOPOLOGY_RESULTS.md` `V3_RESULTS.md` `V4_CONTROL_RESULTS.md` `V4_MULTISEED.md` `V4_RESULTS.md` `VECTOR_NAV_V2_RESULTS.md` `VOCAB_SWEEP_MULTISEED.md` `VOCAB_SWEEP_RESULTS.md` `ZERO_SHOT_TRANSFER_clean.md` `ZERO_SHOT_TRANSFER_clean_brokeninit.md`(del)
- `archive/void/ACTIVE_INFERENCE_RESULTS.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md` `PLANNER_TASK_AUDIT.md`
- `archive/void/AUX_COEF_SWEEP.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/CASCADE_NOSLOW_CONTROL.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/CLONE_TRANSFER_TEST.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/DIMSWEEP_d128.md` (2): `HIERGOAL_ABLATION.md` `HIERGOAL_CLOSEDLOOP.md`
- `archive/void/DIMSWEEP_d256.md` (2): `HIERGOAL_ABLATION.md` `HIERGOAL_CLOSEDLOOP.md`
- `archive/void/DIMSWEEP_d512.md` (2): `HIERGOAL_ABLATION.md` `HIERGOAL_CLOSEDLOOP.md`
- `archive/void/DROPOUT_ABLATION_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/EM_FIX_HIERGOAL.md` (2): `HIERGOAL_ABLATION.md` `HIERGOAL_CLOSEDLOOP.md`
- `archive/void/GOAL_CLOSEDLOOP_RESULTS.md` (1): `PLANNER_TASK_AUDIT.md`
- `archive/void/GOAL_DIRECTED_RESULTS.md` (1): `PLANNER_TASK_AUDIT.md`
- `archive/void/GOAL_TASKS_RESULTS.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md` `PLANNER_TASK_AUDIT.md`
- `archive/void/GSF_MODES_DIAGNOSTIC.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/GSF_NODROP_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/GSF_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/HEX_EMERGENCE_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/HIERGOAL_FIXED.md` (1): `HIERGOAL_ABLATION.md`
- `archive/void/HIERGOAL_FIXED_LONGT.md` (1): `HIERGOAL_ABLATION.md`
- `archive/void/HIERGOAL_MULTISEED.md` (1): `HIERGOAL_ABLATION.md`
- `archive/void/HIERGOAL_RESULTS.md` (3): `CLOCK_SCAN.md` `HIERGOAL_ABLATION.md` `coarse_angles.json`
- `archive/void/HIER_ATTN_MULTIENV.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/LENGTH_DIAGNOSTIC.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/LONGT_EVAL_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/LONG_SEQ_lm200.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/MODEOMEGA_FINEGRAINED.md` (1): `CORRECTED_LM200_LEADERBOARD.md`
- `archive/void/MODEOMEGA_LESION.md` (1): `CORRECTED_LM200_LEADERBOARD.md`
- `archive/void/MODEOMEGA_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/MULTIENV_CLEAN_2x2.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/MULTIENV_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/MULTISIZE_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/OMEGA_RESCALE_lm200.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/PER_VISIT_lm200.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/PROBE_GOAL_DISTANCE.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md` `PLANNER_TASK_AUDIT.md`
- `archive/void/PROBE_GOAL_RESULTS.md` (1): `PLANNER_TASK_AUDIT.md`
- `archive/void/README.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `HIERGOAL_ABLATION.md` `PLANNER_TASK_AUDIT.md`
- `archive/void/ROOMS_MAZE_RESULTS.md` (1): `PLANNER_TASK_AUDIT.md`
- `archive/void/ROOMS_TASK_RESULTS.md` (1): `PLANNER_TASK_AUDIT.md`
- `archive/void/R_T_DISTRIBUTION.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/SINGLE_SIZE_CONTROL.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md` `TEM_CROSSSCALE_DIAGNOSTIC.md`
- `archive/void/SPARSE_LANDMARKS_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/SR_PRETRAIN_RESULTS.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md` `PLANNER_TASK_AUDIT.md`
- `archive/void/SR_PROBE_RESULTS.md` (3): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md` `PLANNER_TASK_AUDIT.md`
- `archive/void/STATE_PROBES.md` (1): `PLANNER_TASK_AUDIT.md`
- `archive/void/TEM_NOVEL_ENV_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/TEM_SCALING_RESULTS.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/VANILLANODROP_CONTROL.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/VARYING_MAZE_RESULTS.md` (1): `PLANNER_TASK_AUDIT.md`
- `archive/void/VECTOR_NAV_RESULTS.md` (1): `PLANNER_TASK_AUDIT.md`
- `archive/void/ZERO_SHOT_TRANSFER_lm200.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `archive/void/ZERO_SHOT_TRANSFER_lm200_brokeninit.md` (2): `CORRECTED_LM200_LEADERBOARD.md` `NOISE_CLEAN_REVALIDATION.md`
- `docs/IMAGE_TASKS_SURVEY_2026-10-25.md` (9): `ALIASING_CONTROLLED.md` `ALLOCENTRIC_RECODING.md` `HABITAT_BUILD.md` `IMAGE_TORUS_PREREG.md` `IMAGE_TORUS_RESULTS.md` `KNOB_SWEEP_n8.md` `MINIGRID_ALLOCENTRIC_2X2X2.md` `PERCEPTION_EXPERIMENT_PLAN.md` `VISITS_TEST.md`
- `docs/LOG.md` (119): `ABLATE_RESULTS.md` `ACCUMULATOR.md` `ACTION_GEOMETRY.md` `ALGORITHMIC_RESULTS.md` `ALIASING_CONTROLLED.md` `AUDIT_2026-09-10.md` `AUDIT_HEADLINE.md` `AUG_RESULTS.md` `BASELINE_TABLE.md` `BF16_RESULTS.md` `CLONE_ANALYSIS_LEVEL15PC.md`(del) `CLONE_TRANSFER_NOBYPASS.md`(del) `CODE_DECAY_RESULTS.md` `CODE_GATES.md` `CODE_PREREG.md` `CODE_RESULTS.md` `CODE_RESULTS_OOD.md` `COMPOSITIONAL_EXPERIMENT.md` `COMPOSITIONAL_MULTISEED.md` `COMPOSITIONAL_RESULTS.md` `COMP_HEADROOM.md` `CORRECTED_LM200_LEADERBOARD.md` `CROSS_RESULTS.md` `DAGGER_RESULTS.md` `DECAY_RESULTS.md` `DOG_RESULTS.md`(del) `DOORKEY_BC_RESULTS.md` `DYCK_DECAY_PROBE.md` `DYCK_FAR_PROBE.md` `DYCK_GATES.md` `DYCK_LADDER_RESULTS.json` `DYCK_LADDER_RESULTS.md` `DYCK_LITERATURE_METRICS.md` `DYCK_RESULTS_POPE.md` `DYCK_RESULTS_bs128.md` `DYCK_T3_RESULTS.md` `EM_WM_STATE.md` `ENWIK8_HIERARCHY.md` `ENWIK8_SEEDS.md` `FAMILY_TREE_RESULTS.md` `FLIPFLOP_RESULTS.md` `FORGET_CLOCK_PREREG.md` `FRONTIER_ALGORITHMIC.md` `GATED_RESULTS.md` `GATE_PROBE.md` `H12_BUDGET_CURVE.md` `HABITAT_BUILD.md` `HIERGOAL_ABLATION.md` `HIER_RECHECK.md` `HIPPOCAMPAL_LEVEL15PC.md`(del) `HORIZON_RESULTS.md` `INDIRECT_ARITHMETIC.md` `INDIRECT_OOD.md` `INDIRECT_RESULTS_200k.md` `JSB_LENGTH_RESULTS.md` `JSB_RESULTS.md` `L15_ABLATION.md` `L15_LOOP_2X2.md` `LANGUAGE_LANDSCAPE.md` `LOCALISATION.md` `LOOPED_PILOT.md` `LOOP_DEPTH_STRATA.md` `LOOP_HEADROOM.md` `LOOP_HIER_COMPUTE.md` `LOOP_HIER_PARITY.md` `LOOP_SAMPLED.md` `MAESTRO_PLAN.md` `MAGONLY_RESULTS.md` `MAPPOPE_R4_RESULTS.md` `MATCH_QUERY_EM.md` `MATCH_QUERY_SCALE.md` `MINIGRID_DOORKEY_LONGT.md` `MINIGRID_DOORKEY_ROPE_DIAG.md` `MINIGRID_FULL_2X2X2.md` `MQAR_RESULTS.md` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md` `N3_AUDIT.md` `NOBYPASS_RESULTS.md` `NODROP_PARETO_RESULTS.md` `NOISE_CLEAN_REVALIDATION.md` `NOISE_REFINE.md` `NOLEAK_RESULTS.md` `PAIRORIGIN_RESULTS.md` `PAIRSPLIT_RESULTS.md` `PAPER_FIG4_REPRO.md` `PAPER_TASK_ACCURACY.md` `PLANNER_TASK_AUDIT.md` `RANK_MATCHED_PREREG.md` `RANK_MATCHED_RESULTS.md` `RANK_SWEEP.json` `RANK_SWEEP.md` `RECENCY_GATE_ABLATION.md` `RECENCY_RESULTS.md` `RECENCY_T3_RESULTS.md` `RECIPE_POWER.md` `RESULTS_PAPER.md` `ROPE_CANONICAL.md` `R_T_DISTRIBUTION_3WAY.md`(del) `SEARCH_PREREG.md` `SEARCH_RESULTS.md` `SELECTIVE_ROPE.md` `SESSION_2026-05-01.md` `SIGN_ABLATION.md` `SPREAD2_RESULTS.md` `SPREAD_RESULTS.md` `T2_RESULTS.md` `T3GEN_RESULTS.md` `TALE_OF_TWO_ALGORITHMS.md` `THEORY_KERNEL.md` `TIMING_BENCHMARK.md` `TORUS_T3_RESULTS.md` `UNFREEZE_RESULTS.md` `V4_CONTROL_RESULTS.md` `V4_MULTISEED.md` `VISITS_TEST.md` `VOCAB_SWEEP_MULTISEED.md` `VOCAB_SWEEP_RESULTS.md` `WARM_RESULTS.md`
- `docs/NORMSTEP_NOTES.md` (2): `LEAK_RESULTS.md` `TW_NORMSTEP_RESULTS.md`
- `docs/RUNBOOK_NEXT_EXPERIMENTS.md` (2): `DELTA_FREE_RESULTS.md` `ROBUST_PREREG.md`
- `docs/SESSION_2026-09-27_to_10-03.md` (14): `CANCEL_RESULTS.md` `CODE_DECAY_RESULTS.md` `CODE_FULLVAL_RESULTS.md` `CODE_RESULTS.md` `CTXSTEP_HSR_PILOT.md` `CTXSTEP_HS_RECIPE.md` `CTXSTEP_PREREG.md` `LEAK_RESULTS.md` `LOOP_RANK_E1800_P1_RESULTS.md` `RANK3_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_WRAP_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md`
- `docs/SESSION_2026-10-12_autonomous.md` (9): `CLOCK_SIDE_RESULTS.md` `CLOCK_SPEED_REP_RESULTS.md` `CLOCK_SPEED_RESULTS.md` `FACTORIAL_RESULTS.md` `GENERALITY_RESULTS.md` `MECHANISM_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/WHAT_WHERE_ANALYSIS.md` (13): `ABLATE_RESULTS.md` `AUDIT_2026-09-10.md` `CTXSTEP_HS_RECIPE.md` `CTXSTEP_PILOT3.md` `EM_WM_STATE.md` `EM_WM_THEORY.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `PAIRSPLIT_RESULTS.md` `PAPER2X2_RESULTS.md` `T2_RESULTS.md` `TEXTWORLD_PROBE.json` `TEXTWORLD_RESULTS.md`
- `docs/audits/2026-09-24/README.md` (2): `FAST_ATTN_RANK.md` `L15_ABLATION.json`
- `docs/audits/2026-09-24/applied/driver_dryrun/results.txt` (4): `RANK_PERHEAD_PILOT.md` `RANK_PERHEAD_PILOT_GEOMETRY.md` `RANK_PERHEAD_PILOT_STRATA.json` `RANK_SWEEP_STRATA.json`
- `docs/audits/2026-09-24/applied/dyck_ladder_check.txt` (2): `DYCK_LADDER_RESULTS.json` `DYCK_LADDER_RESULTS.md`
- `docs/audits/2026-09-24/applied/evals_env.txt` (12): `RANK_MATCHED_e900.json` `RANK_MATCHED_e900.md` `RANK_MATCHED_e900_ANALYSIS.txt` `RANK_MATCHED_e900_GEOMETRY.md` `RANK_MATCHED_e900_STRATA.json` `RANK_MATCHED_e900c_ANALYSIS.txt` `RANK_PERHEAD_PILOT_ANALYSIS.txt` `RANK_PERHEAD_PILOT_GEOMETRY.md` `RANK_PROJ_ANALYSIS.txt` `RANK_PROJ_TRAIN.json` `RANK_PROJ_TRAIN.md` `RANK_PROJ_TRAIN_STRATA.json`
- `docs/audits/2026-09-24/applied/evals_scalefold.txt` (15): `L15_ABLATION.json` `NOISE_REFINE.json` `NOISE_REFINE.md` `RANK_MATCHED_e900.json` `RANK_MATCHED_e900.md` `RANK_MATCHED_e900_ANALYSIS.txt` `RANK_MATCHED_e900_GEOMETRY.md` `RANK_MATCHED_e900_STRATA.json` `RANK_MATCHED_e900c_ANALYSIS.txt` `RANK_PERHEAD_PILOT_ANALYSIS.txt` `RANK_PERHEAD_PILOT_GEOMETRY.md` `RANK_PROJ_ANALYSIS.txt` `RANK_PROJ_TRAIN.json` `RANK_PROJ_TRAIN.md` `RANK_PROJ_TRAIN_STRATA.json`
- `docs/audits/2026-09-24/applied/evals_strata_stats.txt` (15): `RANK_MATCHED_STRATA.json` `RANK_MATCHED_e900.json` `RANK_MATCHED_e900.md` `RANK_MATCHED_e900_ANALYSIS.txt` `RANK_MATCHED_e900_GEOMETRY.md` `RANK_MATCHED_e900_STRATA.json` `RANK_MATCHED_e900c_ANALYSIS.txt` `RANK_MATCHED_e900c_STRATA.json` `RANK_PERHEAD_PILOT_ANALYSIS.txt` `RANK_PERHEAD_PILOT_GEOMETRY.md` `RANK_PROJ_ANALYSIS.txt` `RANK_PROJ_TRAIN.json` `RANK_PROJ_TRAIN.md` `RANK_PROJ_TRAIN_STRATA.json` `RANK_SWEEP_STRATA.json`
- `docs/audits/2026-09-24/applied/geometry_default_check.txt` (2): `RANK_MATCHED_e900_GEOMETRY.md` `RANK_PERHEAD_PILOT_GEOMETRY.md`
- `docs/audits/2026-09-24/applied/verify_perm_repo.txt` (3): `RANK_MATCHED.json` `RANK_MATCHED_e900.json` `RANK_MATCHED_e900c.json`
- `docs/audits/2026-09-24/context/HISTORY_PLAN.md` (7): `DETAILED_RESULTS.md` `DYCK_DEPTH_RESULTS.md` `DYCK_T3_RESULTS.md` `FORGET_CONTROL.md` `MINIGRID_EM.md` `RESULTS_PAPER.md` `T1_PREREG.md`
- `docs/audits/2026-09-24/context/REPORT.md` (10): `DYCK_DEPTH_RESULTS.md` `DYCK_T3_RESULTS.md` `EM_WM_STATE.md` `LANGUAGE_LANDSCAPE.md` `LOOP_DEPTH_STRATA.md` `MAPPOPE_R4_RESULTS.md` `NOISE_REFINE.md` `RANK_MATCHED_RESULTS.md` `RECIPE_POWER.md` `REPORT.md`
- `docs/audits/2026-09-24/efficiency/REPORT.md` (1): `REPORT.md`
- `docs/audits/2026-09-24/experiments/EXPERIMENTS.md` (2): `ABLATE_RESULTS.md` `DYCK_DEPTH_RESULTS.json`
- `docs/audits/2026-09-24/experiments/LEDGER.md` (66): `ABLATE_RESULTS.md` `ACTION_GEOMETRY.md` `ALIASING_CONTROLLED.md` `ALLOCENTRIC_RECODING.md` `AUG_RESULTS.md` `BF16_RESULTS.md` `CODE_DECAY_RESULTS.md` `CODE_GATES.md` `CODE_RESULTS.md` `CROSSOVER_CONVERGED.md` `CROSS_RESULTS.md` `D5_RESULTS.md` `DECAY_RESULTS.md` `DYCK_DEPTH_RESULTS.md` `DYCK_LADDER_RESULTS.json` `DYCK_LADDER_RESULTS.md` `DYCK_RESULTS_bs128.md` `EM_WM_THEORY.md` `FAMILY_TREE_RESULTS.md` `FORGET_CLOCK_PREREG.md` `FORGET_CONTROL.md` `FORGET_GATE.md` `GATED_RESULTS.md` `H12_BUDGET_CURVE.md` `HIER_PARITY.md` `HIER_RECHECK.md` `INDEX_BASELINE_PAPER_TASK_n8.json` `INDIRECT_RESULTS_200k.md` `JSB_LENGTH_RESULTS.md` `JSB_RESULTS.md` `KNOB_SWEEP_n8.md` `L15_ABLATION.md` `L15_LOOP_2X2.md` `LAMBDA_TRACE.md` `LM200_CORRECTED_MULTISEED.md` `LOOP_DEPTH_STRATA.md` `LOOP_HEADROOM.md` `LOOP_SAMPLED.md` `MAGONLY_RESULTS.md` `MAPPOPE_R4_RESULTS.md` `MATCH_QUERY_SCALE.md` `MINIGRID_FULL_2X2X2.md` `MQ_NOISE_2X2_C2.md` `MQ_RANK_2X2.md` `N3_AUDIT.md` `NOLEAK_RESULTS.md` `PAIRSPLIT_RESULTS.md` `PAPER2X2_RAW_CONTRASTS.json` `PAPER2X2_RESULTS.md` `PAPER_FIG4_REPRO.md` `POPE_WRAPPING.md` `POSITION_EFFECT_CONVERGED.md` `RANK_MATCHED_RESULTS.md` `RANK_MATCHED_e900.md` `RANK_PROJ_FROZEN.md` `RANK_SWEEP.md` `RECENCY_EM_RESULTS.md` `RECENCY_RESULTS.md` `RECIPE_POWER.md` `REFINE_RESULTS.md` `REVISIT_DISTANCE.md` `SIGN_ABLATION.md` `STITCH_ATTENTION.md` `TIMING_BENCHMARK.md` `VISITS_TEST.md` `WARM_RESULTS.md`
- `docs/audits/2026-09-24/experiments/PROPAGATION.md` (6): `BASELINE_TABLE.md` `DOG_RESULTS.md`(del) `EM_WM_STATE.md` `INDEX_BASELINE_PAPER_TASK.md` `PAPER2X2_RESULTS.md` `REFINE_RESULTS.md`
- `docs/audits/2026-09-24/experiments/REPORT_TEX_LEDGER.md` (99): `ACCUMULATOR.md` `ACTION_GEOMETRY.md` `ALGORITHMIC_RESULTS.md` `ALIASING_CONTROLLED.md` `AUDIT_2026-09-10.md` `AUG_RESULTS.md` `COMP_HEADROOM.md` `CROSSOVER_CONVERGED.md` `CROSS_RESULTS.md` `D5_RESULTS.md` `DECAY_RESULTS.md` `DOF_RESULTS.md` `DXR_RANK_THRESHOLD.md` `DYCK_DECAY_RESULTS.md` `DYCK_LITERATURE_METRICS.md` `DYCK_RESULTS_POPE.md` `DYCK_RESULTS_bs128.md` `DYCK_STACK_PROBE.md` `ENWIK8_HIERARCHY.md` `FAMILY_TREE_WM_GAP.md` `FORGET_CONTROL.md` `FORGET_GATE.md` `FRONTIER_ALGORITHMIC.md` `GATED_RESULTS.md` `GATED_TORUS.json` `H12_BUDGET_CURVE.md` `HABITAT_BUILD.md` `HIER_PARITY.md` `HIER_RECHECK.md` `INDEX_BASELINE_PAPER_TASK_n8.md` `INDIRECT_RESULTS_200k.md` `JSB_LENGTH_RESULTS.md` `JSB_RESULTS.md` `KNOB_SWEEP.md` `KNOB_SWEEP_n8.json` `KNOB_SWEEP_n8.md` `L15_ABLATION.json` `L15_ABLATION.md` `L15_LOOP_2X2.md` `LAMBDA_TRACE.md` `LM200_ABLATION.md` `LOCALISATION.md` `LOOP_HEADROOM.md` `LOOP_HIER_COMPUTE.md` `LOOP_HIER_PARITY.md` `MAGONLY_RESULTS.md` `MAPPOPE_R4_RESULTS.md` `MATCH_QUERY_EM.md` `MATCH_QUERY_LONGQ.md` `MATCH_QUERY_SCALE.md` `MINIGRID_EM.md` `MINIGRID_EM_FIX.md` `MINIGRID_FULL_2X2X2.json` `MINIGRID_FULL_2X2X2.md` `MINIWORLD_ENDPOINTS.md` `MONOTONE_RAW.json` `MONOTONE_RESULTS.md` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md` `MQ_RANK_2X2.md` `N3_AUDIT.md` `NOLEAK_RESULTS.md` `PAIRCONST_RESULTS.md` `PAIRSPLIT_RESULTS.md` `PAPER2X2_RESULTS.md` `PAPERTASK_RESULTS.md` `PAPER_FIG4_REPRO.md` `PAPER_OOD_EXTENDED_n8.md` `PAPER_OOD_RERUN.md` `PAPER_TASK_ABLATION.md` `PAPER_TASK_ACCURACY.md` `POPE_WRAPPING.md` `POSITION_EFFECT_CONVERGED.md` `RANK_SWEEP.json` `RANK_SWEEP.md` `RECENCY_EM_RESULTS.md` `RECENCY_GATE_ABLATION.md` `RECENCY_H2.md` `RECENCY_RESULTS.md` `RECIPE_POWER.md` `REFINE_RESULTS.md` `REVISIT_DISTANCE.md` `SEARCH_RESULTS.md` `SELECTIVE_ROPE.md` `SIGN_ABLATION.md` `SIGN_PROBE.md` `SPREAD2_RESULTS.md` `T1_RESULTS.md` `T2_RESULTS.md` `T3GEN_RESULTS.md` `T3_RESULTS.md` `THEORY_MAPPOPE.md` `THEORY_SEARCH_AND_LENGTH.md` `TIMING_BENCHMARK.md` `UNFREEZE_RESULTS.md` `VISITS_TEST.md` `VOCAB_EM.md` `WARM_RESULTS.md` `_PAPER2X2_RAW.json`
- `docs/audits/2026-09-24/experiments/RETRACTIONS_LIST.md` (3): `DOG_RESULTS.md`(del) `DYCK_LADDER_RESULTS.md` `LOCALISATION.md`
- `docs/audits/2026-09-24/hygiene/REPORT.md` (7): `ABLATE_RESULTS.md` `BUMP_TOKEN_RESULTS.md` `CODE_RESULTS.md` `CODE_RESULTS_OOD.md` `EXTRAHEAD_CONTROL.md` `REPORT.md` `VECTOR_NAV_V2_RESULTS.md`
- `docs/audits/2026-09-24/rank_review.md` (8): `RANK_MATCHED.json` `RANK_MATCHED_GEOMETRY.md` `RANK_MATCHED_PREREG.md` `RANK_MATCHED_RESULTS.md` `RANK_MATCHED_STRATA.json` `RANK_MATCHED_e900.json` `RANK_SWEEP.json` `RANK_SWEEP_STRATA.json`
- `docs/audits/2026-09-24/rank_review/README.md` (4): `RANK_MATCHED_PREREG.md` `RANK_MATCHED_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md`
- `docs/audits/2026-09-24/rank_review2/README.md` (2): `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md`
- `docs/audits/2026-09-24/theory_REPORT.md` (16): `ABLATE_RESULTS.md` `ACCUMULATOR.md` `AUDIT_2026-09-10.md` `BASELINE_TABLE.md` `CODE_DECAY_RESULTS.md` `DYCK_DEPTH_RESULTS.md` `DYCK_LADDER_RESULTS.md` `DYCK_PREREG.md` `HIER_RECHECK.md` `INDIRECT_ARITHMETIC.md` `MQ_RANK_2X2.md` `PAPER2X2_RESULTS.md` `RANK_PROJ_PREREG.md` `RECENCY_RESULTS.md` `REPORT.md` `SIGN_ABLATION.md`
- `docs/audits/2026-09-25/rank_mi_review/README.md` (2): `RANK_MI_RESULTS.md` `RANK_SEP_RESULTS.md`
- `docs/audits/2026-10-04/rescore_other_compare_out.txt` (2): `LEAK.json` `RANK_ND.json`
- `docs/audits/2026-10-06/gain_phase_leak_validate_out.txt` (1): `LEAK.json`
- `docs/audits/2026-10-06/gain_phase_smoke_out.txt` (1): `GAIN_PHASE_EVAL.json`
- `docs/audits/2026-10-11/CRITICAL_REVIEW.md` (12): `GAIN_GRAIN_ANALYSIS.txt` `GAIN_PHASE_ANALYSIS.txt` `GAIN_PHASE_PREREG.md` `GAIN_PHASE_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `SCORE_RANK_ANALYSIS.txt` `SCORE_RANK_RESULTS.md` `TINYSTORIES_PILOT_RESULTS.md` `TW_NORMSTEP_RESULTS.md` `TW_STATECHANGE_ANALYSIS.txt` `TW_STATECHANGE_PREREG.md` `TW_STATECHANGE_RESULTS.md`
- `docs/audits/2026-10-11/rank_tests_audit.md` (1): `LEAK_RESULTS.md`
- `docs/audits/2026-10-11/rank_tests_checks_out.txt` (1): `CANCEL_PREREG.md`
- `docs/audits/2026-10-12/clock_side_audit.md` (8): `CLOCK_SIDE_ANALYSIS.txt` `CLOCK_SIDE_GATES.json` `CLOCK_SIDE_GATES.md` `CLOCK_SIDE_PREREG.md` `CLOCK_SIDE_VERDICTS.json` `RECENCY_PREREG.md` `RECENCY_RESULTS.md` `SIGN_MATCHED_RESULTS.md`
- `docs/audits/2026-10-12/clock_side_gate8000_out.txt` (1): `CLOCK_SIDE_GATES_8000.md`
- `docs/audits/2026-10-12/clock_speed_audit.md` (3): `CLOCK_SPEED_ANALYSIS.txt` `CLOCK_SPEED_PREREG.md` `CLOCK_SPEED_VERDICTS.json`
- `docs/audits/2026-10-12/clock_speed_rep_audit.md` (4): `CLOCK_SPEED_REP_ANALYSIS.txt` `CLOCK_SPEED_REP_PREREG.md` `CLOCK_SPEED_REP_VERDICTS.json` `CLOCK_SPEED_VERDICTS.json`
- `docs/audits/2026-10-12/factorial_audit.md` (4): `FACTORIAL_PREREG.md` `FACTORIAL_VERDICTS.json` `HORIZON_RESULTS.md` `LOOPED_PILOT.md`
- `docs/audits/2026-10-12/ring_budget_audit.md` (2): `RING_BUDGET_PREREG.md` `RING_DIAL_ANALYSIS.txt`
- `docs/audits/2026-10-13/wrap_existence_audit.md` (1): `WRAP_EXISTENCE_PREREG.md`
- `docs/audits/2026-10-14/accounts_audit.md` (1): `ACCOUNTS_PREREG.md`
- `docs/audits/2026-10-14/cope_diagnosis/PREDICTIONS.txt` (1): `WRAP_COPE_ANALYSIS.txt`
- `docs/audits/2026-10-14/generality_audit.md` (2): `GENERALITY_PREREG.md` `GENERALITY_VERDICTS.json`
- `docs/audits/2026-10-14/mechanism_audit.md` (1): `MECHANISM_PREREG.md`
- `docs/audits/2026-10-14/torus_start_audit.md` (1): `TORUS_START_PREREG.md`
- `docs/audits/2026-10-14/wrap_cope_audit.md` (4): `MECHANISM_VERDICTS.json` `WRAP_COPE_ANALYSIS.txt` `WRAP_COPE_PREREG.md` `WRAP_COPE_VERDICTS.json`
- `docs/audits/2026-10-15/gauge_audit.md` (1): `GAUGE_PREREG.md`
- `docs/audits/2026-10-16/lrsweep_audit.md` (2): `LRSWEEP_PREREG.md` `ROBUST_PREREG.md`
- `docs/audits/2026-10-16/robust_audit.md` (2): `GAUGE_PREREG.md` `ROBUST_PREREG.md`
- `docs/audits/2026-10-20/delta_free_audit.md` (2): `DELTA_FREE_PREREG.md` `PAPER2X2_PREREG.md`
- `docs/audits/2026-10-20/delta_free_check/e2e_ANALYSIS.txt` (1): `DELTA_FREE_PREREG.md`
- `docs/audits/2026-10-23/g3_ssm_audit.md` (2): `G3_SSM_PREREG.md` `G3_SSM_VERDICTS.json`
- `docs/audits/2026-10-24/g4_torus_lrsweep_audit.md` (1): `TORUS_LRSWEEP_PREREG.md`
- `docs/audits/2026-10-25/factorial_n8_audit.md` (4): `FACTORIAL_L1_STRATA.json` `FACTORIAL_N8_PREREG.md` `G3_SSM_PREREG.md` `LM200_ABLATION.md`
- `docs/audits/2026-10-25/g3_a2_audit.md` (4): `G3_SSM_ANALYSIS.txt` `G3_SSM_PREREG.md` `G3_SSM_RESULTS.md` `G3_SSM_VERDICTS.json`
- `docs/audits/2026-10-25/image_torus_amendment2_audit.md` (5): `IMAGE_TORUS_ANALYSIS.txt` `IMAGE_TORUS_PILOT.txt` `IMAGE_TORUS_PILOT2.txt` `IMAGE_TORUS_PREREG.md` `IMAGE_TORUS_VERDICTS.json`
- `docs/audits/2026-10-25/image_torus_audit.md` (1): `IMAGE_TORUS_PREREG.md`
- `docs/audits/2026-10-26/CRITICAL_REVIEW.md` (12): `ALIASING_CONTROLLED.md` `FACTORIAL_N8_RESULTS.md` `G3_SSM_A2_RESULTS.md` `G3_SSM_RESULTS.md` `IMAGE_TORUS_PREREG.md` `IMAGE_TORUS_RESULTS.md` `KNOB_SWEEP_n8.md` `RANKRECIPE_RESULTS.md` `RANKWD_RESULTS.md` `SIGNSEP_RESULTS.md` `TORUS_LRSWEEP_RESULTS.md` `VISITS_TEST.md`
- `docs/audits/2026-10-26/exptorus_audit.md` (1): `EXPTORUS_PREREG.md`
- `docs/audits/2026-10-26/rnn_torus_audit.md` (1): `RNN_TORUS_PREREG.md`
- `docs/audits/2026-10-27/memory_maze_audit.md` (2): `MEMORY_MAZE_GATE_V3.md` `MEMORY_MAZE_PREREG.md`
- `docs/audits/2026-10-28/FULL_REVIEW.md` (33): `DELTA_FREE_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `EXPTORUS_VERDICTS.json` `FACTORIAL_N8_RESULTS.md` `FACTORIAL_N8_VERDICTS.json` `G3_SSM_A2_RESULTS.md` `G3_SSM_A2_VERDICTS.json` `G3_SSM_RESULTS.md` `G3_SSM_VERDICTS.json` `IMAGE_TORUS_RESULTS.md` `IMAGE_TORUS_VERDICTS.json` `LOOP_RANK_E1800_P1_RESULTS.md` `MEMORY_MAZE_GATE_V3.md` `MEMORY_MAZE_RESULTS.md` `MEMORY_MAZE_VERDICTS.json` `PAPER2X2_RESULTS.md` `RANKCLOCK_RESULTS.md` `RANKCLOCK_VERDICTS.json` `RANKRECIPE_RESULTS.md` `RANKRECIPE_VERDICTS.json` `RANKWD_RESULTS.md` `RANKWD_VERDICTS.json` `RANK_PROJ_RESULTS.md` `RNN_TORUS_RESULTS.md` `RNN_TORUS_VERDICTS.json` `RNN_TORUS_anchor.json` `ROBUST_RESULTS.md` `SIGNROUTE_RESULTS.md` `SIGNSEP_RESULTS.md` `SIGNSEP_VERDICTS.json` `SIGN_MATCHED_RESULTS.md` `SROPE_TORUS_RESULTS.md` `TORUS_LRSWEEP_VERDICTS.json`
- `docs/audits/2026-10-28/memory_maze/pose_only_probe_out.txt` (1): `MEMORY_MAZE_ANALYSIS.txt`
- `docs/lit/LIT_CLOCK_MAP.md` (5): `CODE_FULLVAL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `RECENCY_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TINYSTORIES_PILOT_RESULTS.md`
- `docs/lit/LIT_CONTEXT_STEPS.md` (4): `CONTEXT_STEP_DESIGN.md` `CTXSTEP_HSR_PILOT.md` `CTXSTEP_HS_RECIPE.md` `TEXTWORLD_RESULTS.md`
- `docs/lit/LIT_NEW_OBJECTS.md` (1): `PAPER2X2_RESULTS.md`
- `docs/lit/LIT_WHAT_WHERE.md` (2): `PAIRSPLIT_RESULTS.md` `RANK_WRAP_RESULTS.md`
- `docs/paper/APPENDIX_draft5.md` (35): `ACCOUNTS_ANALYSIS.txt` `ACCOUNTS_RESULTS.md` `CANCEL_PREREG.md` `CANCEL_RESULTS.md` `CLOCK_SIDE_RESULTS.md` `CLOCK_SPEED_REP_RESULTS.md` `CLOCK_SPEED_RESULTS.md` `DYCK_MDEPTH_PREREG.md` `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_L1_STRATA.json` `FACTORIAL_RESULTS.md` `GAUGE_RESULTS.md` `GENERALITY_ANALYSIS.txt` `GENERALITY_PREREG.md` `GENERALITY_RESULTS.md` `GENERALITY_TORUS_SO4_STRATA.json` `MECHANISM_ANALYSIS.txt` `MECHANISM_PREREG.md` `MECHANISM_RESULTS.md` `RING_BUDGET_ANALYSIS.txt` `RING_DIAL_ANALYSIS.txt` `RING_DIAL_VERDICTS.json` `ROBUST_ANALYSIS.txt` `ROBUST_RESULTS.md` `SCORE_RANK_PREREG.md` `SIGN_MATCHED.json` `SIGN_MATCHED_ANALYSIS.txt` `SIGN_MATCHED_STRATA.json` `TEXTWORLD_RESULTS.md` `TORUS_START_ANALYSIS.txt` `TORUS_START_PREREG.md` `WRAP_COPE_RESULTS.md` `WRAP_EXISTENCE_ANALYSIS.txt` `WRAP_EXISTENCE_PREREG.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/APPENDIX_draft6.md` (35): `ACCOUNTS_ANALYSIS.txt` `ACCOUNTS_RESULTS.md` `CANCEL_PREREG.md` `CANCEL_RESULTS.md` `CLOCK_SIDE_RESULTS.md` `CLOCK_SPEED_REP_RESULTS.md` `CLOCK_SPEED_RESULTS.md` `DYCK_MDEPTH_PREREG.md` `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_L1_STRATA.json` `FACTORIAL_RESULTS.md` `GAUGE_RESULTS.md` `GENERALITY_ANALYSIS.txt` `GENERALITY_PREREG.md` `GENERALITY_RESULTS.md` `GENERALITY_TORUS_SO4_STRATA.json` `MECHANISM_ANALYSIS.txt` `MECHANISM_PREREG.md` `MECHANISM_RESULTS.md` `RING_BUDGET_ANALYSIS.txt` `RING_DIAL_ANALYSIS.txt` `RING_DIAL_VERDICTS.json` `ROBUST_ANALYSIS.txt` `ROBUST_RESULTS.md` `SCORE_RANK_PREREG.md` `SIGN_MATCHED.json` `SIGN_MATCHED_ANALYSIS.txt` `SIGN_MATCHED_STRATA.json` `TEXTWORLD_RESULTS.md` `TORUS_START_ANALYSIS.txt` `TORUS_START_PREREG.md` `WRAP_COPE_RESULTS.md` `WRAP_EXISTENCE_ANALYSIS.txt` `WRAP_EXISTENCE_PREREG.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/APPENDIX_theory_v2.md` (3): `ACCOUNTS_RESULTS.md` `GAUGE_RESULTS.md` `MECHANISM_RESULTS.md`
- `docs/paper/APPENDIX_theory_v3.md` (3): `ACCOUNTS_RESULTS.md` `GAUGE_RESULTS.md` `MECHANISM_RESULTS.md`
- `docs/paper/BIG_PICTURE_2026-10-24.md` (5): `FACTORIAL_RESULTS.md` `GENERALITY_RESULTS.md` `RANKWD_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md`
- `docs/paper/DRAFT3_changes.md` (18): `ACCOUNTS_ANALYSIS.txt` `ACCOUNTS_RESULTS.md` `CLOCK_SPEED_REP_RESULTS.md` `CLOCK_SPEED_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_RESULTS.md` `GENERALITY_ANALYSIS.txt` `GENERALITY_RESULTS.md` `MECHANISM_ANALYSIS.txt` `MECHANISM_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `TORUS_START_ANALYSIS.txt` `TORUS_START_RESULTS.md` `WRAP_COPE_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/DRAFT4_changes.md` (13): `FACTORIAL_RESULTS.md` `GAUGE_PREREG.md` `GAUGE_RESULTS.md` `GENERALITY_PREREG.md` `GENERALITY_RESULTS.md` `MECHANISM_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_RESULTS.md` `SIGN_MATCHED_PREREG.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `TORUS_START_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/DRAFT5_changes.md` (14): `ACCOUNTS_ANALYSIS.txt` `ACCOUNTS_RESULTS.md` `ACCOUNTS_VERDICTS.json` `FACTORIAL_RESULTS.md` `GAUGE_RESULTS.md` `MECHANISM_PREREG.md` `MECHANISM_RESULTS.md` `RING_DIAL_VERDICTS.json` `ROBUST_RESULTS.md` `SIGN_MATCHED.json` `SIGN_MATCHED_RESULTS.md` `SIGN_MATCHED_STRATA.json` `TORUS_START_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/DRAFT6_changes.md` (3): `EXPTORUS_RESULTS.md` `G3_SSM_A2_RESULTS.md` `GAUGE_RESULTS.md`
- `docs/paper/MAIN_APPENDIX1.md` (34): `ACCOUNTS_RESULTS.md` `CANCEL_PREREG.md` `CANCEL_RESULTS.md` `DYCK_MDEPTH_RESULTS.json` `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_RESULTS.md` `GAIN_PHASE_RESULTS.md` `GAUGE_RESULTS.md` `GENERALITY_ANALYSIS.txt` `LEAK_RESULTS.md` `LOOP_RANK_E1800_P1_ANALYSIS.txt` `LRSWEEP_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `MECHANISM_RESULTS.md` `PAPER_TASK_ACCURACY.md` `RANK3.json` `RANK3_ANALYSIS.txt` `RANK_MI_ANALYSIS.txt` `RANK_ND_RESULTS.md` `RANK_PROJ_ANALYSIS.txt` `RANK_PROJ_RESULTS.md` `RANK_SEP.json` `RANK_SEP_ANALYSIS.txt` `RANK_TESTS_RESULTS.md` `RANK_WRAP_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_VERDICTS.json` `ROBUST_ANALYSIS.txt` `ROBUST_RESULTS.md` `SIGN_MATCHED_ANALYSIS.txt` `TEXTWORLD_ANALYSIS.txt` `TEXTWORLD_RESULTS.md` `TORUS_START_RESULTS.md` `WRAP_EXISTENCE_ANALYSIS.txt`
- `docs/paper/MAIN_DRAFT1_notes.md` (32): `ACCOUNTS_RESULTS.md` `CANCEL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_RESULTS.md` `GAIN_PHASE_RESULTS.md` `GENERALITY_RESULTS.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LRSWEEP_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `MECHANISM_RESULTS.md` `PAPER2X2_RESULTS.md` `PAPER_TASK_ACCURACY.md` `RANK3.json` `RANK3_ANALYSIS.txt` `RANK3_RESULTS.md` `RANK_MI_ANALYSIS.txt` `RANK_MI_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_PROJ_ANALYSIS.txt` `RANK_PROJ_RESULTS.md` `RANK_SEP.json` `RANK_SEP_ANALYSIS.txt` `RANK_SEP_RESULTS.md` `RANK_TESTS_RESULTS.md` `RANK_WRAP_RESULTS.md` `RING_DIAL_RESULTS.md` `ROBUST_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `TINYSTORIES_PILOT_RESULTS.md` `TORUS_START_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/MAIN_DRAFT2_notes.md` (58): `ACCOUNTS_RESULTS.md` `CANCEL_RESULTS.md` `CODE_FULLVAL_RESULTS.md` `DELTA_FREE_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_N8_ANALYSIS.txt` `FACTORIAL_N8_RESULTS.md` `FACTORIAL_N8_VERDICTS.json` `FACTORIAL_RESULTS.md` `G3_SSM_A2_ANALYSIS.txt` `G3_SSM_A2_RESULTS.md` `G3_SSM_A2_VERDICTS.json` `G3_SSM_ANALYSIS.txt` `G3_SSM_RESULTS.md` `G3_SSM_VERDICTS.json` `GAIN_PHASE_RESULTS.md` `GENERALITY_RESULTS.md` `IMAGE_TORUS_ANALYSIS.txt` `IMAGE_TORUS_RESULTS.md` `IMAGE_TORUS_VERDICTS.json` `KNOB_SWEEP_n8.md` `LEAK_RESULTS.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LOOP_RANK_RESULTS.md` `LOOP_RANK_STRATA.json` `LRSWEEP_RESULTS.md` `MEMORY_MAZE_ANALYSIS.txt` `PAPER2X2_RESULTS.md` `PAPER_TASK_ACCURACY.md` `RANKCLOCK.md` `RANKCLOCK_ANALYSIS.txt` `RANKCLOCK_PREREG.md` `RANKCLOCK_RESULTS.md` `RANKCLOCK_VERDICTS.json` `RANKRECIPE_ANALYSIS.txt` `RANKRECIPE_RESULTS.md` `RANKRECIPE_VERDICTS.json` `RANK_MI_RESULTS.md` `RANK_SEP_RESULTS.md` `RANK_TESTS_RESULTS.md` `ROBUST_RESULTS.md` `SCORE_RANK_RESULTS.md` `SEARCH_RESULTS.md` `SIGNROUTE_RESULTS.md` `SIGNSEP_ANALYSIS.txt` `SIGNSEP_RESULTS.md` `SIGNSEP_VERDICTS.json` `SIGN_MATCHED_RESULTS.md` `SPREAD_RESULTS.md` `SROPE_TORUS_RESULTS.md` `TEXTWORLD_RESULTS.md` `TINYSTORIES_PILOT_RESULTS.md` `TORUS_LRSWEEP_ANALYSIS.txt` `TORUS_LRSWEEP_RESULTS.md` `TORUS_LRSWEEP_VERDICTS.json` `TW_NORMSTEP_RESULTS.md` `WARM_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/MOCK_REVIEW_draft3.md` (12): `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_RESULTS.md` `GENERALITY_RESULTS.md` `MECHANISM_ANALYSIS.txt` `MECHANISM_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_RESULTS.md` `SIGN_MATCHED_PREREG.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `TORUS_START_ANALYSIS.txt` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/OUTLINE_main_v2.md` (24): `ACCOUNTS_RESULTS.md` `CANCEL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_RESULTS.md` `GAIN_PHASE_RESULTS.md` `GENERALITY_RESULTS.md` `LEAK_RESULTS.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LRSWEEP_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `MECHANISM_RESULTS.md` `PAPER2X2_RESULTS.md` `RANK3_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md` `RANK_TESTS_RESULTS.md` `RANK_WRAP_RESULTS.md` `RING_DIAL_RESULTS.md` `ROBUST_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `TINYSTORIES_PILOT_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/OUTLINE_overall.md` (39): `ACCOUNTS_RESULTS.md` `ALIASING_CONTROLLED.md` `CANCEL_ANALYSIS.txt` `CANCEL_RESULTS.md` `CODE_FULLVAL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `EM_WM_STATE.md` `FACTORIAL_RESULTS.md` `GAIN_PHASE_RESULTS.md` `GENERALITY_RESULTS.md` `KNOB_SWEEP_n8.md` `LEAK_RESULTS.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LOOP_RANK_RESULTS.md` `LOOP_SAMPLED.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `MECHANISM_RESULTS.md` `PAPER2X2_RESULTS.md` `RANK3_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md` `RANK_TESTS_RESULTS.md` `RANK_WRAP_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_RESULTS.md` `ROBUST_RESULTS.md` `SEARCH_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `SIGN_MATCHED_STRATA.json` `TEXTWORLD_PROBE.json` `TEXTWORLD_RESULTS.md` `TEXTWORLD_SECONDARY.json` `TINYSTORIES_PILOT_RESULTS.md` `TORUS_START_RESULTS.md` `TW_STATECHANGE_RESULTS.md` `VISITS_TEST.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/OUTLINE_workshop.md` (15): `CLOCK_SIDE_RESULTS.md` `CLOCK_SPEED_REP_RESULTS.md` `CLOCK_SPEED_RESULTS.md` `CODE_FULLVAL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_RESULTS.md` `GENERALITY_RESULTS.md` `MECHANISM_RESULTS.md` `PAPER2X2_RESULTS.md` `RING_BUDGET_RESULTS.md` `RING_DIAL_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `TINYSTORIES_PILOT_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/READ_draft4.md` (3): `MECHANISM_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `WRAP_EXISTENCE_RESULTS.md`
- `docs/paper/expansion/DEPTH_THEORY.md` (4): `CANCEL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `FACTORIAL_L4_STRATA.json` `FACTORIAL_RESULTS.md`
- `docs/paper/expansion/LANGUAGE.md` (8): `CODE_FULLVAL_RESULTS.md` `GAIN_PHASE_RESULTS.md` `LEAK_RESULTS.md` `TEXTWORLD_RESULTS.md` `TEXTWORLD_SECONDARY.txt` `TINYSTORIES_PILOT_RESULTS.md` `TW_NORMSTEP_RESULTS.md` `TW_STATECHANGE_RESULTS.md`
- `docs/paper/expansion/MAPPOPE.md` (8): `ABLATE_RESULTS.md` `GAIN_GRAIN_RESULTS.md` `GAIN_PHASE_RESULTS.md` `INDIRECT_RESULTS_200k.md` `JSB_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `PAPER2X2_RESULTS.md` `SCORE_RANK_RESULTS.md`
- `docs/paper/expansion/PAPER_LEVEL.md` (11): `CANCEL_RESULTS.md` `EM_WM_STATE.md` `LOOP_RANK_E1800_PREREG.md` `NOLEAK_RESULTS.md` `PAIRSPLIT_RESULTS.md` `RANK_NOWRAP_PREREG.md` `RECENCY_EM_RESULTS.md` `ROBUST_PREREG.md` `SEARCH_RESULTS.md` `SPREAD_RESULTS.md` `WARM_RESULTS.md`
- `docs/paper/expansion/RANK_THEORY.md` (12): `DXR_RANK_THRESHOLD.md` `LEAK_RESULTS.md` `LOOP_RANK_E1800_P1_RESULTS.md` `RANK3_RESULTS.md` `RANK_MATCHED_RESULTS.md` `RANK_MI_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md` `RANK_TESTS_RESULTS.md` `RANK_WRAP_RESULTS.md` `SCORE_RANK_RESULTS.md`
- `docs/paper/expansion/RECURRENCE.md` (14): `FACTORIAL_RESULTS.md` `L15_LOOP_2X2.md` `LOOPED_PILOT.md` `LOOP_DEPTH_STRATA.md` `LOOP_HEADROOM.md` `LOOP_RANK_ANALYSIS.txt` `LOOP_RANK_E1800_P1_RESULTS.md` `LOOP_RANK_GEOMETRY.md` `LOOP_RANK_PREREG.md` `LOOP_RANK_RESULTS.md` `LOOP_RANK_STRATA.json` `MQ_RANK_2X2.md` `RANK_MI_STRATA.json` `REFINE_RESULTS.md`
- `docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md` (1): `RANK3_RESULTS.md`
- `docs/paper/figs/FIGS_NOTES.md` (14): `DYCK_MDEPTH_RESULTS.json` `GENERALITY_ANALYSIS.txt` `MECHANISM_RESULTS.md` `RANK3_ANALYSIS.txt` `RANK3_STRATA.json` `RANK_MI_STRATA.json` `RANK_ND_ANALYSIS.txt` `RANK_PROJ_ANALYSIS.txt` `RANK_SEP_ANALYSIS.txt` `RANK_TESTS_ANALYSIS.txt` `RANK_WRAP_ANALYSIS.txt` `SIGN_MATCHED_STRATA.json` `TEXTWORLD_SECONDARY.json` `WRAP_EXISTENCE_ANALYSIS.txt`
- `docs/paper/figs/fig_rank_solved_out.txt` (2): `RANK3_ANALYSIS.txt` `RANK_SEP_ANALYSIS.txt`
- `docs/paper/figs/torus_strata_floors_s0-7.json` (3): `RANK3_STRATA.json` `RANK_MI_STRATA.json` `SIGN_MATCHED_STRATA.json`
- `docs/paper/figs/torus_strata_floors_s500-503.json` (3): `FACTORIAL_L1_STRATA.json` `FACTORIAL_L4_STRATA.json` `FACTORIAL_LOOP_STRATA.json`
- `docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md` (3): `DYCK_MDEPTH_RESULTS.json` `SIGN_MATCHED_STRATA.json` `TEXTWORLD_SECONDARY.json`
- `docs/paper/review_draft5/MOCK_REVIEW_draft5.md` (2): `MECHANISM_RESULTS.md` `ROBUST_RESULTS.md`
- `docs/paper/review_draft5/NUMBERS_AUDIT.md` (7): `CANCEL_RESULTS.md` `GAUGE_RESULTS.md` `RING_DIAL_VERDICTS.json` `ROBUST_PREREG.md` `SIGN_MATCHED.json` `TEXTWORLD_ANALYSIS.txt` `TEXTWORLD_PREREG.md`
- `docs/paper/review_draft5/REPRO_AUDIT.md` (2): `ACCOUNTS_RESULTS.md` `SIGN_MATCHED.json`
- `docs/paper/review_draft5/REPRO_AUDIT_nonsign.md` (10): `LEAK.json` `LEAK_SECONDARY.txt` `LOOP_RANK_RESULTS.md` `MATCH_QUERY_LONGQ.json` `MQ_RANK_2X2.md` `RANK_MATCHED_PREREG.md` `RANK_WRAP.json` `TW_NORMSTEP.json` `TW_NORMSTEP_ANALYSIS.txt` `_PAPER2X2_RAW.json`
- `docs/paper/review_draft5/SYNTHESIS.md` (1): `GAUGE_RESULTS.md`
- `docs/plans/2026-10-11_next_gpu.md` (4): `RANK3_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_NOWRAP_PREREG.md` `RANK_SEP_RESULTS.md`
- `docs/prepared/dyck_mdepth/DYCK_MDEPTH_PREREG.md` (2): `DYCK_LADDER_RESULTS.md` `DYCK_MDEPTH_GATES.md`
- `docs/prepared/dyck_mdepth/README.md` (1): `DYCK_MDEPTH_PREREG.md`
- `docs/prepared/dyck_mdepth/mock_out/MOCK_RESULTS.md` (2): `DYCK_MDEPTH_GATES.md` `DYCK_MDEPTH_PREREG.md`
- `docs/prepared/tex_edits_2026-09-25/README.md` (2): `DYCK_MDEPTH_RESULTS.md` `RANK_SEP_RESULTS.md`
- `docs/prepared/tex_edits_2026-09-27/README.md` (3): `DYCK_MDEPTH_RESULTS.md` `LOOP_RANK_RESULTS.md` `RANK_SEP_RESULTS.md`
- `docs/theory/2026-10-04/01_synthesis.md` (22): `CANCEL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `FAMILY_TREE_RESULTS.md` `KNOB_SWEEP_n8.md` `LEAK_RESULTS.md` `LOOP_RANK_E1800_P1_RESULTS.md` `LOOP_RANK_RESULTS.md` `RANK3_RESULTS.md` `RANK_MATCHED_GEOMETRY.md` `RANK_MATCHED_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md` `RANK_WRAP_RESULTS.md` `RECENCY_GATE_ABLATION.md` `RECENCY_RESULTS.md` `SEARCH_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_PROBE.json` `TEXTWORLD_RESULTS.md` `TW_NORMSTEP_RESULTS.md` `VISITS_TEST.md`
- `docs/theory/2026-10-04/02_literature.md` (22): `ALIASING_CONTROLLED.md` `CANCEL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `EM_WM_STATE.md` `HIPPOCAMPAL_ANALYSIS.md`(del) `HIPPOCAMPAL_GRID_FREE.md`(del) `LAP_TRANSFER_NOREWARD.md` `LEAK_RESULTS.md` `MATCH_QUERY_SCALE.md` `PAPER2X2_RESULTS.md` `RANK3_GEOMETRY.md` `RANK3_RESULTS.md` `RANK_ND_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SEP_RESULTS.md` `RANK_WRAP_RESULTS.md` `RECENCY_RESULTS.md` `SEARCH_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `TW_NORMSTEP_RESULTS.md` `VISITS_TEST.md`
- `docs/theory/2026-10-04/03_formal.md` (9): `CANCEL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `LEAK_RESULTS.md` `LOOP_RANK_RESULTS.md` `RANK_MATCHED_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SWEEP.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md`
- `docs/theory/2026-10-04/04_language.md` (9): `CODE_FULLVAL_RESULTS.md` `CONTEXT_STEP_DESIGN.md` `CTXSTEP_HSR_PILOT.md` `CTXSTEP_HS_RECIPE.md` `CTXSTEP_PREREG.md` `DYCK_MDEPTH_RESULTS.md` `JSB_RESULTS.md` `TEXTWORLD_RESULTS.md` `TW_NORMSTEP_RESULTS.md`
- `docs/theory/2026-10-04/05_redteam.md` (13): `ALIASING_CONTROLLED.md` `CANCEL_RESULTS.md` `DYCK_MDEPTH_RESULTS.md` `LEAK_RESULTS.md` `MATCH_QUERY_SCALE.md` `PAPER2X2_RESULTS.md` `RANK3_GEOMETRY.md` `RANK3_RESULTS.md` `RANK_SEP_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `TW_NORMSTEP_RESULTS.md` `VISITS_TEST.md`
- `docs/theory/2026-10-05/neuro_design.md` (15): `CROSS_RESULTS.md` `EM_WM_STATE.md` `ENWIK8_HIERARCHY.md` `FREQ_CONTROL.md` `L15_ABLATION.md` `LEAK_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `MATCH_QUERY_SCALE.md` `MQ_NOISE_2X2.md` `NOISE_REFINE.md` `PAPER2X2_RESULTS.md` `RECENCY_RESULTS.md` `REFINE_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TW_NORMSTEP_RESULTS.md`
- `docs/theory/2026-10-05/neuro_positional.md` (11): `EM_WM_STATE.md` `ENWIK8_HIERARCHY.md` `FORGET_CONTROL.md` `HIPPOCAMPAL_ANALYSIS.md`(del) `L15_ABLATION.md` `LEAK_RESULTS.md` `MAPPOPE_PAIR_RESULTS.md` `PAPER2X2_RESULTS.md` `SIGN_MATCHED_RESULTS.md` `TEXTWORLD_RESULTS.md` `TIMING_BENCHMARK.md`
- `paper/RETRACTED.md` (4): `CORRECTED_LM200_LEADERBOARD.md` `L15_ABLATION.md` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md`
- `paper_rank/STATUS.md` (6): `ACTION_GEOMETRY.md` `GATE_PROBE.md` `ND_GATES.md` `PAPER_FIG4_REPRO.md` `RANK_SWEEP.md` `SELECTIVE_ROPE.md`
- `report/FIXES.md` (84): `ABLATE_FAMILY_TREE.md` `ACTION_GEOMETRY.md` `ALGORITHMIC_RESULTS.md` `ALIASING_CONTROLLED.md` `AUDIT_2026-09-10.md` `BASELINE_TABLE.md` `COMPOSITIONAL_EXPERIMENT.md` `COMP_HEADROOM.md` `CONTINUOUS_ALLOC.md` `CSCG_TASK_GATES.md` `D5_RESULTS.md` `DXR_RANK_THRESHOLD.md` `FAMILY_TREE_D7_RESULTS.md` `FAMILY_TREE_RESULTS.md` `FLIPFLOP_RESULTS.md` `FREQ_CONTROL.md` `GATED_RESULTS.md` `H12_BUDGET_CURVE.md` `HIER_PARITY.md` `HIER_RECHECK.md` `INDEX_BASELINE_PAPER_TASK_n8.md` `KNOB_SWEEP.md` `KNOB_SWEEP_n8.md` `L15_ABLATION.md` `L15_LOOP_2X2.md` `LANGUAGE_LANDSCAPE.md` `LM200_ABLATION.md` `LOCALISATION.md` `LONG_SEQ_clean.md` `LOOPED_PILOT.md` `LOOP_DEPTH_STRATA.md` `LOOP_HEADROOM.md` `MAPPOPE_R4_RESULTS.md` `MATCH_GATES_128_16.md` `MATCH_GATES_64_4.md` `MATCH_QUERY_GATES.md` `MATCH_QUERY_GATES_P010.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `MINIGRID_EM.md` `MINIGRID_EM_FIX.md` `MINIGRID_EM_PREREG.md` `MINIGRID_FULL_2X2X2.md` `MINIWORLD_ENDPOINTS.md` `MINIWORLD_FIXED_FINDINGS.md` `MINIWORLD_FIXED_RESULTS.md` `MONOTONE_RESULTS.md` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md` `MQ_RANK_2X2.md` `N5_RESULTS.md` `NOISE_REFINE.md` `NOLEAK_RESULTS.md` `PAIRCONST_RESULTS.md` `PAIRORIGIN_RESULTS.md` `PAIRSPLIT_RESULTS.md` `PAPER2X2_PREREG.md` `PAPERTASK_RESULTS.md` `PAPER_FIG4_REPRO.md` `PAPER_OOD_EXTENDED_n8.md` `PAPER_OOD_RERUN.md` `PAPER_TASK_ABLATION.md` `PAPER_TASK_ACCURACY.md` `PAPER_TASK_FLOORS.md` `POPE_WRAPPING.md` `RANK_SWEEP.md` `RECENCY_EM_RESULTS.md` `RECENCY_RESULTS.md` `REFINE_RESULTS.md` `SEARCH_RESULTS.md` `SELECTIVE_ROPE.md` `SIGN_ABLATION.md` `SIGN_ABLATION_PREREG.md` `SIGN_PROBE.json` `SPREAD2_RESULTS.md` `SPREAD_RESULTS.md` `STITCH_ATTENTION.md` `TEM_BACKGROUND_BASELINES.md` `TEM_T_MULTISEED.md` `THEORY_SEARCH_AND_LENGTH.md` `TIMING_BENCHMARK.md` `UNFREEZE_RESULTS.md` `VISITS_TEST.md` `VOCAB_EM.md`
- `report/INVENTORY_BRIEF.md` (1): `N3_AUDIT.md`
- `report/STORY.md` (35): `ALGORITHMIC_RESULTS.md` `ALIASING_CONTROLLED.md` `ALLOCENTRIC_RECODING.md` `BASELINE_TABLE.md` `EM_WM_STATE.md` `FORGET_CLOCK_PREREG.md` `FORGET_CONTROL.md` `FORGET_GATE.md` `INDEX_BASELINE_PAPER_TASK_n8.md` `KNOB_SWEEP_n8.md` `LEVEL15_MEETS_GATED_matchq.md` `LOCALISATION.md` `LOOP_HEADROOM.md` `MATCH_QUERY_GATES_P010.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `MINIGRID_EM.md` `MONOTONE_PREREG.md` `PAIRSPLIT_RESULTS.md` `PAPER2X2_RESULTS.md` `PAPERTASK_RESULTS.md` `PAPER_TASK_FLOORS.md` `RANK_MI_RESULTS.md` `RANK_PROJ_RESULTS.md` `RANK_SWEEP.md` `RECENCY_EM_RESULTS.md` `RECENCY_GATE_ABLATION.md` `RECENCY_H2.md` `RECENCY_RESULTS.md` `REFINE_RESULTS.md` `SEARCH_RESULTS.md` `SIGN_ABLATION.md` `SIGN_PROBE.md` `THEORY_NARRATIVE.md` `WARM_RESULTS.md`
- `report/VERIFY.md` (134): `ABLATE_FAMILY_TREE.md` `ACCUMULATOR.md` `ACTION_GEOMETRY.md` `ALGORITHMIC_GATES.md` `ALGORITHMIC_RESULTS.md` `ALIASING_CONTROLLED.md` `ALIASING_GATES.md` `ALLOCENTRIC_RECODING.md` `AP_KERNEL_DIAGNOSTIC.md` `AUDIT_2026-09-10.md` `BASELINE_TABLE.md` `COMPOSITIONAL_EXPERIMENT.md` `COMPOSITIONAL_MULTISEED.md` `COMP_HEADROOM.md` `CONTINUOUS_ALLOC.md` `CORRECTION_COMPOSITIONAL.md` `CROSSOVER_CONVERGED.md` `CSCG_TASK_GATES.md` `D5_RESULTS.md` `DISSOCIATION_SWEEP.md` `DOF_RESULTS.md` `DRIFT_PROBE.md` `DXR_RANK_THRESHOLD.md` `EM_WM_THEORY.md` `ENWIK8_HIERARCHY.md` `FAMILY_TREE_D7_RESULTS.md` `FAMILY_TREE_RESULTS.md` `FAMILY_TREE_WM_GAP.md` `FLIPFLOP_RESULTS.md` `FORGET_CONTROL.md` `FORGET_GATE.md` `FREQ_CONTROL.md` `FRONTIER_ALGORITHMIC.md` `GATED_RESULTS.md` `GATED_SEPARATION.md` `GATED_TORUS.md` `GATE_PROBE.md` `H12_BUDGET_CURVE.md` `HABITAT_BUILD.md` `HIERGOAL_ABLATION.md` `HIERGOAL_CLOSEDLOOP.md` `HIER_PARITY.md` `HIER_RECHECK.md` `INDEX_BASELINE_PAPER_TASK.md` `INDEX_BASELINE_PAPER_TASK_n8.json` `INDEX_BASELINE_PAPER_TASK_n8.md` `KNOB_SWEEP.md` `KNOB_SWEEP_n8.md` `L15_ABLATION.md` `L15_LOOP_2X2.md` `LAMBDA_TRACE.md` `LANGUAGE_LANDSCAPE.md` `LM200_ABLATION.md` `LOCALISATION.md` `LONG_SEQ_clean.md` `LOOPED_PILOT.md` `LOOP_DEPTH_STRATA.md` `LOOP_HEADROOM.md` `LOOP_HIER_COMPUTE.md` `LOOP_HIER_PARITY.md` `LOOP_SAMPLED.md` `MAGONLY_RESULTS.md` `MAPPOPE_R4_RESULTS.md` `MATCH_GATES_128_16.md` `MATCH_QUERY_EM.md` `MATCH_QUERY_GATES.md` `MATCH_QUERY_GATES_P010.md` `MATCH_QUERY_LONGQ.md` `MATCH_QUERY_NOISE_ABLATION.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `MINIGRID_ALLOCENTRIC_2X2X2.md` `MINIGRID_EM.md` `MINIGRID_EM_FIX.md` `MINIGRID_FULL_2X2X2.md` `MINIGRID_REPRO_CONTROL.md` `MINIWORLD_ENDPOINTS.md` `MINIWORLD_FIXED_RESULTS.md` `MINIWORLD_GATE_CONTROL.md` `MONOTONE_RESULTS.md` `MQAR_RESULTS.md` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md` `MQ_RANK_2X2.md` `N3_AUDIT.md` `N5_RESULTS.md` `ND_GATES.md` `NOISE_CLEAN_REVALIDATION.md` `NOISE_REFINE.md` `NOLEAK_RESULTS.md` `PAIRCONST_RESULTS.md` `PAIRORIGIN_RESULTS.md` `PAIRSPLIT_RESULTS.md` `PAPERTASK_RESULTS.md` `PAPER_FIG4_EM.md` `PAPER_FIG4_REPRO.md` `PAPER_OOD_EXTENDED_n8.md` `PAPER_OOD_RERUN.md` `PAPER_TASK_ABLATION.md` `PAPER_TASK_ACCURACY.md` `PAPER_TASK_FLOORS.md` `PAPER_VALIDATION.md` `PLANNER_TASK_AUDIT.md` `POPE_WRAPPING.md` `POSITION_EFFECT_CONVERGED.md` `RANK_SWEEP.md` `RANK_TRUNCATION.md` `RECENCY_EM_RESULTS.md` `RECENCY_GATES_K64.md` `RECENCY_GATE_ABLATION.md` `RECENCY_H2.md` `RECENCY_RESULTS.md` `RECIPE_POWER.md` `REFINE_RESULTS.md` `RESULTS_PAPER.md` `REVISIT_DISTANCE.md` `ROPE_CANONICAL.md` `SEARCH_RESULTS.md` `SELECTIVE_ROPE.md` `SIGN_ABLATION.md` `SIGN_ABLATION_PREREG.md` `SIGN_PROBE.md` `SPREAD2_RESULTS.md` `SPREAD_RESULTS.md` `STITCH_ATTENTION.md` `TEM_BACKGROUND_BASELINES.md` `TEM_NOISE_FFN_RESULTS.md` `TEM_T_MULTISEED.md` `THEORY_SEARCH_AND_LENGTH.md` `TIMING_BENCHMARK.md` `UNFREEZE_RESULTS.md` `VISITS_TEST.md` `VOCAB_EM.md` `WARM_RESULTS.md`
- `report/VERIFY_2X2.md` (9): `BASELINE_TABLE.md` `INDEX_BASELINE_PAPER_TASK_n8.md` `PAPER2X2_PREREG.md` `PAPER2X2_RESULTS.md` `PAPER_TASK_FLOORS.md` `REVISIT_DISTANCE.md` `SIGN_ABLATION.md` `_PAPER2X2_RAW.json` `_SIGN_RAW.json`
- `report/inventory/A_reproduction_baselines.md` (73): `ABLATE_COMPOSITIONAL.md` `ABLATE_FAMILY_TREE.md` `AP_KERNEL_DIAGNOSTIC.md` `AUDIT_2026-09-10.md` `AUDIT_HEADLINE.md` `BASELINE_TABLE.md` `CAPACITY_PERREGIME.md` `CORRECTED_LM200_LEADERBOARD.md` `DETAILED_RESULTS.md` `EM_COMP_SAMEBATCH.md` `EM_FIX_COMP.md` `EM_HOPFIELD_CROSSSCALE.md` `EM_P0_COMP.md` `EM_P0_PAPER.md` `EM_WM_STATE.md` `EM_WM_THEORY.md` `FAMILY_TREE_D7_GATES.json` `FAMILY_TREE_D7_GATES.md` `FAMILY_TREE_D7_RESULTS.md` `FAMILY_TREE_GATES.json` `FAMILY_TREE_GATES.md` `FAMILY_TREE_RESULTS.md` `FAMILY_TREE_WM_GAP.md` `GENERALIZATION_REPORT.md` `HOPFIELD_NOMAINAP_RESULTS.md` `INDEX_BASELINE_PAPER_TASK.md` `INDEX_BASELINE_PAPER_TASK_n8.md` `L15_ABLATION.md` `LONG_SEQ_clean.md` `MAPPOPE_R4_RESULTS.md` `MATCH_QUERY_EM.md` `MATCH_QUERY_GATES.md` `MULTICLASS_MULTISEED_RESULTS.md` `N3_AUDIT.md` `NOISE_CLEAN_REVALIDATION.md` `NUMBERLINE_RESULTS.md` `OMEGA_RESCALE_clean.md` `OOD_GRID_RESULTS.md`(del) `PAPERTASK_PREREG.md` `PAPERTASK_RESULTS.md` `PAPER_OOD_EXTENDED.md` `PAPER_OOD_EXTENDED_n8.md` `PAPER_OOD_PROTOCOL.md` `PAPER_OOD_RERUN.md` `PAPER_OOD_WITH_POPE.md` `PAPER_TASK_ABLATION.md` `PAPER_TASK_ACCURACY.md` `PAPER_TASK_FLOORS.md` `PAPER_VALIDATION.md` `PERSCALE_OMEGA_RESULTS.md` `PER_VISIT_clean.md` `RECENCY_EM_RESULTS.md` `REPORT.md` `REPORT_ADDENDUM.md` `REPORT_v2.md` `RESULTS_PAPER.md` `RESULTS_SUMMARY_2026-05-10.md` `REVISIT_DISTANCE.md` `STITCH_ATTENTION.md` `TEM_BACKGROUND_BASELINES.md` `TEM_CROSSSCALE_DIAGNOSTIC.md` `TEM_NOISE_FFN_RESULTS.md` `TEM_RESULTS.md` `TEM_T_MULTISEED.md` `TEM_T_RESULTS.md` `TIMING_BENCHMARK.md` `TOPOLOGY_RESULTS.md` `VOCAB_EM.md` `VOCAB_EM_PREREG.md` `VOCAB_SWEEP_MULTISEED.md` `VOCAB_SWEEP_RESULTS.md` `ZERO_SHOT_TRANSFER_clean.md` `ZERO_SHOT_TRANSFER_clean_brokeninit.md`(del)
- `report/inventory/B_state_correction.md` (75): `BUMP_TOKEN_RESULTS.md` `CAPACITY_CONTROL.md` `CAPACITY_PERREGIME.md` `CASCADE_MULTISEED_RESULTS.md` `CASCADE_REPRO_TEST.md` `CASCADE_ZEROSHOT_S0.md` `CLONE_ANALYSIS_LEVEL15PC.md`(del) `CLONE_TRANSFER_NOBYPASS.md`(del) `CNAV_HEX_Level15.md`(del) `CNAV_HEX_Level15EM.md`(del) `CNAV_HEX_Vanilla.md`(del) `CNAV_HEX_VanillaEM.md`(del) `CNAV_RESULTS.md`(del) `CONTINUOUS_ALLOC.md` `CORRECTED_LM200_LEADERBOARD.md` `CORRECTION_COMPOSITIONAL.md` `DOG_RESULTS.md`(del) `DRIFT_PROBE.md` `EXTRAHEAD_CONTROL.md` `FAMILY_TREE_WM_GAP.md` `GSF_FULL_RESULTS.md` `H12_BUDGET_CURVE.md` `HIPPOCAMPAL_ANALYSIS.md`(del) `HIPPOCAMPAL_GRID.md`(del) `HIPPOCAMPAL_GRIDL15PC.md`(del) `HIPPOCAMPAL_GRID_FREE.md`(del) `HIPPOCAMPAL_HIDDEN.md`(del) `HIPPOCAMPAL_HIDDEN_GRIDFREE.md`(del) `HIPPOCAMPAL_LEVEL15PC.md`(del) `L15_ABLATION.md` `L15_LOOP_2X2.md` `LEVEL15BETA_RESULTS.md` `LEVEL15EM_CROSSSCALE.md` `LEVEL15_MEETS_GATED_matchq.md` `LEVEL15_MEETS_GATED_paper.md` `LEVEL15_MEETS_GATED_paper50.md` `LM200_ABLATION.md` `LM200_CORRECTED_MULTISEED.md` `LONG_SEQ_clean.md` `MATCH_QUERY_GATES_P010.md` `MATCH_QUERY_NOISE_ABLATION.md` `MATCH_QUERY_RESULTS.md` `MQ_NOISE_2X2.json` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.json` `MQ_NOISE_2X2_C2.md` `MULTICLASS_MULTISEED_RESULTS.md` `MULTICLASS_RESULTS.md`(del) `MULTISEED_FOLLOWUP.md`(del) `MULTISEED_FOLLOWUP_RESULTS.md` `N3_AUDIT.md` `NOBYPASS_RESULTS.md` `NODROP_PARETO_RESULTS.md` `NOISE_CLEAN_REVALIDATION.md` `NOISE_REFINE.json` `NOISE_REFINE.md` `NUMBERLINE_RESULTS.md` `OMEGA_RESCALE_clean.md` `PERSCALE_OMEGA_RESULTS.md` `PER_VISIT_clean.md` `PLANNER_TASK_AUDIT.md` `RANK_SWEEP.json` `RECIPE_CHOICE.json` `RECIPE_POWER.md` `REFINE_RESULTS.md` `RESULTS_PAPER.md` `R_T_DISTRIBUTION_3WAY.md`(del) `SESSION_2026-05-01.md` `SESSION_HIERARCHICAL_CASCADE.md` `V3_RESULTS.md` `V4_CONTROL_RESULTS.md` `V4_MULTISEED.md` `V4_RESULTS.md` `VECTOR_NAV_V2_RESULTS.md` `_L15_LOOP_RAW.md`
- `report/inventory/C_hierarchy_loops_tasks.md` (93): `ABLATE_COMPOSITIONAL.md` `AGGREGATE_EXTRAS.md` `AGGREGATE_MULTISEED.md` `AGGREGATE_TASK_RESULTS.md` `ALGORITHMIC_GATES.md` `ALGORITHMIC_RESULTS.md` `ALIASING_COVARIATE.md` `BASELINE_TABLE.md` `BOUNDED_MEMORY.md` `BOUNDED_MEMORY_RESULTS.md` `COMPOSITIONAL_EXPERIMENT.md` `COMPOSITIONAL_MATCH_QUERY.md` `COMPOSITIONAL_MATCH_QUERY_CURRIC.md` `COMPOSITIONAL_MATCH_QUERY_GATES.md` `COMPOSITIONAL_MATCH_QUERY_RESULTS.md` `COMPOSITIONAL_MATCH_QUERY_STAB.md` `COMPOSITIONAL_MULTISEED.md` `COMPOSITIONAL_PLAIN_RESULTS.md` `COMPOSITIONAL_RESULTS.md` `COMP_HEADROOM.json` `COMP_HEADROOM.md` `COMP_HEADROOM_PREREG.md` `CORRECTION_COMPOSITIONAL.md` `CSCG_TASK_GATES.md` `DAGGER_DK6_RESULTS.md` `DAGGER_EMPTY_RESULTS.md` `DAGGER_RESULTS.md` `DISSOCIATION_SWEEP.md` `DOORKEY_BC_RESULTS.md` `EM_COMP_SAMEBATCH.md` `ENWIK8_2X2.md` `ENWIK8_HIER.md` `ENWIK8_HIERARCHY.md` `ENWIK8_LONG.md` `ENWIK8_SEEDS.md` `FRONTIER_ALGORITHMIC.md` `HIERGOAL_ABLATION.md` `HIERGOAL_CLOSEDLOOP.md` `HIERGOAL_LONGT.md` `HIER_ATTN_LONGT.md` `HIER_PARITY.md` `HIER_RECHECK.json` `HIER_RECHECK.md` `HORIZON_L1d128e16.md` `HORIZON_L1d128e50.md` `HORIZON_L2d128e16.md` `HORIZON_L2d256e16.md` `HORIZON_L4d128e16.md` `HORIZON_L4d256e16.md` `HORIZON_RESULTS.md` `HORIZON_TASK_DISTANCES.md` `L15_LOOP_2X2.md` `LAP_GATES.md` `LAP_TRANSFER.md` `LAP_TRANSFER_NOREWARD.md` `LEVEL15_MEETS_GATED_matchq.md` `LOOPED_L1.md` `LOOPED_L4.md` `LOOPED_Loop4.md` `LOOPED_PILOT.md` `LOOP_DEPTH_STRATA.md` `LOOP_HEADROOM.md` `LOOP_HIER_COMPUTE.md` `LOOP_HIER_PARITY.md` `LOOP_SAMPLED.json` `LOOP_SAMPLED.md` `MAP_QUERY_GATES.md` `MAP_QUERY_RESULTS.md` `MATCH_GATES_128_16.md` `MATCH_GATES_64_16.md` `MATCH_GATES_64_4.md` `MATCH_QUERY_EM.md` `MATCH_QUERY_GATES.md` `MATCH_QUERY_LONGQ.md` `MATCH_QUERY_RESULTS.md` `MATCH_QUERY_SCALE.md` `MINIGRID_MEMORY_RESULTS.md` `MQ_NOISE_2X2.md` `MQ_NOISE_2X2_C2.md` `MQ_RANK_2X2.md` `N3_AUDIT.md` `NOISE_REFINE.md` `PAPER_VALIDATION.md` `PLANNER_TASK_AUDIT.md` `RECIPE_POWER.md` `RECURSIVE_RESULTS.md` `REVISIT_DISTANCE.md` `ROOMS_GOAL_RESULTS.md` `ROUTE_ATTN_RESULTS.md` `SPACETIME_HIER_RESULTS.md` `_RECIPE_C0.md` `_RECIPE_C1.md` `_RECIPE_C2.md`
- `report/inventory/D_environments.md` (73): `ALIASING_CONTROLLED.md` `ALIASING_COVARIATE.md` `ALIASING_GATES.md` `ALLOCENTRIC_RECODING.md` `BASELINE_TABLE.md` `CLOCK_SCAN.md` `COMP_HEADROOM.md` `CONTINUOUS_ALLOC.md` `CROSSOVER_CONVERGED.md` `DISSOCIATION_SWEEP.md` `DXR_RANK_THRESHOLD.md` `EM_WM_STATE.md` `FREQ_CONTROL.md` `H12_BUDGET_CURVE.md` `HABITAT_BUILD.md` `INDEX_BASELINE_PAPER_TASK_n8.md` `KNOB_SWEEP.md` `KNOB_SWEEP_n8.md` `MINIGRID_2X2.md` `MINIGRID_2X2X2.md` `MINIGRID_2X2X2_n8.md` `MINIGRID_ALLOCENTRIC_2X2X2.md` `MINIGRID_ALLOCENTRIC_8CELL.md` `MINIGRID_ALLO_8THCELL.md` `MINIGRID_DK16_RESULTS.md` `MINIGRID_DOORKEY_CACHED.md` `MINIGRID_DOORKEY_LONGT.md` `MINIGRID_DOORKEY_RESULTS.md` `MINIGRID_DOORKEY_ROPE_DIAG.md` `MINIGRID_EM.md` `MINIGRID_EM_FIX.md` `MINIGRID_EM_FIX_PREREG.md` `MINIGRID_EM_PREREG.md` `MINIGRID_FULL_2X2X2.md` `MINIGRID_REPRO_CONTROL.md` `MINIWORLD_FIXED_FINDINGS.md` `MINIWORLD_FIXED_RESULTS.md` `MINIWORLD_FIXED_RESULTS_T1024.md` `MINIWORLD_FRESH_ABLATION.md` `MINIWORLD_FRESH_FINDINGS.md` `MINIWORLD_FRESH_GATES_ALLO.md` `MINIWORLD_FRESH_GATES_RAW.md` `MINIWORLD_FRESH_RESULTS.md` `MINIWORLD_FRESH_RESULTS_T1024.md` `MINIWORLD_GATES_ALLO.md` `MINIWORLD_GATES_FIXED.md` `MINIWORLD_GATE_CONTROL.md` `MINIWORLD_GRID16_GATES.md` `MINIWORLD_GRID24_GATES.md` `MINIWORLD_GRID32_GATES.md` `MINIWORLD_GRID_SWEEP.md` `MINIWORLD_GRID_SWEEP_HIER.md` `MINIWORLD_HIER_ABLATION.md` `MINIWORLD_ORACLE_ABLATION.md` `MINIWORLD_ORACLE_GATES.md` `MINIWORLD_ORACLE_RESULTS_T1024.md` `MINIWORLD_ORACLE_RESULTS_T512.md` `MINIWORLD_PROBE3.md` `MINIWORLD_RESULTS.md` `MINIWORLD_SROPE_COMPONENTS.md` `MINIWORLD_TODO.md` `N3_AUDIT.md` `ND_GATES.md` `OMEGA_RESCALE_clean.md` `PERCEPTION_EXPERIMENT_PLAN.md` `PERSCALE_OMEGA_RESULTS.md` `PLANNER_TASK_AUDIT.md` `POSITION_EFFECT_CONVERGED.md` `PUBLICATION_VENUES.md` `ROPE_CANONICAL.md` `ROPE_CONVERGE.md` `VISITS_PER_CELL.json` `VISITS_TEST.md`
- `report/inventory/E_phase_axes.md` (59): `ACCUMULATOR.md` `ACTION_GEOMETRY.md` `ALGORITHMIC_RESULTS.md` `AUDIT_2026-09-10.md` `CONV_KERNEL_PROBE.md` `DXR_PRELIM.md` `DXR_RANK_THRESHOLD.md` `EM_WM_STATE.md` `ENWIK8_LONG.md` `FLIPFLOP_GATES.md` `FLIPFLOP_RESULTS.md` `FORGET_CLOCK_PREREG.md` `FORGET_CONTROL.md` `FORGET_GATE.md` `GATED_PREREG.md` `GATED_RESULTS.md` `GATED_SEPARATION.md` `GATED_TORUS.md` `GATE_PROBE.md` `LAMBDA_TRACE.md` `LANGUAGE_LANDSCAPE.md` `LEARNED_RANK.md` `LOCALISATION.md` `LOCALISATION_PREREG.md` `LOCALISATION_RANK.md` `MAPPOPE_R4.md` `MAPPOPE_R4_PREREG.md` `MAPPOPE_R4_RESULTS.md` `MAPPOPE_VS_POPE.md` `MONOTONE_PREREG.md` `MQAR_PREREG.md` `MQAR_RESULTS.md` `MQ_RANK_2X2.md` `N3_AUDIT.md` `ND_GATES.md` `PAPER_FIG4_EM.md` `PAPER_FIG4_REPRO.md` `PAPER_OOD_WITH_POPE.md` `POPE_WRAPPING.md` `RANK_SWEEP.json` `RANK_SWEEP.md` `RANK_TRUNCATION.md` `RECENCY_EM_RESULTS.md` `RECENCY_GATES.md` `RECENCY_GATES_K16SET.md` `RECENCY_GATES_K4SET.md` `RECENCY_GATES_K64.md` `RECENCY_GATE_ABLATION.md` `RECENCY_H2.md` `RECENCY_PREREG.md` `RECENCY_RESULTS.md` `SELECTIVE_ROPE.md` `SIGN_ABLATION.md` `SIGN_ABLATION_PREREG.md` `SIGN_PROBE.md` `THEORY_NARRATIVE.md` `WARM_RESULTS.md` `_SELECTIVE_TORUS.md` `_SIGN_RAW.md`
- `report/inventory/F_em_wm_kernel.md` (74): `AP_KERNEL_DIAGNOSTIC.md` `AUDIT_2026-09-10.md` `D5_PREREG.md` `D5_RESULTS.md` `DOF_PREREG.md` `DOF_RESULTS.md` `EM_COMP_SAMEBATCH.md` `EM_FIX_COMP.md` `EM_P0_COMP.md` `EM_P0_PAPER.md` `EM_WM_STATE.md` `EM_WM_THEORY.md` `MAGONLY_PREREG.md` `MAGONLY_RESULTS.md` `MATCH_QUERY_EM.md` `MATCH_QUERY_GATES.md` `MINIGRID_EM.md` `MINIGRID_EM_FIX.md` `MINIGRID_EM_FIX_PREREG.md` `MINIGRID_EM_PREREG.md` `N3_AUDIT.md` `N5_PREREG.md` `N5_RESULTS.md` `NOLEAK_PREREG.md` `NOLEAK_RESULTS.md` `PAIRCONST_PREREG.md` `PAIRCONST_RESULTS.md` `PAIRORIGIN_PREREG.md` `PAIRORIGIN_RESULTS.md` `PAIRSPLIT_PREREG.md` `PAIRSPLIT_RESULTS.md` `PAPERTASK_PREREG.md` `PAPERTASK_RESULTS.md` `PAPER_OOD_EXTENDED_n8.json` `PAPER_OOD_RERUN.md` `PAPER_TASK_ACCURACY.md` `PAPER_TASK_FLOORS.md` `RECENCY_EM_RESULTS.md` `RECENCY_GATES_K16SET.md` `RECENCY_GATES_K4SET.md` `RECENCY_GATES_K64.md` `RECENCY_RESULTS.md` `REC_EM_PREREG.md` `SEARCH_PREREG.md` `SEARCH_RESULTS.md` `SIGN_ABLATION.md` `SPREAD2_PREREG.md` `SPREAD2_RESULTS.md` `SPREAD_PREREG.md` `SPREAD_RESULTS.md` `TALE_OF_TWO_ALGORITHMS.md` `THEORY_KERNEL.md` `THEORY_NARRATIVE.md` `THEORY_SEARCH_AND_LENGTH.md` `UNFREEZE_PREREG.md` `UNFREEZE_RESULTS.md` `VOCAB_EM.json` `VOCAB_EM.md` `VOCAB_EM_PREREG.md` `VOCAB_SWEEP_MULTISEED.md` `WARM_PREREG.md` `WARM_RESULTS.md` `_ACHIEVABLE.json` `_ANATOMY.json` `_ANATOMY_EXT.json` `_COLLISIONS.json` `_DOF_TORUS_RAW.md` `_INIT_GRAD.json` `_MAGONLY.json` `_N5_TORUS_RAW.md` `_NOLEAK.json` `_PAIRORIGIN.json` `_PHASE_SPREAD.json` `_REWIND_PROBE.json`

</details>

## 5. Risks

- **Broken guards and analyses (rule 18).** `test_guards.py` reads `_MAGONLY.json`, `_D5_N24.json`, `_REWIND_PROBE.json` from the root; `analyze_factorial_n8.py`, `analyze_generality.py`, `analyze_srope_torus.py`, `analyze_rank_*`, `analyze_score_rank.py`, `analyze_exptorus.py` and the figure scripts read per-arm `*_STRATA.json` / `*_OWN_s*.json` from the root or an `--eval-dir`. Re-run `python3 -m mapformer.test_guards` and one analysis per family (byte-identical output) after the move, before the commit lands (rule 19 style check).
- **Drivers recreate the root.** 363 code files write `$REPO/NAME`. Nothing is running now (project_state 2026-10-28), but any re-run or new batch built from an old driver will drop files at the root again. Change `lib_driver.sh` / driver templates to take an output dir, or accept that historical drivers are frozen and say so in a README in `results/`.
- **CLAUDE.md, memory and drafts cite by bare name.** The Citable table names 92 files, the drafts 83, the claim map 47, memory 113. Every rewrite is a chance to break a citation; rule 28 (end-to-end read) applies to CLAUDE.md after the edit. Alternative (b) in 4.1 avoids this by leaving CITED files at the root.
- **The shared report.** `report/language_summary.html` names result files; if paths in it change it must be republished WITH `url=` (CLAUDE.md "Where things live") or the user's link breaks.
- **Classification is automated.** References were found by token match on file names. A name mentioned only in prose without the extension and not in backticks may be missed (counted as a bare-stem reference at best); a stem that is also an English or variable word (`REPORT`, `RANK3`) may be over-counted. SUPPORTING vs ORPHAN in particular depends on this. The 36 ORPHAN and 41 archive-bound GENERATED files should be spot-read before archiving (rule 9: verify what a probe measures).
- **Withdrawn records are evidence.** `N3_AUDIT`, `PLANNER_TASK_AUDIT`, `HIERGOAL_ABLATION`, `CORRECTED_LM200_LEADERBOARD`, `AUDIT_2026-09-10` (kept live: named in "Where things live") are the records that justify withdrawals; archive, never delete, and keep CLAUDE.md "Withdrawn" pointing at them.
- **Partially invalid is not void.** The 15 lm200-bannered files keep valid clean/noise rows (report/INVENTORY_BRIEF.md); archive them, do not delete.
- **History.** Use `git mv` (one commit per topic) so `git log --follow` keeps history; do not mix moves with edits in the same commit, or rename detection may fail on the small per-arm files (many are 4-6 lines and near-identical).
- **Registered records must stay whole.** Splitting a family across folders (e.g. moving a cited RESULTS but leaving its ANALYSIS.txt and VERDICTS.json) breaks rule 7 ("commit the script behind every number") in practice. Section 2 always moves a family as a unit.
- **The catalogue is already stale.** RESULTS_INDEX catalogue was regenerated 2026-10-09 (520 files); 153 later files appear only in hand rows or not at all. Regenerate it after the move with the updated `catalog_results_index.py`, then rule 18: grep the code for any hard-coded catalogue counts.
- **Destructive steps (rule 26).** Only 20 deletes are proposed, each byte-identical to a file already in `archive_stale/`; re-verify the md5 immediately before deleting. Everything else is a move or an archive.

Method note: the scan covered CLAUDE.md (split into its Citable / Withdrawn / other regions), RESULTS_INDEX.md (hand rows vs catalogue), the three current drafts and the claim map, WHERE_THINGS_STAND, `.claude-memory/*.md`, every top-level `.md`/`.txt`, `docs/**` (excluding the frozen package copy), `report/**`, top-level `.tex`/`.html`, all `.py`/`.sh` outside `runs/`, `data/`, `papers/`, and `archive*`. Dates are the last commit touching each file (`git log --name-only`).
