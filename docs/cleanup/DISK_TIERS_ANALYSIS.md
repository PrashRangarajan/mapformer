# Disk tiers D2a / D2b / D3: what points to them, what is lost, verdicts (read-only, 2026-10-29)

Analysis only. Nothing was deleted, moved, edited or committed; this file is uncommitted. The one state change was
`git fetch origin` (remote-tracking refs only). Sizes are GiB of checkpoint bytes measured today (`find -printf %s`),
matching `INVENTORY_RUNS.md`. "Tree" = the current working tree outside `runs/`, `.git`, `data/` and PDFs (CLAUDE.md,
`.claude-memory/`, README, `docs/**` incl. drafts, audits, expansion scripts, LOG.md, WHERE_THINGS_STAND, all
top-level `*.md` / `*.py` / `*.sh`, `hyak/`, `paper_figures/`, `report/` incl. `report/language_summary.html`,
`papers/`, `archive/`), plus text files (not JSON/logs) under `runs/`. "History" = every commit on every ref.

## 1. Summary

| verdict | GiB | what |
|---|---:|---|
| **SAFE** | **11.34** | D2a 4.79 (Indirect T3, the Bach side runs, the algorithmic family, `sameblock`, `dyck_t3`); D2b 6.54 (`indirect`, `indirect_200k`, `code/*.best.pt`, `jsb`, `jsb_aug`, `tinystories/p0/*.final.pt`) |
| **SAFE AFTER EXTRACTION** | **6.42** | D2a 3.55 (MiniWorld, Flip-Flop, the 15 no-JSON nav dirs minus `em_fig4`, other nav-trainer dirs); D2b 0.80 (map-extent: `alias_sweep`, `alias_follow`, `rope_converge`, `g8_converge`, `visits_test`); D3 2.08 (TinyStories raw train text) |
| **KEEP** | **1.43** | `code_ablate/*.final.pt` 1.28 (only source for the CLAUDE.md-cited 80.7% delta census, no committed script); `selective` 0.11 (CLAUDE.md rule 9's probe and a cited live negative); `em_fig4` 0.01 (GAIN_GRAIN's registered power table reads its `losses`); TinyStories valid text 0.02 |
| **ASK** | **15.11** | Memory Maze zips 12.55 (sole public source is quota-limited Drive, no mirror, cluster copy on purgeable scratch, no md5 recorded); `code/*.final.pt` 1.28 and `code_decay/*.final.pt` 1.28 (cited full-val numbers are stored, but regeneration is ~57 GPU-h of non-bitwise retraining) |

Findings that decide most rows:

1. **Git history adds no consumer that the current tree does not show.** Every historical reference is still live, or
   sits in a file that moved or was superseded: CLAUDE.md before `670ad10` (2026-09-24; text moved verbatim to
   `docs/LOG.md`, still live); `docs/prepared/dyck_mdepth/pkg/*` (a package copy, deleted in C1 `49db9ac`); drivers
   moved to `archive/code/2026-10/` in B1 `7c7e1ac` (still in the tree); `DYCK_T3_RESULTS.md` and `MINIWORLD_TODO.md`
   (archived in A2 / A3); `runs/` paths later removed from `CODE_RESULTS.md`, `INDIRECT_RESULTS.md`,
   `project_state.md` and `run_code_controls.sh`; cluster / `mm_data` paths redacted from the Memory Maze documents on
   2026-10-28. No version of `report/language_summary.html` (the shared artifact's source) names any of these dirs.
2. **No number cited in CLAUDE.md's citable table, draft 2, draft 6, the claim map or the shared report needs a
   checkpoint re-analysis from D2a / D2b.** Each is stored in tracked JSON or MD (sections 3-4). There are three
   exceptions where a cited number or a registered computation reads values that exist only inside `.pt` files:
   CLAUDE.md rule 2's "r = -0.996 over 57 runs" (MiniWorld training losses, `mw_grid_sweep` / `miniworld_*`); the
   map-extent row's convergence and loss-gap analysis (`agg_alias.py`, `agg_visits.py` read `losses` from the `.pt`);
   GAIN_GRAIN's pre-registered power table (`docs/audits/2026-10-05/gain_grain_power.py` reads `em_fig4` losses). D3
   adds a fourth: CLAUDE.md's "80.7% of `pope_delta` frozen in its clamp" (`code_ablate`, no committed script).
3. **No guard or selftest reads any of these dirs.** `test_guards.py` uses `dof`, `warm`, `unfreeze`, `recency_em` and
   `magonly_repro`. Its and `ckpt_guard.py`'s `"noise_refine"` string is a layout name, not the dir. None of the 16
   `--selftest` analyses names a D2 or D3 dir.
4. **Per-seed evaluations are tracked for almost every nav dir**, including the 15 with no JSON in the run dir:
   `L15_ABLATION.json`, `NOISE_REFINE.json`, `LOOP_SAMPLED.json`, `_L15_LOOP_RAW.json`, `_RECIPE_C{0,1,2}.json`,
   `FORGET_GATE.json`, `FORGET_CONTROL.json`, `VOCAB_EM.json`, `MINIGRID_EM.json`, `MINIGRID_EM_FIX.json`,
   `STALE_MINIGRID_ALLOCENTRIC_2X2X2.json` + `MINIGRID_ALLOCENTRIC_8CELL.json`, `MAPPOPE_R4.json`,
   `PAPER_OOD_RERUN.json`, `_N5_TORUS_RAW.json`, `_SELECTIVE_TORUS.json`, `runs/popewrap/g*/acc.json`,
   `runs/lamtrace/lam_s*.txt`, `docs/audits/2026-10-05/gain_grain_emfig4_T128.json`. What exists only in the
   checkpoint is the **per-epoch training loss** (`losses`) and the run config. The gitignored per-run `.log` files
   carry the loss only every 5 epochs.
5. **No D2 or D3 dir has a `code_md5*.txt`** (the md5 guard came later). Regeneration is "driver + commit + seeds"
   (rule 30: check out the batch's commit), not bitwise-checkable, and the torch version has changed since (2.10 now).

## 2. Git state (GitHub = local, checked)

- After `git fetch origin` (02:11): `main` = `origin/main` = `d151475`. So the history on GitHub equals the local history.
- Remote branches: `origin/main`, `origin/rankdecay-wip` (`b2dba71`, one unmerged commit, the shelved RANKDECAY
  optimizer change; it references none of these dirs). `koopman/main` (`80fc2f4`) is an ancestor of `main`.
- Local-only refs: `backup-coauthor` and `refs/original/refs/heads/main` (both `62075dc`, the pre-rewrite history of
  2026-09-16; their hits duplicate main's), and `stash@{0}` (2026-06-03, before any of these runs; no hits). No tags,
  no git notes.
- Method: one pass of `git log --all -p --no-renames -- . ':(exclude)runs'` (1,208 commits) with the regex
  `runs/<dir>(?![A-Za-z0-9_])` on added and removed lines; bare `mm_data` and `TinyStoriesV2` for the D3 data.
  Cross-checked against `git log --all -G` for `runs/indirect_t3` and `-S` for `runs/visits_test`: identical commit sets.
  Plus `git log --all --name-only -- runs` for commits that touched the dir's tracked files ("content commits").
  The tips of `rankdecay-wip`, `koopman/main`, `backup-coauthor` and the stash were also grepped directly.

## 3. Per-group tables

Columns: **refs now** = files in the tree naming `runs/<dir>` (excluding `docs/cleanup/`); **history** = commits whose
diff adds or removes `runs/<dir>` (first .. last outside the cleanup docs; "+C" = tracked-content commits). Every
history reference is live or covered by finding 1 unless a row says otherwise.

### 3a. D2a: checkpoints of uncited runs (8.47 GiB)

| path(s) | GiB | refs now | history | cited numbers depending | re-analyses lost | extraction first | regenerable (cost) | verdict |
|---|---:|---|---|---|---|---|---|---|
| `runs/indirect_t3` | 1.76 | 2: `run_t3.sh`, `docs/LOG.md` | 3 (`cd1f04c` 09-18 .. `670ad10` 09-24), +1C | none (T3 per-token phase; `T3_RESULTS.md`) | none documented | none (JSON has `loss_curve`, `test_acc`) | `run_t3.sh`, JSON-gated; 16 runs, 32.5 GPU-h | **SAFE** |
| `runs/jsb_len512`, `runs/jsb_centered` | 0.89 | 8 + 3: `analyze_t1.py`, `probe_theory_numbers.py`, `analyze_jsb_base.py`, `run_jsb_len.sh`, `run_jsb_rank.sh`, `run_t1_centered.sh`, `T1_PREREG.md`, LOG | 11 + 6 (`7dc7b2a` 09-17 .. `665bcb7` 09-25) | none cited. Claim map C8's post hoc "Bach alpha ~1.00" is stored in `T1_RESULTS.md` | `analyze_t1.py` (T1, centred accumulator; the THEORY_MAPPOPE account is withdrawn), `probe_theory_numbers.py` (output `THEORY_NUMBERS.md`, tracked) | none | JSON-gated drivers; 50 runs, ~1.5 GPU-h | **SAFE** |
| `runs/jsb_t3`, `runs/jsb_forced` | 0.42 | 3 + 4: `probe_theory_numbers.py`, `run_t3.sh`, `archive/code/2026-10/run_t3gen.sh`, LEDGER, LOG | 6 + 8 (`cd1f04c` 09-18 .. `665bcb7`) | none (T3 / T3GEN cross-task claims withdrawn) | `probe_theory_numbers.py` (stored output) | none | JSON-gated; 20 runs, 0.6 GPU-h | **SAFE** |
| `runs/jsb_cross`, `runs/jsb_decay` | 0.36 | 2 + 2: `run_cross_jsb.sh`, `run_decay.sh`, LOG | 3 + 3 (`7da11e8` 09-19 .. `670ad10`) | the shared report's "48-parameter fix repairs Bach" (`DECAY_RESULTS.md`) is in JSON `test_buckets`. The CLAUDE.md CROSS row is Dyck-only | `CROSS_RESULTS.md` Amendment 1's Bach "r = 1.000" (learned-state vs token distance) was computed by a script that was never committed; only these weights could re-derive it | none | JSON-gated; 20 runs, 0.8 GPU-h | **SAFE** (the r = 1.000 stays as stored text) |
| `runs/jsb_base{512,8192,32768}` | 0.54 | 1 each: `analyze_jsb_base.py` (reads JSON) | 3 each (`7dc7b2a` .. `665bcb7`) | none (OOD omega-base sweep) | none | none | `run_jsb_len.sh`, JSON-gated; 30 runs, 0.9 GPU-h | **SAFE** |
| `runs/mw_grid_sweep` | 0.61 | 8: `run_mw_grid_sweep.sh`, `run_mw_gatectl.sh`, `run_mw_probe3{,_seeds}.sh`, `run_mw_srope_components.sh`, `archive/code/2026-10/run_mw_nope.sh`, `experiment_audit.py` (its documented example), `docs/audits/2026-09-24/applied/experiment_audit_check.txt` | 9 (`db04d91` 08-26 .. `665bcb7`), +1C | **CLAUDE.md rule 2: r = -0.996 over 57 runs** (losses from `.pt`). Rule 1's twin floor 0.150 (GateDeltaCtl vs Vanilla, n=9) uses accuracies, which are in JSON | `experiment_audit.py` on MiniWorld layout and `run_mw_gatectl.sh`'s inline capacity table read `losses` from the `.pt` | **`losses`** (+ variant, flags, sizes) | 6 drivers, all **.pt-gated** (would retrain); ~30 h wall, plus MiniWorld EGL buffer prebuild | **SAFE AFTER EXTRACTION** |
| `runs/{mw_grid_sweep_hier, mw_hier_ablation, miniworld_fixed, miniworld_fresh, miniworld_oracle, miniworld}` | 1.33 | 1-3 each: their drivers, `agg_miniworld.py`, `eval_ablation_miniworld.py`, `run_mw_missing.sh`, `archive/orphan/MINIWORLD_TODO.md` | 1-3 each (08-24 .. 08-26), +1C | rule 2's 57-run correlation draws on this MiniWorld line (exact run set not recorded) | `agg_miniworld.py` (final loss from `.pt`, convergence filter), `eval_ablation_miniworld.py` (context destruction; output `MINIWORLD_ORACLE_ABLATION.md` tracked) | **`losses`** | .pt-gated drivers; ~1-6 h wall each | **SAFE AFTER EXTRACTION** |
| `runs/flipflop` | 0.57 | 1: `run_flipflop.sh` | 1 (`7bce165` 09-07), +1C | none (live negative "does not test our axis") | none | **`losses`** | `.pt`-gated; 48 runs, ~1 h wall | **SAFE AFTER EXTRACTION** |
| `runs/{frontier_alg, algorithmic, hier_parity, loop_hier, rope_canon}` | 0.64 | 1-4 each: drivers, `agg_frontier.py`, `decide_frontier.py`, hygiene patch 13 | 1-4 each (09-03 .. `665bcb7`), +1C | none (hierarchy unpowered) | none (the aggregators read JSON) | none (JSON has `acc`, `final_loss`; the `.pt` holds no losses) | JSON-gated, except `run_frontier.sh`'s Match-Query stage (.pt-gated); <3 h wall total | **SAFE** |
| `runs/selective` | 0.11 | 6: `probe_rank_learned.py`, `probe_rank_truncate.py`, `probe_gate.py`, `probe_conv_kernel.py`, `run_selective.sh`, patch 12 | 8 (`fac85a2` 09-03 .. `665bcb7`), +1C | `SELECTIVE_ROPE.md` (cited live negative) is in `_SELECTIVE_TORUS.json` + parity JSON; **CLAUDE.md rule 9's lesson** ("100% in top-2 SVs") is `LEARNED_RANK.md`, from `selective/torus/p0` | the four probes (outputs `LEARNED_RANK.md`, `CONV_KERNEL_PROBE.md`, both kept as records in A3) | torus `losses` if deleted | torus part .pt-gated; ~2.5 h wall | **KEEP** (0.11 GiB backs a rule's probe; cheap) |
| `runs/sameblock` | 0.19 | 3: `analyze_sameblock.py`, `run_sameblock.sh`, `archive/code/2026-10/run_sameblock_amend.sh` | 9 (`5bd5d9a` 09-15 .. `665bcb7`; 3 pre-rewrite), +1C | none | `analyze_sameblock.py` role cosines (its registered analysis loads checkpoints; output in `SAMEBLOCK_RESULTS.md`) | none (JSON has `curve`, `eval`) | JSON-gated; ~2 h wall | **SAFE** (the registered analysis can no longer re-run its cosine part) |
| `runs/dyck_t3` | 0.01 | 4: `run_t3gen2.sh`, archived `run_t3gen.sh`, LOG, HISTORY_PLAN | 7 (`1c443b6` 09-19 .. `670ad10`) | none | none | none | JSON-gated | **SAFE** |
| 15 no-JSON dirs less `em_fig4`: `vocab_em` (+ `vocab_em_link`), `minigrid_em`, `minigrid_em_fix`, `minigrid_2x2x2_allo`, `l15_loop_2x2`, `l15_ablation`, `noise_refine`, `loop_sampled`, `recipe_power`, `forget`, `forget_ctrl`, `mappope_r4`, `paper_task_rerun`, `lamtrace` | 0.61 | 1-7 each: their drivers; `ablate`/`eval_*` scripts; `report/report.tex` + `report/FIXES.md` (minigrid_em; stale report); `THEORY_KERNEL.md` (minigrid_em_fix, a value in a table); `PAPERTASK_PREREG.md`, `probe_collisions.py`, `analyze_papertask.py` (paper_task_rerun) | 1-9 each (08-24 .. 09-24) | cited live negatives in CLAUDE.md: L15_ABLATION, L15_LOOP_2X2, NOISE_REFINE, FORGET_CONTROL ("+0.086"); "MapPoPE r=4 +0.019 OOD". All are in tracked per-seed JSON (finding 4) | `eval_noise_refine.py` / `eval_loop_sweep.py` re-evals, `probe_forget.py` / LAMBDA_TRACE (the forget gate's "mechanism unidentified" could no longer be probed), `probe_collisions.py` (output `_COLLISIONS.json` tracked) | **`losses` + `config`** (the only full-resolution training curve and the recorded config) | all .pt-gated; each ≤ 6 h wall | **SAFE AFTER EXTRACTION**. Also remove the 120 untracked symlinks in `vocab_em_link/` or they dangle |
| `runs/em_fig4` | 0.01 | 6: `run_em_fig4.sh`, `PAPER_FIG4_EM.md`, `docs/audits/2026-10-05/gain_grain_power.py` + `_emfig4_T128.md`, `report/inventory/E_phase_axes.md`, patch 13 | 6 (`6038ade` 09-04 .. `d332718` 10-05) | GAIN_GRAIN (cited row) pre-registered its power table from em_fig4's MapEM: `gain_grain_power.py` classifies SOLVED from `torch.load(...)["losses"]` | the registered power computation; `probe_paper_fig4.py` (C1 / C2 stats, aggregates only in `PAPER_FIG4_EM.md`) | `losses` would do, but then patch a registered script | .pt-gated; 0.3 h wall | **KEEP** (0.01 GiB) |
| `runs/{n5_phase, counter_batch, recency_budget, cmq_curric, cmq_stab, cmq_sweep, mq_noise, mq_noise_c2, popewrap, mappope_recency}` | 0.43 | 1-4 each: drivers, `agg_cmq.py`, `ablate_match_noise.py`, `COMPOSITIONAL_MATCH_QUERY.md`, `N5_PREREG.md`, `report/inventory/B_state_correction.md` | 1-6 each (08-23 .. `665bcb7`) | MQ_NOISE_2X2(*) cited live negatives: per-seed JSON tracked | `ablate_match_noise.py` (context destruction on the noisy Match-Query; never cited) | **`losses`** (results are in JSON) | JSON-gated: `counter_batch`, `recency_budget`, `mappope_recency`; the rest .pt-gated; ≤ 6 h wall each | **SAFE AFTER EXTRACTION** |

### 3b. D2b: checkpoints of cited runs whose evaluations are stored (7.34 GiB)

| path(s) | GiB | refs now | history | cited numbers depending | re-analyses lost | extraction first | regenerable (cost) | verdict |
|---|---:|---|---|---|---|---|---|---|
| `runs/indirect/**/*.pt` | 3.02 | 6: `run_indirect.sh`, `run_indirect_more.sh`, `run_jsb.sh`, `docs/audits/2026-10-05/pope_task_step_table.py`, LOG, HISTORY_PLAN | 13 (`e043d3d` 09-15 .. `a129fcf` 10-05; 4 pre-rewrite), +1C | CLAUDE.md "1/8 at their 100k"; the shared report's 1/8 and "earlier on 6 of 7 paired seeds, median 70k vs 100k" (`analyze_indirect.py` reads JSON `loss_curve`). All in tracked JSON | `pope_task_step_table.py` (output `pope_task_step_table_out.txt` tracked; post hoc, uncited) | none | JSON-gated drivers; 32 runs, 46.1 GPU-h | **SAFE** |
| `runs/indirect_200k/**/*.pt` | 1.51 | 4: `run_indirect_200k.sh`, `eval_indirect_ood.py`, step table, LOG | 9 (`b7cb3a0` 09-16 .. `a129fcf`) | CLAUDE.md / shared report "7/8 at 200k": JSON | `eval_indirect_ood.py` (`INDIRECT_OOD.md` / `.json` tracked, uncited), step table | none | JSON-gated; 16 runs, 48.6 GPU-h | **SAFE** |
| `runs/code/*.best.pt` | 1.28 | `probe_code_depth0.py`, `probe_code_accum.py`, `eval_code_delims.py` (default `--which best`), `docs/audits/2026-09-24/applied/verify_stale_strict.py` | (code: 16 commits, `b9b34c7` 09-21 .. `1add8f4` 09-27) | none: the full-val numbers read `.final.pt`; `best_val_bpc` is in each run JSON | the two probes behind `CODE_RESULTS.md` (withdrawn OOD); `eval_code_delims` (`CODE_GATES.json` tracked); `verify_stale_strict.py`'s strict-load check (prints MISSING; stored output `verify_stale_strict.txt`) | none | JSON-gated `run_code.sh`; retraining the 12 runs also rebuilds `.final.pt` (26.5 GPU-h) | **SAFE** |
| `runs/jsb/**/*.pt` | 0.36 | 7: `run_jsb.sh`, `run_indirect_more.sh`, `probe_theory_numbers.py`, step table, `JSB_LENGTH_RESULTS.md`, LOG, HISTORY_PLAN | 13 (`e043d3d` .. `a129fcf`) | CLAUDE.md and shared report "Bach -0.032 (5/5)"; MapPoPE - MapWM -0.0165: JSON | `probe_theory_numbers.py` (`THEORY_NUMBERS.md` tracked), step table | none | JSON-gated; 20 runs, 5.1 GPU-h | **SAFE** |
| `runs/{alias_sweep, alias_follow, rope_converge, g8_converge, visits_test}/**/*.pt` | 0.80 | 3 / 6 / 5 / 1 / 1: `run_alias_{sweep,finish,followup,waveb}.sh`, `run_rope_converge.sh`, `run_vanilla_converge.sh`, `run_g8_converge.sh`, `run_visits_test.sh`, `agg_alias.py`, `MINIWORLD_ENDPOINTS.md`, `report/STORY.md`, `report/inventory/D_environments.md`, LEDGER | 3 / 7 / 6 / 1 / 1 (`7a947c4` 08-27 .. `665bcb7`) | CLAUDE.md map-extent row (-0.010 / +0.015 / +0.305): accuracies and per-seed contrasts are in JSON (`MINIWORLD_ENDPOINTS.md`). The convergence flags, loss columns and r(loss, acc) of `ALIASING_CONTROLLED.md` / `VISITS_TEST.md` came from `.pt` losses | `agg_alias.py` (silently drops any run whose `.pt` is missing: `load()` returns None, so a re-run prints INCOMPLETE), `agg_visits.py` (crashes), re-evaluation at new lengths | **`losses`**, then point `agg_alias` / `agg_visits` at the sidecars (verify identical output first) or freeze them | 8 drivers, all **.pt-gated**; ~50 h wall in total (span 16 / 8.8 / 15.6 / 3.6 / 6.5 h) | **SAFE AFTER EXTRACTION** |
| `runs/jsb_aug/**/*.pt` | 0.18 | 2: `run_aug.sh`, LOG | 4 (`b41a9a4` 09-20 .. `670ad10`) | CLAUDE.md "transposition 0.107 NLL"; shared report "0.394 vs their 0.489": JSON | none documented | none | JSON-gated; 10 runs, 2.9 GPU-h | **SAFE** |
| `runs/tinystories/p0/*.final.pt` | 0.20 | (dir: 9, all reading `*.best.pt`: `usage_tick.py`, `tw_alpha_drift.py` (draft 2 App. J4), `alpha_language.py`, `tinystories_probe.py`) | 7 (`f840699` 10-09 .. `f1858ce` 10-20) | draft 2 / claim map C8 "1.5053 vs 1.5050": in each JSON (`final_val_200`, `best_val`) | none (no `.final.pt` reader) | none | `run_tinystories.sh` JSON-gated; keep `.best.pt` | **SAFE** |

### 3c. D3: code `.final.pt` and external data (18.49 GiB)

| path(s) | GiB | refs now | history | cited numbers depending | re-analyses lost | extraction first | regenerable (cost) | verdict |
|---|---:|---|---|---|---|---|---|---|
| `runs/code/*.final.pt` | 1.28 | 20 (`run_code_fullval.sh`, `eval_code_long.py`, `analyze_code_fullval.py`, CODE_FULLVAL / ABLATE pre-registrations and results, LEDGER, PROPAGATION, LOG, ...) | 16 (09-21 .. 09-27), +2C | CLAUDE.md MapPoPE - MapWM code -0.0054 and MapPoPE - PoPE +0.0034 (full val); ABLATE's RoPE row (+3.5885): stored in tracked `runs/code/FULLVAL_512.json` and `CODE_ABLATE_*.json` | the full-val rescore and any matched-length probe of the 512-trained models | none for the numbers; record md5 | retrain 12 runs, 26.5 GPU-h, not bitwise | **ASK** |
| `runs/code_decay/*.final.pt` | 1.28 | 12 (`run_code_decay.sh`, `run_code_fullval.sh`, `run_code_controls.sh`, CODE_DECAY / CODE_FULLVAL docs, audits) | 12 (`54f8574` 09-21 .. `1add8f4` 09-27), +2C | CODE_FULLVAL C2 (envelope): `runs/code_decay/FULLVAL_512.json` (tracked) | full-val rescore | md5 | 12 runs, 30.0 GPU-h | **ASK** |
| `runs/code_ablate/*.final.pt` | 1.28 | 7 (`run_ablate.sh`, `ABLATE_RESULTS.md`, `ckpt_guard.py` docstring, `verify_stale_strict.py`, hygiene REPORT and patch 09, LOG) | 12 (`076f74f` 09-22 .. `665bcb7`) | **CLAUDE.md "80.7% of `pope_delta` frozen in its clamp"**, plus ABLATE's "50.5% channels dead" and logit sd 2.50 / 3.34. A1-A3 are stored (`CODE_ABLATE_INDIST.json`, `CODE_ABLATE_LONG_FINAL.json`); the census scripts were **never committed** (no `*.py` computes them), so these weights are the only way to re-derive a CLAUDE-cited number | A1-A3 rescore; the delta / ReLU census | -- | 12 runs, 26.9 GPU-h | **KEEP** (or commit a census script and its output first, then reconsider) |
| `/home/prashr/mm_data/zips/15x15/{eval,train-part0}.zip` | 12.55 | 12 live files name `mm_data` (`MEMORY_MAZE_PREREG/RESULTS.md`, `train_mm.py`, `mm_encoder.py`, `test_mm.py`, `run_mm_pilot{,3}.sh`, `hyak/mm_sync.sh`, audits 10-27 / 10-28); zips themselves only via `mm_encoder.py prep --zips`, `mm_gate*.py`, `mm_floors.py`, `test_mm.py`, `hyak/mm_download.sh` | `mm_data`: 5 commits (`d5c0ec4` 10-27 .. `fd256da` 10-28; paths later redacted) | Memory Maze (draft 2 K5, claim map C6, registered): no number needs the zips. Every analysis reads the 64 committed `results.json` or `mm_data/arrays/` (2.15 GiB, **keep**: md5 `MANIFEST.json`, the exact batch input). Gate / floor / encoder outputs are tracked under `docs/audits/2026-10-26..28/` | rebuilding the arrays and encoder from frames, re-running the gate (F1-F3) and floors, `test_mm.py` (needs a small zip) | **md5 + byte size of both zips** (none recorded anywhere; MANIFEST covers derived arrays only) | re-download ~13.5 GB from the authors' public Drive folder (ids in `hyak/mm_download.sh`). Drive is the **only** source (no mirror, generation code not public), with a per-file quota: every request returned "Quota exceeded" through the night of 2026-10-26 (`MEMORY_MAZE_GATE.md`). The cluster copy sits on scratch that is purged after ~30 idle days (it was downloaded 2026-10-26); not verified from kalman | **ASK** (lean KEEP until a second, non-purgeable copy exists) |
| `data/tinystories/TinyStoriesV2-GPT4-train.txt` (+ `-valid.txt` 0.02) | 2.08 (+0.02) | 1: `tinystories_data.py` (the builder) | `TinyStoriesV2`: 1 commit (`0b59d4a` 10-09) | none: every TinyStories analysis reads `train.bin` / `val.bin` / `vocab.json` (kept) | rebuilding the `.bin` files (~10-20 min CPU) | **md5 of both files**, plus a note of the source (`huggingface.co/datasets/roneneldan/TinyStories`, V2 GPT-4 split, downloaded 2026-10-09, `TINYSTORIES_PILOT.md`; HF revision not recorded) | public HF download (2.2 GB, no quota) | train: **SAFE AFTER EXTRACTION**; valid: KEEP (21 MB, the readable text of the val stream) |

## 4. Drivers that would silently retrain or overwrite after deletion

Rule 21's trap: most drivers skip a run only when its `.pt` exists.

- **.pt-gated (would retrain the deleted runs and overwrite their JSON and logs):** `run_mw_grid_sweep.sh`,
  `run_mw_grid_sweep_hier.sh`, `run_mw_hier_ablation.sh`, `run_mw_gatectl.sh`, `run_mw_probe3.sh`,
  `run_mw_probe3_seeds.sh`, `run_mw_srope_components.sh`, `run_miniworld.sh`, `run_mw_missing.sh`,
  `run_mw_fresh_factorial.sh`, `run_mw_oracle.sh`, `run_alias_sweep.sh`, `run_alias_finish.sh`,
  `run_alias_followup.sh`, `run_alias_waveb.sh`, `run_rope_converge.sh`, `run_vanilla_converge.sh`,
  `run_g8_converge.sh`, `run_visits_test.sh`, `run_flipflop.sh`, `run_frontier.sh` (Match-Query stage),
  `run_selective.sh` (torus stage), `run_vocab_em.sh`, `run_minigrid_em.sh`, `run_minigrid_em_fix.sh`,
  `run_pope_hier_allo.sh`, `run_n5_phase.sh`, `run_cmq_curric.sh`, `run_cmq_curric_full.sh`, `run_cmq_stab.sh`,
  `run_cmq_sweep.sh`, `run_recipe_power.sh`, `run_papertask_rerun.sh`, `run_em_fig4.sh`, `run_mq_noise_2x2.sh`,
  `run_mq_noise_c2.sh`, `run_l15_loop_2x2.sh`, `run_l15_loop_supp.sh`, `run_l15_ablation.sh`, `run_noise_refine.sh`,
  `run_loop_sampled.sh`, `run_forget.sh`, `run_forget_control.sh`, `run_lambda_trace.sh`, `run_popewrap.sh`,
  `run_mappope_r4.sh` (r4 stage), `run_t3gen2.sh` (its torus_t3 stage, already D1). Archived ones with the same gate:
  `archive/code/2026-10/run_mw_nope.sh`, `run_minigrid_allo.sh`, `run_cmq_topup.sh`, `run_cmq_{curric,stab}_topup.sh`.
- **JSON-gated (safe to relaunch, they skip):** `run_t3.sh`, `run_jsb_len.sh`, `run_jsb_rank.sh`, `run_t1_centered.sh`,
  `run_cross_jsb.sh`, `run_decay.sh`, `run_aug.sh`, `run_jsb.sh`, `run_indirect.sh`, `run_indirect_more.sh`,
  `run_indirect_200k.sh`, `run_code.sh`, `run_code_decay.sh`, `run_ablate.sh`, `run_tinystories.sh`,
  `run_algorithmic.sh`, `run_hier_parity.sh`, `run_loop_hier.sh`, `run_rope_canonical.sh`, `run_sameblock.sh`,
  `run_counter_batch.sh`, `run_t3gen2.sh` (dyck_t3 stage), archived `run_recency_budget.sh`, `run_sameblock_amend.sh`.
- **Eval-only drivers that would overwrite tracked outputs:** `run_code_fullval.sh`: with no `*.final.pt`,
  `eval_code_long.py` writes `{}` over the tracked `runs/{code,code_decay}/FULLVAL_512.json`, and then the analysis
  fails. `git checkout` restores them. `run_papertask_eval.sh`: `eval_paper_ood.py` skips missing checkpoints and
  rewrites `PAPER_OOD_RERUN.json`.
- Aggregators that degrade silently: `agg_alias.py` drops runs that have no `.pt`; `agg_miniworld.py` returns no loss,
  which changes its convergence filter; `experiment_audit.py` (MiniWorld layout) reads `losses` from the `.pt`.

## 5. Recommended deletion list

Checkpoints only (`*.pt`); every `.json`, `.log`, `.txt`, `.md` and `.done` stays.

| step | paths | GiB | when |
|---|---|---:|---|
| R1 | D2a SAFE: `runs/{indirect_t3, jsb_len512, jsb_centered, jsb_t3, jsb_forced, jsb_cross, jsb_decay, jsb_base512, jsb_base8192, jsb_base32768, frontier_alg, algorithmic, hier_parity, loop_hier, rope_canon, sameblock, dyck_t3}/**/*.pt` | 4.79 | after X0-X1 |
| R2 | D2b SAFE: `runs/indirect/**/*.pt`, `runs/indirect_200k/**/*.pt`, `runs/code/*.best.pt`, `runs/jsb/**/*.pt`, `runs/jsb_aug/**/*.pt`, `runs/tinystories/p0/*.final.pt` | 6.54 | after X0-X1 |
| R3 | SAFE AFTER EXTRACTION, nav: `runs/{mw_grid_sweep, mw_grid_sweep_hier, mw_hier_ablation, miniworld_fixed, miniworld_fresh, miniworld_oracle, miniworld, flipflop, vocab_em, minigrid_em, minigrid_em_fix, minigrid_2x2x2_allo, l15_loop_2x2, l15_ablation, noise_refine, loop_sampled, recipe_power, forget, forget_ctrl, mappope_r4, paper_task_rerun, lamtrace, n5_phase, counter_batch, recency_budget, cmq_curric, cmq_stab, cmq_sweep, mq_noise, mq_noise_c2, popewrap, mappope_recency, alias_sweep, alias_follow, rope_converge, g8_converge, visits_test}/**/*.pt`, plus the 120 symlinks in `runs/vocab_em_link/` | 4.34 | after X0-X4 |
| R4 | `data/tinystories/TinyStoriesV2-GPT4-train.txt` | 2.08 | after X6 |
| **total recommended** | | **17.75** | |
| ASK | `/home/prashr/mm_data/zips/15x15/*.zip` (12.55; after X7), `runs/code/*.final.pt` and `runs/code_decay/*.final.pt` (2.56) | 15.11 | only on the user's decision |
| KEEP | `runs/code_ablate/*.final.pt`, `runs/selective`, `runs/em_fig4`, TinyStories valid text | 1.43 | -- |

`safe_clear.sh` is the wrong tool here: it removes whole dirs and refuses most of these (completion markers). Work from
an md5 manifest as D1 did.

## 6. Extraction checklist (before R1-R4)

- **X0. Preconditions (rule 26).** Check that no `python3` training or eval process runs (rule-23 `ps` awk), that
  `nvidia-smi` is empty, that `lsof +D runs` shows nothing open under the target dirs, and that `git status` is clean.
  No other session may be active.
- **X1. Manifest and snapshot.** Write `docs/cleanup/D2_MANIFEST.md`: md5 and path of every file to delete, with the
  reason per dir as in `D1_MANIFEST.md`. Commit it, and re-verify it with `md5sum -c` just before deleting. Tar every
  non-checkpoint file under `runs/` (JSON, `.log`, `.txt`, `.md`, `.done`, about 0.1 GB) to a dated archive outside the
  repo and record its path and md5. The gitignored `.log` files are the only out-of-checkpoint loss curves (5-epoch
  resolution).
- **X2. Sidecars: the values that exist only in the checkpoints.** For every `.pt` in R3 (about 1,240 files, about 1,250 with `em_fig4`), dump
  every non-tensor key to one `runs/<dir>/CKPT_META.json` per dir, keyed by the `.pt` path relative to the dir:
  - all dirs: **`losses`** (full per-epoch list, 40-600 floats), `variant`, `seed`;
  - the 15 no-JSON dirs (`vocab_em, minigrid_em, minigrid_em_fix, minigrid_2x2x2_allo, l15_loop_2x2, l15_ablation,
    noise_refine, loop_sampled, recipe_power, forget, forget_ctrl, mappope_r4, em_fig4, paper_task_rerun, lamtrace`;
    `em_fig4` is KEEP, but dump it too): **`losses` and `config`** (`vocab_size, d_model, n_heads, n_layers,
    grid_size, n_obs_types, p_empty, env` or `n_landmarks`). These checkpoints hold no `results`; their per-seed
    evaluations are already in the tracked JSON of finding 4;
  - MiniWorld / alias / visits dirs: `losses`, `results`, `allocentric`, `oracle`, `vocab_size`, `d_model`,
    `n_heads`, `n_layers`;
  - `flipflop`: `losses`, `results`, `grid_size`; `counter_batch`, `recency_budget`, `mappope_recency`: `losses`,
    `results`, `k_*`, `n_symbols`, `min_gap`; `cmq_*`, `mq_noise*`: `losses`, `results`.

  Commit the sidecars. That is a few MB of JSON; gzip it if it is too noisy. Check: the number of entries equals the
  number of `.pt` files per dir, and each `losses[-1]` matches the last "loss=" line of the run's `.log` within its
  printing precision.
- **X3. Readers.** Either add a sidecar fallback to `agg_alias.py`, `agg_visits.py`, `agg_miniworld.py`,
  `experiment_audit.py` (MiniWorld layout) and `run_mw_gatectl.sh`'s inline table, and verify byte-identical output
  against the `.pt` path before deleting (rule 19), or leave them unchanged and record that they no longer re-run.
  Move `experiment_audit.py`'s documented example to a KEEP dir either way. Not md5-recorded, but they are analysis
  code: run `test_guards` and the 16 selftests after any edit.
- **X4. Notes in place.** Put a `CHECKPOINTS_DELETED.md` in each touched dir with the date, the manifest, and "do not
  relaunch <driver>" (section 4). Add the section-4 list to `EXECUTION_LOG.md`.
- **X5. After.** Confirm that `git ls-files -d runs` is empty, that `find runs -xtype l` prints nothing (remove
  `vocab_em_link/` first), and that the before/after listing differs by exactly the manifest. Run
  `python3 -m mapformer.test_guards` (13/13) and the 16 `--selftest` analyses.
- **X6. TinyStories.** Record `md5sum` of `TinyStoriesV2-GPT4-{train,valid}.txt`, the HF dataset URL and the download
  date in the D2 manifest (`data/` is gitignored), then delete the train file only.
- **X7. Memory Maze zips, only if the user approves deletion.** Record the md5 and byte size of `eval.zip` and
  `train-part0.zip`. Confirm whether the cluster scratch copy still exists, and copy it, or these, to a non-purgeable
  location first. Record in the D2 manifest that the arrays (with `MANIFEST.json`), not the zips, are the batch input.
  Keep `/home/prashr/mm_data/arrays/` and `batch_runs/` regardless.
- **Optional, to release the KEEP row later:** commit a script that recomputes ABLATE's delta census and dead-channel
  share from `code_ablate/*.final.pt`, and its output. Then `code_ablate` joins the ASK row. Patching
  `gain_grain_power.py` to read a sidecar would do the same for `em_fig4` (0.01 GiB, not worth it).

## 7. Limits

- Readers were found by literal path (`runs/<dir>`), by quoted dir names in `.py` / `.sh` files, and through
  `--runs-dir` arguments in the drivers. The scripts behind the large dirs, and every script flagged as loading
  checkpoints, were read. A script that builds a dir name at runtime from pieces would be missed. The two uncommitted
  computations found (CROSS Amendment 1's Bach r, ABLATE's census) show that some past numbers have no script at all.
- GPU costs: summed per-run `wall_s` from the run JSON where present (GPU-hours, language line). Otherwise
  checkpoint-mtime span, which is batch wall-clock with about 2 jobs per GPU and can include top-up waves
  (`minigrid_2x2x2_allo`'s 50 h span is two waves).
- The cluster copy of the Memory Maze zips and the Drive quota's current state were not checked from here.
