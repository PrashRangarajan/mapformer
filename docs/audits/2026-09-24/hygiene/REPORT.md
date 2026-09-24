I found four high-severity problems: a statistics rule the results depend on is miscalibrated at small n, and three shared guards either fail open or do nothing. The harness blocked writing `REPORT.md`, so the full report is below; the 13 patch files are in `/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_hygiene/patches/`. I changed nothing in the repo: `git status` is as it was. All 13 patches apply cleanly in a dry run. The guard tests pass 13/13 as committed and 13/13 with patch 01 swapped in.

**Top findings:**
1. **The "DETECTABLE" verdict is too loose at n=3 (A1).** `|delta| > 2.8*sd/sqrt(n)` is really `|t| > 2.8`, which is only a 10.7% false-positive test at n=3 (6.8% at n=4, 2.7% at n=8). The quoted MDE is too small by 92% at n=3 and 16% at n=8. Below 6 seeds, no distribution-free test can reach p < .05.
   - The current C1 headline "MapPoPE−PoPE +0.0033 DETECTABLE against" has p = 0.097.
   - Bimodal seeds are not the problem at n=8; small n is.
2. **`experiment_audit.py` does nothing on current run layouts and misjudges convergence (A2).** On `runs/rank_matched_e900` and `runs/dyck_ladder` it prints "no pairs found" and exits 0. Its flatness rule calls 7 of 8 r=2 runs "flat" where the registered rule calls 4 of them DESCENDING.
3. **`safe_clear.sh` fails open (A3).** Run from /home/prashr, it offered to delete the completed `runs/rank_matched_e900` as a "partial run". I answered `n`; nothing was deleted.
4. **The stale `runs/code_ablate/PoPE-NoSigma_s2.best.pt` is still on disk (A4).** It is the iteration-15000 checkpoint from the 09-22 story. `eval_code_delims.py` and two code probes would read it without any check. I recommend renaming it to `.stale`.
5. **Rank pipeline (B1–B5).** No current rank readout is wrong; these are latent risks.
   - `build_rank_proj.py` copies the r=4 loss curve into the projected r=2 checkpoints.
   - `--data-seed-offset` is silently ignored without data workers, and offset 1 reuses seed s+1's walks.
   - `--init-from` checks only the variant name, not seed, grid or vocabulary.
   - `run_rank_proj.sh` waits forever if the e900c driver fails.
6. **Relative output paths (B6).** Eight results files ended up in /home/prashr. `MINIWORLD_GATES.md` is cited in the repo but exists only there, and three others differ from their repo copies.
7. **Hygiene (C1–C3).** 36 scripts compute the MDE inline, and none written since 09-15 imports `stats_guard`. About 29% of the .py files and 34% of the drivers look dormant.

**Apply-order warning:** do not apply patches 03, 05, 06, 09 or 11 while the current batches run. `run_rank_matched.sh` checksums `train.py` and `train_variant.py` and would abort a re-run, and `run_rank_proj.sh` is executing now.

---

# Audit report: correctness, statistics and hygiene (2026-09-24)

VERIFIED means I ran something; JUDGED means I only read the code.

## A. Results that may be wrong, and the statistics tooling

**A1. HIGH (VERIFIED): the verdict rule and the MDE are miscalibrated at small n.**
- **Where:** `stats_guard.py:65,77-79` and the 36 scripts that copy the rule. The verdict `|delta| > 2.8*sd/sqrt(n)` is the test `|t| > 2.8`.

| n | actual false-positive rate | exact 80%-power multiplier | quoted MDE too small by |
|---|---|---|---|
| 3 | 0.107 | 5.36 | 92% |
| 4 | 0.068 | 4.16 | 49% |
| 5 | 0.049 | 3.72 | 33% |
| 8 | 0.027 | 3.26 | 16% |
| 24 | 0.010 | 2.93 | 5% |

- **No exact test below 6 seeds:** the smallest possible sign-flip p is 2/2^n, which is 0.25 at n=3.
- **Current headline affected:** CLAUDE.md says of Code C1 that "position +0.0055 and MapPoPE − PoPE +0.0033 [are] both DETECTABLE" (`CODE_RESULTS.md:46-47`).
  - MapPoPE − PoPE +0.0033: t = 2.98, p = 0.097. The exact-t MDE is 0.0059, bigger than the effect.
  - Position +0.0055: p = 0.043. NoDelta − Full −0.0061 (`ABLATE_RESULTS.md:60`): p = 0.035.
- **Other files:** of 16 n=3 "DETECTABLE" contrasts I parsed, two have p > 0.05 (`CODE_RESULTS_OOD.md:55` p = 0.065, `:22` p = 0.058).
- **Bimodal seeds are not the cause:** a simulated bimodal null at n=8 fired the verdict 2.7% of the time, the same as the normal case.
- **Edge case:** if every seed's difference is identical, sd = 0 makes the MDE 0 and any nonzero delta reads DETECTABLE.

Fixes:
- **Patch 01** adds exact-t MDE, t-test p, exact sign-flip p and an "n<6" tag to `stats_guard`. The house verdict itself is unchanged, so committed tables still reproduce.
- **Patch 02** adds a new `stats_core.py`: the exact MDE and alpha helpers, paired and sign-flip tests, a vectorised two-sample permutation test and interval, Fisher's test on solved counts, the run classifier with rising runs flagged, and a bimodality flag.
  - Its permutation p equals `analyze_rank_matched.perm` to 1e-12 on the e900 JSON at three lengths and four shifts.
  - Its classifier matches `classify()` on 200 random loss curves.

**A2. HIGH (VERIFIED): `experiment_audit.py` is a no-op on most layouts and its convergence rule is wrong.**
- **No-op:** `:176-177` only understands MiniWorld json/pt pairs and exits 0 otherwise. Confirmed on `rank_matched_e900` and `dyck_ladder`.
- **Unsound flatness test:** `:39,64-65` compare two single epochs against a fixed threshold.
  - On e900 it calls 7/8 r=2 runs flat where the registered rule calls 4 DESCENDING.
  - On `mw_grid_sweep` it calls RoPE flat 0/9 where the windowed rule gives STALLED 7/9.
- **Patch 07:** checks convergence from the checkpoints alone, exits 2, and prints the registered windowed classes alongside.

**A3. HIGH (VERIFIED): `safe_clear.sh` fails open.**
- **Why:** `:8-9` looks for a relative marker path, which never resolves from /home/prashr.
- **What I saw:** on `runs/rank_matched_e900`, which has `.rank_matched_e900_done` and `.train_done`, it offered deletion as a "partial run".
- **Coverage:** only 46 of 351 run directories have a marker with the default name.
- **Patch 08:** resolves markers against the repo, refuses when `.train_done` or any marker naming the directory exists, and refuses when a running process references the directory.

**A4. HIGH (VERIFIED): a stale checkpoint is still on disk.**
- **What:** `runs/code_ablate/PoPE-NoSigma_s2.best.pt` is at iteration 15000 with val bpc 0.9962; its run JSON says 0.9156. It was the only stale one of 96 code checkpoints I scanned.
- **Who would read it:** `eval_code_delims.py:138-149` and `probe_code_depth0.py`/`probe_code_accum.py` have no staleness check. The two probes also load with `strict=False`, though every arm loads cleanly today.
- **Patch 09:** adds `ckpt_guard.check_not_stale()` (tested: seed 1 passes, seed 2 raises), calls it from all three readers, and makes the loads strict.

## B. Silent-failure risks in the code committed 09-23/24

- **B1 (VERIFIED):** `build_rank_proj.py:45` saves the r=4 run's 900-epoch loss curve as the projected r=2 checkpoint's own. Through `--init-from`, every `rank_proj_train` checkpoint will record it as its prior history. `analyze_rank_proj` reads only the fresh losses, so its output is fine. **Patch 04.**
- **B2 (VERIFIED/JUDGED):** `--data-seed-offset` and `--init-from` (`train.py:92-98`, `train_variant.py:516-521`).
  - The offset only applies when data workers are used, so a serial continuation silently replays its parent's walks. The rank drivers use workers, so this is not triggered now.
  - Offset 1 makes continuation seed s train on exactly the walks original seed s+1 used (checked by arithmetic on `_seed_for`).
  - `--init-from` checks only the variant name and tensor shapes, so a parent from another seed would silently continue on a different map.
  - **Patch 03** fixes these and saves all command-line arguments into the checkpoint.
- **B3:** checkpoints do not record the environment settings (noise levels, action mode, boundary and so on). `eval_noise_refine.py:69,99` always evaluates on a 64-grid default torus and infers the noise from the directory name. No current driver misuses it. **Patches 03 and 06.**
- **B4:** `run_rank_proj.sh:48` waits forever if the e900c driver fails, since a failure sets no marker. `:26` reuses any existing checkpoint without a config or code check. **Patch 05**, to apply after the script exits.
- **B5:** `analyze_rank_matched.py`.
  - A run whose loss rose is filed as DESCENDING. That follows Amendment 2's wording, but it counts toward the "unreadable" rule in e900c, where the restart deliberately perturbs solved runs.
  - The confidence-interval grid is fixed at ±0.6, so it can clip silently. It also did not finish within 115 s under current load.
  - The R2 branch passes if either of two tests does (as registered), so both p-values should be reported.
  - **Patch 11**, which needs patch 02.
- **B6 (VERIFIED):** 46 modules default their output or runs path to a relative path. Eight results files are in /home/prashr: `MINIWORLD_GATES.md`, `HEX_EMERGENCE_RESULTS.md` and two `MODEOMEGA_*` files exist only there; `BUMP_TOKEN_RESULTS.md`, `EXTRAHEAD_CONTROL.md` and `VECTOR_NAV_V2_RESULTS.md` differ from the repo copies. **Patch 12.**
- **B7, lower:**
  - `pgrep -f` is still used in 13 old drivers, their `-af` helpers and `wire_data_parallel.py:97` (**patch 13**).
  - `analyze_dyck_ladder.py` and `analyze_dyck_depth.py` pair arms by position in a string-sorted file list and hard-code "/8". Both committed batches are complete, so no current number is wrong (**patch 10**).
  - 30 scripts silently skip missing checkpoints, and `eval_rank_strata` exits 0 when some are missing.
  - 15 broad `except: pass` handlers; `agg_refine.py:33` silently drops the gate reading.
  - `train.py` divides the epoch loss by all batches, including skipped ones; this is negligible at T ≥ 128.
  - Checked and fine: the stratum classification in `eval_rank_strata` agrees with the environment on 6,190 revisits, and the rank projection matches `ActionToLieAlgebra` exactly.

## C. Hygiene

- **Duplicated statistics:** 36 scripts compute the MDE inline, 12 hand-roll loss-matching, and nothing has imported `stats_guard` since 09-15.
- **Duplicated scheduling:** 52 drivers define their own GPU picker and 67 their own job counter; only 9 of 283 use `flock`. A shared `lib_sched.sh` would cover these.
- **Dead or dormant files (heuristic):**
  - 77 .py modules (21%) are unreachable from anything touched since 08-09.
  - A further 27 model modules are imported only by `train_variant` and have no checkpoint since the lm200 retraction, making about 29% of .py dormant.
  - 95 drivers (34%) are older than 08-09, and 61 point only at deleted run directories.
  - 45 .md files carry a superseded/retracted/void banner.
- **Consolidation that breaks no `python3 -m mapformer.X` command:**
  - Keep every .py at the top level. Make `VARIANT_MAP` load lazily, and generate a `MANIFEST.md` marking modules active, dormant or void.
  - Move drivers whose outputs are gone or void to `archive/drivers/`.
  - Leave results .md/.json at the top level, because `test_guards` and the analysis scripts read them by fixed path; archive only the 45 bannered files, after checking references.
  - Send new outputs to `results/` and `logs/` through one `paths.py`; move the existing logs only once no driver is writing to them.

## Patch index

| patch | fixes | apply now? |
|---|---|---|
| 01_stats_guard_small_n | A1 | yes |
| 02_new_stats_core | A1, C | yes |
| 03_train_seed_offset_and_init_checks | B2, B3 | **no** |
| 04_build_rank_proj_provenance | B1 | yes (future builds) |
| 05_run_rank_proj_guards | B4 | **no** |
| 06_eval_noise_refine_env_from_cfg | B3 | after the batches |
| 07_experiment_audit_fail_loudly | A2 | yes |
| 08_safe_clear_fail_closed | A3 | yes |
| 09_code_ckpt_stale_and_strict | A4 | after the batches (it edits `ckpt_guard.py`, which the drivers' final evaluation imports) |
| 10_dyck_ladder_pair_by_seed | B7 | yes |
| 11_analyze_rank_matched_shared_stats | B5 (needs 02) | after e900c's analysis step |
| 12_absolute_default_paths | B6 (46 files) | yes |
| 13_pgrep_f_to_ps_comm | B7 (27 files) | yes |

The scratch directory also holds the simulation and check scripts (`calib.py`, `verify_stats_core.py`), the scan outputs (`relative_paths.txt`, `modgraph.json`, `variants.json`, `dead_estimate.json`), and the before/after file copies the patches were built from.