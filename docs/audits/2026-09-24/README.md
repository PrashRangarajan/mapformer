# Audits of 2026-09-24

Five read-only audits run while the rank continuation and stability batches trained, plus
the end-to-end review of the rank matched-length line. Nothing here is applied unless a
commit says so. Patches are against the tree as of commit ba42ec5.

| audit | report | what is in it |
|---|---|---|
| context / tokens | `context/REPORT.md` | CLAUDE.md classification; APPLIED in 670ad10 (183 KB -> 21 KB, log in `docs/LOG.md`) and the memory merge b277599 / c0b2d31 / ab05f91 |
| experiments | `experiments/LEDGER.md`, `PROPAGATION.md`, `EXPERIMENTS.md` | 58 claims by status; 44 places in 11 documents still citing retracted results; cheapest decisive experiments |
| theory / documents | `theory_REPORT.md` | withdrawn theory still asserted, over-claims, novelty claims the corpus contradicts, per-head vs shared bottleneck, minimal edits per document |
| efficiency | `efficiency/REPORT.md`, `efficiency/*.patch`, `efficiency/proposed/`, `efficiency/verify_*.py` + `.out` | vectorised walk (byte-identical, 23x), SDPA without TF32 (new series only), scale fold, sync-free loop, vectorised evaluators and permutation test; `verify_gpu_bitexact.py` NOT yet run |
| correctness / stats / hygiene | `hygiene/REPORT.md`, `hygiene/patches/01..13` | small-n MDE calibration, experiment_audit no-op, safe_clear fails open, stale checkpoint (quarantined), rank-pipeline latent risks, relative output paths, pgrep -f |
| rank matched-length review | `rank_review.md` | the review behind Amendment 2 |

Apply order (from the reports): after the running batches finish and their drivers exit --
env fast walk + permutation vectorisation (byte-identical), evaluators, then scale fold and
sync-free loop after the GPU bit-exact check, then SDPA for new series only. Hygiene
patches 03/05/06/09/11 touch files the running drivers use; apply after they exit.

## Applied (2026-09-24, evening; nothing was training)

Every change was checked against a pristine worktree of `5b6da26`; scripts and raw outputs
are in `applied/`. Training-path changes: the same short configs run old and new on the same
GPU (`cmp_train.sh`, 11 configs: rank T1024 bs16 with 0 and 3 data workers for Vanilla /
Vanilla_r4 / Vanilla_r2ph, torus default T128 bs128 serial and with workers, VanillaEM, d_head
48, Level15_DoG with action noise, Looped, an `--init-from --data-seed-offset 1`
continuation) gave bitwise-identical per-epoch losses, parameters and optimizer state after
every training-path commit. Evaluators and analyses: `run_evals.sh` / `run_evals_extra.sh`
regenerate the committed rank files, `NOISE_REFINE.*`, `L15_ABLATION.json` and five strata
JSONs byte for byte.

| patch | status | commit | evidence / note |
|---|---|---|---|
| efficiency #2 `environment_fastwalk` | applied (guard tightened: default ACTION_DELTAS, CPU int64 obs_map) | 7aff4b2 | `verify_env_repo` 131/131; 20x per batch |
| efficiency #6 `eval_rank_strata_vec` | applied | 8ab06f0 | 5 strata JSONs identical; 61 s -> 10.6 s |
| efficiency #7 `analyze_perm_vec` + hygiene 02 + 11 | applied as ONE implementation in `stats_core.py` (reworked so the exact branch reproduces the old arithmetic; fixed-grid CI kept, clipping reported) | 1c0c0f1 | 846/846 p-values ==; CI 6.6 s -> 0.08 s; RISING note added as a new line, not appended |
| hygiene 01 `stats_guard_small_n` | applied, via `stats_core`; the sd=0 verdict relabel NOT applied (note in `row_full` instead) | 6ef2c4a | test_guards 13/13 |
| hygiene 07 `experiment_audit` | applied (classifier from `stats_core`) | cabb432 | e900: exit 2, 4 STALLED / 4 DESCENDING |
| hygiene 08 `safe_clear` | applied + two more refusals (any `.*done*` inside; repo marker naming a PREFIX, which the patch missed for `rank_proj_train`) | 48889ce | `test_safe_clear.sh`, scratch dirs only |
| hygiene 09 stale checkpoint / strict loads | applied | 11d579c | 95/95 pass; stale copy refused; 24 strict loads |
| hygiene 10 dyck ladder pairing | applied | d954709 | old == new; JSON == committed |
| efficiency #3 scale fold, #5 sync-free loop | applied (default ON: bit-identical on GPU) | 64a463e | `verify_gpu_bitexact_repo` part 1 BITWISE EQUAL x3 |
| hygiene 03 seed offset / init checks | applied, rebased onto `--init-from`/`--save-full-state`; n_steps mismatch is a NOTE, not an error; offset semantics unchanged | 62334d9 | `test_init_from_guards.txt` |
| hygiene 04 `build_rank_proj` provenance | applied (not in the requested list; future builds only) | 62334d9 | existing `runs/rank_proj` untouched |
| hygiene 06 eval env from config | applied | da0838f | only popewrap used another grid, and passes it |
| efficiency #1 SDPA `--fast-attn` | applied opt-in (default off), with `--deterministic` made STRICT: the patch's `warn_only=True` was measured not to make SDPA deterministic | 01ecdaa | `FAST_ATTN_RANK.md`: 2.2x (CLI, 3 workers), 1.25x deterministic |
| efficiency #8 `lib_driver.sh` + hygiene 05 + MAXPG | applied; `run_rank_matched/proj/perhead_pilot.sh` converted, MAXPG default 2 | 52ff3e3 | stub dry run, `driver_dryrun/` |
| hygiene 12 absolute default paths | applied | 6ab2ad5 | 46 modules, import outcome == HEAD |
| hygiene 13 pgrep -> ps comm | applied | 7f0da5f | 26 drivers `bash -n`; only comments mention pgrep |
| new: `probe_action_geometry --space delta` | applied opt-in | ed7d951 | default output == committed |

Left for a decision: the md5 guard in `run_rank_matched.sh` now ABORTS a re-invocation of any
existing series (environment/model/train/train_variant changed, bit-identically) -- regenerate
a run dir's `code_md5.txt` deliberately if a series must be resumed; `analyze_dyck_depth.py`
has the same pairing-by-position pattern as the ladder and was not changed; jobs per GPU were
not re-measured after SDPA; the eight results files under /home/prashr were not moved.
