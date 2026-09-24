---
name: project-state
description: LIVE STATE ONLY -- what is running, what is queued, what the user must decide, the last few results. History is docs/LOG.md; citable and withdrawn claims are in CLAUDE.md.
metadata:
  type: project
---

Written 2026-09-24 ~03:00. It goes stale within hours: check `git log`, the `.done` markers and
the result files before acting on it. `main` is 152 commits ahead of `origin/main` (not pushed).

## Running (owned by the main session, which has watchers on both; do not touch `runs/`, drivers or processes)

| batch | run dir / driver | pre-registration | done marker, outputs | expected |
|---|---|---|---|---|
| Rank continuation: all 16 runs warm-restarted from their 900-epoch checkpoints for 900 more (same recipe, fresh walk stream) | `runs/rank_matched_e900c`, `run_rank_matched.sh` (`EPOCHS=900 TAG=_e900c INIT_TAG=_e900`) | `RANK_MATCHED_PREREG.md` Amendment 3: Amendment 2 classes on the continuation's own 900 epochs; UNREADABLE again means stop and report | `.rank_matched_e900c_done`; `RANK_MATCHED_e900c.md`, `_GEOMETRY.md`, `_ANALYSIS.txt` | ~07:00 |
| Warm-start stability: each seed's rank-2 projection of the solved r=4, trained with the continuation's recipe; control = the continuation's r=4 arm | `runs/rank_proj_train`, `run_rank_proj.sh` (its analysis waits for the e900c marker) | `RANK_PROJ_PREREG.md`: S1 STABLE (>= 6/8 solved) means r=2's failure is search; S2 UNSTABLE (<= 2/8) triggers the gentle version (lr 1e-4, 300 ep); control < 6/8 is uninformative | `.rank_proj_done`; `RANK_PROJ_TRAIN.md/.json`, `_STRATA.json`, `_GEOMETRY.md`, `RANK_PROJ_ANALYSIS.txt` | ~08:00 |

## Last results (newest first)

- **Rank-2 projection, frozen** (2026-09-24): each solved r=4 projected to rank 2 scores 0.995 at
  T=1024 on 8/8 seeds (0.957 at T=2048). r=2 can represent the solution. `RANK_PROJ_FROZEN.md` and
  `.json` are UNTRACKED; commit them with the rank write-up.
- **Rank at matched length, 900 ep** (eca6d2c): r=4 solves 8/8, r=2 0/8 (4 stalled, 4 descending),
  +0.103 at T=1024; registered verdict UNREADABLE. `RANK_MATCHED_RESULTS.md`.
- **Bottleneck deviation found**: our `ActionToLieAlgebra` shares one r-dim latent across heads; the
  paper's is per head, so our r=2 has half its latent dims at 2 heads. Now a CLAUDE.md invariant.
- **Dyck ladder rescoped**: training is L32 D4; the L32 D12 headline is 3x the training depth
  (`DYCK_LADDER_RESULTS.md`, whose own text still says "training length" -- it has no banner yet).
- **Code C1 closed at matched length**: encoding -0.0030 (MDE 0.0046) against -3.694 extrapolating;
  the "reversal" is underpowered (paired t-test p 0.043 for position, ~0.1 for MapPoPE - PoPE).
  `CODE_RESULTS.md`.
- **Housekeeping, 2026-09-24**: CLAUDE.md 183 KB -> 21 KB (670ad10, log verbatim in `docs/LOG.md`);
  memory 35 -> 25 files; `RESULTS_INDEX.md` regenerated with ledger statuses.

## Queued for after both batches land

1. **Rank write-up**: `RANK_MATCHED_RESULTS.md` + the stability verdict; commit the RANK_PROJ files;
   update CLAUDE.md's "Open: rank" paragraph, [[project-rank-and-selective-rope]] and this file.
2. **Apply the audited code patches, after verifying each, once no batch spawns from the modules**
   (CLAUDE.md rule 22). They sit in the session scratchpad
   `/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/`, which
   is NOT durable -- copy them into the repo first if that session is ending.
   - `audit_efficiency/`: fast walk generation (8-16x, 75/75 byte-exact), sync-free train loop and
     scale-folded model (17/17 bitwise), vectorised permutation CI (50 s -> 0.12 s, identical),
     vectorised strata eval (identical), `train_variant` fast-attn flag, `lib_driver.sh`.
   - `audit_hygiene/patches/01-13`: small-n t-based MDE in `stats_guard` + a shared `stats_core`;
     train seed offset and init checks; rank-proj provenance and driver guards; `eval_noise_refine`
     env from config; `experiment_audit` fails loudly; `safe_clear.sh` fails closed; stale/strict
     code-checkpoint check; Dyck ladder paired by seed; `analyze_rank_matched` on shared stats;
     absolute default paths; `pgrep -f` -> `ps comm=`. Not yet verified here.

## Decisions pending with the user

- **Shared-report corrections** (`report/language_summary.html`, republish with `url=`): the
  navigation "+0.461, index on a 0.506 floor" (converged: +0.243, index 0.805); "plain PoPE still
  wins" (contradicted on code); the unsourced "six cases"; the cross-batch "best of eight".
- **.tex / README edits**: ~44 live propagations of retracted claims across the review, results
  paper, record, reports, README, BASELINE_TABLE and EM_WM_STATE (the 2026-09-24 audit's
  PROPAGATION list); e.g. README still says the InEKF wrap bounds theta.
- **New experiments**, in the audit's order: (1) full-val rescore of `runs/code2048` and
  `runs/code_decay` `.final.pt` (eval-only, <= 2 GPU-h); (2) Dyck matched-depth control (needs an
  opt-in training-depth flag in `train_dyck.py`, ~1.3 GPU-h) plus an index-arm budget extension
  (~2 GPU-h); (3) sign at matched length, `Abs_r4` trained and tested at T=1024, 900 ep (10-20
  GPU-h); (4) a per-head r=2 arm (the paper's bottleneck). Lower: converged rerun of
  rotate/allocentric; forget-clock rerun (the finished batch was deleted) with a matched-length arm;
  Level15 at matched length; torus and MiniWorld index-arm budget extensions.
- **Zero-GPU**: reclassify every "flat"-converged run dir with Amendment 2's classes; re-read every
  n <= 5 DETECTABLE under the t-based MDE.
