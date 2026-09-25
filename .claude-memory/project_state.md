---
name: project-state
description: LIVE STATE ONLY -- what is running, what the user must decide, the last results. History is docs/LOG.md; citable and withdrawn claims are in CLAUDE.md.
metadata:
  type: project
---

Written 2026-09-24 ~23:00. Goes stale fast: check `git log`, `.done` markers and results files first.
Not pushed (`main` well ahead of `origin/main`).

## Running
Nothing.

## Last results (all committed)
- **Rank line resolved, budget-scoped** (`RANK_MI_RESULTS.md`): T=1024 torus, 900 epochs, matched init --
  our shared r=2 0/8 solved, a per-head r=2 2/8, shared r=4 8/8; which of per-head rank, sharing or
  W_out scale matters is unseparated (review 2026-09-25).
  A rank-2 solution exists and is held (`RANK_PROJ_RESULTS.md`, S1 STABLE). Continuations were UNREADABLE
  (`RANK_MATCHED_RESULTS.md`). Separating arms designed: C_bd and per-head r=4 (`RANK_MI_RESULTS.md`).
- **Audits of 2026-09-24** (`docs/audits/2026-09-24/`): CLAUDE.md cut 183 KB -> 21 KB (log in `docs/LOG.md`);
  code fixes applied in 17 commits 7aff4b2..ac3f338, default path verified bit-identical; opt-in
  `--fast-attn` (2.2x, not reproducible) and `--fast-attn --deterministic` (1.25x, bitwise); drivers use
  `lib_driver.sh` (2 jobs/GPU).

## Waiting on the user
1. Correct and republish the shared report (https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc, source
   `report/language_summary.html`): it headlines Dyck as training-length (it is 3x training depth), quotes
   navigation +0.461 (converged +0.243), contradicts itself on PoPE, and predates the rank resolution.
2. Document edits from the theory audit (`docs/audits/2026-09-24/theory_REPORT.md`, per-file line lists) for
   `positional_review.tex`, `axes_measured.tex`, `mapformer_math.tex`, `report/*.tex`, `README.md`.
3. Experiments, cheapest first: code checkpoints rescored on the full val file (eval-only, 1-2 GPU-h; settles
   the code "reversal", p ~0.1); Dyck control at matched nesting depth (~3 GPU-h); sign at matched length
   (10-20 GPU-h); rank-2 at r=4's W_out init scale (~2 h).

## Known loose ends
- md5 guards abort resuming any pre-2026-09-24 series (training files changed bit-identically); regenerate
  `code_md5.txt` deliberately to resume.
- `analyze_dyck_depth.py` pairs arms by list position (same bug fixed in the ladder); committed numbers fine.
- Eight results files sit in /home/prashr, outside the repo (`MINIWORLD_GATES.md` exists only there).
- Jobs per GPU not re-measured with `--fast-attn`.
