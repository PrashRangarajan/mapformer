---
name: project-state
description: LIVE STATE ONLY -- what is running, what the user must decide, the last results. History is docs/LOG.md; citable and withdrawn claims are in CLAUDE.md; orientation is docs/WHERE_THINGS_STAND.md.
metadata:
  type: project
---

Updated 2026-09-30. Goes stale fast: check `git log`, `.done` markers and the results files first.
New here? Read `docs/WHERE_THINGS_STAND.md` next.

## Running
Nothing. No GPU jobs.

## Finished since 2026-09-27 (one line each; numbers in the file)
- Rank 3 per head: RANK 3 SUFFICES at its boundary, 6/8, Holm 0.054 -- `RANK3_RESULTS.md`
- Sign at matched length: SIGN IS CAPABILITY, Abs - Signed -0.177, 0/8 vs 8/8 -- `SIGN_MATCHED_RESULTS.md`
- H1 part 1: 2x budget solves 0/8 rank-2 runs (5/8 still descending) vs 7/8 rank 4 -- `LOOP_RANK_E1800_P1_RESULTS.md`
- H3 cancellation knob: no registered branch; exchange rate 3 layers (knife-edge at p_plus 0.9) -- `CANCEL_RESULTS.md`
- Text world: PATH WINS IN WORDS 0.969 vs 0.505; step table B no branch, map on 8/8 -- `TEXTWORLD_RESULTS.md`
- Code full-val rescore: no sign flips; 4 contrasts keep t p < .05 at n=3 -- `CODE_FULLVAL_RESULTS.md`
- Jericho feasibility: NOT a good test (6/57 games clean, near-trees) -- `/home/prashr/jericho_data/feasibility/table_final.txt`
- Context-step pilots 1-3 and the HS recipe pilot -- `CTXSTEP_PILOT{1,2,3}.md`, `CTXSTEP_HS_RECIPE.md`
- 2026-09-30 audit: pilot seeds reused in TEXTWORLD and CANCEL (fresh-seed numbers in
  `docs/audits/2026-09-27/fresh_seeds.txt`), text-world clock is real, swap tests committed.

## Open / waiting
- **Context step** (`CONTEXT_STEP_DESIGN.md`): the window result (context gate and SR generator fail
  past their 4-token window, n=1 per cell) is ready to pre-register; exclude or declare seeds 0, 1, 5;
  SR is not one-knob (full rank, no omega) -- add a rank-4 SR arm or say so.
- **HS fix untested**: step from `emb(x_t) + LN(h1_t)` instead of LN(h1) alone, so it starts as the
  context-free model. Pilot it before registering HS.
- **H1 part 2** (loop and 4-layer rank-2 arms at 1800 ep, the registered H1 primary): deferred.
- **Documents**: four .tex/PDFs corrected 2026-09-30 only where contradicted (sign, code full-val;
  `mapformer_math` needed nothing); they do not carry rank 3, H1 part 1, H3, the text world or the context step. The shared
  report source `report/language_summary.html` updated and republished 2026-09-30 as v10 (same link).

## Loose ends
- `runs/rank_mi/p0/` is untracked; `Vanilla_r2ph_s0/s1` there are symlinks into
  `runs/rank_perhead_pilot/p0/`. Do not delete the pilot dir.
- Trainers duplicated by sed (`train_textworld.py`, `train_ctxstep{,2,3}.py`, `train_cancel.py`);
  consolidation into one trainer proposed in the 2026-09-30 audit, not done.
- Checkpoints are the only copy of the per-epoch loss curves for these batches; do not clear run dirs.
- `CANCEL` eval stream for seed-0 runs overlaps training batch 0 (see `CANCEL_RESULTS.md` block).
- 2026-10-01: 3D rank test DONE (`RANK_ND_RESULTS.md`): no registered branch; rank = D hard in 2D and 3D (1/8 each); rank D+1 rescues 2D (8/8) but 3D only 4/8. Failures on wrap-only revisits. What/where analysis: `docs/WHAT_WHERE_ANALYSIS.md`.
- 2026-10-02: rank x wrap DONE (`RANK_WRAP_RESULTS.md`): registered WRAP DRIVES IT; clean half is 3D (grid 18: rank 4 8/8 vs grid 10: 4/8); 2D high-wrap cell memorised its 100-cell map, so its half is invalid.
