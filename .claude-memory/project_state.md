---
name: project-state
description: LIVE STATE ONLY -- what is running, what the user must decide, the last results. History is docs/LOG.md; citable and withdrawn claims are in CLAUDE.md; orientation is docs/WHERE_THINGS_STAND.md.
metadata:
  type: project
---

Updated 2026-10-03 ~12:00. Goes stale fast: check `git log`, `.done` markers and the results files first.
New here? Read `docs/WHERE_THINGS_STAND.md` next.

## Running
- **Leak-remedy batch** (`LEAK_PREREG.md`, `run_leak.sh`, `runs/leak/p0`, marker `.leak_done`): MapWM vs ActOnly
  (action-only step) vs NormStep (step reads LayerNorm(emb)) on the new-object task, 8 seeds, 24 runs, launched
  2026-10-03 09:20, ~10 s/epoch at 8 concurrent, expected done ~17:00-18:00. On completion the driver writes
  `LEAK_ANALYSIS.txt` + `LEAK.json`; then write `LEAK_RESULTS.md` (registered branch per remedy arm: REMEDY /
  REMEDY WITH A COST / NO REMEDY; label x4 as robustness, rule 10), update CLAUDE.md and this file, commit, push.
- A code-verification agent was launched 2026-10-03 ~11:35 on the leak batch (blind to results). Its findings may
  require an amendment BEFORE reading results; likely issue: NormStep is invariant to embedding norm by construction
  (LN), so its x4 REMEDY branch may be guaranteed -- the informative contrast for NormStep is x1 (cost).

## Finished since 2026-09-30 (one line each; numbers in the file)
- 3D rank (`RANK_ND_RESULTS.md`) + rank x wrap (`RANK_WRAP_RESULTS.md`): rank = D hard in 2D and 3D; D+1 suffices on
  large tori (3D grid 18: 8/8); failures on wrap-only revisits; the 2D 100-cell cell memorised its map.
- Context step: the fixed hidden-state step HSR (step = W(emb + alpha*LN(h1)), alpha init 0) learns a step 4/4 and
  ignores far decoys (`CTXSTEP_HSR_PILOT.md`); the registered batch was STOPPED before any result for cost
  (`CTXSTEP_PREREG.md` STOPPED block). Pilots only.
- What/where: analysis `docs/WHAT_WHERE_ANALYSIS.md` (+ correction banner: key-step error fixed 2026-10-03), checks
  `docs/WHAT_WHERE_CHECKS.md` (causal: converged path models do not need the content x position interaction; fixed
  32x32 map separates without redraw; leak = pre-LayerNorm step).
- New-object transfer pilot (`runs/newobj_pilot`, seed 100, n=1): path arms copy unseen objects ~0.98-0.99; transfer
  to iid codes is known prior art (Chen 2019); the train-test "gap" was blank calibration.
- Literature reviews: `docs/lit/LIT_CONTEXT_STEPS.md`, `LIT_WHAT_WHERE.md`, `LIT_NEW_OBJECTS.md` (what is prior art).

## Open / waiting
- Leak batch (above). Then candidate next steps from the reviews: window limit as a cue-distance curve (~10-12 h);
  Mamba-3-style gate at depth vs HSR (~13 h); separation vs data at matched map size (~3.5 h, 48 runs);
  anisotropic SWAP code shift for the leak (gate on CPU first).
- H1 part 2 (loop and 4-layer rank-2 arms at 1800 ep): deferred.
- Documents: the .tex papers lack rank 3, H1 part 1, H3, text world, 3D rank/wrap, what/where checks. Shared report
  v10 (`report/language_summary.html`, https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc) lacks the 3D/wrap and
  what/where results.
- Trainer consolidation (sed-derived trainers: train_cancel, train_textworld, train_ctxstep{,2,3}, train_newobj).

## Loose ends
- `runs/rank_mi/p0/` is untracked; `Vanilla_r2ph_s0/s1` there are symlinks into `runs/rank_perhead_pilot/p0/`.
- Checkpoints are the only copy of per-epoch loss curves; do not clear run dirs.
- `CANCEL` eval stream for seed-0 runs overlaps training batch 0; `train_newobj`/ctxstep trainers use eval seed 10**6.
- `lib_driver.sh` now counts every mapformer.train_* job (shared slot budget) since 2026-09-30.
- Jericho environment and feasibility outputs live outside the repo: `/home/prashr/jericho_env`, `/home/prashr/jericho_data`.
