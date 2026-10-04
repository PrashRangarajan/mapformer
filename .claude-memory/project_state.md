---
name: project-state
description: LIVE STATE ONLY -- what is running, what the user must decide, the last results. History is docs/LOG.md and docs/SESSION_*.md; citable and withdrawn claims are in CLAUDE.md; orientation is docs/WHERE_THINGS_STAND.md.
metadata:
  type: project
---

Updated 2026-10-03 21:25. Goes stale fast: check `git log`, `.done` markers and results files first.
New here? Read `docs/WHERE_THINGS_STAND.md` next; this week's narrative is `docs/SESSION_2026-09-27_to_10-03.md`.

## Running
- NormStep on the text world (`TW_NORMSTEP_PREREG.md`, + Amendments 1-2, committed 302d901 before launch): 32 runs
  (MapWM, NormStep, NormStepNB, DirOnly x seeds 10-17), `run_tw_normstep.sh` launched 2026-10-03 21:21, MAXPG 3,
  ETA ~00:30. Done marker `.tw_normstep_done`; analysis `TW_NORMSTEP_ANALYSIS.txt` / `TW_NORMSTEP.json` (written by the
  driver). Verification agent already ran (Amendment 1). Then: write TW_NORMSTEP_RESULTS.md, fill the `<!-- TWNS -->`
  slot in `report/language_summary.html` and republish WITH url= (3D rank, what/where, leak already added locally).
- Found in its pilot (Amendment 2, `docs/audits/2026-10-03/dropout_mode_check_out.txt`): eval mode under-reports
  clock-type solutions by 0.08-0.15 (attention-probability dropout; train mode 0.99). Unchecked on other batches.

## Last results (newest first; one line each, numbers in the file)
- NormStep analysis (`docs/NORMSTEP_NOTES.md`): scale robustness is by construction; zeroing its obs steps removes a
  per-move gauge, not leak; identity leak ~5x smaller than MapWM's (2 seeds). ANALYSIS.
- Leak remedies (`LEAK_RESULTS.md`, REG): ActOnly and NormStep +0.0107 unseen-object acc in distribution (p 0.0002)
  = MapWM's leak; 16/16 SOLVED vs MapWM 8/8 descending; x4 test unfailable (Amendment 1).
- What/where checks (`docs/WHAT_WHERE_CHECKS.md`, POST HOC): path models do not need the content x position
  interaction; separation without map redraw; leak = pre-LayerNorm step.
- Literature reviews (`docs/lit/LIT_CONTEXT_STEPS.md`, `LIT_WHAT_WHERE.md`, `LIT_NEW_OBJECTS.md`): what is prior art.
- New-object pilot (`runs/newobj_pilot`, n=1): path arms 0.976-0.982 on unseen codes; transfer is prior art.
- Rank x wrap (`RANK_WRAP_RESULTS.md`, REG): 3D grid 18 rank 4 8/8; the 3D shortfall was the small grid; 2D 100-cell
  cell memorised its map.
- 3D rank (`RANK_ND_RESULTS.md`, REG, no branch): rank = D hard in 2D and 3D; failures on wrap-only revisits.
- What/where analysis (`docs/WHAT_WHERE_ANALYSIS.md`, section 6 corrected 2026-10-03). ANALYSIS.
- Context step (`CTXSTEP_HSR_PILOT.md`, PILOT): HSR learns a step 4/4; registered batch STOPPED (`CTXSTEP_PREREG.md`).
- Earlier this week (REG): H1 part 1 (`LOOP_RANK_E1800_P1_RESULTS.md`), text world (`TEXTWORLD_RESULTS.md`), H3
  (`CANCEL_RESULTS.md`), sign matched (`SIGN_MATCHED_RESULTS.md`), rank 3 (`RANK3_RESULTS.md`), code full-val
  (`CODE_FULLVAL_RESULTS.md`).

## Open decisions (user picks; costs are wall-clock estimates on both GPUs from comparable batches -- re-measure s/epoch before launching)
1. NormStep on the text world + a bias-free NormStep arm (word-count-clock risk) -- ~3 h.
2. Separation vs data at matched map size (`LIT_WHAT_WHERE.md` P2) -- 48 runs, ~3.5 h.
3. Window limit as a cue-distance curve (`LIT_CONTEXT_STEPS.md` P3) -- ~10-12 h; Mamba-3-style gate vs HSR (P1) ~13 h;
   HSR decomposition (P2) ~9 h.
4. Leak test that can fail: anisotropic codes + SWAP shift (`LIT_NEW_OBJECTS.md` E2; CPU gate first) -- ~10 h.
5. H1 part 2 (loop and 4-layer rank-2 arms at 1800 ep) -- ~6 h.
6. No GPU: trainer consolidation (train_cancel, train_textworld, train_ctxstep{,2,3}, train_newobj; verify
   loss-exact); documents -- the .tex papers lack rank 3, H1 part 1, H3, text world, context step, 3D rank/wrap,
   what/where, leak; shared report v10 (`report/language_summary.html`, republish WITH url=) lacks 3D/wrap,
   what/where, leak.

## Loose ends
- `runs/rank_mi/p0/` is untracked on purpose; `Vanilla_r2ph_s0/s1` there are symlinks into `runs/rank_perhead_pilot/p0/`.
- Checkpoints are the only copy of per-epoch loss curves; do not clear run dirs.
- `CANCEL` eval stream for seed-0 runs overlaps training batch 0; `train_newobj` / ctxstep trainers use eval seed 10**6.
- Killing a trainer orphans its `--data-workers`; after stopping runs check `ps` for parent-1 python3 processes.
- Jericho environment and feasibility outputs live outside the repo: `/home/prashr/jericho_env`, `/home/prashr/jericho_data`.
