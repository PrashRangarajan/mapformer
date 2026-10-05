---
name: project-state
description: LIVE STATE ONLY -- what is running, what the user must decide, the last results. History is docs/LOG.md and docs/SESSION_*.md; citable and withdrawn claims are in CLAUDE.md; orientation is docs/WHERE_THINGS_STAND.md.
metadata:
  type: project
---

Updated 2026-10-04 00:45. Goes stale fast: check `git log`, `.done` markers and results files first.
New here? Read `docs/WHERE_THINGS_STAND.md` next; this week's narrative is `docs/SESSION_2026-09-27_to_10-03.md`.

## Running
Nothing. GPUs idle (2026-10-05 00:00).

## Last results (newest first; one line each, numbers in the file)
- MapPoPE separated (`MAPPOPE_PAIR_RESULTS.md`, REG): the score rule carries the gain (+0.024, 16/16 vs 10/16 SOLVED at
  r2); the doubled angle count adds +0.0002, CI [-0.0004, +0.0008]. Open: does PoPE's score rescue rank 2 at T=1024?
- NormStep on words (`TW_NORMSTEP_RESULTS.md`, REG): accuracy no detectable difference (+0.006, MDE 0.096); predicted
  word-count clock fired but tiny (+0.057 rad optional-word drift, p 0.027, 0/64 channels), not from the LN bias.
  Post hoc: DirOnly oracle capped at 0.972 by aside objects (100% of its errors); eval mode under-reports runs below
  ceiling (attention dropout; up to +0.16 in train mode) -- UNCHECKED on other batches (next: eval-only CPU audit).
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

## Theory review (2026-10-04)
Five-agent review: `docs/theory/2026-10-04/00_PLAN.md` (theory T1-T7 and a ranked plan; reports 01-05). Corrections
applied: TW_NORMSTEP verdict B is the aside offset, not a word clock; the eval-mode gap is the 1/(1-p) attention
scale (CORRECTED block, verified). Plan step 1 (re-score) done: no verdict changes. Next: 1D ring rank 1 vs 2;
rank escape test.

## Open decisions (user picks; costs are wall-clock estimates on both GPUs from comparable batches -- re-measure s/epoch before launching)
1. DONE 2026-10-04: dropout-scale re-score of every committed batch (`docs/audits/2026-10-04/DROPOUT_RESCORE.md`): no
   registered verdict changes; only text-world path arms move. Next from the plan: 1D ring rank 1 vs 2; rank escape test.
2. Separation vs data at matched map size (`LIT_WHAT_WHERE.md` P2) -- 48 runs, ~3.5 h.
3. Window limit as a cue-distance curve (`LIT_CONTEXT_STEPS.md` P3) -- ~10-12 h; Mamba-3-style gate vs HSR (P1) ~13 h;
   HSR decomposition (P2) ~9 h.
4. Leak test that can fail: anisotropic codes + SWAP shift (`LIT_NEW_OBJECTS.md` E2; CPU gate first) -- ~10 h.
5. H1 part 2 (loop and 4-layer rank-2 arms at 1800 ep) -- ~6 h.
6. No GPU: trainer consolidation (train_cancel, train_textworld, train_ctxstep{,2,3}, train_newobj; verify
   loss-exact); documents -- the .tex papers lack rank 3, H1 part 1, H3, text world, context step, 3D rank/wrap,
   what/where, leak, NormStep on words; shared report v14 (`report/language_summary.html`, republish WITH url=) is current.

## Loose ends
- `runs/rank_mi/p0/` is untracked on purpose; `Vanilla_r2ph_s0/s1` there are symlinks into `runs/rank_perhead_pilot/p0/`.
- Checkpoints are the only copy of per-epoch loss curves; do not clear run dirs.
- `CANCEL` eval stream for seed-0 runs overlaps training batch 0; `train_newobj` / ctxstep trainers use eval seed 10**6.
- Killing a trainer orphans its `--data-workers`; after stopping runs check `ps` for parent-1 python3 processes.
- Jericho environment and feasibility outputs live outside the repo: `/home/prashr/jericho_env`, `/home/prashr/jericho_data`.
