---
name: project-state
description: LIVE STATE ONLY -- what is running, what the user must decide, the last results. History is docs/LOG.md; citable and withdrawn claims are in CLAUDE.md; orientation is docs/WHERE_THINGS_STAND.md.
metadata:
  type: project
---

Updated 2026-09-27 ~18:15. Goes stale fast: check `git log`, `.done` markers and the results files
first. New here? Read `docs/WHERE_THINGS_STAND.md` next.

## Running (all detached; judge completion by the .done marker AND the artifacts)
Five queues, launched 2026-09-27 17:47-18:10; each pre-registered and committed before launch.
The three train_variant drivers share the 2-jobs/GPU picker and run in this order as slots free:
1. **Rank 3 per head** -- DONE 2026-09-28 02:41 (`RANK3_RESULTS.md`): registered RANK 3 SUFFICES, at
   its boundary. Per-head r=3 6/8 SOLVED, acc 0.987 (r=2 0.885, r=4 0.999); 3 - 2 fires on accuracy
   only (perm p 0.027, Fisher 0.13, Holm 0.054); 4 - 3 UNMEASURED. Reproduction exact. The two
   unsolved seeds sit in the non-cancelling basin (opposition 1.6-1.8). Next test: 3D torus.
2. **Sign at matched length** -- DONE 2026-09-28 14:08 (`SIGN_MATCHED_RESULTS.md`): registered SIGN IS
   CAPABILITY. Trained and tested at T=1024: Abs - Signed -0.177 (perm p 0.0002), solved 0/8 vs 8/8;
   opposition signed 0.06 vs monotone 1.92-1.97. Monotone arms stalled (budget-scoped).
3. **H1 at 1800 epochs** -- PAUSED AGAIN 2026-09-28 18:11 (user: free the GPUs for the text world).
   Kept: `Vanilla_s0`, `Vanilla_r4mi_s0`. Killed at ~epoch 780/1800: Looped s0/s1, Vanilla_L4 s0;
   Vanilla_L4 s1 at 245. Restart: `setsid nohup ./run_loop_rank_e1800.sh >/dev/null 2>&1 </dev/null &`
   (skips finished runs; killed ones restart from scratch).
5. **Text world** (navigation told in words): `environment_textworld.py`, `train_textworld.py`,
   `probe_textworld.py`, gate `docs/audits/2026-09-27/gate_textworld.py` (vocab 58, floor 0.512,
   revisit 0.231). Pilot `runs/textworld_pilot` running; pre-registration next.
4. **H3** -- DONE 2026-09-28 14:37 (`CANCEL_RESULTS.md`): no registered branch (index 1-layer acc
   non-monotone in p_plus, dip at 0.75); secondary: exchange rate 3 wherever steps cancel, 1 at the
   clock, path 1 layer 32/32 -- replicates Dyck's 1 ~ 3 layers.
Plus, outside the picker:
4. **Code full-val rescore** (`CODE_FULLVAL_PREREG.md`, `run_code_fullval.sh`, `.code_fullval_done`),
   eval-only on cuda:1, ~6 min per 2048 checkpoint.
5. **H3 pilot** (`runs/cancel_pilot/run.sh`, `.done` in that dir): 16 short runs on the biased 1D
   ring (`environment_cancel.py`, `train_cancel.py`), to set the recipe and noise floor BEFORE the
   H3 pre-registration is written. H3 itself is NOT registered yet.
Do not edit train_variant.py, model*.py, train.py, environment*.py while 1-3 run (md5 guards, rule 22).

## Last results (all committed)
- **H1: search aids partly recover rank 2, registered verdict UNMEASURED** (`LOOP_RANK_RESULTS.md`,
  2026-09-27). At T=1024, 900 ep, 8 seeds: r=2 + loop x4 (bit-identical params/init to r=2) solves
  2/8, acc 0.973; r=2 at 4 real layers 5/8, 0.990; plain r=2 0/8, 0.894; r=4 8/8, 0.998. Both aids
  fire on accuracy, only depth on solved count. Three loss regimes; nothing at rank 2 reaches r=4's.
  The registered 0.05 cutoff sits inside the new arms' spread (at 0.08+ H1 passes), so uncertain,
  not negative. Depth with 4x params beats the matched-param loop: NOT "search at constant capacity".
- **Rank resolves to PER-HEAD rank** (`RANK_SEP_RESULTS.md`, 2026-09-25): per-head rank 2 -> 0/8,
  2/8, 2/8; per-head rank 4 -> 8/8, 8/8. Decisive contrast D - C_bd (both block-diagonal), Fisher
  and permutation p 0.0070. Sharing (D-C) and `W_out` per-entry scale (C_bd-B) UNMEASURED; initial
  angle scale excluded. Supersedes the "unseparated" caveat in `RANK_MI_RESULTS.md`.
- **Dyck CLOSES at matched depth** (`DYCK_MDEPTH_RESULTS.md`, 210 runs, 2026-09-25): trained AND
  tested at L32 D12, every 4-layer arm at ceiling (index 0.997-0.998, path 1.000, floor 0.594),
  effect +0.002. The ladder's +0.168 was depth EXTRAPOLATION from D4. Survives: depth-substitution
  (+0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers) and mixture training over D 4..12 (+0.110 at
  D12). Closed in the same batch: the RoPE base confound at 4L; the 1x ladder budget WAS limiting
  the index arms (+0.021).
- Shared report at v9 (2026-09-28: sign, rank 3, code full-val, H3). The 2026-09-24 audits and code fixes are in `docs/audits/2026-09-24/`; the
  review scripts behind them, and behind the rank line, in `docs/audits/2026-09-24/rank_review*/`
  and `docs/audits/2026-09-25/rank_mi_review/` (rescued from a scratchpad 2026-09-27).

## Done 2026-09-27
- Pushed (207 commits); documents rewrite committed (75445b0): the five .tex files and
  RESULTS_INDEX.md now carry Dyck matched-depth, per-head rank and LOOP_RANK; PDFs rebuilt.

## Waiting on the user
0. (old item, done) **The documents rewrite.** The five `.tex` files and `RESULTS_INDEX.md` were corrected
   2026-09-25, BEFORE the two batches above, so they still carry the old Dyck framing (+0.168 at 3x
   depth as the language line's positive result) and the "unseparated" rank caveat. Per-file list
   in `docs/WHERE_THINGS_STAND.md`; use `docs/prepared/tex_edits_2026-09-25/rep.py`.
2. **Which experiment next** -- all five cheap items were launched 2026-09-27 (see Running). Still
   unlaunched: H3 proper (after its pilot), more loops (8, 16) and the loop at rank 4.

## Known loose ends
- md5 guards abort resuming any pre-2026-09-24 series (training files changed bit-identically);
  regenerate `code_md5.txt` deliberately to resume.
- `runs/rank_mi/p0` reuses ten arms by ABSOLUTE symlink into `runs/rank_matched_e900` and
  `runs/rank_perhead_pilot`. Deleting either breaks the rank_mi batch. Not committed for that reason.
- `analyze_dyck_depth.py` pairs arms by list position (same bug fixed in the ladder); committed
  numbers fine.
- Jobs per GPU not re-measured with `--fast-attn`.
- `/home/prashr/*.md` (outside the repo, rule 25 damage) is now accounted for: `MINIWORLD_GATES.md`
  is byte-identical to the repo's `MINIWORLD_GATES_FIXED.md`; `HEX_EMERGENCE_RESULTS.md` is a
  smaller-sample variant of a void'd file; `MODEOMEGA_{FINEGRAINED,LESION}.md` were archived to
  `archive/void/`. Nothing there is unique any more.
