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
5. **Text world** -- DONE 2026-09-28 21:14 (`TEXTWORLD_RESULTS.md`): PATH WINS IN WORDS (0.969 vs
   RoPE 0.505, RoPE 2L 0.772); step-table verdict no branch (4/8), but opposites cancel on 8/8 after
   the common component; half the seeds add a per-step clock. Jericho feasibility: NOT a good test
   (6/57 games clean, near-trees; /home/prashr/jericho_data/feasibility/).
