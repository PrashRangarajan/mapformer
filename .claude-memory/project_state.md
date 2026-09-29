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
3. **H1 at 1800 epochs** -- SPLIT by Amendment 1 (2026-09-28 22:50). Part 1 running:
   `run_loop_rank_e1800_budget.sh` (A r=2 and C r=4 only, 8 seeds; marker `.loop_rank_e1800_p1_done`;
   reads only branch RANK 2 WAS BUDGET). Part 2 (Looped, Vanilla_L4) deferred; its runs were killed.
4. **H3** -- DONE (`CANCEL_RESULTS.md`).
5. **Text world** -- DONE 2026-09-28 21:14 (`TEXTWORLD_RESULTS.md`): PATH WINS IN WORDS (0.969 vs
   RoPE 0.505, RoPE 2L 0.772); step-table verdict no branch (4/8), but opposites cancel on 8/8 after
   the common component; half the seeds add a per-step clock. Jericho feasibility: NOT a good test
   (6/57 games clean, near-trees; /home/prashr/jericho_data/feasibility/).
6. **Context-dependent step** (`CONTEXT_STEP_DESIGN.md`): decoy text world
   (`environment_textworld_ctx.py`), `model_context_step.py` (CtxGateWM, HiddenStepWM),
   `train_ctxstep.py`, gate `docs/audits/2026-09-27/gate_ctxstep.py` (all pass). Pilot
   `runs/ctxstep_pilot` (CF, SR, CG, HS x 2 seeds, p_decoy 0.3) running since 23:05; pre-registration
   after it.
