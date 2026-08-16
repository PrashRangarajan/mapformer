# Archived: void results

Every file here is **void** and retained only for the record. Each carries a
banner naming the specific evidence that invalidated it. Do not cite anything in
this directory.

Three causes, all established 2026-08-09:

1. **lm200 / landmarks** — checkpoints never converged (final CE ~1.0 vs ~0.005).
   The reported ranking tracked training convergence, not architecture.
   See `CORRECTED_LM200_LEADERBOARD.md` (kept at top level).
2. **hier-goal** — the task is solvable from the action prefix. Randomising the
   goal AND the whole explore phase leaves accuracy unchanged (0.912 -> 0.913);
   closed-loop success 0.013-0.037 vs a 0.010 random floor.
   See `HIERGOAL_ABLATION.md` (kept at top level).
3. **planner-demonstration tasks** — n-grams on the ACTION STREAM ALONE score
   0.650-0.969 against a chance of 0.250. See `PLANNER_TASK_AUDIT.md` (kept at
   top level).

The diagnostics that established each cause are deliberately NOT archived; they
are current results in their own right.
