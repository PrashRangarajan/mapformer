# Check A: are hier-goal models navigating, or exploiting a shortcut?

T_explore=128 (OOD), T_navigate=64, n=3 seeds, interleaved task.
Copy-previous-action floor on this task = **0.327**. A genuine navigator should fall toward it when the goal is randomised or the explore path is destroyed; a model that barely moves is not using that information.

| variant | intact | random_goal | shuffle_explore | destroy_context |
|---|---|---|---|---|
| MapWM-Flat | 0.912 ± 0.015 | 0.913 ± 0.012 | 0.910 ± 0.010 | 0.913 ± 0.012 |
| MapWM-Hier | 0.834 ± 0.020 | 0.812 ± 0.027 | 0.831 ± 0.022 | 0.807 ± 0.010 |
| Plain-Flat | 0.915 ± 0.008 | 0.911 ± 0.015 | 0.913 ± 0.011 | 0.916 ± 0.011 |
| Plain-Hier | 0.694 ± 0.051 | 0.676 ± 0.047 | 0.686 ± 0.046 | 0.691 ± 0.046 |
| PoPE-Flat | 0.935 ± 0.003 | 0.940 ± 0.003 | 0.936 ± 0.001 | 0.938 ± 0.006 |
| MapPoPE-Hier | 0.928 ± 0.004 | 0.908 ± 0.031 | 0.925 ± 0.005 | 0.907 ± 0.030 |

## Verdict: the hier-goal task does not measure navigation. Both versions.

`destroy_context` randomises the goal tokens AND every explore action AND every
explore observation -- everything a cognitive map could be built from. Accuracy
is UNCHANGED for all six variants (MapWM-Flat 0.912 -> 0.913, Plain-Flat
0.915 -> 0.916, PoPE-Flat 0.935 -> 0.938). The models solve the task from the
navigate-phase action prefix alone.

### Root cause: the interleave "fix" hid the shortcut instead of removing it

Markov predictors fit to the navigate ACTION SEQUENCE ALONE -- no goal, no
explore, no observations, no model:

| task version | order-1 | order-3 | order-5 |
|---|---|---|---|
| raw BFS (original task) | **0.969** | 0.969 | 0.969 |
| interleaved (current "fixed" task) | 0.320 | **0.971** | 0.974 |

The original task was solvable by copying the previous action (order-1, 0.969).
`_interleave` is a DETERMINISTIC function of the action counts (most-remaining
first, tie-break lowest id), so the sequence remains fully self-predictable --
just at order 3 instead of order 1. The fix dropped order-1 to 0.320 and left
order-3 at 0.971, slightly HIGHER than the original shortcut.

The copy-previous-action baseline (0.327) was the only check used to validate
the fix. It tests order 1 only, so it certified a task whose order-3 shortcut is
stronger than the one being removed. A transformer learns order-3 trivially.

### Scope of invalidation

Every hier-goal number in this repo, both task versions:
`HIERGOAL_RESULTS.md`, `HIERGOAL_MULTISEED.md`, `HIERGOAL_FIXED.md`,
`HIERGOAL_FIXED_LONGT.md`, and the hier-goal sections of `CLAUDE.md`.

Claims that do NOT survive:
- "MapWM-Hier is best at OOD explore length by a wide margin" / the +0.09-0.10
  super-additive 2x2 interaction.
- "Hierarchy is net harmful on the fixed task" (the reversal was measured on a
  shortcut too).
- "PoPE-Flat holds 0.936 across a 32x length extrapolation." This is now
  EXPLAINED rather than impressive: continuing a deterministic order-3 pattern
  is length-invariant by construction, which is exactly why it was suspiciously
  flat and why its seed spread was +/-0.002.

Also suspect for the same reason (BFS demonstrations, never checked beyond
order 1): `GOAL_DIRECTED_RESULTS.md`, `PROBE_GOAL_RESULTS.md`,
`GOAL_CLOSEDLOOP_RESULTS.md`.

### Rule going forward

A demonstration-based task must be validated against an n-gram predictor fit to
the ACTION SEQUENCE ALONE at order 1..5, not a single copy-previous baseline.
Report that number next to every headline. Teacher-forced action-match accuracy
on optimal-planner demonstrations is intrinsically vulnerable: the planner's
output is structured, and structure is predictable without the state.

## Addendum: why the two task versions gave opposite hierarchy verdicts

They did not disagree about hierarchy. Per-seed re-evaluation of the ORIGINAL
(raw-BFS) checkpoints at T_explore=128, measured 2026-08-09:

| variant | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| MapWM-Flat | 0.526 | 0.942 | 0.470 | 0.646 ± 0.258 |
| MapWM-Hier | 0.869 | 0.891 | 0.934 | 0.898 ± 0.033 |

MapWM-Flat is **bimodal**: one seed trains, two collapse at OOD explore length.
The reported "hierarchy wins 0.907 vs 0.656" was therefore mostly two failed flat
runs, not an architectural effect. On the interleaved task all three flat seeds
train (0.911 ± 0.007), which is the entire reason its mean "improved" -- the
model did not get better, the runs stopped failing.

**The decisive point: the copy-previous-action baseline on the original task is
0.969, and EVERY model is below it** -- best flat seed 0.942, all of hierarchy
0.898. The original table ranked models by how far short of a one-line heuristic
they fell.

So the sequence is: original table = noise below a trivial baseline;
interleaved table = order-3 pattern continuation; neither = navigation
(closed-loop 0.013-0.037 vs a 0.010 random floor). Both are void for the same
underlying reason, and neither supports a claim about hierarchy in either
direction.
