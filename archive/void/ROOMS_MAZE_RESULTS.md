> **VOID -- planner-demonstration shortcut (2026-08-09).** This file's task
> (`RoomsMazeWorld`) scores next-action prediction on optimal-planner demonstrations.
> An n-gram fitted on the ACTION STREAM ALONE -- no goal, no observations, no
> position, no model -- scores **0.791** at order 1 against a chance of **0.250**
> (`PLANNER_TASK_AUDIT.md`). The task is solvable without the capability it
> claims to test, so no number here measures navigation. Same failure that
> invalidated hier-goal, where closed-loop success was 0.013-0.037 against a
> 0.010 random floor.

# Hierarchical planning: rooms MAZE (spanning-tree doors)

Greedy-optimal only 0.706 here (vs 1.000 open-plan): real route planning.
Action accuracy at BFS-optimal steps, chance=0.25, bucketed by room distance.
Prediction: hierarchy's advantage GROWS with room distance.

## T_explore=64 (train)

| Variant | d=1 | d=2 | d=3 | d=4 | all |
|---|---|---|---|---|---|
| Level15 (n=3) | 0.926±0.002 | 0.931±0.007 | 0.944±0.005 | 0.944±0.006 | 0.939±0.005 |
| HierAttn (n=3) | 0.927±0.003 | 0.933±0.002 | 0.943±0.002 | 0.943±0.002 | 0.939±0.001 |
| HierAttn_LocalOnly (n=1) | 0.934 | 0.934 | 0.947 | 0.951 | 0.944 |
| HierAttn_CoarseOnly (n=1) | 0.743 | 0.767 | 0.773 | 0.778 | 0.770 |

## T_explore=128 (OOD)

| Variant | d=1 | d=2 | d=3 | d=4 | all |
|---|---|---|---|---|---|
| Level15 (n=3) | 0.907±0.012 | 0.928±0.001 | 0.940±0.004 | 0.947±0.004 | 0.938±0.004 |
| HierAttn (n=3) | 0.909±0.004 | 0.930±0.000 | 0.944±0.001 | 0.950±0.001 | 0.940±0.001 |
| HierAttn_LocalOnly (n=1) | 0.919 | 0.931 | 0.945 | 0.956 | 0.943 |
| HierAttn_CoarseOnly (n=1) | 0.762 | 0.779 | 0.806 | 0.810 | 0.799 |

