> **SUSPECT — planner-demonstration task (2026-08-09).** Scores next-action
> prediction on optimal-planner demonstrations. That family of task is
> self-predictable from the action stream alone: on hier-goal an n-gram scored
> 0.969 at order 1 and 0.971 at order 3, and every model there collapsed to
> ~0.02 closed-loop against a 0.010 random floor. This file has NOT been
> re-validated with an action-only n-gram control at orders 1-5, which is the
> check that would settle it.

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

