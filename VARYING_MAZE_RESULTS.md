> **SUSPECT — planner-demonstration task (2026-08-09).** Scores next-action
> prediction on optimal-planner demonstrations. That family of task is
> self-predictable from the action stream alone: on hier-goal an n-gram scored
> 0.969 at order 1 and 0.971 at order 3, and every model there collapsed to
> ~0.02 closed-loop against a 0.010 random floor. This file has NOT been
> re-validated with an action-only n-gram control at orders 1-5, which is the
> check that would settle it.

# VARYING maze: build a cognitive map, then plan on it

Fresh maze + landmarks every episode -> memorisation impossible by construction.
(Fixed-maze version collapsed 0.94 -> 0.68 on a novel maze: it had memorised.)
Chance = 0.25. Greedy wall-ignoring policy = 0.73, so >0.73 means the model
is USING the map it built during exploration.

| Variant | len 1-5 | len 6-10 | len 11-15 | len 16+ | all |
|---|---|---|---|---|---|
| Level15 (n=3) | 0.446±0.008 | 0.482±0.001 | 0.534±0.007 | 0.532±0.001 | 0.503±0.003 |
| HierAttn (n=3) | 0.454±0.008 | 0.486±0.005 | 0.531±0.006 | 0.541±0.004 | 0.506±0.005 |
| HierAttn_LocalOnly (n=1) | 0.424 | 0.481 | 0.530 | 0.534 | 0.499 |
| HierAttn_CoarseOnly (n=1) | 0.431 | 0.472 | 0.524 | 0.538 | 0.495 |
