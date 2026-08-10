> **SUSPECT — planner-demonstration task (2026-08-09).** Scores next-action
> prediction on optimal-planner demonstrations. That family of task is
> self-predictable from the action stream alone: on hier-goal an n-gram scored
> 0.969 at order 1 and 0.971 at order 3, and every model there collapsed to
> ~0.02 closed-loop against a 0.010 random floor. This file has NOT been
> re-validated with an action-only n-gram control at orders 1-5, which is the
> check that would settle it.

# Nested-room (hierarchical space) task: flat Level15 vs HierAttn

One trained model, two metrics. revisit = needle/retrieval;
room_novel = infer the room THEME for a never-visited cell (spatial aggregate).
Ceiling on room_novel ~0.333 (1/theme_size); blind chance 0.0625.

## revisit
| Variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| Level15 | 0.999 | 0.994 | 0.971 | 0.916 |
| HierAttn | 0.966 | 0.931 | 0.874 | 0.798 |

## room_novel
| Variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| Level15 | 0.444 | 0.437 | 0.420 | 0.382 |
| HierAttn | 0.412 | 0.424 | 0.422 | 0.414 |

