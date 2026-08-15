> **VOID -- planner-demonstration shortcut (2026-08-09).** This file's task
> (`RoomsGoalWorld`) scores next-action prediction on optimal-planner demonstrations.
> An n-gram fitted on the ACTION STREAM ALONE -- no goal, no observations, no
> position, no model -- scores **0.969** at order 1 against a chance of **0.250**
> (`PLANNER_TASK_AUDIT.md`). The task is solvable without the capability it
> claims to test, so no number here measures navigation. Same failure that
> invalidated hier-goal, where closed-loop success was 0.013-0.037 against a
> 0.010 random floor.

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

