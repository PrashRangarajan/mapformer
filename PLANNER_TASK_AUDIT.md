# Planner-task audit: n-grams on the ACTION STREAM ALONE

No model, no goal, no observations, no position -- just the scored action
sequence predicting itself. Far above chance => the task is solvable without
the capability it claims to test.

`hier_goal(control)` is a POSITIVE CONTROL and must come out ~0.97.

| task | chance | o1 | o2 | o3 | o4 | o5 | verdict |
|---|---|---|---|---|---|---|---|
| hier_goal(control) | 0.250 | 0.969 | 0.968 | 0.968 | 0.968 | 0.969 | **VOID** |
| goal | 0.250 | 0.969 | 0.968 | 0.967 | 0.967 | 0.966 | **VOID** |
| rooms_goal | 0.250 | 0.969 | 0.968 | 0.968 | 0.968 | 0.968 | **VOID** |
| rooms_maze | 0.250 | 0.791 | 0.792 | 0.793 | 0.793 | 0.792 | **VOID** |
| maze_varying | 0.250 | 0.650 | 0.645 | 0.643 | 0.623 | 0.567 | **VOID** |

Thresholds: **VOID** if any order exceeds chance+0.25, *suspect* if above chance+0.10.
