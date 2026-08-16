# Family-tree (non-commutative relational structure) — results

The task MapFormer's appendix B.2.2 motivates and never runs: mother and
father do not commute. 8 relational actions, scored at revisited nodes,
n=3 seeds. Trained at 64 steps, also evaluated at 128 (OOD length).

**chance 0.125; effective floor is the HUB baseline 0.163** — shallow nodes
are revisited more, so answering with the most-visited node's observation
already scores 0.163. Read every number against 0.163.

| variant | n_steps=64 (train) | n_steps=128 (OOD) |
|---|---|---|
| MapEM-NC-L (non-commutative, linear) | 0.720 ± 0.011 | 0.671 ± 0.006 |
| MapEM-NC-NL (non-commutative, MLP) | 0.729 ± 0.010 | 0.672 ± 0.012 |
| MapEM single-p0 (COMMUTATIVE control) | 0.715 ± 0.008 | 0.659 ± 0.014 |
| Plain-Flat (index position, no PI) | 0.600 ± 0.011 | 0.550 ± 0.031 |

## Per seed (n_steps=64)

| variant | s0 | s1 | s2 |
|---|---|---|---|
| MapEM-NC-L (non-commutative, linear) | 0.713 | 0.733 | 0.714 |
| MapEM-NC-NL (non-commutative, MLP) | 0.720 | 0.740 | 0.726 |
| MapEM single-p0 (COMMUTATIVE control) | 0.712 | 0.725 | 0.709 |
| Plain-Flat (index position, no PI) | 0.589 | 0.610 | 0.603 |
