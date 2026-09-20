# Hierarchical goal-directed navigation (rooms + distant goals)

Action-prediction accuracy at BFS-optimal navigate steps. Chance = 0.25.
Bucketed by ROOM DISTANCE (room-hops to goal). Prediction: hierarchy's
advantage should GROW with room distance.

## T_explore=64 (train length)

| Variant | d=1 | d=2 | d=3 | d=4 | all |
|---|---|---|---|---|---|
| Level15 (n=3) | 0.889±0.003 | 0.937±0.001 | 0.958±0.001 | 0.967±0.000 | 0.953±0.001 |
| HierAttn (n=3) | 0.887±0.003 | 0.940±0.002 | 0.959±0.000 | 0.967±0.000 | 0.953±0.000 |
| HierAttn_LocalOnly (n=1) | 0.882 | 0.936 | 0.958 | 0.968 | 0.953 |
| HierAttn_CoarseOnly (n=1) | 0.786 | 0.874 | 0.834 | 0.857 | 0.846 |
| Level15_train128 (n=1) | 0.888 | 0.930 | 0.956 | 0.967 | 0.951 |

## T_explore=128 (OOD explore length)

| Variant | d=1 | d=2 | d=3 | d=4 | all |
|---|---|---|---|---|---|
| Level15 (n=3) | 0.887±0.003 | 0.929±0.002 | 0.956±0.004 | 0.960±0.007 | 0.948±0.004 |
| HierAttn (n=3) | 0.885±0.002 | 0.928±0.002 | 0.957±0.002 | 0.963±0.001 | 0.950±0.002 |
| HierAttn_LocalOnly (n=1) | 0.873 | 0.928 | 0.957 | 0.965 | 0.950 |
| HierAttn_CoarseOnly (n=1) | 0.878 | 0.905 | 0.935 | 0.947 | 0.930 |
| Level15_train128 (n=1) | 0.891 | 0.929 | 0.961 | 0.963 | 0.952 |

