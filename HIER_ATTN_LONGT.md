# Hierarchical attention vs flat attention (Level 1.5), long-T clean

Both trained at n_steps=256, seed 0. Same params; differ only in attention.

Level15: final train loss 0.0013
HierAttn: final train loss 0.1197

## Accuracy
| Variant | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| Level15 | 1.000 | 1.000 | 1.000 | 0.983 | 0.861 |
| HierAttn | 0.974 | 0.938 | 0.892 | 0.833 | 0.769 |

## NLL
| Variant | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| Level15 | 0.000 | 0.000 | 0.000 | 0.059 | 0.781 |
| HierAttn | 0.109 | 0.248 | 0.440 | 0.681 | 0.956 |
