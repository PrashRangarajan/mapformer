# Aggregate-task extras: ablations, long-T, training-length control

## Component ablation + long-T (trained n_steps=256)

| Variant | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| Level15 | 0.878 | 0.715 | 0.542 | 0.383 | 0.290 |
| HierAttn | 0.823 | 0.741 | 0.628 | 0.531 | 0.475 |
| HierAttn_CoarseOnly | 0.617 | 0.631 | 0.586 | 0.522 | 0.434 |
| HierAttn_LocalOnly | 0.593 | 0.520 | 0.480 | 0.453 | 0.427 |

## Control: flat Level15 trained at n_steps=512 (does flat just need longer training?)

| Variant | T=512 | T=1024 | T=2048 |
|---|---|---|---|
| Level15 (train n_steps=512) | 0.746 | 0.629 | 0.539 |
