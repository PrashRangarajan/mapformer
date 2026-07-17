# Attention hierarchy at MULTI-ENV transfer (held-out envs, seed 0)

Matched flat Level15 vs HierAttn; 50 train / 50 held-out envs.

| Variant | Config | train loss | train acc | held T=128 | held T=512 OOD |
|---|---|---|---|---|---|
| Level15 | clean | 0.0378 | 0.991 | 0.990 | 0.947 |
| HierAttn | clean | 0.0425 | 0.993 | 0.994 | 0.947 |
| Level15 | lm200 | 0.0091 | 0.999 | 0.998 | 0.993 |
| HierAttn | lm200 | 0.0791 | 0.986 | 0.989 | 0.941 |
