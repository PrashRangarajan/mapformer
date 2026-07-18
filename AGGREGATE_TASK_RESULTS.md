# Aggregate-query task: flat Level15 vs HierAttn (clean, n_steps=256)

Windowed-majority obs-type (W_agg=128). Chance ~0.11. Contrast with the
RETRIEVAL task (HIER_ATTN_LONGT.md) where flat wins.

| Variant | train loss | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|---|
| Level15 | 0.6308 | 0.878 | 0.715 | 0.542 | 0.383 |
| HierAttn | 0.6931 | 0.823 | 0.741 | 0.628 | 0.531 |
