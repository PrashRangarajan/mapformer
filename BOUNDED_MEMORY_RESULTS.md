# Bounded-memory test: how to spend a fixed read budget (M=128)

Retrieval task (revisit prediction), trained n_steps=256, clean.
BoundedFlat = all budget on recency. BoundedHier = M/2 recent + M/2
summaries spanning ALL history. Level15 = UNBOUNDED full attention (ceiling).

| Variant | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| BoundedFlat | 0.971±0.000 | 0.943±0.000 | 0.892±0.000 | 0.833±0.000 | 0.770±0.000 |
| BoundedHier | 0.984±0.004 | 0.924±0.001 | 0.866±0.000 | 0.814±0.000 | 0.752±0.000 |
| Level15 (unbounded, n=1) | 1.000 | 1.000 | 1.000 | 0.982 | 0.860 |
