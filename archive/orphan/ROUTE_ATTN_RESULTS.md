# Identity-preserving routing vs pooled hierarchy vs flat

Retrieval task (revisit prediction), clean, trained n_steps=256, all arms
trained in the SAME batch. Same param count (253,973) for every arm.

RouteAttn: coarse top-2 chunk SELECTION (pooled keys used only to route),
then FULL-RESOLUTION read of the selected chunks -- nothing read from a pool.
HierAttn: reads from mean-pooled summaries -- token identity destroyed.

| Variant | T=256 | T=512 | T=1024 | T=2048 | T=4096 |
|---|---|---|---|---|---|
| Level15 | 1.000±0.000 | 0.999±0.002 | 0.994±0.004 | 0.953±0.009 | 0.849±0.011 |
| RouteAttn | 0.915±0.091 | 0.890±0.088 | 0.846±0.085 | 0.778±0.078 | 0.708±0.069 |
| HierAttn | 0.970±0.002 | 0.937±0.003 | 0.890±0.004 | 0.829±0.006 | 0.764±0.007 |
| RouteAttn_NoBias | 0.995±0.002 | 0.985±0.006 | 0.956±0.012 | 0.900±0.018 | 0.832±0.018 |
| RouteAttn_K4 | 0.984 | 0.947 | 0.881 | 0.790 | 0.698 |
