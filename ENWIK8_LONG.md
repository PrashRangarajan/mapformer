# enwik8 — 36k iters, rank 4 (both 12k caveats fixed)

12k run was budget-limited (every arm still improving; remaining gain > the between-model spread) and used r=2 where MapFormer's own language run uses r=4. Both fixed. seq 512, bs 16, lr 2e-4, param-matched to 0.03%. **Lower is better.** n=1.

| model | position | encoding | val bpc | vs RoPE | 12k value | note |
|---|---|---|---|---|---|---|
| RoPE | index | RoPE | **1.3817** |  | 1.5221 | baseline |
| PoPE-Flat | index | PoPE | **1.3746** | -0.0072 | 1.5303 |  |
| Vanilla_r4 | path-int | RoPE | **1.3758** | -0.0059 | 1.5505 | r=4 |
| MapPoPE-Flat_r4 | path-int | PoPE | **1.3723** | -0.0094 | 1.5193 | **combination**, r=4 |

Final-checkpoint slopes (negative = still improving; if these are still
large the run is STILL budget-limited and the ordering is not settled):
- RoPE: -0.0032 per 1000 iters
- PoPE-Flat: -0.0036 per 1000 iters
- Vanilla_r4: -0.0081 per 1000 iters
- MapPoPE-Flat_r4: -0.0020 per 1000 iters
