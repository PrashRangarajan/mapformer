# Match-Query results (n=3 seeds)

Blind continuation: explore with observations revealed, then continue with
them withheld and predict the observation at each cell. Scored at cells
visited during explore and non-blank; each cell scored once per episode.

Trained TE=512 TQ=256, 200 epochs. Held-out env (seed=10000).
**Chance 0.0625.** Gates in `MATCH_QUERY_GATES.md` (all at chance).

## Match accuracy

| variant | T_query=256 (train) | T_query=512 (OOD) |
|---|---|---|
| MapWM-Flat | 0.888 ± 0.140 | 0.902 ± 0.117 |
| MapWM-Hier | 0.786 ± 0.227 | 0.771 ± 0.220 |
| Plain-Flat | 0.153 ± 0.012 | 0.127 ± 0.010 |
| Plain-Hier | 0.155 ± 0.020 | 0.131 ± 0.004 |
| PoPE-Flat | 0.117 ± 0.011 | 0.109 ± 0.008 |
| MapPoPE-Hier | 0.847 ± 0.132 | 0.823 ± 0.155 |

## Match NLL (lower better)

| variant | T_query=256 | T_query=512 |
|---|---|---|
| MapWM-Flat | 0.376 ± 0.441 | 0.341 ± 0.382 |
| MapWM-Hier | 0.745 ± 0.778 | 0.817 ± 0.777 |
| Plain-Flat | 2.578 ± 0.042 | 2.635 ± 0.016 |
| Plain-Hier | 2.590 ± 0.057 | 2.636 ± 0.006 |
| PoPE-Flat | 2.668 ± 0.041 | 2.692 ± 0.022 |
| MapPoPE-Hier | 0.510 ± 0.443 | 0.576 ± 0.498 |
