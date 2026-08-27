# MiniWorld grid sweep -- HIERARCHY (Hourglass k=2), fresh-map oracle recode

MapWM-Hier=path-int+hier, Plain-Hier=index+hier (both 2.38M, internally matched). Question: does the hier-pair crossover grid sit HIGHER than the flat-pair (Vanilla-RoPE) crossover, i.e. does hierarchy let index substitute over longer distances? chance 0.0625.

## T=512

| grid | MapWM-Hier (path-int) | Plain-Hier (index) | effect (H-I) |
|---|---|---|---|
| 8 | 0.683 | 0.974 | **-0.292** (n=3) |
| 16 | 0.943 | 0.597 | **+0.346** (n=3) |
| 24 | 0.978 | 0.585 | **+0.393** (n=3) |
| 32 | 0.986 | 0.598 | **+0.388** (n=3) |

## T=1024

| grid | MapWM-Hier (path-int) | Plain-Hier (index) | effect (H-I) |
|---|---|---|---|
| 8 | 0.607 | 0.628 | **-0.022** (n=3) |
| 16 | 0.883 | 0.395 | **+0.488** (n=3) |
| 24 | 0.876 | 0.396 | **+0.480** (n=3) |
| 32 | 0.927 | 0.432 | **+0.495** (n=3) |

