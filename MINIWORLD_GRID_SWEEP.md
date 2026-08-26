# MiniWorld grid sweep -- attention substitutability (fresh-map, oracle recode)

Vanilla=path-int, RoPE=index. Prediction: path-int - index climbs from -0.53 (grid 8) toward positive as the grid grows and attention can no longer integrate position over the longer revisit distances. chance 0.0625.

## T=512

| grid | Vanilla (path-int) | RoPE (index) | effect (V-R) |
|---|---|---|---|
| 8 | 0.448 | 0.977 | **-0.529** (n=3) |
| 16 | 0.886 | 0.738 | **+0.148** (n=3) |
| 24 | 0.694 | 0.618 | **+0.076** (n=3) |
| 32 | 0.703 | 0.615 | **+0.087** (n=3) |

## T=1024

| grid | Vanilla (path-int) | RoPE (index) | effect (V-R) |
|---|---|---|---|
| 8 | 0.289 | 0.543 | **-0.253** (n=3) |
| 16 | 0.712 | 0.379 | **+0.333** (n=3) |
| 24 | 0.506 | 0.298 | **+0.208** (n=3) |
| 32 | 0.505 | 0.333 | **+0.172** (n=3) |

