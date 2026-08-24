# Compositional Match-Query — results (mean ± std over seeds)

Blind continuation in a repeated-motif world. chance = 0.0625. `exact` = path-integration matching; `cross` = path integration AND motif abstraction (the synergy target). Held-out env (seed=10000).

## cross_acc
| variant | TQ=256 | TQ=512 |
|---|---|---|
| Vanilla (n=6) | 0.112 ± 0.043 | 0.108 ± 0.050 |
| Hourglass_k2 (n=6) | 0.100 ± 0.017 | 0.097 ± 0.017 |
| Hourglass_CoarseIdx (n=6) | 0.182 ± 0.189 | 0.183 ± 0.189 |
| PlainFlat (n=6) | 0.117 ± 0.007 | 0.109 ± 0.010 |
| PlainHourglass (n=6) | 0.114 ± 0.006 | 0.104 ± 0.006 |

## exact_acc
| variant | TQ=256 | TQ=512 |
|---|---|---|
| Vanilla (n=6) | 0.396 ± 0.293 | 0.350 ± 0.245 |
| Hourglass_k2 (n=6) | 0.462 ± 0.307 | 0.414 ± 0.288 |
| Hourglass_CoarseIdx (n=6) | 0.316 ± 0.336 | 0.293 ± 0.347 |
| PlainFlat (n=6) | 0.193 ± 0.025 | 0.168 ± 0.008 |
| PlainHourglass (n=6) | 0.183 ± 0.021 | 0.155 ± 0.012 |

## all_acc
| variant | TQ=256 | TQ=512 |
|---|---|---|
| Vanilla (n=6) | 0.183 ± 0.093 | 0.172 ± 0.089 |
| Hourglass_k2 (n=6) | 0.194 ± 0.081 | 0.179 ± 0.075 |
| Hourglass_CoarseIdx (n=6) | 0.217 ± 0.228 | 0.210 ± 0.227 |
| PlainFlat (n=6) | 0.137 ± 0.008 | 0.124 ± 0.009 |
| PlainHourglass (n=6) | 0.131 ± 0.007 | 0.117 ± 0.007 |

