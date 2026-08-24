# Compositional Match-Query — results (mean ± std over seeds)

Blind continuation in a repeated-motif world. chance = 0.0625. `exact` = path-integration matching; `cross` = path integration AND motif abstraction (the synergy target). Held-out env (seed=10000).

## cross_acc
| variant | TQ=256 | TQ=512 |
|---|---|---|
| Vanilla (n=3) | 0.102 ± 0.001 | 0.098 ± 0.005 |
| Hourglass_k2 (n=3) | 0.118 ± 0.018 | 0.121 ± 0.025 |
| Hourglass_CoarseIdx (n=3) | 0.274 ± 0.218 | 0.271 ± 0.227 |
| PlainFlat (n=3) | 0.118 ± 0.005 | 0.117 ± 0.008 |
| PlainHourglass (n=3) | 0.115 ± 0.005 | 0.107 ± 0.010 |

## exact_acc
| variant | TQ=256 | TQ=512 |
|---|---|---|
| Vanilla (n=3) | 0.333 ± 0.285 | 0.308 ± 0.289 |
| Hourglass_k2 (n=3) | 0.385 ± 0.291 | 0.353 ± 0.314 |
| Hourglass_CoarseIdx (n=3) | 0.514 ± 0.427 | 0.499 ± 0.446 |
| PlainFlat (n=3) | 0.204 ± 0.030 | 0.160 ± 0.008 |
| PlainHourglass (n=3) | 0.200 ± 0.025 | 0.160 ± 0.004 |

## all_acc
| variant | TQ=256 | TQ=512 |
|---|---|---|
| Vanilla (n=3) | 0.159 ± 0.070 | 0.152 ± 0.073 |
| Hourglass_k2 (n=3) | 0.188 ± 0.068 | 0.178 ± 0.070 |
| Hourglass_CoarseIdx (n=3) | 0.337 ± 0.274 | 0.332 ± 0.290 |
| PlainFlat (n=3) | 0.140 ± 0.005 | 0.128 ± 0.009 |
| PlainHourglass (n=3) | 0.137 ± 0.003 | 0.121 ± 0.009 |

