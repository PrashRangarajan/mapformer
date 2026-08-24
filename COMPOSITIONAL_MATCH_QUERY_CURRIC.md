# Compositional Match-Query — results (mean ± std over seeds)

Blind continuation in a repeated-motif world. chance = 0.0625. `exact` = path-integration matching; `cross` = path integration AND motif abstraction (the synergy target). Held-out env (seed=10000).

## cross_acc
| variant | TQ=256 | TQ=512 |
|---|---|---|
| Vanilla (n=6) | 0.153 ± 0.128 | 0.154 ± 0.145 |
| Hourglass_k2 (n=6) | 0.177 ± 0.174 | 0.180 ± 0.184 |
| Hourglass_CoarseIdx (n=6) | 0.227 ± 0.188 | 0.231 ± 0.183 |
| PlainFlat (n=6) | 0.115 ± 0.005 | 0.109 ± 0.009 |
| PlainHourglass (n=6) | 0.113 ± 0.006 | 0.107 ± 0.004 |

## exact_acc
| variant | TQ=256 | TQ=512 |
|---|---|---|
| Vanilla (n=6) | 0.607 ± 0.284 | 0.559 ± 0.281 |
| Hourglass_k2 (n=6) | 0.778 ± 0.179 | 0.729 ± 0.217 |
| Hourglass_CoarseIdx (n=6) | 0.709 ± 0.304 | 0.681 ± 0.315 |
| PlainFlat (n=6) | 0.185 ± 0.025 | 0.158 ± 0.013 |
| PlainHourglass (n=6) | 0.184 ± 0.028 | 0.160 ± 0.010 |

## all_acc
| variant | TQ=256 | TQ=512 |
|---|---|---|
| Vanilla (n=6) | 0.269 ± 0.155 | 0.258 ± 0.166 |
| Hourglass_k2 (n=6) | 0.334 ± 0.161 | 0.320 ± 0.177 |
| Hourglass_CoarseIdx (n=6) | 0.351 ± 0.199 | 0.346 ± 0.202 |
| PlainFlat (n=6) | 0.133 ± 0.008 | 0.121 ± 0.008 |
| PlainHourglass (n=6) | 0.131 ± 0.008 | 0.120 ± 0.006 |

