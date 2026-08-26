# MiniWorld fresh-map — context-destruction ablation (validity gate F2)

A genuine in-context map must COLLAPSE when context is destroyed. chance nb_acc = 1/16 = 0.0625. Held-out T=512.

| variant | enc | seed | intact | obs-shuffle | action-shuffle | marginal | verdict |
|---|---|---|---|---|---|---|---|
| Vanilla | raw | 0 | 0.355 | 0.043 | 0.022 | 0.076 | PASS |
| Vanilla | raw | 1 | 0.319 | 0.014 | 0.017 | 0.076 | PASS |
| Vanilla | raw | 2 | 0.237 | 0.035 | 0.028 | 0.076 | PASS |
| RoPE | raw | 0 | 0.396 | 0.053 | 0.022 | 0.076 | PASS |
| RoPE | raw | 1 | 0.401 | 0.056 | 0.064 | 0.076 | PASS |
| RoPE | raw | 2 | 0.404 | 0.041 | 0.028 | 0.076 | PASS |
| MapPoPE-Flat | raw | 0 | 0.309 | 0.045 | 0.041 | 0.076 | PASS |
| MapPoPE-Flat | raw | 1 | 0.300 | 0.029 | 0.043 | 0.076 | PASS |
| MapPoPE-Flat | raw | 2 | 0.327 | 0.043 | 0.043 | 0.076 | PASS |
| PoPE-Flat | raw | 0 | 0.395 | 0.055 | 0.051 | 0.076 | PASS |
| PoPE-Flat | raw | 1 | 0.382 | 0.054 | 0.055 | 0.076 | PASS |
| PoPE-Flat | raw | 2 | 0.374 | 0.055 | 0.039 | 0.076 | PASS |
| Vanilla | allo | 0 | 0.281 | 0.011 | 0.022 | 0.076 | PASS |
| Vanilla | allo | 1 | 0.275 | 0.020 | 0.048 | 0.076 | PASS |
| Vanilla | allo | 2 | 0.303 | 0.007 | 0.016 | 0.076 | PASS |
| RoPE | allo | 0 | 0.512 | 0.088 | 0.059 | 0.076 | PASS |
| RoPE | allo | 1 | 0.490 | 0.054 | 0.038 | 0.076 | PASS |
| RoPE | allo | 2 | 0.509 | 0.052 | 0.042 | 0.076 | PASS |
| MapPoPE-Flat | allo | 0 | 0.168 | 0.028 | 0.030 | 0.076 | PASS |
| MapPoPE-Flat | allo | 1 | 0.267 | 0.041 | 0.025 | 0.076 | PASS |
| MapPoPE-Flat | allo | 2 | 0.261 | 0.034 | 0.032 | 0.076 | PASS |
| PoPE-Flat | allo | 0 | 0.361 | 0.046 | 0.032 | 0.076 | PASS |
| PoPE-Flat | allo | 1 | 0.367 | 0.048 | 0.033 | 0.076 | PASS |
| PoPE-Flat | allo | 2 | 0.353 | 0.035 | 0.029 | 0.076 | PASS |
