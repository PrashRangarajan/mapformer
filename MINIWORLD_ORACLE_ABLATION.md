# MiniWorld fresh-map — context-destruction ablation (validity gate F2)

A genuine in-context map must COLLAPSE when context is destroyed. chance nb_acc = 1/16 = 0.0625. Held-out T=512.

| variant | enc | seed | intact | obs-shuffle | action-shuffle | marginal | verdict |
|---|---|---|---|---|---|---|---|
| Vanilla | allo | 0 | 0.280 | 0.010 | 0.024 | 0.070 | PASS |
| Vanilla | allo | 1 | 0.271 | 0.023 | 0.047 | 0.070 | PASS |
| Vanilla | allo | 2 | 0.302 | 0.008 | 0.015 | 0.070 | PASS |
| RoPE | allo | 0 | 0.511 | 0.083 | 0.059 | 0.070 | PASS |
| RoPE | allo | 1 | 0.485 | 0.054 | 0.043 | 0.070 | PASS |
| RoPE | allo | 2 | 0.506 | 0.052 | 0.047 | 0.070 | PASS |
| MapPoPE-Flat | allo | 0 | 0.166 | 0.027 | 0.029 | 0.070 | PASS |
| MapPoPE-Flat | allo | 1 | 0.268 | 0.041 | 0.023 | 0.070 | PASS |
| MapPoPE-Flat | allo | 2 | 0.261 | 0.044 | 0.035 | 0.070 | PASS |
| PoPE-Flat | allo | 0 | 0.363 | 0.051 | 0.032 | 0.070 | PASS |
| PoPE-Flat | allo | 1 | 0.371 | 0.046 | 0.030 | 0.070 | PASS |
| PoPE-Flat | allo | 2 | 0.356 | 0.040 | 0.028 | 0.070 | PASS |
| Vanilla | oracle | 0 | 0.540 | 0.058 | 0.026 | 0.070 | PASS |
| Vanilla | oracle | 1 | 0.403 | 0.037 | 0.034 | 0.070 | PASS |
| Vanilla | oracle | 2 | 0.402 | 0.041 | 0.020 | 0.070 | PASS |
| RoPE | oracle | 0 | 0.982 | 0.083 | 0.051 | 0.070 | PASS |
| RoPE | oracle | 1 | 0.981 | 0.080 | 0.038 | 0.070 | PASS |
| RoPE | oracle | 2 | 0.969 | 0.087 | 0.045 | 0.070 | PASS |
| MapPoPE-Flat | oracle | 0 | 0.343 | 0.041 | 0.018 | 0.070 | PASS |
| MapPoPE-Flat | oracle | 1 | 0.339 | 0.035 | 0.022 | 0.070 | PASS |
| MapPoPE-Flat | oracle | 2 | 0.291 | 0.026 | 0.027 | 0.070 | PASS |
| PoPE-Flat | oracle | 0 | 0.949 | 0.074 | 0.045 | 0.070 | PASS |
| PoPE-Flat | oracle | 1 | 0.967 | 0.078 | 0.050 | 0.070 | PASS |
| PoPE-Flat | oracle | 2 | 0.898 | 0.069 | 0.047 | 0.070 | PASS |
