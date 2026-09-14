# PAPER2X2 raw evaluation

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | RoPE | PoPE-Flat | Vanilla | MapPoPE-Flat | Vanilla_r4 | MapPoPE_r4 |
|---|---|---|---|---|---|---|
| 0 | 0.805 ± 0.012 | 0.679 ± 0.058 | 0.971 ± 0.043 | 0.999 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.001 |

## T=512

| p_action_noise | RoPE | PoPE-Flat | Vanilla | MapPoPE-Flat | Vanilla_r4 | MapPoPE_r4 |
|---|---|---|---|---|---|---|
| 0 | 0.470 ± 0.093 | 0.639 ± 0.046 | 0.915 ± 0.052 | 0.974 ± 0.011 | 0.985 ± 0.006 | 0.982 ± 0.018 |

## T=1024

| p_action_noise | RoPE | PoPE-Flat | Vanilla | MapPoPE-Flat | Vanilla_r4 | MapPoPE_r4 |
|---|---|---|---|---|---|---|
| 0 | 0.374 ± 0.135 | 0.607 ± 0.033 | 0.777 ± 0.094 | 0.921 ± 0.023 | 0.928 ± 0.012 | 0.941 ± 0.028 |

