# Sign ablation, raw evaluation

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | Signed_r4 | Abs_r4 | Pos_r4 | CARoPE_r4 | Vanilla_r4 | RoPE |
|---|---|---|---|---|---|---|
| 0 | 1.000 ± 0.000 | 0.946 ± 0.070 | 0.977 ± 0.035 | 0.900 ± 0.138 | 1.000 ± 0.000 | 0.799 ± 0.018 |

## T=512

| p_action_noise | Signed_r4 | Abs_r4 | Pos_r4 | CARoPE_r4 | Vanilla_r4 | RoPE |
|---|---|---|---|---|---|---|
| 0 | 0.978 ± 0.014 | 0.675 ± 0.040 | 0.798 ± 0.080 | 0.809 ± 0.133 | 0.984 ± 0.009 | 0.449 ± 0.083 |

## T=1024

| p_action_noise | Signed_r4 | Abs_r4 | Pos_r4 | CARoPE_r4 | Vanilla_r4 | RoPE |
|---|---|---|---|---|---|---|
| 0 | 0.922 ± 0.027 | 0.558 ± 0.027 | 0.584 ± 0.080 | 0.645 ± 0.090 | 0.927 ± 0.015 | 0.345 ± 0.118 |

