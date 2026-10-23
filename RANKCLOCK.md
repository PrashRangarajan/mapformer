# RANKCLOCK (G-1): rank clock routes, trained and tested at T=1024

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=512

| p_action_noise | Vanilla | Vanilla_r2clockxy | Vanilla_r2clock | Vanilla_r2DF | Vanilla_r2AO | Vanilla_r4mi |
|---|---|---|---|---|---|---|
| 0 | 0.954 ± 0.042 | 0.984 ± 0.037 | 0.979 ± 0.023 | 1.000 ± 0.000 | 0.990 ± 0.011 | 1.000 ± 0.000 |

## T=1024

| p_action_noise | Vanilla | Vanilla_r2clockxy | Vanilla_r2clock | Vanilla_r2DF | Vanilla_r2AO | Vanilla_r4mi |
|---|---|---|---|---|---|---|
| 0 | 0.918 ± 0.053 | 0.972 ± 0.046 | 0.951 ± 0.045 | 1.000 ± 0.000 | 0.962 ± 0.029 | 1.000 ± 0.001 |

## T=2048

| p_action_noise | Vanilla | Vanilla_r2clockxy | Vanilla_r2clock | Vanilla_r2DF | Vanilla_r2AO | Vanilla_r4mi |
|---|---|---|---|---|---|---|
| 0 | 0.851 ± 0.056 | 0.927 ± 0.059 | 0.891 ± 0.075 | 1.000 ± 0.001 | 0.890 ± 0.072 | 0.986 ± 0.018 |

