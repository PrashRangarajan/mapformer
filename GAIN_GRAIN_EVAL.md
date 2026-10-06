# Gain granularity between MapEM and MapPoPE, paper torus, rank 2, trained at T=128

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | Vanilla | MapPoPE-Pair | GainScalar | GainMod4 | VanillaEM | VanillaEM_NonNeg |
|---|---|---|---|---|---|---|
| 0 | 0.990 ± 0.021 | 0.999 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.997 ± 0.010 | 0.982 ± 0.028 |

## T=512

| p_action_noise | Vanilla | MapPoPE-Pair | GainScalar | GainMod4 | VanillaEM | VanillaEM_NonNeg |
|---|---|---|---|---|---|---|
| 0 | 0.939 ± 0.038 | 0.976 ± 0.016 | 0.986 ± 0.015 | 0.980 ± 0.016 | 0.932 ± 0.086 | 0.865 ± 0.122 |

## T=1024

| p_action_noise | Vanilla | MapPoPE-Pair | GainScalar | GainMod4 | VanillaEM | VanillaEM_NonNeg |
|---|---|---|---|---|---|---|
| 0 | 0.832 ± 0.081 | 0.927 ± 0.026 | 0.942 ± 0.034 | 0.932 ± 0.033 | 0.829 ± 0.102 | 0.717 ± 0.172 |

