# GAIN_GRAIN dropout-scale re-score (declared secondary)

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | Vanilla | MapPoPE-Pair | GainScalar | GainMod4 | VanillaEM | VanillaEM_NonNeg |
|---|---|---|---|---|---|---|
| 0 | 0.990 ± 0.021 | 1.000 ± 0.001 | 1.000 ± 0.001 | 1.000 ± 0.001 | 0.997 ± 0.010 | 0.982 ± 0.028 |

