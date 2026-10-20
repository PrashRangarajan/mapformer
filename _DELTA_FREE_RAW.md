# DELTA_FREE raw evaluation (batch)

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | PoPE-Flat | PoPE-FreeDelta | MapPoPE-Flat | MapPoPE-FreeDelta | MapPoPE-FrozenDelta |
|---|---|---|---|---|---|
| 0 | 0.649 ± 0.063 | 0.771 ± 0.043 | 0.999 ± 0.002 | 0.981 ± 0.041 | 0.999 ± 0.001 |

## T=512

| p_action_noise | PoPE-Flat | PoPE-FreeDelta | MapPoPE-Flat | MapPoPE-FreeDelta | MapPoPE-FrozenDelta |
|---|---|---|---|---|---|
| 0 | 0.615 ± 0.040 | 0.536 ± 0.108 | 0.980 ± 0.019 | 0.901 ± 0.153 | 0.929 ± 0.067 |

## T=1024

| p_action_noise | PoPE-Flat | PoPE-FreeDelta | MapPoPE-Flat | MapPoPE-FreeDelta | MapPoPE-FrozenDelta |
|---|---|---|---|---|---|
| 0 | 0.599 ± 0.023 | 0.442 ± 0.134 | 0.933 ± 0.029 | 0.831 ± 0.168 | 0.832 ± 0.115 |

