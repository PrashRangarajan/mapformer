# DELTA_FREE raw evaluation (pilot)

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | PoPE-Flat | PoPE-FreeDelta | MapPoPE-Flat | MapPoPE-FreeDelta | MapPoPE-FrozenDelta |
|---|---|---|---|---|---|
| 0 | 0.671 | 0.816 | 1.000 | 1.000 | 1.000 |

## T=512

| p_action_noise | PoPE-Flat | PoPE-FreeDelta | MapPoPE-Flat | MapPoPE-FreeDelta | MapPoPE-FrozenDelta |
|---|---|---|---|---|---|
| 0 | 0.623 | 0.525 | 0.985 | 0.969 | 0.937 |

## T=1024

| p_action_noise | PoPE-Flat | PoPE-FreeDelta | MapPoPE-Flat | MapPoPE-FreeDelta | MapPoPE-FrozenDelta |
|---|---|---|---|---|---|
| 0 | 0.599 | 0.401 | 0.939 | 0.926 | 0.801 |

