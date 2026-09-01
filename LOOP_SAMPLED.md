# Can the loop's length trade-off be trained away?

A model trained at a FIXED 4 passes peaks at 4 passes on its training length and
at 2 on a 4x longer one -- every pass past the second hurts out of distribution.
`LoopedSampled` draws the count from {2..6} each training batch instead, so the
count becomes a runtime knob the model has been trained across.

Torus paper task, held-out map, evaluated under the noise it trained on.

## p_action_noise = 0

**Vanilla** (no loop): T=128: 0.965 ± 0.058 | T=512: 0.892 ± 0.068 | T=1024: 0.767 ± 0.091

**Looped** — accuracy by (sequence length × loops at eval)

| loops at eval | T=128 | T=512 | T=1024 |
|---|---|---|---|
| 1 | 0.821 ± 0.191 | 0.659 ± 0.122 | 0.573 ± 0.055 |
| 2 | 0.987 ± 0.028 | 0.823 ± 0.108 | 0.651 ± 0.066 |
| 3 | 0.999 ± 0.001 | 0.821 ± 0.123 | 0.646 ± 0.081 |
| 4 | 1.000 ± 0.000 | 0.816 ± 0.132 | 0.642 ± 0.089 |
| 6 | 1.000 ± 0.001 | 0.811 ± 0.139 | 0.639 ± 0.096 |

best count per length — T=128: 4 loops (1.000) · T=512: 2 loops (0.823) · T=1024: 2 loops (0.651)

**LoopedSampled** — accuracy by (sequence length × loops at eval)

| loops at eval | T=128 | T=512 | T=1024 |
|---|---|---|---|
| 1 | 0.998 ± 0.003 | 0.914 ± 0.043 | 0.732 ± 0.068 |
| 2 | 0.999 ± 0.001 | 0.915 ± 0.058 | 0.736 ± 0.082 |
| 3 | 0.999 ± 0.001 | 0.904 ± 0.071 | 0.726 ± 0.094 |
| 4 | 0.999 ± 0.001 | 0.898 ± 0.078 | 0.719 ± 0.099 |
| 6 | 0.999 ± 0.001 | 0.890 ± 0.086 | 0.709 ± 0.102 |

best count per length — T=128: 6 loops (0.999) · T=512: 2 loops (0.915) · T=1024: 2 loops (0.736)

## p_action_noise = 0.1

**Vanilla** (no loop): T=128: 0.762 ± 0.019 | T=512: 0.693 ± 0.029 | T=1024: 0.619 ± 0.039

**Looped** — accuracy by (sequence length × loops at eval)

| loops at eval | T=128 | T=512 | T=1024 |
|---|---|---|---|
| 1 | 0.444 ± 0.087 | 0.437 ± 0.088 | 0.440 ± 0.075 |
| 2 | 0.670 ± 0.072 | 0.577 ± 0.045 | 0.524 ± 0.026 |
| 3 | 0.876 ± 0.016 | 0.678 ± 0.078 | 0.582 ± 0.097 |
| 4 | 0.892 ± 0.018 | 0.678 ± 0.087 | 0.585 ± 0.104 |
| 6 | 0.877 ± 0.017 | 0.660 ± 0.089 | 0.580 ± 0.096 |

best count per length — T=128: 4 loops (0.892) · T=512: 4 loops (0.678) · T=1024: 4 loops (0.585)

**LoopedSampled** — accuracy by (sequence length × loops at eval)

| loops at eval | T=128 | T=512 | T=1024 |
|---|---|---|---|
| 1 | 0.629 ± 0.100 | 0.599 ± 0.064 | 0.564 ± 0.048 |
| 2 | 0.856 ± 0.012 | 0.694 ± 0.075 | 0.600 ± 0.082 |
| 3 | 0.873 ± 0.011 | 0.695 ± 0.075 | 0.595 ± 0.086 |
| 4 | 0.878 ± 0.010 | 0.692 ± 0.071 | 0.592 ± 0.085 |
| 6 | 0.878 ± 0.010 | 0.689 ± 0.069 | 0.589 ± 0.082 |

best count per length — T=128: 6 loops (0.878) · T=512: 3 loops (0.695) · T=1024: 2 loops (0.600)

