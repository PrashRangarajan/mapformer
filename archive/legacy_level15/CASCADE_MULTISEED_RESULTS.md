# Level15Cascade vs Level15 — multi-seed length generalization
Generated: Wed Jul 15 05:43:10 PM PDT 2026

############ clean ############
# Long-Sequence Evaluation
Config: clean, n_landmarks: 0, lengths: [128, 512, 2048]
Seeds: [0, 1, 2]
## Accuracy
| Variant | T=128 | T=512 | T=2048 |
|---------|-------|-------|-------|
| Level15 | 1.000±0.000 | 0.994±0.003 | 0.879±0.013 |
| Level15Cascade | 0.999±0.001 | 0.989±0.012 | 0.881±0.044 |
## NLL
| Variant | T=128 | T=512 | T=2048 |
|---------|-------|-------|-------|
| Level15 | 0.000 | 0.032 | 0.705 |
| Level15Cascade | 0.002 | 0.066 | 0.795 |

############ noise ############
# Long-Sequence Evaluation
Config: noise, n_landmarks: 0, lengths: [128, 512, 2048]
Seeds: [0, 1, 2]
## Accuracy
| Variant | T=128 | T=512 | T=2048 |
|---------|-------|-------|-------|
| Level15 | 0.948±0.019 | 0.866±0.026 | 0.676±0.043 |
| Level15Cascade | 0.959±0.005 | 0.881±0.016 | 0.700±0.033 |
## NLL
| Variant | T=128 | T=512 | T=2048 |
|---------|-------|-------|-------|
| Level15 | 0.242 | 0.532 | 1.183 |
| Level15Cascade | 0.201 | 0.491 | 1.149 |

############ lm200 ############
# Long-Sequence Evaluation
Config: lm200, n_landmarks: 200, lengths: [128, 512, 2048]
Seeds: [0, 1, 2]
## Accuracy
| Variant | T=128 | T=512 | T=2048 |
|---------|-------|-------|-------|
| Level15 | 0.921±0.022 | 0.841±0.033 | 0.664±0.036 |
| Level15Cascade | 0.985±0.020 | 0.949±0.053 | 0.771±0.091 |
## NLL
| Variant | T=128 | T=512 | T=2048 |
|---------|-------|-------|-------|
| Level15 | 0.405 | 0.721 | 1.437 |
| Level15Cascade | 0.074 | 0.264 | 1.536 |
