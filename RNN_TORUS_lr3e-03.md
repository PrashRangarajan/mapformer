# RNN torus lr3e-03

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=512

| p_action_noise | LSTM_L1_h313 | LSTM_L1_h157 | LSTM_L2_h111 | GRU_L2_h128 |
|---|---|---|---|---|
| 0 | 0.604 ± 0.007 | 0.640 ± 0.005 | 0.686 ± 0.008 | 0.662 ± 0.008 |

## T=1024

| p_action_noise | LSTM_L1_h313 | LSTM_L1_h157 | LSTM_L2_h111 | GRU_L2_h128 |
|---|---|---|---|---|
| 0 | 0.572 ± 0.005 | 0.607 ± 0.005 | 0.649 ± 0.002 | 0.626 ± 0.004 |

## T=2048

| p_action_noise | LSTM_L1_h313 | LSTM_L1_h157 | LSTM_L2_h111 | GRU_L2_h128 |
|---|---|---|---|---|
| 0 | 0.532 ± 0.005 | 0.567 ± 0.005 | 0.605 ± 0.005 | 0.586 ± 0.003 |

