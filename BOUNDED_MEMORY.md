# Bounded-memory eval: does PoPE need to RETRIEVE the action history?

Prefix-preserving sliding window: each query sees the first 2 tokens (goal) plus the last W. T_explore=64 (in-distribution), n_trials=100, seeds=[0, 1, 2]. Eval-only: models were TRAINED with full attention.

| variant | W=inf | W=256 | W=128 | W=64 | W=32 | W=16 |
|---|---|---|---|---|---|---|
| PoPE-Flat | 0.951 ± 0.002 | 0.951 ± 0.002 | 0.951 ± 0.002 | 0.951 ± 0.002 | 0.857 ± 0.007 | 0.731 ± 0.009 |
| MapPoPE-Flat | 0.951 ± 0.002 | 0.951 ± 0.002 | 0.951 ± 0.002 | 0.951 ± 0.002 | 0.872 ± 0.003 | 0.751 ± 0.006 |
| MapPoPE-Hier | 0.951 ± 0.003 | 0.951 ± 0.003 | 0.951 ± 0.003 | 0.951 ± 0.003 | 0.927 ± 0.013 | 0.821 ± 0.006 |
| MapWM-Flat | 0.958 ± 0.008 | 0.958 ± 0.008 | 0.902 ± 0.045 | 0.698 ± 0.177 | 0.794 ± 0.122 | 0.832 ± 0.118 |
| MapWM-Hier | 0.964 ± 0.002 | 0.964 ± 0.002 | 0.939 ± 0.038 | 0.758 ± 0.212 | 0.756 ± 0.207 | 0.742 ± 0.196 |
| MapWM-Hier-CoarsePI | 0.960 ± 0.000 | 0.960 ± 0.000 | 0.956 ± 0.004 | 0.777 ± 0.170 | 0.765 ± 0.133 | 0.763 ± 0.100 |
| Plain-Flat | 0.965 ± 0.003 | 0.965 ± 0.003 | 0.965 ± 0.003 | 0.965 ± 0.003 | 0.965 ± 0.003 | 0.965 ± 0.003 |
