# Paper-Ready Results

Generated: 2026-04-22 20:05:07.419914

All trainings used 50 epochs × 156 batches × 128 batch size = ~1M sequences. Multi-seed results below: mean ± std over 3 training seeds (0, 1, 2).

## Config: clean (n_landmarks=0)

Two evaluations per model:
- **in-dist (trained env)**: obs_map matches training seed — tests memorization + path integration
- **OOD (fresh env)**: obs_map from seed+1000 — tests path integration generalization only

### In-distribution (obs_map seen during training)

| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |
|---------|-----------|-----------|-----------|-----------|
| Vanilla | 0.990 ± 0.009 | 0.031 ± 0.023 | 0.921 ± 0.037 | 0.388 ± 0.132 |
| RoPE | 0.663 ± 0.084 | 1.106 ± 0.223 | 0.475 ± 0.016 | 1.975 ± 0.025 |
| LSTM | N/A | N/A | N/A | N/A |
| CoPE | N/A | N/A | N/A | N/A |
| MambaLike | N/A | N/A | N/A | N/A |
| Level1 | 0.942 ± 0.041 | 0.203 ± 0.147 | 0.890 ± 0.069 | 0.417 ± 0.245 |
| Level15 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.995 ± 0.003 | 0.026 ± 0.015 |
| PC | 0.961 ± 0.032 | 0.104 ± 0.070 | 0.821 ± 0.015 | 1.214 ± 0.217 |

### OOD (fresh obs_map, tests path-integration generalization)

| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |
|---------|-----------|-----------|-----------|-----------|
| Vanilla | 0.992 ± 0.006 | 0.025 ± 0.018 | 0.913 ± 0.037 | 0.443 ± 0.121 |
| RoPE | 0.635 ± 0.072 | 1.216 ± 0.173 | 0.463 ± 0.026 | 2.028 ± 0.050 |
| LSTM | N/A | N/A | N/A | N/A |
| CoPE | N/A | N/A | N/A | N/A |
| MambaLike | N/A | N/A | N/A | N/A |
| Level1 | 0.939 ± 0.044 | 0.214 ± 0.152 | 0.880 ± 0.070 | 0.452 ± 0.247 |
| Level15 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.993 ± 0.003 | 0.039 ± 0.025 |
| PC | 0.968 ± 0.027 | 0.085 ± 0.059 | 0.815 ± 0.005 | 1.246 ± 0.196 |

## Config: noise (n_landmarks=0)

Two evaluations per model:
- **in-dist (trained env)**: obs_map matches training seed — tests memorization + path integration
- **OOD (fresh env)**: obs_map from seed+1000 — tests path integration generalization only

### In-distribution (obs_map seen during training)

| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |
|---------|-----------|-----------|-----------|-----------|
| Vanilla | 0.951 ± 0.006 | 0.247 ± 0.021 | 0.744 ± 0.051 | 1.225 ± 0.385 |
| RoPE | 0.638 ± 0.087 | 1.186 ± 0.211 | 0.479 ± 0.013 | 1.890 ± 0.018 |
| LSTM | N/A | N/A | N/A | N/A |
| CoPE | N/A | N/A | N/A | N/A |
| MambaLike | N/A | N/A | N/A | N/A |
| Level1 | 0.887 ± 0.035 | 0.464 ± 0.108 | 0.794 ± 0.036 | 0.810 ± 0.106 |
| Level15 | 0.947 ± 0.018 | 0.242 ± 0.072 | 0.865 ± 0.025 | 0.537 ± 0.076 |
| PC | 0.945 ± 0.006 | 0.241 ± 0.017 | 0.759 ± 0.094 | 1.298 ± 0.930 |

### OOD (fresh obs_map, tests path-integration generalization)

| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |
|---------|-----------|-----------|-----------|-----------|
| Vanilla | 0.954 ± 0.008 | 0.243 ± 0.026 | 0.739 ± 0.062 | 1.267 ± 0.437 |
| RoPE | 0.608 ± 0.070 | 1.286 ± 0.153 | 0.469 ± 0.027 | 1.958 ± 0.044 |
| LSTM | N/A | N/A | N/A | N/A |
| CoPE | N/A | N/A | N/A | N/A |
| MambaLike | N/A | N/A | N/A | N/A |
| Level1 | 0.887 ± 0.042 | 0.466 ± 0.136 | 0.783 ± 0.041 | 0.851 ± 0.128 |
| Level15 | 0.949 ± 0.015 | 0.241 ± 0.057 | 0.851 ± 0.026 | 0.585 ± 0.069 |
| PC | 0.941 ± 0.007 | 0.246 ± 0.016 | 0.752 ± 0.095 | 1.328 ± 0.914 |

## Config: lm200 (n_landmarks=200)

Two evaluations per model:
- **in-dist (trained env)**: obs_map matches training seed — tests memorization + path integration
- **OOD (fresh env)**: obs_map from seed+1000 — tests path integration generalization only

### In-distribution (obs_map seen during training)

| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |
|---------|-----------|-----------|-----------|-----------|
| Vanilla | 0.841 ± 0.021 | 0.722 ± 0.081 | 0.726 ± 0.049 | 1.282 ± 0.259 |
| RoPE | 0.650 ± 0.044 | 1.360 ± 0.166 | 0.512 ± 0.024 | 2.224 ± 0.211 |
| LSTM | N/A | N/A | N/A | N/A |
| CoPE | N/A | N/A | N/A | N/A |
| MambaLike | N/A | N/A | N/A | N/A |
| Level1 | 0.812 ± 0.039 | 0.794 ± 0.109 | 0.733 ± 0.036 | 1.083 ± 0.145 |
| Level15 | 0.920 ± 0.021 | 0.409 ± 0.070 | 0.841 ± 0.033 | 0.719 ± 0.111 |
| PC | 0.868 ± 0.040 | 0.575 ± 0.145 | 0.748 ± 0.075 | 1.083 ± 0.287 |

### OOD (fresh obs_map, tests path-integration generalization)

| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |
|---------|-----------|-----------|-----------|-----------|
| Vanilla | 0.837 ± 0.031 | 0.804 ± 0.085 | 0.715 ± 0.059 | 1.396 ± 0.286 |
| RoPE | 0.626 ± 0.031 | 1.603 ± 0.155 | 0.495 ± 0.013 | 2.353 ± 0.147 |
| LSTM | N/A | N/A | N/A | N/A |
| CoPE | N/A | N/A | N/A | N/A |
| MambaLike | N/A | N/A | N/A | N/A |
| Level1 | 0.802 ± 0.042 | 0.947 ± 0.133 | 0.721 ± 0.040 | 1.237 ± 0.141 |
| Level15 | 0.909 ± 0.012 | 0.544 ± 0.027 | 0.821 ± 0.025 | 0.898 ± 0.079 |
| PC | 0.860 ± 0.040 | 0.681 ± 0.130 | 0.733 ± 0.071 | 1.229 ± 0.256 |

## Level 1.5 Ablations (single seed)

### Config: clean

| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |
|---------|-----------|-----------|-----------|-----------|
| Level15 | 1.000 | 0.000 | 0.993 | 0.038 |
| L15_ConstR | 0.795 | 0.663 | 0.672 | 1.123 |
| L15_NoMeas | 0.904 | 0.368 | 0.831 | 0.640 |
| L15_NoCorr | 0.940 | 0.204 | 0.833 | 0.803 |
| L15_DARE | 1.000 | 0.000 | 0.992 | 0.040 |

### Config: lm200

| Variant | T=128 acc | T=128 NLL | T=512 acc | T=512 NLL |
|---------|-----------|-----------|-----------|-----------|
| Level15 | 0.896 | 0.496 | 0.800 | 0.845 |
| L15_ConstR | 0.783 | 0.776 | 0.658 | 1.247 |
| L15_NoMeas | 0.905 | 0.459 | 0.836 | 0.772 |
| L15_NoCorr | 0.898 | 0.518 | 0.679 | 1.541 |
| L15_DARE | 0.943 | 0.329 | 0.881 | 0.580 |


---
*Auto-generated by orchestrator_finalize.sh.*

