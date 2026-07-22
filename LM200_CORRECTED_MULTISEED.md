# CORRECTED lm200 leaderboard — fresh multi-seed (current code)

Supersedes all lm200 tables built on the April checkpoints, which never
converged (loss ~1.0 vs ~0.005) and ranked training-convergence, not
architecture. See feedback_lm200_stuck_baselines.

| Variant | fresh loss | T=128 | T=512 | OLD T=512 (stuck) |
|---|---|---|---|---|
| Level15 (n=3) | 0.0083 | 0.996±0.003 | 0.990±0.005 | 0.800 |
| TEMFaithful (n=3) | 0.0009 | 1.000±0.000 | 0.974±0.008 | 0.982 |
| Level15GSF (n=3) | 0.0860 | 0.982±0.025 | 0.967±0.034 | 0.994 |
| Level15NoDrop (n=3) | 0.0918 | 0.981±0.014 | 0.956±0.029 | 0.915 |
| Level15EM (n=3) | 0.3537 | 0.938±0.045 | 0.823±0.110 | 0.786 |
| Vanilla (n=3) | 0.7741 | 0.814±0.073 | 0.742±0.075 | 0.710 |
| VanillaEM (n=3) | 0.9719 | 0.830±0.050 | 0.656±0.130 | 0.741 |
| PC (n=3) | 0.5377 | 0.888±0.041 | 0.716±0.012 | 0.854 |
| MambaLike (n=3) | 1.8174 | 0.562±0.010 | 0.549±0.013 | 0.549 |
| RoPE (n=3) | 1.5318 | 0.636±0.042 | 0.482±0.023 | 0.523 |

## Fresh ranking (T=512)

- Level15: 0.990 ± 0.005 (n=3)
- TEMFaithful: 0.974 ± 0.008 (n=3)
- Level15GSF: 0.967 ± 0.034 (n=3)
- Level15NoDrop: 0.956 ± 0.029 (n=3)
- Level15EM: 0.823 ± 0.110 (n=3)
- Vanilla: 0.742 ± 0.075 (n=3)
- PC: 0.716 ± 0.012 (n=3)
- VanillaEM: 0.656 ± 0.130 (n=3)
- MambaLike: 0.549 ± 0.013 (n=3)
- RoPE: 0.482 ± 0.023 (n=3)
