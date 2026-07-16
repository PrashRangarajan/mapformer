# Corrected lm200 leaderboard (seed 0, current code)

Fresh = retrained under current code. OLD = April/May stored checkpoints.
Reveals the April training-convergence artifact.

| Variant | OLD loss | OLD T512 | FRESH loss | FRESH T128 | FRESH T512 |
|---|---|---|---|---|---|
| Vanilla | 1.218 | 0.710 | 0.566 | 0.882 | 0.835 |
| Level15 | 1.011 | 0.800 | 0.005 | 1.000 | 0.996 |
| Level15NoDrop | 0.244 | 0.915 | 0.244 | 0.960 | 0.915 |
| Level15GSF | 0.001 | 0.994 | -- | -- | -- |
| TEMFaithful | 0.000 | 0.982 | 0.000 | 1.000 | 0.982 |
| VanillaEM | 1.557 | 0.741 | 0.628 | 0.901 | 0.807 |
| Level15EM | 1.015 | 0.786 | 0.121 | 0.981 | 0.860 |
| RoPE | 1.546 | 0.523 | 1.369 | 0.694 | 0.513 |
| PC | 0.936 | 0.854 | 0.318 | 0.946 | 0.721 |
| MambaLike | 1.946 | 0.549 | 1.813 | 0.577 | 0.567 |

## Fresh ranking (T512):
- Level15: 0.996
- TEMFaithful: 0.982
- Level15NoDrop: 0.915
- Level15EM: 0.860
- Vanilla: 0.835
- VanillaEM: 0.807
- PC: 0.721
- MambaLike: 0.567
- RoPE: 0.513
