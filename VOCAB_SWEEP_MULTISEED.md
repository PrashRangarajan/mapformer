# Vocab sweep, MULTI-SEED (n=3), same training batch

Supersedes the single-seed table in VOCAB_SWEEP_RESULTS.md, whose EM row could not be distinguished from a collapsed seed.

`VanillaEM` = paper-faithful separate q0/k0 (App. A.4). `VanillaEM_P0` = single-p_0 ablation.

## n_obs = 16

| variant | T=128 | T=512 |
|---|---|---|
| Vanilla | 0.997 ± 0.000 | 0.950 ± 0.010 |
| VanillaEM | 1.000 ± 0.001 | 0.977 ± 0.006 |
| VanillaEM_P0 | 0.999 ± 0.001 | 0.977 ± 0.005 |

- Vanilla per-seed @T=512: 0.959, 0.952, 0.939
- VanillaEM per-seed @T=512: 0.973, 0.984, 0.975
- VanillaEM_P0 per-seed @T=512: 0.983, 0.975, 0.974

## n_obs = 256

| variant | T=128 | T=512 |
|---|---|---|
| Vanilla | 0.913 ± 0.090 | 0.675 ± 0.103 |
| VanillaEM | 0.761 ± 0.050 | 0.590 ± 0.020 |
| VanillaEM_P0 | 0.806 ± 0.262 | 0.773 ± 0.234 |

- Vanilla per-seed @T=512: 0.652, 0.788, 0.587
- VanillaEM per-seed @T=512: 0.593, 0.568, 0.608
- VanillaEM_P0 per-seed @T=512: 0.910, 0.502, 0.906

## n_obs = 4096

| variant | T=128 | T=512 |
|---|---|---|
| Vanilla | 0.465 ± 0.018 | 0.483 ± 0.007 |
| VanillaEM | 0.493 ± 0.004 | 0.497 ± 0.001 |
| VanillaEM_P0 | 0.499 ± 0.006 | 0.499 ± 0.001 |

- Vanilla per-seed @T=512: 0.480, 0.480, 0.491
- VanillaEM per-seed @T=512: 0.498, 0.498, 0.496
- VanillaEM_P0 per-seed @T=512: 0.499, 0.500, 0.498

