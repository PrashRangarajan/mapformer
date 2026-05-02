# TEM baselines vs MapFormer (re-run with orthogonal W_a)

Two TEM variants vs Vanilla / Level15 on torus, 3 regimes, single-seed.
- **TEM**: GRU + factorised g/x + outer-product Hebbian memory.
- **TEMFaithful**: per-action W_a parametrised as exp(skew(A_a)) — 
  orthogonal by construction, never blows up. Modern-Hopfield memory.
  This is the EM-RNN method the MapFormer paper claims to subsume.

## clean

| Variant | T=128 OOD | T=512 OOD | T=128 NLL | T=512 NLL |
|---|---|---|---|---|
| **Vanilla** | 0.983 | 0.862 | 0.050 | 0.603 |
| **Level15** | 1.000 | 0.991 | 0.000 | 0.050 |
| **TEM** | 0.772 | 0.692 | 0.747 | 1.135 |
| **TEMFaithful** | 0.440 | 0.423 | 1.486 | 1.545 |

## noise

| Variant | T=128 OOD | T=512 OOD | T=128 NLL | T=512 NLL |
|---|---|---|---|---|
| **Vanilla** | 0.751 | 0.672 | 0.695 | 1.206 |
| **Level15** | 0.749 | 0.696 | 0.753 | 1.021 |
| **TEM** | 0.662 | 0.584 | 1.048 | 1.476 |
| **TEMFaithful** | 0.470 | 0.471 | 1.621 | 1.713 |

## lm200

| Variant | T=128 OOD | T=512 OOD | T=128 NLL | T=512 NLL |
|---|---|---|---|---|
| **Vanilla** | 0.848 | 0.710 | 0.825 | 1.401 |
| **Level15** | 0.895 | 0.790 | 0.581 | 0.991 |
| **TEM** | 0.711 | 0.612 | 1.140 | 1.622 |
| **TEMFaithful** | 0.411 | 0.420 | 1.775 | 1.887 |

