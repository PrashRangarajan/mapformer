# TEM baselines vs MapFormer (re-run with orthogonal W_a)

Two TEM variants vs Vanilla / Level15 on torus, 3 regimes, single-seed.
- **TEM**: GRU + factorised g/x + outer-product Hebbian memory.
- **TEMFaithful**: per-action W_a parametrised as exp(skew(A_a)) —
  orthogonal by construction. Modern-Hopfield memory.

> **STALE — TEMFaithful rows removed.** Every TEMFaithful number originally
> in this file was PRE-bug-fix (predict-then-update bug: memory queried with
> pre-action `g`, capping accuracy at chance ~0.42). The fix took TEMFaithful
> lm200 OOD T=512 from 0.42 to 0.97. Current TEMFaithful numbers live in
> `TEM_BACKGROUND_BASELINES.md` (clean), `GSF_NODROP_RESULTS.md` (lm200),
> `TEM_NOVEL_ENV_RESULTS.md` (novel-env). The TEM (GRU), Vanilla, Level15
> rows below are unaffected and kept for the historical single-seed record.

## clean

| Variant | T=128 OOD | T=512 OOD | T=128 NLL | T=512 NLL |
|---|---|---|---|---|
| **Vanilla** | 0.983 | 0.862 | 0.050 | 0.603 |
| **Level15** | 1.000 | 0.991 | 0.000 | 0.050 |
| **TEM** | 0.772 | 0.692 | 0.747 | 1.135 |

## noise

| Variant | T=128 OOD | T=512 OOD | T=128 NLL | T=512 NLL |
|---|---|---|---|---|
| **Vanilla** | 0.751 | 0.672 | 0.695 | 1.206 |
| **Level15** | 0.749 | 0.696 | 0.753 | 1.021 |
| **TEM** | 0.662 | 0.584 | 1.048 | 1.476 |

## lm200

| Variant | T=128 OOD | T=512 OOD | T=128 NLL | T=512 NLL |
|---|---|---|---|---|
| **Vanilla** | 0.848 | 0.710 | 0.825 | 1.401 |
| **Level15** | 0.895 | 0.790 | 0.581 | 0.991 |
| **TEM** | 0.711 | 0.612 | 1.140 | 1.622 |
