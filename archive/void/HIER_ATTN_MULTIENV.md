> **INVALID — lm200 (2026-08-09).** Every result in this file was trained on
> landmark (lm200) checkpoints that never converged (final CE ~1.0 instead of
> ~0.005). The reported ranking tracks training convergence, not architecture.
> See the RETRACTION section of `CLAUDE.md` and `CORRECTED_LM200_LEADERBOARD.md`.
> Corrected ranking (fresh, current code): Level15 0.996 > TEMFaithful 0.982 >
> NoDrop 0.915 > Level15EM 0.860 > Vanilla 0.835. Clean and noise results
> elsewhere are unaffected — they retrain bit-identically
> (`NOISE_CLEAN_REVALIDATION.md`).

# Attention hierarchy at MULTI-ENV transfer (held-out envs, seed 0)

Matched flat Level15 vs HierAttn; 50 train / 50 held-out envs.

| Variant | Config | train loss | train acc | held T=128 | held T=512 OOD |
|---|---|---|---|---|---|
| Level15 | clean | 0.0378 | 0.991 | 0.990 | 0.947 |
| HierAttn | clean | 0.0425 | 0.993 | 0.994 | 0.947 |
| Level15 | lm200 | 0.0091 | 0.999 | 0.998 | 0.993 |
| HierAttn | lm200 | 0.0791 | 0.986 | 0.989 | 0.941 |
