> **INVALID — lm200 (2026-08-09).** Every result in this file was trained on
> landmark (lm200) checkpoints that never converged (final CE ~1.0 instead of
> ~0.005). The reported ranking tracks training convergence, not architecture.
> See the RETRACTION section of `CLAUDE.md` and `CORRECTED_LM200_LEADERBOARD.md`.
> Corrected ranking (fresh, current code): Level15 0.996 > TEMFaithful 0.982 >
> NoDrop 0.915 > Level15EM 0.860 > Vanilla 0.835. Clean and noise results
> elsewhere are unaffected — they retrain bit-identically
> (`NOISE_CLEAN_REVALIDATION.md`).

# Long-Sequence Evaluation

Generated: 2026-07-15 18:05:06.317137

Config: lm200, n_landmarks: 200, lengths: [128, 512, 2048]
Seeds: [0, 1, 2]

## Accuracy

| Variant | T=128 | T=512 | T=2048 |
|---------|-------|-------|-------|
| Level15 | 0.921±0.022 | 0.841±0.033 | 0.664±0.036 |
| Level15CascadeNoSlow | 1.000±0.000 | 0.992±0.005 | 0.844±0.019 |
| Level15Cascade | 0.985±0.020 | 0.949±0.053 | 0.771±0.091 |

## NLL

| Variant | T=128 | T=512 | T=2048 |
|---------|-------|-------|-------|
| Level15 | 0.405 | 0.721 | 1.437 |
| Level15CascadeNoSlow | 0.001 | 0.054 | 1.344 |
| Level15Cascade | 0.074 | 0.264 | 1.536 |
