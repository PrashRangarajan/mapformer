# NOISE + CLEAN re-validation (seed 0, fresh vs old)

Retrained clean/noise variants under current code and compared against the
stored April/May checkpoints. **Fresh == old, bit-identical**, on every
clean and noise variant. This pins the stuck-training root cause to the
**landmark-cell-selection RNG** (`rng.permutation(n_cells)[:n_landmarks]`),
which only runs when `n_landmarks > 0`. Clean and noise (no landmarks) draw
identical data April-vs-now, so training reproduces exactly and those
results are valid. Only lm200 draws the shifted landmark layout, and lm200
training is basin-sensitive to it — which is why only lm200 broke.

## NOISE T=512 (env seed 0, 120 trials, eval seed 1000)

| Variant | OLD loss | OLD acc | FRESH loss | FRESH acc |
|---|---|---|---|---|
| Vanilla       | 0.737 | 0.811 | 0.737 | 0.811 |
| Level15       | 0.814 | 0.853 | 0.814 | 0.853 |
| VanillaEM     | 0.745 | 0.906 | 0.745 | 0.906 |
| Level15EM     | 0.738 | 0.879 | 0.738 | 0.879 |
| Level15NoDrop | 0.679 | 0.892 | 0.679 | 0.892 |
| TEMFaithful   | 1.014 | 0.907 | 1.014 | 0.907 |

## CLEAN T=512 spot-check

| Variant | OLD loss | OLD acc | FRESH loss | FRESH acc |
|---|---|---|---|---|
| RoPE    | 1.206 | 0.467 | 1.206 | 0.467 |
| Vanilla | 0.119 | 0.868 | 0.119 | 0.868 |

## Conclusion

- **Clean: valid.** Reproduces exactly. (RoPE's low 0.467 is genuine weakness,
  not a stuck-training artifact — it reproduces.)
- **Noise: valid.** Reproduces exactly. Earlier "noise degraded" worry was
  wrong: the April-vs-May loss gaps were genuine variant differences
  (NoDrop/GSF really train lower), not convergence artifacts.
- **lm200: the only compromised config** (see CORRECTED_LM200_LEADERBOARD.md).
