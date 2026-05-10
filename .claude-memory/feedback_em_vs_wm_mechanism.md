---
name: EM vs WM is a regime claim, not a universal ordering
description: MapFormer-EM's multiplicative AND-gate (A_X ⊙ A_P) wins when A_X is the noisy channel; MapFormer-WM's additive scoring wins when A_X is the signal channel. The paper's "EM scales better" claim is along axes that make A_X noisier; our extended regimes go the other way.
type: feedback
originSessionId: continued-2026-05-10
---

**Mechanism (verified empirically across our regimes 2026-05-10):**

- **MapFormer-EM** attention: `A = softmax(A_X ⊙ A_P)` — content-score and position-score combined multiplicatively. Both must be high simultaneously → AND-gate.
- **MapFormer-WM** attention: combined additively in the score → OR-gate (a strong content match can compensate for slightly-off position match).

**Regime predictions and observations:**

| Regime | A_X channel | A_P channel | Predicted winner | Observed |
|---|---|---|---|---|
| Aliased obs, short l (paper main task) | Noisy | Sharp | EM (filter A_X cleanly) | EM 0.972 > WM 0.913 ✓ |
| Aliased obs, **large vocab**, short l (paper Fig 4c) | Increasingly noisy | Sharp | EM (more bottleneck on A_X) | EM > WM ✓ (per paper) |
| Aliased obs, **long sequence** | Noisy | Drift-degraded | EM AND-gate fails → WM | WM wins at OOD T=512 ✓ |
| Landmarks (rare unique content) | Sharp signal | Drift-degraded | WM (additive compensates for weak A_P) | WM 0.715 > EM 0.605 ✓ |
| Landmarks + correction (Level15) | Sharp signal | Repaired | Both helped; WM still slight edge | Level15-WM 0.821 > Level15-EM 0.730 ✓ |
| Noise (transition stochasticity) | Noisy | Severely drift-degraded | Both struggle; closer call | Level15-WM ≈ Level15-EM ✓ |

**Paper's "EM scales better" (Figure 4):**
- (a) Head size 16→128, l=256: WM degrades; EM stable. Compatible with mechanism — smaller h reduces A_P-channel capacity → EM's filtering relatively more useful.
- (b) Sequence length 16→384, h=48: WM degrades. **But** the paper's worst l is 384, much shorter than our T=512 OOD; their A_P degradation is mild. At our OOD lengths EM is worse, not better. Open question whether their claim survives at T=512+ (our vocab sweep tests this).
- (c) Vocab 10→10000, l=16: **EM dominates decisively at large vocab.** This is the strongest paper claim and l=16 is short → A_P is sharp → mechanism predicts EM advantage. Untested by us at our l=128 / OOD T=512.

**Conclusion:** "EM is the better model" is a paper-task claim, not a universal one. Backbone choice is regime-dependent and predictable from whether A_X or A_P is the bottleneck. Use this in writeups instead of "EM is better at scale."

**Open empirical question (vocab sweep in flight 2026-05-10):** does Level15-WM > Level15-EM survive at the paper's vocab-scaling regime (n_obs=256, 4096) when we evaluate at T=512 OOD? If yes → the long-l-regime dominates the vocab-regime under correction. If no → the paper's factorization argument reasserts at large vocab even with correction.
