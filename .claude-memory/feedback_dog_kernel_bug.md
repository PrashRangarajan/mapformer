---
name: DoG kernel must use normalised Gaussians
description: Earlier hex tests were vacuous due to a DoG kernel bug; targets were silently zero. Always verify DoG signal is nonzero before reading hex results.
type: feedback
---

On 2026-05-01 we discovered that `model_level15_dog.py` and `continuous_nav.py` (early version) used the wrong DoG formula:

```python
gE = exp(-d² / (2 σE²))     # at d=0: gE = 1
gI = exp(-d² / (2 σI²))     # at d=0: gI = 1
target = max(0, gE - gI)    # at d=0: target = 0
```

Both unnormalised Gaussians equal 1 at the centre; their difference is 0. ReLU gives 0 everywhere. **The DoG target is silently all zeros.** The earlier `DOG_RESULTS.md` (max grid score 0.036, no hex emergence) was on broken targets — a vacuous negative, NOT a real test of Sorscher's three conditions.

**The fix** uses normalised 2D Gaussians (1/σ² prefactor), so the narrower (excitatory) Gaussian has higher peak amplitude:

```python
gE = (1/σE²) * exp(-d² / (2 σE²))   # at d=0: 1/σE² = 0.444
gI = (1/σI²) * exp(-d² / (2 σI²))   # at d=0: 1/σI² = 0.111
target = max(0, gE - gI)            # at d=0: 0.333
```

This gives the correct Mexican-hat shape: positive bump at centre, zero (after ReLU) outside.

**How to apply:**
- When working with any DoG-supervised hex experiment, first sanity-check that `target.max() > 0.1` and `target.mean() > 0`. If targets are all near zero, the kernel is broken and any "no hex" result is invalid.
- The earlier `DOG_RESULTS.md` is vacuous — superseded by `DOG_RESULTS_FIXED.md` (re-run pending GPU availability).
- Anywhere else in the codebase that computes DoG: check it uses normalised Gaussians.
- Sorscher's theory predicts hex emergence under three conditions: (1) path integration, (2) ReLU non-negativity bottleneck, (3) DoG-supervised place-cell targets. Our discrete tests historically only reliably met (1). The continuous-nav experiment now meets all three (verified targets are nonzero); it's the first valid Sorscher-conditions test we've run.
