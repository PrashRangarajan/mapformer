# Does the forget gate work through its trajectory?

Effective decay per step is `lambda * E[sigmoid]`; lambda alone is not
interpretable. `gain` is `Forget - Vanilla` at T=1024, from the same seeds (the torus retrains bit-identically,
so these runs reproduce the stored checkpoints exactly).

| seed | peak decay | at frac of training | final decay | gain | peak interior |
|---|---|---|---|---|---|
| 0 | +0.04415 | 0.01 | +0.00320 | +0.002 | no (monotone) |
| 1 | +0.04044 | 0.01 | -0.00010 | +0.132 | no (monotone) |
| 2 | +0.04235 | 0.01 | -0.00037 | +0.093 | no (monotone) |
| 3 | +0.04248 | 0.01 | -0.00054 | +0.013 | no (monotone) |
| 4 | +0.04399 | 0.01 | +0.00267 | +0.147 | no (monotone) |
| 5 | +0.04142 | 0.01 | -0.00107 | +0.214 | no (monotone) |
| 6 | +0.05579 | 0.04 | -0.00152 | +0.058 | yes |
| 7 | +0.09900 | 0.10 | +0.01931 | -0.015 | yes |

- **r(peak decay, gain) = -0.531**  (pre-registered: > 0)
- r(final decay, gain) = -0.511
- peak is interior in **2/8** seeds (rise-then-fall rather than monotone drift)

**Verdict.** NOT supported. The transient-aid hypothesis does not survive its own pre-registered test, and the mechanism behind the +0.081 remains unidentified.
