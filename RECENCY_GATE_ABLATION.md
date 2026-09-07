# Is the content gate the mechanism? A causal test

`Signed_r4`, n=8 seeds, T=1024, eval-only intervention on Delta. Chance 0.0625; the index arms sit at 0.234.

| condition | accuracy | delta vs baseline | sd | MDE | seeds worse |
|---|---|---|---|---|---|
| `none` | 1.0000 +/- 0.0000 | +0.0000 | 0.0000 | 0.0000 | 0/8 |
| `zero_filler` | 0.9997 +/- 0.0008 | -0.0003 | 0.0008 | 0.0008 | 1/8 |
| `equalize` | 0.0855 +/- 0.0075 | -0.9145 | 0.0075 | 0.0074 | 8/8 |
| `scale_match` | 0.1096 +/- 0.0231 | -0.8904 | 0.0231 | 0.0229 | 8/8 |
| `uniform_content` | 0.7826 +/- 0.1150 | -0.2174 | 0.1150 | 0.1138 | 8/8 |
| `uniform_all` | 0.1886 +/- 0.0248 | -0.8114 | 0.0248 | 0.0245 | 8/8 |
| `zero_content` | 0.0680 +/- 0.0172 | -0.9320 | 0.0172 | 0.0171 | 8/8 |
| `zero_query` | 0.8232 +/- 0.0797 | -0.1768 | 0.0797 | 0.0789 | 8/8 |

## Reading it -- and a correction to my own criterion

**The pre-stated criterion was: `zero_filler` a no-op AND `equalize` collapses AND `zero_content` destroys. Two of those hold, but the `equalize` leg is CONFOUNDED and I am not counting it.** Making filler count roughly doubles how fast theta advances, and `scale_match` -- which changes ONLY the scale, still counting content alone -- collapses just as hard (0.110 vs 0.086). So this model is acutely sensitive to theta's absolute scale, and any intervention that moves it is destructive for reasons that have nothing to do with what is counted. `equalize` therefore proves nothing on its own.

**What does establish it is the magnitude-matched pair, added after `scale_match` failed.** `uniform_content` and `uniform_all` both replace Delta with a CONSTANT and both make theta travel the same total distance as the baseline. They differ in exactly one thing: which tokens the constant lands on. The gap is the isolated value of counting CONTENT rather than TOKENS.

The chain that survives:

1. `zero_filler` is a no-op -- filler increments contribute nothing, so a gate exists. Unconfounded: this LOWERS theta's rate and costs nothing, while scale changes in either direction are otherwise fatal.
2. `zero_content` destroys -- the positive control bites, so the intervention reaches the pathway that matters.
3. `uniform_content` (0.783) vs `uniform_all` (0.189) -- **+0.594 at 8/8 seeds, magnitude-matched**. What is counted is what matters.

A constant increment on content alone recovers 0.783 of the baseline 1.000, so the learned Delta structure beyond "constant on counted tokens, zero elsewhere" is worth only ~0.22: the gate is most of the mechanism, not a component of it. And `uniform_all` (0.189) lands BELOW the index arms (0.234) -- a token clock inside this architecture is no better than an index code, which is what it is.

Caveat on provenance: the isolated pair was designed AFTER seeing `scale_match` fail. It is a control for a confound, not a second bite at the hypothesis, but it was not pre-registered and is labelled here.
