# Is the position effect about ALIASING, or about map size?

Grid size FIXED at 32; only `n_obs` varies, which RELABELS the obs_map and
changes nothing else -- the gates confirm identical label mass (50.4 scored
per trajectory) and identical revisit lag (median 33) across all three
conditions. Same walks, same scored positions, different amounts of aliasing.

400 epochs, warmup+cosine, oracle recode, n=5 seeds, T=512. Verdict
thresholds were hard-coded in this script before the runs finished.

## Reproducibility of the reused anchor

RoPE n_obs=16 s0 retrained in THIS batch: 0.725 vs stored 0.725 (drift +0.000).

Cross-batch anchor is LICENSED.

## Effect by aliasing level

| n_obs | cells/token | Vanilla (path-int) | RoPE (index) | effect | per-seed | all flat? |
|---|---|---|---|---|---|---|
| 16 | 32 | 0.936 | 0.758 | **+0.178** | +0.274, +0.210, +0.033, +0.230, +0.142 | YES |
| 64 | 8 | 0.999 | 0.689 | **+0.310** | +0.277, +0.291, +0.330, +0.341 | **no** |
| 256 | 2 | 0.981 | 0.607 | **+0.374** | +0.394, +0.364, +0.408, +0.299, +0.407 | **no** |

Measured noise floor (two function-identical models): **0.150**. No effect smaller than this is reportable.

## Distance above each condition's own floor

Raw accuracy is NOT comparable across `n_obs` -- more classes means a lower
marginal. What is comparable is how far each arm sits above ITS floor.

| n_obs | non-blank marginal (floor) | Vanilla above floor | RoPE above floor |
|---|---|---|---|
| 16 | 0.077 | +0.859 | +0.681 |
| 64 | 0.031 | +0.968 | +0.658 |
| 256 | 0.013 | +0.968 | +0.594 |

## The same comparison in TRAINING LOSS (rule 9)

Held-out accuracy here is an affine readout of final training loss
(r = -0.996 over 57 runs). So the accuracy effect above is a loss gap in
disguise, and the loss gap is the more direct measurement of it. If the
aliasing story is right, THIS should shrink with aliasing too.

| n_obs | Vanilla loss | RoPE loss | loss gap (negative = path-int fits better) | ranges overlap? |
|---|---|---|---|---|
| 16 | 0.120 [0.004-0.430] | 0.438 [0.406-0.499] | **-0.318** | yes |
| 64 | 0.026 [0.006-0.040] | 0.560 [0.512-0.613] | **-0.534** | **no** |
| 256 | 0.073 [0.007-0.286] | 0.664 [0.599-0.704] | **-0.591** | **no** |

Where the ranges do NOT overlap, no loss-matched residual can be computed
without extrapolating, so this study cannot separate 'path integration
optimises better' from 'path integration represents better at equal fit'. It
measures the former. That limit is inherent to these runs, not a choice of
analysis.

## Convergence sensitivity

| n_obs | all seeds | both arms flat only | direction of the bias |
|---|---|---|---|
| 16 | +0.178 (n=5) | +0.178 (n=5) | unchanged |
| 64 | +0.310 (n=4) | +0.316 (n=3) | conditioning RAISES it -> non-convergence is not inflating the effect |
| 256 | +0.374 (n=5) | +0.408 (n=1) | conditioning RAISES it -> non-convergence is not inflating the effect |

## Power (rule 11)

| n_obs | n | mean effect | sd | MDE | detectable? |
|---|---|---|---|---|---|
| 16 | 5 | +0.178 | 0.094 | 0.118 | yes |
| 64 | 4 | +0.310 | 0.031 | 0.043 | yes |
| 256 | 5 | +0.374 | 0.046 | 0.057 | yes |

## Verdict

**NOT ALL ARMS CONVERGED. Nothing here is interpretable** -- an unconverged arm makes this a time-to-solve comparison, which is exactly the confound that inverted the grid-8 sign (rule 10).

## Scope

One environment (MiniWorld OneRoom), one map size, n=5. Aliasing is varied by
relabelling the obs_map, which is the cleanest available manipulation but is
not the only way aliasing could be varied (p_empty and map size both also
change it, and are held fixed here).
