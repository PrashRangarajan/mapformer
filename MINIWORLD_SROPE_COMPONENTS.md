# Selective RoPE components on navigation — 3 seeds, convergence-conditioned

ConvDelta (SRoPE's conv1d before the cumsum) and GateDelta (SRoPE's sigmoid gate on Delta) vs Vanilla MapFormer. Fresh-map, oracle recode, grids 16/24/32, n=3. **Accuracy (final train loss)** — the hierarchy ablation showed effects here are frequently convergence, not capability, so the conditioned row is the one that counts.

## T=512

### ConvDelta vs Vanilla

| grid | seed | Vanilla | ConvDelta | delta | both converged? |
|---|---|---|---|---|---|
| 16 | 0 | 0.947 <sub>(0.17)</sub> | 0.872 <sub>(0.28)</sub> | -0.075 | YES |
| 16 | 1 | 0.949 <sub>(0.16)</sub> | 0.992 <sub>(0.06)</sub> | +0.044 | YES |
| 16 | 2 | 0.761 <sub>(0.44)</sub> | 0.772 <sub>(0.47)</sub> | +0.011 | no |
| 24 | 0 | 0.623 <sub>(0.76)</sub> | 0.969 <sub>(0.10)</sub> | +0.346 | no |
| 24 | 1 | 0.838 <sub>(0.36)</sub> | 0.667 <sub>(0.67)</sub> | -0.171 | no |
| 24 | 2 | 0.621 <sub>(0.77)</sub> | 0.634 <sub>(0.71)</sub> | +0.013 | no |
| 32 | 0 | 0.559 <sub>(0.85)</sub> | 0.808 <sub>(0.43)</sub> | +0.249 | no |
| 32 | 1 | 0.967 <sub>(0.13)</sub> | 0.782 <sub>(0.50)</sub> | -0.185 | no |
| 32 | 2 | 0.582 <sub>(0.82)</sub> | 0.664 <sub>(0.69)</sub> | +0.082 | no |

- **pooled** (9 pairs): **+0.035**
- **both-converged only** (2 pairs): **-0.016**

### GateDelta vs Vanilla

| grid | seed | Vanilla | GateDelta | delta | both converged? |
|---|---|---|---|---|---|
| 16 | 0 | 0.947 <sub>(0.17)</sub> | 0.591 <sub>(0.67)</sub> | -0.356 | no |
| 16 | 1 | 0.949 <sub>(0.16)</sub> | 1.000 <sub>(0.01)</sub> | +0.051 | YES |
| 16 | 2 | 0.761 <sub>(0.44)</sub> | 0.694 <sub>(0.62)</sub> | -0.067 | no |
| 24 | 0 | 0.623 <sub>(0.76)</sub> | 0.745 <sub>(0.53)</sub> | +0.122 | no |
| 24 | 1 | 0.838 <sub>(0.36)</sub> | 1.000 <sub>(0.00)</sub> | +0.162 | YES |
| 24 | 2 | 0.621 <sub>(0.77)</sub> | 0.633 <sub>(0.74)</sub> | +0.012 | no |
| 32 | 0 | 0.559 <sub>(0.85)</sub> | 0.668 <sub>(0.69)</sub> | +0.109 | no |
| 32 | 1 | 0.967 <sub>(0.13)</sub> | 1.000 <sub>(0.00)</sub> | +0.033 | YES |
| 32 | 2 | 0.582 <sub>(0.82)</sub> | 0.992 <sub>(0.11)</sub> | +0.410 | no |

- **pooled** (9 pairs): **+0.053**
- **both-converged only** (3 pairs): **+0.082**

## T=1024

### ConvDelta vs Vanilla

| grid | seed | Vanilla | ConvDelta | delta | both converged? |
|---|---|---|---|---|---|
| 16 | 0 | 0.786 <sub>(0.17)</sub> | 0.676 <sub>(0.28)</sub> | -0.110 | YES |
| 16 | 1 | 0.803 <sub>(0.16)</sub> | 0.883 <sub>(0.06)</sub> | +0.080 | YES |
| 16 | 2 | 0.548 <sub>(0.44)</sub> | 0.584 <sub>(0.47)</sub> | +0.035 | no |
| 24 | 0 | 0.394 <sub>(0.76)</sub> | 0.733 <sub>(0.10)</sub> | +0.339 | no |
| 24 | 1 | 0.694 <sub>(0.36)</sub> | 0.494 <sub>(0.67)</sub> | -0.199 | no |
| 24 | 2 | 0.430 <sub>(0.77)</sub> | 0.463 <sub>(0.71)</sub> | +0.033 | no |
| 32 | 0 | 0.320 <sub>(0.85)</sub> | 0.630 <sub>(0.43)</sub> | +0.310 | no |
| 32 | 1 | 0.828 <sub>(0.13)</sub> | 0.614 <sub>(0.50)</sub> | -0.214 | no |
| 32 | 2 | 0.366 <sub>(0.82)</sub> | 0.453 <sub>(0.69)</sub> | +0.087 | no |

- **pooled** (9 pairs): **+0.040**
- **both-converged only** (2 pairs): **-0.015**

### GateDelta vs Vanilla

| grid | seed | Vanilla | GateDelta | delta | both converged? |
|---|---|---|---|---|---|
| 16 | 0 | 0.786 <sub>(0.17)</sub> | 0.386 <sub>(0.67)</sub> | -0.400 | no |
| 16 | 1 | 0.803 <sub>(0.16)</sub> | 1.000 <sub>(0.01)</sub> | +0.197 | YES |
| 16 | 2 | 0.548 <sub>(0.44)</sub> | 0.518 <sub>(0.62)</sub> | -0.030 | no |
| 24 | 0 | 0.394 <sub>(0.76)</sub> | 0.522 <sub>(0.53)</sub> | +0.128 | no |
| 24 | 1 | 0.694 <sub>(0.36)</sub> | 1.000 <sub>(0.00)</sub> | +0.306 | YES |
| 24 | 2 | 0.430 <sub>(0.77)</sub> | 0.457 <sub>(0.74)</sub> | +0.026 | no |
| 32 | 0 | 0.320 <sub>(0.85)</sub> | 0.424 <sub>(0.69)</sub> | +0.105 | no |
| 32 | 1 | 0.828 <sub>(0.13)</sub> | 1.000 <sub>(0.00)</sub> | +0.172 | YES |
| 32 | 2 | 0.366 <sub>(0.82)</sub> | 0.944 <sub>(0.11)</sub> | +0.578 | no |

- **pooled** (9 pairs): **+0.120**
- **both-converged only** (3 pairs): **+0.225**

> If the both-converged effect is ~0, this is trainability (as hierarchy was).
> If it survives, a component from the LANGUAGE paper improves the NAVIGATION
> model — the cross-domain transfer neither paper tested.


## CORRECTION (adversarial review, 2026-08-27): 'null' is UNSUPPORTED

Paired-delta SD across the 9 (grid,seed) pairs is 0.176 (ConvDelta) / 0.203
(GateDelta). Minimum detectable effect at 80% power, n=9: **0.165 / 0.190**.
The ConvDelta data are consistent with anything from -0.09 to +0.16 -- and a
+0.16 effect would be among the LARGEST architectural effects in this repo.
Detecting 0.05 at this noise level needs n ~ 100 pairs.

So: **ConvDelta and GateDelta are UNMEASURED, not null.** The 'both-converged'
subsets (n=2, n=3) and the gate control's n=1 have MDE 0.29 to >0.5 and should
not be reported as point estimates at all.

Independently: the measured run-to-run noise floor for this setup is 0.150
(two provably function-identical models, GateDeltaCtl vs Vanilla, n=9). Every
component effect chased here (+0.035, +0.053, +0.081) is inside it.
