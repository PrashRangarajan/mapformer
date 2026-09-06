# Does PoPE's gain scale with how much the phase wraps?

`MapPoPE` minus `Vanilla`: same path-integrated phase, PoPE's magnitude and
one frequency per element instead of one per pair. +320 parameters at every
grid size. 8 seeds, one batch.

omega spans [2pi/N, 2pi] over a fixed 32 blocks, so the spectrum covers
log2(N) octaves and thins as N grows -- the quantity the prediction is about.

## T = 128

| grid | octaves | `Vanilla` | `MapPoPE` | gain | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 0.623 ± 0.255 | 0.838 ± 0.235 | **+0.215** | 0.448 | 0.444 | 6/8 | unmeasured |
| 32 | 5 | 0.999 ± 0.002 | 1.000 ± 0.000 | **+0.001** | 0.002 | 0.002 | 5/8 | unmeasured  *(ceiling)* |
| 64 | 6 | 0.993 ± 0.017 | 1.000 ± 0.001 | **+0.007** | 0.018 | 0.017 | 6/8 | unmeasured  *(ceiling)* |
| 128 | 7 | 0.996 ± 0.011 | 1.000 ± 0.000 | **+0.004** | 0.011 | 0.011 | 3/8 | unmeasured  *(ceiling)* |

## T = 512

| grid | octaves | `Vanilla` | `MapPoPE` | gain | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 0.589 ± 0.257 | 0.762 ± 0.261 | **+0.174** | 0.471 | 0.466 | 6/8 | unmeasured |
| 32 | 5 | 0.900 ± 0.033 | 0.957 ± 0.049 | **+0.057** | 0.070 | 0.069 | 7/8 | unmeasured |
| 64 | 6 | 0.944 ± 0.028 | 0.982 ± 0.011 | **+0.037** | 0.032 | 0.032 | 7/8 | DETECTABLE |
| 128 | 7 | 0.952 ± 0.052 | 0.993 ± 0.010 | **+0.042** | 0.055 | 0.054 | 5/8 | unmeasured |

## T = 1024

| grid | octaves | `Vanilla` | `MapPoPE` | gain | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|---|---|
| 16 | 4 | 0.586 ± 0.251 | 0.726 ± 0.259 | **+0.139** | 0.459 | 0.455 | 6/8 | unmeasured |
| 32 | 5 | 0.753 ± 0.045 | 0.875 ± 0.099 | **+0.122** | 0.139 | 0.137 | 6/8 | unmeasured |
| 64 | 6 | 0.834 ± 0.064 | 0.935 ± 0.014 | **+0.101** | 0.065 | 0.065 | 8/8 | DETECTABLE |
| 128 | 7 | 0.855 ± 0.134 | 0.978 ± 0.019 | **+0.123** | 0.140 | 0.139 | 6/8 | unmeasured |

## Reading it

The prediction is a **trend**, not a single contrast: the gain should rise
with octaves and with T. A row marked *(ceiling)* carries no information
either way -- both arms above 0.99 leaves nothing for a mechanism to buy.

## Verdict: the octave prediction is REFUTED; the length prediction holds

Loss-matched against final training loss (last-5-epoch mean from the
checkpoints), paired, n=8.

| grid | octaves | r(loss,acc) @1024 | raw gain @1024 | loss-matched | seeds + |
|---|---|---|---|---|---|
| 16 | 4 | **-0.933** | +0.139 | **-0.077** | 1/8 |
| 32 | 5 | -0.510 | +0.122 | +0.077 | 6/8 |
| 64 | 6 | **-0.167** | +0.101 | **+0.095** | **8/8 DETECTABLE** |
| 128 | 7 | -0.656 | +0.123 | +0.079 | 5/8 |

**Grid 16 is void, and not because it is a ceiling -- it is BIMODAL.** Vanilla
converges in 2 of 8 seeds (accuracy 0.98/0.99 at final loss 0.000) and fails in
the other 6 (0.37-0.51 at loss 0.16-0.26); MapPoPE converges in 4 of 8. So the
apparent +0.139 there is a difference in *convergence rate*, r(loss, acc) =
-0.933, and loss-matched the sign **flips negative** (1/8). Nothing
representational can be read from that row.

**The octave prediction fails on the three clean grids.** Loss-matched gains at
5, 6 and 7 octaves are +0.077 / +0.095 / +0.079 -- flat. The pre-registered
claim was that they should rise with the spectrum thinning, and they do not.

**The length prediction holds, 3/3.** At every clean grid the gain rises with
evaluation length: grid 32 gives +0.001 / +0.057 / +0.122 at T = 128 / 512 /
1024, grid 64 gives +0.007 / +0.037 / +0.101, grid 128 gives +0.004 / +0.042 /
+0.123.

### The corrected mechanism, and why the original was wrong

Grid size sets the **range** of the frequency spectrum, `[2pi/N, 2pi]`. It does
not set how many times the phase actually wraps. The highest-frequency block
wraps roughly once per cell of displacement regardless of N, so at fixed T the
agent's accumulated angle -- and therefore the number of wraps to disambiguate --
is essentially the same on a 16-grid and a 128-grid. I attributed the wrapping to
map extent when it is set by **path length**.

So what survives is: **PoPE's extra frequencies pay in proportion to accumulated
angle, which is temporal, not spatial.** That is supported 3/3 on the length axis
and refuted on the size axis, and the corrected statement is the one the data
chose rather than the one predicted.

### The honest caveat

"Helps more at out-of-distribution length" is this project's **universal
signature**. The rank effect, the InEKF, the forget gate and now PoPE all show it,
and it is the one axis on which everything that has ever helped here helps. So
this batch did not isolate PoPE's mechanism -- it added PoPE to that list. The
one prediction that *would* have been distinctive to the frequency-count account,
grid dependence, is the one that failed.
