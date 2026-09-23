# Pre-registration: is Dyck's in-distribution position effect the signed accumulator, or 1-layer capacity?

Written before any run. This is the last standing positive result in the language
line, so it decides whether that line has an in-distribution claim at all.

## Why, and a confound in my own earlier ladder

`DYCK_DEPTH_RESULTS.md` compared 1L against 2L and found the A2 position main
effect dropping 4.4x at the training cell (+0.357 -> +0.081). I read that as
depth substitution. **It is not a clean depth comparison**: the 1L arms were run
at n_heads=1 / d_model=64 / ~51k params and the 2L arms at n_heads=2 /
d_model=128 / ~398k. Depth, width, head count and a 7.8x parameter difference all
move together. **The existing "depth substitutes" reading is confounded** and is
withdrawn pending this batch.

## Design

**Fixed width throughout: n_heads=2, d_model=128.** Only DEPTH varies.

Depths **1, 2, 3, 4** x arms **RoPE, PoPE, MapWM, MapPoPE** x **8 seeds** = 128
runs, **all in one batch** (rule 3 -- the existing 2L arms are retrained here, not
reused). Params scale linearly with depth and are matched across arms to <0.25%:

| | 1L | 2L | 3L | 4L |
|---|---|---|---|---|
| RoPE | 199,813 | 398,085 | 596,357 | 794,629 |
| MapPoPE | 200,581 | 398,981 | 597,381 | 795,781 |

Recipe identical to `runs/dyck_depth`. ~76 s per run.

## Primary readout, fixed now

**A2 (Hewitt distance-averaged closing accuracy, chance 0.500) at L32 D4 and at
L32 D12** -- both at the TRAINING length, so there is no extrapolation confound
and no exposure to the F1 shape inversion documented in `DYCK_DEPTH_RESULTS.md`.

**L32 D12 is the better co-primary**: at 2 layers the index arms sit at 0.740 /
0.765 with real headroom, and the position effect there is +0.209. At L32 D4 the
path arms are already at 0.997, so the largest effect the cell can show is 0.086.
F1 and invalid mass reported alongside, never selectively. Floor from
`DYCK_GATES.md` beside every cell. MDE = 2.8*sd/sqrt(8) per paired contrast.

## Falsifiers

- **F1, kills the last positive result.** If `position(4L)` is within its MDE at
  BOTH cells, Dyck's effect is depth-substitution and **the language line has no
  surviving in-distribution positive result.**
- **F2.** If the effect plateaus (roughly flat from 2L to 4L) while the index arms
  stop climbing, path integration buys something depth does not, on a task whose
  positional variable is signed. That is a converged, floor-reported,
  matched-length claim.
- **F3, the confound check.** If `position(1L)` at n_heads=2 differs materially
  from the +0.357 measured at n_heads=1, the original ladder was measuring width
  and the 4.4x drop must be re-attributed.

## Registered prediction

I will not predict the direction. Six of my seven mechanistic predictions in this
line have failed, and the one structural prediction I made for the earlier ladder
(collapse on the RoPE row, survival on the PoPE row) was unmeasured on all three
metrics. Recording that abstention rather than manufacturing a forecast.

## Void conditions

Any arm unconverged by the registered slope rule (-0.005/1k); arms split across
batches; a gate regression.
