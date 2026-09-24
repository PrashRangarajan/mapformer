> **CORRECTED 2026-09-24 (audit).** L32 D12 is NOT the training cell: training is L32 **D4**
> (`train_dyck.py`, the paper's recipe), so D12 is matched in LENGTH but 3x the training
> nesting DEPTH -- an extrapolation in depth. At the training cell L32 D4 the position effect
> is +0.293 / +0.081 / +0.048 / +0.019 at 1-4 layers (8/8 each; index RoPE 0.979 at 4L): real
> but shrinking to a ceiling with depth. The large depth-resistant effect below exists only at
> unseen depth. The convergence line uses an arm-median final slope, not the SOLVED/STALLED
> classes, and the index arms' losses are still falling. Read "training length" below as
> "training length, 3x training depth".

# READING -- the last standing positive result SURVIVES depth

Pre-registration `DYCK_LADDER_PREREG.md`. All arms converged (no median slope below
-0.005/1k). Width fixed at n_heads=2 / d=128; only depth varies.

## F2 fires; F1 does not

At **L32 D12** -- the co-primary with real headroom, at the TRAINING length:

| depth | index arms (RoPE / PoPE) | path arms (MapWM / MapPoPE) | POSITION main |
|---|---|---|---|
| 1L | 0.657 / 0.667 | 0.915 / 0.988 | **+0.290** (MDE 0.033, 8/8) |
| 2L | 0.740 / 0.765 | 0.937 / 0.987 | **+0.209** (MDE 0.010, 8/8) |
| 3L | 0.760 / 0.777 | 0.934 / 0.920 | **+0.159** (MDE 0.035, 8/8) |
| 4L | 0.755 / 0.781 | 0.923 / 0.949 | **+0.168** (MDE 0.024, 8/8) |

**The index arms STOP CLIMBING** at ~0.76-0.78 after two to three layers, while the
path arms hold at ~0.92-0.95. The position effect falls by ~42% from 1L to 3L and
then **PLATEAUS** (3L +0.159 and 4L +0.168 are within each other's MDE). At 4 layers
path integration still buys **+0.168, 8/8 seeds**. Depth does not remove it.

**F1 (kills the claim) did not fire**: position(4L) is far outside its MDE at L32 D12.
**F2 fired**: the effect plateaus while the index arms stop climbing. Path integration
buys something depth does not, on a task whose positional variable is signed, at
matched length, converged, floor-reported, width-fixed, 8 seeds, one batch.

## Why L32 D4 shrinks toward zero -- a ceiling, not depth substitution

At L32 D4 the path arms are at 0.995-0.999 from ONE layer. The position effect falls
+0.293 -> +0.081 -> +0.048 -> +0.019 because the index arms climb into the ceiling
(0.699 -> 0.979); at 4L the largest effect the cell could show is 1 - 0.979 = 0.021.
The pre-registration named D4 as the weaker readout for exactly this reason. It is
still detectable at every depth (8/8).

## F3, the width confound: CONFIRMED, and it overstated depth substitution

At fixed width the 1L training-cell effect is **+0.293, not the +0.357** measured when
the 1L arms were narrower (n_heads=1, d=64). The old ladder's "4.4x drop from 1L to 2L,
depth substitutes" mixed depth with width and a 7.8x parameter change. **At fixed width
on the deep cell, depth takes the effect only from +0.290 to +0.168 over three extra
layers, and then stops.** The earlier "depth substitution" reading is WITHDRAWN.

## It does NOT extrapolate in length

At L128 D12 (reference only) the effect is +0.091 at 1L, +0.048 at 2L, and **gone at 3L
and 4L** (-0.000, +0.002), with every arm near 0.59-0.61 against a floor of ~0.51. The
advantage is an in-distribution stack effect, consistent with the training cell being
where it is largest.

## Caveats

- The plateau rests on two depths (3L, 4L). Four depths fix the shape better than the
  two-point line this batch replaced, but 6-8 layers are untested.
- The path arms do not improve with depth and may slightly degrade (MapWM 0.937 ->
  0.923; MapPoPE non-monotone, 0.920 at 3L). The advantage persists because the index
  arms plateau LOWER, not because path integration keeps gaining.
- One task, one width, 8 seeds.

# Dyck depth ladder at FIXED width

n_heads=2, d_model=128 throughout; only depth varies. 4 arms x 8 seeds x 4 depths, one batch.
Primary A2 (chance 0.500) at the TRAINING length.

## L32D4  (co-primary)

| depth | RoPE | PoPE | MapWM | MapPoPE | POSITION main | ENCODING main | interaction |
|---|---|---|---|---|---|---|---|
| 1L | 0.699 | 0.707 | 0.995 | 0.997 | +0.293 (0.004, 8/8)** | +0.005 (0.003, 7/8)** | -0.005 (0.004, 0/8)** |
| 2L | 0.914 | 0.919 | 0.998 | 0.997 | +0.081 (0.003, 8/8)** | +0.002 (0.003, 6/8) | -0.007 (0.004, 0/8)** |
| 3L | 0.950 | 0.950 | 0.999 | 0.997 | +0.048 (0.004, 8/8)** | -0.001 (0.004, 2/8) | -0.003 (0.008, 4/8) |
| 4L | 0.979 | 0.980 | 0.999 | 0.999 | +0.019 (0.003, 8/8)** | +0.000 (0.002, 4/8) | -0.002 (0.004, 3/8) |

`**` = clears MDE. Cells show mean (MDE, seeds positive).

## L32D12  (co-primary)

| depth | RoPE | PoPE | MapWM | MapPoPE | POSITION main | ENCODING main | interaction |
|---|---|---|---|---|---|---|---|
| 1L | 0.657 | 0.667 | 0.915 | 0.988 | +0.290 (0.033, 8/8)** | +0.041 (0.035, 8/8)** | +0.063 (0.069, 7/8) |
| 2L | 0.740 | 0.765 | 0.937 | 0.987 | +0.209 (0.010, 8/8)** | +0.038 (0.010, 8/8)** | +0.024 (0.029, 7/8) |
| 3L | 0.760 | 0.777 | 0.934 | 0.920 | +0.159 (0.035, 8/8)** | +0.002 (0.029, 4/8) | -0.031 (0.048, 3/8) |
| 4L | 0.755 | 0.781 | 0.923 | 0.949 | +0.168 (0.024, 8/8)** | +0.025 (0.044, 6/8) | -0.001 (0.105, 6/8) |

`**` = clears MDE. Cells show mean (MDE, seeds positive).

## L128D12  (extrapolation, reference only)

| depth | RoPE | PoPE | MapWM | MapPoPE | POSITION main | ENCODING main | interaction |
|---|---|---|---|---|---|---|---|
| 1L | 0.539 | 0.558 | 0.615 | 0.663 | +0.091 (0.022, 8/8)** | +0.034 (0.038, 6/8) | +0.029 (0.085, 5/8) |
| 2L | 0.574 | 0.578 | 0.612 | 0.636 | +0.048 (0.042, 7/8)** | +0.014 (0.038, 4/8) | +0.020 (0.074, 5/8) |
| 3L | 0.585 | 0.597 | 0.594 | 0.587 | -0.000 (0.021, 4/8) | +0.003 (0.026, 3/8) | -0.018 (0.053, 4/8) |
| 4L | 0.590 | 0.612 | 0.600 | 0.607 | +0.002 (0.030, 4/8) | +0.014 (0.036, 6/8) | -0.015 (0.070, 4/8) |

`**` = clears MDE. Cells show mean (MDE, seeds positive).

## Convergence

- 1L: median final slope -0.00163/1k, worst -0.01016 (void if median < -0.005)
- 2L: median final slope -0.00203/1k, worst -0.00743 (void if median < -0.005)
- 3L: median final slope -0.00216/1k, worst -0.00562 (void if median < -0.005)
- 4L: median final slope -0.00223/1k, worst -0.00764 (void if median < -0.005)
