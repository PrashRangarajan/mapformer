> **CORRECTION 2026-09-23 (audit).** This file, and every summary built on it, quoted
> the position effect at **L128 D12 -- the cell where it is SMALLEST**. The full A2
> position main effect by cell, recomputed from `DYCK_DEPTH_RESULTS.json`:
>
> | cell | 1 layer | 2 layers |
> |---|---|---|
> | **L32 D4 (the TRAINING cell)** | **+0.357** (MDE 0.005, 8/8) | **+0.081** (MDE 0.003, 8/8) |
> | L32 D12 (depth x3, length matched) | +0.343 (MDE 0.027, 8/8) | **+0.209** (MDE 0.010, 8/8) |
> | L128 D4 | +0.213 (MDE 0.099, 8/8) | +0.080 (MDE 0.068, 7/8) |
> | L128 D12 (quoted everywhere) | +0.136 (MDE 0.051, 8/8) | +0.048 (MDE 0.042, 7/8) |
>
> **Dyck's position effect is LARGEST IN DISTRIBUTION and SHRINKS with length.** It is a
> depth/stack effect that degrades under extrapolation, not a length-extrapolation effect.
>
> **And F1 does not merely inflate it -- F1 INVERTS ITS SHAPE.** Same checkpoints, same
> sequences, 1 layer, training cell -> L128 D12: **A2 falls 2.6x (+0.357 -> +0.136)
> while F1 rises 4.1x (+0.075 -> +0.306)**. The "path integration helps out of
> distribution" signature on Dyck is manufactured by the metric out of data whose
> chance-anchored reading says the opposite.

# READING

**E2: the frequency-ladder confound is DEAD.** F3 does not fire -- giving the index
arms the path arms' ladder (slowest wavelength 47,116 -> 180 tokens) closes -3% to
+3% of the gap. A real uncontrolled difference that explains nothing. This
STRENGTHENS the position claim. Reproduction control is exact: fresh
`RoPE-1L_b10000` gives A2 0.535 and CE 1.0248 against the stored 0.535 / 1.0248.

**E1: SPLIT across the two registered primary readouts.**
- **A2 (primary)**: position +0.1357 (1L, 8/8, DET) -> **+0.0482 (2L, 7/8, DET)**.
  Shrinks 65% and SURVIVES.
- **A3 d33+ (co-primary)**: +0.1165 (1L, 8/8, DET) -> **-0.0002 (2L, 4/8,
  unmeasured)**. ELIMINATED.
So falsifier F1 fires on the co-primary and not on the primary.

**Depth LEVELS rather than substitutes.** Position main effect by bucket:
| | d 9-32 | d 33+ |
|---|---|---|
| 1 layer | +0.278 | +0.117 |
| 2 layers | +0.250 | **-0.0005** |
At mid distance path integration wins ~0.25 whatever the depth. At the longest
distance depth converges the arms to ~0.60 by LIFTING the index arms (RoPE 0.539
-> 0.592) and LOWERING the path arms (MapPoPE 0.730 -> 0.640). **MapPoPE-1L
(0.730) is still the single best arm at d33+, better than every 2-layer arm**, so
depth is not a dominating substitute.

**At 2 layers and long distance the ENCODING is the only detectable axis**
(+0.0482, 7/8) while position is exactly zero.

**My registered prediction FAILED.** I predicted position would collapse on the
RoPE row and survive on the PoPE row, i.e. a detectable interaction. The
interaction is unmeasured at 2 layers on all three metrics (+0.0196 / +0.0051 /
+0.0089). Sixth failed prediction in this line.

**Second independent indictment of F1.** At 2 layers F1 COLLAPSES for every arm
(MapWM 0.868 -> 0.617, RoPE 0.567 -> 0.497) while the stack metrics IMPROVE,
because invalid mass roughly doubles (MapWM 0.054 -> 0.113). Two-layer models
rank the right closer better and spread more mass on illegal tokens; F1 punishes
that. Do not use F1 as a headline -- now demonstrated twice.

**Convergence**: no arm's median slope crosses the registered -0.005/1k. Not void.

# Dyck: depth-matched 2x2 (E1) and the frequency ladder (E2)

Pre-registration `DYCK_DEPTH_PREREG.md`. Cell L128 D12. **Primary readout A2 = Hewitt distance-averaged closing accuracy, chance 0.500.** F1 and invalid mass alongside, never selectively.

## Levels

| arm | n | A2 (primary) | A3 d33+ | A3 d9-32 | F1 | invalid mass |
|---|---|---|---|---|---|---|
| MapWM-1L | 8 | 0.638 | 0.611 | 0.799 | 0.868 | 0.054 |
| MapPoPE-1L | 8 | 0.719 | 0.730 | 0.971 | 0.926 | 0.018 |
| RoPE-1L | 8 | 0.535 | 0.539 | 0.570 | 0.567 | 0.220 |
| PoPE-1L | 8 | 0.551 | 0.569 | 0.645 | 0.616 | 0.195 |
| MapWM-2L | 8 | 0.612 | 0.589 | 0.836 | 0.617 | 0.113 |
| MapPoPE-2L | 8 | 0.636 | 0.640 | 0.959 | 0.601 | 0.099 |
| RoPE-2L | 8 | 0.574 | 0.592 | 0.602 | 0.497 | 0.229 |
| PoPE-2L | 8 | 0.578 | 0.638 | 0.694 | 0.472 | 0.281 |
| RoPE-1L_b32 | 8 | 0.532 | 0.542 | 0.548 | 0.559 | 0.227 |
| PoPE-1L_b32 | 8 | 0.546 | 0.559 | 0.625 | 0.711 | 0.151 |
| RoPE-1L_b128 | 8 | 0.538 | 0.538 | 0.560 | 0.559 | 0.223 |
| PoPE-1L_b128 | 8 | 0.550 | 0.564 | 0.661 | 0.660 | 0.178 |
| RoPE-1L_b10000 | 8 | 0.535 | 0.539 | 0.570 | 0.567 | 0.220 |
| PoPE-1L_b10000 | 8 | 0.551 | 0.569 | 0.645 | 0.616 | 0.195 |

## E1 -- does the position effect survive at 2 layers?

**1 LAYER (stored)** (A2_close_acc_dist)
- POSITION main: +0.1357 (MDE 0.0507, 8/8 positive) **DETECTABLE**
- ENCODING main: +0.0487 (MDE 0.0532, 6/8 positive) unmeasured
- INTERACTION: +0.0646 (MDE 0.1182, 6/8 positive) unmeasured

**2 LAYERS (new)** (A2_close_acc_dist)
- POSITION main: +0.0482 (MDE 0.0417, 7/8 positive) **DETECTABLE**
- ENCODING main: +0.0138 (MDE 0.0384, 4/8 positive) unmeasured
- INTERACTION: +0.0196 (MDE 0.0735, 5/8 positive) unmeasured

**1 LAYER (stored)** (A3_d33+)
- POSITION main: +0.1165 (MDE 0.0694, 8/8 positive) **DETECTABLE**
- ENCODING main: +0.0744 (MDE 0.0602, 6/8 positive) **DETECTABLE**
- INTERACTION: +0.0886 (MDE 0.1455, 6/8 positive) unmeasured

**2 LAYERS (new)** (A3_d33+)
- POSITION main: -0.0002 (MDE 0.0497, 4/8 positive) unmeasured
- ENCODING main: +0.0482 (MDE 0.0410, 7/8 positive) **DETECTABLE**
- INTERACTION: +0.0051 (MDE 0.1108, 3/8 positive) unmeasured

**1 LAYER (stored)** (C1_paper_f1)
- POSITION main: +0.3059 (MDE 0.0463, 8/8 positive) **DETECTABLE**
- ENCODING main: +0.0536 (MDE 0.0296, 8/8 positive) **DETECTABLE**
- INTERACTION: +0.0094 (MDE 0.0729, 4/8 positive) unmeasured

**2 LAYERS (new)** (C1_paper_f1)
- POSITION main: +0.1243 (MDE 0.0585, 8/8 positive) **DETECTABLE**
- ENCODING main: -0.0203 (MDE 0.0554, 3/8 positive) unmeasured
- INTERACTION: +0.0089 (MDE 0.0772, 4/8 positive) unmeasured

## E2 -- does giving the index arms the path arms' frequency ladder close the gap?

Fresh base-10000 controls: RoPE 0.535 (stored 0.535), PoPE 0.551 (stored 0.551)

| ladder | RoPE-1L | closes | PoPE-1L | closes |
|---|---|---|---|---|
| base 10000 | 0.535 | +0% | 0.551 | +0% |
| base 32 | 0.532 | -2% | 0.546 | -3% |
| base 128 | 0.538 | +3% | 0.550 | -1% |

Gap to close: MapWM-1L - RoPE-1L(b10000) = +0.103; MapPoPE-1L - PoPE-1L(b10000) = +0.168

**F3 fires if either ladder closes >= 50% of its gap.**

