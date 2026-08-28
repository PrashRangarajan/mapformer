# GateDelta capacity control

GateDeltaCtl = GateDelta's parameters (3,206,682, identical) with the gate multiplied out: functionally identical to Vanilla (max|diff| 0.00e+00), gate gradient exactly zero. If Ctl ~= GateDelta the win is capacity/RNG; if Ctl ~= Vanilla the win is the GATE.

## T=512

| grid | seed | Vanilla | GateDeltaCtl | GateDelta | Gate−Ctl | all converged? |
|---|---|---|---|---|---|---|
| 16 | 0 | 0.947 <sub>(0.17)</sub> | 0.860 <sub>(0.33)</sub> | 0.591 <sub>(0.67)</sub> | -0.269 | no |
| 16 | 1 | 0.949 <sub>(0.16)</sub> | 0.721 <sub>(0.55)</sub> | 1.000 <sub>(0.01)</sub> | +0.279 | no |
| 16 | 2 | 0.761 <sub>(0.44)</sub> | 0.612 <sub>(0.76)</sub> | 0.694 <sub>(0.62)</sub> | +0.082 | no |
| 24 | 0 | 0.623 <sub>(0.76)</sub> | 0.565 <sub>(0.86)</sub> | 0.745 <sub>(0.53)</sub> | +0.180 | no |
| 24 | 1 | 0.838 <sub>(0.36)</sub> | 0.978 <sub>(0.08)</sub> | 1.000 <sub>(0.00)</sub> | +0.022 | YES |
| 24 | 2 | 0.621 <sub>(0.77)</sub> | 0.567 <sub>(0.86)</sub> | 0.633 <sub>(0.74)</sub> | +0.065 | no |
| 32 | 0 | 0.559 <sub>(0.85)</sub> | 0.969 <sub>(0.12)</sub> | 0.668 <sub>(0.69)</sub> | -0.300 | no |
| 32 | 1 | 0.967 <sub>(0.13)</sub> | 0.758 <sub>(0.56)</sub> | 1.000 <sub>(0.00)</sub> | +0.242 | no |
| 32 | 2 | 0.582 <sub>(0.82)</sub> | 0.565 <sub>(0.85)</sub> | 0.992 <sub>(0.11)</sub> | +0.428 | no |

- **GateDelta − Control, pooled** (9): **+0.081**
- **GateDelta − Control, all-converged** (1): **+0.022**

## T=1024

| grid | seed | Vanilla | GateDeltaCtl | GateDelta | Gate−Ctl | all converged? |
|---|---|---|---|---|---|---|
| 16 | 0 | 0.786 <sub>(0.17)</sub> | 0.709 <sub>(0.33)</sub> | 0.386 <sub>(0.67)</sub> | -0.323 | no |
| 16 | 1 | 0.803 <sub>(0.16)</sub> | 0.558 <sub>(0.55)</sub> | 1.000 <sub>(0.01)</sub> | +0.442 | no |
| 16 | 2 | 0.548 <sub>(0.44)</sub> | 0.437 <sub>(0.76)</sub> | 0.518 <sub>(0.62)</sub> | +0.081 | no |
| 24 | 0 | 0.394 <sub>(0.76)</sub> | 0.375 <sub>(0.86)</sub> | 0.522 <sub>(0.53)</sub> | +0.147 | no |
| 24 | 1 | 0.694 <sub>(0.36)</sub> | 0.846 <sub>(0.08)</sub> | 1.000 <sub>(0.00)</sub> | +0.154 | YES |
| 24 | 2 | 0.430 <sub>(0.77)</sub> | 0.346 <sub>(0.86)</sub> | 0.457 <sub>(0.74)</sub> | +0.110 | no |
| 32 | 0 | 0.320 <sub>(0.85)</sub> | 0.824 <sub>(0.12)</sub> | 0.424 <sub>(0.69)</sub> | -0.399 | no |
| 32 | 1 | 0.828 <sub>(0.13)</sub> | 0.582 <sub>(0.56)</sub> | 1.000 <sub>(0.00)</sub> | +0.418 | no |
| 32 | 2 | 0.366 <sub>(0.82)</sub> | 0.395 <sub>(0.85)</sub> | 0.944 <sub>(0.11)</sub> | +0.549 | no |

- **GateDelta − Control, pooled** (9): **+0.131**
- **GateDelta − Control, all-converged** (1): **+0.154**



## CORRECTION (2026-08-27)

The 'all-converged (1): +0.022' row is n=1 and should not be read as a number
(MDE > 0.5). More importantly, the framing was wrong: because the gate output is
PROVABLY multiplied out (zero grad, zero grad-norm contribution, bit-identical
function), capacity is ruled out A PRIORI rather than tested. What GateDeltaCtl
actually varies is training RNG -- it is a second Vanilla seed.

That makes it valuable for a different reason: **Ctl-vs-Vanilla IS a direct
measurement of this setup's run-to-run variance.** mean|delta| 0.150 (T=512) /
0.163 (T=1024), sd 0.198/0.230, range -0.228..+0.410. Use it as the error bar
for every effect measured in this environment.
