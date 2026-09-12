# PAPERTASK rerun report (PAPERTASK_PREREG.md)

50 epochs cosine, logs kept. Raw accuracy, then the floor-normalised primary.

| condition | floor | Vanilla | VanillaEM_P0 | MapPoPE-Flat | Vanilla (norm) | VanillaEM_P0 (norm) | MapPoPE-Flat (norm) |
|---|---|---|---|---|---|---|---|
| IID  l=128 g=64  pe=0.5 | 0.522 | 0.968 | 0.985 | 1.000 | **0.932** | **0.969** | **0.999** |
| OOD-d l=64 g=32  pe=0.2 | 0.216 | 0.943 | 0.980 | 0.993 | **0.927** | **0.974** | **0.991** |
| OOD-s l=256 g=128 pe=0.8 | 0.803 | 0.984 | 0.988 | 0.998 | **0.921** | **0.940** | **0.989** |
| OOD-s l=512 g=128 pe=0.8 | 0.799 | 0.964 | 0.978 | 0.991 | **0.822** | **0.889** | **0.955** |
| ext-s l=1024 g=128 pe=0.8 | 0.801 | 0.927 | 0.964 | 0.978 | **0.632** | **0.818** | **0.891** |
| ext-s l=2048 g=128 pe=0.8 | 0.802 | 0.886 | 0.942 | 0.963 | **0.423** | **0.709** | **0.812** |

## Convergence check (before any verdict)

| arm | IID | >= 0.99 |
|---|---|---|
| Vanilla | 0.968 | **NO** |
| VanillaEM_P0 | 0.985 | **NO** |
| MapPoPE-Flat | 1.000 | yes |

-> **FAIL -- P1/P2 not interpreted; this is a budget-curve point only** (the 16-epoch batch gave WM 0.969)

## Contrasts, floor-normalised (paired by seed, n=8)

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| EM - WM norm OOD-s l=512 g=128 pe | +0.067 | 0.170 | 0.168 | 4/8 | unmeasured |
| PoPE - EM norm OOD-s l=512 g=128 pe | +0.066 | 0.080 | 0.079 | 6/8 | unmeasured |
| EM - WM norm ext-s l=1024 g=128 p | +0.186 | 0.187 | 0.185 | 8/8 | DETECTABLE |
| PoPE - EM norm ext-s l=1024 g=128 p | +0.073 | 0.091 | 0.090 | 6/8 | unmeasured |
| EM - WM norm ext-s l=2048 g=128 p | +0.287 | 0.196 | 0.194 | 8/8 | DETECTABLE |
| PoPE - EM norm ext-s l=2048 g=128 p | +0.102 | 0.129 | 0.128 | 6/8 | unmeasured |

## Rule 9 -- the check the deleted logs made impossible

rule 9: r(final loss, acc) = -0.461 over 24 runs; acc = 0.942 -0.152*loss, resid sd 0.036

Mean final loss: Vanilla 0.1306, VanillaEM_P0 0.0762, MapPoPE-Flat 0.0168

## Registered verdicts

- **P1 (the effect is real)**: EM - WM norm at l=2048 = +0.287 (MDE 0.194; needs >= +0.20 AND detectable) -> **NOT READ (convergence failed)**
- **P3 (per-pair counterexample)**: PoPE - EM norm at l=2048 = +0.102 (MDE 0.128) -> **still unmeasured, as predicted**
- **P4 (falsifier)**: convergence FAILED and the l=2048 gap is above its MDE -> **does not fire**
