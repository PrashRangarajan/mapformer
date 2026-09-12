# PAIRORIGIN report (PAIRORIGIN_PREREG.md)

## Per-arm

| arm | acc T=1024 | acc T=2048 | final loss |
|---|---|---|---|
| EMPair_r4 | **0.880 +/- 0.136** | 0.784 | 0.422 |
| VanillaEM_P0_r4 | **0.600 +/- 0.126** | 0.510 | 1.340 |
| Vanilla_r4 | **0.975 +/- 0.072** | 0.947 | 0.064 |

## Contrasts (paired by seed, n=8)

| contrast | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| EMPair - P0 (P1/P3) | +0.280 | 0.218 | 0.216 | 7/8 | DETECTABLE |
| EMPair - WM (P2) | -0.095 | 0.137 | 0.135 | 0/8 | unmeasured |
| P0 - WM (the gap being closed) | -0.375 | 0.156 | 0.154 | 0/8 | DETECTABLE |
| EMPair - P0 at T=2048 | +0.273 | 0.234 | 0.231 | 7/8 | DETECTABLE |

## Rule 9

rule 9: r(final loss, acc) = -0.983 over 24 runs; acc = 1.012 -0.319*loss, resid sd 0.036
  WARNING |r| > 0.98: held-out accuracy is an affine readout of training loss, so any raw 'effect' is a loss gap. The loss-matched residual is the honest contrast -- BUT loss-matching conditions on a MEDIATOR: a function-class limit also shows up as worse fit, so a zero residual cannot by itself distinguish optimisation from representation (AUDIT_2026-09-10.md finding 7). That needs an existence construction (rule 29), not this statistic.

## Registered verdicts

- **P1 (kernel sharing)**: EMPair - P0 = +0.280 (MDE 0.216, needs >= +0.20 AND detectable) -> **CONFIRMED**
- **P2 (recovery to WM)**: EMPair - WM = -0.095 (MDE 0.135) -> **MET (within MDE)**
- **P3 (per-token search)**: |EMPair - P0| below MDE -> **NOT CONFIRMED**

## P4 (mechanism) and manipulation check 3

| arm | rewind fraction of solved cells | phase spread across pairs |
|---|---|---|
| EMPair_r4 | 0.185 | 0.000 |
| VanillaEM_P0_r4 | 0.962 | 0.000 |

P4: EMPair's solved cells use the per-token rewind -0.777 vs P0. Lower is what P1 predicts (per-pair freedom should retrieve WITHOUT moving the query token).
Check 3: EMPair phase spread must be > 0 (P0 is 0.000 by construction) -> FAIL

## Manipulation checks 1-2 (from the batch)

```
CHECK 1 same function at init: max|logit diff| 0.000e+00 -> PASS
CHECK 2 origin pathway moved: min over 16 tensors 3.272e-01, max 8.408e-01 -> PASS
```
