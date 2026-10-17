# Does the sign cost follow the relative sign of the moves? -- results (registered, `GAUGE_PREREG.md` + Amendment 1)

**CORRECTED 2026-10-16** (numbers audit of draft 5, `docs/paper/review_draft5/NUMBERS_AUDIT.md` error 8): on +1/-1, 89.3% of
revisits are backtracks on the registered eval stream (138,800 of 155,456, `GAUGE_ANALYSIS.txt`); the 87.5% below is the pre-launch audit's sample. Body unchanged.

Batch 2026-10-15 21:00-22:40, 40 runs, seeds 1300-1307; ring 32 cells, observation-step-free SignedAO_r4 (SO) / AbsAO_r4
(AO, |.| step), RING_DIAL recipe, 300 epochs; moves +1/-1 (`_m`, an inverse pair) vs +1/+2 (`_p`, both forward; revisits
only from laps). Analysis `analyze_gauge.py` -> `GAUGE_ANALYSIS.txt`, `GAUGE_VERDICTS.json`. Independent code check before
launch (`docs/audits/2026-10-15/gauge_audit.md`, Amendment 1); results read only after it. `experiment_audit`: every arm's
windowed classes agree with the analysis (SO 13 SOLVED, AO 3, RoPE 0).

**Reading (registered): AS FALLS.** The sign cost replicates on the inverse pair; whether it disappears when the moves step
the same way is UNMEASURED, because +1/+2 is harder for the signed model too.

## Registered decisions
| decision | contrast | result | label |
|---|---|---|---|
| G2 | AO_m vs SO_m (+1/-1) | 0.931, SOLVED **1/8** vs 1.000, **8/8**; d -0.069 [95% CI -0.095, -0.045], perm p 0.0014, Fisher p 0.0014 | **SIGNED LEARNS BETTER** |
| G1 (primary) | AO_p vs SO_p (+1/+2) | 0.883, SOLVED 2/8 vs 0.947, 5/8; d -0.064 [-0.200, +0.075], perm p 0.34, Fisher p 0.31 | **UNMEASURED** |
| G3 | AO_p vs AO_m, SOLVED | 2/8 vs 1/8, Fisher p 1.00 | NO DIFFERENCE DETECTED |

Gates passed (SO_m 8/8, SO_p 5/8 >= 4). SUPPORTED needed G1 = NO SIGN COST, G2 and G3; NOT SUPPORTED needed G1 = SIGNED
LEARNS BETTER. Neither holds.

## What this shows
1. **The sign cost replicates on fresh seeds** (third ring replication; G2): |.| solves 1/8 against 8/8, every signed run at
   1.000, on the +1/-1 walk of RING_DIAL / MECHANISM.
2. **+1/+2 needs integration.** Index RoPE: 0.649, 0/8 SOLVED (floor: predict blank ~0.50); 0.468 at 4T.
3. **On +1/+2 the signed model learns a sign-coherent code** (declared secondary): 0.92 of channels end with both (wrapped)
   move angles of one sign (per seed 0.57-1.00; initial ~0.50-0.70), against 0.01 on +1/-1. What signed steps learn follows
   the moves' relative sign, as Lemma 1 allows: on +1/+2 the solution they find lies in the |.| class up to the gauge.
4. **The |.| model still does not learn +1/+2 reliably** (2/8) although such a code exists and needs no zero crossing.
   The registered prediction that the cost disappears is NOT confirmed; it is not refuted either: the signed model also
   finds +1/+2 harder (5/8 vs 8/8 on +1/-1), and the CI on the difference spans -0.20..+0.08.

## Declared secondaries (no verdicts)
| cell | mean acc T | SOLVED | acc 4T | loss AUC | backtrack | lap | gap < 16 | gap >= 16 |
|---|---|---|---|---|---|---|---|---|
| SO_m | 1.0000 | 8/8 | 0.997 | 0.090 | 1.000 | 1.000 | 1.000 | 1.000 |
| AO_m | 0.9309 | 1/8 | 0.871 | 0.537 | 0.936 | 0.893 | 0.952 | 0.902 |
| SO_p | 0.9470 | 5/8 | 0.913 | 0.515 | -- | 0.947 | -- | 0.947 |
| AO_p | 0.8834 | 2/8 | 0.863 | 0.661 | -- | 0.884 | -- | 0.884 |
| RoPE_p | 0.6494 | 0/8 | 0.468 | 1.341 | -- | 0.651 | -- | 0.651 |

Revisit strata (Amendment 1): on +1/-1, 87.5% of revisits are backtracks; on +1/+2 every revisit is a lap with gap >= 16.
On the comparable stratum (laps), |.| scores about the same on both tasks (0.893 vs 0.884) and signed drops from 1.000 to
0.947: the difficulty +1/+2 adds lies in the task (long-gap laps), and on laps the |.| model is no better when the moves
do not oppose. Descriptive, post hoc reading of a declared secondary; the point estimates of the sign gap are about equal
on the two tasks (-0.069, -0.064), so nothing here suggests the cost shrinks without an inverse pair -- but G1 cannot say so.

Per seed (accuracy at T = 128, class):
```
1300 | SO_m 1.0000 SOLV | AO_m 0.9039 STAL | SO_p 0.9873 STAL | AO_p 0.9874 DESC | RoPE_p 0.6419 STAL
1301 | SO_m 1.0000 SOLV | AO_m 0.9530 DESC | SO_p 1.0000 SOLV | AO_p 0.9095 STAL | RoPE_p 0.6627 STAL
1302 | SO_m 1.0000 SOLV | AO_m 0.9573 STAL | SO_p 1.0000 SOLV | AO_p 0.9039 STAL | RoPE_p 0.6545 STAL
1303 | SO_m 1.0000 SOLV | AO_m 1.0000 SOLV | SO_p 1.0000 SOLV | AO_p 0.5901 STAL | RoPE_p 0.6608 STAL
1304 | SO_m 1.0000 SOLV | AO_m 0.9220 STAL | SO_p 0.9354 STAL | AO_p 0.7680 STAL | RoPE_p 0.6542 STAL
1305 | SO_m 1.0000 SOLV | AO_m 0.9282 STAL | SO_p 1.0000 SOLV | AO_p 1.0000 SOLV | RoPE_p 0.6017 STAL
1306 | SO_m 1.0000 SOLV | AO_m 0.8801 STAL | SO_p 0.6536 DESC | AO_p 0.9078 STAL | RoPE_p 0.6598 STAL
1307 | SO_m 1.0000 SOLV | AO_m 0.9025 STAL | SO_p 1.0000 SOLV | AO_p 1.0000 SOLV | RoPE_p 0.6595 STAL
```
Three runs were still descending at 300 epochs (AO_m 1301, AO_p 1300, SO_p 1306): budget-scoped, as every ring result.

Pilot (seed 1390, outside the batch, never pooled): SO_p 1.000 SOLVED, AO_p 0.745 unsolved, RoPE_p 0.657 -- in line.

## Consequences
- Lemma 1 (the relative-sign gauge) is an identity and stands; its corollaries about what the |.| class can REPRESENT stand.
  Its LEARNABILITY prediction (no cost without an inverse pair) is untested at this power and must not be stated as a
  prediction the data bear out. Paper: contribution 3 and Sec. 4.5 rewritten accordingly.
- The barrier-at-zero account explains why |.| cannot follow the signed route on an inverse pair (G2, MECHANISM); it does
  not explain why |.| also struggles on +1/+2, where no zero crossing is needed. Open; a larger budget or a +1/+2 task as
  easy as +1/-1 for the signed model would be needed to read G1.
