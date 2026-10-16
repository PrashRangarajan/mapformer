# Is the sign learnability cost one recipe or one scale? -- results (registered, `ROBUST_PREREG.md` + Amendment 1)

Batch 2026-10-16 04:07-14:26, 80 runs, seeds 1400-1407 (fresh). Observation-step-free models, signed step (SO) vs |.| step
(AO). Ring: `train_gauge.py --deltas=1,-1` (32 cells, T = 128, 300 epochs; the GAUGE m-cell / WRAP_EXISTENCE E3 setting) in
four recipes; torus: `train_variant` with GENERALITY's recipe (T = 1024, 900 epochs) at 2 layers, d_model 256 (1.59M
parameters vs 0.20M). Independent check before launch (`docs/audits/2026-10-16/robust_audit.md`, Amendment 1); torus
pilot on seed 1490 (SO SOLVED, AO DESCENDING; launch rule met). Analysis `analyze_robust.py` -> `ROBUST_ANALYSIS.txt`,
`ROBUST_VERDICTS.json`. `experiment_audit`: windowed classes agree (torus AO 2 SOLVED / 6 DESCENDING; ring AO 5 SOLVED,
SO 30).

**Reading (registered): PARTLY UNMEASURED -- the cost holds in four of five cells (default, weight decay 0, lr /3, torus at
2 layers d 256); at lr x3 it is UNMEASURED and much smaller.**

## Registered decisions (AO vs SO; Fisher on SOLVED, permutation on accuracy)
| cell | recipe | AO acc, SOLVED | SO acc, SOLVED | d [95% CI] | perm p | Fisher p | label |
|---|---|---|---|---|---|---|---|
| ring def | lr 1e-3, wd 0.05 | 0.911, 1/8 | 1.000, 8/8 | -0.089 [-0.125, -0.055] | 0.0002 | 0.0014 | **SIGNED LEARNS BETTER** |
| ring wd0 | lr 1e-3, wd 0 | 0.873, 0/8 | 1.000, 8/8 | -0.127 [-0.200, -0.075] | 0.0002 | 0.0002 | **SIGNED LEARNS BETTER** |
| ring lrx3 | lr 3e-3, wd 0.05 | 0.987, 4/8 | 1.000, 8/8 | -0.013 [-0.025, -0.005] | 0.026 | 0.077 | **UNMEASURED** |
| ring lrd3 | lr 3.3e-4, wd 0.05 | 0.841, 0/8 | 0.950, 6/8 | -0.109 [-0.215, +0.010] | 0.089 | 0.0070 | **SIGNED LEARNS BETTER** |
| torus d256L2 | 2 layers, d 256 | 0.967, 2/8 | 0.999, 8/8 | -0.032 [-0.055, -0.015] | 0.0005 | 0.0070 | **SIGNED LEARNS BETTER** |

Gates passed in every cell (SO >= 4/8). lrx3 is UNMEASURED, not NO SIGN COST: SOLVED differs by 4 (Fisher p 0.077) and the
accuracy gap (perm p 0.026) is smaller than the registered 0.05 materiality.

## What this shows
1. **The default replicates on fresh seeds** (fourth observation-step-free ring batch): 1/8 vs 8/8.
2. **Weight decay is not the cause.** With no weight decay the barrier class solves 0/8 vs 8/8 (the largest accuracy gap,
   -0.127). Together with ACCOUNTS T1 (no weight-decay effect on the held code), the cost is not an AdamW decay artefact.
3. **A slower recipe keeps the cost; a faster one shrinks it.** At lr /3 the signed model still solves 6/8 and |.| 0/8. At
   lr x3, |.| solves 4/8 and reaches 0.987; the signed model 8/8 at 1.000. Note the lr cells also change the effective
   decay (lr x wd) and the cosine floor (Amendment 1); only wd0 isolates decay. So the cost's SIZE depends on the learning
   rate; whether it vanishes at still larger rates is not measured.
4. **The cost survives a larger model on the torus**: at 2 layers and d 256 the barrier class solves 2/8 vs 8/8 (Fisher
   p 0.007; accuracy -0.032, p 0.0005). The accuracy gap is much smaller than at 1 layer d 128 (GENERALITY E3: 0.880 vs
   1.000, -0.120; an out-of-batch comparison, Amendment 1 MINOR 3), and 6/8 |.| runs were still DESCENDING at 900 epochs
   (final losses 0.018-0.144): at this scale part of the cost may be speed. Budget-scoped.
5. **The |.| runs that solve do not use the wrapped code** (declared secondary): wrap-cancel share 0.04-0.13 in every |.|
   cell, including lrx3 (4/8 SOLVED) and the torus, vs 0.43-1.00 for signed. How they solve is not identified.

## Declared secondaries (no verdicts)
| cell | arm | acc 4T (ring 512) / 2T (torus 2048) | loss AUC | SOLVED at loss 0.1 | wrap-cancel | opposite-sign |
|---|---|---|---|---|---|---|
| ring def | SO / AO | 0.996 / 0.773 | 0.117 / 0.607 | 8/8 / 1/8 | 0.85 / 0.07 | 0.99 / 0.04 |
| ring wd0 | SO / AO | 0.978 / 0.758 | 0.150 / 0.796 | 8/8 / 1/8 | 0.43 / 0.11 | 0.86 / 0.05 |
| ring lrx3 | SO / AO | 1.000 / 0.934 | 0.077 / 0.453 | 8/8 / 4/8 | 1.00 / 0.13 | 1.00 / 0.07 |
| ring lrd3 | SO / AO | 0.911 / 0.747 | 0.294 / 0.825 | 7/8 / 0/8 | 0.69 / 0.04 | 0.79 / 0.03 |
| torus | SO / AO | 0.983 / 0.888 | 0.070 / 0.319 | 8/8 / 5/8 | 1.00 / 0.11 | 1.00 / 0.06 |

Threshold sensitivity (SOLVED at loss 0.1, Fisher): def 1/8 vs 8/8 (p 0.0014), wd0 1/8 vs 8/8 (0.0014), lrx3 4/8 vs 8/8
(0.077), lrd3 0/8 vs 7/8 (0.0014), torus 5/8 vs 8/8 (0.20): on the torus the SOLVED contrast does NOT survive a looser
threshold; the accuracy contrast is the evidence there.
Torus strata (T = 1024): SO 1.000 / 0.999 / 0.992 (plain lag < 128 / >= 128 / wrap-only); AO 0.989 / 0.911 / 0.799. The
|.| deficit is in long-gap and wrap-only revisits, where a map is needed.
Floors: ring blank ~0.50, n-gram 0.554, retrace 0.860 (`docs/audits/2026-10-16/ring_floors_out.txt`, post hoc); torus on
this batch's own walks: blank 0.509, n-gram 0.578, retrace 0.844 (`docs/audits/2026-10-16/robust_floors_out.txt`, post hoc); every
run of both arms is above retrace (lowest AO 0.910).
Across recipes at matched seeds (descriptive): AO SOLVED def 1/8, wd0 0/8, lrx3 4/8, lrd3 0/8; SO 8/8, 8/8, 8/8, 6/8.

Per seed (accuracy at the training length, class):
```
ring  seed | def:SO | def:AO | wd0:SO | wd0:AO | lrx3:SO | lrx3:AO | lrd3:SO | lrd3:AO
1400 | 1.0000 SOLV | 0.8230 STAL | 1.0000 SOLV | 0.9078 DESC | 1.0000 SOLV | 1.0000 DESC | 1.0000 SOLV | 0.8893 STAL
1401 | 1.0000 SOLV | 0.9155 STAL | 1.0000 SOLV | 0.6239 STAL | 1.0000 SOLV | 0.9596 STAL | 0.6194 STAL | 0.6221 STAL
1402 | 1.0000 SOLV | 0.9237 STAL | 1.0000 SOLV | 0.8878 STAL | 1.0000 SOLV | 0.9702 STAL | 1.0000 SOLV | 0.8319 STAL
1403 | 1.0000 SOLV | 0.9136 STAL | 1.0000 SOLV | 0.8800 STAL | 1.0000 SOLV | 0.9995 SOLV | 1.0000 SOLV | 0.9167 STAL
1404 | 1.0000 SOLV | 0.9118 STAL | 1.0000 SOLV | 0.9119 STAL | 1.0000 SOLV | 1.0000 SOLV | 1.0000 SOLV | 0.8313 STAL
1405 | 1.0000 SOLV | 0.8791 STAL | 1.0000 SOLV | 0.8858 STAL | 1.0000 SOLV | 0.9650 STAL | 1.0000 SOLV | 0.8526 STAL
1406 | 1.0000 SOLV | 0.9211 DESC | 1.0000 SOLV | 0.9110 STAL | 1.0000 SOLV | 1.0000 SOLV | 0.9794 STAL | 0.9062 STAL
1407 | 1.0000 SOLV | 0.9997 SOLV | 1.0000 SOLV | 0.9735 DESC | 1.0000 SOLV | 0.9998 SOLV | 1.0000 SOLV | 0.8801 STAL
torus 1400 SO 1.0000 SOLV | AO 0.9941 DESC     1404 SO 1.0000 SOLV | AO 0.9935 SOLV
      1401 SO 1.0000 SOLV | AO 0.9981 SOLV     1405 SO 1.0000 SOLV | AO 0.9706 DESC
      1402 SO 0.9941 SOLV | AO 0.9705 DESC     1406 SO 1.0000 SOLV | AO 0.9098 DESC
      1403 SO 1.0000 SOLV | AO 0.9454 DESC     1407 SO 1.0000 SOLV | AO 0.9544 DESC
```

## Consequences for the paper
- "One recipe" is no longer the scope: the cost holds without weight decay, at a third of the learning rate and in a
  2-layer d 256 torus model; it shrinks at 3x the learning rate (4/8 vs 8/8, -0.013, unmeasured) and in accuracy at the
  larger torus model (-0.032, 6/8 |.| still descending). State both as limits, not as robustness.
- Not tested: other optimisers, lr x wd interaction, larger lr, more depth or width, longer torus budgets at scale.
