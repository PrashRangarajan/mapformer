# Warm-start stability test -- results (2026-09-24)

Pre-registration `RANK_PROJ_PREREG.md`. Runs `runs/rank_proj` (frozen), `runs/rank_proj_train`
(trainable). Full output `RANK_PROJ_ANALYSIS.txt` (`python3 -m mapformer.analyze_rank_proj`).

## Registered verdict: S1 STABLE

| arm | SOLVED | STALLED | DESCENDING | T=1024 accuracy |
|---|---|---|---|---|
| **TRAINABLE**: r=2 started from the rank-2 projection of a solved r=4 | **7** | 0 | 1 | 0.994 |
| **CONTROL**: r=4 re-trained from its own solved weights, same recipe / seed / data | 7 | 0 | 1 | 0.996 |
| reference: r=2 continued from its own unsolved 900-epoch weights | 1 | 3 | 4 | 0.905 |

Primary: SOLVED 7/8 vs 7/8, Fisher p 1.00. Accuracy trainable vs control: permutation p 0.67;
trainable vs the r=2 reference: p 0.0009.

**FROZEN** (existence): the projected r=2 models score 0.9955 at T=1024 against their r=4
sources' 0.9966, >= 0.95 on 8/8 seeds (min 0.981).

## What it says

- **r=2 can hold the solution, and recovers it after being knocked out.** The restart to lr
  1e-3 drove every trainable run to loss 0.37-0.46 (the r=4 control peaked lower,
  0.26-0.30), and 7/8 came back below 0.05 at epochs 553-654, the same window as the control
  (497-669). Its action code stays cancelling (opposition 0.004-0.018, against 1.27 for r=2
  trained from scratch).
- **So r=2's from-scratch failure is SEARCH.** The solution exists (frozen), is an attractor
  once reached (trainable), and r=2 trained from scratch does not reach it: 1/8 after 900 +
  900 epochs, where four of its eight runs end the second cycle at or above the loss they
  ended the first at (0.308 -> 0.357, 0.321 -> 0.392, 0.177 -> 0.184, 0.404 -> 0.376).
- Exactly the pattern of EM's recency deficit (`SEARCH_RESULTS.md`): present, stable,
  not found. The extra latent dimensions of r=4 change what training finds, not what the
  model can represent.

Caveats: one task, one width (d=128), one recipe (batch 16, lr 1e-3, warm restart); our
bottleneck is shared across heads where the paper's is per head, so "r=2" here has half the
paper's latent dimensions. The kick is larger at r=2 than at r=4, so "as stable as r=4" is
about the endpoint, not the path.
