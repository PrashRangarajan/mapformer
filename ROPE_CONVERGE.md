# Does RoPE (index) solve grid 32 given enough optimisation?

400 epochs (4x), warmup+cosine schedule, grid 32, oracle recode, n=3.
Baseline for comparison -- the SAME arm at 100 epochs with linear decay:
final loss 0.84 / 0.78 / 0.72, nb_acc 0.615 (mean).

| seed | final loss | slope over last 10% | FLAT? | nb_acc T=512 | nb_acc T=1024 |
|---|---|---|---|---|---|
| 0 | 0.4990 | -0.00029/ep | YES | 0.725 | 0.336 |
| 1 | 0.4055 | -0.00032/ep | YES | 0.789 | 0.388 |
| 2 | 0.4334 | -0.00037/ep | YES | 0.748 | 0.391 |

**mean final loss 0.4460, mean nb_acc 0.754, 3/3 runs flat**

**AMBIGUOUS.** Converged but mid-range; a paired Vanilla batch is required.
