# Paper-task held-out revisit ACCURACY

Paper config: 1 layer, 2 heads, d=128, T=128, 200K sequences (16 epochs x 98 batches x 128).
Paper Table 2 (2D columns), IID: MapWM **0.99**, MapEM-os **1.0**. (An earlier version of this file cited 0.955 / 0.999; those numbers appear in no table of the paper and were retracted in CLAUDE.md on 2026-08-09.)

`same-map` = new trajectories on the trained obs_map; `fresh-map` = unseen obs_map (in-context map learning).

| variant | same-map acc | fresh-map acc |
|---|---|---|
| Vanilla | 0.989 ± 0.010 | 0.989 ± 0.011 |
| VanillaEM_P0 | 0.987 ± 0.012 | 0.987 ± 0.012 |
| MapPoPE-Flat | 1.000 ± 0.001 | 1.000 ± 0.001 |
| RoPE | 0.513 ± 0.007 | 0.514 ± 0.004 |
| PlainFlat | 0.514 ± 0.018 | 0.513 ± 0.014 |
| PoPE-Flat | 0.505 ± 0.009 | 0.510 ± 0.002 |

## The 2x2: the axis is path integration, not the encoding scheme

Every cell trained with the same recipe, hyperparameters and seeds as the rows
above; parameters matched within 0.4% (204k). Fresh-map revisit accuracy, n=3,
against a **measured** always-predict-blank floor of **0.506**.

| encoding | index position | path-integrated |
|---|---|---|
| **RoPE** | 0.514 +/- 0.004 | 0.989 +/- 0.011 (`Vanilla`) |
| **PoPE** | 0.509 +/- 0.004 | **1.000 +/- 0.001** (`MapPoPE-Flat`) |

Both index cells sit at the floor. Both path-integrated cells solve the task.
The encoding scheme moves the result by ~0.005 in the index row and ~0.011 in the
path-integrated row; path integration moves it by ~0.48. **The axis is where the
angle comes from, not how it is applied to Q and K.**

This reproduces on the paper's own task what `MATCH_QUERY_RESULTS.md` found on a
gated task (PoPE-Flat 0.117 vs MapPoPE-Hier 0.847, chance 0.0625), on the
corrected PoPE implementation -- the d-band fix (3fb40a4) landed 2026-08-09
00:44 and those checkpoints were trained 22:00 the same day.

### PoPE is not a weak encoding; it is an inert one without path integration

`MapPoPE-Flat` reaches **1.000 +/- 0.001** and a final training loss of
0.012-0.044, the lowest of anything run on this task (Vanilla 0.070-0.138,
VanillaEM_P0 0.101-0.151). It matches the paper's MapEM-os IID figure of 1.0
using the WM backbone, which the paper's own MapWM does not (0.99).

State the limits of that: this is the IID same-map/fresh-map setting at T=128,
n=3 seeds, one config. It is NOT the paper's OOD-d / OOD-s protocol
(`PAPER_OOD_PROTOCOL.md`), where the ordering could differ, and PoPE has never
been run there. A 1.000 on an in-distribution metric with a 0.506 floor is a
ceiling effect as much as an achievement, and the honest reading is
"indistinguishable from perfect on the IID metric", not "the best model".
