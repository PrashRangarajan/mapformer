# MiniWorld — HIERARCHY ABLATION (isolates pooling)

MapWM-Hier vs MapWM-FlatHG: the SAME Hourglass 3-block scaffold at IDENTICAL parameter count (2,384,026 both), differing ONLY in whether the middle block sees pooled (k=2) or full-resolution tokens. Both retrained in ONE batch, so this is within-batch by construction. Fresh-map, oracle recode, n=3, chance 0.0625.

## T=512

| grid | MapWM-Hier (pooled) | MapWM-FlatHG (no pooling) | effect of POOLING |
|---|---|---|---|
| 8 | 0.683 | 0.347 | **+0.335** (n=3; +0.262, +0.575, +0.168) |
| 16 | 0.943 | 0.879 | **+0.064** (n=3; +0.304, -0.112, -0.001) |
| 24 | 0.978 | 0.827 | **+0.151** (n=3; -0.006, +0.459, +0.000) |
| 32 | 0.986 | 0.755 | **+0.231** (n=3; +0.480, +0.009, +0.202) |

## T=1024

| grid | MapWM-Hier (pooled) | MapWM-FlatHG (no pooling) | effect of POOLING |
|---|---|---|---|
| 8 | 0.607 | 0.234 | **+0.373** (n=3; +0.315, +0.599, +0.204) |
| 16 | 0.883 | 0.761 | **+0.122** (n=3; +0.488, -0.117, -0.004) |
| 24 | 0.876 | 0.719 | **+0.157** (n=3; -0.040, +0.511, -0.000) |
| 32 | 0.927 | 0.583 | **+0.344** (n=3; +0.658, +0.001, +0.372) |

> Read this table INSTEAD of the earlier MapWM-Hier vs Vanilla comparison,
> which differed in scaffold, depth and parameter count (2.38M vs 3.17M) and
> ran across two batches. Only the numbers here isolate pooling.

## VERDICT: pooling has NO reliable accuracy effect — it improves TRAINABILITY

Per-seed values disagree in sign at every grid >=16, and accuracy tracks final
training loss (the r=-0.996 pattern from H12_BUDGET_CURVE reproduces here).
Conditioning on BOTH arms converging. **CORRECTED 2026-08-27 (audit):** the four
rows below are threshold **0.2**, not 0.4 as originally written. At the stated 0.4
there are FIVE pairs, mean **-0.022** (the extra pair is g16 s1, delta -0.112, the
largest-magnitude converged pair). The failure counts below DO use 0.4. Both
thresholds give an effect ~0 against the claimed +0.283, so the conclusion stands;
the specific numbers were computed at an unstated threshold. Rows at 0.2:

| grid | seed | Hier loss->acc | FlatHG loss->acc | delta |
|---|---|---|---|---|
| 16 | 2 | 0.030 -> 0.999 | 0.015 -> 1.000 | -0.001 |
| 24 | 0 | 0.139 -> 0.956 | 0.127 -> 0.962 | -0.006 |
| 24 | 2 | 0.001 -> 1.000 | 0.000 -> 1.000 | +0.000 |
| 32 | 1 | 0.177 -> 0.963 | 0.176 -> 0.954 | +0.009 |

**Mean effect among converged pairs: +0.001 at threshold 0.2, -0.022 at 0.4.** Every apparent "pooling win" is a
run where ONE arm failed to train -- and it goes BOTH ways (g16 s1 is the POOLED
arm failing). The means in the tables above are outlier-driven; do not cite them.

**What IS real: convergence reliability.** Runs with final loss > 0.4:
  MapWM-FlatHG  7 of 12 failed (all 3 at grid 8)
  MapWM-Hier    2-3 of 12 failed
Pooling makes the model EASIER TO OPTIMISE, not more capable. Reportable, but a
different claim from "hierarchy amplifies path integration".

**RETRACTS** the earlier "hierarchy adds +0.283 to path integration" reading of
MINIWORLD_GRID_SWEEP_HIER.md, which was confounded twice over: scaffold+param
mismatch (MapWM-Hier 2.38M vs Vanilla 3.17M, different depth) AND convergence
failures.

**Cross-batch drift was NOT a problem** (the worry that motivated retraining both
arms): retrained MapWM-Hier reproduced its prior-batch numbers EXACTLY
(0.683/0.943/0.978/0.986 at T=512, both batches). Training is deterministic given
seed+code+buffer. The confounds were scaffold and convergence, not batch.

**STILL TRUE, unaffected:** the position-code CROSSOVER (path-int vs index flips
sign between grid 8 and 16) replicates in BOTH the flat pair and the hier pair,
and those are within-pair, parameter-matched comparisons.
