# Does MapPoPE beat or match PoPE everywhere PoPE was run?

**No.** It wins overwhelmingly on four task families, matches on one, and **loses on
MiniWorld** — and the losses are one coherent, already-diagnosed class.

The contrast is matched on hierarchy, so the ONLY difference is the phase: index
(`PoPE`) against path-integrated (`MapPoPE`). Compiled from existing batches; no new
runs. Void files (hier-goal, planner tasks) and the ungated `CLOCK_SCAN` excluded.

## Where MapPoPE wins

| task | PoPE | MapPoPE | delta |
|---|---|---|---|
| **torus paper task**, held-out map (n=8) | 0.509 | **0.994** | **+0.485** |
| **torus, paper's OOD protocol** (IID / OOD-d / OOD-s / l=512) | 0.508 / 0.226 / 0.799 / 0.804 | **1.000 / 0.995 / 0.999 / 0.996** | up to **+0.77** |
| **MiniGrid allocentric**, flat (T=128/512/1024) | 0.871 / 0.828 / 0.807 | **0.873 / 0.833 / 0.818** | +0.002 to +0.011 |
| **MiniGrid allocentric**, hier | 0.873 / 0.831 / 0.817 | **0.877 / 0.840 / 0.825** | +0.004 to +0.009 |
| **MiniWorld fixed map**, raw / allocentric | 0.655 / 0.807 | **0.662 / 0.819** | +0.007 / +0.012 |
| **bounded memory**, longest two budgets | 0.857 / 0.731 | **0.872 / 0.751** | +0.015 / +0.020 |

On the torus the comparison is not close: **PoPE with an index phase sits ON the
0.506 blank floor.** Everything PoPE contributes there requires the path-integrated
phase to be present at all.

## Where MapPoPE loses

| task | PoPE | MapPoPE | delta |
|---|---|---|---|
| **MiniGrid raw** (no recoding), flat T=1024 | **0.953** | 0.919 | **-0.034** |
| **MiniGrid raw**, hier T=1024 | **0.955** | 0.942 | -0.013 |
| **MiniWorld fresh map**, raw | **0.384** | 0.308 | **-0.076** |
| **MiniWorld fresh map**, allocentric | **0.364** | 0.232 ⚠ | **-0.132** |
| **MiniWorld oracle** (exact cell displacement) | **0.938** | 0.324 ⚠ | **-0.614** |
| **compositional**, longer T | 0.162 / 0.123 | 0.137 / 0.090 | -0.025 / -0.033 |

## The pattern, and where it breaks

Four of the six losses are **rotation-based action spaces**, which is this project's
oldest diagnosed failure: MapFormer cumsums a fixed per-token increment, and under
turn/forward the displacement depends on accumulated heading, which that form cannot
represent. On MiniGrid, **allocentric recoding flips the sign** — raw `-0.034`,
recoded `+0.011`.

**But it does NOT flip on MiniWorld, and that is the honest problem.** Recoding makes
MapPoPE *worse* there (raw 0.308 -> allocentric 0.232), and the oracle column is the
sharpest failure in the whole comparison: given the **exact per-step cell
displacement** — the cleanest possible action signal — the index arm reaches 0.938
and MapPoPE reaches 0.324.

Two things must be said about that. Both MapPoPE MiniWorld cells carry the
**non-convergence flag** (final train loss > 1.5), and this project's own rule is
that accuracy tracks final loss (r has reached -0.996) — so these may be
optimisation failures rather than capability ones, and they have never been re-run
at the better recipe. And MiniWorld's forward step has **continuous magnitude**
(CV = 0.49 from navmesh sliding), so even an oracle direction leaves the step
*length* unmodelled — a quantised angular code cannot integrate it.

## The answer

**On tasks where a usable *where* can be formed, MapPoPE dominates PoPE** — by up to
+0.485, and on the paper's own task PoPE alone cannot leave the floor. **On
rotation-action environments it loses**, recoverably on MiniGrid and, on the evidence
so far, not on MiniWorld.

So the defensible claim is scoped, not universal: *adding path integration to PoPE is
free-to-large where displacement is representable as a fixed per-token increment, and
harmful where it is not.* Claiming "MapPoPE ≥ PoPE everywhere" is contradicted by six
cells, three of them by more than 0.03.

## The one run that would settle the open half

MiniWorld MapPoPE has never been trained at the **cosine / lr 1e-3** recipe, and both
its failing cells are flagged non-converged. Until that is run, "MapPoPE fails on
continuous-geometry navigation" and "MapPoPE was undertrained there" are not
separated — and `COMP_HEADROOM.md` showed exactly that confusion costing a task 0.16.
