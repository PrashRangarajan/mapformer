# Does the position effect track VISITS PER CELL?

Three predictors were perfectly confounded across grid 8/16/32 at T=512 --
distinct cells visited, prior visits at a scored position, and map extent all
move together. These two conditions vary T at fixed grid to break that, each
matched on DISTINCT CELLS VISITED with a condition already measured.

Prior-visit counts are MEASURED (probe_visits_per_cell.py), not inferred from
T/n_occupied -- the walk is directed, so realised counts are 8.64/4.61/3.05
where the arithmetic predicted 16/4/1.

400 epochs, warmup+cosine, fast-attn (licensed: +0.392 vs +0.374 reference),
n=3, trained AND evaluated at the training length.

| cond | config | statistics | matched against | reference | effect | per-seed | flat |
|---|---|---|---|---|---|---|---|
| A | grid 32, T=128 | 48 distinct, 1.95 prior, 512 occupied | grid 8, T=512 | -0.010 | **+0.275** | +0.192, +0.345, +0.288 | 3/3 |
| B | grid 16, T=1024 | 153 distinct, 6.20 prior, 128 occupied | grid 32, T=512 | +0.374 | **+0.010** | +0.006, +0.005, +0.018 | 1/3 |

Measured noise floor: **0.150**. 'large' below means above it.

## Verdict

**Visits-per-cell SURVIVES, and so does map extent -- this pair cannot separate them.** A (prior 1.95, map 512) is large and B (prior 6.20, map 128) is not, and those two predictors point the same way in both cells. Distinct cells visited IS ruled out: B has 153, essentially the 158 that produced +0.374, and shows nothing. A third condition varying prior visits at fixed map extent is needed.

(A +0.275 vs its reference -0.010; B +0.010 vs its reference +0.374.)

## Scope

Varying T changes sequence length as well as visit statistics, so 'T itself' is
an uncontrolled alternative in every cell here. The design controls it only by
matching each new condition to a reference on distinct cells; it does not
eliminate it. n=3 per condition.

---

## Post-hoc pooling (NOT pre-registered) — and it does separate them

The pre-registered verdict above is pairwise: each new condition against one
matched reference. That logic concluded visits-per-cell and map extent could not
be separated. Pooling all five conditions **does** separate them, because two map
sizes now have TWO episode lengths each — so prior visits vary *at fixed map
extent*.

| grid | occupied | T | prior visits | effect | index-arm acc |
|---|---|---|---|---|---|
| 8 | 32 | 512 | 8.64 | −0.010 | — |
| 16 | 128 | 512 | 4.61 | +0.015 | — |
| 16 | 128 | 1024 | 6.20 | +0.010 | 0.987 |
| 32 | 512 | 128 | 1.95 | **+0.275** | 0.674 |
| 32 | 512 | 512 | 3.05 | **+0.305** | 0.676 |

**Within a map size**, varying T moves prior visits 1.34× (grid 16) and 1.56×
(grid 32) and moves the effect by **0.005 and 0.030** — both far below the 0.150
noise floor.

**Across map sizes**, the effect moves **0.285** (≤128 occupied: +0.005;
512 occupied: +0.290).

The decisive contrast: grid 16 at prior 4.61 gives +0.015, while grid 32 at prior
3.05 gives +0.305. A 1.5× change in prior visits *across* a map boundary flips
everything; a 1.6× change *within* a map does nothing. If visits-per-cell were the
driver, those two should resemble each other far more than either resembles its
own map-size group. They don't.

**MAP EXTENT drives the position effect. Visits-per-cell does not.**

### What condition B actually showed

B is a **ceiling condition, not an informative null**. The index arm *solves* it —
0.9888 / 0.9893 / 0.9824 against path integration's 0.9949 / 0.9945 / 1.0000. So
153 distinct cells visited and 6.20 prior visits are no obstacle to an index model
when the map holds only 128 occupied cells. That is stronger than "no effect", and
it kills distinct-cells-visited from the opposite direction to condition A.

(Only 1/3 of B's arms reach a flat loss, but with both arms at 0.98–1.00 there is
nowhere for further training to go; the non-convergence cannot be hiding an effect.)

### Honest limit

Prior visits were varied only 1.3–1.6× at fixed map extent, against a 4.4× total
range (1.95–8.64) across the study. This rules out visits-per-cell **over the range
accessible at a fixed map size**, not over its whole range — and the two cannot be
crossed in MiniWorld, because prior-visit ranges per grid size do not overlap
(grid 8 spans 5.67–18.35; grid 32 spans 1.95–4.13). Small maps force frequent
revisits. Confirming this outside MiniWorld would need an environment where map
extent and revisit frequency are independently settable.
