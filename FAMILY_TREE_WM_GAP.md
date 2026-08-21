# Family tree with the missing arms: plain WM beats every published variant

`FAMILY_TREE_RESULTS.md` compared four models and had **no plain-WM arm at all** —
a gap flagged in `BASELINE_TABLE.md`. This retrains all four in ONE batch
alongside MapWM-Flat and Level15 (rule 3), so every number below is
within-batch comparable.

n=3 seeds · depth 5, 8 observation types, 2 layers, 100 epochs
**floor 0.163** (hub-node baseline — the 0.125 chance is NOT the floor)

| model | T=64 (train) | T=128 (OOD) | per-seed (T=64) |
|---|---|---|---|
| **Level15** | **0.843 ± 0.015** | **0.789 ± 0.027** | 0.830 / 0.859 / 0.841 |
| **MapWM-Flat** | **0.805 ± 0.072** | 0.746 ± 0.080 | 0.835 / 0.858 / 0.724 |
| MapEM-NC-NL (non-commutative) | 0.729 ± 0.010 | 0.672 ± 0.012 | 0.720 / 0.740 / 0.726 |
| MapEM-os (commutative control) | 0.715 ± 0.008 | 0.659 ± 0.015 | 0.712 / 0.725 / 0.709 |
| Plain-Flat (index) | 0.601 ± 0.011 | 0.550 ± 0.031 | 0.589 / 0.610 / 0.603 |

## The batch reproduces the published numbers exactly

| arm | `FAMILY_TREE_RESULTS.md` | this batch |
|---|---|---|
| MapEM-NC-NL | 0.729 ± 0.010 | **0.729 ± 0.010** |
| MapEM-os | 0.715 ± 0.008 | **0.715 ± 0.008** |
| Plain-Flat | 0.600 ± 0.011 | **0.601 ± 0.011** |

Three arms, three decimal places, independently retrained. The cross-batch worry
that motivated re-running them was unfounded — but it could not have been known
without doing it, and it is what licenses reading the two new rows against the
old ones.

## 1. The missing arm was the best of the published set

MapWM-Flat scores **0.805**, beating MapEM-NC-NL by **+0.076** — five times the
margin the non-commutativity comparison itself turns on.

This does not overturn `FAMILY_TREE_RESULTS.md`'s conclusion, which was that
non-commutativity buys +0.014 over a commutative control for 34x the compute.
That still holds: 0.729 vs 0.715 here, matching. What changes is **what the
conclusion is about**. The comparison ran among EM variants while the simplest
path-integrating backbone — never tested on this task — beat all of them. The
right summary is now "non-commutative machinery costs 34x and lands below plain
MapWM-Flat", which is a considerably stronger negative than the original.

## 2. Level 1.5 helps here, but by REDUCING VARIANCE, not raising the mean

Level15 0.843 vs MapWM-Flat 0.805 looks like +0.038. The per-seed pairing says
otherwise:

| seed | MapWM-Flat | Level15 | Δ |
|---|---|---|---|
| 0 | 0.835 | 0.830 | **−0.005** |
| 1 | 0.858 | 0.859 | **+0.001** |
| 2 | 0.724 | 0.841 | **+0.117** |

Two seeds are exact ties. The entire mean difference comes from seed 2, where
MapWM-Flat collapses to 0.724 and Level15 does not. Paired t on n=3 gives
t≈0.89 — **not significant**, and it should not be reported as a mean improvement.

What is visible is the spread: **±0.015 for Level15 against ±0.072 for
MapWM-Flat**, a 5x reduction. That is consistent with the reframing already in
CLAUDE.md — that the correction's contribution is stabilisation rather than
inference — and it is the first non-landmark task where the correction does
anything measurable at all. On Match-Query it showed no advantage of either kind
(0.876 vs 0.888, and the variance was worse, not better).

**One task, n=3, one bad seed.** Distinguishing "reduces variance" from "got
lucky on the seed that mattered" needs more seeds, and that is the honest state
of this finding.
