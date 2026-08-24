# Continuous displacement: does allocentric recoding survive Habitat's conditions?

H=12 headings (Habitat turns 30 degrees), position real-valued, n=3, 980 batches.
`allocnoise` adds 0.15 rad of Gaussian noise to each executed turn, so the
recorded direction drifts off the true displacement -- Habitat actuation noise.

| condition | floor | Vanilla | RoPE (index) | position effect |
|---|---|---|---|---|
| commanded | 0.509 | 0.618 ± 0.008 | 0.509 ± 0.005 | **+0.110** |
| allocentric | 0.509 | 0.772 ± 0.099 | 0.509 ± 0.005 | **+0.263** |
| allocnoise | 0.509 | 0.739 ± 0.045 | 0.509 ± 0.008 | **+0.230** |

Reference, H=4 discrete (n=8): commanded **+0.050**, allocentric **+0.488**,
translate baseline **+0.438**.

> **SUPERSEDED 2026-08-20, and the correction itself PARTLY CORRECTED
> 2026-08-23.** The budget sweep below shows the 980-batch number was too low,
> so the "partial recovery" reading of this section is withdrawn. But the sweep's
> own conclusion — that the effect climbs with budget — did NOT survive its third
> point: nb=4000 sends it back down to +0.286. Read the correction at the end,
> then `H12_BUDGET_CURVE.md`, before this section.

## Result at a fixed 980-batch budget (superseded): partial recovery

| condition | position effect | vs H=4 reference |
|---|---|---|
| H=12 commanded (turn/forward) | +0.110 | H=4: +0.050 |
| **H=12 allocentric** | **+0.263** | H=4: **+0.488** |
| H=12 allocentric + 0.15 rad noise | +0.230 | — |
| *translate baseline (H=4)* | — | *+0.438* |

**Allocentric recoding still helps a lot** — it more than doubles the position
effect at 12 headings (+0.110 → +0.263), reproducing the direction of the H=4
result. **But it no longer restores the baseline.** At H=4 the recovery was
total (+0.050 → +0.488, exceeding the +0.438 translate baseline); at H=12 it
recovers roughly 40% of that.

**Actuation noise costs almost nothing.** Adding 0.15 rad of Gaussian error to
every executed turn — so the recorded direction drifts off the true displacement
and the error compounds — moves the effect from +0.263 to +0.230. A −0.033 cost
for noise that makes every token systematically wrong. So the barrier is **not**
the token being an approximation of a noisy process.

What changed between H=4 and H=12 is instead that **position became
real-valued**. With 12 headings the agent lands on fractional coordinates and
the observed cell is a `floor()` of a continuous position, so two visits to
"the same place" can straddle a cell boundary and return different observations.
That is a property of the task, not of the encoding.

## The alternative explanation, which is NOT ruled out

The H=12 conditions have a scored rate of **0.022** against the torus baseline's
0.225 — ten times less supervision. These runs used 980 batches (10x) to
compensate, but `Vanilla` reaches only 0.772 ± 0.099 against a 0.509 floor,
which is well above floor but nowhere near the 0.996 it reaches at H=4. **It may
simply be undertrained**, and the partial recovery an artifact of budget rather
than of continuous position.

Distinguishing these needs a budget sweep at H=12 — the same rule-5 check that
turned `rotate` from +0.004 (both arms on the floor) into +0.050. Until that is
run, "the fix degrades with finer heading quantisation" and "the fix is fine but
the H=12 task needs more compute" are both live.

## What this means for a Habitat port

Habitat turns 30 degrees (12 headings) and moves 0.25 m, so H=12 is its regime
exactly. On this evidence:

- allocentric recoding **helps substantially** there and is worth doing;
- it is **not** a complete fix, unlike the discrete four-direction case;
- **actuation noise is not the obstacle**, which is the encouraging part — the
  realistic-setting noise models should not break it;
- and the remaining gap is either continuous position or budget, unresolved.

That is a weaker headline prediction than the H=4 result suggested, and it should
be stated that way rather than extrapolated from the discrete case.


---

# CORRECTION: it was the budget, not the quantisation

This file flagged two live explanations for the partial recovery at H=12 and said
a budget sweep would separate them. It does, decisively, in favour of
undertraining.

| budget | Vanilla | RoPE | position effect |
|---|---|---|---|
| 980 (this file's runs) | 0.772 ± **0.101** | 0.508 ± 0.006 | +0.264 |
| **2000** | **0.891 ± 0.005** | 0.508 ± 0.006 | **+0.383** |
| *H=4 reference (n=8)* | — | — | *+0.488* |
| *translate baseline (n=8)* | — | — | *+0.438* |

Doubling the budget moves the effect **+0.264 → +0.383** and collapses the seed
spread from **±0.101 to ±0.005**. That variance collapse is the diagnostic: at
980 batches the three seeds sat at different points on the learning curve, which
is what undertraining looks like; at 2000 they agree to half a percent.

**So the claim "allocentric recoding generalises only partially to Habitat's 12
headings" is withdrawn.** It generalises: at every budget tested the weakest
path-integrated seed (0.661) beats the strongest index seed (0.555) against a
0.508 floor.

This is the third false negative from a fixed budget in one day — `rotate`
(+0.004 → +0.050 once both arms cleared the floor) and Level 1.5 on the paper
task (0.938 at 16 epochs → 1.000 at 50) were the others. A weak number at one
budget is not a result.

### The nb=4000 point (2026-08-23): the trend does not hold

This section originally closed with "within 0.055 of the translate baseline and
still climbing", and a `nb=4000` run described as confirmation. It ran, and it
disconfirms:

| batches | Vanilla | RoPE | effect |
|---|---|---|---|
| 980 | 0.772 ± 0.101 | 0.508 ± 0.006 | +0.264 |
| 2000 | 0.891 ± 0.005 | 0.508 ± 0.006 | **+0.383** |
| 4000 | 0.837 ± 0.060 | 0.551 ± 0.007 | +0.286 |

Two of three seeds converged WORSE at 4000 (final loss 0.834 / 0.815) than every
seed at 2000 (0.507–0.552), while the third converged better than any run in the
sweep (0.422) — on 2× fresh data and the same LR-schedule shape. Accuracy tracks
final training loss at r = −0.996 across all 18 runs, so this is bimodal basin
selection, not a dose-response curve. Separately the index arm leaves the floor
at 4000 (0.542–0.555), shrinking the measured effect on its own.

"Still climbing" is withdrawn. nb=2000 is the best point measured. Per-seed table
and open questions in `H12_BUDGET_CURVE.md`.

## What still stands

- **Actuation noise is not the barrier**: 0.15 rad of Gaussian error on every
  executed turn costs only −0.033 at the 980 budget.
- **`HABITAT_BUILD.md` remains a genuine complication and is untouched by this.**
  Habitat's navmesh slides the agent on 69–91% of forward moves, so real
  displacement is continuous in MAGNITUDE as well as direction. The experiments
  here quantise direction only with a fixed magnitude, so they still do not model
  that case.
