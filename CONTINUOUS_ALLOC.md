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

## Result: the fix generalises PARTIALLY, and noise is not the barrier

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
