# The crossing: a decay envelope's metric matters, and path integration makes a model insensitive to it

Pre-registration: `CROSS_PREREG.md`. Dyck-2, 8 seeds per cell, closer accuracy at distance >= 9
(chance 0.500), F1 alongside for comparability with the earlier tables.

| | decay over TOKEN distance | decay over LEARNED STATE |
|---|---|---|
| **index phase** | 0.503 (F1 0.878) | **0.784** (F1 0.817) |
| **path-integrated phase** | **0.873** (F1 0.962) | 0.906 (F1 0.956) |

- **Swapping the metric on the INDEX row: +0.281 (MDE 0.047, 8/8 seeds) -- DETECTABLE.** An index
  model whose envelope decays over a learned state keeps long-range retrieval; the same model with the
  same envelope decaying over token distance is at chance. The envelope's damage was the METRIC'S
  fault, not the envelope's.
- **Swapping the metric on the PATH row: -0.033 (MDE 0.084, 3/8) -- unmeasured.** A path-integrated
  model keeps its retrieval whichever metric the envelope uses.

So the outcome follows neither the pure column (C1) nor the pure row (C2), and both readings are
partly right in a way the 2x2 separates cleanly:

1. **The metric is what does the damage.** Where an envelope can destroy retrieval -- the index row --
   the destruction is entirely attributable to measuring proximity in token distance. Give the same
   envelope a learned state to decay over and retrieval survives (0.784 against 0.503).
2. **Path integration confers insensitivity.** Its phase already carries the state, so the model can
   locate the right key even when the envelope's prior is expressed in the wrong units.

**The mechanism in the index/state arm is worth noting on its own**: that model's increment map feeds
ONLY the decay metric -- the attention phase remains `t * theta_c`. It therefore learns a positional
metric with no phase role at all, purely through a distance-proportional bias, and that is enough to
recover most of the long-range retrieval an index model otherwise loses. Nothing in either source
paper has this shape.

**A third instance of the metric disagreement.** The index/state arm has the best closer accuracy of
the two index cells (0.784 vs 0.503) and the WORSE F1 (0.817 vs 0.878), because F1 rewards being
reliably local. Any conclusion drawn from F1 alone here would be the opposite of the one the
stack-sensitive metric supports.

## Status of the claim

This is the one positive theoretical result in the line, and it is now tested by a crossing rather
than inferred from a confounded contrast. Scope: one task, one model size (1 layer, 1 head), one
envelope family (ALiBi-style linear-in-distance), 8 seeds. It has not been tested on Bach, where the
accumulator is a clock and the metric and token distance nearly coincide -- which is the obvious next
test and would say whether "the metric is what matters" survives where the two metrics agree.

## Amendment 1: the same crossing on Bach, where the two metrics coincide

5 seeds, 512-crop setup, test NLL by position bucket (lower is better).

| | decay over TOKEN distance | decay over LEARNED STATE |
|---|---|---|
| **index phase** | 0.6262 | 0.6314 |
| **path-integrated phase** | 0.6332 | 0.6223 |

- **X1 CONFIRMED.** Swapping the envelope's metric moves the index row by **+0.005** (MDE 0.019) and
  the path row by **+0.011** (MDE 0.014) at 2-4x beyond the training context. Both unmeasured, and
  both an order of magnitude smaller than Dyck's +0.281.
- **The registered explanation is now measured rather than inferred.** On Bach the learned-state
  distance and token distance correlate at **r = 1.000 +/- 0.000** across 5 seeds; on Dyck the same
  quantity correlates 0.278 with token distance and 0.755 with stack depth. The metrics do not merely
  behave alike on Bach, they are the same metric to three decimals, so there is nothing for the swap
  to change. That is exactly why the result is uninformative about the claim and was registered as
  the weak outcome in advance.
- **X3 does not fire**: neither swap hurts detectably, so nothing here contradicts the Dyck result.

## Where the claim stands after both tasks

**A decay envelope is a proximity prior in whatever metric it is given.** The claim is supported where
the metrics differ (Dyck: +0.281 on the index row, 8/8, detectable) and untestable where they
coincide (Bach: r = 1.000, swap worth +0.005). The clock/map distinction earns a narrow, measured
role here that it did not earn as a rule about kernels: **it predicts WHEN the choice of metric can
matter at all** -- a clock accumulator is a token counter, so its state metric and token distance are
the same thing, while a map accumulator encodes something else and the choice becomes load-bearing.

That is a smaller claim than the one withdrawn on 2026-09-20, and unlike it, it now has a crossed
design behind it on one task and a measured null-by-construction on the other.
