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
