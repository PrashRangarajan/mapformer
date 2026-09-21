# The crossing: a decay envelope's metric matters, and path integration makes a model insensitive to it

> **AUDIT REVISION (2026-09-20).** An independent verification reproduced every number here from the
> checkpoints, and confirmed both new classes do what their docstrings say (including a decisive
> isolation test: with the envelope killed, randomising the index arm's metric map changes the logits
> by exactly 0.0). It also found that the two CONCLUSIONS drawn from those numbers overreach. The
> title's second half is withdrawn, the first half is qualified, and the corrections are in the
> section "What the audit changed" at the end. Read that first.


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

## What the audit changed (2026-09-20)

**1. "Path integration confers insensitivity to the metric" is WITHDRAWN -- it is a null, and the
pooling hid a large point estimate.** Broken out by distance bucket (state minus token, 8 seeds):

| bucket | path row | index row |
|---|---|---|
| d 3-8 | -0.005 | +0.165 (8/8) |
| d 9-32 | -0.028 | +0.308 (8/8) |
| **d 33+** | **+0.154** (6/8, MDE 0.194) | +0.224 (8/8) |

d 9-32 is a CEILING cell for the path row (0.997 against 0.969), so pooling over d >= 9 dilutes the
one bucket where that row is off the ceiling. At d 33+ the path row's metric effect is **69% the size
of the index row's**, and is unmeasured only because its MDE is larger than the effect. Under rule 11
that is "unmeasured", not insensitivity, and a positive claim cannot be built on it.

**2. The index-row effect is COLLINEAR WITH A CONVERGENCE GAP and cannot be loss-matched.**
r(final training loss, closer accuracy at d >= 9) = **-0.995** over the 16 index-row runs. Final
training losses do not overlap: token metric [1.0807, 1.0847], state metric [0.9595, 1.0014]. By this
project's own rule, loss-matching requires overlapping losses, so the +0.281 cannot be separated from
"the state-metric arm simply trains better". This file reported no training loss at all.

**3. The index row is not parameter-matched** (50,822 against 51,078; +256 for the metric map), and
no control separated "the envelope reads a state metric" from "the model gained a trainable
content-dependent accumulator". **A frozen-metric control is now running**: identical parameters,
still a state metric, but the increment map is frozen at initialisation. If the effect survives, it
is about having a non-token metric; if it vanishes, it is about learning one; if it tracks training
loss, finding 2 explains it.

**4. No floor was reported here, and two cells sit below it.** The best n-gram gives acc >= 9 of
0.508 and F1 0.884. The index/token cell (0.502) is AT the chance floor -- fine, since that metric is
chance-anchored -- but **both index-row F1 values (0.876 and 0.816) are below the no-stack F1 floor**,
so the paragraph above that draws a conclusion from their difference is not supportable and is
withdrawn.

**5. The registered lambda readout was never reported, and it supports the claim.** Dyck, per head,
initialised at 0.500: index/token **0.408**, index/state **0.593**, path/token 0.453, path/state
0.494. Every arm kept its envelope and the index/state arm RAISED it -- so no arm declined the prior.

**6. Direct evidence that was available and unreported.** The index/state arm's own metric map
correlates **0.744 +/- 0.185 with stack-depth difference and 0.205 +/- 0.016 with token distance**
(8 seeds). That is the mechanism the claim asserts, measured on the very arm the claim is about, and
is stronger than the `MapPoPE_decay` correlation cited above.

**7. Scope, corrected.** "Clock/map predicts WHEN the metric choice can matter" is a two-point
generalisation with no discriminating cell: one task where the metrics differ, one where they
coincide, and none where a clock accumulator meets non-local dependencies. The Bach null is equally
well predicted by "the metric never matters anywhere", so it removes a falsifier and adds no positive
evidence. The claim that survives is narrower still: **on Dyck-2, an index model's decay envelope
destroys long-range retrieval when it decays over token distance and does not when it decays over a
learned state -- subject to the convergence and parameter confounds in 2 and 3.**

**8. Minor**: the two swap bullets used opposite sign conventions (both are +0.28 and +0.03 as
state minus token); "an order of magnitude smaller" compares an NLL difference with an accuracy
difference across tasks and should be dropped for the within-task statement; the Bach r = 1.000 is
0.998 when measured on the arm that actually uses the metric, and the convention was not stated.
This file is also cross-batch (runs/dyck_cross is 22 h after runs/dyck_decay) with no provenance note;
code for the pre-existing arms is byte-identical across those batches.
