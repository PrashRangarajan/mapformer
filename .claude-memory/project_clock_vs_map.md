---
name: project-clock-vs-map
description: Cancellation is not a quality axis — it chooses whether the accumulator is a map or a clock. Explains the sign result, the rank result, and probably the forget gate.
metadata:
  type: project
---

Attention sees only the interval sum `θ_s − θ_t = ω·Σ_{t<u≤s} Δ_u`. What that
quantity **is** depends entirely on whether increments cancel:

| | Δ signed | Δ monotone |
|---|---|---|
| the interval sum is | **net displacement** | **path length** |
| two routes A→B | agree | disagree |
| injective in | space, blind to time | time, blind to place |
| the accumulator is | a **map** | a **clock** |

Mutually exclusive by construction, and **each is correct for one job**. A cognitive
map NEEDS same-place-different-time to collide — that IS revisit prediction. Language
needs the opposite. So CARoPE/GRAPE-AP/CoPE being monotone is not an oversight; on
text a monotone clock is the right object, which is why RoPE is the default there and
why nothing went visibly wrong.

**Do not say a signed increment is simply better.** That framing was in the documents
and is wrong.

## The growth exponent is a readout of which question, not a score

Measured (`LOCALISATION.md`, eval-only). All arms start from the same accumulator
range at training length (88–94) and separate by exponent, `range(S) ~ T^α`:

| arm | α | opposition |
|---|---|---|
| signed r=4 | **0.518** | 0.125 |
| signed r=4 (rank sweep) | 0.524 | 0.092 |
| signed r=2 | 0.619 | 0.495 |
| monotone r=4 | **0.943** | 1.849 |

α≈0.5 = diffusive random walk (measuring position); α≈0.94 = ballistic (measuring
elapsed time). `r(opposition, α) = +0.9995` across two independent batches. **One
mechanism covers the sign result and the rank result.** The ideal for a bounded map
would be α=0 and nothing achieves it — everything accumulates without wrapping.

## Scope: it explains TWO of four, not "almost everything"

`ACCUMULATOR.md`: neither the forget gate (−0.051, MDE 0.092) nor PoPE (+0.006, MDE
0.177) nor Level15 (+0.009, MDE 0.191) changes α. All unmeasured.

**And the positive control FAILED.** "The InEKF's wrap bounds the accumulator" is
FALSE — it wraps the INNOVATION, so `θ̂ = θ_path + bounded correction` and
`range(θ̂) = 285.6` vs `range(θ_path) = 283.9` at T=1024. Asserted twice without
measuring. Withdrawn.

## Candidate account of the forget gate (consistent, untested)

A signed phase cannot represent *how long ago*. The forget gate adds a second,
content-dependent, **monotone** accumulator (`Σ log γ`, measured α = **+0.956**),
restoring the clock. Fits everything: needs a LIVE λ (frozen → baseline, −0.016) but
not DECAY; 5/8 seeds learn λ<0 and gain most (r = −0.516); negative λ is still
monotone, counting up. **It was never a forget gate — it's a clock.**
Predictions, untested: any monotone content-dependent increment of either sign should
reproduce the gain; a content-INDEPENDENT constant should reproduce much of it; the
gain should vanish where there is no recency structure.

See [[project-sign-axis]], [[reference-positional-landscape]].

## alpha is a RE-DESCRIPTION, not a third cause (added 2026-09-08)

Two corrections, both from the user asking what alpha is actually for.

**It is nearly collinear with the opposition score: r = +0.9995.** Opposition is
read straight off the learned action code, is simpler and more interpretable, and
alpha carries almost no information it does not. Nothing rests on alpha that could
not have rested on opposition. Its contribution is **economy** — it makes the sign
result and the rank result one finding at two severities rather than two.

**And "vary alpha deliberately and check that degradation follows" is MALFORMED.**
alpha is not a parameter of any model here; it is a statistic fitted to a trained
one. Both ways to move it fail: interpolating signed -> monotone moves alpha but
destroys the map in the same stroke, and bounding the accumulator is either a NO-OP
(theta enters through cos/sin, so theta and theta+2pi are already identical) or it
breaks additivity and with it interval-relativity. **Within this frame alpha may not
be independently controllable at all** — it is set by the cancellation, whose only
levers are the sign and rank axes.

**The prior question, which I had skipped:** why should a growing accumulator cost
anything when the code is periodic? The standard answer — low-frequency channels at
untrained phases — was imported here and **refuted** (ablating them costs MORE at
OOD). So alpha's association with length-degradation is established and its **route
is not**.

**One narrow use survives**: opposition needs labelled opposite actions and cannot
be computed on text at all; alpha needs only trajectories. It is the form this
measurement takes in the setting the literature actually works in. Not yet measured
there — the language trainer saves no checkpoints.
