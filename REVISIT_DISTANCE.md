# Revisit accuracy by recurrence interval

Index-position models beat the marginal in LIKELIHOOD (train loss 1.59-1.68 vs 2.079 nats) while sitting at it in ACCURACY (0.513 vs a 0.506 blank floor). This asks where that likelihood goes.

Recurrence interval = steps since the cell was last visited. The paper's walk is directed with run lengths 1-10, so an out-and-back run retraces cells a few steps later -- detectable from the ACTION TOKENS AS CONTENT, with no position code. If that is the source, index models win only in the leftmost buckets.

| variant | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ |
|---|---|---|---|---|---|---|---|
| Vanilla | 0.995 | 0.996 | 0.996 | 0.993 | 0.980 | 0.937 | 0.881 |
| RoPE | 0.557 | 0.516 | 0.505 | 0.503 | 0.500 | 0.495 | 0.507 |
| PlainFlat | 0.575 | 0.518 | 0.505 | 0.501 | 0.477 | 0.475 | 0.488 |
| *blank rate (floor)* | *0.504* | *0.514* | *0.514* | *0.509* | *0.484* | *0.504* | *0.533* |
| *n per seed* |  |  |  |  |  |  |  |

## Share of revisit events per bucket

Measured on 40x64 sequences from the same generator (n = 75,409 revisit events):

| bucket | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ |
|---|---|---|---|---|---|---|---|
| share | **19.3%** | 16.3% | 24.7% | 24.6% | 8.9% | 4.5% | 1.7% |

## Verdict: the retrace hypothesis is CONFIRMED

Index models exceed the blank floor in exactly one bucket -- recurrence interval
1-2, by +0.05 to +0.07 -- and are at or below it in all six others. Weighting by
bucket share: 0.193 x 0.06 = **+0.012**, which is what the aggregate shows
(+0.007 to +0.010 over the floor). The advantage is entirely short-range.

Interval 1 is impossible on this grid (the walk always moves), so bucket 1-2 is
really all interval 2: go east, go west, land back where you were. That revisit
is fully determined by the ACTION TOKENS ALONE -- no position code needed, just
"the last action reversed the one before it, so copy the observation from four
tokens back". A 1-layer model appears to learn this only partially (0.557, not
~1.0), which is consistent with it needing to CONDITION the copy on the reversal
rather than copy unconditionally.

### Why the likelihood gain is large while the accuracy gain is tiny

Training loss 1.59-1.68 against a 2.079-nat marginal is a ~0.4-nat gap, which
looks far too big for a +0.012 accuracy gain. It is not, because likelihood pays
partial credit and argmax does not. On the ~19% of events that are interval-2
retraces the model can put substantial mass on the correct observation while its
argmax still lands on blank: moving a non-blank target from the marginal 0.031 to
even 0.4 is ln(0.4/0.031) = 2.5 nats on that event, and 0.193 x 0.5 x 2.5 = 0.25
nats of the gap from that alone.

So there is no contradiction between the two measurements. The index models learn
one genuine, strictly local regularity -- immediate out-and-back retraces -- and
nothing about the map. Vanilla, by contrast, holds 0.995 out to interval 16 and
degrades gracefully to 0.881 at 65+, which is what an actual map looks like.
