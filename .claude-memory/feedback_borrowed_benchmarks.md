---
name: A borrowed benchmark usually does not test your axis — predict its result first
description: Flip-Flop and MQAR were both run as external anchors and both returned nulls that our own data predicted in advance. Check what a benchmark DISCRIMINATES before running it.
metadata:
  type: feedback
---

Twice in one session I proposed a standard synthetic as an external anchor, and
twice it could not have discriminated the thing under test. The second time I caught
it before the batch; the first time I did not.

## Flip-Flop LM — a null our own per-offset curve had already given

6 arms x 8 seeds. Every contrast unmeasured (path-integrated minus index -1.23pp
against an MDE of 3.99); all arms at 0.00% in-distribution error.

**Our own recency data said so in advance**: index accuracy is **0.99 at offset
k=1** and only collapses from k=8. **Flip-Flop only ever asks k=1** — "the bit of
the most recent write". It varies the DISTANCE to the governing write and never the
ORDINAL DEPTH of the query, and ordinal depth is what separates a content-gated
counter from an index.

## MQAR — predicted a ceiling, got a floor, and stopped at 4 pilots

MQAR exists to measure **how close a sub-quadratic model gets to softmax
attention** — its difficulty knob is STATE SIZE. Every arm here is full softmax
attention, so the quantity it varies is one we hold fixed at infinity. And it is
content-addressed: match the key, the value is at +1, i.e. k=1 again.

Nothing learned it at our scale — all arms at "guess among this episode's values"
(0.273 / 0.264 / 0.274 against 0.250). **Running the index arms is what made that
interpretable**: a tempting mechanism story (content-dependent rotation sabotages
content matching) died because RoPE and PlainFlat fail identically. Four pilots,
~15 min, against a 48-run batch that would have measured noise.

## The rule

Before adopting a benchmark, ask **what it discriminates and for whom**. Both of
these were designed to separate architecture *classes* we are not comparing. Write
the predicted result down first — a pre-registered null is a scope statement worth
having; an unpredicted one looks like a failure.

**Of the field's standard synthetics, `parity` does discriminate this axis** (path
integration +0.316 at L=16, 8/8). Adding more borrowed synthetics is not obviously
worth it.
