# MQAR: pre-registered as a probable null, and run anyway

## The prediction, written before the environment existed

**I expect MQAR not to discriminate our axis, and I expect every arm to be at or
near ceiling.** Two reasons, both from the corpus rather than from intuition:

1. MQAR exists to measure how close a **sub-quadratic** model gets to softmax
   attention. Verbatim: *"while subquadratic models can achieve competitive
   language modeling performance to Transformers, [they] lag behind softmax
   attention in recall-intensive tasks."* Its difficulty knob is **state size**.
   Every arm here is full softmax attention with an unbounded KV cache, so the
   quantity MQAR varies is one we hold fixed at infinity.
2. It is **content-addressed**, not position-addressed. You find the key by
   matching content and the value sits at $+1$. That is offset $k=1$ -- exactly
   where `RECENCY_RESULTS` measured an index code at **0.99**, and exactly why
   Flip-Flop returned a null.

This is the second time this pattern has come up. Flip-Flop was run as an external
anchor and returned a null for a reason our own per-offset curve had already given.
Writing the prediction down first is the difference between repeating that and
learning from it.

## Why run it at all

Three reasons, none of which is "we expect to win":

- **Scope.** "MQAR does not test this axis" is a claim worth having measured rather
  than argued. It bounds what the positional axes are for.
- **Credibility.** MQAR appears in 6 of the 40 sources and is the standard synthetic
  for this family. A reader will ask; "we ran it, here is the number" beats "it does
  not apply."
- **It is cheap**, and a genuinely surprising result would be highly informative.

## What would change my mind

- **Any arm below ceiling with a spread across arms exceeding the 0.150 floor.**
  That would mean MQAR is harder for softmax attention than the literature implies,
  and that positional structure matters for it after all.
- **An index/path-integration gap.** Predicted absent; if present, the
  content-addressed reading above is wrong and the recency per-offset account needs
  revisiting.

## Verdict rule, fixed in advance

If every arm lands above 0.95 the result is reported as **"MQAR is at ceiling for
softmax attention and does not discriminate positional mechanisms"** -- a scope
statement, not a failure -- and no arm ordering is read from it, whatever the
ordering happens to be. Reading a ranking off a ceiling is how three claims in this
project were retracted.
