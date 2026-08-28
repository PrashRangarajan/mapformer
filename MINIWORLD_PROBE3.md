# MiniWorld — seed-0 probe: NoPE, ConvDelta, GateDelta

**n=1, low confidence by design** — lands the table shape first, seeds after. Final train loss shown beside every accuracy: the hierarchy ablation showed differences here are often convergence, not capability.

## T=512

| grid | Vanilla | RoPE | NoPE | ConvDelta | GateDelta |
|---|---|---|---|---|---|
| 8 | — | — | 0.045 <sub>(1.73)</sub> | 0.374 <sub>(1.22)</sub> | 0.443 <sub>(1.05)</sub> |
| 16 | 0.947 <sub>(0.17)</sub> | 0.603 <sub>(0.76)</sub> | 0.091 <sub>(1.71)</sub> | 0.872 <sub>(0.28)</sub> | 0.591 <sub>(0.67)</sub> |
| 24 | 0.623 <sub>(0.76)</sub> | 0.623 <sub>(0.75)</sub> | 0.113 <sub>(1.68)</sub> | 0.969 <sub>(0.10)</sub> | 0.745 <sub>(0.53)</sub> |
| 32 | 0.559 <sub>(0.85)</sub> | 0.574 <sub>(0.84)</sub> | 0.114 <sub>(1.66)</sub> | 0.808 <sub>(0.43)</sub> | 0.668 <sub>(0.69)</sub> |

## T=1024

| grid | Vanilla | RoPE | NoPE | ConvDelta | GateDelta |
|---|---|---|---|---|---|
| 8 | — | — | 0.016 <sub>(1.73)</sub> | 0.276 <sub>(1.22)</sub> | 0.269 <sub>(1.05)</sub> |
| 16 | 0.786 <sub>(0.17)</sub> | 0.319 <sub>(0.76)</sub> | 0.037 <sub>(1.71)</sub> | 0.676 <sub>(0.28)</sub> | 0.386 <sub>(0.67)</sub> |
| 24 | 0.394 <sub>(0.76)</sub> | 0.306 <sub>(0.75)</sub> | 0.041 <sub>(1.68)</sub> | 0.733 <sub>(0.10)</sub> | 0.522 <sub>(0.53)</sub> |
| 32 | 0.320 <sub>(0.85)</sub> | 0.297 <sub>(0.84)</sub> | 0.046 <sub>(1.66)</sub> | 0.630 <sub>(0.43)</sub> | 0.424 <sub>(0.69)</sub> |

> Values are nb_acc with (final train loss) beneath. Compare NoPE vs RoPE
> (is the ordinal rotation a handicap?), ConvDelta vs Vanilla (does
> learning how-much-to-accumulate help when full accumulation is already
> correct?), GateDelta vs Vanilla (does an explicit action/obs gate help?).


## CORRECTION (2026-08-27)

'NoPE collapses to chance' overstates n=1 data. NoPE is **seed 0 only**, and at
g32 its loss is still descending monotonically at epoch 100 (1.693@80 ->
1.680@90 -> 1.662@100). That is indistinguishable in kind from Vanilla's 0/3
non-convergence at grid 8, which this session treats as an optimisation artifact.
Say: *NoPE did not leave the plateau within 100 epochs, n=1, still descending.*

The floor is also misquoted: the gates report a non-blank MARGINAL of
0.070-0.079, not 1/16 = 0.0625 -- which is why NoPE reads 0.045, *below* the
stated 'chance'.

And the inference does not follow. 'NoPE < RoPE, therefore index-RoPE is not a
straw man' is a non-sequitur: the straw-man objection asks for a STRONG index
baseline (learned absolute embeddings, ALiBi, T5 relative bias, or RoPE with a
tuned base -- ours uses base=10000.0, never tuned). Removing position entirely
says nothing about any of those.
