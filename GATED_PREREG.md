# Gated signed increment: pre-registration

`Delta = sigmoid(W_g x + b) * (W_out W_in x)`. CoPE's selection on MapFormer's
direction. Written BEFORE the batch; no checkpoint has been read.

## Why

MapFormer's what/where separator is a linear bottleneck: `Delta` must reach zero on
observation tokens by cancellation inside `W_out W_in`, and nothing makes zero a
natural output. CoPE's separator is a sigmoid, where zero is the resting state. The
evidence that MapFormer WANTS this is that it re-derives it: on contextual counting
an unconstrained increment learns a content gate in 8/8 seeds, and a
magnitude-matched intervention shows the gate is most of the mechanism
(`RECENCY_GATE_ABLATION.md`). CoPE's own gate cannot be adopted as-is because it is
non-negative, hence a clock and not a map; this keeps the sign.

Deliberately weaker than CoPE in one way: the gate is per TOKEN, not per query-key
pair, which keeps the prefix scan. Cost 258 parameters (+0.13%).

## Arms, and what each isolates

| arm | isolates |
|---|---|
| `Vanilla_r4` | the ungated baseline, retrained in this batch (rule 3) |
| `Gated_r4` | the mechanism |
| `Gated_r4_frozen` | gate present but untrainable -- separates "a gate that ADAPTS" from "an extra module and a constant 0.982 rescale" |
| `Gated_r2` | does an explicit gate SUBSTITUTE for rank? The bottleneck is the identification mechanism, so if the gate takes over the separation, r=2's skewed-basis deficit should shrink |

8 seeds, one batch per task. Torus recipe verbatim from `run_sign.sh`, recency
recipe verbatim from `run_recency.sh`, so both are comparable to their published
batches.

## Predictions

**P1 (torus).** `Gated_r4 - Vanilla_r4 > 0` at OOD length (T=512/1024), where there
is headroom; T=128 is at ceiling for the baseline and cannot show it. The torus is
where the what/where split is literally the action/observation split, so this is the
mechanism's home ground.

**P2 (recency).** `Gated_r4 >= Vanilla_r4` at T=2048, the only recency length with
headroom (the baseline is 1.000 at T=1024). This is the pre-registered "explicit
gate vs learned gate" contrast from the review, now testable: the recency ablation
says the gate is worth +0.594 when supplied by intervention, so supplying it by
construction should be free or better.

**P3 (the diagnostic, and the one that identifies the mechanism).** On the torus the
learned gate should be markedly HIGHER on action tokens than on observation tokens.
This is the pre-registered discriminator: an accuracy gain with NO gate separation
means the module helped for some other reason. Reference point -- Selective RoPE's
gate was measured at 1.35x on this exact contrast and that was judged NOT
suppression, so 1.35x is the floor this must clear to mean anything, not the target.

**P4 (rank substitution).** If the gate takes over the separation, `Gated_r2 -
Vanilla_r4` should exceed `Gated_r4 - Vanilla_r4`, i.e. the gate helps r=2 more than
r=4 because r=2 is the arm whose separator is impaired.

## Falsifiers

- `Gated_r4 ~= Gated_r4_frozen` -> the gain, if any, is the module and the constant
  rescale, not adaptation. This is the control that makes the rest readable.
- Gate ratio at or below 1.35x with an accuracy gain -> P3 fails; the gate is not
  separating what from where and any gain is attributed elsewhere, named before it
  is explained.
- No effect at any length on either task -> the explicit gate buys nothing over the
  learned one, and the review's borrow recommendation is withdrawn.

## Checks required before reading

Convergence (final-10% loss slope, rule 10); r(final loss, accuracy) per length
before leaning on any loss-matched residual (rule 9); MDE beside every contrast,
"unmeasured" where the effect is under it (rule 11).

Pre-flight, already passing: bit-identical to `Vanilla_r4` with the gate forced
open (0.000e+00); gradient 1.9e-01 at init, so escapable; gate starts open AND
token-independent (spread 0.00e+00), so nothing is separated at initialisation;
zero causal leak.
