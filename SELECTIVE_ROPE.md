# Selective RoPE's angle generator vs MapFormer's

Both papers put a content-dependent cumulative sum in the rotation phase,
posted three days apart with neither citing the other. This swaps the
generator and changes nothing else, with one arm per difference so any
effect can be attributed.

**Not parameter-matched, and it cannot be** — removing the rank bottleneck
IS the design difference. Read every contrast against the params column.

| arm | what it adds | params | vs MapFormer |
|---|---|---|---|
| RoPE | index position (floor reference) | 199,042 | -448 |
| Vanilla | MapFormer: r=2, no conv, no gate | 199,490 | — |
| ConvAngle | + causal conv only | 199,683 | +193 |
| NoBottleneck | + full rank only | 207,363 | +7,873 |
| GateAngle | + sigmoid gate only | 207,683 | +8,193 |
| SRoPEGen | Selective RoPE: all three | 215,875 | +16,385 |

## Parity, L=16 (training length)

| arm | accuracy | vs MapFormer | verdict |
|---|---|---|---|
| RoPE | 0.676 ± 0.166 | -0.295 (sd 0.167, MDE 0.117, 0/16) | DETECTABLE NEGATIVE |
| Vanilla | 0.971 ± 0.030 | — | baseline |
| ConvAngle | 0.952 ± 0.055 | -0.019 (sd 0.064, MDE 0.045, 8/16) | UNMEASURED |
| NoBottleneck | 0.929 ± 0.072 | -0.042 (sd 0.068, MDE 0.048, 6/16) | UNMEASURED |
| GateAngle | 0.923 ± 0.056 | -0.048 (sd 0.070, MDE 0.049, 2/16) | UNMEASURED |
| SRoPEGen | 0.963 ± 0.027 | -0.008 (sd 0.041, MDE 0.029, 8/16) | UNMEASURED |

## Parity, L=128 (extrapolation)

| arm | accuracy | vs MapFormer | verdict |
|---|---|---|---|
| RoPE | 0.519 ± 0.019 | -0.078 (sd 0.024, MDE 0.017, 0/16) | DETECTABLE NEGATIVE |
| Vanilla | 0.598 ± 0.013 | — | baseline |
| ConvAngle | 0.578 ± 0.017 | -0.020 (sd 0.018, MDE 0.013, 3/16) | DETECTABLE NEGATIVE |
| NoBottleneck | 0.578 ± 0.027 | -0.020 (sd 0.025, MDE 0.018, 4/16) | DETECTABLE NEGATIVE |
| GateAngle | 0.568 ± 0.015 | -0.030 (sd 0.020, MDE 0.014, 3/16) | DETECTABLE NEGATIVE |
| SRoPEGen | 0.588 ± 0.017 | -0.009 (sd 0.025, MDE 0.017, 6/16) | UNMEASURED |

## Parity, L=256 (extrapolation)

| arm | accuracy | vs MapFormer | verdict |
|---|---|---|---|
| RoPE | 0.510 ± 0.009 | -0.038 (sd 0.013, MDE 0.009, 0/16) | DETECTABLE NEGATIVE |
| Vanilla | 0.548 ± 0.007 | — | baseline |
| ConvAngle | 0.538 ± 0.009 | -0.011 (sd 0.009, MDE 0.006, 3/16) | DETECTABLE NEGATIVE |
| NoBottleneck | 0.538 ± 0.014 | -0.010 (sd 0.013, MDE 0.009, 4/16) | DETECTABLE NEGATIVE |
| GateAngle | 0.534 ± 0.008 | -0.014 (sd 0.011, MDE 0.008, 3/16) | DETECTABLE NEGATIVE |
| SRoPEGen | 0.544 ± 0.008 | -0.004 (sd 0.012, MDE 0.009, 6/16) | UNMEASURED |

## Torus paper task

| arm | T=128 | T=512 | T=1024 | vs MapFormer @T=512 |
|---|---|---|---|---|
| RoPE | 0.798 | 0.497 | 0.412 | -0.448 (sd 0.050, MDE 0.050, 0/8) |
| Vanilla | 0.993 | 0.944 | 0.834 | baseline |
| ConvAngle | 0.991 | 0.913 | 0.770 | -0.031 (sd 0.049, MDE 0.048, 2/8) |
| NoBottleneck | 0.994 | 0.967 | 0.892 | +0.022 (sd 0.025, MDE 0.025, 6/8) |
| GateAngle | 1.000 | 0.984 | 0.920 | +0.040 (sd 0.030, MDE 0.030, 7/8) |
| SRoPEGen | 1.000 | 0.975 | 0.882 | +0.031 (sd 0.030, MDE 0.030, 7/8) |

## Verdict

**Selective RoPE's full generator does not beat MapFormer's on either task**, and
costs $+8.2\%$ parameters. Parity $-0.009$ (unmeasured); torus $+0.031$ at T=512
and $+0.048$ at T=1024, the second unmeasured. Its three components pull in
**opposite directions on the two tasks**, which is the actual finding:

| knob | params | parity L=128 | torus T=1024 |
|---|---|---|---|
| causal conv | +193 | -0.020 DETECTABLE | -0.064 (t -1.99, 2/8) |
| no bottleneck | +7,873 | -0.020 DETECTABLE | +0.058 (t 2.13, 7/8) |
| sigmoid gate | +8,193 | -0.030 DETECTABLE | **+0.086 (t 3.05, 8/8, sign p=0.008)** |
| all three | +16,385 | -0.009 unmeasured | +0.048 (t 1.36) |

**1. The two parameter-adding knobs help on the torus and are indistinguishable
from each other.** GateAngle minus NoBottleneck is $+0.018$ (MDE 0.026) at T=512
and $+0.028$ (MDE 0.045) at T=1024 -- both unmeasured. Two different ways of
spending ~8k parameters on the angle generator buy the same $+0.02$ to $+0.09$.
**The gate is not shown to be special; the parameters are doing the work.** Any
reading of "the sigmoid gate helps MapFormer" has to explain why an unrelated
+8k addition helps just as much.

**2. The conv is the one knob that is free, and the only one that hurts on both.**
$+193$ parameters, $-0.020$ on parity and $-0.064$ on the torus. Not a capacity
effect in either direction.

**3. My pre-registered prediction was half right, and I am recording the half that
was wrong.** I predicted the conv would hurt on the torus specifically -- where the
cumsum is a literal path integral and smoothing the increment blurs an exact
displacement -- and be neutral on parity, where the increment is a learned clock
rate with no geometric meaning. It does hurt more on the torus (7.7% of the
baseline against 3.3% on parity), but it hurts on parity too, which the mechanism
story did not predict. The directional part survives; the "neutral on parity" part
does not.

**4. The sign flip between tasks is the result worth carrying.** Every knob that
helps on the torus hurts on parity. Whatever the extra capacity buys a
path-integrated model on navigation, it costs on the iterative task -- so
"better-engineered angle generator" is not a task-independent property, and
Selective RoPE's design choices should not be imported into this codebase on the
strength of their language-modelling results.

## Scope

This swaps the GENERATOR and keeps MapFormer's PLACEMENT: the angle is
computed once from token embeddings before the blocks, not per-head and
per-layer from the query. At 1 layer the two are close, since the query is
itself a learned linear map of the token. **A negative result here does
not refute Selective RoPE** — it is evidence about the generator's
components in this setting, on tasks their paper does not run.

---

## CONFOUND found on audit, 2026-09-05: the arms are NOT single-knob

Verified by diffing named parameters against `MapFormerWM`. Every arm above
(`ConvAngle`, `NoBottleneck`, `GateAngle`, `SRoPEGen`) **removes**
`path_integrator.omega`, `action_to_lie.w_in` and `action_to_lie.w_out`, because
`SelectiveAngle` has no omega -- its readout is `A = tau * I` with a single
scalar `log_temp`.

So each arm changes **two** things: the named knob, and the readout
`A: diag(omega) W_out -> tau I`. The second discards 64 learnable,
geometrically-initialised per-(head, block) frequencies spanning
`[2pi/64, 2pi]`.

**Consequence.** The per-knob rows cannot attribute their effects to the conv,
the rank, or the gate. The "sign flip between tasks" is still a real observation
about the arms as built, but its attribution to individual components is not
supported. The parameter deltas (+193 / +7,873 / +8,193 / +16,385) are unaffected
and verify exactly.

**The missing arm** is one that keeps `diag(omega) W_out` and adds only the conv,
or only the gate. It has not been run. Until it is, the cheapest honest reading of
this file is that a generator swap which also drops the learnable frequency
spectrum costs a little on parity and buys a little on the torus, and no
finer-grained claim survives.

This does not disturb `RANK_SWEEP.md`, whose arms differ only in `bottleneck_r`
and keep the readout intact.
