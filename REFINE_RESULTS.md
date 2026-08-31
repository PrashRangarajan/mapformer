# Does REFINING theta each pass beat re-reading a fixed one?

`Looped` computes theta once from the token embeddings and re-reads it every
pass. `LoopedRefine` carries and corrects it:
`theta = theta_0 + gate * tanh(refine(x))` -- this repo's InEKF idea moved from
the sequence axis to the DEPTH axis. gate starts at 0, so at init the two models
are bit-identical (verified 0.00e+00); the gate gradient at 0 is 1.9e-03, so the
no-op init is escapable. +385 params (0.19%).

Match-Query 128^2, TQ=256, chance 0.0625, both arms retrained in one batch.

| arm | n | mean | sd | min | per-seed |
|---|---|---|---|---|---|
| Looped (theta fixed) | 8 | **0.735** | 0.256 | 0.123 | 0.89 0.92 0.12 0.85 0.78 0.74 0.74 0.84 |
| LoopedRefine (theta refined) | 8 | **0.893** | 0.082 | 0.762 | 0.87 0.87 1.00 0.86 0.83 0.94 0.76 1.00 |

**Refine − fixed: +0.158** (sd 0.302, MDE(n=8) 0.299, 6/8 positive, per-seed -0.016, -0.043, +0.877, +0.009, +0.053, +0.196, +0.027, +0.159)

**NO DETECTABLE DIFFERENCE.** The loop's benefit is ITERATION alone; carrying and correcting a position estimate adds nothing on top. That extends this project's standing finding -- the Kalman win was stabilisation and token-type gating, not inference -- to a second axis.

## The learned gate (diagnostic, whatever the verdict)

(gate not recoverable from the checkpoints)

## Scope

One task, one loop count (4), correction applied to theta and
bounded by tanh. A correction applied to delta (odometry) and re-integrated is a
different model and is untested.

---

## Corrected analysis: the runs are NOT reproducible, so pairing was doing nothing

`Looped` was retrained here with the SAME variant, seeds and settings as the
earlier `loop_headroom` batch. It did not reproduce:

| seed | batch 1 | batch 2 | drift |
|---|---|---|---|
| 0 | 0.766 | 0.888 | +0.122 |
| 1 | 1.000 | 0.917 | −0.083 |
| 2 | 0.772 | **0.123** | **−0.649** |
| 3 | 0.838 | 0.851 | +0.013 |
| 4 | 1.000 | 0.781 | −0.219 |
| 5 | 0.868 | 0.744 | −0.124 |
| 6 | 0.942 | 0.735 | −0.206 |
| 7 | 0.777 | 0.841 | +0.064 |

Mean per-seed drift **0.185**. On this task the seed does NOT determine the run:
outcomes are drawn from a bimodal distribution and numerical noise (SDPA kernels,
atomics) decides the basin. Two consequences:

1. **RETRACTED: "the loop arm never fails — 8/8 ≥ 0.77, sd 0.099."** That was one
   lucky batch. Pooled over 16 draws the loop arm is **0.803 ± 0.200 with 1/16
   catastrophic failures**.
2. **Seed-pairing added nothing**, so every Match-Query comparison should be read
   unpaired. Redone that way, with the loop arm pooled over both batches:

| arm | n | mean | sd | failures <0.30 |
|---|---|---|---|---|
| path-int, 1 layer | 8 | 0.456 | 0.220 | 1/8 |
| path-int + loop (pooled) | 16 | 0.803 | 0.200 | 1/16 |
| path-int + loop + REFINE | 8 | 0.893 | 0.082 | 0/8 |
| path-int, 3 real layers | 8 | 0.771 | 0.263 | 1/8 |

| comparison | delta | se | t | verdict |
|---|---|---|---|---|
| loop vs no loop | **+0.346** | 0.092 | **3.75** | **DETECTABLE** |
| refine vs fixed theta | +0.090 | 0.058 | 1.56 | not significant |
| loop vs 3 real layers | +0.032 | 0.105 | 0.30 | not significant |

**The loop result SURVIVES and strengthens** (t=3.75 unpaired, against the paired
+0.414 that only just cleared its MDE). **Refinement adds nothing detectable.**
**The loop matches three real layers** at a third of the parameters — confirming the
earlier retraction of "beats depth".

## The learned gate: the model declined to refine

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| gate | +0.058 | +0.135 | +0.143 | −0.029 | −0.046 | −0.126 | +0.059 | −0.069 |

mean +0.016, mean|gate| 0.083, max|gate| 0.143 — **and the sign is inconsistent
across seeds**. The correction is tanh-bounded, so |gate| caps the correction in
radians: at most **0.14 rad**, against a theta that spans roughly 2*pi*T. That is a
rounding error on the scale of the position code.

So the null is not a failure to express the correction — the gate was escapable
(gradient 1.9e-03 at zero, verified) and the model moved it off zero, then never
made it large enough to matter. **The optimiser declined to refine.** Same tell as
the learnable-beta red herring, where learned betas barely moved from init and a
1.2x sharpening could not explain a +12pp effect.

**Reading:** the loop's benefit is ITERATION, not a better position estimate. This
extends the project's standing finding — the Kalman win was stabilisation and
token-type gating, not inference — from the sequence axis to the depth axis, by an
independent route.
