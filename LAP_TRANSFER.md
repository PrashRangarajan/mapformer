# Does learning the lap task degrade the cognitive map?

One model, shared vocabulary. Phase 1 Match-Query, phase 2 either LAP or
(control) MORE Match-Query for the same number of steps, phase 3 re-measure.
Match-Query chance = 0.0625. Lap circuit has exactly zero net displacement,
so faithful path integration gives theta drift ~0.

| arm | MQ before | MQ after | delta | theta drift before | after | obs/act Delta before | after |
|---|---|---|---|---|---|---|---|
| **lap** | 0.377 | 0.083 | -0.293 | 4.37 | 4.73 | 0.252 | 0.060 |
| **control** | 0.377 | 0.378 | +0.002 | 4.37 | 4.35 | 0.252 | 0.245 |

Lap-arm lap `exact` after phase 2: **0.993** (random-boundary floor 0.250).

## Per seed

| arm | seed | MQ before | MQ after | delta | drift before | drift after |
|---|---|---|---|---|---|---|
| lap | 0 | 0.394 | 0.077 | -0.317 | 4.62 | 4.74 |
| control | 0 | 0.394 | 0.420 | +0.026 | 4.62 | 4.67 |
| lap | 1 | 0.405 | 0.098 | -0.308 | 4.42 | 4.74 |
| control | 1 | 0.405 | 0.390 | -0.015 | 4.42 | 4.43 |
| lap | 2 | 0.330 | 0.075 | -0.255 | 4.06 | 4.72 |
| control | 2 | 0.330 | 0.324 | -0.007 | 4.06 | 3.96 |

## What this does and does not establish

**ESTABLISHED.** Phase 2 on the lap task drops Match-Query from 0.377 to 0.083
(chance 0.0625) on 3/3 seeds, while the same-task control is flat (+0.002). The
model simultaneously reaches lap `exact` 1.000 / 0.980 / 1.000. So lap
competence and cognitive-map competence are not co-held by this architecture at
this scale.

**REFUTED -- the predicted mechanism.** The pre-registered prediction was that
theta drift would RISE and observation tokens would start displacing. Measured:

| | before | after (lap arm) |
|---|---|---|
| theta drift | 4.37 | 4.73 (barely moved) |
| obs/act \|Delta\| | 0.252 | **0.060** (DROPPED, 3/3 seeds) |

Observations displace LESS after lap training, the opposite of the prediction and
of the from-scratch lap probe (0.188). The collapse is NOT the position code
degrading. The mechanism is unaccounted for.

**CONFOUND -- the control is too weak.** Phase 2 of the control is the SAME task,
so it cannot forget. These numbers are equally consistent with ordinary
catastrophic forgetting: 800 steps of ANY different task on a 600K-parameter
model might wipe Match-Query just as thoroughly. Nothing here separates "the lap
task conflicts with cognitive maps" from "sequential training overwrites".

REQUIRED NEXT ARM: phase 2 on a DIFFERENT but MAP-PRESERVING task. If Match-Query
survives that and dies on lap, the conflict is specific to laps. If it dies on
both, this is forgetting and the interesting claim does not survive.

**BASELINE CAVEAT.** Phase 1 reached 0.377, not the sweep's 0.888, because of the
stated constant-LR choice. The map that got damaged was mediocre to begin with.
