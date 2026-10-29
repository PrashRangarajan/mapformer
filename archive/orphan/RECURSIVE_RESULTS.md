# Recursive hierarchy (learned readout + residual) vs flat and vs routing

Coarse level consumes the fine level's PROCESSED output via a LEARNED
attention-weighted readout (not a mean of raw K/V), and its result is added
back by residual. Fine level is local-window, so distant tokens are reachable
only through the coarse path -- this asks whether a learned abstractive
summary can carry retrievable content where an average provably cannot.

Retrieval task, clean, trained n_steps=256. Params: Recursive 485K,
Level15 2-layer 452K (depth control), Level15 1-layer 254K.

| Variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| Recursive | 0.972±0.000 | 0.939±0.000 | 0.893±0.000 | 0.833±0.000 |
| Level15_L2 | 0.997±0.004 | 0.991±0.012 | 0.974±0.036 | 0.945±0.071 |
| Level15 (1-layer) | 1.000±0.000 | 0.999±0.002 | 0.994±0.004 | 0.953±0.009 |
| RouteAttn | 0.915±0.091 | 0.890±0.088 | 0.846±0.085 | 0.778±0.078 |
| HierAttn (pooled) | 0.968 | 0.933 | 0.885 | 0.822 |

## Did the learned readout actually become selective?

Uniform weights over a 64-token chunk = entropy 4.159 nats (i.e. a mean).
Lower entropy = the readout learned to pick specific tokens.

- seed 0: readout entropy = 3.823 nats (SELECTIVE)
- seed 1: readout entropy = 3.866 nats (SELECTIVE)
- seed 2: readout entropy = 3.925 nats (still ~uniform = behaving as a mean)


## POST-HOC: the coarse level is INERT (read the table with this)

Diagnostic (the check built for exactly this): zeroing coarse_proj changes
SpaceTimeHier accuracy by 0.0000 at T=1024, and ||coarse contribution|| /
||fine stream|| = 0.030. Training drove coarse_proj -> 0. SpaceTimeHier and
Recursive produce byte-identical accuracy because at inference both ARE just
their fine (local-window) level; the coarse path contributes nothing.

Same inert-module outcome as the Kalman cascade (K_slow -> 0). Readout entropy
stayed 3.82-3.93 vs a 4.16 uniform ceiling -- barely selective, no gradient
pressure to be.

DESIGN FLAW owned: the fine level was made LOCAL, so the coarse path was the
ONLY route to distant tokens; yet the model still scored 0.833 at T=2048, i.e.
the window alone sufficed and the coarse level was never NEEDED. A test where
the coarse level has no job. The informative version keeps the fine level
GLOBAL and asks whether the coarse map path ADDS anything -- everything so far
says such an additive term goes to zero too, which is itself the answer.
