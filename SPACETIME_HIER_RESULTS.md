# Space+time hierarchy with the map property preserved at BOTH levels

Coarse level carries its own theta (low-omega blocks = region scale) and
rotates its queries/keys by it, so coarse scores depend on RELATIVE region
displacement. Verified: shifting all coarse theta by a constant changes the
coarse output by 0.000000, i.e. translation equivariance holds above the
fine level too. Chunk content is a learned readout of processed fine output,
never a mean of rotated keys.

Retrieval task, clean, n_steps=256. Controls trained in the same session at
identical config. Params: SpaceTimeHier 485K, Recursive 485K, Level15_L2 452K.

| Variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| SpaceTimeHier | 0.972±0.000 | 0.939±0.000 | 0.893±0.000 | 0.833±0.000 |
| Recursive (no coarse theta) | 0.972 | 0.939 | 0.893 | 0.833 |
| Level15 2-layer | 0.991 | 0.974 | 0.923 | 0.845 |
| Level15 1-layer | 1.000 | 1.000 | 0.998 | 0.955 |
| RouteAttn | 0.892±0.104 | 0.875±0.104 | 0.839±0.103 | 0.776±0.096 |
| HierAttn (pooled) | -- | -- | -- | -- |


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
