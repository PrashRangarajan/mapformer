# Does the sigmoid gate suppress observation tokens?

MapFormer's `action_to_lie` reads every token and must LEARN that
`Delta ~= 0` on observations. A sigmoid gate is the natural shape for
that. If it is what the gate is doing, it explains the sign flip: the
torus is half observation tokens that should contribute nothing, while on
parity every bit must contribute.

Prediction, written before looking: **gate(action) >> gate(observation)**
on the torus, and no such split on parity.

## Torus

| seed | gate on ACTION tokens | gate on OBSERVATION tokens | ratio |
|---|---|---|---|
| 0 | 0.5424 | 0.4341 | 1.25x |
| 1 | 0.5755 | 0.4205 | 1.37x |
| 2 | 0.6084 | 0.3853 | 1.58x |
| 3 | 0.5553 | 0.4230 | 1.31x |
| 4 | 0.5069 | 0.4415 | 1.15x |
| 5 | 0.5617 | 0.4262 | 1.32x |
| 6 | 0.5615 | 0.4205 | 1.34x |
| 7 | 0.5698 | 0.3731 | 1.53x |
| **mean** | **0.5602** | **0.4155** | **1.35x** |

## Parity

| seed | gate on bit=1 | gate on bit=0 | ratio |
|---|---|---|---|
| 0 | 0.6086 | 0.3570 | 1.70x |
| 1 | 0.4320 | 0.2797 | 1.54x |
| 2 | 0.6138 | 0.4197 | 1.46x |
| 3 | 0.3563 | 0.2578 | 1.38x |
| 4 | 0.5549 | 0.4077 | 1.36x |
| 5 | 0.5949 | 0.3293 | 1.81x |
| **mean** | **0.5267** | **0.3419** | **1.54x** |

## Verdict

**NOT confirmed**: the gate is only 1.35x larger on actions than on observations. The token-suppression story does not explain the torus win, and the capacity reading stands — GateAngle and NoBottleneck buy the same thing for the same ~8k parameters.

Inference only, on existing checkpoints. n=8 torus seeds, 24 trajectories each.
