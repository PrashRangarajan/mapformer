# Does refining theta help when the ACTIONS ARE NOISY?

The refine-theta loop was first tested on Match-Query, where actions are clean
and the query phase is blind -- neither half of the InEKF premise holds there, and
the null merely replicated a known negative. Action noise is the regime the
mechanism was built for: the action RECORD is corrupted while the agent moves per
the true action, so the path integral drifts and the observations (which reflect
TRUE position) carry the correction signal.

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | RoPE | Vanilla | ConvAngle | NoBottleneck | GateAngle | SRoPEGen |
|---|---|---|---|---|---|---|
| 0 | 0.798 ± 0.029 | 0.993 ± 0.017 | 0.991 ± 0.019 | 0.994 ± 0.014 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## T=512

| p_action_noise | RoPE | Vanilla | ConvAngle | NoBottleneck | GateAngle | SRoPEGen |
|---|---|---|---|---|---|---|
| 0 | 0.497 ± 0.058 | 0.944 ± 0.028 | 0.913 ± 0.046 | 0.967 ± 0.020 | 0.984 ± 0.011 | 0.975 ± 0.022 |

## T=1024

| p_action_noise | RoPE | Vanilla | ConvAngle | NoBottleneck | GateAngle | SRoPEGen |
|---|---|---|---|---|---|---|
| 0 | 0.412 ± 0.094 | 0.834 ± 0.064 | 0.770 ± 0.096 | 0.892 ± 0.028 | 0.920 ± 0.030 | 0.882 ± 0.068 |

