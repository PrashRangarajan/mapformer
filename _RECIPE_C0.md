# Does refining theta help when the ACTIONS ARE NOISY?

The refine-theta loop was first tested on Match-Query, where actions are clean
and the query phase is blind -- neither half of the InEKF premise holds there, and
the null merely replicated a known negative. Action noise is the regime the
mechanism was built for: the action RECORD is corrupted while the agent moves per
the true action, so the path integral drifts and the observations (which reflect
TRUE position) carry the correction signal.

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | Vanilla | Looped |
|---|---|---|
| 0 | 0.936 ± 0.086 | 1.000 ± 0.001 |

## T=512

| p_action_noise | Vanilla | Looped |
|---|---|---|
| 0 | 0.863 ± 0.096 | 0.910 ± 0.073 |

## T=1024

| p_action_noise | Vanilla | Looped |
|---|---|---|
| 0 | 0.735 ± 0.110 | 0.733 ± 0.138 |

