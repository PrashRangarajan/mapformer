# Does refining theta help when the ACTIONS ARE NOISY?

The refine-theta loop was first tested on Match-Query, where actions are clean
and the query phase is blind -- neither half of the InEKF premise holds there, and
the null merely replicated a known negative. Action noise is the regime the
mechanism was built for: the action RECORD is corrupted while the agent moves per
the true action, so the path integral drifts and the observations (which reflect
TRUE position) carry the correction signal.

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | Vanilla | Level15 | Looped | Level15Looped | LoopedSampled |
|---|---|---|---|---|---|
| 0 | 0.947 ± 0.062 | 0.990 ± 0.030 | 0.999 ± 0.004 | 0.994 ± 0.014 | 0.997 ± 0.007 |

## T=512

| p_action_noise | Vanilla | Level15 | Looped | Level15Looped | LoopedSampled |
|---|---|---|---|---|---|
| 0 | 0.876 ± 0.071 | 0.953 ± 0.056 | 0.872 ± 0.119 | 0.929 ± 0.089 | 0.905 ± 0.066 |

## T=1024

| p_action_noise | Vanilla | Level15 | Looped | Level15Looped | LoopedSampled |
|---|---|---|---|---|---|
| 0 | 0.749 ± 0.084 | 0.878 ± 0.092 | 0.730 ± 0.166 | 0.830 ± 0.168 | 0.745 ± 0.120 |

