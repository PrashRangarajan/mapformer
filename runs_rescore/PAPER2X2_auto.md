# Does refining theta help when the ACTIONS ARE NOISY?

The refine-theta loop was first tested on Match-Query, where actions are clean
and the query phase is blind -- neither half of the InEKF premise holds there, and
the null merely replicated a known negative. Action noise is the regime the
mechanism was built for: the action RECORD is corrupted while the agent moves per
the true action, so the path integral drifts and the observations (which reflect
TRUE position) carry the correction signal.

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | MapPoPE-Flat | MapPoPE_r4 | PoPE-Flat | RoPE | Vanilla | Vanilla_r4 |
|---|---|---|---|---|---|---|
| 0 | 0.999 ± 0.001 | 1.000 ± 0.001 | 0.692 ± 0.027 | 0.816 ± 0.015 | 0.971 ± 0.043 | 1.000 ± 0.000 |

