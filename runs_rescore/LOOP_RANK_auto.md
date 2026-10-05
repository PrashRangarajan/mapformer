# Does refining theta help when the ACTIONS ARE NOISY?

The refine-theta loop was first tested on Match-Query, where actions are clean
and the query phase is blind -- neither half of the InEKF premise holds there, and
the null merely replicated a known negative. Action noise is the regime the
mechanism was built for: the action RECORD is corrupted while the agent moves per
the true action, so the path integral drifts and the observations (which reflect
TRUE position) carry the correction signal.

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=1024

| p_action_noise | Looped | Vanilla_L4 |
|---|---|---|
| 0 | 0.965 ± 0.009 | 0.973 ± 0.012 |

