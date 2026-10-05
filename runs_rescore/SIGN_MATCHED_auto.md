# Does refining theta help when the ACTIONS ARE NOISY?

The refine-theta loop was first tested on Match-Query, where actions are clean
and the query phase is blind -- neither half of the InEKF premise holds there, and
the null merely replicated a known negative. Action noise is the regime the
mechanism was built for: the action RECORD is corrupted while the agent moves per
the true action, so the path integral drifts and the observations (which reflect
TRUE position) carry the correction signal.

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=1024

| p_action_noise | Signed_r4 | Abs_r4 | Pos_r4 | RoPE |
|---|---|---|---|---|
| 0 | 0.998 ± 0.004 | 0.821 ± 0.120 | 0.781 ± 0.193 | 0.735 ± 0.009 |

