# Does refining theta help when the ACTIONS ARE NOISY?

The refine-theta loop was first tested on Match-Query, where actions are clean
and the query phase is blind -- neither half of the InEKF premise holds there, and
the null merely replicated a known negative. Action noise is the regime the
mechanism was built for: the action RECORD is corrupted while the agent moves per
the true action, so the path integral drifts and the observations (which reflect
TRUE position) carry the correction signal.

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | Vanilla | Looped | LoopedRefine | Level15 |
|---|---|---|---|---|
| 0 | 0.954 ± 0.076 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.966 ± 0.059 |
| 0.1 | 0.764 ± 0.017 | 0.901 ± 0.009 | 0.891 ± 0.010 | 0.787 ± 0.012 |
| 0.25 | 0.638 ± 0.030 | 0.843 ± 0.028 | 0.848 ± 0.020 | 0.641 ± 0.015 |

## T=512

| p_action_noise | Vanilla | Looped | LoopedRefine | Level15 |
|---|---|---|---|---|
| 0 | 0.914 ± 0.074 | 0.780 ± 0.146 | 0.786 ± 0.060 | 0.931 ± 0.107 |
| 0.1 | 0.677 ± 0.021 | 0.629 ± 0.035 | 0.631 ± 0.031 | 0.705 ± 0.026 |
| 0.25 | 0.569 ± 0.009 | 0.606 ± 0.066 | 0.601 ± 0.068 | 0.593 ± 0.006 |

## Refinement gain vs noise level

| p_action_noise | T | refine − fixed θ | se | t |
|---|---|---|---|---|
| 0 | 128 | -0.001 | 0.000 | -1.95 |
| 0.1 | 128 | -0.011 | 0.008 | -1.35 |
| 0.25 | 128 | +0.005 | 0.020 | 0.25 |
| 0 | 512 | +0.006 | 0.091 | 0.07 |
| 0.1 | 512 | +0.003 | 0.027 | 0.09 |
| 0.25 | 512 | -0.005 | 0.055 | -0.10 |

**The pre-registered prediction is a POSITIVE SLOPE in this column.** A gain
that does not grow with noise is not the correction mechanism working -- the
premise is that noise creates drift and refinement removes it. A flat or
negative slope says the loop's benefit is iteration, on any input.
