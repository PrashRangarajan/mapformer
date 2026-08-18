# Context-destruction ablation -- family task

The check that exposed hier-goal (0.912 -> 0.913, i.e. nothing was learned) and that Match-Query passed (0.918 -> 0.074). Both tasks here are in the citable set and neither had been through it.

Metric: **revisit accuracy**. Floor: **0.163** (hub-node baseline; the 0.125 chance is NOT the floor).

`shuffle` permutes slots, which also destroys the walk's autocorrelation; `resample` substitutes the stream from an INDEPENDENT episode -- a valid walk that simply does not match the observations beside it, so it stays on-manifold. On the paper task these agreed (0.231 vs 0.178).

**A genuine result COLLAPSES here. Surviving destruction is the failure mode.**

| variant | intact | shuffle_actions | resample_actions | shuffle_obs | resample_obs |
|---|---|---|---|---|---|
| MapEM_NC_NL | 0.7282 ± 0.0088 | 0.2486 ± 0.0100 | 0.2740 ± 0.0071 | 0.1626 ± 0.0034 | 0.2487 ± 0.0027 |
| VanillaEM_P0 | 0.7126 ± 0.0087 | 0.2477 ± 0.0085 | 0.2709 ± 0.0056 | 0.1720 ± 0.0016 | 0.2551 ± 0.0057 |
| PlainFlat | 0.6123 ± 0.0335 | 0.2432 ± 0.0025 | 0.2615 ± 0.0023 | 0.1852 ± 0.0083 | 0.2565 ± 0.0041 |

## Verdict: PASSES

All three variants collapse. `MapEM_NC_NL` falls 0.728 -> 0.249 (actions
shuffled) / 0.274 (actions resampled), i.e. it loses ~80% of its headroom above
the 0.163 hub floor. The relation tokens are load-bearing; the number was not
coming from node-visit frequency.

**The cleanest signal is `shuffle_obs` -> 0.1626, which lands on the hub-node
floor of 0.163 to three decimals.** Destroy the map and the model scores exactly
what "always answer with the most-visited node's observation" scores. That is the
behaviour a correct metric should have, and it was not built in -- the floor came
from `validate_family_tree.py` and the 0.1626 from a trained model.

Contrast with the failure case: hier-goal under the analogous manipulation went
0.912 -> 0.913.

### Residual above the floor, stated rather than smoothed over

`resample_actions` leaves 0.274, not 0.163. That residual is explicable: with
relations replaced the OBSERVATION stream is still intact, and observation values
recur, so a model that predicts from recent observation history alone gets
partial credit without knowing which person it is standing on. It is not evidence
of a shortcut in the intact task -- the intact score is 0.728 -- but the task's
true "no context at all" floor is nearer 0.25 than 0.163 for the
actions-destroyed condition specifically.

### shuffle and resample do NOT agree here, unlike on the paper task

On the paper task `resample` was MORE destructive than `shuffle` (0.178 vs
0.231). Here it is LESS (0.274 vs 0.249), on both tasks in this file. So the two
conditions are not interchangeable and neither is a safe stand-in for the other;
both are reported.
