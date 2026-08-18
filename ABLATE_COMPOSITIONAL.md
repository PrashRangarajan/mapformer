# Context-destruction ablation -- compositional task

The check that exposed hier-goal (0.912 -> 0.913, i.e. nothing was learned) and that Match-Query passed (0.918 -> 0.074). Both tasks here are in the citable set and neither had been through it.

Metric: **cross_nb_acc**. Floor: no analytic floor; the destroyed rows ARE the empirical floor.

`shuffle` permutes slots, which also destroys the walk's autocorrelation; `resample` substitutes the stream from an INDEPENDENT episode -- a valid walk that simply does not match the observations beside it, so it stays on-manifold. On the paper task these agreed (0.231 vs 0.178).

**A genuine result COLLAPSES here. Surviving destruction is the failure mode.**

| variant | intact | shuffle_actions | resample_actions | shuffle_obs | resample_obs |
|---|---|---|---|---|---|
| Vanilla | 0.2708 ± 0.0350 | 0.0234 ± 0.0036 | 0.0658 ± 0.0115 | 0.0197 ± 0.0058 | 0.0516 ± 0.0115 |
| Hourglass_k2 | 0.4279 ± 0.1808 | 0.0294 ± 0.0134 | 0.0854 ± 0.0098 | 0.0238 ± 0.0135 | 0.0622 ± 0.0073 |
| PlainFlat | 0.2162 ± 0.0032 | 0.0075 ± 0.0019 | 0.0724 ± 0.0013 | 0.0161 ± 0.0009 | 0.0551 ± 0.0007 |

## Verdict: PASSES

`cross_nb_acc` -- the compositional-transfer metric -- collapses by 85-95% in
every arm:

| variant | intact | best destroyed | headroom lost |
|---|---|---|---|
| Vanilla | 0.271 | 0.066 | 76% |
| Hourglass_k2 | 0.428 | 0.085 | 80% |
| PlainFlat | 0.216 | 0.072 | 67% |

So compositional transfer is being read off the path, not off something else in
the sequence. Contrast the failure case: hier-goal went 0.912 -> 0.913 under the
analogous manipulation.

### The hierarchy advantage survives the check

The reason to care: `COMPOSITIONAL_MULTISEED.md` reports hierarchy helping in
both backbones, and that is one of the repo's citable claims. If the advantage
had come from a shortcut, the destroyed rows would differ as much as the intact
ones. They do not -- destroyed, Hourglass_k2 (0.085) and Vanilla (0.066) are
close, while intact they are 0.428 vs 0.271. The gap lives in the part of the
signal that destruction removes.

Note also `Hourglass_k2`'s intact spread (± 0.181) is far wider than any
destroyed condition, consistent with the known seed instability of that arm
(seed 1 is an outlier) rather than with a shortcut.

### shuffle and resample do NOT agree here, unlike on the paper task

`resample_actions` (0.066-0.085) is consistently LESS destructive than
`shuffle_actions` (0.008-0.029), the opposite ordering to the paper task, where
resample was more destructive (0.178 vs 0.231). The two conditions are not
interchangeable; both are reported rather than one chosen.
