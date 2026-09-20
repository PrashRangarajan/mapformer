# T2_PREREG -- force a clock inside Dyck-2 (the within-task test of the boundary)

## Why

The clock/map boundary -- a per-token phase pays where the accumulator leaves its trained range and
not where it is bounded -- currently rests on comparing Bach (alpha 1.00, phase worth -3.9 NLL) with
Dyck-2 and the torus (bounded, phase worth -0.05). Those three differ in dataset, model size
(6 layers / 8 heads vs 1 / 1) and metric as well as in accumulator, so the boundary is a BETWEEN-TASK
association. `THEORY_MAPPOPE.md` registered T2 as the within-task version and it has not been run.

On Dyck the increments cancel because opens and closes move the accumulator in opposite directions.
Constraining them to be non-negative (`SignConstrainedActionToLie(mode="abs")`, the repo's existing
machinery, weights loaded from the signed parent so the arms start as the same function up to the
absolute value) turns the same task's accumulator into a clock and changes nothing else.

## Arms

`MapWM_abs` and `MapPoPE_abs`, 8 seeds, the `DYCK_PREREG.md` recipe verbatim (1 layer, 1 head, 560k
sequences at L=32 D=4). Signed baselines in hand from the same recipe: MapWM-1L 0.868 and MapPoPE-1L
0.927 at L128 D12, i.e. **PoPE's encoding is worth +0.059 on a bounded accumulator**.

## Registered verdicts

- **M (manipulation check, first)** alpha for the monotone arms rises toward 1.0 and the accumulator
  no longer returns to its starting value within a sequence. If alpha does not move, nothing else here
  is interpretable.
- **T2 (the primary contrast, a difference of differences)**
  `(MapPoPE_abs - MapWM_abs) - (MapPoPE - MapWM)` at L128 D12, paired by seed.
  **Registered prediction: NEGATIVE and detectable.** On a bounded accumulator PoPE's encoding helps
  (+0.059); if the account is right, making the same task's accumulator a clock should turn that into
  a cost, because PoPE's kernel then has an out-of-range argument it cannot compensate for.
- **Levels are NOT the test.** Both monotone arms will be much worse in absolute terms -- a monotone
  increment cannot represent push/pop at all, which this project measured on the torus (a monotone
  code beats an index code nowhere). That is why the test is the difference of differences.
- **Falsification**: if MapPoPE keeps its advantage over MapWM once the accumulator is a clock, the
  boundary is not about the accumulator within a task, and the three-task pattern is explained by
  something else those tasks differ in -- dataset, model size or metric.
- Also reported: alpha per arm, the per-distance closer accuracy (does the monotone pair lose the far
  buckets specifically?), and whether either monotone arm clears the no-stack n-gram floor at all.
