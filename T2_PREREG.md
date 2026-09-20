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

## Amendment 1 (2026-09-20) -- T2b: does the phase pay once Dyck's accumulator IS a clock?

T2 falsified its own prediction: making Dyck's accumulator a clock (alpha 0.6 -> 1.05, verified) did
not make PoPE collapse; its encoding helped MORE (+0.159 against +0.058). That kills the claim
"PoPE collapses on clocks". The account's OTHER half -- "a per-token phase pays on clocks" -- is
separable and untested here, and this amendment tests it inside the same task.

Arms, 8 seeds, same recipe: `MapPoPE_abs_T3` (monotone increments + per-token phase, forced init 0.1)
and `MapPoPE_abs_T3inert` (identical parameters, phase gated to zero). Baseline in hand:
`MapPoPE_abs` 0.836 at L128 D12. On SIGNED Dyck the same phase contrast was -0.046 / -0.047 against
its inert twin, i.e. harmful on a map accumulator.

- **T2b** phase - inert twin at L128 D12. **Registered prediction: POSITIVE and detectable.** If the
  phase pays here, where it did not on the signed version of the same task at the same size and
  recipe, the boundary survives in the narrow form "the compensating phase pays on clocks", with the
  strong form ("PoPE collapses on clocks") already dead.
- **Falsification**: a null or negative result means BOTH halves of the account are Bach-specific,
  and the honest position becomes that the whole clock/map story is an association across three tasks
  that no within-task manipulation reproduces.
- Reported: the same contrast at the training cell (should be small), and the learned phase magnitude.
