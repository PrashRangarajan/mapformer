# Compositional transfer: can the largest headroom in the project be claimed?

`cross_nb` on the compositional-motif task is **0.415** against a floor of ~0.072
and a ceiling of 1.0 -- **0.585 unclaimed**, where the torus (0.994) and Match-Query
(0.986) have essentially none. It also carries the **largest seed variance** here
(+/-0.096, with one seed at 0.625 against ~0.30 for the rest).

## Why the recipe is the first bet, not the loop

Every published compositional number was trained with `LinearLR(1.0 -> 0.0)` from
step one at `lr 3e-4` for 50 epochs -- the regime standing rule 10 exists for, since
it decays with no warmup and cannot escape a plateau late. `RECIPE_POWER.md` measured
what fixing it does elsewhere: seed sd fell **3.5x** at T=512 with a HIGHER mean at
every length, and rule 5 was bought by an arm that moved **0.448 -> 0.990** on the
same task from a budget change alone.

So on the task with the most headroom and the most variance, the cheapest untried
change is the one this project already knows is worth the most.

## Arms (8 seeds each, one batch)

| arm | schedule | lr | epochs | isolates |
|---|---|---|---|---|
| A `Hourglass_k2` | linear | 3e-4 | 50 | **the published recipe** -- must reproduce ~0.415 |
| B `Hourglass_k2` | cosine | 1e-3 | 50 | schedule + lr, **budget held** |
| C `Hourglass_k2` | cosine | 1e-3 | 150 | schedule + lr + budget |
| D `LoopedHourglass` | cosine | 1e-3 | 150 | the loop, at whatever recipe is best |

A is the reproduction control: if it does not land near 0.415 the batch is not
comparable to the published table and nothing else here is readable.

## Predictions

**P1 (the bet).** C > A by a margin larger than the 0.150 noise floor. The
mechanism is not subtle -- the published runs may simply be undertrained.

**P2 (which half).** If B ~= A and C > B, it is budget. If B > A, it is the
schedule. Stated in advance because "the recipe helped" is uninformative without it.

**P3 (variance, and the sharper claim).** The recipe should compress **spread** at
least as much as it lifts the mean: sd 0.096 -> below 0.05. A mean gain with
unchanged variance would mean something other than optimisation is happening.

**P4 (the loop).** D is **not parameter-matched** -- 209,256 against 605,800, a
2.9x deficit, because the loop shares one block where the hourglass has three. So D
beating C would be remarkable and D matching C is the expected good outcome; this is
a parameter-efficiency test, as it was on parity and Match-Query where the loop
matched or beat 3x its parameters. **D losing to C is not evidence against the
loop.**

## Falsifiers and checks

- If A does not reproduce ~0.415, stop: the batch is not comparable.
- If C ~= A, the published numbers were converged after all and the headroom is
  real capability rather than optimisation -- which redirects the search entirely.
- Convergence (rule 10), r(final loss, accuracy) per arm (rule 9), and MDE beside
  every contrast (rule 11) before any reading.
- The task's own gates are unchanged; only the optimiser and the arm differ.
