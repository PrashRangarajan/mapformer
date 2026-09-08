# Is the forget gate a clock? The one live account of the length axis

## Why this is the experiment worth running

Four separate mechanisms help at extrapolated length; the accumulator's growth
exponent accounts for **two** and demonstrably not the others, and the imported
long-context account was refuted here. That is the largest unexplained axis in the
project.

One account survives, and it is currently **untested**. The forget gate carries its
own accumulator, `sum log gamma`, whose exponent is **alpha = +0.956** -- ballistic,
against the phase accumulator's 0.6. On the clock/map reading a signed phase gives
up the clock in exchange for the map, so a second monotone accumulator restores it.
That story already explains every otherwise-odd detail: the gate needs a **live**
lambda but not **decay** (frozen at zero it lands on the baseline, -0.016), and 5 of
8 seeds learn `lambda < 0` -- still monotone, counting up instead of down -- and
those gain most.

`positional_review.tex` states the prediction and marks it untested: *"the gain
should vanish on a task with no recency structure."* Both halves of that test now
exist.

## Arms -- rank-matched, which the obvious choice was not

`Forget_r4` is rank 4 and `Forget_Frozen` is rank 2, so pairing them would confound
the gate with the rank. The r=2 family is used throughout: it is internally matched
AND it is where the published +0.086 was measured.

| arm | rank | params | isolates |
|---|---|---|---|
| `Vanilla` | 2 | 204,373 | baseline |
| `Forget` | 2 | 205,660 | the mechanism |
| `Forget_Frozen` | 2 | 205,659 | gate present, cannot adapt -- separates "a gate that runs" from "an extra module" |

Two tasks, 8 seeds, one batch each.

## The ceiling trap, checked in advance

The recency baseline is **exactly 1.000** at T=1024 and leaves only ~0.023 at
T=2048, so neither could show a gain of any size. **T=4096 is added as the primary
clock-side length**, and the headroom at each length is reported before any contrast
is read. This is the third time in this project a pre-registration has been written
against a ceiling; it is checked here rather than discovered afterwards.

## Predictions

**P1 (the clock claim).** The gate helps MORE on recency, which needs a clock, than
on the torus, which does not. The quantity is the **interaction**
`gain(recency) - gain(torus)`, predicted positive.

**P2 (where on the torus).** Its torus benefit sits at OOD length and not at
training length -- the signature of restoring a code that has left its trained
range, not of added capacity.

**P3 (the control).** `Forget_Frozen` lands on `Vanilla`, reproducing the published
-0.016. If it does not, the module and not the gate is doing the work and nothing
else here is readable.

**P4 (the mechanism readout).** `sum log gamma` should show alpha ~ 0.95 on both
tasks -- it is monotone by construction -- while the phase accumulator's alpha
should differ between them, as it did in the crossover (0.591 vs 0.967).

## Falsifiers

- **Interaction <= 0, i.e. the gate helps the map task as much as the clock task:
  the clock account is withdrawn**, and the length axis is left with no explanation
  at all rather than a partial one.
- `Forget_Frozen` not landing on `Vanilla`: the batch is unreadable, stop.
- Everything inside the 0.150 noise floor: unmeasured, not null.

## Checks before reading

Convergence (rule 10), r(final loss, accuracy) per task and length (rule 9),
MDE beside every contrast (rule 11), and the headroom at each eval length stated
before the contrasts.
