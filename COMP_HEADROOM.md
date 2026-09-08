# Compositional headroom: is 0.415 a capability limit or a recipe limit?

`cross_nb` on a fresh held-out environment (seed 10000), the same evaluator as the published table. Floor ~0.072, ceiling 1.0. Pre-registration: `COMP_HEADROOM_PREREG.md`.

| arm | recipe | final loss | cross_nb T=256 | cross_nb T=512 | n |
|---|---|---|---|---|---|
| **A** | published recipe: linear, 3e-4, 50ep | 0.7466 | **0.354 ± 0.070** | **0.237 ± 0.068** | 8 |
| **B** | cosine, 1e-3, 50ep (budget held) | 0.5579 | **0.460 ± 0.151** | **0.352 ± 0.124** | 8 |
| **C** | cosine, 1e-3, 150ep | 0.4760 | **0.514 ± 0.126** | **0.390 ± 0.126** | 8 |
| **D** | loop, cosine, 1e-3, 150ep (1/3 the params) | 0.4095 | **0.542 ± 0.170** | **0.440 ± 0.183** | 8 |

## Pre-registered contrasts

| contrast | tests | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| `C - A` @ T=256 | P1: the whole recipe | +0.160 | 0.141 | 0.140 | 7/8 | **DETECTABLE** |
| `B - A` @ T=256 | P2: schedule + lr alone | +0.107 | 0.151 | 0.150 | 6/8 | unmeasured |
| `C - B` @ T=256 | P2: budget alone | +0.053 | 0.133 | 0.132 | 6/8 | unmeasured |
| `D - C` @ T=256 | P4: the loop, at 1/3 params | +0.029 | 0.168 | 0.166 | 4/8 | unmeasured |
| `C - A` @ T=512 | P1: the whole recipe | +0.153 | 0.144 | 0.143 | 7/8 | **DETECTABLE** |
| `B - A` @ T=512 | P2: schedule + lr alone | +0.115 | 0.132 | 0.131 | 7/8 | unmeasured |
| `C - B` @ T=512 | P2: budget alone | +0.038 | 0.101 | 0.100 | 6/8 | unmeasured |
| `D - C` @ T=512 | P4: the loop, at 1/3 params | +0.050 | 0.211 | 0.209 | 4/8 | unmeasured |

## P3: does the recipe compress spread?

| arm | sd at T=256 |
|---|---|
| A | 0.070 |
| B | 0.151 |
| C | 0.126 |
| D | 0.170 |

Pre-registered: the recipe should take sd from $0.096$ to below $0.05$. A mean gain with unchanged variance means something other than optimisation.

**A is the reproduction control.** If it does not land near $0.415$ the batch is not comparable to the published table and nothing above is readable.

---

## Reading it

**The reproduction control passes, but not tightly.** A gives $0.354 \pm 0.070$
against the published $0.415 \pm 0.096$ at the same recipe, seeds, variant and
evaluation protocol (n_traj=200, matched after finding the aggregator defaults
differed). The gap is $0.061$ with a pooled se of $0.042$, $t = 1.45$ --- the same
distribution, on a task whose seed spread is this wide. The batch is comparable; it
is not a bit-reproduction, and nothing here should be read to three decimals.

## P1 CONFIRMED, and it is the largest single effect on this task

**`C - A` is +0.160 at T=256 and +0.153 at T=512, detectable on 7/8 seeds.** The
recipe is worth more than any architectural ingredient ever measured here on
compositional transfer --- more than hierarchy itself (+0.13). The published
compositional numbers were **undertrained**, not capability-limited.

## P2 unresolved: schedule and budget cannot be separated at n=8

`B - A` (schedule + lr, budget held) is $+0.107/+0.115$ and `C - B` (budget alone)
is $+0.053/+0.038$. Both point the same way, neither clears its own \textsc{mde},
and they sum to the detectable total. The reading is that the schedule and learning
rate carry most of it and the budget adds a little, but that ordering is
**unmeasured** and stated as such.

## P3 REFUTED, and the refutation is informative

Pre-registered: the recipe should take the seed sd from $0.096$ to below $0.05$,
and *"a mean gain with unchanged variance means something other than optimisation is
happening."* Measured, variance went the other way:

    A 0.070  ->  B 0.151  ->  C 0.126  ->  D 0.170

**The spread roughly doubled.** So this is not the same phenomenon as `RECIPE_POWER`
on the torus, where the identical change cut seed sd 3.5x. Here a higher learning
rate and a longer budget raise the mean and *widen* the distribution --- some seeds
find much more than others. By the criterion written down in advance, that is not
pure optimisation, and the mechanism is not identified.

## P4 as predicted: the loop matches 3x its parameters

`D - C` is $+0.029/+0.050$, unmeasured, 4/8 --- at **209,256 parameters against
605,800**. Pre-registered as the expected good outcome: matching is the result, and
D beating C would have been remarkable. The loop reproduces on compositional what it
did on parity and Match-Query.

## The consequence that outranks all of this

**Every published compositional conclusion was measured under the recipe that arm A
reproduces, and the recipe is worth more than the effects those conclusions turn
on.** In particular *"hierarchy buys compositional transfer"* rests on
`MapWM-Hier - MapWM-FlatHG` $= +0.130$, measured entirely inside arm A's regime,
against a recipe effect of $+0.160$. That comparison has to be re-run at the C
recipe before the claim can stand. It is the first thing to do next, and it takes
precedence over the queued follow-ups.
