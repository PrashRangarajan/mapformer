# Which environment property flips the position effect?

Position effect = **Vanilla (path-integrated) − RoPE (index)**, both trained in the same batch at every condition (rule 3), n=3, 16 epochs, 1 layer. Knobs turned ONE AT A TIME from the torus baseline.

Floors are **measured per condition** — they move a lot here, so an accuracy without its column is meaningless (rule 4).

| condition | floor | Vanilla | RoPE (index) | **position effect** |
|---|---|---|---|---|
| baseline | 0.505 | 0.989 ± 0.010 | 0.511 ± 0.012 | **+0.478** |
| rotate | 0.504 | 0.990 ± 0.000 | 0.990 ± 0.000 | **+0.000** |
| ego | 0.508 | 0.800 ± 0.013 | 0.536 ± 0.016 | **+0.265** |
| wall | 0.502 | 0.862 ± 0.032 | 0.611 ± 0.004 | **+0.251** |
| small | 0.502 | 0.732 ± 0.257 | 0.409 ± 0.003 | **+0.324** |
| richobs | 0.505 | 0.805 ± 0.162 | 0.506 ± 0.024 | **+0.299** |
| allcombined | 0.558 | 0.725 ± 0.100 | 0.809 ± 0.019 | **-0.084** |

Reference points: torus paper task **+0.461** (n=8), MiniGrid-DoorKey-16x16 **−0.060** (n=3, frequency-controlled).


## Gate results — run AFTER training, which was a mistake

Standing rule 1 says n-gram the answer stream before training a new task. I ran
the sweep first. Gating afterwards (200 episodes per condition, n-gram on the
scored-answer stream alone, no model involved):

| condition | marginal | n-gram o1 | n-gram o3 | verdict |
|---|---|---|---|---|
| baseline | 0.516 | 0.517 | 0.449 | clean |
| **rotate** | 0.508 | **0.932** | **0.913** | **VOID** |
| ego | 0.513 | 0.510 | 0.436 | clean |
| wall | 0.493 | 0.488 | 0.561 | clean |
| small | 0.507 | 0.506 | 0.526 | clean |
| richobs | 0.516 | 0.512 | 0.467 | clean |
| allcombined | 0.536 | 0.509 | **0.634** | caveat, below |

### The `rotate` row is VOID and must be discarded

Its answer stream is **93% predictable from the previous answer alone**. Turns do
not translate, so an agent that samples "turn left" and repeats it for a run of
k spins in place; with revisit keyed on `(x, y, heading)` it returns to the same
state every 4 turns and emits the same observation. Copying the last answer
solves it.

That is why both arms read 0.990 ± 0.000 and the position effect reads +0.000.
**The task became trivial, not position-independent** — the +0.000 says nothing
about whether rotation actions break path integration, which was the question the
condition existed to answer. It needs a redesign (score only steps where the
agent actually translated, or drop heading from the revisit key and accept that
the egocentric observation then varies) and a re-run.

### `allcombined` carries a caveat, not a verdict

Order-3 reaches 0.634 against a 0.536 marginal (+0.098). Both trained arms exceed
it (0.725 and 0.809), so the condition is not solved by the shortcut, but its
−0.084 should be read as approximate.

## What the sweep actually shows

Discarding `rotate`:

| condition | position effect | change from baseline |
|---|---|---|
| baseline | **+0.478** | — |
| small (16² not 64²) | +0.324 | −0.154 |
| richobs (64 types not 16) | +0.299 | −0.179 |
| ego | +0.265 | −0.213 |
| wall | +0.251 | −0.227 |
| **allcombined** | **−0.084** | **−0.562** |

**My pre-registered prediction is REFUTED.** I predicted aliasing and size would
drive the effect and the embodiment knobs would not. Instead **every knob
reduces it by a similar amount, 0.15 to 0.23**, and none of them individually
comes close to flipping the sign. Egocentric observation and walls — the two I
expected to be irrelevant — are the two *largest* single reductions.

**No single property explains the flip. The combination does.** Turning all five
gives **−0.084**, which lands on MiniGrid-DoorKey-16x16's independently measured
**−0.060**. That agreement is the sweep's real result: it says the five knobs are
jointly sufficient to reproduce the target environment's behaviour in the torus
codebase, so nothing important about MiniGrid is missing from the list.

## Honest limits

- **One condition void, one caveated**, and the void one was the most
  theoretically interesting of the seven.
- **Two conditions have an unstable Vanilla seed**: `small` (+0.038 / +0.526 /
  +0.408) and `richobs` (+0.085 / +0.410 / +0.403). Their means are carried by
  two seeds each; the ±0.257 and ±0.162 are real.
- **n=3, 16 epochs, 1 layer** — the paper's recipe, but the smallest budget used
  anywhere in this repo.
- The decomposition is **not additive**: the individual reductions sum to −0.77
  while the combined effect is −0.56, so the knobs interact and single-knob
  deltas cannot be read as independent contributions.
