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

---

# ROTATE, REDONE — and it overturns the section above

The `rotate` row above is void. Redone with two fixes, gated BEFORE training this
time (orders 1-5 at 0.501/0.472/0.440/0.462 against a 0.507 marginal, PASS):

1. revisit keyed on the observation-determining state, not `(x, y, heading)`
2. `--score-moves-only`: skip steps where the observed cell did not change

**Fix 1 alone changed nothing** (order-1 stayed at 0.899). Spinning at an
already-seen cell still emits the same observation. It took fix 2.

## Two budgets, because the fix costs supervision

The scored rate falls from 0.727 to **0.054** (baseline 0.225), so at equal
epochs rotate gets ~4x fewer gradient-contributing events than every other
condition. `matched` gives it 4x the batches; `standard` keeps the sweep recipe.

| condition | floor | Vanilla | RoPE (index) | position effect | paired |
|---|---|---|---|---|---|
| rotate, standard budget | 0.508 | 0.512 ± 0.002 | 0.508 ± 0.007 | +0.004 | −0.000/+0.003/+0.009 |
| **rotate, matched budget** | 0.508 | **0.557 ± 0.023** | **0.508 ± 0.007** | **+0.049** | +0.063/+0.053/+0.031 |

At the standard budget **both arms sit exactly on the floor** — the condition is
unlearnable there, and its +0.004 would have been a false negative. Rule 5 again.

At the matched budget the index model is still *exactly* at the floor (0.508 vs
0.508, it learns nothing at all) while the path-integrated model clears it by
+0.049 on 3/3 seeds.

## This is comparable to baseline, and it makes rotate the dominant knob

`score_moves_only` is a **no-op in translate mode** — the agent moves on every
step, so no step is skipped. Baseline, ego, small and richobs are therefore
unaffected by the rule change, and rotate's position effect can be read against
them directly. (Only `wall` would be affected, since bumps do not move.)

| knob | position effect | reduction from baseline |
|---|---|---|
| baseline | +0.478 | — |
| **rotate** | **+0.049** | **−0.429** |
| small | +0.324 | −0.154 |
| richobs | +0.299 | −0.179 |
| ego | +0.265 | −0.213 |
| wall | +0.251 | −0.227 |
| allcombined | −0.084 | −0.562 |

**The "every knob contributes about equally" conclusion above is WRONG and is
withdrawn.** Rotation actions cut the position effect by 0.429 — roughly twice
the next largest knob and 90% of the total available. A single knob very nearly
accounts for the whole flip; the other four together add the remaining ~0.13.

## What this does and does not vindicate

It **partly** supports the mechanism I proposed for MiniGrid: MapFormer's path
integrator cumsums fixed per-token deltas, and under turn/turn/forward the
displacement depends on accumulated heading, which that form cannot represent.
Rotation actions are indeed the single most damaging property.

It does **not** support the strong version. Path integration does not become
harmful here — it stays positive on 3/3 seeds. What happens is that the task
becomes nearly unlearnable for everyone: Vanilla falls from 0.989 to 0.557, and
the index model, which reached 0.511 at baseline, learns literally nothing
(0.508 against a 0.508 floor). Rotation does not invert the ordering; it
collapses the ceiling.

The remaining negative sign at `allcombined` (−0.084) must therefore come from
rotate's collapse combined with the other four knobs, not from rotate alone.

## Still open

Whether MapFormer recovers under an allocentric action recoding — emitting
absolute N/S/E/W displacements computed from the known heading instead of
turn/forward. If it does, the mis-specification account is confirmed and the fix
is stated. That experiment is now well motivated by a measured 0.429 effect
rather than by a single-seed MiniGrid observation.
