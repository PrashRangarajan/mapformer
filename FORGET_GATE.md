# MapFormer with a forget gate: the empty Re G cell

Torus paper task, held-out map, evaluated under the same noise it trained on.

## T=128

| p_action_noise | Vanilla | Vanilla_r4 | Forget | Forget_r4 |
|---|---|---|---|---|
| 0 | 0.993 ± 0.017 | 1.000 ± 0.000 | 0.991 ± 0.021 | 1.000 ± 0.000 |

## T=512

| p_action_noise | Vanilla | Vanilla_r4 | Forget | Forget_r4 |
|---|---|---|---|---|
| 0 | 0.944 ± 0.028 | 0.982 ± 0.005 | 0.967 ± 0.038 | 0.978 ± 0.008 |

## T=1024

| p_action_noise | Vanilla | Vanilla_r4 | Forget | Forget_r4 |
|---|---|---|---|---|
| 0 | 0.834 ± 0.064 | 0.919 ± 0.012 | 0.914 ± 0.041 | 0.918 ± 0.018 |

## What the forget gate learned

| arm | lambda (mean ± sd) | per-seed | E[sigmoid] | bias at lag 128 |
|---|---|---|---|---|
| `Forget` | +0.0118 ± 0.0334 | +0.016 -0.001 -0.003 -0.003 +0.011 -0.006 -0.010 +0.092 | 0.253 | -0.38 logits |
| `Forget_r4` | +0.0141 ± 0.0048 | +0.011 +0.015 +0.005 +0.022 +0.014 +0.016 +0.017 +0.013 | 0.253 | -0.46 logits |

lambda > 0 = decay, ~0 = declined (escapable start verified), < 0 = anti-recency.

## Verdict: the gate helps, and NOT by forgetting

Paired, $n=8$, loss-matched against final training loss (the last-5-epoch mean
from the checkpoints -- **not** the eval NLL in the JSON, which is a readout of
the same softmax as the accuracy and so is circular).

| contrast | T=512 raw | T=512 matched | T=1024 raw | T=1024 matched |
|---|---|---|---|---|
| gate at r=2 | +0.022 | +0.029 | **+0.081** (7/8) | **+0.086** (8/8) |
| gate at r=4 | -0.004 | -0.004 | -0.002 | -0.002 |
| rank, no gate | **+0.038** (8/8) | **+0.029** (7/8) | **+0.085** (8/8) | **+0.078** (8/8) |
| rank, with gate | +0.011 | -0.005 | +0.003 | -0.010 |
| **interaction** | -0.027 | | **-0.082** (2/8) | |

r(final training loss, accuracy) is -0.716 at T=512 and **-0.311** at T=1024, so
loss-matching barely moves the T=1024 column and the effects are not convergence
artifacts.

**PRE-REGISTERED PREDICTION WRONG ON ACCURACY.** I predicted neutral-to-negative.
The gate is worth **+0.086** at $r=2$, $8/8$ seeds.

**PRE-REGISTERED PREDICTION RIGHT ON MECHANISM, and the two are separable because
lambda was measured.** The gain is *anti-correlated* with how much the model
actually forgets:

| seed | Vanilla | Forget | gain | lambda |
|---|---|---|---|---|
| 5 | 0.729 | 0.943 | **+0.214** | -0.0063 (anti-recency) |
| 4 | 0.749 | 0.895 | +0.147 | +0.0106 |
| 1 | 0.820 | 0.951 | +0.132 | -0.0007 (flat) |
| 2 | 0.864 | 0.956 | +0.093 | -0.0026 (anti-recency) |
| 6 | 0.874 | 0.932 | +0.058 | -0.0101 (anti-recency) |
| 3 | 0.912 | 0.925 | +0.013 | -0.0035 (anti-recency) |
| 0 | 0.869 | 0.871 | +0.002 | +0.0156 |
| 7 | 0.856 | 0.841 | **-0.015** | **+0.0916** (most decay) |

- **r(lambda, gain) = -0.516.** Seeds with lambda <= 0 gain **+0.102**; seeds that
  actually decay gain **+0.045**; the seed that decays most is the only one that
  **loses**.
- 5 of 8 seeds learn lambda < 0 -- mild **anti-recency**, the opposite of the
  design principle -- and those are the seeds that gain most.

**So the decay mechanism is falsified as the explanation**, by the arm's own
learned coefficient. What remains is $+259$ parameters on the attention path,
whose effect is concentrated where the baseline is worst: gain **+0.119** on the
worst four Vanilla seeds against **+0.042** on the best four, sd $0.064 \to 0.041$,
min $0.729 \to 0.841$.

**Gate and rank are SUBSTITUTES.** Each alone is worth ~$+0.08$; together, nothing
extra (interaction $-0.082$, $2/8$). Both raise the floor at $r=2$ and neither adds
once the other has. This is the same shape as `SELECTIVE_ROPE.md`'s finding that
two unrelated ways of spending ~8k parameters bought the same thing -- now at 259
parameters, and now with the named mechanism explicitly ruled out rather than
merely unsupported.

**The control this needs**, and it is the same one `V4_MULTISEED.md` needed: an arm
with the gate parameters present but lambda frozen at zero, so the mechanism is
provably off while parameter count and initialisation RNG are matched. If it
matches `Forget`, the gain is parameters and not decay, definitively. 8 runs.

## On the rotation-and-decay principle

Selective RoPE's principle is that recall needs both. On a cognitive-map task the
optimiser, from a start verified escapable (bit-identical to Vanilla at
lambda = 0; |grad| 5.0e-03 against a 4.8e-04 median; lambda leaves zero at step 1),
**chooses essentially no decay and slightly negative decay in most seeds**. The
accumulated bias it settles on is -0.38 to -0.46 logits at lag 128. Decay does not
transfer out of the language and finite-state-recall regimes it was derived in --
which is what the task structure predicted, since the scored event is retrieval of
the *first* visit to a cell.
