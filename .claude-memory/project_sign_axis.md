---
name: project-sign-axis
description: A monotone phase increment cannot represent a -1 action; measured, and it costs MapFormer its entire advantage over RoPE. Also: the sign axis is NOT ours.
metadata:
  type: project
---

**The sign axis is published prior art — do not claim it.** Sarrof et al.
(2405.17394) observed that SSM gates are "always nonnegative due to exponential or
sigmoid parameterizations", with Theorem 2: such models cannot recognise PARITY.
Grazzi et al. (2411.12537, ICLR 2025) proved positive-only eigenvalues cannot solve
parity and fixed it by extending to [-1,1]. PaTH adopts it deliberately, RWKV-7
restricts it only for stability, and **Selective RoPE §4.2 already carried it into
content-dependent phase in softmax attention** — citing Grazzi, arguing its
rotations "model flips", and demonstrating single-layer Transformer parity. What is
untaken is the *navigation* regime and a clean isolation.

**Measured 2026-09-06** (`SIGN_ABLATION.md`, clean torus, 6 arms x 12 seeds, one
batch, r=4, identical parameter count 204,757):

| arm | Δ | T=128 | T=512 | T=1024 | final loss |
|---|---|---|---|---|---|
| Signed | `W_out W_in x` | 1.000 | 0.978 | 0.922 | 0.0002 |
| Abs | `\|W_out W_in x\|` | 0.946 | 0.675 | 0.558 | 0.171 |
| Pos | `softplus` (GRAPE-AP) | 0.977 | 0.798 | 0.584 | 0.068 |
| CARoPE | `1/(softplus+1)` | 0.900 | 0.809 | 0.645 | 0.293 |
| RoPE | index | 0.799 | 0.449 | 0.345 | 0.784 |

**The headline: at matched training loss, monotone path integration beats RoPE
NOWHERE** (-0.028 at T=128, unmeasured beyond), while signed beats it +0.123/+0.195
at T=512/1024 on 12/12 seeds. Take the sign away and content-dependent phase is
worth nothing over an index clock.

**Mechanism confirmed at the level of the learned code** (`SIGN_PROBE.md`).
Opposition score `||Δ(+x)+Δ(-x)||/mean||Δ||`, where 0 = perfect cancellation and
2 = identical: signed **0.11-0.13**, monotone **1.85-1.98**. CARoPE's own
parameterisation reaches 1.98 of 2.00 and collapses its N/E axes (|cos| 0.93 vs
0.13-0.22). They cannot represent a -1 action, and they don't.

## Two method lessons, both mine

1. **My pre-registered discriminator required a signal in a cell at ceiling**
   (baseline 1.000 +/- 0.000, headroom 0.054, MDE 0.057) — the rule-11 ceiling trap
   written into my own verdict rule. When a verdict branch fires, check whether its
   cell *could* have gone the other way.
2. **The training-length effect was in the LOSS, not the accuracy** (12/12 seeds,
   Abs +0.171). Loss-matching at T=128 partials out the very quantity the
   constraint causes, and r(loss,acc) = -0.978 there. Ceiling on accuracy does not
   mean no effect — look at what the model could not fit.

See [[reference-positional-landscape]], [[reference-paper-corpus]].
