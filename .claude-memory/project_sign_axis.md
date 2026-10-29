---
name: project-sign-axis
description: The sign of a rotary phase increment is a LEARNABILITY cost, not a representational one (a non-negative rotary step can represent -1 mod 2 pi; CORRECTED 2026-10-12). It survives matched length (T=1024, -0.177, 0/8 vs 8/8), best rates on ring and torus, and an SSM; it is the |.| and softplus increments (exp: no cost in the SSM at lr 1e-3, a cost in attention on the torus). The representational sign cost (real eigenvalues) is published; the learnability result is ours. Current statement: CLAUDE.md sign row.
metadata:
  type: project
---

**CORRECTED 2026-10-29 -- read this first.** CLAUDE.md's sign row is the current statement; this file is the history
up to 2026-10-13. Where the body below says a monotone or non-negative code "cannot represent" a -1 action or push/pop, read
"does not learn": rotary steps wrap mod 2 pi, so the constrained class represents the map (RING_DIAL audit M1). Since
2026-10-13, all registered: MECHANISM (|.| removes the path through zero; a past-pi start learns by wrapping on the ring, not
on the torus -- TORUS_START), GENERALITY (torus 0/8 vs 8/8; the exact non-negative solution is perturbed and re-found),
ACCOUNTS (the '87% -> 18%' drift below counted unused channels; usage-weighted the held code stays 0.98), GAUGE (+1/+2
UNMEASURED), ROBUST (partly unmeasured), LRSWEEP and TORUS_LRSWEEP (cost at each arm's best rate, ring and torus), SIGNROUTE
(the cost is the |.| increment, not a non-negative readout), SIGNSEP (route vs barrier UNMEASURED), G3_SSM and its A2 block
(cost in a diagonal complex SSM; exp no cost at lr 1e-3), EXPTORUS (exp and softplus pay in attention on the torus).

**Matched-length control, 2026-09-28 (`SIGN_MATCHED_RESULTS.md`): SIGN IS CAPABILITY.** Trained AND
tested at T=1024: Abs - Signed -0.177 (perm p 0.0002), solved 0/8 vs 8/8; signed opposition 0.06 vs
monotone 1.92-1.97. The earlier "cost is extrapolation-only" reading (T=128 training) is superseded.
Monotone arms stalled, so budget-scoped.

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

**SCOPE (2026-09-24): the accuracy cost is extrapolation-only.** Trained T=128; at T=128 the
monotone arms score 0.90-0.98 against index 0.80, and the training-length evidence is the LOSS (12/12
worse). No arm has been trained and tested at matched length (a trained-at-T=1024 `Abs_r4` is the
direct test; `Vanilla_r4` there solves 8/8). Treat the headline below as robustness until then.

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

See [[reference-positional-landscape]], [[reference-review-documents]].

**REPLICATED ON A THIRD TASK (2026-09-20, `T2_RESULTS.md` manipulation check).** Constraining the
increment to be non-negative on Dyck-2 moves the accumulator from a map to a clock within one task
(alpha 0.578/0.618 -> 1.056/1.044, range growth 2.3x -> 4.2x over L32->L128) and costs 0.191 (MapWM)
and 0.091 (MapPoPE) F1 at L128 D12 -- a monotone code cannot represent push/pop, now shown on a
formal language as well as on navigation. This half of T2 stands independently of the PoPE corollary
that T2 withdrew.

**CORRECTION 2026-10-12 (RING_DIAL audit M1, verified by construction).** For a ROTARY phase, "monotone" is not a
representational limit: angles wrap mod 2 pi, so a second token's non-negative step 2 pi / omega - Delta(+1) equals
-Delta(+1) mod 2 pi, and Abs / Pos / CARoPE can reproduce any signed code exactly. The matched-length cost (-0.177, 0/8 vs
8/8 on the torus) stands as a LEARNABILITY result (search within 900 epochs); the opposition score (1.92-1.97) is not
wrap-aware and does not show inability. Howard's Case II vs III distinction holds for real (decaying) rates, not for
rotations. On the 1D ring (T=128) a pilot Abs model solved at p_plus 0.5 (n=1, RING_DIAL pilot).

**Existence (2026-10-13, `WRAP_EXISTENCE_RESULTS.md`, registered):** on the ring with observation-step-free classes, an exact
non-negative solution (balanced construction: orthogonal action latents, omega' = max(|omega|, 1)) is HELD under training
(8/8 SOLVED, never lost); from scratch the class solves 0/8 vs signed 8/8. So the sign cost is search. But the held model
drifts off the cancelling code (87% -> 18% of channels), and the naive wrap construction (keep W_in) is ill-conditioned and
gets lost under Adam -- do not read that as intrinsic fragility.
