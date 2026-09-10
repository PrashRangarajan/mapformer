# A kernel theory of positional encoding for cognitive maps

Written 2026-09-09, after the `runs/recency_em` batch refuted two of three
pre-registered predictions (`RECENCY_EM_RESULTS.md`). It is a **retrodiction** of
that batch and of six earlier results; Sec 8 lists what it predicts that has not
been measured, which is the only part that can earn it anything.

## 1. The object every result in this project is about

For any model in this family the positional part of the attention score between
query `t` and key `s` is

    A_P[t,s] = (R(theta_t) q_0) . (R(theta_s) k_0) = q_0^T R(theta_s - theta_t) k_0

Expanding per frequency block `i` (each a 2D rotation plane, `n_b` of them):

    A_P[t,s] = sum_i a_i cos( omega_i * (S_t - S_s) + phi_i )  =:  kappa( Delta S_ts )

with `S_t = sum_{u<=t} Delta_u` the accumulator, `a_i = |q_0i||k_0i|` and
`phi_i = angle(q_0i, k_0i)` the per-block phase offset.

**`A_P` is a shift-invariant kernel `kappa` evaluated at an accumulator
difference.** `kappa` is real and even iff all `phi_i = 0`; in general it is the
real part of a Fourier series with spectral weights `a_i` at frequencies
`omega_i` and phases `phi_i` (Bochner).

This is exhaustive for the family. There are exactly three design choices:

| axis | what it sets | knobs in this project |
|---|---|---|
| **A. the argument** | what `Delta S` measures | signed vs monotone `Delta`; rank of the action->`Delta` map; index (`Delta == 1`) |
| **B. the kernel** | the shape of `kappa` | phases `phi` (coherence); magnitudes `a` (PoPE); frequencies `omega` (RoPE schedule) |
| **C. the composition** | how `kappa` meets content | product (`A_X (*) A_P`, EM) vs sum (WM) |

## 2. What a task requires: the retrieval offset

Let `s*(t)` be the key query `t` must retrieve. Define the **retrieval offset**

    delta(t) = S_t - S_{s*(t)}

Two properties of `delta` decide everything:

- its **value** -- where `kappa` must be large;
- whether it is **constant across queries** or **query-dependent**.

Two archetypes, and this project has both:

| | map / retrieval (torus, Match-Query, MiniGrid) | clock / k-back (recency) |
|---|---|---|
| `s*` | previous visit to the same cell | the answer `k` steps back |
| `delta` | **0**, for every query | **k(t)**, drawn per query |
| needs (A) | `S` cancels on loops | `S` strictly monotone |
| needs (B) | `kappa` peaked at 0 | `kappa` injective over the range |

## 3. Theorem 1 (axis A) -- cancellation is exclusive, so the crossover is a
contradiction rather than a trade-off

A map requires `sum_loop Delta = 0` for every closed action loop. A clock
requires `S_t - S_s` strictly increasing in `t - s`, hence `> 0` on every
interval, hence `> 0` on a closed loop. These cannot both hold. **No single
accumulator is both a homomorphism onto the displacement group and a counter.**

Measured (`RECENCY_RESULTS.md`): forcing a monotone increment costs **-0.280 on
the torus** (12/12 seeds) and **-0.004 on recency** (inside MDE); interaction
`~ +0.28`. The growth exponent `alpha` (`range(S) ~ T^alpha`) is 0.591 on the
torus and 0.967 on recency in the same unconstrained architecture, with every
constrained arm pinned at `~1.0` as a control.

**The rank result is the same theorem.** At `r=2` the four actions are forced
into a badly conditioned 2-plane: opposition score 0.495 (against 0.092 at
`r=4`), i.e. `N + S != 0`. The accumulator fails to cancel, so `Delta S` is not
the net displacement -- an axis-A defect, partway from map to clock. That is why
`r(opposition, alpha) = +0.9995`: both statistics measure the same thing, how
close `S` is to a homomorphism.

## 4. Theorem 2 (axis B) -- coherence, and why `q_0`/`k_0` is a clock/map choice

Define the **coherence**

    rho := kappa(0) / sum_i a_i = ( sum_i a_i cos phi_i ) / ( sum_i a_i )   in [-1, 1]

- **Single `p_0`** (`q_0 = k_0 = p_0`): every `phi_i = 0`, so `rho = 1` **exactly**,
  and `kappa` attains its unique maximum at `Delta S = 0`. A matched filter for
  "same place", by construction and independent of the draw.
- **Separate `q_0`, `k_0`** drawn independently: `phi_i` iid, so `E[rho] = 0` and

        sd(rho) ~ sqrt( sum_i a_i^2 / 2 ) / sum_i a_i   ~   1 / sqrt(2 n_b)

  Each block peaks at its own `Delta S = -phi_i / omega_i`, so the sum is
  incoherent and `kappa`'s maximum sits at a random nonzero offset.

**Verified at the actual init of the recency batch** (`probe_ap_coherence.py`):
measured `rho` over 16 head-values has mean **-0.059**, sd **0.147**, against the
theory's `E = 0`, sd **0.160**. Single `p_0` returns `rho = 1.000` on every seed.
3 of 8 seeds start with every head at `rho < 0` -- the kernel actively
*down-weights* the same location.

**Corollary, and the point of the whole section: `rho = 1` is not "correct".**
It is correct for `delta = 0` and wrong for `delta != 0`. So the `q_0`/`k_0`
choice is a *second* instance of the clock/map axis, living on the kernel instead
of on the accumulator. It predicts a sign flip between task kinds, which is what
the five measured cells show (`RECENCY_EM_RESULTS.md`): `+0.089 / +0.167 /
+0.358 / collapse-removed` on four map tasks, **`-0.237` on the one clock task**.

The per-offset curve is the mechanism made visible: the coherent EM arm matches WM
at `k <= 2` (where `delta ~ 0` and a zero-peaked kernel is right) and falls to
**0.37 at `k = 64`** (where it is exactly wrong).

## 5. Theorem 3 (axis C) -- the composition, and the constant-offset condition

Write the score as `A_X (*) kappa` (EM) or `A_X + kappa` (WM). Then

    d(score)/d(A_X) = kappa      (product)          d(score)/d(A_X) = 1      (sum)

**In EM the content pathway's gradient is gated by the position kernel**: it
vanishes where `kappa ~ 0` and *inverts* where `kappa < 0`. In WM it is untouched.
This is the exact statement behind "AND-gate vs OR-gate", and it explains why an
identical axis-B defect is a per-seed lottery in EM (1 collapse in 8 on MiniGrid,
worst-to-second-worst gap 0.137) and invisible in WM -- WM has no `q_0`/`k_0` at
all, because it rotates the content-derived `q`, `k` directly and needs no
position-only stream to seed.

**The constant-offset corollary (the new part).** A product forces one fixed
`kappa` to be large at *every* required offset and small elsewhere. If `delta` is
constant across queries this is one constraint and the gate is free -- even
helpful, since it suppresses content matches at the wrong place. If `delta` is
query-dependent, `kappa` must be large on the whole range `{delta(t)}`, which is
to say flat, which is to say uninformative. A sum has no such problem: content
supplies the query-specific part and `kappa` is a bias.

    EM ~ WM   when delta is constant across queries
    EM << WM  when delta varies across queries

Measured: torus / MiniGrid / vocab sweep, `delta == 0` constant -> `EM - WM` =
+0.000 to +0.004 across four cells. Recency, `delta = k(t)` drawn per query ->
**-0.375, 0/8 seeds**. That is the largest EM-vs-WM effect in the project and the
only nonzero one.

## 6. Every result, placed

| result | axis | statement |
|---|---|---|
| signed beats monotone on navigation (+0.123/+0.195), nowhere on recency | A | Thm 1 |
| `alpha` collinear with opposition (`r = +0.9995`) | A | both measure homomorphism failure |
| `r = 2` is a skewed basis, not too small a one | A | cancellation defect, Thm 1 |
| index codes cannot count contextually (+0.750) | A | `Delta S = t - s`, but `delta = k(t)` is not a function of `t - s` |
| `q_0`/`k_0` init lottery, 5 tasks, sign tracks task kind | B | Thm 2 + corollary |
| EM ~ WM on four map cells, `-0.375` on recency | C | Thm 3 corollary |
| EM's 1-in-8 collapse where WM has none | B x C | Thm 2 defect x Thm 3 gating |
| MapPoPE's `r = 4` gain is small (`n_b = 64` vs 32) | B | more blocks, weaker rank bottleneck |

## 7. What it does NOT explain

Stated because a frame that explains everything explains nothing, and because
this project has been burned by exactly that (`ACCUMULATOR.md`: `alpha` covers two
of four).

- **The map-size threshold** (flat at 32 and 128 occupied cells, `+0.305` at 512).
  Nothing in `kappa` has a knee between 128 and 512.
- **Loop / recursion results.** Depth is not in this frame at all.
- **"Helps at OOD length" as the universal signature** -- rank, InEKF, forget gate
  and PoPE all show it and the frame says nothing about why.
- **The InEKF.** It modifies `S` (axis A) yet measurably does not change `alpha`,
  and its wrap acts on the innovation, not on `S` (`ACCUMULATOR.md`, a positive
  control that FAILED).
- **Optimisation.** Everything above is about the function class. The compositional
  recipe finding (`+0.160`, larger than any architecture effect on that task) is
  outside it entirely.

## 8. What it predicts that has not been measured

The only part that can earn the frame anything.

**N1 -- fixed-`k` recency removes EM's handicap.** Thm 3's corollary says EM's
`-0.375` is caused by `delta` varying per query, not by `k` being large. Train on
recency with `k` FIXED at 32 (so `delta` is constant, and large). Predicted:
`EM - WM` returns to within MDE, while an index arm still fails. This isolates
"query-dependent offset" from "large offset" and is the sharpest available test --
one environment flag, 3 arms x 8 seeds.
*Falsified if* EM still loses by more than its MDE at fixed `k`.

**N2 -- coherence variance scales as `1/sqrt(2 n_b)`.** PoPE runs `n_b = 64`
against MapWM's 32, so the separate-form lottery should be **tighter**:
predicted sd(`rho`) 0.113 vs 0.160, hence fewer catastrophic seeds and fewer lucky
ones. Computable on existing MapPoPE checkpoints with no training.
*Falsified if* the measured sd ratio is not near `sqrt(2)`.

**N3 -- `k_0 = q_0` at init recovers the map-task behaviour at separate-form
capacity.** Keep both vectors, initialise them equal (`rho = 1` at step 0, free to
drift). Predicted: matches single-`p_0` on the torus and starts poorly then partly
recovers on recency. This separates "the second vector is harmful" from "the
random relative phase is harmful" -- Thm 2 says only the latter.
*Falsified if* it tracks the random-init separate form on the torus.

**N4 -- TESTED THE SAME DAY, AND NOT ESTABLISHED.** Thm 3 says the content
gradient inverts where `kappa < 0`, so `rho` at init should predict a seed's fate.
Run on both checkpoint sets with `probe_ap_coherence.py`:

| batch | task kind | `r(rho_init, final_loss)` | n | final-loss range |
|---|---|---|---|---|
| `runs/recency_em` | clock | **+0.142** | 8 | 0.430-1.136 |
| `runs/minigrid_em_fix` | map | **+0.029** | 8 | 0.307-0.357 |

Neither is a correlation. Read them differently, though: the recency row is a
genuine null at n=8, while the MiniGrid row has **almost no dynamic range in the
outcome** (every seed lands within 0.05) and is UNMEASURED rather than null. The
falsifier as written fires: **Thm 2's defect is established -- the algebra is
exact and the init measurement matches the predicted sd to within 8% -- but its
route to damage is unidentified.** `rho` at init does not predict which seed
suffers.

An unregistered observation from the same probe, worth more than N4 was:
**`rho` does not converge toward 1 during training.** Mean `rho` drifts by 0.25-0.27
and ends up *more* negative than it started (recency -0.059 -> -0.107; MiniGrid
-0.055 -> -0.227). The optimiser does not align `q_0` with `k_0` on either task.
That is consistent with the chicken-and-egg argument in Sec 5 -- the gradient that
would fix the gate passes through the gate -- but it is now measured rather than
asserted, and it means the separate form's kernel stays incoherent for the whole
of training rather than being a transient handicap.

**N5 -- the sharpest surviving test, replacing N4.** Since `rho` neither predicts
fate nor self-corrects, the causal test is to SET it. Train EM on the torus and on
recency with `k_0` initialised equal to `q_0` (N3) and, separately, with `q_0`,`k_0`
frozen at a chosen `rho` in {-1, 0, +1}. Thm 2's corollary predicts the accuracy
ordering REVERSES between the two tasks as `rho` goes from +1 to -1. An
intervention, not a correlation -- and the recency gate ablation showed that only
a magnitude-matched intervention establishes anything here.
