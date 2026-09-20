# Why MapPoPE does not combine the two mechanisms' strengths

Written 2026-09-17 after two pre-registered mechanism accounts were refuted (rank, omega base) and
one diagnostic dissociation landed. Every number below is measured, and the two refutations are
kept rather than buried.

## The two score functions, from the code

**MapWM** rotates the content-derived Q and K by the path angle, so for a query at t and key at s

    a_ts = sum_pairs |q_p| |k_p| cos( omega_p (S_t - S_s) + psi_p(content) )

with `psi_p` a per-pair phase set by content and the amplitude effectively SIGNED (it is the
projection of a rotated q onto k). Content can move the kernel's peak and can cancel a contribution.

**MapPoPE** (`model_pope.py:53-70`) keeps PoPE's polar decomposition -- magnitude from content,
phase from position alone -- and swaps the index phase for the path-integrated one:

    a_ts = (1/sqrt d) sum_c mu_q,tc mu_k,sc cos( omega_c (S_t - S_s) - delta_c ),   mu = softplus(.) >= 0

Every amplitude is NON-NEGATIVE and every phase offset `delta_c` is a fixed learned constant, shared
by all tokens. That is PoPE's whole selling point: position is the sole locator and content cannot
contaminate it. The consequence is that the kernel is a fixed function of `S_t - S_s` alone --
content can scale channels but cannot shift a peak or cancel a term.

## What the accumulator does on music (measured)

On the 512-context checkpoints, over full 2048-token test pieces:

| arm | frequencies per head | alpha (range(S) ~ T^alpha) | range(S) at 512 -> 2048 |
|---|---|---|---|
| MapWM | 16 | **1.005** | 151 -> 599 (4.0x) |
| MapPoPE | 32 | **1.003** | 163 -> 651 (4.0x) |

alpha = 1 means the accumulator is a **CLOCK**: increments do not cancel, so `S` grows linearly with
position and `S_t - S_s` for distant pairs leaves the trained range in proportion to the length.
**Both arms do this identically.** So the accumulator is NOT what separates them -- this is the first
thing the diagnostic rules out, and it kills the obvious story ("MapPoPE's angle runs away").

## The dissociation that identifies the difference (eval-only)

Scale `omega` at evaluation to compress the accumulated phase back toward its trained range -- the
path-integration analogue of linear position interpolation. Test NLL by bucket, 5 seeds:

| arm | scale | 0-512 | 512-1024 | 1024-2048 |
|---|---|---|---|---|
| MapWM | 1.00 | 0.538 | 0.783 | **1.397** |
| MapWM | 0.25 | 2.883 | 3.020 | 2.839 |
| MapPoPE | 1.00 | 0.523 | 1.954 | **4.616** |
| MapPoPE | 0.25 | 2.914 | 3.076 | **2.913** |

Compression **improves MapPoPE out of distribution by 1.70 nats** and **worsens MapWM by 1.44**, and
at scale 0.25 the two are equal (2.91 vs 2.84). Same accumulator, opposite response.

Reading: MapPoPE's far-bucket collapse is substantially an OUT-OF-RANGE `S_t - S_s` problem -- bring
the argument back into range and most of the catastrophe goes away, at the cost of local resolution.
MapWM's robustness is NOT that its argument stays in range (it does not; same alpha, same 4x growth)
but that it has a second, content-carried degree of freedom which compression destroys.

## The account

Path integration and PoPE are robust for CONTRADICTORY reasons.

- PoPE's advantage comes from taking content OUT of the phase: a clean positional kernel that can be
  read without content interference. Its cost is that the kernel is calibrated absolutely in the
  position variable and has no content-side compensation.
- Path integration's robustness comes from content being IN the phase: per-pair phases let attention
  place its peak by content, so a mis-scaled position variable can be absorbed.

Stack them and you keep PoPE's uncompensatable kernel while handing it a position variable that is
itself learned, content-driven and, on data without cancellation, unbounded. The failure is a
conflict of requirements, not an implementation defect.

## Why this RETRODICTS what we measured

(Every row was known when the table was written; the registered predictions are T1-T3 below it.)

| condition | accumulator | MapPoPE result |
|---|---|---|
| Dyck-2, length 32 -> 128 | BOUNDED: opens +, closes -, cancel; depth <= 12 | best arm, 0.927 at the hardest cell |
| Indirect Indexing, fixed 56-token sequences | no extension at all | TIE: 8/8 vs PoPE's 7/8, Fisher p = 1.0 |
| Bach Chorales, in distribution (512-crop runs) | clock, inside its trained range | best at 0.5235 but AT the MDE boundary, and it INVERTS at full context (worse than PoPE, 5/5) |
| Bach Chorales, 2-4x beyond context | clock, 4x out of range | collapse, 4.616 vs MapWM's 1.397 |

The dividing line is not the task or the encoding but whether the accumulator's argument stays in
the range training calibrated. This is the clock/map axis this project already measured
(`project_clock_vs_map` in memory): signed increments that cancel give a bounded MAP, monotone ones
give an unbounded CLOCK. The new consequence is that **PoPE-style decoupling is only safe on the map
side.**

It also retro-explains the inverted omega-base result (`JSB_LENGTH_RESULTS_BASE.md`): a wide
frequency spread (base 32768) adds very slow channels that stay coherent far outside the trained
range and therefore produce confident wrong kernel values, while a narrow spread (base 512) lets the
kernel decay toward zero out of range so attention falls back on content magnitudes. Both rows
improved with the smaller base, which is what that reading says should happen.

## Registered predictions -- T1 and T3 have now run (`T1_RESULTS.md`, `T3_RESULTS.md`)

**Status: the conjunction account has interventional support on BOTH halves, and its trade-off
corollary is withdrawn.** T1 (shrink the accumulator) rescued MapPoPE 5.7x more than MapWM.
T3 (restore the pairwise phase, accumulator untouched, inert twin controlled) removed the collapse
almost entirely: 4.616 -> 0.733 at 2-4x, better than MapWM's 1.397. But T3's predicted COST to
pure indexing did not appear (5/8 vs 5/8 on Indirect Indexing), so the re-entanglement corollary
is withdrawn. (The replacement reading offered there -- "optional freedom, not forced entanglement"
-- was ITSELF refuted by `T3GEN_RESULTS.md` G3: forcing the phase away from zero HELPS monotonically
on Bach, and on Dyck the model keeps 0.825 rad while doing worse than its own inert twin. The model
does not decline the freedom where it is useless.)


- **T1** Bounding the accumulator should rescue MapPoPE on music and should NOT help MapWM much.
  Two ways: wrap `S` into a fixed interval, or drive the increments to cancel (zero-mean penalty).
  If a bounded accumulator does not rescue it, this account is wrong.
- **T2** MapPoPE should collapse on any task whose increments do not cancel, including Dyck-2 if the
  increments are forced monotone -- the `MONOTONE` machinery in this repo can do that.
- **T3, restated after a correction.** MapPoPE's phase IS already content-dependent in one sense --
  `phi_t = omega * S_t` with `S` a cumsum of content-driven increments -- so "the fix is a
  content-dependent phase" as first written was wrong. The two senses must be separated:

  1. **Path phase** (MapPoPE has it): a property of the POSITION. Every query at position t shares
     the same `omega (S_t - S_s)`; it is content-dependent only through history.
  2. **Pairwise phase** (MapWM has it, PoPE and MapPoPE do not): rotating the content vectors Q, K
     makes the cosine argument `omega_p (S_t - S_s) + (angle q_p - angle k_p)`, so the CURRENT query
     and key shift the peak, and the amplitude can be negative. Two queries at the SAME position can
     want different offsets. `pope_delta` is `nn.Parameter(zeros(n_heads, d_head))` -- one constant
     per head and frequency, shared by every token of every sequence.

  Removing (2) is precisely PoPE's what/where decoupling, so the prediction is: make `delta_c`
  depend on the token, `delta_c(x_t)` and `delta_c(x_s)`, which restores (2) while keeping (1). This
  is the same phase freedom already measured on MapEM in this repo (`MAGONLY_RESULTS.md`, +0.146).

  **The prediction has a cost attached, which is what makes it falsifiable**: restoring pairwise
  phase partially re-entangles what and where, the thing PoPE exists to prevent. So it should
  recover length extrapolation on music AND give back part of PoPE's pure-indexing advantage -- the
  Indirect Indexing solve rate is where that would show. If it recovers extrapolation at NO cost to
  indexing, this account is too simple and should be replaced.

## T2 status (2026-09-19): registered and NOT run

T2 -- force monotone increments on Dyck with the repo's `MONOTONE` machinery and check that MapPoPE
then collapses -- is the only registered test that moves alpha WITHIN a task. It has not been run.
Until it is, the clock/map boundary is a BETWEEN-TASK association: Bach, Dyck and the torus differ in
accumulator, but also in dataset, model size (6 layers / 8 heads against 1 / 1) and metric. The
interventions that have run (T1 centring, T3 phase, the decay envelope) each manipulate one factor
within one task and support the account's two halves; none establishes that alpha itself is the axis
across tasks.
