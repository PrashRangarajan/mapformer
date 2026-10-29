# Sign in rotary position is a matter of learnability: non-negative phase steps can represent a map but, within our budgets, rarely learn one

*Draft 3, 2026-10-15. Replaces draft 2 (`DRAFT_workshop.md`, kept unchanged) after the advisor plan
(`docs/plans/2026-10-14_next_steps.md`), the literature note (`docs/lit/LIT_REPRESENTABILITY_2026-10-14.md`), the theory
appendix (`docs/paper/APPENDIX_theory.md`) and the ACCOUNTS and TORUS_START batches. Every number is from a pre-registered
batch unless marked (post hoc); sources in `docs/paper/DRAFT3_changes.md`.*

## Abstract

Rotary position encodings accumulate a phase, theta_t = omega * sum_{u <= t} Delta_u, whose step can be constant (index
RoPE), non-negative and content-dependent (CARoPE) or signed (MapFormer, Selective RoPE, Mamba-3). In the rotary slot a
non-negative step loses no representational power: angles wrap mod 2 pi, the cyclic-group fact Grazzi et al. and Mamba-3
use to build Z_m and parity from rotations. We ask whether the sign matters for learning maps, where opposite moves must
cancel. It does. At the training length, non-negative steps learn the map on 0-3 of 8 seeds against 8 of 8 for signed
steps, on a 2D torus and a 1D ring. An exact non-negative solution exists at a trained weight scale and training keeps
(ring) or re-finds (torus) it: the failure is one of search. The exact core is a gauge theorem: the absolute value fixes no
channel's direction, only the relative signs of the tokens that step in a channel, and the codes it excludes are exactly
those in which an inverse pair cancels through zero. On the ring, signed steps escape the non-negative model's own start
through zero (7/8) by a collective move of their shared latents (1/8 when frozen); non-negative steps learn only from a start
whose opposite angles sum past pi (8/8 vs 0/8). That start fails on the torus (0/8). Faithful CARoPE and CoPE also fail these
maps. Every negative is scoped to our training budgets (300-900 epochs, one recipe per task).

## 1. Introduction

A rotary position code is an integrator. Index RoPE adds a constant per token; content-dependent variants let each token
set its own step. Some steps are non-negative by construction (CARoPE's rate in (0, 1); in a different slot, CoPE's
sigmoid-gated counts). Others take either sign (MapFormer's path-integrated phase, Selective RoPE, Mamba-3).

For real-valued recurrences the sign matters for what can be represented: transitions with real non-negative eigenvalues
cannot track parity or count modulo m (Sarrof et al. 2024; Grazzi et al. 2025). Those theorems do not reach the rotary
slot: a rotation by a non-negative angle has complex eigenvalues, and Grazzi et al. and Mamba-3 build cyclic groups from
such rotations. In rotary attention the sign of the step carries no representational content per channel. We show that the
usual advice, allow a sign, still holds there for an optimisation reason, and we identify the reason.

**Contributions.**
1. *Sign cost at matched length.* On a torus and a ring, non-negative rotary steps learn the map on 0-3/8 seeds against
   8/8; their codes do not cancel opposite moves; on the ring 3x the budget does not help (Sec. 4.1).
2. *Existence.* An exact non-negative solution built at a trained scale is held (ring) or re-found (torus) under training,
   while the same class from scratch solves 0/8 (Sec. 4.2).
3. *What the constraint removes, exactly* (Theorem 1): only relative signs within a channel; for inverse pairs, exactly the
   codes that cancel through zero (Sec. 2).
4. *The mechanism on the ring*: signed steps leave the non-negative start through zero, collectively through their shared
   latents; non-negative steps learn only from past the wrap. That start lever fails on the torus (Secs. 4.3-4.4).
5. *Published monotone methods* (faithful CARoPE, CoPE) fail these maps too, CoPE although it can represent them (Sec. 4.5).
6. *Depth.* One signed path layer beats index attention at the depths we tried, a learnability measurement (Sec. 4.6).

## 2. Background and theory

**Three integrators.** Howard et al. (2014) drive a bank of integrators by a rate alpha(t): dF/dt = alpha(t) (-s F + f(t)).
A constant rate yields elapsed time, a speed path length (a clock), a signed velocity position (a map). Per token, alpha_t
is the step: the selective-SSM Delta_t, or MapFormer's W_out W_in x_t. *Proposition A0 (appendix E):* with
Phi_t = sum_{u <= t} Delta_u, channel c of a rotary score is Re(conj(q~_c) k~_c exp(-i omega_c (Phi_t - Phi_s))), i.e.
Howard's propagator with imaginary s = i omega. The identity "complex SSM = data-dependent RoPE" is Mamba-3's Prop. 3; ours
is only the mapping onto Howard's cases (Table 1). Rotary attention has no counterpart of Howard's input gain.

| case (Howard) | rate alpha_t | interval sum | rotary family | examples |
|---|---|---|---|---|
| I | constant | elapsed tokens | index RoPE | RoPE |
| II | non-negative, content-dependent | path length (a clock) | monotone content phase | CARoPE; CoPE (a gated count read through a table) |
| III | signed, content-dependent | net displacement (a map) | signed content phase | MapFormer, Selective RoPE, Mamba-3 (sign in theta_t) |
| II, real s | non-negative, decaying | a forgetting bias | additive / decay biases | GRAPE-AP, FoX |

*Table 1. In the real slot non-negativity makes the bias a forgetting term and nothing wraps; in the rotary slot Cases II
and III can represent the same codes.*

**Representability.** Per channel a non-negative step reaches any angle mod 2 pi (-1 = m - 1 in Z_m; Grazzi et al.'s
theta_w = 2 pi w / m, Mamba-3's R(pi x_t)). Our step is a shared rank-r map, so we need a scoped version.
*Proposition P1(a) (proved).* If at most r tokens have non-zero steps, with embeddings independent of the others', a
non-negative rank-r model reproduces the signed angle table exactly mod 2 pi. This covers the observation-step-free models
below (ring: 2 stepping tokens, torus: 4, rank 4). When observations also step, exact representation is not guaranteed
(proved for r = 1, conjectured for r >= 2; appendix E), so the observation-step-free arms carry the learnability claim.

**Theorem 1 (relative-sign gauge; proved).** In a MapWM model with any number of layers, negate every token's step in a set
of channels (equivalently omega_c, or row c of W_out) and negate the second coordinate of each such channel's query and key
projections (weights and biases, every layer). Every attention score is unchanged. *Proof:* with J = diag(1, -1),
J R(phi) J = R(-phi). Checked bitwise on trained checkpoints, a random two-layer model and MapEM.
*Corollaries.* (i) A |.| model realises a signed angle table exactly iff every row is sign-coherent
(S_{c,x} S_{c,y} >= 0). So |.| fixes no channel's direction; it removes only the *relative* signs of the tokens that step in
a channel. (ii) With one stepping token (parity, mod-m counting) the classes coincide and nothing is cut; we predict no sign
cost there (untested). (iii) The excluded codes are exactly the rows with strictly opposite-signed entries. For an inverse
pair (x, y), every channel cancelling through zero (Theta_{c,x} = -Theta_{c,y} != 0) is such a row, while cancellation by
wrapping (Theta_{c,x} + Theta_{c,y} in 2 pi Z \ {0}) stays allowed. The constraint removes the through-zero branch, not
cancellation.

**One channel's geometry (proved; appendix E).** *P3 (cut circle):* both classes attain the same losses on the angle torus;
(a, 2 pi - a) is the point (a, -a). From a start with sum S < pi the nearest cancelling point is at S / sqrt 2 for signed
steps (through zero) and (2 pi - S) / sqrt 2 for |.|; they coincide iff S >= pi, and every |.| path to cancellation crosses
S = pi. *P4 (sticky wall):* with a = |s|, dL/ds = sign(s) dL/da, so where the loss wants an angle smaller, s = 0 is a strict
local minimum; this is the known fact that sign-constrained reparameterisations cannot change sign (Woodworth et al. 2020;
Amid and Warmuth 2020). *P5 (pi-mirror):* on an even ring (a+, a-) -> (pi - a+, pi - a-) leaves the same-cell part of the
retrieval objective exactly invariant and maps the through-zero line onto the wrap segment; it is not a symmetry of the full
loss.

**Remark R (the run-level mechanism is an account).** P3-P5 give geometry, not dynamics. One-channel gradient flow on the
exact small-beta potential reproduces neither "past pi flows to the wrap" (only channels within ~0.5 rad of 2 pi are
captured) nor the signed escape (12% of channels cross zero). The candidate is collective: for signed steps the cancelling
set {z(-1) = -z(+1)} is a linear subspace of the move latents, so one latent move cancels every channel; under |.| the same
move gives a(-1) = a(+1), a clock. Sec. 4.3 tests this and finds it supported.

## 3. Setup

**Model.** MapFormer-WM (Rambaud et al.): one layer, 2 heads, d = 128, query-key rotation by
theta = omega * cumsum(W_out W_in x_t) with one rank-4 latent shared across heads. Sign arms differ only in the step: Signed,
|.| (the sign removed, nothing else) and softplus. Observation-step-free (AO) versions give steps to the move tokens only.
Index RoPE uses the canonical frequency ladder.
**Tasks.** A 2D torus walk (64 x 64, T = 1024 moves interleaved with observations, 900 epochs): predict the observation at
revisited cells, with the observation map redrawn for every evaluation trajectory (accuracy is transfer). A 1D ring (32
cells, T = 128, 300 epochs) with forward-step probability p_plus (0.5: balanced). A line (a 256-cell ring the walk never
wraps). Navigation told in English words.
**Protocol.** Each contrast was pre-registered with its decision rule, run on n = 8 fresh seeds (n = 4 in the depth
factorial, 16 for CoPE on the line), trained and tested at the same length, and code-checked independently before launch.
SOLVED = training loss over the final 5% of training < 0.05. Floors (predict-blank, best action-stream n-gram) are given
beside accuracies. "Unmeasured" means below the registered threshold, not "no effect". "Does not learn" always means within
the stated budget of AdamW with warm-up and cosine decay.

## 4. Results

### 4.1 Non-negative steps do not learn the map

On the torus at the training length (T = 1024), |.| costs -0.177 in accuracy (0.821 vs 0.998; permutation p 0.0002) and
solves 0/8 against 8/8 (softplus -0.218, 1/8). On the ring, |.| solves 2/8 and 1/8 at p_plus 0.5 and 0.75 against 8/8
(Fisher p 0.007, 0.0014); at 0.9 the contrast is unmeasured (4/8 vs 8/8). Three times the training leaves |.| at 3/8. The
ring accuracy gap is small (-0.03; |.| reaches 0.95-0.97, above the 0.554 floor and the one-layer index model's 0.72-0.73),
so the ring evidence is mainly convergence. The learned codes show what is missing (Fig. 1): signed models cancel opposite
moves mod 2 pi in 64-98% of channels (group means), |.| and softplus models in 3-11%, including the runs that solve. The
observation-step-free versions, where an exact non-negative solution provably exists, fail the same way: ring 0/8 vs 8/8
(two batches), torus 0/8 vs 8/8 and 1/8 vs 8/8 (two batches).

The |.| ring models are not maps that just miss a loss threshold (post hoc, Fig. C1): across three batches their revisit
accuracy falls with the gap since the last visit (batch means 0.96-0.99 at gaps 2-7, 0.87-0.96 at 32-127; on 8/8 runs in
each observation-step-free batch, sign-flip p 0.0078). Signed models are at 1.000 in every stratum.

### 4.2 The non-negative solution exists and survives training

We convert each solved signed observation-step-free model exactly into non-negative form (P1(a)): move latents re-solved
orthogonal, each channel's steps set to (omega Delta mod 2 pi) / omega' with omega' = max(|omega|, 1). Every angle is
unchanged mod 2 pi (error <= 1.4e-6 rad) and frozen accuracy equals the source's on 8/8 seeds; on the ring the weights stay
within ~5x of the signed model's. A naive construction keeping the signed latents is also exact but ill-conditioned (they
are nearly antiparallel, cosine -0.987 to -0.998), and Adam loses it within ~10 epochs: the test must be posed at a trained
scale.

On the ring, the constructed model trained 300 more epochs stays solved on 8/8 seeds and never leaves the solved region.
On the torus it is knocked out early (peak loss 0.11-0.21 vs <= 0.017 for the signed control; 6/8 leave the solved region)
and re-finds the solution (8/8 SOLVED, 1.000). (Continuation learning rates differ, 1e-3 ring and 1e-4 torus, so the two
labels are not directly comparable.) From scratch the same classes solve 0/8 on both tasks. A non-negative solution exists
and is stable in task terms; training does not find it. Weighted by how much attention reads each channel, the held ring
model keeps its cancelling code (99% of used channels at construction, 98% at the end).

### 4.3 The mechanism on the ring: the constraint removes the path through zero

We vary only the initial move angles of observation-step-free ring models; every other weight is identical at a seed.

| initial move angles | signed | non-negative |
|---|---|---|
| default (small, signs independent) | 8/8 | 0/8 |
| \|default\|: both moves rotate the same way, small (the \|.\| start) | **7/8** | 0/8 |
| large, the two angles summing to less than pi | -- | 2/8 |
| large, summing to more than pi | 8/8 | **8/8** |

*Table 2. SOLVED, ring p_plus 0.5, 300 epochs, n = 8 per cell.*

Four registered contrasts decide it. (i) From the same start, the sign matters: 0/8 vs 7/8 (Fisher p 0.0014; accuracy
-0.092 [-0.135, -0.045]). (ii) A start past pi rescues the non-negative step: 8/8 vs 0/8 (+0.109 [+0.085, +0.135],
p 0.0002). (iii) It is the sum, not the size: a large start with sums below pi solves 2/8 (Fisher p 0.007 vs 8/8). (iv) From
past pi the sign makes no difference (8/8 vs 8/8). How the models cancel matches the geometry (post hoc, Fig. 2): signed
models from the default or the same-direction start end with 98-99% of channels opposite-signed (from 0% at the
same-direction start) and none of their cancelling channels wrapped; from past pi both signs cancel by wrapping (88% and
100% of cancelling channels).

**The escape is collective (registered; fresh seeds).** From the same-direction start, freezing the two move latents
(W_out and omega stay free per channel) blocks the signed escape: 1/8 vs 7/8 SOLVED (Fisher p 0.010; -0.075
[-0.130, -0.020]); 15% of channels end opposite-signed against 82% with free latents. From the default start freezing costs
nothing (7/8 vs 8/8). The signed model moves one move's latent toward the negative of the other's and cancels its channels
together, the route of Remark R; under |.| that move makes a clock.

So, on the ring: by Theorem 1(iii) the non-negative class keeps only wrapped cancelling codes; from a small start the
nearest lies across S = pi (P3); the signed class reaches the through-zero branch by one latent move. Caveat: a start nearer
the wrap is also a larger-angle start that moves 13-22x faster per Adam step; the sub-pi control (~5x) separates sum from
size only partly.

### 4.4 The start lever does not transfer to the torus

On the torus (observation-step-free, 4 moves, n = 8), the past-pi start does not rescue the non-negative step: 0/8 vs 1/8
from the default start (+0.005 [-0.060, +0.070]; NO EFFECT DETECTED). The same start slows the signed model to 2/8 against
8/8 (secondary), with 5/8 runs per large-start arm still descending at 900 epochs. The default-start sign cost replicates
(1/8 vs 8/8, Fisher p 0.0014). With 64 cells per axis and two inverse pairs sharing every channel, a random large-angle code
is far from any map for either sign. The start manipulation is a ring result; "initialise a monotone rotary step near the
wrap" is not a general rule. The sign cost, existence and the exact geometry carry to the torus.

### 4.5 Published monotone methods

Implemented faithfully, neither learns these maps. CARoPE (a per-head rate in (0, 1), channel i stepping f^i, initialised at
index RoPE) cannot represent cancellation (*P6, proved*: every token advances each channel by (0, 1], so the shortest return
advances it by (0, 4] < 2 pi, and every channel is a clock or blind). It scores 0.667 on the torus, below index RoPE's 0.722
(p 0.040), and 0.741 vs 0.730 on the ring (unmeasured). CoPE can represent ring cancellation (a periodic position table read
at a gated count; constructed) but solves 0/8 (0.887 vs signed 1.000, p 0.0002), while its gated count beats the index by
+0.157. Starting its table near the wrap does not rescue it measurably (unmeasured; bimodal seeds). On the line CoPE still
fails (1/16 vs 0/16 on the ring; no effect detected) while signed steps solve it 8/8 and the index does not (0.772): the wrap
is not CoPE's obstacle. CoPE's failure is learnability with an unidentified obstacle; a plausible account is the precision
its count needs (appendix H).

### 4.6 Where a map is needed, one path layer beats index attention at the depths we tried

On the torus at T = 1024, one signed path layer scores 1.000 (4/4 SOLVED) against index RoPE at one layer 0.719, four layers
0.860 and a block looped four times 0.864 (0/4 each, all stalled within 900 epochs; complete separations, p 0.029; floors
0.507 / 0.576). Depth and looping help only revisits within 128 moves (0.74-0.79 -> 0.90-0.96); long-gap and wrap-only
revisits stay at 0.37-0.53 (Fig. 3). Told in English words: path 0.969 vs index 0.505 (p 0.0002, 7/8 vs 0/8).
This is not a capacity result. The torus walk is the group Z_64 x Z_64, and Liu et al. (2023) show that one attention layer
with an MLP can simulate such a group (a prefix sum and a learned modulus; their Lemmas 6 and 8); a second layer could
retrieve the observation. Their construction uses an absolute position embedding, an MLP whose width grows with nT and
weights growing with T, so it does not show that our 128-wide RoPE models can represent the task, and we do not claim they
cannot. We measure what training finds. Liu et al.'s experiments point the same way: cyclic-group shortcuts are learned
unreliably even at T = 100 with every state labelled (median of 20 runs below 90% for C_4 at 1-15 layers), and less reliably
with sparse labels; ours are sparse and indirect. On Dyck-2 at matched depth one path layer is worth about three index
layers (appendix F).

### 4.7 The other direction is not established

A non-negative step might learn a content clock faster than a signed one. On k-back recency it trends that way and is
unmeasured twice: loss AUC -0.197 (paired p 0.053, n = 16), then -0.106 on fresh seeds (p 0.19, n = 16). We do not claim a
double dissociation. A consistent but untested account (appendix J): signed steps contain every non-negative path, so a
clock costs them a delay (moving minority signs through zero), while a map costs |.| a block.

## 5. Related work

**Integrators and position.** Howard et al. (2014) drive leaky integrators by a constant, speed or signed-velocity rate,
giving time, path length or position; their work and the Howard and Tiganj groups' models keep the decay real and the rate
given. GridPE derives RoPE from a velocity-controlled oscillator. MapFormer (Rambaud et al.) and TEM introduced learned
path-integrated position for cognitive maps.

**Content-dependent rotary position.** Signed: Selective RoPE (Movahedi et al.), Mamba-3 (Lahoti et al.; complex SSM =
data-dependent RoPE, their Prop. 3), MapFormer. Non-negative: CARoPE and, as a gated count read through a table, CoPE
(Golovneva et al.). GRAPE (Zhang et al.) unifies multiplicative and additive slots; its non-negative content phase is an
additive bias in the real slot, as in FoX.

**Sign and state tracking.** Sarrof et al. (2024) and Grazzi et al. (2025) prove that recurrences with real non-negative
eigenvalues cannot track parity or count modulo m (Khavari et al. 2025: the sign must be input-dependent in the same layer).
Rotations by non-negative angles have complex eigenvalues and are not covered: Grazzi et al. implement Z_m with rotations by
2 pi w / m >= 0, and Mamba-3 writes parity as R(pi x_t). We give the same advice, allow a sign, for a different reason.

**Sign-constrained parameterisations and initialisation.** A reparameterised weight cannot change sign under gradient flow
(Woodworth et al. 2020; Amid and Warmuth 2020, where w = |u| is gradient descent on the non-negative orthant). The LRU
(Orvieto et al. 2023) parameterises phases as positive and must initialise them near zero to learn PathX. With inverse pairs
sharing a channel the constraint removes only relative signs (Theorem 1); on the ring the useful start was past the wrap, and
on the torus no start we tried helped. To our knowledge (one session of search and a 40-paper corpus) neither the
relative-sign gauge nor the loss of the through-zero branch for inverse pairs has been stated; we claim their conjunction
and the measurements, not any single ingredient.

**Depth versus recurrence.** Liu et al. (2023) show that shallow transformers can represent solvable automata (abelian
groups in one layer with absolute positions, width and weights growing with T, O(log T) precision) and that such shortcuts
are learned unstably, worse under sparse supervision. Hahn and Rofin (2024) tie this to sharp minima for sensitive
functions. Selective RoPE and PaTH (Yang et al.) solve parity or Flip-Flop in one layer with content-dependent position.

## 6. Limitations

"Does not learn" is scoped to our budgets and one recipe per task (AdamW, warm-up and cosine; 300 epochs ring, 900 torus);
3x the ring budget did not help, and recipe robustness (no weight decay, other learning rates) is untested. Small models
(one layer, 2 heads, d = 128), synthetic tasks, n = 8 (n = 4 in the depth factorial), no recurrent baselines. The run-level
mechanism is an account on proved geometry with one tested component, shown on the ring only. On the torus the large start
also slows the signed model, and 5/8 runs per large-start arm were still descending; any start nearer the wrap is also a
more mobile one. The ring accuracy gap is small. CoPE was tested on the ring and line only (~8 h per torus run). The clock
side is unestablished; natural text is not studied here.

## Figures and tables

- Table 1 (Sec. 2): three integrators. Table 2 (Sec. 4.3): start x constraint.
- **Fig. 1** (`docs/paper/figs/fig_wrap_share.png`; Sec. 4.1). Per run, the share of rotary channels in which opposite moves
  cancel mod 2 pi (within 0.1 rad; wrap-invariant), ring p_plus 0.5 / 0.75 / 0.9 and torus; filled = SOLVED. Signed means
  0.79 / 0.78 / 0.64 / 0.98; |.| 0.06 / 0.03 / 0.06 / 0.11; softplus (torus) 0.03. Post hoc readout of registered checkpoints.
- **Fig. 2** (`docs/paper/figs/fig_mechanism.png`; Sec. 4.3). Move angles (a(+1), a(-1)) of every rotary channel at the
  start (grey) and after training (dots), 8 ring runs per panel. Cancelling lines: blue a(-1) = -a(+1) (through zero; needs
  opposite signs), red a(+1) + a(-1) = +-2 pi (wrapping). Shading: outside the non-negative quadrant; dotted: sum = pi.
  Panels: signed, same-direction start (7/8 SOLVED; cancelling share 0.06 -> 0.89, all through zero); |.|, same start (0/8;
  0.04); |.|, large sub-pi start (2/8; 0.05, by wrapping); |.|, start past pi (8/8; 0.07 -> 0.23, by wrapping). Whole turns
  removed toward zero (post hoc).
- **Fig. 3** (`docs/paper/figs/fig_factorial_strata.png`; Sec. 4.6). Torus T = 1024, accuracy by revisit stratum (within
  128 moves, >= 128 moves, wrap-only) for path 1 layer and index 1 layer, 4 layers, looped x4 (n = 4; means, range, floor).
- **Fig. C1** (appendix; `docs/paper/figs/fig_ring_strata.png`). Ring revisit accuracy by gap and wrap-only stratum (post hoc).

## Appendix plan

- **A. Models and recipes.** Architectures; step parameterisations (Signed, |.|, softplus, AO); learning rates, schedules,
  epochs, seeds per batch; `--fast-attn`.
- **B. Floors and gates.** Predict-blank and best action-stream n-gram floors per task.
- **C. Per-seed tables** for every registered contrast; Fig. C1 and the ring strata table (post hoc; the gap-matched
  wrap-only deficit varies between batches: -0.063, -0.007, -0.032).
- **D. Constructions.** The balanced conversion (P1(a)); why the naive one is exact but ill-conditioned; continuation protocol
  and LOST rule; ring vs torus continuation learning rates.
- **E. Theory** (`docs/paper/APPENDIX_theory.md`; checks in `docs/paper/appendix_checks/`): A0; Theorem 1 and corollaries with
  bitwise checks; P1(a), P1(c) proved, P1(d) approximation at large angle scale, P1(e) conjecture; P2 rescaling sensitivity as
  mathematics only (the signed cancelling code is invariant under a channel rescaling, the wrapped one errs by 2 pi eps n-;
  frozen 1% omega noise: wrapped 0.80, signed 0.998), not as an explanation of drift (there is none in used channels); P3,
  P4 (with Woodworth et al., Amid and Warmuth), P5 with its scope; Remark R with the per-channel flow and the frozen-latent
  test.
- **F. Dyck-2 at matched depth.** +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers.
- **G. Navigation in words.** Text world, floors, step table; dropout-scale re-score (path 0.969 -> 0.993).
- **H. Faithful CARoPE and CoPE.** Implementations; P6; the CoPE ring construction; P7 (gate lattice, proved under its
  setup); the CoPE diagnosis (post hoc: three regimes; gate snap 0.63 -> 0.90 on wrap-only revisits), read with the line
  result (the lattice is not the main obstacle); precision margins; the near-wrap table start.
- **I. The toy** (three lines, exploratory): reproduces the |.| failure without a transformer, fails the signed default and
  shows no signed escape.
- **J. The clock side.** Both recency batches; the block-vs-delay account, untested.
