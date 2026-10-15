# Theory review of the workshop draft (2026-10-14)

Reviewer: independent theory agent. Scope: every claim in `docs/paper/DRAFT_workshop.md` (Draft 1), classified as
(A) theoretically justified, (B) consistent with a plausible account that is stated but not proven, (C) purely empirical
with no account, (D) in tension with theory. Nothing here is registered. All computations are CPU, exploratory, in this
directory (scripts + `_out.txt`); none changes a registered verdict. CoPE is being diagnosed separately
(`docs/audits/2026-10-14/cope_diagnosis/`, pending) and is only referenced here.

New computations in this directory (all exploratory; n small; labelled where used):
- `potential.py` -> `potential_out.txt`: the exact one-channel small-beta (linear-response) retrieval potential on the ring
  walk, and gradient flows on it from the MECHANISM starts.
- `toy_probe.py` (imports `toy_wrap.py`, does not edit it) -> `toy_runs/`, `toy_summary_out.txt`: 44 toy runs: (i) weight
  decay and frozen omega on a held wrapped code; (ii) a near-uniform-softmax start for the |.| start.
- `path_cut.py` -> `path_cut_out.txt`: the loss along the collective latent route from the |.| start to the signed code.
- `mech_channels.py`, `mech_bins.py`, `mech_capture.py` -> `_out.txt`: per-channel post hoc readouts of the committed
  MECHANISM checkpoints (`runs/wrap_mech/p0`).
- `sharpness.py` -> `sharpness_out.txt`: sensitivity of a held wrapped vs signed code to a channel rescaling, ring vs torus.

---

## Summary table

| # | claim (draft section) | verdict | one line |
|---|---|---|---|
| 1 | Howard Cases I/II/III = index / monotone / signed rotary phase (Sec. 2, Table 1) | **A for the propagator, with three corrections** | exact for the position factor of the score; GRAPE-AP is misfiled (it is real-s); Mamba-3's sign sits in theta_t not in Delta_t; rotary attention lacks Howard's input gain |
| 2 | Wrapping remark: a non-negative rotary step represents any signed code (Sec. 2, abstract) | **A when scoped, D as written** | true per channel and whenever at most r tokens step; false in general under a rank-r step map with stepping observations |
| 3 | Sign is learnability, not representation (4.2-4.3) | **A for the existence half, C for the learnability half** | existence is a construction (proof); "training does not find it" is empirical and budget-scoped; no theorem says gradient descent fails |
| 4 | Mechanism: the constraint removes the path through zero; a start past pi rescues (4.4) | **B** | the cut-circle picture is exact as topology; the pi midpoint is an exact symmetry of the attractive term; but per-channel gradient flow does NOT reproduce "past pi -> wrap" or "signed crosses zero" -- the account needs collective (multi-channel / latent) dynamics that are not derived |
| 5 | Held (ring) vs perturbed-and-re-found (torus); drift 87% -> 18% (4.3) | **B, with a new causal candidate** | the wrapped code is not invariant under a channel rescaling while the signed one is; AdamW weight decay is such a rescaling; in the toy wd = 0 removes the drift completely (0.99 vs 0.43); sensitivity ~ gap^2 explains torus > ring |
| 6 | One path layer beats 4 index layers on the torus; ~3 suffice on Dyck; short gaps helped by depth (4.1) | **B, and the framing is in tension with a known theorem (D)** | Liu et al. 2023 construct gridworld in 2 layers: index depth CAN represent it, so the factorial result is learnability too; precision ~1/T and an oscillatory MLP decode explain why it is hard; strata consistent, not predicted sharply |
| 7 | Navigation in words (4.1) | **A** for the path model (construction), **C** for the RoPE failure | |
| 8 | Clock side not established; effect one-sided (4.6) | **B** | asymmetry follows from reachability: signed contains the non-negative orthant and can cross zero, the reverse cannot; predicts a delay (small) on the clock side and a block on the map side -- what is observed |
| 9 | Natural text learns a clock, phase ~ token index (4.6) | **A as a genericity statement, C as an explanation of why** | any non-zero mean step gives linear growth; mean-zero (alpha 0.5) is non-generic and needs cancelling supervision |
| 10 | CARoPE cannot represent cancellation; scores below index (4.5) | **A** for the first (proof below; the draft's reason is slightly wrong), **B** for the second (phase leak) | |
| 11 | CoPE can represent ring cancellation, does not learn it; near-wrap start does not transfer (4.5) | **A** for representability (audit construction), **C** for the rest | pending `cope_diagnosis` |
| 12 | Naive construction exact but ill-conditioned; Adam loses it (4.3) | **A** (conditioning) + **B** (Adam) | |
| 13 | Large start is 13-22x more mobile under Adam (4.4) | **A** (direction), **B** (size) | omega's update moves every angle in proportion to the angle |

The bluntest points a reviewer will hit are at the end ("Prioritised gaps").

---

## 1. Howard's three cases in the rotary slot

**Statement in the draft.** dF/dt = alpha(t)(-s F + f(t)); with s = i omega, F_t = exp(i omega int alpha); Cases I/II/III =
index / monotone content / signed content rotary phases.

**What is exact (A).** For the homogeneous part, the propagator of dF/dt = -s alpha(t) F from t' to t is
exp(-s sum_{t'<u<=t} alpha_u). With s real it is a decay (the selective-SSM / forget-gate factor); with s = -i omega it is
the rotation R(omega (Phi_t - Phi_t')), Phi_t = sum_{u<=t} alpha_u -- exactly the relative factor in the rotary score
q^T R(theta_t - theta_t') k. So the position-dependent part of a rotary score is the Howard propagator with imaginary s.
This is a one-line identity, and it is already published as "complex SSM = data-dependent RoPE" (Mamba-3 Prop. 3,
`papers/txt/mamba3.txt:662`). What is ours is only the mapping onto Howard's three rate cases. A reviewer will call it
obvious; the draft already says so. Fine.

**Corrections a reviewer will catch.**
1. **GRAPE-AP is not a rotary phase.** In GRAPE, the content-dependent non-negative "phase" Phi = sum omega_l, omega_l =
   g(x_l) >= 0, is used as an ADDITIVE logit bias b = -theta_h (Phi_t - Phi_j) (`papers/txt/grape.txt:682-686`, the
   path-integral additive GRAPE that contains FoX). That is Case II with REAL s (a monotone penalty), where the wrapping
   remark does not apply and non-negativity is required for a decaying bias. Table 1 must move GRAPE-AP to a real-s
   row, and the "Pos/softplus" arm should be described as "a softplus rate in the rotary slot (GRAPE-AP's rate, not its
   slot)". As written, Table 1 and Sec. 3 ("softplus (GRAPE-AP-like)") conflate the two slots -- the one distinction the
   paper's thesis rests on.
2. **Mamba-3's sign is not in Delta_t.** Mamba-3's per-step angle is Delta_t theta_t[i] with Delta_t = softplus(.) >= 0
   and theta_t a signed data-dependent projection (`mamba3.txt:637`). Howard's alpha is a scalar rate shared by a FIXED
   spectrum s; in Mamba-3 both the rate (Delta_t, non-negative) and the "spectrum" (theta_t, signed) are content-dependent.
   So "alpha_t is the selective-SSM Delta_t" is exact for Mamba-1/2 (real s: Howard's equation including his input gain,
   B_bar = Delta B multiplies the input as alpha multiplies f) and only loose for Mamba-3, where Case III comes from the
   signed theta_t. Say: "alpha_t is Delta_t in real-s selective SSMs; in data-dependent RoPE the per-channel angle step
   Delta_t theta_t[i] plays alpha_t s_i."
3. **The input gain is missing in attention.** Howard's alpha multiplies f too: a token with alpha = 0 writes nothing.
   In rotary attention a zero-step token still writes its key at full strength; softmax normalisation has no Howard
   analogue. The identification is of the propagator only. One sentence.
4. **Vector rates.** MapFormer's step is an r-vector latent (W_in x) read per channel by f_c = omega_c w_c: this is the
   D-dimensional Laplace/Fourier generalisation (Howard & Hasselmo 2020; s -> wavevector), not Howard's scalar alpha.
   CARoPE's channel steps f^i are a different non-linear function of one scalar per channel -- Case II per channel, but
   not "one rate driving a bank". CoPE's gate depends on the query (each query runs its own integrator) and its readout
   is a free table, not exp(i omega p): it is Case II in the integrator only.
5. **"For real s, Cases II and III differ in what they can represent" (Remark)** is true per channel and for any
   non-negative combination of channels (a sum of monotone penalties is monotone). It is false for a population read with
   signed weights: Howard 2014's own push-pull pairs (one integrator driven by speed in +x, another in -x) give position
   by subtraction. In attention the per-head bias is a sum with theta_h >= 0 in GRAPE-AP/FoX, so the statement holds
   there. Scope it: "per channel, and for non-negatively combined channels".

**Prediction it makes / tested.** None beyond the taxonomy; it is a frame. No test needed.

---

## 2. The wrapping remark: exact statement and the rank condition

**Draft:** "a step 2 pi / omega - Delta reproduces -Delta exactly in every channel where it is used, and a non-negative rotary
code can represent any signed one (constructed in Sec. 4.3)."

**Per channel (A).** Rotation angles live in R / 2 pi Z. For any target angle A in R, |M| omega' = (A mod 2 pi) is solvable
with |M| >= 0 for any omega' > 0. So channel by channel, token by token, a non-negative step reproduces any signed angle
mod 2 pi. Exact; trivial.

**Under the step map (the precise representability condition).** The model's per-token angles form the matrix
A = diag(omega) W_out W_in E (channels x tokens), rank <= r. A non-negative model of the same form produces
A' = diag(omega') |W'_out W'_in E|. It reproduces A mod 2 pi iff there exist a sign matrix Sigma (entries +-1) and a
non-negative integer matrix K with

  rank( Sigma o (A mod 2 pi + 2 pi K) ) <= r      (rows may be rescaled freely by omega'; o = entrywise product),

and the embeddings let W'_in send the stepping tokens to that factorisation (generic embeddings with d_model >= V suffice).

- **Proposition 1 (sufficient, exact).** If at most r tokens have non-zero angles in the signed code, the condition holds
  with Sigma = 1, K = 0: the non-zero columns of (A mod 2 pi) are at most r, so any such matrix has rank <= r; zero
  angles stay zero. This is the observation-step-free construction (ring: 2 moves, r = 4; torus: 4 moves, r = 4 -- exactly
  at the limit). The draft's Sec. 4.3 construction is this proposition. **A.**
- **Proposition 2 (generic failure when more than r tokens step; sketch).** If m > r tokens have non-zero angles, the
  wrapped matrix is a perturbation of A by 2 pi times an integer pattern on its negative entries. For fixed (Sigma, K) the
  rank condition is a polynomial condition (vanishing (r+1)-minors) on A; it holds identically only for special sign
  patterns (e.g. A >= 0 already). Over the countably many (Sigma, K) the exceptional set has measure zero in the rank-r
  variety. So a generic signed code with stepping observations (m = 19-21 tokens > r = 4) has NO exact rank-r non-negative
  representation. Status: proof sketch; the "not identically satisfied" step should be checked symbolically for one
  (Sigma, K) per pattern class before it goes in a paper. **B.**
- **What this means for the paper.** "A non-negative rotary code can represent any signed one" is true for an
  unconstrained (full-rank) step table and for observation-step-free models. For the arms with observation steps
  (RING_DIAL Abs_r4, SIGN_MATCHED Abs_r4/Pos_r4) the exact equivalence holds only for those signed codes in which the
  observations' steps are exactly zero (their wrapped value is then zero) -- a small negative observation step wraps to
  ~2 pi, which costs rank. The project already found that trained models use observation steps the rank-4 bottleneck
  cannot also wrap; Proposition 2 is the general form of that finding. **As written the remark is D (overstated); scoped
  to "any signed code in which at most r tokens step, e.g. the observation-step-free models" it is A.** The torus and ring
  "observation-step-free versions show the same" sentence (4.2) is what rescues the learnability reading for the arms with
  observation steps; say so explicitly.
- **Weight scale.** Wrapping a slow channel needs |M| ~ 2 pi / omega'_c; with omega_min = 2 pi / 64 that is up to 64x the
  map step. omega is learnable, so this is a conditioning cost, not a representational one (the min-norm construction's
  16,000x is this; the balanced construction sets omega' = max(|omega|, 1)). Worth one sentence: the representable set is
  the same, the norm needed to reach it is not.

---

## 3. Sign is learnability, not representation

**Existence half (A).** Proposition 1 + the frozen check (angle error <= 1.4e-6 rad, frozen accuracy = source on 8/8) is a
proof that the solution exists in the class, at a stated weight scale. Rule 14 is met. Nothing to add.

**"Training does not find it" (C as a theorem, empirical as a fact).** No result says gradient descent cannot reach the
wrapped code; it is a budget-scoped, recipe-scoped observation (0/8, 3x budget 3/8 vs 3/8). That is fine for a workshop
paper if worded as such; "learnability" here means "not reached in 300-900 epochs of AdamW from standard starts". The
draft mostly says this; the abstract's "does not learn one" should carry "within our budgets".

**Gradient dynamics in the minimal model (what I could derive; `potential.py`).**
One channel, move angles a+ (for +1), a- (for -1). Over an interval with n+ forward and n- backward moves, the phase
difference is phi = a+ n+ + a- n- = sigma g + delta d, with S = a+ + a- = 2 sigma, delta = (a+ - a-)/2, g = n+ + n- (gap),
d = n+ - n- (displacement). Retrieval at inverse temperature beta; at small beta (a fresh transformer: q.k/sqrt(d) std ~0.3)
L = L0 - beta V + O(beta^2) with the exact linear-response potential

  V(a+, a-) = E_q[ mean over same-cell keys of cos phi  -  mean over all keys of cos phi ].

Exact facts:
1. **Every revisit interval has even length.** Same cell on a ring of even size N means d = 0 mod N, so d is even, and
   g = d mod 2. (Checked: all same-cell gaps even, `potential_out.txt` line 1.)
2. **Cancelling sets.** A revisit's phase must vanish mod 2 pi for every revisit gap; at d = 0 this is S g / 2 in 2 pi Z for
   all even g, i.e. S in 2 pi Z. With the lattice condition delta N in 2 pi Z for wrap revisits. Signed: the line
   a- = -a+ through the origin (S = 0). Non-negative: S = 2 pi (and 4 pi, ...), since S = 0 forces a+ = a- = 0 (no map).
3. **The cut circle.** On the angle torus (R / 2 pi Z)^2 the two parameterisations have the SAME loss function: the point
   (a+, 2 pi - a+) IS the point (a+, -a+). The |.| constraint removes nothing from the reachable set mod 2 pi; it cuts each
   circle at 0, so the nearest cancelling point of a small same-direction start (just across a- = 0) is behind a wall and
   the only one on the near side is the far branch S = 2 pi. This is the exact content of "the constraint removes the path
   through zero". **A (topology).**
4. **The wall is sticky, not reflecting.** With a = |s|, dL/ds = sign(s) dL/da. Where the loss wants a smaller (dL/da > 0)
   s is pushed toward 0 from both sides: an attracting kink (under momentum, a chatter of amplitude ~lr). So the |.| model's
   angles collapse toward the wall/origin, they do not bounce back up. Matches the committed checkpoints: AO / AO_absmag
   channels shrink (|a+| + |a-| x0.61 / x0.45 for the few "cancelling" ones, which are near-origin, not wraps;
   `mech_channels_out.txt`). Weight decay (below) adds a deterministic pull toward the origin. **A.**
5. **The midpoint pi is an exact symmetry.** The same-cell part of V satisfies V_s(S, delta) = V_s(2 pi - S, delta) exactly
   (fact 1 makes cos(S g / 2) invariant under S -> 2 pi - S at d = 0; at d = mN use the walk's d -> -d symmetry at
   p_plus = 0.5). Numerically V_s is mirror-symmetric to 3 decimals (`potential_out.txt`). So the attractive term treats the
   through-zero cancel (S = 0) and the wrap (S = 2 pi) as mirror images about S = pi; a start with S > pi is on the wrap's
   side of the mirror. The all-keys term breaks the symmetry only near S = 0, delta = 0 (where every key looks alike).
   **A, as a symmetry.**

What does NOT follow (and I checked):
6. **Per-channel gradient flow does not reproduce the registered pattern.** V along S is rugged (dV/dS changes sign every
   ~0.3 rad: the run-length structure of the walk puts weight on many gap values). Plain and sign-normalised gradient
   ascent on V from 40 channels per start (`potential_out.txt`): |.| from U(1.7, 4.5) reaches the wrap in 28% of channels,
   and only those starting within ~0.5 rad of 2 pi (`mech_capture_out.txt`: capture 1.00 below 0.5 rad, 0.00 beyond);
   signed from the |.| start crosses zero in only 12% of channels; signed from the default start cancels in 47-50%
   (about the share that START opposite-signed). So in the one-channel model, "S past pi flows to the wrap" and "signed
   crosses zero" are both false. **The registered mechanism is not derivable channel by channel.**
7. **The transformer's per-channel capture is broad, not a sharp divide.** In AO_high the final cancel share falls gently
   with the initial distance from 2 pi (0.34 within 0.25 rad, 0.19-0.25 at 0.5-2 rad, 0.08 at 2-3 rad); SO_high the same
   (0.43-0.52 -> 0.11). In AO (default start) the 20 channels that start with S > pi do NOT wrap (0/20). In AO_mid the 5%
   that wrap are not those with the largest initial sums (2.42 vs 2.38); they grew (x2.6) past pi. In SO_absmag the
   initially smaller angle is the one that crosses zero in 64% of single-flip channels (chance 50%), and the through-zero
   cancel share falls with the initial sum (1.00 at S < 0.5 -> 0.65 at S > 2) (`mech_bins_out.txt`). So the run-level
   result (8/8 vs 2/8 vs 0/8) is collective: success needs ENOUGH channels to reach a cancelling code, and the start shifts
   how many can.
8. **The collective route exists for signed and not for |.|, but it starts flat.** In the balanced gauge z(+) = c e1,
   z(-) = c e2, rotating the minus move's latent z(-)(tau) = c((1 - tau) e2 - tau e1) moves a- to -a+ in every channel at
   once (each channel crossing zero at its own tau). Along this one direction the toy's loss falls from 3.32 to 2.91-2.99
   (small beta; accuracy 0.00 -> 0.88 with nothing else trained) and from 28 to 3.5-4.7 (beta 1), with at most a 0.3 bump
   near tau 0.1-0.2; under |.| the same route never improves (`path_cut_out.txt`, 3 seeds). But for tau < 0.5 the
   small-beta loss is flat to 3 decimals: a plateau, not a slope. This is the best account I can give of "signed escapes
   the |.| start": a single latent direction makes every channel cancel at once for signed steps (the cancelling set
   {z(-) = -z(+)} is a linear subspace of latent space), it does not exist under |.| (|W z(-)| = |W z(+)| gives a- = a+, a
   clock), and leaving the |.| start along it needs the model to cross a plateau. Whether training crosses it is then a
   matter of noise and the other parameters -- which is where the toy (0/8, and 0/6 at a near-uniform softmax start) and the
   transformer (7/8) differ. **B.**

**Toy vs transformer (asked).** I tested the obvious candidate -- the toy starts in a saturated softmax (score scale
~44, initial loss 20-28, accuracy 0) while a transformer starts near-uniform. Starting the toy at log beta = -4 (score
scale ~0.8): signed |.|-start 0/6 SOLVED (median 0.52, 52% opposite-sign channels), |.| 0/6, signed default 2/4
(`toy_summary_out.txt`). The softmax temperature is NOT the difference (negative result). Remaining candidates, untested:
(i) the transformer retrieves an observation, and with 16 types and 50% blanks many non-same-cell keys carry the right
answer, which smooths the loss along the plateau; (ii) its content path (Q/K per channel) can mute channels that have not
cancelled, so partial progress pays; the toy's channel weights are tied to a positive softplus. Cheapest test: the toy
with an observation-retrieval target (same-observation keys count as correct), signed |.|-start, 8 seeds, CPU ~10 min.

**What the account predicts, tested or not.**
- P3a (tested, consistent): the |.| angles collapse toward the wall (AO, AO_absmag shrink).
- P3b (tested, consistent at run level only): a start on the wrap's side of the S = pi mirror rescues (8/8); per channel
  the capture is broad and graded, not a divide at pi.
- P3c (untested): the collective route predicts that a signed model whose MOVE LATENTS are frozen (so z(-) cannot rotate
  toward -z(+)), with only W_out trainable, escapes the |.| start much less often. Ring, 8 seeds, ~1 GPU-h.
- P3d (untested, cheap, CPU): in the toy, the per-channel flow fails but the collective route works; adding latent-only
  training (W_out frozen) to the toy should reproduce the transformer's escape if the route is what matters.

---

## 4. Held (ring) vs re-found (torus); drift off the cancelling code (87% -> 18%)

**Exact sensitivity statement (A).** Rescale channel c by (1 + eps): omega_c -> (1 + eps) omega_c (or both of its move steps
by the same factor). Signed cancelling code a- = -a+: the phase at a zero-displacement revisit is a+(n+ - n-) = 0, unchanged
for every eps. Wrapped code a+ + a- = 2 pi: the phase is n (a+ + a-)(1 + eps) = 2 pi n (1 + eps) = 2 pi n eps mod 2 pi, with
n = number of backward moves in the interval. So **the signed cancelling set is invariant under every channel rescaling
(it is a linear subspace through the origin); the wrapped set is an affine set off the origin and is moved by every
rescaling, with a phase error proportional to the path length of the interval.** This is Howard's Case II showing up in
the error: a wrapped non-negative code's error grows with distance travelled, the signed code's with net displacement
(zero at 86-93% of revisits, `sharpness_out.txt`).

Consequences, each checked:
1. **The frozen fragility probe follows.** omega x (1 + 0.01 N(0,1)): error 2 pi 0.01 n ~ 0.6 rad at n = 10 backward moves
   (ring revisit gaps rms 29): destroys long-gap revisits; at 1e-3 the error is <= 0.2 rad. Observed: wrapped 0.802 vs
   signed 0.998 at 1%, both 1.000 at 0.1% (WRAP_EXISTENCE). The W_out probe differs by the angle scale: signed channels
   carry small angles, wrapped ones ~2 pi per pair. **A (derivation matches).**
2. **AdamW's decoupled weight decay is a rescaling, and it is applied to omega and W_out** (`train.py:57`, all parameters,
   wd 0.05). Per step it multiplies them by (1 - lr wd): a pure rescaling of every channel's steps. For the signed code
   this moves along the cancelling set (only the map frequency delta shrinks, which the loss restores); for the wrapped code
   it moves off it (S = 2 pi -> 2 pi (1 - lr wd)^k). Over the ring continuation the integrated decay is
   wd * sum(lr) ~ 0.05 x 1e-3 x 29,400 x ~0.5 ~ 0.7 (an unopposed shrink to ~0.5 per factor); over the torus continuation
   (lr 1e-4) ~0.07. Prediction: the wrapped code drifts off far more on the ring than on the torus continuation.
   Observed: ring 87% -> 18%, torus 100% -> 69%. Consistent.
3. **Causal test in the toy (new, exploratory, n = 4 per arm; `toy_summary_out.txt`).** Start exactly on the wrapped code
   (|.|, 'band'), 3000 AdamW steps: wd 0.05 -> wrap share 0.43 at the end (from 1.00); **wd 0 -> 0.99**; wd 0.05 with omega
   frozen -> 0.54; wd 0 with omega frozen -> 0.99. Signed band: 0.98 (wd 0.05) and 1.00 (wd 0). All 24 runs stay SOLVED.
   So in the toy the whole drift is weight decay, mostly through W_out (freezing omega removes only a little). **This turns
   the drift from "not identified" into a specific, testable cause.** Prediction for the transformer: the AO_bal ring
   continuation with weight decay off (or off for omega, W_out, W_in only) keeps ~87% of channels cancelling and ends with
   T = 512 accuracy equal to the signed copy's. 8 ring runs, ~1-2 GPU-h. **B (strong; toy-level causal evidence).**
4. **Why the torus is knocked out and the ring is not.** The loss curvature along a rescaling at the wrapped code scales
   as E[(2 pi n)^2] over revisits. Measured on the two walks (`sharpness_out.txt`): 1.0e4 (ring, T 128) vs 3.5e5 (torus,
   T 1024) for the most recent visit, ratio 34 (18 over all earlier visits); the signed code's is 0 at the 86-93% of
   revisits with zero displacement. Weighted by the continuation learning rates (ring 1e-3, torus 1e-4) the torus wrapped
   code is still ~3.5x "hotter" than the ring's; the torus lr 1e-3 arm ~35x (observed: ring held; torus lr 1e-4 a
   transient knock-out, peak loss 0.11-0.21; torus lr 1e-3 peak 0.46-0.63 and 6/8 re-solve). The ordering matches; the
   absolute threshold is not derived (Adam's preconditioning changes lr x curvature by an unknown factor). Your suggested
   scaling is right in spirit and sharper than stated: the relevant raw phase of the non-negative code grows linearly in
   the gap (path length) and the signed one as the displacement (sqrt(gap), and exactly 0 at d = 0 revisits); sensitivity
   ratios go as the squares. **B.**
5. **What it predicts that is untested.** (a) Hessian top eigenvalue along log omega at the constructed AO_bal vs SO_bal,
   ring and torus: ratio >> 1, larger on the torus (eval-only, CPU, minutes). (b) Torus AO_bal continuation with wd 0 at
   lr 1e-4: no knock-out (or a smaller one). (c) The from-scratch |.| models: weight decay also pulls small same-direction
   starts toward the origin (the sticky wall), so wd 0 should help the |.| model from scratch a little, not rescue it
   (the wall is there without wd).

**Caveat for the paper.** The draft's parenthesis "(Training also moves the held ring model off the exact cancelling code,
87% -> 18%)" is likely a weight-decay artefact of the recipe, not a property of the class. Do not interpret it until the
wd-0 continuation is run; at minimum, add "under AdamW weight decay 0.05, which rescales the steps".

---

## 5. One path layer vs several index layers (torus) and ~3 on Dyck

**What the theory says (and it is in tension with the draft's framing: D).** Liu et al. 2023 (shortcuts to automata; the
lit note lists "gridworld in 2 layers", `docs/lit/LIT_CLOCK_MAP.md:93`, abstract-level read) construct constant-depth
transformers for solvable/abelian semiautomata, gridworld included. The mechanism is the project's own Theorem 5.2
(`docs/theory/2026-10-04/03_formal.md` sec. 5): layer 1, uniform causal attention over a value +-1 per move, returns
(n_E - n_W)/t and (n_N - n_S)/t; an MLP multiplies by t (available from a second averaging head) and emits cos/sin of
2 pi k x / N; layer 2 matches those Fourier features (exactly what one path layer supplies for free). So **2-3 index
layers can represent the torus map**; 4 layers failing (0/4) is, like the sign result, a learnability/precision result,
not a capacity result. The draft (4.1, abstract: "beats ... four layers") will be read as capacity by a reviewer who knows
Liu et al. State it as "does not learn within 900 epochs, although constant-depth constructions exist (Liu et al.)" --
it actually strengthens the paper's single theme (representable, not learned).

**Why it is hard: the precision and decoding cost (B).** The averaged count has resolution 1/t; recovering x mod 64 needs
the MLP to realise ~T/N = 16 periods of cos(2 pi x / 64) per axis on an input whose consecutive values differ by ~1/t ~ 1e-3
(slopes ~ 2 pi t / N ~ 100). That is a high-frequency function of a near-continuous scalar: the setting where learned
transformers fail (parity/modular counting with uniform attention; Hahn 2020, Bhattamishra et al. 2020 -- check before
citing). On Dyck at L32 D12 the same route needs a counter in [0, 12] at resolution 1/32 and no modulus: a low-frequency,
low-precision decode. **Prediction: the index exchange rate grows with T / N (number of wraps per decode) and with the
range of the prefix sum, not with the task's "complexity".** Dyck (no wrap, T 32) ~3 layers; ring (CANCEL, T 128, N 32)
~3 layers at p_plus 0.5; torus (T 1024, N 64, 2D) > 4. Consistent ordering; not a fitted law. The formal note's
prediction (index-2L shortfall grows with T/N at fixed N; CANCEL T in {64, 128, 256}) is the cheapest decisive test and
has not been run.

**Strata (B, consistent, not sharp).** The prefix-sum route is all-or-nothing: if learned it solves every stratum. The
observed pattern (depth helps only revisits within 128 moves; long-gap and wrap-only stay at 0.37-0.53 at every depth)
says that route is NOT learned and depth buys a local one: relative-offset attention can compose net displacement over a
recent window (RoPE's smooth relative kernels give windows; L layers extend the reach), so error grows with the gap.
This predicts (untested): accuracy vs gap for RoPE_L1/L4/Loop is a decreasing curve whose half-accuracy gap grows with
depth; wrap-only revisits (which close a non-contractible loop and need the modulus) are no better than long-gap ones at
any depth. Eval-only on the committed FACTORIAL checkpoints (fine gap bins), minutes. A one-layer lower bound
(communication-complexity style; the formal note's Conjecture 5.3) is not proved anywhere in the project; do not claim
"1 index layer cannot".

**Navigation in words (A for path, C for RoPE).** Theorem 5.1 (formal note) constructs the path solution for any context-
free step table with direction words carrying +-e_j and all other words 0; a 1-layer path model can therefore solve the
text world exactly. RoPE's 1-layer failure is the same unproven lower bound.

---

## 6. The asymmetry: no established clock-side advantage

**Account (B, derivable in one paragraph; worth putting in the paper).** The signed parameterisation contains the
non-negative one and every path inside it; the converse fails. On a map task the |.| model must reach the cancelling
set, whose near branch is behind the wall (Sec. 3): a BLOCK (only the far wrap is reachable, from far starts). On a
content-clock task (k-back recency) the target is "all counted tokens step the same way in a channel". The |.| model is
there at initialisation in every channel. The signed model starts with random signs: a channel has all K content-token
steps of one sign with probability 2^(1-K), so it must move the minority signs through zero -- reachable, so a DELAY, not
a block. Predictions: (i) clock side: |.| faster early, no difference at convergence; observed: |.| leaves the plateau
earlier in both batches (epoch 172/187 vs 210/213), not lower at epoch 300 in the replication (1.07 vs 0.93), loss-AUC
-0.20 then -0.11 (pooled -0.15, post hoc p 0.018) -- exactly a delay. (ii) The size of the delay should grow with the
number of distinct counted tokens K (more sign flips needed) and vanish at K = 1. Untested; a K-sweep on recency (2 arms x
3 K x 8 seeds) is the decisive test. (iii) The magnitude is not derivable here: a naive Adam estimate (a W_out entry
~0.5 crossing zero at lr 1e-3) gives a few epochs, the observed delay is ~40 epochs, because early gradients are
plateau-limited, not step-limited.

**So the one-sidedness is predicted, not an embarrassment.** "The other direction is not established" should be framed as
the expected asymmetry of a nested parameterisation: block vs delay. That is a cleaner contribution than a double
dissociation would have been.

---

## 7. Natural text learns a clock (growth exponent ~1)

**Account (A as genericity).** theta_t = sum_u Delta(x_u) = t mu + sum_u (Delta(x_u) - mu), mu = the frequency-weighted mean
step. If mu != 0 and the fluctuations are weakly dependent, range(theta) ~ t (alpha = 1) and R^2(theta ~ t) -> 1 as the
clock share mu^2 / E[Delta^2] grows; alpha = 1/2 needs mu = 0 in every used channel, a measure-zero condition the loss must
enforce. A map task enforces it (revisits penalise mu, Sec. 4 above); next-token prediction on narrative text does not
-- it rewards a recency kernel (the index-RoPE-like solution), which needs mu != 0. Observed clock shares 0.79-0.87
(TinyStories), 0.75-0.92 (code), R^2 0.994-0.999: as expected for a dominant mean. So the observation is the generic
outcome, not a finding that needs a mechanism. **C** for the deeper "why does language reward a clock": that is the
standard recency/locality bias of LM attention (ALiBi-style), cited, not derived. Prediction already confirmed: when the
text encodes cancellation (the text world), the model learns a map in words.

**Caveat (C8 in the claim map) stands.** The clock tasks used 6-9-layer models, the map tasks 1 layer; the genericity
argument does not depend on depth, so this confound matters only for "buys nothing over the index", not for "learns a clock".

---

## 8. Published methods

**CARoPE cannot represent cancellation (A; the draft's reason is slightly off).** Faithful CARoPE (`model_mono_baselines.py`)
has channel i advance by f_h(x_t)^i in (0, 1] for EVERY token, observations included, with no omega multiplier.
Proposition 3: per move (move + observation token) channel i advances by sigma_i g + delta_i d with
sigma_i = (f(+)^i + f(-)^i)/2 + (mean observation step) in (0, 2] and |delta_i| <= sigma_i. At a zero-displacement revisit the
phase is sigma_i g with g even, so it is 0 mod 2 pi for all revisit gaps iff 2 sigma_i in 2 pi Z, i.e. sigma_i in pi Z;
sigma_i in (0, 2] < pi rules it out. Hence no channel of CARoPE is a cancelling channel, at any parameter value. Moreover
(approximate version): a channel whose revisit phase stays within eps of 0 for all gaps up to G needs sigma_i <= eps / G
(for sigma_i in [pi/2, 2] the gap-2 phase is already >= 2 pi - 4 = 2.28 rad away), and then |delta_i d| <= sigma_i |d| <= eps,
so it cannot resolve cells either: **every CARoPE channel is either a clock or blind over the context.** The draft's
"(its per-token steps are at most 1 rad)" is the wrong reason: steps of at most 1 rad could still sum to 2 pi over a loop
if they wrapped as a pair; the correct reason is that the shortest loop's total advance lies in (0, 4] < 2 pi
(two moves + two observations) and, for the clock part, sigma_i < pi. Replace with Proposition 3 (two lines).

**CARoPE below index RoPE (-0.055, p 0.04) (B).** CARoPE starts exactly at index RoPE and its observation tokens get content-
dependent steps: an observation-dependent random walk in every channel's phase, variance ~ (#observations in the interval)
x Var(f(o)^i) -- the same gap-accumulated leak as the project's Derivation 4.1 (formal note). The index model has no such
noise. Prediction (eval-only, minutes): replacing CARoPE's per-observation rate by its mean at test time raises its
accuracy to >= the index's, and its deficit vs the index sits in long-gap revisits. Untested.

**CoPE (A for representability; rest C; pending).** The GENERALITY audit's construction (gates 17/16 and 15/16 per
(move, observation) pair, a period-2 table, 1.000 revisit accuracy) is an existence proof. Why it is not learned, and why
the table goes period-2 from any start while the gate asymmetry is not found, is the subject of
`docs/audits/2026-10-14/cope_diagnosis/` (pending); I do not duplicate it. Theory note for that agent: in CoPE the
monotone object is the gated count and the wrap lives in a free per-count table, so the cut-circle argument (Sec. 3) does
not apply to the table; it applies, if at all, to the gate asymmetry, which must be learned through a sigmoid of q.k and
is a precision problem (a 1/16 asymmetry read through mod 2 of a count up to 255).

---

## 9. Smaller claims

- **Naive construction exact but ill-conditioned; Adam loses it within ten epochs (A + B).** Keeping the signed model's
  move latents (cos -0.987..-0.998, nearly antiparallel) and solving W_out for wrapped steps inverts a 2x2 Gram matrix with
  condition number ~ 1/(1 - |cos|) ~ 80-500, so W_out entries blow up (up to 16,000x). Under Adam every parameter moves ~lr
  per step regardless of scale, and the angle error from a W_out step scales with |z(+) - (-z(-))|^-1: tiny relative moves
  of the near-parallel latents produce large angle errors. Derivation of the blow-up: A; "Adam loses it" as a consequence:
  B (consistent, not computed).
- **Large starts are more mobile under Adam (A direction, B size).** Angle a_c = omega_c s_c; an Adam step changes omega_c by
  ~lr, so every angle of the channel moves by a_c lr / omega_c -- proportional to the angle. Starts with |a| ~3 (AO_high)
  vs ~0.5 (default) give ~6x from this term alone; W_out entries are also larger in the balanced gauge (up to 7.6 vs 0.5),
  which changes the W_out term. The measured 13-22x (AO_high) and ~5x (AO_mid) are in this range; the exact factor depends
  on the preconditioner. The mobility confound in M2b is therefore structural, not incidental: any start nearer the wrap is
  a larger-angle start. Only a start that is near the wrap at SMALL angle (impossible: S ~ 2 pi forces large angles) or
  a mobility-matched sub-pi start with omega frozen would separate them. Say this in Limitations; it cannot be fully
  removed by design.
- **Signed models cancel in 64-98% of channels, |.| in 3-11% (C, descriptive).** No account of the exact shares; the
  cut-circle picture predicts |.| shares near zero from small starts (observed) and says nothing about which signed
  channels are used.
- **Design implication "initialise a Case-II rotary step near the wrap" (B, one task).** Supported by M2 and the
  symmetry fact; add the weight-decay corollary: a held wrapped code is eroded by decoupled weight decay on the step
  parameters (toy), so the implication should read "near the wrap, and without weight decay on the step map".

---

## Prioritised gaps (what a reviewer hits first, and what to add)

1. **Table 1 misfiles GRAPE-AP** (real-s additive bias, not a rotary phase) and calls the softplus arm "GRAPE-AP-like"
   without saying it moves GRAPE-AP's rate into the rotary slot. This is the paper's central distinction (real vs imaginary
   s); getting one example wrong invites the reviewer to distrust the frame. Fix: move GRAPE-AP/FoX to a real-s Case II
   row (a one-row extension of Table 1 that also makes the "wrapping is a property of the imaginary slot" point
   visually). Also fix "alpha_t is Delta_t" for Mamba-3 (Sec. 1 above). Cost: text only.
2. **The wrapping remark is overstated.** Add Propositions 1 (exact, with proof) and 2 (scoped, sketch) as a short
   "Representability" paragraph or appendix D: "a rank-r non-negative step reproduces a signed code mod 2 pi whenever at
   most r tokens step; with more stepping tokens it generically cannot". Then say explicitly that the observation-step-free
   arms are what make the learnability reading exact. Cost: text; a symbolic check of Prop. 2's non-degeneracy (CPU).
3. **"One path layer beats four index layers" reads as capacity and contradicts Liu et al.'s constant-depth gridworld
   construction.** Reframe 4.1 as learnability (representable in 2-3 index layers, not learned), cite Liu et al., give
   the precision/decoding account in two sentences. This makes 4.1 an instance of the paper's thesis instead of an
   exception to it. Cost: text. Decisive cheap experiment if wanted: the CANCEL T-sweep (index-2L shortfall vs T/N).
4. **The mechanism (4.4) is presented as an explanation but is only exact as topology and symmetry.** Put in the paper:
   (a) the cut-circle statement (same loss on the angle torus; |.| cuts each circle at 0; the wall is sticky) -- a proof;
   (b) the even-gap symmetry V_s(S) = V_s(2 pi - S), which is what "past the midpoint" means -- a proof; (c) an honest
   sentence that per-channel gradient flow does not reproduce the run-level result (rugged landscape, narrow per-channel
   capture) and that success is collective; (d) the collective latent route (z(-) -> -z(+)) as the account of signed
   escape, labelled as an account. Cost: half a page of appendix; scripts here.
5. **The 87% -> 18% drift is very likely weight decay.** Before the paper interprets it ("the exact map code is not an
   attractor"), run the AO_bal ring continuation with wd 0 (8 runs, ~1-2 GPU-h). The toy says wd 0 keeps 0.99 vs 0.43;
   the integrated-decay estimate (ring ~0.7 vs torus ~0.07) matches ring 18% vs torus 69% remaining. If it holds, the
   draft gains an exact one-line reason (the signed cancelling set is rescaling-invariant, the wrapped one is not) and a
   recipe implication. The exact ω-sensitivity derivation (error 2 pi eps n) should replace the descriptive fragility
   sentence. Note the toy wd result was obtained on 4 seeds per arm; replicate on 8 before citing.
6. **The clock-side asymmetry should be presented as predicted** (block vs delay, Sec. 6 above): one paragraph. The
   K-sweep is the test if a reviewer asks for one.
7. **CARoPE's reason**: replace "(its per-token steps are at most 1 rad)" with Proposition 3. Text only. The "below index"
   result needs either the phase-leak account with its eval-only test or no interpretation.
8. **Toy vs transformer**: the appendix E plan should state that the toy's signed model fails from the |.| start at both
   softmax temperatures (0/8 at beta 1; 0/6 at beta e^-4, exploratory), so the toy is NOT a model of the signed escape;
   it is a model of the |.| failure and of the wrap rescue only.
9. **Learnability claims are budget-scoped**: keep "within 300-900 epochs of AdamW" next to every "does not learn" in the
   abstract; no theorem backs the stronger reading.

**Suggested shape for the paper:** a half-page "Theory" box in Sec. 2 with (i) the propagator identity (cite Mamba-3 Prop. 3),
(ii) Proposition 1 (representability, observation-step-free), (iii) the cut-circle + sticky-wall statement, (iv) the
rescaling (in)variance of the two cancelling sets with the 2 pi eps n error; an appendix with Proposition 2, Proposition 3
(CARoPE), the even-gap symmetry, the block-vs-delay argument, and the Liu-et-al. depth construction with its precision cost.
Everything in that list is either proved above or a two-line derivation; the parts that are only accounts (collective
escape, torus knock-out threshold, strata, clock-side delay size) should be labelled as such.

## Reproduce
From this directory (CPU): `python3 potential.py potential_out.txt` (~20 min with 4 threads); `./run_toys.sh` (44 toy runs,
20 at a time, ~15 min) then `python3 toy_summary.py`; `python3 path_cut.py`; `python3 mech_channels.py`,
`python3 mech_bins.py`, `python3 mech_capture.py` (read `runs/wrap_mech/p0`); `python3 sharpness.py`.
