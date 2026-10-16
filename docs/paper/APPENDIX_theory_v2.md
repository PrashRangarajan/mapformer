# Appendix: theory (v2)

*v2, 2026-10-15, for draft 4 (`docs/paper/DRAFT4_workshop.md`, appendix E). Replaces `APPENDIX_theory.md` (kept unchanged)
for that draft. What changed from v1: a new section B0 (non-negative values vs barrier parameterisations); T1 is presented as
a lemma with its corollaries qualified; the drift / weight-decay account under P2 is removed (ACCOUNTS T1: no drift in the
channels attention reads; weight decay changes nothing detected); P4 gains the trained-model data; Remark R is updated with
ACCOUNTS T2 (the signed collective route, SUPPORTED) and the loose fit on the |.| side; P7 is updated with ACCOUNTS T4 (CoPE
fails a non-wrapping line too); a pointer to the GAUGE test (result: `GAUGE_RESULTS.md`); draft section references updated. Every check pointer of
v1 is kept. Sources: the theory review (`docs/audits/2026-10-14/theory_review/THEORY_REVIEW.md`), the CoPE diagnosis
(`docs/audits/2026-10-14/cope_diagnosis/COPE_DIAGNOSIS.md`), `ACCOUNTS_RESULTS.md`, `MECHANISM_RESULTS.md`,
`docs/paper/figs/FIGS_NOTES.md`, `docs/audits/2026-10-12/ring_dial_audit.md` (M1). Every statement is numbered and labelled **proved**,
**proved (special case)**, **sketch**, **conjecture** or **account**. After each one, a "Checked" line names a script in
`docs/paper/appendix_checks/`; its output sits beside it as `*_out.txt`. All checks are CPU-only and eval-only, and none
trains a model. The checkpoints used are `runs/wrap_mech/p0/{SO,AO}_s800` (MECHANISM, ring) and
`runs/generality/ring/CARoPEF_s70x`. Nothing here is registered, and nothing changes a registered verdict. Statements that
cite registered batches (ACCOUNTS, MECHANISM, WRAP_EXISTENCE) say so; post hoc readouts are marked.*

## Notation and conventions

- **Channel.** A rotary channel is a pair c = (h, b), meaning head h and block b. It acts on query and key coordinates
  (2b, 2b+1) of head h (`model._apply_rope`). Write q_c, k_c in R^2 for those coordinates and
  R(phi) = [[cos phi, -sin phi], [sin phi, cos phi]]. The code rotates Q by R(theta_t) and K by R(theta_s), so channel c
  contributes q_c^T R(theta_s - theta_t) k_c to the score of query t and key s. Every coordinate of a head is rotated, and the
  score is the sum over the head's channels.
- **Step table and angle table.** The phase is theta_{t,c} = omega_c * sum_{u <= t} S_{c, x_u}. Here S = W_out W_in E is the
  C x V step table, with C = H * n_b channels, V tokens and E the embedding matrix; its rank is at most r. The angle table is
  Theta = diag(omega) S. In MapWM the row of W_out for channel (h, b) is h * n_b + b. omega is a free real parameter: nothing
  constrains its sign (`model.PathIntegrator`). Write M_r for the set of C x V tables of rank at most r.
- **Classes.** The *signed* class takes steps S. The *|.|* class (`model_sign`, mode "abs") takes steps |S| entrywise, so its
  angle table is diag(omega) |X| for some X in M_r. The *softplus* class (mode "pos") takes softplus(S). The
  observation-step-free versions (AO, `model_wrap_exist`) give non-zero steps only to the move tokens.
- **Angle sum.** For an inverse pair in one channel, sigma = Theta_{c,x} + Theta_{c,y} (v1 wrote S for this; S is the step
  table).
- **Ring intervals.** Each move token a_t at position 2t is a query that predicts the observation o_t. A revisit key is the
  observation token o_s at position 2s+1 from an earlier visit to the same cell. The tokens in (2s+1, 2t] are g = t - s moves
  (n+ of them +1 and n- of them -1) and g - 1 observations. Write d = n+ - n-. On a ring of N cells, the pair is a same-cell
  pair iff d = 0 mod N.
- For an inverse pair of tokens (x, y), such as the +1 and -1 moves, channel c *cancels through zero* if
  Theta_{c,x} + Theta_{c,y} = 0 with Theta_{c,x} != 0. It *cancels by wrapping* if Theta_{c,x} + Theta_{c,y} is in 2 pi Z \ {0}.
  It is *blind* if both angles are 0 mod 2 pi.

---

## A0. The propagator identity (proved; prior art for the identity itself)

**Proposition A0.** Let alpha(t) be piecewise constant, with alpha = Delta_u on (u-1, u], and let Phi_t = sum_{u <= t} Delta_u.
The homogeneous part of Howard et al.'s integrator, dF/dt = -s alpha(t) F, has the propagator

  F(t) = exp(-s (Phi_t - Phi_{t'})) F(t').

The solution with input is F(t) = e^{-s Phi_t} F(0) + sum_u e^{-s (Phi_t - Phi_u)} Delta_u f_u, exactly, for piecewise-constant
f. Take s = i omega. Channel c of the rotary score is then

  q_c^T R(theta_s - theta_t) k_c = Re( conj(q~_c) k~_c G_c(t, s) ),  G_c(t, s) = exp(-i omega_c (Phi_t - Phi_s)),

with q~ = q_{2b} + i q_{2b+1}. So the position factor of a rotary score is exactly Howard's propagator with an imaginary
spectrum, driven by the rate Delta.

*Proof.* On each interval the ODE is linear with constant coefficient -s Delta_u, so the flow over one token is
exp(-s Delta_u). Flows compose by multiplication, and the exponents add to Phi_t - Phi_{t'}. Variation of constants gives the
inhomogeneous term. For the score, R(theta) acts on q~ as multiplication by e^{i theta}. So
(R(theta_t) q)^T (R(theta_s) k) = Re( conj(q~ e^{i theta_t}) k~ e^{i theta_s} ) = Re( conj(q~) k~ e^{-i (theta_t - theta_s)} ), and
theta_t - theta_s = omega (Phi_t - Phi_s). The zero-order-hold discretisation is the exact flow, not an approximation. []

**What the identification does not carry.**
1. *Input gain.* In Howard's equation alpha also multiplies the input f, so a token with zero rate writes nothing. In attention,
   a zero-step token still writes its key and value at full strength. Softmax normalisation has no counterpart either.
2. *Real s.* Real s gives a decay exp(-s (Phi_t - Phi_s)) in (0, 1]. Its logarithm, -s (Phi_t - Phi_s), is the additive logit
   bias of GRAPE-AP / FoX / ALiBi (`papers/txt/grape.txt:682-686`). In that slot, non-negativity of the rate is what makes the
   bias a forgetting term, and wrapping does not exist. GRAPE-AP is Case II with real s, not a rotary phase.
3. *Mamba-3.* The per-channel angle is Delta_t theta_t[i], with Delta_t = softplus(.) >= 0 and theta_t a signed, data-dependent
   projection (`papers/txt/mamba3.txt:637`). This is the propagator of dF/dt = -i Omega_t F with a content-dependent spectrum
   Omega_t = Delta_t theta_t. Howard's scalar rate times a fixed spectrum is the special case where theta_t is constant. The sign
   therefore sits in theta_t, not in Delta_t. The identity "complex SSM = data-dependent RoPE" is Mamba-3's Proposition 3
   (`papers/txt/mamba3.txt:662`). Ours is only the mapping onto Howard's three rate cases.
4. *Vector rates.* MapWM's step is an r-dimensional latent read per channel, which is the D-dimensional (wave-vector)
   generalisation, not one scalar rate driving a bank.
5. *The sign of s.* s = +i omega and s = -i omega give the same model up to the reflection of Theorem T1. The convention is a
   gauge.

Checked (`a0_propagator.py`): the RK4 propagator of the ODE matches exp(-i omega (Phi_t - Phi_s)) to 3e-14 for signed,
non-negative and constant rates. The `_apply_rope` score matches sum_c Re(conj(q~) k~ G) to 4e-15.

---

## B0. Non-negative values versus barrier parameterisations (proved; elementary)

**Remark B0.** (a) *Non-negativity of the step value is not a constraint on the model.* For any signed step Delta and
omega != 0, Delta' = Delta mod (2 pi / |omega|) lies in [0, 2 pi / |omega|), and
omega Delta' = omega Delta - 2 pi k sign(omega) = omega Delta (mod 2 pi) for either sign of omega. The score reads each
channel's angle only through cos and sin, so the model with Delta' is the same function. If the step is parameterised as
s mod (2 pi / |omega|), its derivative in s is 1 except on a null set, so gradient descent on s is gradient descent on the
signed model. *Rank caveat:* the entrywise remainder of a rank-r step table need not have rank r; it stays in the rank-r
class when at most r tokens step (its non-zero columns then number at most r), which is P1(a) and covers every model in
the paper.
(b) *What differs is the parameterisation.* |s| has a fold at s = 0 (P4); softplus(s) has no fold, but its slope
sigmoid(s) vanishes as the step goes to 0, and it cannot reach 0. In both, a step cannot be carried from one side of zero to
the other by a small parameter change: we call them *barrier parameterisations*. The paper's experiments compare the
signed parameterisation with these two; "non-negative" in the paper means barrier-parameterised.
(c) *Initial steps.* softplus(0) = 0.69, so the softplus class starts with every token at about the same step (an index
clock with rescaled omega), unlike |.| and signed, which start near 0 (`model_sign.py` docstring; the pre-registration calls
this an initialisation confound).

*Proof.* (a) remainder arithmetic and cos/sin periodicity. (b) chain rule. []

Not checked by script: (a) is the construction used, per channel, in P1 and in `model_wrap_exist.wrap_construct`.

---

## T1. The relative-sign gauge (proved; Lemma 1 in the draft)

*Status.* T1 is an elementary identity, J R(phi) J = R(-phi) applied per channel. It is stated as a lemma because its
corollaries say exactly which parameter points the |.| class excludes, and corollary (ii) makes a prediction (tested
in GAUGE, not confirmed; see the end of this section).

**Theorem T1.** Take a MapWM model with any number of layers. Fix a set F of channels and form a second model by two changes:
(a) negate every token's step in each channel c in F; (b) negate output coordinate h * d_head + 2b + 1 of q_proj and k_proj,
both the weight row and the bias entry, in every layer, for each c = (h, b) in F. Change (a) can be made by negating row c of
W_out, which is a reparameterisation of W_out and leaves its rank and the latent unchanged. It can also be made by negating
omega_c. The two models have identical attention scores at every layer, and hence identical logits.

*Proof.* Let J = diag(1, -1). Then J R(phi) J = R(-phi), and J is orthogonal. Change (a) replaces theta_{t,c} by -theta_{t,c}
for every t, because the phase is linear in the steps and in omega_c. Change (b) replaces q_c by J q_c and k_c by J k_c at every
position. This is exact including biases, because negating one output coordinate of an affine map negates its weight row and
its bias entry, and nothing between the projection and the rotation mixes coordinates (`norm1` acts before the projection). The
new channel term is

  (J q_c)^T R(-(theta_s - theta_t)) (J k_c) = q_c^T J R(-(theta_s - theta_t)) J k_c = q_c^T R(theta_s - theta_t) k_c.

Other channels, V, O, FFN, the norms and the output head are untouched. Every layer uses the same cos/sin, so applying (b) in
each layer makes every layer's scores identical. Inputs to the next layer are therefore identical, and so are the logits, by
induction. In MapEM the same argument applies with the reflection on the c-th pair of q0_pos and k0_pos. []

**Remarks.**
- *The gauge group per channel is O(2).* A common rotation U in SO(2) of (q_c, k_c) commutes with R(phi) and leaves the step
  unchanged. A reflection reverses the step. The sign of a channel's step is the determinant of its O(2) gauge element.
- *Floating point.* IEEE negation is exact, and cos(-x) = cos x and sin(-x) = -sin x hold bitwise. The two models therefore
  agree bitwise, not only to rounding.

**Corollary T1(i): sign-coherence characterisation (proved).** Let S be in M_r, with angle table Theta = diag(omega) S. The |.|
class realises Theta exactly *as real numbers* (no wrapping), with the same omega up to sign and the same content weights
(the same latent), if and only if every row of S is *sign-coherent*: S_{c,x} S_{c,y} >= 0 for all tokens x, y.

*Scope note.* This does not contradict P1(a). P1(a) reproduces opposite-sign rows exactly *mod 2 pi* with a *different*
latent (W_in', W_out'), when at most r tokens step. T1(i) is about the same table as real numbers.

When it holds, take D = diag(sign of row c) and a |.| model with W_out unchanged and omega' = D omega. Then
omega'_c |S_{c,x}| = omega_c S_{c,x}. No reflection is needed, because omega carries the channel's sign. If omega were
constrained positive, T1 would absorb D into the q/k reflection instead.

Conversely, every row of a |.| angle table, omega'_c |X_{c,.}|, is sign-coherent. So, modulo T1, the |.| constraint does not
fix the direction of any channel. It removes only the *relative* signs of the tokens that step in a channel: the |.| class is
"in each channel, every stepping token rotates the same way". Two side remarks. The |.| class also contains tables |X| of rank
greater than r, when X has mixed-sign rows; all of these are sign-coherent. On the torus the four moves form two inverse pairs,
and the constraint makes all four rotate the same way in each channel.

**Corollary T1(ii): one stepping token, no sign cost (proved; the prediction is untested).** Suppose a single token x steps, so
S = s e_x^T. Then the signed and |.| classes realise the same angle tables: {omega_c |s_c|} = {omega_c s_c} = R^C. There is also
no cut, because omega_c is a continuous parameter that crosses zero, so any channel's angle moves continuously through 0 in
either class. Parity, written as R(pi x_t) (Mamba-3 Sec. 3.2, `papers/txt/mamba3.txt:576-578`), and Z_m counting by rotations
(Grazzi et al.) are cases of this kind. The prediction is no sign cost on single-stepping-token tasks. It is not tested here.
The |.| reduced loss still has a kink at s = 0 (P4), so the corollary says nothing about learning speed.

*Extension (by T1(i)).* The same holds whenever the task has a solution code whose rows are sign-coherent, for any number of
stepping tokens. Example: a ring walked by moves +1 and +2. The map code a(+2) = 2 a(+1) has both angles of one sign in every
channel, so it lies in the |.| class without wrapping and can be reached from a small same-sign start without crossing zero.
The prediction is no sign cost there, against a cost for +1/-1 (an inverse pair).

*Test (registered, `GAUGE_RESULTS.md`).* Ring 32 cells, observation-step-free SignedAO_r4 / AbsAO_r4, moves +1/-1 vs
+1/+2, n = 8, seeds 1300-1307. +1/-1: |.| 1/8 vs signed 8/8 (Fisher p 0.0014). +1/+2 (primary G1): |.| 2/8 vs signed 5/8,
d -0.064 [-0.200, +0.075]: UNMEASURED; |.| on +1/+2 vs +1/-1: 2/8 vs 1/8. The signed model learns a sign-coherent code on
+1/+2 (0.92 of channels same-sign vs 0.01 on +1/-1), so T1(i)'s representational statement is visible in what signed steps
learn; the learnability prediction is NOT confirmed (the task is harder for both arms), and nothing measured suggests the
|.| cost needs an inverse pair.

**Corollary T1(iii): the excluded codes (proved).** Up to the gauge, the signed tables that the |.| class excludes (without
wrapping) are exactly the tables with a row containing entries of strictly opposite sign. For an inverse pair (x, y), every
through-zero cancelling channel (Theta_{c,x} = -Theta_{c,y} != 0) is such a row. The only unwrapped cancellation the |.| class
allows is a blind channel. Cancellation by wrapping, Theta_{c,x} + Theta_{c,y} in 2 pi Z \ {0}, is allowed: it is
representable whenever P1 applies. So the constraint removes the through-zero branch of the inverse-pair problem, not
cancellation itself. The sign question is about inverse pairs, not about wrapping.

*Functions versus parameter points.* A through-zero code (a, -a) and the wrapped code (a, 2 pi - a) are the same function
(B0(a)). What T1(iii) excludes are parameter points; when P1 applies, every excluded function has a representative in the
|.| class. The cost the paper measures is therefore a cost of reaching that representative from where training starts
(P3), not of representing it. "Exactly the through-zero codes" holds within the set of cancelling codes of an inverse pair;
the full excluded set is every row with strictly opposite-signed entries.

Checked (`t1_gauge.py`).
- SO_s800, with 35 random channels flipped (70 move angles change sign) and reflected: logits are bitwise identical in float32
  and float64. The control, a flip without the reflection, moves the logits by 6.6.
- AO_s800 with the flip carried by omega: identical (control: 22).
- A random 2-layer Signed_r4 in which every token steps and the q/k biases are N(0, 0.5): identical (control: 0.23).
- MapEM with q0_pos and k0_pos reflected: identical.
- (i), constructive: AO_s800 is rewritten as a signed model with steps D|S|. Logits agree to 1e-11 with the reflection, and
  bitwise with the sign in omega instead.
- (iii): SO_s800 has 53/64 opposite-sign channels, and its weights loaded into the |.| class change the logits by up to 35.

---

## P1. Representability by wrapping

**Lemma P1.0 (per channel; proved).** For any target angle A and any omega' != 0 there is a step Delta' >= 0 with
omega' Delta' = A (mod 2 pi). For example, Delta' = remainder(A, 2 pi) / |omega'| with omega' > 0. []

**Proposition P1(a) (at most r stepping tokens; proved).** Let S be a signed table with at most m <= r non-zero columns (the
stepping tokens x_1..x_m). Assume the stepping tokens' embeddings are linearly independent modulo the span of the other
embeddings. This holds generically when V <= d_model (here 19 <= 128). In the AO models it is automatic, because only the
moves' embeddings enter the step path. Then a |.| model with the same embeddings and content weights has an angle table equal
to Theta mod 2 pi, entrywise.

*Proof.* Pick W_in' with W_in' e_{x_j} = u_j, the j-th unit vector of R^r, and W_in' e_y = 0 for every other token. This is
possible by the independence assumption, since m <= r. Set omega' = max(|omega|, 1) per channel and
W_out'[:, j] = remainder(Theta_{., x_j}, 2 pi) / omega' >= 0 (Theta = diag(omega) S), with all other columns zero. Then W_out' W_in' E has
non-negative entries, so |.| leaves it unchanged. It reproduces each stepping column mod 2 pi and leaves every other column at
zero. []

This is the observation-step-free construction of draft 4 Sec. 4.2 (`model_wrap_exist.wrap_construct`, mode `balanced`). On
the ring m = 2 and r = 4. On the torus m = 4 = r, exactly at the limit. P1(a) is sufficient but not necessary: by T1(i), any
table with sign-coherent rows is representable for any m, with no wrapping at all.

*The full (observation-stepping) |.| arms.* Representing the *task* does not require reproducing a given signed table in
which observations step: choose W_in' to annihilate every observation embedding, which P1(a)'s independence assumption
allows when V <= d_model, so that only the moves step. Every |.| arm in the paper can therefore represent the map. The
ring_dial audit built this in the real modules on the ring (`docs/audits/2026-10-12/ring_dial_audit.md` M1;
`wrap_construction_t2.py`): Abs steps >= 0 (min 5e-11), observation-token |Delta| <= 1.7e-6, cos/sin equal to the signed
code's to 6.3e-4 over a 512-step walk, circular mismatch <= 4.6e-4 on 2,000 same-cell pairs. P1(c) and P1(e) below concern
reproducing a *given* table with more than r stepping tokens; they are not needed for the paper's claims.

**P1(b) (the general condition; definition-level).** A signed angle table Theta has an exact |.| rank-r representation mod 2 pi
iff there exist an integer matrix K and a sign matrix Sigma with two properties: every row of Theta' = (Theta mod 2 pi) + 2 pi K
is sign-coherent, and rank(Sigma o Theta') <= r. Here o is the entrywise product, and the rows of Theta' may be rescaled freely
by omega'. Realisability through the embeddings is as in P1(a). This merely restates the definition.

**Proposition P1(c) (generic failure, rank 1; proved).** Let r = 1 and C >= 2. Let the signed table be Theta = alpha v^T with
alpha in R^C, and suppose two tokens step with v_1 > 0 > v_2. Then the set of alpha for which the |.| rank-1 class represents
Theta exactly mod 2 pi has Lebesgue measure zero. The same holds for any m >= 2 stepping tokens whose v has entries of strictly
opposite sign: restrict to such a pair.

*Proof.* A rank-1 |.| angle table is diag(omega') |w z^T| = a b^T, with a = omega' o |w| arbitrary and b = |z| >= 0. So we
need a_c b_j = alpha_c v_j + 2 pi k_{cj} for j = 1, 2, with integers k_{cj}.

If b_1 = 0, then alpha_c v_1 is in 2 pi Z for every c, a null set. Otherwise scale b_1 = 1. Then a_c = alpha_c v_1 + 2 pi k_{c1}.
This is non-zero except on a null set (if a_c = 0, alpha_c v_2 is in 2 pi Z). So b_2 = (alpha_c v_2 + 2 pi k_{c2}) /
(alpha_c v_1 + 2 pi k_{c1}), and this must be the same for c = 1 and c = 2.

Clearing denominators gives, for each integer quadruple (k_1, k_2, l_1, l_2), an equation that is *affine* in
(alpha_1, alpha_2). The alpha_1 alpha_2 terms cancel:

  2 pi alpha_1 (l_1 v_2 - l_2 v_1) + 2 pi alpha_2 (k_2 v_1 - k_1 v_2) + 4 pi^2 (k_2 l_1 - k_1 l_2) = 0.

Suppose it holds identically. Then k_2 v_1 = k_1 v_2 and l_2 v_1 = l_1 v_2, which forces b_2 = v_2 / v_1 < 0 in each channel,
contradicting b_2 >= 0. Otherwise its zero set is a line in the (alpha_1, alpha_2) plane. There are countably many quadruples,
so the admissible set is a countable union of lines, a null set in R^2, and its product with R^{C-2} is null in R^C. []

**Proposition P1(d) (approximation at rank 1, any m; proved; the scale is heuristic).** Let b = (1, b_2, ..., b_m) >= 0 be
linearly independent over Q. By Kronecker's theorem, t -> (t b_j mod 2 pi)_j is dense in the m-torus. So for every channel
there is a_c with |a_c b_j - Theta_{c,j}|_{mod 2 pi} < eps for all j. A rank-1 non-negative code therefore approximates *any*
angle table mod 2 pi to any precision. The cost is the angle size. Heuristically, |a_c| ~ eps^{-(m-1)}: this is the return time
of a linear flow on T^m, measured roughly, not proved.

So "cannot represent" in P1(c) and in the conjecture below means *exactly*. Approximately, representation is never the
obstruction. Its price is weight scale, which is also the rescaling sensitivity of P2.

**Conjecture P1(e) (r >= 2; sketch only).** For r >= 2 and m > r, consider signed rank-r tables that have at least one row with
strictly opposite-sign entries. Among these, those with an exact |.| rank-r representation mod 2 pi form a null set in the rank-r
variety.

Sketch (from the review). For fixed (Sigma, K), the condition is the vanishing of the (r+1)-minors of Sigma o Theta', which is
polynomial in Theta on each cell where Theta mod 2 pi is affine. There are countably many (Sigma, K). The missing step is to show
that no minor condition holds identically on the mixed-sign set. For r = 1 that is the computation above. For r >= 2, |X| can
have rank greater than r, so the r = 1 argument does not transfer. The restriction to mixed-sign rows is needed: the
sign-coherent set and the m <= r set have positive measure and are representable (T1(i), P1(a)).

Checked (`p1_wrap.py`).
- (a) A random rank-4 signed model whose W_in annihilates all but m stepping embeddings (m = 1..4; 39-56 of 64 channels have
  opposite-signed steps) is rebuilt in the |.| class. The built table is >= 0, the angle error mod 2 pi is <= 7e-14 and the
  logit difference is <= 8e-14 (float64).
- (c) Sympy confirms that the compatibility numerator is affine in alpha with the stated coefficients, and that the identically
  vanishing case forces b_2 = v_2 / v_1. A search over integer wraps |k| <= 5 / 20 / 80 never finds an exact solution, but the
  b_2 mismatch shrinks (1e-3 -> 2e-8), which is the approximation of (d).
- (d) Max-over-64-channels error vs allowed wrap count K = 10..10^4: m = 2: 0.17 -> 2e-4 rad. m = 3: 1.15 -> 0.065.
  m = 4: 1.53 -> 0.15. This is roughly the eps^{-(m-1)} scaling, and fitted to nothing.

---

## P2. Rescaling sensitivity (proved; mathematics only)

**Proposition P2.** Fix a channel with move angles a+ and a-. Over an interval with n+ forward and n- backward moves, the raw
phase is phi = a+ n+ + a- n-. Multiply the channel's angles by (1 + eps), whether through omega, through W_out's row, or
through any factor of the step. The phase changes by eps * phi exactly, and hence by eps * phi mod 2 pi.

- Signed cancelling code, a- = -a+: phi = a+ d. The error is eps a+ d. It is zero at every d = 0 revisit for every eps: the set
  {a+ + a- = 0} is a linear subspace, invariant under rescaling.
- Wrapped code, a- = 2 pi - a+: phi = a+ d + 2 pi n-. The error is eps (a+ d + 2 pi n-), which is 2 pi eps n- at d = 0. It grows
  with the number of backward moves, a path length, not with the displacement. The set {a+ + a- = 2 pi} is affine and off the
  origin, so it is not invariant.
- Wrap revisits (d = +-N) need the lattice condition a+ N in 2 pi Z in both codes. That condition is not rescaling-invariant
  in either code. The signed code's error there is eps a+ d = 2 pi k eps.

*Proof.* phi is linear in (a+, a-). Substitute each code. []

**Remark P2.1 (AdamW decoupled decay is a rescaling; proved).** AdamW applies p <- (1 - lr_t wd) p to every parameter
tensor (`train.py:57`), separately from the gradient step. A move angle is the product of four decayed tensors (omega_c, the
row of W_out, W_in, the move embedding), so the decay term alone is a uniform eps < 0 rescaling of every angle, to which P2
applies.

**Status (v2).** v1 offered P2 and P2.1 as an account of a "drift" of the held non-negative ring model off its cancelling code
(87% -> 18% of channels) and of the torus knock-out. Both uses are withdrawn. ACCOUNTS T1 (registered): weighted by how strongly
attention reads each channel, the held model keeps 98% of its cancelling code (0.99 -> 0.98); the unweighted drop was in
channels attention barely reads; and removing weight decay changes nothing detected (D1, -0.003). P2 remains a statement
about sensitivity: the wrapped code is not invariant under rescaling and the signed cancelling code is. Its frozen-model
consequence is measured (WRAP_EXISTENCE secondary, a probe of frozen models, not a basin width): 1% Gaussian noise on omega
leaves the signed solution at 0.998 and the wrapped one at 0.802 (W_out: 0.975 vs 0.815). Nothing here says how training
moves either code.

Checked (`p2_rescale.py`).
- 106,602 same-cell pairs from 400 registered ring walks (0.787 at d = 0). The measured error matches the formula to <= 1e-13
  rad for both codes at eps = 1e-3 and 1e-2. rms(n-) = 26.6, so the predicted rms wrapped-code error at eps = 0.01 is 1.67 rad.
- On AO_bal / SO_bal built from SO_s800 (exact to 7e-7), omega x (1 + eps) in every channel gives the following revisit
  accuracies:

  | eps | wrapped | signed |
  |---|---|---|
  | 1e-3 | 1.000 | 1.000 |
  | 3e-3 | 0.941 | 1.000 |
  | 1e-2 | 0.700 | 1.000 |
  | 3e-2 | 0.485 | 0.866 (the lattice term) |

- v1's integrated-decay figures (wd * sum(lr) = 0.805 ring, 0.080 torus) are still printed by the script; they are no longer
  used.

---

## P3. The cut circle (proved)

Take one channel and an inverse pair (x, y) with angles (a_x, a_y), and hold everything else fixed. The model depends on the
angles only through cos and sin, so the loss is a function on the angle torus (R / 2 pi Z)^2.

**Proposition P3.**
(a) The closed quadrant [0, inf)^2 maps onto the torus, so the signed class and the |.| class attain the same set of loss values
in this channel. The point (a, 2 pi - a) is the point (a, -a).
(b) The non-blind cancelling set is {a_x + a_y in 2 pi Z}, excluding both angles = 0 mod 2 pi. Its part in the quadrant
consists of the segments a_x + a_y = 2 pi k with k >= 1. The through-zero line k = 0 meets the quadrant only at the blind origin.
(c) Take a start in the open quadrant with sigma = a_x + a_y < 2 pi. The infimum Euclidean distance to a non-blind cancelling
point is min(sigma, 2 pi - sigma) / sqrt 2 in the signed class and (2 pi - sigma) / sqrt 2 in the |.| class. They coincide iff
sigma >= pi.
(d) Any continuous path in the |.| class, with omega's sign free, from |sigma(0)| < pi to a non-blind cancelling point passes
through |sigma| = pi. A signed path to the through-zero branch exists that keeps |sigma| <= |sigma(0)|: the straight segment to
the projection onto sigma = 0, perturbed off a_x = a_y if needed.

*Proof.* (a) and (b) are immediate from the quotient map. (c): the distance from a point to the line a_x + a_y = 2 pi k is
|sigma - 2 pi k| / sqrt 2. The signed class may use k = 0 or k = 1. In the |.| class, k = 0 is only the origin, which is blind,
leaving k >= 1, and k = 1 is nearest. With omega < 0 the class also contains the negative quadrant, which is farther. (d): along
a path in the |.| class, |sigma| is continuous and starts below pi. A non-blind cancelling point has |sigma| >= 2 pi. Apply the
intermediate value theorem. []

So the constraint removes no solution. It cuts each channel's circle at zero. From a same-direction start with sigma < pi, the
nearest non-blind cancelling point lies across the cut, and the only branch on the near side is the wrap, which is reached
through sigma = pi. From sigma >= pi the two classes have the same nearest point. P3 is a statement about distances; it does
not say that a channel started past pi flows to the wrap (Remark R).

Checked (`p3_p5_geometry.py`, P3): a brute-force grid (step 0.025) from 60 random starts reproduces both distance formulas to
2e-4.

---

## P4. The sticky wall (proved for the reduced coordinate)

**Proposition P4.** Let s be the pre-|.| step of one token in one channel, as a function of the parameters (W_out row, latent),
and let a = omega |s|. Write L(s) = l(|s|), with l C^1 near 0 and everything else fixed.
(a) For s != 0, dL/ds = sign(s) l'(|s|). Through the chain rule, the gradient with respect to every parameter that moves s is
sign(s) l'(|s|) grad s.
(b) If l'(0) > 0, meaning the loss wants that angle smaller at the wall, then s = 0 is a strict local minimum of L:
L(s) - L(0) = l'(0) |s| + o(|s|). The signed model's loss l, extended to negative arguments, decreases into a < 0 at the same
point. So the stationary points of |.| pinned at the wall are exactly the points where the signed model has a descent direction
through zero.
(c) Gradient descent with step eta, with l' in [mu, M] on [0, rho] and eta M <= rho, gives
|s_{k+1}| <= max(|s_k| - eta mu, eta M). The iterate enters |s| <= eta M in finitely many steps and stays there: a chatter of
amplitude eta M, not a bounce.

*Proof.* (a) is the chain rule for |.|. (b) is a first-order expansion. (c): s_{k+1} = s_k - eta sign(s_k) l'(|s_k|), so
|s_{k+1}| = | |s_k| - eta l'(|s_k|) |. Split into the cases |s_k| >= eta l' and |s_k| < eta l'. []

Under Adam the normalised step makes the chatter amplitude ~lr. This is a remark, not proved. Weight decay (P2.1) adds a
deterministic pull toward 0.

**Trained models (post hoc, eval-only; `docs/paper/figs/fig_mechanism_out.txt`, MECHANISM seeds 800-807).** Share of channels
whose forward-move angle ends within 0.05 rad of 0 (mod 2 pi): |.| from the same-direction (|default|) start (AO_absmag) 0.441;
signed from the same start (SO_absmag) 0.166; |.| from a large start with sums below pi (AO_mid) 0.002; |.| from a start past
pi (AO_high) 0.014. Of AO_absmag's registered 0.04 cancelling share, 0.035 is same-sign channels near the origin, not
cancellation. This is what P4 predicts if the loss pushes those steps toward zero: they settle at the wall. It is a competing
(or additional) account of the |.| failure beside P3's unreachable wrap. *Not established:* whether those channels started
near zero; whether attention reads them (the counts are unweighted, and ACCOUNTS T1 showed unweighted counts can mislead);
and whether l'(0) > 0 at them. The first-order certificate on trained endpoints (channels pinned at s ~ 0 with l' > 0) would
test the last; it is post hoc, its predictions must be written first, and it is not done.

Checked (`p3_p5_geometry.py`, P4): on AO_s800 (float64, revisit cross-entropy), autograd gives dL/ds = sign(s) dL/da exactly
(difference 0). 69 of 128 pre-|.| entries are negative.

---

## P5. The pi-mirror (proved, with the scope stated)

**Lemma P5.0.** On a ring of even size N, every same-cell pair has an even move count g.

*Proof.* d = 0 mod N with N even gives d even, and g = n+ + n- = d + 2 n- = d mod 2. []

**Proposition P5.** Let Pm act on every channel as (a+, a-) -> (pi - a+, pi - a-). This sends sigma -> 2 pi - sigma and delta -> -delta,
where delta = (a+ - a-) / 2.
(a) Pm maps the raw phase of every interval to pi g - phi. So it negates every even-g phase mod 2 pi, and in particular every
same-cell phase, and it sends every odd-g phase to pi - phi.
(b) Consequently, any functional of the same-cell phases through cos is exactly invariant under Pm. This includes the
attractive term V_s(a) = E_q[mean over same-cell keys of cos phi] of the review's small-beta potential, and it needs no
assumption on the walk.
(c) *Model level.* In MapWM (AO), Pm applied to the move angles of every channel, together with the T1 reflection of every
channel, leaves every even-g attention score unchanged and negates every odd-g score. Odd-g pairs are never same-cell pairs.
(d) Pm maps the signed through-zero cancelling line {a- = -a+, 0 < |a+| < pi} onto the wrap segment
{a+ + a- = 2 pi, a+ and a- in (0, 2 pi)}, which lies in the non-negative quadrant. The fixed set of the mirror in sigma is sigma = pi.
(e) The mirror sigma -> 2 pi - sigma *at fixed delta*, i.e. (a+, a-) -> (pi - a-, pi - a+), leaves V_s invariant under either of two
conditions: (1) the law of (g, d) over same-cell pairs is symmetric under d -> -d, which holds in expectation at p_plus = 0.5
by the walk's reflection symmetry; or (2) delta N is in pi Z, including the wrap lattice. It is not a symmetry of the all-keys
term V_a, nor of the transformer's loss.

*Proof.* (a): (pi - a+) n+ + (pi - a-) n- = pi g - phi. For g even, pi g is in 2 pi Z. (b): cos is even, and same-cell pairs have
g even (P5.0). (c): an even-g phase becomes -phi, and with T1's J the channel term returns to its original value. An odd-g phase
becomes pi - phi, and q^T J R(pi - phi) J k = q^T R(phi - pi) k = -q^T R(phi) k. Every channel is transformed, and every
coordinate of a head is a rotary channel, so the whole score is negated. (d): substitution. (e): the phase is sigma g/2 + delta d.
Under the map it becomes pi g - sigma g/2 + delta d = -(sigma g/2 - delta d) mod 2 pi. This equals the original after d -> -d, or when
2 delta d is in 2 pi Z for d in N Z. []

**Scope.** P5 is a symmetry of the same-cell (attractive) part of the objective and of the even-gap scores. It is not a
symmetry of the loss, whose odd-gap competitors are negated, nor of the dynamics. "A start with sigma > pi is on the wrap's side"
means that it lies on the wrap's side of this mirror and is nearer the wrap than the through-zero branch (P3(c)). It does not
mean that the flow from such a start reaches the wrap.

Checked (`p3_p5_geometry.py`, P5; `p5_pm_accuracy.py`).
- Same-cell move counts are all even in 58k revisit queries at p_plus 0.5 and 0.75.
- |V_s(Pm a) - V_s(a)| <= 2e-15 on 20 random points per sample, at both p_plus values.
- The fixed-delta mirror is off by 7e-3 on a plain p_plus 0.5 sample, exact (9e-16) once the sample is closed under walk
  reflection, off by 0.46 at p_plus 0.75, and exact (2e-15) on the lattice.
- The all-keys term changes by 0.06-0.18.
- Model level on SO_s800 (float64): even-g scores are unchanged to 1e-10 and odd-g scores negated to 1e-10, on a score scale of
  121. The control without the reflection moves even-g scores by 45. The image has non-negative angles on 128/128 entries, and
  its opposite-sign channels' sums are 2 pi - sigma in [5.69, 6.67].
- Descriptive: the mirror image, untrained, scores 0.850 revisit accuracy against the source's 1.000. The odd-gap negation
  costs accuracy, as (c) allows.

---

## Remark R. The collective route (account; signed side tested)

The mechanism in the draft is: "a signed model escapes the same-direction start by moving one move's angle through zero,
collectively through the move latents; a |.| model cannot, and reaches the map reliably only from a start past pi". P3-P5
give the geometry. The dynamics are not derived.

*Per channel the flow does not reproduce the run-level pattern.* The review computed the exact one-channel small-beta
potential (the retrieval objective at a fresh model's small attention scale, q.k / sqrt(d) std ~0.3, where softmax is close to
linear in the scores) and ran gradient flow on it (`theory_review/potential.py`, `mech_capture.py`). |.| starts reach the wrap
only from within ~0.5 rad of 2 pi (28% of channels from U(1.7, 4.5)), not from everywhere past pi. Signed starts from the |.|
start cross zero in only 12% of channels.

*The collective route.* In latent space the signed cancelling set {z(-) = -z(+)} is a linear subspace, so a single latent
direction cancels every channel at once. Under |.| that route yields a(-) = a(+), a clock. In the review's toy the route
starts on a plateau (`path_cut_out.txt`).

*Signed side: tested (ACCOUNTS T2, registered, seeds 1000-1007).* From the |.| start, freezing the two move latents (W_out
and omega free per channel) blocks the escape: 1/8 vs 7/8 SOLVED (Fisher p 0.010; -0.075 [-0.130, -0.020]); 15% vs 82% of
channels end opposite-signed. From the default start freezing has no detected effect (7/8 vs 8/8). SUPPORTED.

*|.| side: open, and it fits loosely (post hoc).*
- The |.| (|default|) start has per-channel sums 0.01-6.58 rad (`fig_mechanism_out.txt`); 5% of channels start past pi
  (review `mech_channels_out.txt`). Among the AO channels whose
  forward step does not end collapsed (|wrap(a(+1))| > 0.05; 321 channels over 8 runs), the 20 that start with sums in
  [pi, 3 pi / 2) end with none cancelling, and fewer than 5 start beyond 3 pi / 2 (review `mech_bins_out.txt`). Past pi is
  not sufficient per channel.
- From past pi (AO_high, 8/8 SOLVED) only 23% of channels end cancelling (SO_high 34%; 7% at the start by chance), and the
  final cancel share falls gently with the initial distance from 2 pi (0.34 within 0.25 rad, 0.19-0.25 at 0.5-2 rad, 0.08 at
  2-3 rad; SO_high 0.43-0.52 -> 0.11). The per-channel flow's 28% is close to the observed 23%. So the |.| rescue may be
  per-channel capture of a minority of channels that suffices for the task rather than a collective move; a channel-by-channel
  comparison of predicted and observed capture has not been made. What the other 77% of channels do is not identified.
- From the small start, the wall (P4, trained-model data above) is a competing account of the failure.

So the run-level result (8/8 vs 2/8 vs 0/8) needs enough channels to reach a cancelling code, and the start shifts how many
can. What is proved is the geometry (P3-P5); what is tested is the signed collective route; the |.| dynamics are open.

---

## P6. CARoPE cannot cancel (proved)

CARoPE (`model_mono_baselines.MapFormerWM_CARoPEF`): every token, observations included, advances channel i of head h by
c(x) = f_h(x)^i, with f in (0, 1), so c is in (0, 1]. Channel i = 0 has c = 1 for every token, which is an index clock.

**Proposition P6.**
(a) The phase of any interval containing k >= 1 tokens lies in (0, k]. The shortest return is out and back, g = 2. Between the
move-token query and the observation key it contains 3 tokens (2 moves and 1 observation), so its phase is in (0, 3]. Between
two observation tokens it contains 4 tokens and its phase is in (0, 4]. Either way the phase is in (0, 2 pi), never 0 mod 2 pi.
No channel, at any parameter value, reads every return at phase 0.
(b) *Clock or blind.* Fix eps in (0, pi - 2). Consider the excursion J moves forward and then J back, with nested returns
j = 1..J. If a channel reads each of its returns within eps of 0 mod 2 pi, then J (c+ + c-) <= eps. The move-driven phase of any
interval with at most 2J moves then lies in [0, 2 eps]. Over displacements up to J, the channel separates cells by at most
2 eps of phase from moves. So every channel is either a clock (it misses some short return by more than eps) or blind over that
range.

*Proof.* (a): sum k numbers in (0, 1]. (b): the return at depth j spans 2j moves and 2j - 1 observations. Its phase P_j
satisfies P_1 in (0, 3] and P_{j+1} - P_j = c+ + c- + c(o) + c(o') in (0, 4], where the two observations are at the same cell.
The arc (eps, 2 pi - eps) has length 2 pi - 2 eps > 4 when eps < pi - 2, so a sequence that starts in (0, eps] with increments
of at most 4 cannot jump over it. Since every P_j must stay within eps of 0 mod 2 pi, induction gives P_j in (0, eps] for all j.
Hence J (c+ + c-) <= P_J <= eps, and 2J max(c+, c-) <= 2 eps. []

This replaces the draft's earlier reason, "steps are at most 1 rad", with the review's corrected argument. The (0, 4] in the
review is the observation-observation pair; the move-query pair gives (0, 3]. The conclusion is the same.

Checked (`p6_carope.py`).
- Eight trained ring CARoPEF checkpoints: the g = 2 advance lies in [0.000, 3.000] (channel 0 is exactly 3).
- 0-15 channels per run read every return within 0.1 rad. All of them are blind: their per-token step is <= 1e-3 rad, and their
  phase over *all* query-key pairs, any cells, is <= 0.105 rad.
- 10,331 random (rates, excursion) draws satisfying the premise of (b): the bound held in all.

---

## P7. CoPE on the ring: the lattice family

**Setup (exact for this model; `COPE_DIAGNOSIS.md` Sec. 1).**
- The model is one layer with no other position signal, so each gate is a table G[h, query type, key type].
- Only move queries carry loss.
- The count from the key o_s to the query a_t sums one move gate and one observation gate per step:
  p = n+ alpha + n- beta, with alpha = G[a, +] + cbar and beta = G[a, -] + cbar, when the observation gates are constant
  (tau = 0).
- The position logit is a table z read by linear interpolation between integer knots.

**Lemma P7.0 (proved).** A periodic, non-constant, piecewise-linear function with knots at the integers has an integer period
P >= 2.

*Proof.* Its kink set K is a non-empty subset of Z, since a function with no kinks is affine, and a periodic affine function is
constant. K is invariant under the period, so the period is a difference of integers. Period 1 makes all knot values equal,
which is constant. []

**Proposition P7 (lattice; proved under the setup).** Let z have period P and a unique maximum per period at p = 0 mod P, as the
triangle-type tables do. Suppose the walk produces out-and-back returns (D = 0), wrap returns (D = +-32) and every D with
1 <= |D| <= 31; the ring walks do. Then z(p) is maximal exactly at the returns, the counts with D = 0 mod 32, iff all three of the
following hold:
- (i) alpha + beta = P, and since all four gates lie in (0, 1), alpha + beta < 4, so P is 2 or 3;
- (ii) alpha = k P / 32;
- (iii) k is odd.

*Proof.* Write p = n- (alpha + beta) + D alpha. (i): D = 0, n- = 1 requires alpha + beta in P Z, and 0 < alpha + beta < 4 with
P >= 2 leaves alpha + beta = P in {2, 3}. Then p = D alpha mod P. (ii): D = 32 requires 32 alpha / P in Z. (iii): no non-return
may read as a return. If k = 2k', then D = 16 gives 16 k P / 32 = k' P = 0 mod P, an alias. If k is odd, D k / 32 is in Z iff
32 | D. Every non-return count then sits at a non-zero multiple of P/32 mod P, strictly below the peak. []

The family is: P = 2, alpha - beta = (k - 16) / 8 with k odd, i.e. g+ - g- an odd multiple of 1/8; or P = 3 with k odd. At
initialisation, gates of ~1/2 already give alpha + beta = 2.

**Precision (proved as a sufficient condition).** Returns sit at 0 mod P, and the nearest non-returns sit at +-P/32. For a table
strictly decreasing in the distance to P Z, the readout is therefore unchanged if every count stays within P/64 of its lattice
value:

  N |d sigma| + |D| |d k| P/32 + |sum of observation-gate deviations| < P/64

for all (N, D) in context. Here sigma = (alpha + beta)/2, CoPE's mean count per step (in P7 only; not the angle sum of P3),
and N = g. So |d k| < 1/(2 D_max), not 1/64. The diagnosis's 1/64 is
the |D| = 32 term alone, and with D_max = 97 in T = 128 walks it is not enough. The other terms give
|d sigma| < P/(64 N_max) and tau sqrt(N) < P/64.

**Why a non-wrapping line removes the lattice (proved; the margin caveat is proved too).** On a line the only returns are D = 0.
Condition (i) is unchanged. Conditions (ii)-(iii) become "D alpha / P is not in Z for 1 <= |D| <= D_max". That excludes a
finite set of rationals, so the exact solution set in alpha is open, dense and of full measure, and there is no target point to
hit.

The ring-lattice points are *not* line solutions: they alias D = 32. By Dirichlet's theorem, every alpha has some
1 <= |D| <= D_max with dist(D alpha / P, Z) <= 1/(D_max + 1). So the line's worst-case margin is at most P/(D_max + 1), below
the ring lattice's P/32 once D_max > 31. The line trades a measure-zero target for a smaller margin.

**ACCOUNTS T4 (registered, seeds 1100-1115).** CoPE fails the non-wrapping line too: 1/16 SOLVED (0.914) vs 0/16 on the ring
(0.873); L1 NO EFFECT DETECTED. Signed steps solve the line 8/8 and index RoPE does not (0.772; floor 0.503). So the wrap
(and the lattice it creates) is not CoPE's main obstacle. The precision margin above remains a plausible account, not a
tested one.

Checked (`p7_cope.py`, with the paper's CoPE module from `model_mono_baselines` and constructed gate logits).
- The module's count equals n+ alpha + n- beta exactly.
- Over 11,600 ring revisit queries (1,452 wrap-only) and 10,148 line returns:
  - Lattice points (P = 2: k = 13, 15, 17; P = 3: k = 11, 21) read every ring return (1.000, wrap-only 1.000). On the line the
    same gates read only 0.83-0.98 (the D = 32 aliases).
  - Even k: k = 18 reads 0.289 (wrap-only 0.000). k = 16 (parity only) reads 0.001.
  - Off-lattice k = 17.3 reads ring returns 0.723 (wrap-only 0.000) and line returns 1.000.
  - k = 17 + 0.95/(2 D_max) reads 1.000. k = 17 + 1/64 reads 0.926 (wrap-only 0.679).
  - sigma + 0.95 P/(64 N_max) reads 1.000. sigma + 4 P/(64 N_max) reads 0.369.

What P7 does not cover (from the diagnosis):
- The NONE regime (20/40 runs: no locked count, identity-dependent observation gates).
- The sigma = 0.5 / 0.75 / 1.25 locks.
- Why training stops off the lattice. The barrier on the straight path is measured (11/11 runs) but not derived.
- Non-periodic tables are not excluded by the proposition.

---

## Summary of status

| # | statement | status | check |
|---|---|---|---|
| A0 | rotary position factor = Howard propagator with s = i omega (exact; ZOH discretisation exact) | proved (identity is Mamba-3 Prop. 3) | `a0_propagator.py` |
| B0 | Delta mod (2 pi / \|omega\|) is non-negative and the same model; \|.\| and softplus are barrier parameterisations | proved (elementary) | -- |
| T1 | channel step flip + q/k reflection (rows and biases, every layer; MapEM: q0/k0) leaves all scores unchanged | proved (elementary identity) | `t1_gauge.py` |
| T1(i) | the \|.\| class realises a signed table exactly as real numbers (same latent) iff its rows are sign-coherent; only relative signs are removed | proved | `t1_gauge.py` (d, d2) |
| T1(ii) | one stepping token, or any task with a sign-coherent solution code (e.g. moves +1/+2): no representational sign cost | proved; "no learning cost" is a prediction, NOT confirmed: registered GAUGE G1 UNMEASURED (+1/+2 harder for both arms; \|.\| 2/8 vs signed 5/8) | `GAUGE_RESULTS.md` |
| T1(iii) | excluded codes = opposite-sign rows, which include every through-zero cancelling channel | proved | `t1_gauge.py` (e) |
| P1(a) | <= r stepping tokens: exact non-negative representation mod 2 pi | proved | `p1_wrap.py` |
| P1(c) | r = 1, opposite-signed pair, C >= 2: exact representation fails for a.e. alpha | proved (special case) | `p1_wrap.py` |
| P1(d) | rank-1 non-negative codes approximate any table mod 2 pi (Kronecker) | proved; the eps^-(m-1) scale is a heuristic | `p1_wrap.py` |
| P1(e) | r >= 2, m > r, mixed-sign rows: exact representation generically fails | conjecture (sketch) | -- |
| P2 | rescaling error eps phi_raw: 0 (signed, d = 0) vs 2 pi eps n- (wrapped); AdamW decay = rescaling | proved (mathematics only; the drift account is withdrawn, ACCOUNTS T1) | `p2_rescale.py` |
| P3 | cut circle: same loss on the torus; distances min(sigma, 2pi - sigma)/sqrt2 vs (2pi - sigma)/sqrt2; paths cross sigma = pi | proved | `p3_p5_geometry.py` |
| P4 | sticky wall: attracting kink; GD chatter <= eta M | proved (reduced coordinate); trained-model collapse 44% (post hoc), causal role untested | `p3_p5_geometry.py`, `fig_mechanism_out.txt` |
| P5 | even gaps; Pm negates even-g phases; model-level even-g scores invariant, odd-g negated; sigma-mirror conditions | proved, scope stated | `p3_p5_geometry.py`, `p5_pm_accuracy.py` |
| R | signed escapes the \|.\| start through zero, collectively; \|.\| reaches the map reliably only from past pi | account; signed collective route tested (ACCOUNTS T2, SUPPORTED); \|.\| side open (loose fit) | review's `potential.py`, `mech_capture.py`; `ACCOUNTS_RESULTS.md` |
| P6 | CARoPE: no channel cancels; every channel a clock or blind | proved | `p6_carope.py` |
| P7 | CoPE lattice family, precision, line removes the lattice target (smaller margin) | proved under the stated setup; the line also fails (ACCOUNTS T4), so the lattice is not the main obstacle | `p7_cope.py` |

**Corrections found while writing.**
1. P3: from a start with sigma >= pi, the signed class's nearest cancelling point is also the wrap. The through-zero branch is
   nearer only for sigma < pi. (My first numerical check assumed sigma/sqrt 2 and failed; the statement above is the corrected
   one.)
2. P1: "a generic signed code has no exact rank-r non-negative counterpart" is true only in the exact sense. It is proved for
   r = 1 and conjectural for r >= 2. Approximately, a rank-1 non-negative code represents anything, at a large angle scale.
3. P7: the precision on k is 1/(2 D_max), not 1/64. The ring-lattice gates are not line solutions, and the line's margin is at
   most P/(D_max + 1).
4. P6: the shortest-return bound is (0, 3] for the move-query / observation-key pair. (0, 4] is the observation-observation pair.
5. T1: with omega free in sign, the |.| class already contains both directions of every channel. T1 shows that the direction is
   a gauge even if omega were constrained positive. The |.| class also contains tables |X| of rank greater than r, all of them
   sign-coherent.
6. P2.1: with the exact schedule, wd * sum(lr) is 0.805 on the ring and 0.080 on the torus (review: ~0.7 and ~0.07). An angle is
   a product of four decayed tensors.

**Corrections in v2.**
7. P2: the drift / weight-decay account is withdrawn (ACCOUNTS T1). P2 is kept as mathematics with its frozen-model probe.
8. Remark R: the signed collective route has been tested (ACCOUNTS T2, SUPPORTED); the |.| side is stated as open, with the
   numbers that fit it loosely.
9. P7: the line was tested (ACCOUNTS T4); CoPE fails it too.
10. T1(i): qualified "as real numbers, same latent", which removes an apparent contradiction with P1(a).
11. P1(a): stated to cover the full |.| arms (observation steps annihilated), with the ring_dial M1 construction.
12. P3: the angle sum is written sigma (v1 used S, which is also the step table).

## Reproduce

From `docs/paper/appendix_checks/` (CPU, OMP_NUM_THREADS=4; each step takes about 1-5 min):

```
python3 a0_propagator.py
python3 t1_gauge.py
python3 p1_wrap.py
python3 p2_rescale.py
python3 p3_p5_geometry.py
python3 p5_pm_accuracy.py
python3 p6_carope.py
python3 p7_cope.py
```

`common.py` holds the ring-walk and reflection helpers. The scripts read only the committed `runs/wrap_mech/p0` and
`runs/generality/ring` checkpoints.
