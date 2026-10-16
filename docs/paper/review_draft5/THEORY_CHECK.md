# Theory check: draft 5 (Secs. 2-3) and appendix E (`APPENDIX_theory_v2.md`)

Referee pass, 2026-10-16. Scope: every lemma / proposition / corollary / remark in `APPENDIX_theory_v2.md`, their use in
`DRAFT5_workshop.md`, and the code they describe (`model.py`, `model_sign.py`, `model_wrap_exist.py`,
`model_mono_baselines.py`). The draft and appendices were not edited. Verdicts: CORRECT / CORRECT WITH FIX / GAP / WRONG.

## 0. Summary

| item | verdict | one line |
|---|---|---|
| A0 homogeneous propagator; score identity | CORRECT | checked (3e-14, 4e-15) |
| A0 inhomogeneous formula ("exactly, for piecewise-constant f") | **WRONG** | the input term is (1 - e^{-s Delta_u})/s f_u, not Delta_u f_u; counterexample below |
| A0 notes 2-5 | CORRECT WITH FIX | "in (0, 1]" needs a non-negative rate |
| B0 (a)-(c) | CORRECT WITH FIX | "covers every model in the paper" overclaims; (b)'s "one side of zero to the other" is vacuous for a value that is always >= 0 |
| T1 / Lemma 1 (identity) | CORRECT WITH FIX | for the \|.\| class the flip can only be carried by omega, not W_out; bitwise claim is implementation-dependent |
| T1(i) | CORRECT WITH FIX | the iff fails at omega_c = 0; state it on Theta or assume omega != 0 |
| T1(ii) | CORRECT | representational only, correctly labelled |
| Lemma 1(ii) in the main text | **WRONG as worded** | "excludes no map code" is false for +1/+2 (code (1, 2 - 2 pi) is excluded); "function" is meant |
| T1(iii) | CORRECT | |
| P1.0, P1(a), P1(b), P1(c), P1(d) | CORRECT | P1(c) algebra re-derived; checks reproduce |
| P1(e) | CORRECT (conjecture, correctly labelled) | a dimension count supports it |
| Softplus representability (main Sec. 3, App. D4, Sec. 6) | **GAP** | it is provable and exact: annihilate the observations and set omega' = 2 pi / ln 2; built and checked (logits to 3e-11) |
| P2, P2.1 | CORRECT | |
| P3 | CORRECT | main text needs "in angle coordinates" and \|sigma\| |
| P4 (appendix) | CORRECT WITH FIX | (c) needs mu > 0 and \|s_0\| <= rho; GD on s, not on the parameters |
| P4 (main text) | **WRONG** | "with a = omega \|s\|, dL/ds = sign(s) dL/da" drops the factor omega; "strict local minimum" needs "along s" |
| P5.0, P5 | CORRECT WITH FIX | scope: moves +-1 and AO in (a); "onto" misses (pi, pi) |
| Remark R / main "collective route" | CORRECT WITH FIX | {z(-1) = -z(+1)} is the all-channel cancelling set only if W_out has full column rank |
| Main Sec. 4.3 "an unreachable wrap, P3" | CORRECT WITH FIX | P3 says reachable only through sigma = pi, not unreachable |
| P6 | CORRECT | matches `CARoPEF` exactly |
| P7.0, P7 lattice, line | CORRECT | |
| P7 precision ("proved as a sufficient condition") | GAP (small) | the tau sqrt(N) term is a heuristic, not a bound |
| App. H "move-gate difference an odd multiple of P/32" | **WRONG** | it is P/16 (alpha is the odd multiple of P/32) |
| Main Sec. 1 eigenvalue sentence | CORRECT WITH FIX | rotation by pi has the real eigenvalue -1 |

Check scripts: all 8 re-run from a copy in the scratchpad (so the committed `*_out.txt` were not overwritten), CPU,
1.5-20 s each. **Every output is byte-identical to the committed `*_out.txt`**, and every number quoted in appendix E matches
(two text slips listed in Sec. 9). Two new referee checks: `docs/paper/review_draft5/theory_check_extra.py` (CPU, ~5 s).

---

## 1. A0 (propagator identity)

**Homogeneous part and score identity: CORRECT.** Howard et al.'s equation is dF/dt = alpha(t) (-s F + f) (confirmed in
`docs/lit/LIT_HOWARD_TRANSFORMER.md` l.7 and Alipour et al.). With alpha piecewise constant the flow over token u is
exp(-s Delta_u) and the propagator is exp(-s (Phi_t - Phi_t')). Code: `_apply_rope` computes R(theta) x
(out1 = x1 cos - x2 sin, out2 = x1 sin + x2 cos), so the score is (R(theta_t) q)^T (R(theta_s) k) = q^T R(theta_s - theta_t) k
= Re(conj(q~) k~ e^{-i(theta_t - theta_s)}), as stated. Re-run: 2.9e-14 (ODE) and 3.55e-15 (score).

**Inhomogeneous formula: WRONG.** The appendix states F(t) = e^{-s Phi_t} F(0) + sum_u e^{-s (Phi_t - Phi_u)} Delta_u f_u,
"exactly, for piecewise-constant f". On (u-1, u] with alpha = Delta_u, f = f_u constant, variation of constants gives
F(u) = e^{-s Delta_u} F(u-1) + f_u (1 - e^{-s Delta_u}) / s. So the u-th weight is (1 - e^{-s Delta_u})/s, which equals
Delta_u only to first order in s Delta_u. Counterexample (`theory_check_extra.py` (1), RK4, s = 0.7i, T = 12, F(0) = 0):
|ODE - stated| = 0.857 with |F| = 2.19; |ODE - corrected| = 1.1e-14. The A0 check script tests only the homogeneous part.
Not used in the main text.

*Fix (appendix A0, replace the sentence "The solution with input is ... exactly, for piecewise-constant f."):*
> The solution with input is F(t) = e^{-s Phi_t} F(0) + sum_{u <= t} e^{-s (Phi_t - Phi_u)} ((1 - e^{-s Delta_u}) / s) f_u,
> exactly, for piecewise-constant f (zero-order hold); the weight (1 - e^{-s Delta_u})/s is Delta_u + O(s Delta_u^2), and a
> zero-rate token writes nothing. For an impulse input f_u delta(t - u) the weight is exactly Delta_u.

**Note 2: CORRECT WITH FIX.** "Real s gives a decay exp(-s (Phi_t - Phi_s)) in (0, 1]" holds only for s > 0 and a
non-negative rate (Phi non-decreasing); with a signed rate it can exceed 1. *Fix:* "Real s > 0 with a non-negative rate gives
a decay ... in (0, 1]". Note 3: Mamba-3's text says Delta_t in R_{>0} (`mamba3.txt:154`); "softplus(.)" is an implementation
detail -- write "Delta_t > 0". Notes 1, 4, 5: CORRECT.

**Main text Sec. 3 ("Three integrators"): CORRECT WITH FIX.** Phi is per channel in MapWM (the step is a vector). *Fix:*
"with Phi_{c,t} = sum_{u <= t} Delta_{c,u}, channel c ... depends on position only through exp(-i omega_c (Phi_{c,t} -
Phi_{c,t'}))". Status "proved; the identity is Mamba-3's Prop. 3" is right (`mamba3.txt:662`).

## 2. B0 (non-negative values vs barrier parameterisations)

(a) remainder arithmetic: CORRECT (omega Delta' = omega Delta - 2 pi k sign(omega)). The gradient remark is right for omega
too: locally the angle is omega s - 2 pi k sign(omega), so d/domega and d/ds equal the signed ones.
*Overclaim:* "it stays in the rank-r class when at most r tokens step ..., which is P1(a) and covers every model in the
paper." The full arms' trained signed tables have every token stepping (V = 19 or 21 > r = 4), so their remainder is not
rank r. *Fix:* "... which is P1(a); it covers every observation-step-free model, and, with the observation embeddings
annihilated, a map code for every |.| arm (P1(a), full arms)."

(b) CORRECT WITH FIX. "a step cannot be carried from one side of zero to the other by a small parameter change" is vacuous:
the value is never negative. What is excluded is a change of *relative* sign within a channel. *Fix:* "In both, the step is
never negative, so no path moves one token's angle in a channel through zero while another's stays put (omega flips a whole
channel; T1)."

(c) CORRECT (softplus(0) = 0.693; `model_sign.py` mode "pos").

## 3. T1 / Lemma 1 (relative-sign gauge)

**Identity: CORRECT WITH FIX.** J R(phi) J = R(-phi); (Jq)^T R(-psi) (Jk) = q^T R(psi) k. Code agreement: Q is viewed
(B, T, H, d_head) and `_apply_rope` pairs coordinates (2b, 2b+1), so the reflected output index h * d_head + 2b + 1 is right;
the W_out row of channel (h, b) is h * n_b + b (`delta.view(B, T, H, nb)`); cos/sin are computed once from the embeddings and
shared by every layer (`MapFormerWM.forward`), so the per-layer reflection and induction are right; `norm1` precedes the
projection. MapEM: A_P = (R q0)(R k0)^T, so reflecting q0_pos/k0_pos is right.
Fixes:
1. For the |.| class, negating row c of W_out does nothing (|.| is applied after W_out); the flip must be carried by omega_c.
   Main text Lemma 1: "(equivalently omega_c, or row c of W_out)" -> "(equivalently omega_c; in the signed class also row c of
   W_out)". Appendix T1: add "In the |.| class only the omega_c form is available."
2. "cos(-x) = cos x and sin(-x) = -sin x hold bitwise" is a property of the libm/kernel used, not of IEEE 754. *Fix:* "hold
   bitwise in the CPU kernels used (checked)".
3. The MapEM sub-check is uninformative: its control (flip without reflection) moves the logits by only 4.7e-5
   (`t1_gauge_out.txt` (f)), because q0_pos/k0_pos are at their 0.02 init. Scale q0_pos/k0_pos (e.g. x 50) before the check,
   or drop "identical" from the MapEM bullet.

Main text "*Proof:* ... The identity is elementary; its consequences are what we use. (i) ...": with omega free in sign, (i)
follows from omega carrying the channel sign, not from the reflection (appendix correction 5 says so). *Fix (optional):*
"(i) (with omega's sign free this needs only omega; the identity gives the same if omega's sign were fixed)".

**T1(i): CORRECT WITH FIX.** Both directions check: if rows are sign-coherent, omega' = D omega with W_out unchanged gives
omega'_c |S_cx| = omega_c S_cx; conversely any |.| table omega'_c |X_c.| is row-sign-coherent, so (omega_c != 0) S's row is.
Edge case: if omega_c = 0, Theta's row c is 0 and is realised whatever the signs of S's row, so the iff on S fails. *Fix:* "iff
every row of Theta = diag(omega) S is sign-coherent" (or "assume omega_c != 0").

**T1(ii): CORRECT.** {omega_c |s_c|} = {omega_c s_c} = R^C; omega crosses zero continuously and with one stepping token
flipping omega flips only that token. Label (representation proved; learnability NOT confirmed by GAUGE) is right.

**Lemma 1(ii) in the main text: WRONG as worded.** "So the barrier excludes no map code where one with sign-coherent rows
exists: one stepping token ... or two moves that step the same way (+1 and +2)." For +1/+2 the code a(+1) = 1,
a(+2) = 2 - 2 pi is a valid map code (2 a(+1) mod 2 pi) with an opposite-sign row, and |.| excludes it as a parameter point.
What is true is that no map *function* is excluded, and a sign-coherent representative exists without wrapping. *Fix:*
> (ii) So the barrier excludes no map *function* for which a code with sign-coherent rows exists, and that code needs no
> wrapping: one stepping token (parity, mod-m counting; Mamba-3 Sec. 3.2, Grazzi et al. Thm. 3) or two moves that step the
> same way (+1 and +2: a(+2) = 2 a(+1)).

**T1(iii): CORRECT.** Through-zero cancellation needs Theta_x = -Theta_y != 0, an opposite-sign row; the only unwrapped
cancellation with same-sign entries is the blind channel. "The cost ... is therefore a cost of reaching that representative"
is an interpretation (the existence results support it); acceptable as worded with "(P3)" pointing to geometry.

Checks (`t1_gauge.py`): re-run identical; 35 channels flipped, logits 0.00 vs control 6.62 (float32), AO via omega 0.00 vs
22.0, 2-layer biased 0.00 vs 0.23, D|S| rewrite 1.2e-11 / bitwise via omega, SO_s800 53/64 opposite-sign, |.| reload changes
logits by 35.5. All as stated.

## 4. P1 (representability by wrapping)

**P1.0: CORRECT.**

**P1(a): CORRECT.** W_in' with W_in' e_{x_j} = u_j and W_in' e_y = 0 exists iff the stepping embeddings are independent
modulo the span of the others; omega' = max(|omega|, 1) > 0; W_out'[:, j] = remainder(Theta_{., x_j}, 2 pi)/omega' >= 0, so
|.| is the identity on it and angles are Theta mod 2 pi. Matches `wrap_construct(mode="balanced")` (W_in re-solved by
pinv(e_act) so the latents are c e_j; omega' = |omega|.clamp_min(1); steps = remainder(omega Delta, 2 pi)/omega').
Vocabularies are 19 (ring) and 21 (torus) <= 128. Re-run: built tables >= 0, angle error <= 7.4e-14, logits <= 7.7e-14.

Main text use: CORRECT WITH FIX. "so P1(a) covers every |.| arm here" -- P1(a) reproduces a *given* table with <= r stepping
tokens; for the full arms what it gives is: any code in which only the moves step, in particular the trained signed
observation-step-free solutions. *Fix:* "...leaves the moves as the only stepping tokens (2 on the ring, 4 on the torus;
r = 4), so every |.| arm here can represent any code in which only the moves step, including the trained signed
observation-step-free solutions of Sec. 4.2 (built in the real modules on the ring; appendix D)."

**P1(b): CORRECT** (definition-level; rank(Sigma o Theta') is invariant under the row scaling by omega').

**P1(c): CORRECT.** Re-derived the cross-multiplied condition: 2 pi alpha_1 (l_1 v_2 - l_2 v_1) + 2 pi alpha_2 (k_2 v_1 -
k_1 v_2) + 4 pi^2 (k_2 l_1 - k_1 l_2) = 0, affine in alpha; the identically-zero case gives b_2 = v_2 / v_1 < 0, inadmissible;
otherwise a line; countable union of lines is null. The b_1 = 0 and a_c = 0 cases are null as stated (a_c = 0 forces
alpha_c v_1 in 2 pi Z). The restriction-to-a-pair argument for m >= 2 is valid (necessary condition). Not used in the main
text.

**P1(d): CORRECT.** Kronecker (continuous flow, Q-independent b) per channel with b shared; b >= 0 and a arbitrary sign is
exactly the rank-1 |.| class. The eps^{-(m-1)} scale is labelled heuristic; the re-run errors (m = 2: 0.17 -> 2e-4; m = 4:
1.53 -> 0.15 over K = 10 -> 1e4) fit K^{-1/(m-1)} roughly (m = 3 is slower than K^{-1/2}: 1.15 -> 0.065).

**P1(e): conjecture, correctly labelled.** A dimension count supports it: for r = 2, m = 3 the rank-2 C x 3 variety and each
wrapped |.| stratum have dimension 2C + 2 in R^{3C}, so for C > 2 their intersection is lower-dimensional unless they
coincide locally, which a mixed-sign row prevents (it would need a non-zero integer vector in a generic row space, or a
codimension-1 condition). Not a proof.

**Softplus representability: GAP (missed result; the stated reason is wrong).** Main Sec. 3: "Softplus cannot set a step to
zero; whether the softplus arms can represent the map, exactly or approximately, was not tested." App. D4: "softplus cannot
reach 0, so observation steps cannot be annihilated; its representability is untested." Sec. 6: "untested
representability". The observation steps need not be 0, only 0 mod 2 pi:

> **P1(f) (softplus, exact).** Let at most r tokens step in a signed table Theta, with embeddings as in P1(a). Choose W_in'
> sending the stepping embeddings to u_1..u_m and every other embedding to 0, set omega'_c = 2 pi / ln 2 for every c, and
> W_out'[c, j] = softplus^{-1}((remainder(Theta_{c,x_j}, 2 pi) + 2 pi) / omega'_c). Every non-stepping token then steps
> softplus(0) = ln 2, an angle of exactly 2 pi, and every stepping token's angle is Theta mod 2 pi. (Any omega'_c = 2 pi k / ln 2,
> k >= 1, works.)

Checked (`theory_check_extra.py` (2)): a full `MapFormerWM_Pos_r4` built this way from SO_s800 has min step 0.693 > 0, angle
error mod 2 pi 8.4e-15 rad over all 19 tokens, and logits equal to SO_s800's to 2.8e-11 (scale 21.6). On the torus m = 4 = r,
so the same construction applies. *Fixes:* main Sec. 3, replace the last sentence of "Representability" with "Softplus
cannot set a step to zero, but it need not: with the observation embeddings annihilated every observation steps
softplus(0) = ln 2, and omega' = 2 pi / ln 2 makes that a full turn, so P1(a) holds for softplus too (appendix E). Its
existence under training was not tested." D4: replace its last sentence accordingly. Sec. 6: "Softplus carries an
initialisation confound; its representability is proved but its existence under training was not tested."

## 5. P2 and P2.1 (rescaling): CORRECT

phi = a+ n+ + a- n- is linear; signed cancelling code error eps a+ d (0 at d = 0); wrapped code a+ d + 2 pi n- gives
2 pi eps n- at d = 0; the wrap lattice a+ N in 2 pi Z is not scale-invariant. P2.1: `train.py:57` builds
`AdamW(model.parameters(), weight_decay=wd)`, so omega, W_out, W_in and the embedding all decay; the angle is their product,
so the decay term rescales it by (1 - lr wd)^4. Status (mathematics; drift account withdrawn) is right; not used in the main
text. Re-run identical (pairs 106,602; error <= 1.0e-13; rms n- 26.6; accuracies 1.000/0.941/0.700/0.485 vs signed).

## 6. P3 (cut circle): CORRECT

(a) the closed quadrant maps onto the torus. (b) cancelling set {a_x + a_y in 2 pi Z} minus blind points; quadrant part
k >= 1 (minus the blind points (2 pi j, 2 pi (k - j)); trivial). (c) distances |sigma - 2 pi k| / sqrt 2; in the |.| class
(union of the closed positive and negative quadrants, omega's sign free) the projection onto sigma = 2 pi from the open
quadrant is (a_x + delta, a_y + delta), delta = (2 pi - sigma)/2, non-blind and inside the quadrant; the negative quadrant is
farther. (d) |sigma| = |omega| (|s_x| + |s_y|) is continuous along any |.| path; non-blind cancelling points have
|sigma| >= 2 pi; IVT. (d) holds per channel in the multi-channel model too, since every channel's pair stays in the union of
quadrants. Re-run: grid matches both formulas to 2e-4.

Main text: CORRECT WITH FIX. (1) "the nearest cancelling point" is nearest in angle coordinates (Euclidean in (a_x, a_y)),
not in parameter space, and an infimum. *Fix:* "the nearest cancelling point (in angle coordinates) lies through zero".
(2) "every |.| path to cancellation crosses sigma = pi" -> "crosses |sigma| = pi" (omega < 0 gives the negative quadrant).
(3) Sec. 4.3 "why |.| fails from a small start (an unreachable wrap, P3; ...)": P3 says the wrap is reachable, only through
|sigma| = pi and farther than the cut-off branch. *Fix:* "(a wrap that lies past sigma = pi, P3; the wall, P4; ...)".

## 7. P4 (sticky wall)

**Appendix: CORRECT WITH FIX.** (a) chain rule; (b) L(s) - L(0) = l'(0)|s| + o(|s|), a strict local minimum *along s with
every other parameter fixed* -- in parameter space s = w_c . z_x is shared (w_c moves every token's step in channel c, z_x every
channel's step for token x), so this is not a local minimum of the loss. "Exactly the points where the signed model has a
descent direction through zero" -> "exactly the points (with l'(0) != 0) where the signed loss decreases into s < 0".
(c) the induction needs mu > 0 and |s_0| <= rho (then |s_k| <= max(rho, eta M) = rho stays in the bound region; checked), and
it is GD on s itself; GD on (w, z) moves s by eta l' sign(s) (|w|^2 + |z|^2) + O(eta^2). *Fix:* "(c) Gradient descent on s
with step eta, with l' in [mu, M] on [0, rho], mu > 0, eta M <= rho and |s_0| <= rho, gives ...".
Check wording: `p3_p5_geometry.py` defines a = |s| (the pre-omega table), so "dL/ds = sign(s) dL/da exactly" should read
"dL/ds = sign(s) dL/d|s|" (re-run: difference 0, 69/128 negative entries).

**Main text: WRONG.** "*P4 (sticky wall):* with a = omega |s|, dL/ds = sign(s) dL/da" omits the factor omega
(dL/ds = omega sign(s) dL/da). *Fix:*
> *P4 (sticky wall):* with a = omega |s|, dL/ds = omega sign(s) dL/da, so where the loss at the wall increases with |s|,
> s = 0 is a strict local minimum along s, other parameters fixed (cf. Woodworth et al. 2020; Amid and Warmuth 2020,
> Prop. 2 at tau = 0, where w = |u|).

The P4 trained-model paragraph is labelled post hoc and MIXED; consistent with main Sec. 4.3.

## 8. P5 (pi-mirror): CORRECT WITH FIX

P5.0: d = 0 mod N, N even -> d even -> g = d + 2 n- even; requires moves +-1 only (it fails on the +1/+2 task; P5 is not used
there). Pm: sigma -> 2 pi - sigma, delta -> -delta; phase -> pi g - phi; even g -> -phi, odd g -> pi - phi; with T1's J the
even-g term returns and the odd-g term is negated (q^T J R(pi - phi) J k = -q^T R(phi) k in the appendix's sign convention).
(e): phase = sigma g/2 + delta d; the map gives -(sigma g/2 - delta d) mod 2 pi; conditions (1) and (2) are right.
Fixes: (a) "the raw phase of every interval" assumes the AO model (no observation steps); say so. (d) the image of the
punctured line {0 < |a+| < pi} is the wrap segment minus (pi, pi) (whose preimage is the blind a+ = 0): "onto ... minus
(pi, pi)". Not used in the main text. Re-run identical (all even; 5.6e-16 / 8.9e-16 / 2.0e-15; mirror 7.4e-3, 8.9e-16,
0.46; model level 1.0e-10 vs control 45.2; Pm accuracy 0.850).

## 9. Remark R / main "The collective route": CORRECT WITH FIX

Labelled account / hypothesis; the signed side is the registered ACCOUNTS T2 result; consistent with main Sec. 4.3.
"the cancelling set {z(-1) = -z(+1)} is a linear subspace": the set of latents at which *every* channel cancels through zero
is {W_out (z(+1) + z(-1)) = 0}, which equals {z(-1) = -z(+1)} iff W_out has full column rank (C = 64 >= r = 4, generically);
otherwise it is a larger linear subspace; either way linear in the latents at fixed W_out (it is bilinear jointly).
*Fix (main Sec. 3):* "the latents at which every channel cancels through zero, {z(-1) = -z(+1)} (for W_out of full column
rank), form a linear subspace". Under |.| z(-1) = -z(+1) gives a(-1) = a(+1): correct.

## 10. P6 (CARoPE cannot cancel): CORRECT

Code: `MapFormerWM_CARoPEF.phase` uses f = 1/(softplus(x W + b) + 1) in (0, 1) per head from the raw token embedding, step
f^i for block i = 0..31, cumsum with no omega multiplier; channel 0 steps exactly 1. So every per-token advance is in (0, 1],
as stated. (a) move-query/observation-key return g = 2 spans 3 tokens -> (0, 3]; observation pair 4 tokens -> (0, 4]; both
inside (0, 2 pi). (b) P_1 in (0, 3], increments in (0, 4]; the arc (eps, 2 pi - eps) is longer than 4 iff eps < pi - 2; induction
gives P_j in (0, eps]; hence J (c+ + c-) <= eps and the move phase of <= 2J moves is <= 2 eps. Main Sec. 4.6's "(0, 1] ...
(0, 4] < 2 pi" is the observation-pair version, valid. Re-run identical (g = 2 advance in [0.000, 3.000]; channels that read
every return are blind, step <= 1.0e-3, phase <= 0.105; 10,331 draws, bound held). The "[0.000" lower end is numerical
underflow of f^i (f^31), i.e. blind channels; fine.

## 11. P7 (CoPE lattice)

**Setup: CORRECT for `CoPEF`.** One layer, q and k from norm1(token embedding), so the gate sigmoid(q.k/sqrt(d_h)) is a table
over (head, query token, key token); p_ij = sum_{k=j}^{i} g_ik (flip-cumsum-flip); from key o_s to query a_t the sum covers g
moves and g observations (o_s .. o_{t-1}), so p = n+ alpha + n- beta with constant observation gates. Re-run: exact (0.0).

**P7.0: CORRECT** (the kink set is a non-empty subset of Z invariant under the period; period 1 forces a constant). The table
is finite (npos_max 2048); "periodic" means on the used range; harmless.

**P7 lattice: CORRECT.** p = n- (alpha + beta) + D alpha; (i) out-and-back forces alpha + beta in P Z with
0 < alpha + beta < 4, so alpha + beta = P in {2, 3}; (ii) D = 32 forces 32 alpha / P in Z; (iii) k even aliases D = 16, k odd
gives Dk/32 in Z iff 32 | D. Family: alpha - beta = 2 alpha - P = (k - 16) P / 16, i.e. 1/8 times an odd number for P = 2 (as
stated).

**Precision: GAP (small).** The first two terms are a valid sufficient condition (returns at 0 mod P, nearest non-returns at
+-P/32, so a P/64 perturbation budget). "tau sqrt(N) < P/64" is a typical-size estimate for a sum of N observation-gate
deviations, not a bound. *Fix:* "|sum of observation-gate deviations| <= N max_o |dg_o| < P/64 is sufficient; for independent
deviations of size tau the typical size is tau sqrt(N) (heuristic)."

**Line (Dirichlet): CORRECT.** For any alpha some 1 <= |D| <= D_max has dist(D alpha / P, Z) <= 1/(D_max + 1).

**App. H: WRONG.** "the exact solutions form a lattice (period P = 2 or 3; move-gate difference an odd multiple of P/32)".
alpha = kP/32 with k odd, so the difference alpha - beta = (k - 16) P / 16 is an odd multiple of **P/16** (1/8 at P = 2, as P7
says). *Fix:* "...; alpha an odd multiple of P/32, i.e. move-gate difference an odd multiple of P/16".

Main Sec. 4.6 uses P6 and P7 consistently (CoPE representable, constructed; obstacle unidentified; precision "a plausible
account").

## 12. Other main-text statements

- **Sec. 1:** "a rotation by a non-negative angle has complex eigenvalues" -- R(pi) has the real eigenvalue -1 (Mamba-3's own
  parity example). *Fix:* "a rotation by theta has eigenvalues e^{+-i theta}, which are real and non-negative only when theta
  is a multiple of 2 pi". The cited theorems: Sarrof et al. Thm. 2 (parity), Grazzi et al. Thm. 1 (parity) and Thm. 2
  (mod m, m not a power of two, needs a non-real eigenvalue); cite them by number.
- **Abstract:** "(Delta mod 2 pi / omega is the same model)" -> "(Delta mod 2 pi/|omega| gives every channel the same angle)":
  |omega|, and per channel (B0's rank caveat).
- **Sec. 2:** model, |.| placement (after W_out, per channel: `SignConstrainedActionToLie.forward` applies `.abs()` to the
  (B, T, H, nb) output of W_out), observation-step-free classes (`step_table` zero except `act`), softplus start, omega
  ladder (omega_i = 2 pi (1/grid)^(i/31), free sign) all match the code. CORRECT.
- **Sec. 3 Representability:** "Per channel a non-negative step reaches any angle mod 2 pi (-1 = m - 1 in Z_m)" -- add
  "(Grazzi et al. 2025, Thm. 3 and App. C.1: Z_m by rotations through 2 pi/m; Mamba-3 Sec. 3.2: parity by R(pi x_t))".
- **Sec. 4.5:** "By Lemma 1(ii) a |.| step represents a map whose moves step the same way without wrapping or crossing
  zero" -- consistent with (ii) as representation; the learnability reading is correctly called unconfirmed.
- **Sec. 5:** "A reparameterised weight cannot change sign under gradient flow (Woodworth ...; Amid and Warmuth 2020: w = |u|
  is gradient descent on the non-negative orthant)" -- verified against Amid and Warmuth Prop. 2 (w = q_tau(u) ~ |u|^{2/(2-tau)};
  tau = 0 recovers GD on the non-negative orthant). Cite "Prop. 2, tau = 0".
- **Sec. 6:** "Softplus carries ... untested representability" -> see Sec. 4 (P1(f)).

## 13. Check scripts and appendix housekeeping

| script | runtime | output vs committed | claims in appendix E |
|---|---|---|---|
| `a0_propagator.py` | 1.5 s | identical | match (homogeneous only; see Sec. 1) |
| `t1_gauge.py` | 4.4 s | identical | match; MapEM control 4.7e-5 is uninformative |
| `p1_wrap.py` | 3.2 s | identical | match; see slips below |
| `p2_rescale.py` | 7.0 s | identical | match |
| `p3_p5_geometry.py` | 6.9 s | identical | match; P4 line uses a = \|s\| |
| `p5_pm_accuracy.py` | 2.6 s | identical | match (0.850) |
| `p6_carope.py` | 7.5 s | identical | match |
| `p7_cope.py` | 19.6 s | identical | match |

Slips: (1) P1 "Checked" (a): "(m = 1..4; 39-56 of 64 channels have opposite-signed steps)" -- m = 1 has 0/64; write
"m = 1..4; for m = 2..4, 39-56 of 64". (2) The script labels its r = 1 part "(b)" and the approximation "(c)"; the appendix
calls them P1(c) and P1(d). (3) "Reproduce: each step takes about 1-5 min" -- they take 1.5-20 s. (4) Header and P1(a) still
point to draft 4 ("for draft 4", "draft 4 Sec. 4.2"); draft 5 uses this appendix. (5) App. D2's prose "steps to
remainder(Delta, 2 pi / |omega|) rescaled to omega'" reads as |omega| Delta mod 2 pi, wrong for omega < 0; the code
(`remainder(omega Delta, 2 pi) / omega'`) is right -- write it that way.

## 14. Citations for standard facts

| fact | where used | citation to add |
|---|---|---|
| Z_m by rotations; -1 = m - 1 | Sec. 1, Sec. 3 Representability, T1(ii) | Grazzi et al. 2025 Thm. 3, App. C.1 |
| parity by R(pi x_t) | Sec. 3 Lemma 1(ii), T1(ii) | Mamba-3 Sec. 3.2 (`mamba3.txt:576-578`) |
| non-negative real eigenvalues cannot do parity / mod m | Sec. 1 | Sarrof et al. Thm. 2; Grazzi et al. Thms. 1-2 |
| complex SSM = data-dependent RoPE | Sec. 3, A0 | Mamba-3 Prop. 3 (cited) |
| w = \|u\| is GD on the orthant; reparameterised weights keep sign | Sec. 3 P4, Sec. 5 | Amid and Warmuth 2020 Prop. 2 (tau = 0); Woodworth et al. 2020 (cited) |
| Kronecker; Dirichlet approximation | P1(d), P7 | named; a textbook reference (e.g. Hardy and Wright) would do |

## 15. GAP / WRONG items and fixes (list)

1. **A0 inhomogeneous formula -- WRONG.** Replace Delta_u f_u by ((1 - e^{-s Delta_u})/s) f_u ("exactly, for
   piecewise-constant f"); Delta_u is exact only for impulse input. Counterexample 0.857 vs |F| 2.19.
2. **Main Sec. 3 P4 -- WRONG.** "dL/ds = sign(s) dL/da" with a = omega |s| -> "dL/ds = omega sign(s) dL/da"; add "along s,
   other parameters fixed" to "strict local minimum". Appendix P4(c): add mu > 0, |s_0| <= rho, "GD on s".
3. **Main Sec. 3 Lemma 1(ii) -- WRONG as worded.** "excludes no map code" -> "excludes no map *function* for which a code
   with sign-coherent rows exists" (counterexample: +1/+2 code (1, 2 - 2 pi)).
4. **App. H P7 summary -- WRONG.** "move-gate difference an odd multiple of P/32" -> "P/16" (alpha is the odd multiple of
   P/32).
5. **Softplus representability -- GAP.** Provably exact (P1(f) above: observations annihilated, omega' = 2 pi / ln 2);
   built and checked (logits to 2.8e-11). Fix main Sec. 3 last sentence of Representability, App. D4's last sentence, and
   Sec. 6.
6. **P7 precision -- GAP (small).** tau sqrt(N) is heuristic; the sufficient condition is N max |dg_o| < P/64.

CORRECT WITH FIX (wording, no change of claim): B0(a) "covers every model"; B0(b); T1 omega-only for |.|; T1 bitwise caveat;
T1(i) omega_c = 0 edge; MapEM check control; P3 main text ("angle coordinates", |sigma|); Sec. 4.3 "unreachable wrap";
Remark R full-column-rank; P5 scope and "onto"; A0 note 2; Sec. 1 eigenvalue sentence; abstract |omega|; Phi per channel;
P1(a) main-text "covers"; check-text slips (Sec. 13).

Status labels: all correct as given, except A0 (its inhomogeneous clause is not proved as stated) and the P7 precision
paragraph (part heuristic). No result labelled "proved" is only numerical. No single-channel result is used in the main text
for a multi-channel claim beyond what it supports, except the P4 "strict local minimum" (needs "along s") and the
collective-route set (needs full column rank). No ring result is applied to the torus in the theory sections (P5's even-gap
lemma is ring-only and is not used for the torus; Sec. 4.4 states the start lever does not transfer).
