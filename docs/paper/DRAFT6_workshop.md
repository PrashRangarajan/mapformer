# Rotary phase steps parameterised to be non-negative rarely learn a map, even when they can represent it (one-layer models, synthetic maps)

*Draft 6, 2026-10-16. Changes and word counts: `docs/paper/DRAFT6_changes.md`. Appendices A-D and F-K:
`docs/paper/APPENDIX_draft6.md`; appendix E (theory): `docs/paper/APPENDIX_theory_v3.md`. Delete this line before submission.*

## Abstract

Content-dependent rotary encodings accumulate a phase theta_t = omega * sum_{u <= t} Delta_u. A non-negative step *value*
costs nothing there (Delta mod 2 pi/|omega| gives the same angles), but a non-negative *parameterisation*,
|s| or softplus(s), puts a barrier at zero that gradient steps cannot cross. In one-layer models trained and tested at equal
length on synthetic maps (a 2D torus, a 1D ring), barrier steps solve the map on 0-4 of 8 seeds per batch against
8 of 8 for signed steps at the default recipe (6 of 8 at a third of the learning rate); on the ring |.| reaches 0.95-0.97
accuracy but rarely an exact map. Both barrier classes can represent the map exactly; built |.| solutions are kept (ring) or
re-found (torus) by training, yet from scratch the class solves 0-1 of 8. The barrier removes the through-zero route by
which signed steps cancel an inverse pair, but that does not explain the failure: with moves +1/+2, where no step need cross
zero, |.| still solves 2 of 8 (signed 5 of 8; unmeasured). The cost holds without weight decay, at a third of the learning
rate and in a 2-layer d 256 torus model; it shrinks at three times the learning rate and, in accuracy, in the larger model.
Negatives are scoped to our budgets and learning rates.

## 1. Introduction

A rotary position code is an integrator. Index RoPE adds a constant per token; content-dependent variants let each token set
its own step, non-negative by construction (CARoPE's rate in (0, 1), Veisi et al.; in a different slot, CoPE's gated counts,
Golovneva et al.) or of either sign (MapFormer's path-integrated phase, Rambaud et al.; Selective RoPE, Movahedi et al.;
Mamba-3, Lahoti et al.).

Sign decides what real-valued recurrences can represent: real non-negative eigenvalues cannot track parity (Sarrof et al.
2024, Thm. 2; Grazzi et al. 2025, Thm. 1), and real eigenvalues cannot count modulo m for m not a power of two (Grazzi et
al., Thm. 2). Rotations escape this: a rotation by theta has eigenvalues e^{+-i theta}, real and non-negative only at
multiples of 2 pi, and rotations by non-negative angles build cyclic groups (Grazzi et al., Thm. 3; Mamba-3, Sec. 3.2). A
non-negative step *value* is harmless there; a non-negative *parameterisation* may not be: |s| folds and softplus(s)
flattens the step at zero, so training cannot move a step through zero. Trained signed models cancel opposite moves with
antipodal latents (Rambaud et al., Figs. 3d, 9b), a through-zero code the barrier excludes; signed steps reach it
collectively, but losing this route does not explain the barrier's failure (Sec. 4.5).

**Contributions.**
1. *A learnability cost at matched length, within budget*: on a torus and a ring, barrier steps rarely reach an exact map
   (Sec. 4.1).
2. *Existence.* Both barrier classes represent the map exactly; built |.| solutions are held (ring) or re-found (torus) by
   training, yet from scratch the class solves 0-1/8 (Secs. 3, 4.2).
3. *Lemma 1*: the barrier fixes only relative signs within a channel; for an inverse pair it removes the through-zero route,
   not the function. Its learnability prediction was not confirmed (Sec. 4.5).
4. *The route signed steps use* (ring): through zero, collectively via their latents; |.| learns reliably only from a start
   whose opposite angles sum past pi, which fails on the torus within 900 epochs (Secs. 4.3-4.4).
5. *Published monotone methods* (faithful CARoPE, CoPE) do not learn these maps within budget, CoPE although it can
   represent the ring map (Sec. 4.6).

## 2. Setup and terms

**Model.** MapFormer-WM (MapWM; Rambaud et al.): one layer, 2 heads, d = 128. Query and key are rotated per rotary channel c
(a head and a 2D block) by theta_{t,c} = omega_c * sum_{u <= t} Delta_{c,u}, with Delta_u = W_out W_in x_u through a rank-4
latent shared across heads (MapFormer's is per head). Theta_{c,x} = omega_c Delta_{c,x} is token x's angle in channel c.
**Step parameterisations.** Signed: Delta = s = W_out W_in x. |.|: Delta = |s|. Softplus: Delta = softplus(s), which starts
every token at the same step (softplus(0) = 0.69), an initialisation confound |.| does not have. |.| and softplus are
*barrier parameterisations*: no gradient step carries Delta through zero. *Observation-step-free* versions give steps to the
move tokens only.
**Tasks.** A 2D torus walk (64 x 64, T = 1024 moves interleaved with observations, 900 epochs): predict the observation at
revisited cells, scored on a held-out map. A 1D ring (32 cells, T = 128, 300 epochs; maps redrawn for every walk) with
forward-step probability p_plus (0.5 is balanced). A line: a 256-cell ring the walk never wraps.
**Terms.** The +1 and -1 moves form an *inverse pair*. A channel *cancels* it if Theta_{c,+} + Theta_{c,-} = 0 mod 2 pi
(Theta_{c,+} != 0 mod 2 pi): *through zero* if the sum is 0, *by wrapping* if a non-zero multiple of 2 pi.
*SOLVED*: training loss over the final 5% of training < 0.05.
**Protocol.** Every verdict was pre-registered with its decision rule (post hoc readouts are marked); every model is trained
and tested at the same length; n = 8 seeds per arm unless stated. Every main-text batch but the torus sign batch (Sec. 4.1;
authors' pre-launch gate, audit after the run) was checked before launch by an independent code-verification agent, an
automated reviewer blind to results whose findings were amended into the pre-registration. Pilots: outside seeds.
"Unmeasured": below the registered threshold; "does not learn": within the stated budget and learning rate.

## 3. Theory

**Representability.** Per channel a non-negative step reaches any angle mod 2 pi (-1 = m - 1 in Z_m; Grazzi et al., Thm.
3), but our step is a shared rank-r map. *P1(a) (proved).* If at most r tokens step, with embeddings independent of the
others', a |.| rank-r model reproduces a signed angle table exactly mod 2 pi. Annihilating the observation embeddings in
W_in leaves only the moves stepping (2 on the ring, 4 on the torus; r = 4), so every |.| arm can represent any code in which
only the moves step (built; appendix D). *P1(f) (proved).* Softplus cannot set a step to zero but need
not: with the observations annihilated each steps softplus(0) = ln 2, a full turn at omega' = 2 pi / ln 2 (built from a
solved ring model, logits equal to 2.8e-11; appendix E). Its existence under training is untested.

**Lemma 1 (relative-sign gauge).** Negate every token's step in a set of channels (equivalently omega_c; in the signed class
also row c of W_out) and the second coordinate of each such channel's query and key projections (weights and biases, every
layer). Every attention score is unchanged. *Proof:* with J = diag(1, -1), J R(phi) J = R(-phi) (checked bitwise on CPU;
appendix E). (i) Without wrapping, a |.| model realises a signed table with the same latent iff every row of the
angle table is sign-coherent (all stepping tokens in a channel step the same way): |.| fixes only *relative* signs within a
channel. (ii) It excludes no map *function* that has a sign-coherent code, which needs no wrapping: one stepping token
(parity, mod-m counting) or moves stepping the same way (+1/+2: a(+2) = 2 a(+1)); that learning is then cheap is a
prediction (Sec. 4.5). (iii) Through-zero cancellation of an inverse pair needs opposite signs and is excluded; the function
remains as the wrapped code (a, 2 pi - a) (P1): the barrier removes a route to cancellation, not cancellation.

**One channel's geometry (proved; appendix E).** Let sigma = Theta_{c,+} + Theta_{c,-}. *P3 (cut circle):* from a start in
the open non-negative quadrant with sigma < pi, the nearest cancelling point (in angle coordinates) lies through zero, out
of |.|'s reach, and every |.| path to cancellation crosses |sigma| = pi; from pi <= sigma < 2 pi both classes are equally
near, at the wrap. *P4 (sticky wall):* with a = omega |s|, dL/ds = omega sign(s) dL/da, so where the loss at the wall
increases with |s|, s = 0 is a strict local minimum along s, other parameters fixed (cf. Amid and Warmuth 2020, Prop. 2).
Softplus has no fold, but its slope vanishes toward zero.

**The collective route (hypothesis).** P3-P4 are geometry, not dynamics (appendices C, E). For signed steps, the latents at
which every channel cancels through zero, {z(-1) = -z(+1)} with z = W_in x (W_out of full column rank), form a linear
subspace: one latent move cancels every channel, while under |.| it gives a(-1) = a(+1), a clock. Sec. 4.3 tests whether
signed models take it.

## 4. Results

### 4.1 Barrier parameterisations rarely learn the map within budget

On the torus at the training length, |.| costs -0.177 in accuracy (0.821 vs 0.998; permutation p 0.0002) and solves 0/8
against 8/8, with 3/8 |.| runs still descending at 900 epochs (Fig. 1); softplus costs -0.218 (1/8; initialisation
confound). Post hoc floors (appendix B) on the same walks: blank 0.506, best n-gram 0.575, and 0.843 for "retrace", a
map-free rule that copies the observation 2j steps back while a run of moves reverses the previous one; the |.| mean is
below retrace (3/8 runs), as are softplus (0.781; 4/8) and index RoPE (every run, 0.72-0.74). On the ring, |.| solves 2/8
and 1/8 at p_plus 0.5 and 0.75 against 8/8 (Fisher p 0.007, 0.0014; at 0.9 unmeasured, 4/8 vs 8/8), and 3/8 with three times
the training. Ring |.| models reach 0.95-0.97, above the floors (n-gram 0.554; retrace 0.860, post hoc) and the one-layer
index model (0.730 at p_plus 0.5): they rarely reach an exact map, and the ring evidence is the SOLVED criterion. Post hoc,
at a loss threshold of 0.1 the p_plus 0.5 contrast no longer fires (4/8 vs 8/8, Fisher p 0.077); every other contrast in
Table C10 holds at 0.02, 0.1 and accuracy >= 0.99.

Barrier codes do not cancel (Fig. 2, post hoc): signed models cancel opposite moves mod 2 pi in 64-98% of channels, |.| and
softplus models in 3-11%, solved runs included (unweighted; weighted by use 0.70-0.93 vs 0.01-0.10).
Observation-step-free versions fail too: ring 0/8 vs 8/8 (two batches), torus 0/8 vs 8/8 (-0.120) and 1/8 vs 8/8 (-0.104)
(unlike the torus sign batch: fused attention kernel, observation-step-free classes; appendix A3).

*Recipe and scale (registered, fresh seeds; appendix C13).* On the observation-step-free ring the cost holds with weight
decay 0 (0/8 vs 8/8, -0.127), on SOLVED at a third of the learning rate (0/8 vs 6/8, Fisher p 0.007; accuracy p 0.089), and
at the default (1/8 vs 8/8). At three times the rate it shrinks and is unmeasured (4/8 vs 8/8, Fisher p 0.077): |.| SOLVED
rises with the rate (0/8, 1/8, 4/8), and all other comparisons use the default rate, 1e-3. A 2-layer d 256 torus model (8x
the parameters) keeps it (2/8 vs 8/8, Fisher p 0.007; 0.967 vs 0.999, p 0.0005), with a quarter of the one-layer accuracy
gap (out of batch), 6/8 |.| runs still descending and no SOLVED contrast at loss 0.1 (5/8 vs 8/8, p 0.20; a declared
secondary): at scale part of the cost may be speed. In every cell |.| shows little wrapped code, even at lr x3 with 4/8
solved (unweighted arm share 0.04-0.13 vs 0.43-1.00 signed; a declared secondary).

### 4.2 The |.| solution exists: kept on the ring, re-found on the torus

We convert each solved signed observation-step-free model exactly into |.| form (P1(a); appendix D): angles unchanged mod
2 pi (error <= 1.4e-6 rad), frozen accuracy equal to the source's on 8/8 seeds, weights within ~5x of the signed model's. A
naive construction keeping the signed latents is also exact but ill-conditioned, and Adam loses it within ~10 epochs: the
test must be posed near a trained scale.

On the ring, the constructed model trained 300 more epochs stays solved on 8/8 seeds and never leaves the solved region;
weighted by how strongly attention reads each channel, it keeps its cancelling code (99% of used channels at construction,
98% at the end; a registered readout of ACCOUNTS T1, whose retrained runs are bit-identical to these: determinism, not
replication). On the torus it is knocked out early (peak loss 0.11-0.21 vs <= 0.017 for the signed control; 6/8 leave the
solved region) and re-finds the solution (8/8 SOLVED, 1.000) at lr 1e-4, fixed before launch (at 1e-3, descriptive, 6/8).
From scratch the class solves 0/8 on both tasks.

### 4.3 The mechanism on the ring: the route signed steps take

We vary only the initial move angles of observation-step-free ring models, every other weight identical at a seed (Table
1).

| initial move angles | signed | \|.\| |
|---|---|---|
| default (small, signs independent) | 8/8 | 0/8 |
| \|default\|: both moves rotate the same way (the \|.\| start) | **7/8** | 0/8 |
| large, summing to less than pi | -- | 2/8 |
| large, summing to more than pi | 8/8 | **8/8** |

*Table 1. SOLVED, ring p_plus 0.5, 300 epochs, n = 8 per cell; threshold checks in Table C10.*

Four registered contrasts read Table 1. (i) From the same start the sign matters: 0/8 vs 7/8 (Fisher p 0.0014; accuracy
-0.092 [-0.135, -0.045]). (ii) A start past pi rescues |.|: 8/8 vs 0/8 (+0.109 [+0.085, +0.135], p 0.0002). (iii) Size alone
does not: a large start with sums below pi solves 2/8 (Fisher p 0.007 vs 8/8; sum and size are only partly separated,
appendix C12). (iv) From past pi the sign makes no difference (8/8 vs 8/8). Post hoc (Fig. 3): signed models from the small
starts end with 98-99% of channels opposite-signed (from 0% at the same-direction start), no cancelling channel wrapping;
from past pi both signs cancel by wrapping (88% and 100% of cancelling channels).

**Signed steps escape collectively (registered; fresh seeds).** From the same-direction start, freezing the two move latents
(W_out and omega free per channel) blocks the escape: 1/8 vs 7/8 SOLVED (Fisher p 0.010; -0.075 [-0.130, -0.020]); 15% of
channels end opposite-signed against 82% with free latents. From the
default start freezing has no detected effect (7/8 vs 8/8). MapFormer described the end state; this is the causal test.

**Loose ends (post hoc; appendix C12).** Rescued |.| models cancel in only 23% of channels, not those the one-channel flow
predicts (kappa +0.09); the final gradient points into the wall for only about half of the collapsed small-start |.| steps
(mixed, by a rule fixed in advance); a transformer-free toy does not reproduce the signed escape (appendix I). Why |.| fails
from a small start (a wrap past |sigma| = pi, P3; the wall, P4; both; neither, Sec. 4.5) is open.

### 4.4 The start lever fails on the torus within 900 epochs

On the torus (observation-step-free, 4 moves) the past-pi start does not rescue |.|: 0/8 vs 1/8 from the default start
(+0.005 [-0.060, +0.070]; no effect detected), and it slows the signed model to 2/8 against 8/8 (secondary); 5/8 runs per
large-start arm are still descending. The default-start cost replicates (1/8 vs 8/8, Fisher p 0.0014).

### 4.5 Does the cost follow the relative sign? (registered)

By Lemma 1(ii) a |.| step represents, without wrapping or crossing zero, a map whose moves step the same way. Does the cost
then disappear? Ring models as in Sec. 4.3, fresh seeds; moves +1/-1 (an inverse pair)
vs +1/+2 (map code a(+2) = 2 a(+1), sign-coherent).

| moves | signed | \|.\| | difference (95% CI) | label |
|---|---|---|---|---|
| +1/-1 | 1.000, 8/8 | 0.931, 1/8 | -0.069 [-0.095, -0.045], Fisher p 0.0014 | signed learns better |
| +1/+2 | 0.947, 5/8 | 0.883, 2/8 | -0.064 [-0.200, +0.075], Fisher p 0.31 | unmeasured |

*Table 2. Accuracy (T = 128) and SOLVED, n = 8 per cell.*

The cost replicates on the inverse pair. +1/+2 needs integration (index RoPE
0.649, 0/8; floor ~0.50), and signed models learn it with a sign-coherent code (0.92 of channels end with both move angles of
one sign, 0.01 on +1/-1; a declared secondary), as Lemma 1 allows. The registered prediction is not confirmed: |.| solves
+1/+2 on only 2/8, no more than +1/-1 (1/8; Fisher p 1.00), though no zero crossing is needed; the comparison is unmeasured
because +1/+2 (all revisits long-gap laps) is harder for signed steps too (5/8). On laps, the stratum both tasks share, |.|
scores 0.893 and 0.884 (post hoc reading of a declared secondary). So losing the through-zero route does not account for
|.|'s difficulty here; Lemma 1 is a statement about representation only.

### 4.6 Published monotone methods (within budget)

Faithful CARoPE (a per-head rate in (0, 1)) cannot represent cancellation (*P6, proved*: the shortest return advances each
channel by (0, 4] < 2 pi); it scores 0.667 on the torus, below index RoPE's 0.722 (p 0.040), and 0.741 vs 0.730 on the ring
(unmeasured). MapFormer reports CoPE failing 1D and 2D navigation (their Table 2) and attributes it to the gate's sign. But
CoPE can represent ring cancellation (constructed; appendix H), so its obstacle is not representational; it solves 0/8
(0.887 vs signed 1.000, p 0.0002; 1/32 runs at the paper's initialisation over three ring batches, pooled post hoc). On
the line CoPE solves 1/16 (ring 0/16), signed steps 8/8, and the index fails (0.772). For CoPE the loss criterion is
fragile: 5/16 line runs score >= 0.9997 held out yet miss the (dropout-on) loss cut, and at held-out accuracy >= 0.999 the
line reads 5/16 vs 2/16 (post hoc), so whether the wrap adds to CoPE's difficulty is unmeasured. The precision its count
needs is a plausible, untested account (appendix H).

## 5. Related work

**Integrators and position.** Per channel the rotary position factor is the propagator of Howard et al.'s (2014)
rate-driven integrators with an imaginary rate constant (appendix E; Mamba-3, Prop. 3): constant, non-negative and signed
rates give elapsed time (index RoPE), a clock and a map. TEM-t (Whittington et al. 2022) and MapFormer put
action-driven position codes into transformers. GRAPE's (Zhang et al.) non-negative
content phase is a real-slot bias, as in FoX (Lin et al.).

**Sign, state tracking and learnability.** In diagonal SSMs, parity needs input dependence and negative eigenvalues in the
same layer (Khavari et al. 2025, Thm. 4.4); none of these theorems restricts an input-dependent rotation by a non-negative
angle. Shakerinava et al. (2026) find a learnability gap for input-dependent rotations in unit-modulus SSMs:
non-Abelian solutions exist but are not reached from scratch, and near-solution starts recover them; ours is a gap created
by one parameterisation choice, on single-layer Abelian tasks that their signed-angle models learn. We give Sarrof et al.'s
and Grazzi et al.'s advice, allow a sign, for a different reason.

**Sign-constrained parameterisations.** A reparameterised weight cannot change sign under gradient flow (Woodworth et al.
2020; Amid and Warmuth 2020, Prop. 2 at tau = 0: w = |u| is gradient descent on the non-negative orthant). In orthogonal
RNNs the +/-1 diagonal of the scaled Cayley parameterisation is fixed during training, its count of -1s a per-task
hyperparameter (Helfrich et al. 2018). The LRU (Orvieto et al. 2023) needs its positive phases initialised near zero to
learn PathX; they are input-independent, so by Lemma 1(ii) their sign is a convention. Dale's-law networks ask the same
question, constraint or start (Cornford et al. 2021; Li et al. 2023). With fixed frequencies the parameter symmetries of
rotary attention are per-plane complex scalars, with no reflection (Wang and Wang 2025; Tran et al. 2026); to our knowledge
the reflection a learnable step adds (Lemma 1), the loss of the through-zero route for inverse pairs, and the learnability
cost of a non-negative rotary step have not been stated. We claim their conjunction and the measurements, not any single
ingredient.

## 6. Limitations

"Does not learn" is scoped to our budgets and learning rates: 3x the ring budget did not help; the cost shrinks at three
times the learning rate (unmeasured) and, in accuracy, in a larger torus model whose |.| runs were mostly still descending;
we ran no per-arm learning-rate sweep, other optimisers or longer budgets at scale. Small models (one layer, 2 heads, d =
128, a rank-4 latent shared across heads), synthetic tasks, n = 8, no recurrent, SSM or exp(.)-type (LRU) baselines. On the
ring the evidence is mainly a loss criterion, which at p_plus 0.5 does not survive a looser threshold (0.1; post hoc). On
the torus the barrier class has less latent slack (the wrapped code also needs the 2 pi k / omega offsets), and per-head
rank, known to decide torus learnability for signed steps, is uncontrolled. Softplus carries an initialisation confound; its
representability is proved, its existence under training untested. The route is shown on the ring only and does not explain
the |.| failure. CARoPE and CoPE were tested at one layer (CoPE on ring and line). Natural text is not studied.

## References

Titles and author lists checked against arXiv, OpenReview, Crossref, ACL Anthology or PMLR metadata on 2026-10-15/16;
venues marked * were not re-checked. Liu et al., Hahn and Rofin, and Yang et al. are cited in appendix K only.

- Amid, E. and Warmuth, M. K. Reparameterizing mirror descent as gradient descent. NeurIPS 2020*; arXiv:2002.10487.
- Cornford, J., Kalajdzievski, D., Leite, M., Lamarquette, A., Kullmann, D. M. and Richards, B. A. Learning to live with
  Dale's principle: ANNs with separate excitatory and inhibitory units. ICLR 2021.
- Golovneva, O., Wang, T., Weston, J. and Sukhbaatar, S. Contextual position encoding: learning to count what's important
  (CoPE). arXiv:2405.18719, 2024.
- Grazzi, R., Siems, J., Zela, A., Franke, J. K. H., Hutter, F. and Pontil, M. Unlocking state-tracking in linear RNNs
  through negative eigenvalues. ICLR 2025*; arXiv:2411.12537.
- Hahn, M. and Rofin, M. Why are sensitive functions hard for transformers? ACL 2024 (long papers); arXiv:2402.09963.
- Helfrich, K., Willmott, D. and Ye, Q. Orthogonal recurrent neural networks with scaled Cayley transform. ICML 2018*;
  arXiv:1707.09520.
- Howard, M. W., MacDonald, C. J., Tiganj, Z., Shankar, K. H., Du, Q., Hasselmo, M. E. and Eichenbaum, H. A unified
  mathematical framework for coding time, space, and sequences in the hippocampal region. J. Neurosci. 34(13):4692-4707,
  2014.
- Khavari, B., Shakerinava, M., Khullar, J., Huang, J., Rivest, F., Ravanbakhsh, S. and Chandar, S. Parity requires unified
  input dependence and negative eigenvalues in SSMs. ICML 2025 Workshop on Methods and Opportunities at Small Scale;
  arXiv:2508.07395.
- Lahoti, A., Li, K. Y., Chen, B., Wang, C., Bick, A., Kolter, J. Z., Dao, T. and Gu, A. Mamba-3: improved sequence modeling
  using state space principles. ICLR 2026; arXiv:2603.15569.
- Li, P., Cornford, J., Ghosh, A. and Richards, B. Learning better with Dale's law: a spectral perspective. NeurIPS 2023.
- Lin, Z., Nikishin, E., He, X. O. and Courville, A. Forgetting Transformer: softmax attention with a forget gate (FoX).
  ICLR 2025; arXiv:2503.02130.
- Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A. and Zhang, C. Transformers learn shortcuts to automata. ICLR 2023;
  arXiv:2210.10749.
- Movahedi, S., Carstensen, T., Afzal, A., Hutter, F., Orvieto, A. and Cevher, V. Selective rotary position embedding.
  ICLR 2026; arXiv:2511.17388.
- Orvieto, A., Smith, S. L., Gu, A., Fernando, A., Gulcehre, C., Pascanu, R. and De, S. Resurrecting recurrent neural
  networks for long sequences (LRU). ICML 2023*; arXiv:2303.06349.
- Rambaud, V., Mascarenhas, S. and Lakretz, Y. MapFormer: self-supervised learning of cognitive maps with input-dependent
  positional embeddings. arXiv:2511.19279 (v4, 10 May 2026).
- Sarrof, Y., Veitsman, Y. and Hahn, M. The expressive capacity of state space models: a formal language perspective.
  NeurIPS 2024; arXiv:2405.17394.
- Shakerinava, M., Khavari, B., Ravanbakhsh, S. and Chandar, S. The expressive limits of diagonal SSMs for state-tracking.
  ICLR 2026; arXiv:2603.01959.
- Tran, V.-H., Bui, V. K., Trinh, V.-H., Lai Ngoc, T. and Nguyen, T. M. Functional equivalence in attention: a
  comprehensive study with applications to linear mode connectivity. ICML 2026 (PMLR 306); arXiv:2606.17830.
- Veisi, A., Fartoot, D. and Amirzadeh, H. Context-aware rotary position embedding (CARoPE). arXiv:2507.23083, 2025.
- Wang, H. and Wang, K. Complete characterization of gauge symmetries in transformer architectures. Symmetry and Geometry
  in Neural Representations (NeurReps), PMLR 282:622-663, 2025.
- Whittington, J. C. R., Warren, J. and Behrens, T. E. J. Relating transformers to models and neural representations of the
  hippocampal formation (TEM-t). ICLR 2022; arXiv:2112.04035.
- Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E., Savarese, P., Golan, I., Soudry, D. and Srebro, N. Kernel and
  rich regimes in overparametrized models. COLT 2020*; arXiv:2002.09277.
- Yang, S., Shen, Y., Wen, K., Tan, S., Mishra, M., Ren, L., Panda, R. and Kim, Y. PaTH attention: position encoding via
  accumulating Householder transformations. NeurIPS 2025; arXiv:2505.16381.
- Zhang, Y., Chen, Z., Liu, Y., Qin, Z., Yuan, H., Xu, K., Yuan, Y., Gu, Q. and Yao, A. C.-C. Group representational
  position encoding (GRAPE). ICLR 2026; arXiv:2512.07805.

## Figures and tables

- Table 1 (Sec. 4.3): start x step parameterisation. Table 2 (Sec. 4.5): moves +1/-1 vs +1/+2. Appendix tables C1-C13
  (and C9b) are in `APPENDIX_draft6.md`.
- **Fig. 1** (`docs/paper/figs/fig_loss_curves.png`; Sec. 4.1). Training loss per epoch for the signed step (blue) and the non-negative steps |.| (orange) and softplus (green), every run of four registered batches (thin lines; thick: median of 8 runs; log scale; training loss with dropout on). Dashed line: the registered SOLVED threshold (final-5% mean training loss < 0.05); n/8 = runs SOLVED. (a) Torus, T = 1024, one layer (SIGN_MATCHED): signed 8/8, |.| 0/8, softplus 1/8. (b) Ring of 32 cells, observation-step-free (WRAP_EXISTENCE): 8/8 vs 0/8. (c) Torus, observation-step-free (GENERALITY): 8/8 vs 0/8. (d) As (c) at two layers and d_model 256 (ROBUST): 8/8 vs 2/8. The signed median crosses the threshold at epoch 480 / 57 / 309 / 183 of 900 / 300 / 900 / 900 and keeps descending; the |.| median never does, and most |.| runs flatten at 0.3-0.5 (STALLED on the ring 8/8; 5-6/8 on the 1-layer torus). The 2-layer |.| runs are still descending at 900 epochs (6/8), so there part of the cost may be speed (budget-scoped).
- **Fig. 2** (`docs/paper/figs/fig_wrap_share.png`; Sec. 4.1). Share of rotary channels in which opposite moves cancel mod 2 pi, per run, weighted by how much attention uses each channel (dots; filled = SOLVED) and unweighted (faint dashes, joined to the same run). A channel cancels when |wrap(a(move) + a(opposite move))| < 0.1 rad; channels whose move angle has collapsed to within 0.05 rad of 0 mod 2 pi are excluded from the denominator. Usage weight: mean query norm of the channel at move tokens times mean key norm at observation tokens (eval walks). Ring of 32 cells at three move biases p_plus (RING_DIAL, n = 8 each) and the torus at T = 1024 (SIGN_MATCHED, n = 8). Bars: arm means of the weighted share. Signed steps: 0.70-0.93; |.| steps: 0.02-0.10; softplus: 0.01. Seven of the eight non-negative runs that solve stay at or below 0.07.
- **Fig. 3** (`docs/paper/figs/fig_mechanism.png`; Sec. 4.3; post hoc). Move angles (a(+1), a(-1)) of every rotary channel, at the start (grey) and after training (dots). There are 8 runs per panel, on the 1D ring. A channel cancels opposite moves when a(+1) + a(-1) = 0 mod 2 pi, on the blue line (through zero; reached only with opposite signs) or the red line (by wrapping). Grey shading marks the two mixed-sign quadrants, the only region the |.| step cannot reach (omega's sign is free, so both angles may be negative); the dotted line is a(+1) + a(-1) = pi; dashed boxes are the U(lo, hi) start boxes. The panels: - Signed step from the same-sign start (SO_absmag): 7/8 runs solve. One move's angle crosses zero: 6% of channels cancel at the start, 89% at the end (88% with opposite signs, none by wrapping). - The |.| step from the same start (AO_absmag): 0/8 runs solve, with 4% cancelling. - The |.| step from a large start whose sums stay below pi (AO_mid): 2/8 runs solve, with 5% cancelling, all by wrapping. - The |.| step from a large start whose sums exceed pi (AO_high): 8/8 runs solve, and cancelling rises from 7% to 23%, all by wrapping. The cancelling share is the registered wrap share: |wrap(a(+1) + a(-1))| < 0.1 rad among channels with |wrap(a(+1))| > 0.05.
- **Fig. C1** (appendix C; `docs/paper/figs/fig_ring_strata.png`). Ring revisit accuracy by gap and wrap-only stratum (post hoc).
- **Fig. K1** (appendix K; `docs/paper/figs/fig_factorial_strata.png`). Revisit accuracy by stratum on the torus at T = 1024 (FACTORIAL; trained and tested at T = 1024; bars: means of n = 4 runs, dots: runs). Strata: plain revisits with a gap of fewer than 128 moves since the last visit, plain revisits with longer gaps, and wrap-only revisits (the cell was seen before only from a different unwrapped position). Dashed: always-blank (the best constant) within each stratum, 0.508 / 0.515 / 0.517; dotted: the retrace rule within the short-gap stratum, 0.912 (elsewhere it equals the constant). Floors are computed on this batch's own evaluation walks. Path integration with one layer is at 1.000 in every stratum. Four index layers or a loop of four lift short-gap revisits to 0.94, just above retrace; at long gaps and wrap-only revisits every index model stays at or below the constant (0.46-0.51).