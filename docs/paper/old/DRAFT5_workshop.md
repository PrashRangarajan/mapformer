# Rotary phase steps parameterised to be non-negative rarely learn a map, even when they can represent it (one-layer models, synthetic maps)

*Draft 5, 2026-10-16. Changes and word counts: `docs/paper/DRAFT5_changes.md`. Appendices A-D and F-K:
`docs/paper/APPENDIX_draft5.md`; appendix E (theory): `docs/paper/APPENDIX_theory_v2.md`. Delete this line before submission.*

## Abstract

Content-dependent rotary encodings accumulate a phase theta_t = omega * sum_{u <= t} Delta_u. A non-negative step *value*
costs nothing there (Delta mod 2 pi / omega is the same model), but a non-negative *parameterisation*, |s| or softplus(s),
puts a barrier at zero that gradient steps cannot cross. In one-layer rotary models trained and tested at equal length on
synthetic maps (a 2D torus, a 1D ring), barrier steps solve the map on 0-4 of 8 seeds per batch against 8 of 8 for signed
steps. Exact |.| solutions exist at a trained scale and training keeps (ring) or re-finds (torus) them; from scratch the
class solves 0-1 of 8. A per-channel identity shows the barrier removes, for an inverse pair, the through-zero route to
cancellation, not the function. On the ring signed steps take that route collectively; |.| steps learn reliably only from a
start whose opposite angles sum past pi, which fails on the torus. Where no step need cross zero (moves +1/+2), both classes
struggle and the comparison is unmeasured. The cost holds without weight decay, at a third of the learning rate and in a 2-layer d 256 torus model; it shrinks at three times the learning rate. Negatives hold within our budgets (300-900 epochs).

## 1. Introduction

A rotary position code is an integrator. Index RoPE adds a constant per token; content-dependent variants let each token set
its own step, non-negative by construction (CARoPE's rate in (0, 1); in a different slot, CoPE's gated counts) or of either
sign (MapFormer's path-integrated phase, Selective RoPE, Mamba-3).

For real-valued recurrences the sign decides what can be represented: transitions with real non-negative eigenvalues cannot
track parity or count modulo m (Sarrof et al. 2024; Grazzi et al. 2025). Those theorems do not reach the rotary slot: a
rotation by a non-negative angle has complex eigenvalues, and Grazzi et al. and Mamba-3 build cyclic groups from such
rotations. A non-negative step *value* is harmless there; a non-negative *parameterisation* may not be: |s| folds and
softplus(s) flattens the step at zero, so training cannot move a step through zero. For learning maps, where opposite moves
must cancel, this matters within our budgets. We identify a route to the map that the barrier removes; we do not explain why
barrier steps fail from small starts, or with +1/+2 moves, where no step need cross zero (Sec. 4.5).

**Contributions.**
1. *A learnability cost at matched length* on a torus and a ring; barrier codes do not cancel opposite moves (Sec. 4.1).
2. *Existence.* An exact |.| solution at a trained scale is held (ring) or re-found (torus) under training; from scratch the
   class solves 0-1/8 (Sec. 4.2).
3. *Which codes the barrier excludes.* Lemma 1 (relative-sign gauge): only relative signs within a channel; for an inverse
   pair, a route to cancellation, not the function. A registered test of its learnability prediction was inconclusive:
   moves +1/+2 were harder for both classes (Sec. 4.5).
4. *A route, on the ring*: signed steps leave the barrier's start through zero, collectively through their latents; |.|
   learns reliably only from a start whose opposite angles sum past pi, which fails on the torus (Secs. 4.3-4.4).
5. *Published monotone methods* (faithful CARoPE, CoPE) fail these maps, CoPE although it can represent the ring map (4.6).

## 2. Setup and terms

**Model.** MapFormer-WM (MapWM; Rambaud et al.): one layer, 2 heads, d = 128. Query and key are rotated per rotary channel c
(a head and a 2D block) by theta_{t,c} = omega_c * sum_{u <= t} Delta_{c,u}, with Delta_u = W_out W_in x_u through a rank-4
latent shared across heads (MapFormer's is per head). Theta_{c,x} = omega_c Delta_{c,x} is token x's angle in channel c.
Recipes: appendix A.
**Step parameterisations.** Signed: Delta = s = W_out W_in x. |.|: Delta = |s| (the sign removed, nothing else). Softplus:
Delta = softplus(s), which starts every token at the same step (softplus(0) = 0.69), an initialisation confound |.| does not
have. |.| and softplus are *barrier parameterisations*: no gradient step carries Delta through zero. *Observation-step-free
(AO)* versions give steps to the move tokens only. Index RoPE uses the canonical ladder.
**Tasks.** A 2D torus walk (64 x 64, T = 1024 moves interleaved with observations, 900 epochs): predict the observation at
revisited cells; each run trains on one map and is scored on a held-out map (accuracy is transfer). A 1D ring (32 cells,
T = 128, 300 epochs; maps redrawn for every walk) with forward-step probability p_plus (0.5 is balanced). A line: a 256-cell
ring the walk never wraps. Floors (predict-blank, best n-gram) are in appendix B.
**Terms.** The +1 and -1 moves form an *inverse pair*. A channel *cancels* it if Theta_{c,+} + Theta_{c,-} = 0 mod 2 pi
(Theta_{c,+} != 0 mod 2 pi): *through zero* if the sum is 0, *by wrapping* if a non-zero multiple of 2 pi.
*SOLVED*: training loss over the final 5% of training < 0.05.
**Protocol.** Every contrast was pre-registered with its decision rule, trained and tested at the same length, n = 8 seeds
per arm unless stated. Every main-text batch except the torus sign batch of Sec. 4.1 was checked by an independent
code-verification agent (an automated reviewer, blind to results, whose findings were amended into the pre-registration
before any result was read); that batch had a pre-launch gate by the authors and an independent audit after the run. Pilots
used seeds outside the batch. "Unmeasured" means below the registered threshold; "does not learn", within the stated budget.

## 3. Theory

**Three integrators.** Howard et al. (2014) drive integrators by a rate: constant gives elapsed time, a speed path length (a
clock), a signed velocity position (a map). Per token the rate is Delta_t; with Phi_t = sum_{u <= t} Delta_u, channel c of
the score of query t and key t' depends on position only through exp(-i omega_c (Phi_t - Phi_{t'})): Howard's propagator
with an imaginary rate constant (appendix E; for complex SSMs, Mamba-3's Prop. 3). Index RoPE is the constant case; the
methods of Sec. 1 are the non-negative and signed cases. In the real (decay) slot nothing wraps; in the rotary slot the last
two cases represent the same per-channel codes.

**Representability.** Per channel a non-negative step reaches any angle mod 2 pi (-1 = m - 1 in Z_m); our step is a shared
rank-r map, so we need a scoped version. *P1(a) (proved).* If at most r tokens have non-zero steps, with embeddings
independent of the others', a |.| rank-r model reproduces a signed angle table exactly mod 2 pi. A W_in that annihilates the
observation embeddings leaves the moves as the only stepping tokens (2 on the ring, 4 on the torus; r = 4), so P1(a) covers
every |.| arm here (built in the real modules on the ring; appendix D). Softplus cannot set a step to zero; whether the
softplus arms can represent the map, exactly or approximately, was not tested.

**Lemma 1 (relative-sign gauge).** Negate every token's step in a set of channels (equivalently omega_c, or row c of W_out)
and the second coordinate of each such channel's query and key projections (weights and biases, every layer). Every
attention score is unchanged. *Proof:* with J = diag(1, -1), J R(phi) J = R(-phi) (checked bitwise; appendix E). The
identity is elementary; its consequences are what we use. (i) Without wrapping, a |.| model realises a signed table with the
same latent iff each row is sign-coherent (all tokens stepping in a channel step the same way): |.| fixes only the
*relative* signs within a channel. (ii) So the barrier excludes no map code where one with sign-coherent rows exists: one
stepping token (parity, mod-m counting) or two moves that step the same way (+1 and +2). That it then costs no learnability
is a prediction, not a consequence; Sec. 4.5 does not confirm it. (iii) For an inverse pair, a channel cancelling through
zero has opposite signs and is excluded; the same function remains as the sign-coherent wrapped code (a, 2 pi - a) (P1). The
barrier removes a route to cancellation, not cancellation.

**One channel's geometry (proved; appendix E).** Let sigma = Theta_{c,+} + Theta_{c,-}. *P3 (cut circle):* from a start in
the open non-negative quadrant with sigma < pi, the nearest cancelling point lies through zero, out of |.|'s reach, and every
|.| path to cancellation crosses sigma = pi; from pi <= sigma < 2 pi both classes are equally near, at the wrap. *P4 (sticky wall):* with a = omega |s|, dL/ds = sign(s) dL/da, so where the loss wants an angle smaller, s = 0
is a strict local minimum (cf. Woodworth et al. 2020; Amid and Warmuth 2020). Softplus has no fold, but its slope vanishes
toward zero.

**The collective route (hypothesis).** P3-P4 are geometry, not dynamics; gradient flow on one channel reproduces neither the
signed escape nor a sharp past-pi rescue (appendices C, E). But for signed steps, with z(+1), z(-1) the moves' latents
W_in x, the cancelling set {z(-1) = -z(+1)} is a linear subspace, so one latent move cancels every channel at once; under
|.| the same move gives a(-1) = a(+1), a clock. Sec. 4.3 tests this.

## 4. Results

### 4.1 Barrier parameterisations do not learn the map

On the torus at the training length, |.| costs -0.177 in accuracy (0.821 vs 0.998; permutation p 0.0002) and solves 0/8
against 8/8, with 3/8 |.| runs still descending at 900 epochs; softplus costs -0.218 (1/8; initialisation confound). On the
same walks (post hoc) predicting blank scores 0.506, the best n-gram 0.575, and a map-free rule that copies the observation
2j steps back while a run of moves reverses the previous one ("retrace") 0.843: the |.| mean is below retrace (3/8 runs
below), softplus too (0.781; 4/8), index RoPE on every run (0.72-0.74). On the
ring, |.| solves 2/8 and 1/8 at p_plus 0.5 and 0.75 against 8/8 (Fisher p 0.007, 0.0014); at 0.9 the contrast is unmeasured
(4/8 vs 8/8; accuracy -0.046). Three times the training leaves |.| at 3/8. The ring accuracy gap is small (-0.03; |.|
reaches 0.95-0.97, above the map-free floors -- n-gram 0.554, retrace 0.860 (post hoc) -- and the one-layer index model's
0.72-0.73 at p_plus 0.5), so the ring evidence is
the SOLVED criterion; the accuracy evidence is the torus. Post hoc, at a loss threshold of 0.1 the p_plus 0.5 contrast no
longer fires (4/8 vs 8/8, Fisher p 0.077); every other sign contrast holds at thresholds 0.02 and 0.1 and under accuracy >=
0.99 (appendix C).

The codes show what is missing (Fig. 1, post hoc): signed models cancel opposite moves mod 2 pi in 64-98% of channels (arm
means), |.| and softplus models in 3-11%, including the runs that solve. Observation-step-free versions fail the same way:
ring 0/8 vs 8/8 (two batches), torus 0/8 vs 8/8 (-0.120) and 1/8 vs 8/8 (-0.104).

*Recipe and scale (registered, fresh seeds; appendix C).* On the observation-step-free ring the cost holds with weight
decay 0 (|.| 0/8 vs 8/8, -0.127) and at a third of the learning rate (0/8 vs 6/8, Fisher p 0.007), and replicates at the
default (1/8 vs 8/8). At three times the learning rate it shrinks and is unmeasured (4/8 vs 8/8, Fisher p 0.077; accuracy
0.987 vs 1.000). A larger torus model (2 layers, d 256, 8x the parameters) keeps it (2/8 vs 8/8, Fisher p 0.007; accuracy
0.967 vs 0.999, p 0.0005), but the accuracy gap is a quarter of the one-layer gap (an out-of-batch comparison), 6/8 |.| runs
were still descending at 900 epochs, and the SOLVED contrast does not survive a 0.1 loss threshold (5/8 vs 8/8; post hoc):
at scale part of the cost may be speed. In every cell the |.| runs that solve do so without the wrapped code (cancelling
share 0.04-0.13 vs 0.43-1.00 signed).

### 4.2 The |.| solution exists and survives training

We convert each solved signed observation-step-free model exactly into |.| form (P1(a); the balanced construction of
appendix D). Every angle is unchanged mod 2 pi (error <= 1.4e-6 rad) and frozen accuracy equals the source's on 8/8 seeds;
on the ring the weights stay within ~5x of the signed model's. A naive construction that keeps the signed latents is also
exact but ill-conditioned, and Adam loses it within ~10 epochs: the test must be posed at a trained scale.

On the ring, the constructed model trained 300 more epochs stays solved on 8/8 seeds and never leaves the solved region;
weighted by how strongly attention reads each channel, it keeps its cancelling code (99% of used channels at construction,
98% at the end). On the torus it is knocked out early (peak loss 0.11-0.21 vs <= 0.017 for the signed control; 6/8 leave the
solved region) and re-finds the solution (8/8 SOLVED, 1.000), at lr 1e-4 (fixed before launch: in a pilot, lr 1e-3 knocked
out the signed control); at 1e-3 (descriptive) it re-solves 6/8. From scratch the class solves 0/8 on both tasks. The
solution is stable in task terms on the ring and re-found from nearby on the torus.

### 4.3 The mechanism on the ring: a route through zero

We vary only the initial move angles of observation-step-free ring models; every other weight is identical at a seed (Table
1).

| initial move angles | signed | \|.\| |
|---|---|---|
| default (small, signs independent) | 8/8 | 0/8 |
| \|default\|: both moves rotate the same way (the \|.\| start) | **7/8** | 0/8 |
| large, summing to less than pi | -- | 2/8 |
| large, summing to more than pi | 8/8 | **8/8** |

*Table 1. SOLVED, ring p_plus 0.5, 300 epochs, n = 8 per cell. Post hoc, each contrast below reads the same at loss
thresholds 0.02 and 0.1 and under accuracy >= 0.99 (appendix C).*

Four registered contrasts read Table 1. (i) From the same start the sign matters: 0/8 vs 7/8 (Fisher p 0.0014; accuracy
-0.092 [-0.135, -0.045]). (ii) A start past pi rescues |.|: 8/8 vs 0/8 (+0.109 [+0.085, +0.135], p 0.0002). (iii) The sum,
not only the size: a large start with sums below pi solves 2/8 (Fisher p 0.007 vs 8/8). (iv) From past pi the sign makes no
difference (8/8 vs 8/8). Post hoc (Fig. 2): signed models from the default or the same-direction start end with 98-99% of
channels opposite-signed (from 0% at the same-direction start) and none of their cancelling channels wrap; from past pi both
signs cancel by wrapping (88% and 100% of cancelling channels).

**The signed escape is collective (registered; fresh seeds).** From the same-direction start, freezing the two move latents
(W_out and omega free per channel) blocks the escape: 1/8 vs 7/8 SOLVED (Fisher p 0.010; -0.075 [-0.130, -0.020]); 15% of
channels end opposite-signed against 82% with free latents, which take the latent route of Sec. 3 that |.| lacks. From the
default start freezing has no detected effect (7/8 vs 8/8).

**What fits loosely (post hoc; appendix C).** Rescued |.| models cancel in only 23% of channels, and the one-channel flow
does not predict which (Cohen's kappa +0.09). Many small-start |.| steps collapse to zero (24-42% of ring move steps), but
the final gradient points into that wall for only about half of them (mixed, by a rule fixed in advance). A transformer-free
toy reproduces the |.| failure but not the signed escape (appendix I).

So the barrier removes a collective route through zero (registered); why |.| fails from a small start (an unreachable wrap,
P3; the wall, P4; both; or neither, Sec. 4.5) is not separated.

### 4.4 The start lever does not transfer to the torus

On the torus (observation-step-free, 4 moves), the past-pi start does not rescue |.|: 0/8 vs 1/8 from the default start
(+0.005 [-0.060, +0.070]; no effect detected) and slows the signed model to 2/8 against 8/8 (secondary); 5/8 runs per
large-start arm are still descending at 900 epochs. The default-start sign cost replicates (1/8 vs 8/8, Fisher p 0.0014). An
untested account: with 64 cells per axis and two inverse pairs per channel, a random large-angle code is far from any map.
The start lever is a ring result, not a design rule.

### 4.5 Does the cost follow the relative sign? (registered)

By Lemma 1(ii) a |.| step represents a map whose moves step the same way without wrapping or crossing zero. Does the
learning cost then disappear? Observation-step-free ring models as in Sec. 4.3, n = 8 fresh seeds per cell; moves +1/-1 (an
inverse pair) vs +1/+2 (map code a(+2) = 2 a(+1), sign-coherent).

| moves | signed | \|.\| | difference (95% CI) | label |
|---|---|---|---|---|
| +1/-1 | 1.000, 8/8 | 0.931, 1/8 | -0.069 [-0.095, -0.045], Fisher p 0.0014 | signed learns better |
| +1/+2 | 0.947, 5/8 | 0.883, 2/8 | -0.064 [-0.200, +0.075], Fisher p 0.31 | unmeasured |

*Table 2. Accuracy (T = 128) and SOLVED; setup as Table 1, n = 8 per cell.*

The cost replicates on the inverse pair (a third observation-step-free ring batch). +1/+2 needs integration (index RoPE
0.649, 0/8; floor ~0.50), and the signed model learns it with a sign-coherent code (0.92 of channels end with both move
angles of one sign, against 0.01 on +1/-1), as Lemma 1 allows. But the registered prediction is not confirmed: |.| solves
+1/+2 on only 2/8, no more than +1/-1 (1/8; Fisher p 1.00), although its code there needs no zero crossing; the comparison
is unmeasured because +1/+2, whose revisits are all long-gap laps, is harder for signed steps too (5/8). The gap's point
estimates are about equal (Table 2; on laps, the stratum both tasks share, |.| scores 0.893 and 0.884: a post hoc reading of
a declared secondary). Nothing here suggests the cost needs an inverse pair, and the barrier account does not explain |.|'s
difficulty on +1/+2: Lemma 1 stands as a statement about representation only.

### 4.6 Published monotone methods

Faithful CARoPE (a per-head rate in (0, 1), started at index RoPE) cannot represent cancellation (*P6, proved*: every token
advances each channel by (0, 1], so the shortest return advances it by (0, 4] < 2 pi). It scores 0.667 on the torus, below
index RoPE's 0.722 (p 0.040), and 0.741 vs 0.730 on the ring (unmeasured). CoPE can represent ring cancellation (a periodic
position table read at a gated count; constructed), contrary to MapFormer's statement that it "cannot represent an action
and its inverse", but solves 0/8 (0.887 vs signed 1.000, p 0.0002; 1 of 32 runs at the paper's initialisation over three
ring batches, pooled post hoc), while its gated count beats the index by +0.157. On the line CoPE still fails at the
registered criterion (1/16 vs 0/16 on the ring) while signed steps solve it 8/8 and the index does not (0.772): removing the
wrap does not make CoPE learnable. At a looser threshold half the line runs pass (post hoc; appendix C), so whether the wrap
adds to CoPE's difficulty is unmeasured. CoPE's obstacle is unidentified; the precision its count needs is a plausible
account (appendix H).

## 5. Related work

**Integrators and position.** TEM-t (Whittington et al. 2022) puts an action-driven position code into a transformer;
MapFormer (Rambaud et al.) makes it a path-integrated rotary phase.

**Content-dependent rotary position.** Sec. 1's methods are Selective RoPE (Movahedi et al.), Mamba-3 (Lahoti et al.),
CARoPE (Veisi et al.) and CoPE (Golovneva et al.). GRAPE (Zhang et al.) unifies multiplicative and additive slots; its
non-negative content phase is a real-slot bias, as in FoX (Lin et al.).

**Sign and state tracking.** Khavari et al. (2025) add that in diagonal SSMs input dependence and negative eigenvalues must
occur in the same layer; none of these theorems covers rotations by non-negative angles. We give the same advice, allow a
sign, for a different reason.

**Sign-constrained parameterisations and initialisation.** A reparameterised weight cannot change sign under gradient flow
(Woodworth et al. 2020; Amid and Warmuth 2020: w = |u| is gradient descent on the non-negative orthant). The LRU (Orvieto et
al. 2023) parameterises phases as positive and needs them initialised near zero to learn PathX. Dale's-law networks raise
the same question, constraint or starting point (Cornford et al. 2021; Li et al. 2023 find the spectrum at initialisation
matters more than the sign constraint, though both contribute). To our knowledge neither the relative-sign gauge nor the
loss of the through-zero route for inverse pairs has been stated; we claim their conjunction and the measurements, not any
single ingredient.

**Depth.** Shallow transformers learn automaton shortcuts unreliably (Liu et al. 2023; Hahn and Rofin 2024), while Selective
RoPE and PaTH (Yang et al.) solve parity or Flip-Flop in one layer; appendix K compares one path layer with deeper index
attention on our torus.

## 6. Limitations

"Does not learn" is scoped to our budgets; 3x the ring budget did not help, and a one-factor recipe check (Sec. 4.1) holds the cost without weight decay and at a third of the learning rate but finds it
shrinking at three times the learning rate (unmeasured) and, in accuracy, in a 2-layer d 256 torus model whose |.| runs
were mostly still descending; optimisers, larger learning rates and longer budgets at scale were not tested.
Small models (one layer, 2 heads, d = 128, a rank-4 latent shared across heads), synthetic tasks, n = 8, no recurrent or SSM
baselines, no exp(.)-type (LRU) parameterisation. On the ring the evidence is mainly a loss criterion, and at p_plus 0.5 it
does not survive a looser threshold (0.1; post hoc). On the torus the barrier class has less latent slack (the wrapped code
also needs the 2 pi k / omega offsets), and per-head rank is known to decide torus learnability for signed steps; this is
uncontrolled. Softplus carries an initialisation confound and untested representability (the existence results are for |.|).
The mechanism is shown on the ring only, with its |.| side unresolved; on the torus the large start also slows the signed
model. Lemma 1's learnability prediction is untested at our power, and the barrier account does not explain the |.|
difficulty on +1/+2 (Sec. 4.5). CARoPE and CoPE were tested at one layer, CoPE on the ring and line only. Natural text is
not studied.

## References

Titles and author lists checked against arXiv, OpenReview, Crossref or ACL Anthology metadata on 2026-10-15
(`papers/new_2026-10-15/`); venues marked * are taken from the literature note and were not re-checked.

- Amid, E. and Warmuth, M. K. Reparameterizing mirror descent as gradient descent. NeurIPS 2020*; arXiv:2002.10487.
- Cornford, J., Kalajdzievski, D., Leite, M., Lamarquette, A., Kullmann, D. M. and Richards, B. A. Learning to live with
  Dale's principle: ANNs with separate excitatory and inhibitory units. ICLR 2021.
- Golovneva, O., Wang, T., Weston, J. and Sukhbaatar, S. Contextual position encoding: learning to count what's important
  (CoPE). arXiv:2405.18719, 2024.
- Grazzi, R., Siems, J., Zela, A., Franke, J. K. H., Hutter, F. and Pontil, M. Unlocking state-tracking in linear RNNs
  through negative eigenvalues. ICLR 2025*; arXiv:2411.12537.
- Hahn, M. and Rofin, M. Why are sensitive functions hard for transformers? ACL 2024 (long papers); arXiv:2402.09963.
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
- Veisi, A., Fartoot, D. and Amirzadeh, H. Context-aware rotary position embedding (CARoPE). arXiv:2507.23083, 2025.
- Whittington, J. C. R., Warren, J. and Behrens, T. E. J. Relating transformers to models and neural representations of the
  hippocampal formation (TEM-t). ICLR 2022; arXiv:2112.04035.
- Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E., Savarese, P., Golan, I., Soudry, D. and Srebro, N. Kernel and
  rich regimes in overparametrized models. COLT 2020*; arXiv:2002.09277.
- Yang, S., Shen, Y., Wen, K., Tan, S., Mishra, M., Ren, L., Panda, R. and Kim, Y. PaTH attention: position encoding via
  accumulating Householder transformations. NeurIPS 2025; arXiv:2505.16381.
- Zhang, Y., Chen, Z., Liu, Y., Qin, Z., Yuan, H., Xu, K., Yuan, Y., Gu, Q. and Yao, A. C.-C. Group representational
  position encoding (GRAPE). ICLR 2026; arXiv:2512.07805.

## Figures and tables

- Table 1 (Sec. 4.3): start x step parameterisation. Table 2 (Sec. 4.5): moves +1/-1 vs +1/+2. Appendix tables C1-C12
  (and C9b) are in `APPENDIX_draft5.md`.
- **Fig. 1** (`docs/paper/figs/fig_wrap_share.png`; Sec. 4.1; post hoc readout of registered checkpoints). Per run, the share
  of rotary channels in which opposite moves cancel mod 2 pi (within 0.1 rad; wrap-invariant), ring p_plus 0.5 / 0.75 / 0.9
  and torus; filled = SOLVED. Signed means 0.79 / 0.78 / 0.64 / 0.98; |.| 0.06 / 0.03 / 0.06 / 0.11; softplus (torus) 0.03.
- **Fig. 2** (`docs/paper/figs/fig_mechanism.png`; Sec. 4.3; post hoc). Move angles (a(+1), a(-1)) of every rotary channel at
  the start (grey) and after training (dots), 8 ring runs per panel. Cancelling lines: blue a(-1) = -a(+1) (through zero;
  needs opposite signs), red a(+1) + a(-1) = +-2 pi (wrapping). Shading: outside the non-negative quadrant (the barrier
  side, closed to |.|); dotted: sum = pi.
  Panels: (a) signed step, same-direction start (7/8 SOLVED; cancelling share 0.06 -> 0.89, almost all through zero);
  (b) |.| step, same start (0/8; 0.04, of which 0.035 is collapsed same-sign channels near the origin, drawn black, not
  cancellation); (c) |.| step, large start with sums below pi (2/8; 0.05, by wrapping); (d) |.| step, start past pi (8/8;
  0.07 -> 0.23, by wrapping). Whole turns removed toward zero.
- **Fig. C1** (appendix C; `docs/paper/figs/fig_ring_strata.png`). Ring revisit accuracy by gap and wrap-only stratum (post hoc).
- **Fig. K1** (appendix K; `docs/paper/figs/fig_factorial_strata.png`). Torus T = 1024, accuracy by revisit stratum (within 128
  moves, >= 128 moves, wrap-only) for path 1 layer and index 1 layer, 4 layers, looped x4 (n = 4; means, range). Dashed line:
  the best-constant (predict-blank) floor, averaged over strata and runs (overall 0.507); the n-gram floor 0.576 is not drawn.
