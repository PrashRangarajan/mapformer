# A barrier at zero: rotary phase steps parameterised to be non-negative can represent a map but rarely learn one (one-layer models, synthetic maps)

*Draft 4, 2026-10-15. Changes and sources: `docs/paper/DRAFT4_changes.md`. Delete this line before submission.*

## Abstract

Content-dependent rotary encodings accumulate a phase theta_t = omega * sum_{u <= t} Delta_u. In the rotary slot the sign of the step
carries no representational content: Delta mod (2 pi / omega) is a non-negative step defining the same model (the
cyclic-group fact behind Grazzi et al.'s and Mamba-3's constructions). We ask whether parameterisations with a barrier at
zero, |s| and softplus(s), cost learnability. In one-layer rotary models trained and tested at equal length on synthetic
navigation maps (a 2D torus, a 1D ring), barrier steps solve the map on 0-4 of 8 seeds per batch (0-2 wherever a registered
contrast fires) against 8 of 8 for signed steps. Exact non-negative solutions exist at a trained weight scale and training
keeps (ring) or re-finds (torus) them; from scratch the class solves 0-1 of 8. A per-channel reflection identity shows that
the barrier removes only relative signs within a channel: for an inverse pair, the through-zero route to cancellation, not
the function; whether the cost requires an inverse pair is unmeasured (moves +1/+2 are harder for both classes). On the
ring, signed steps leave the barrier's start through zero by a collective move of the move latents (7/8; 1/8 with them
frozen); barrier steps learn reliably only from a start whose opposite angles sum past pi (8/8 vs 0/8),
which does not help on the torus. Negatives hold within our budgets (300-900 epochs, one recipe per task).

## 1. Introduction

A rotary position code is an integrator. Index RoPE adds a constant per token; content-dependent variants let each token
set its own step, non-negative by construction (CARoPE's rate in (0, 1); in a different slot, CoPE's gated counts) or of
either sign (MapFormer's path-integrated phase, Selective RoPE, Mamba-3).

For real-valued recurrences the sign decides what can be represented: transitions with real non-negative eigenvalues cannot
track parity or count modulo m (Sarrof et al. 2024; Grazzi et al. 2025). Those theorems do not reach the rotary slot: a
rotation by a non-negative angle has complex eigenvalues, and Grazzi et al. and Mamba-3 build cyclic groups from such
rotations. A non-negative step *value* is harmless there; a non-negative *parameterisation* may not be: |s| folds and
softplus(s) flattens the step at zero, so training cannot move a step through zero. For learning maps, where opposite moves
must cancel, this barrier matters within our budgets. We identify a route to the map that signed steps use and the barrier
removes; we do not fully explain why barrier steps fail from small starts, nor why they also struggle on a ring walked
by +1/+2 moves, where no step need cross zero (Sec. 4.5).

**Contributions.**
1. *A learnability cost at matched length* on a torus and a ring; barrier codes do not cancel opposite moves (Sec. 4.1).
2. *Existence.* An exact non-negative solution at a trained scale is held (ring) or re-found (torus) under training; from
   scratch the class solves 0-1/8 (Sec. 4.2).
3. *What the barrier removes* (Lemma 1, a one-line identity): only relative signs within a channel; for an inverse pair, a
   route, not a function. A registered test of its learnability corollary (moves stepping the same way, Sec. 4.5)
   replicates the cost on an inverse pair but cannot read the same-direction case: that task is harder for signed steps too.
4. *A route, on the ring*: signed steps leave the barrier's start through zero, collectively through their shared latents;
   barrier steps learn reliably only from a start whose opposite angles sum past pi, a start that fails on the torus
   (Secs. 4.3-4.4).
5. *Published monotone methods* (faithful CARoPE, CoPE) fail these maps too, CoPE although it can represent the ring map
   (Sec. 4.6).

## 2. Setup and terms

**Model.** MapFormer-WM (MapWM; Rambaud et al.): one layer, 2 heads, d = 128. Query and key are rotated per rotary channel
c (a head and a 2D block) by theta_{t,c} = omega_c * sum_{u <= t} Delta_{c,u}, with Delta_u = W_out W_in x_u through a rank-4
latent shared across heads (MapFormer's is per head). Theta_{c,x} = omega_c Delta_{c,x} is token x's angle in channel c.
**Step parameterisations.** Signed: Delta = s = W_out W_in x. |.|: Delta = |s| (the sign removed, nothing else). Softplus:
Delta = softplus(s), which starts every token at the same step (softplus(0) = 0.69), an initialisation confound |.| does not
have. |.| and softplus are *barrier parameterisations*: no gradient step carries Delta through zero. *Observation-step-free
(AO)* versions give steps to the move tokens only. Index RoPE uses the canonical ladder.
**Tasks.** A 2D torus walk (64 x 64, T = 1024 moves interleaved with observations, 900 epochs): predict the observation at
revisited cells, with the observation map redrawn for every evaluation walk (accuracy is transfer). A 1D ring (32 cells,
T = 128, 300 epochs) with forward-step probability p_plus (0.5 is balanced). A line: a 256-cell ring the walk never wraps.
**Terms.** The +1 and -1 moves form an *inverse pair*. A channel *cancels* it if Theta_{c,+} + Theta_{c,-} = 0 mod 2 pi
(Theta_{c,+} != 0 mod 2 pi): *through zero* if the sum is 0, *by wrapping* if a non-zero multiple of 2 pi. A *wrap-only*
revisit has all earlier visits a whole lap away. *SOLVED*: training loss over the final 5% of training < 0.05. Floors
(predict-blank, best action-stream n-gram) are given beside accuracies.
**Protocol.** Every contrast was pre-registered with its decision rule, trained and tested at the same length, n = 8 seeds
per arm unless stated. Every main-text batch except the torus sign batch of Sec. 4.1 (pre-launch gate by the authors,
independent audit after the run) was code-checked by an independent agent before any result was read, and any pilot used
seeds outside the batch. In the appendix, navigation in words reused 2 pilot seeds of 8 per arm (we quote the 6 fresh ones)
and the Dyck batch predates independent checks.
"Unmeasured" means below the registered threshold; "does not learn" means within the stated budget (AdamW, warm-up, cosine).

## 3. Theory

**Three integrators.** Howard et al. (2014) drive integrators by a rate: constant gives elapsed time, a speed path length
(a clock), a signed velocity position (a map). Per token the rate is Delta_t; with Phi_t = sum_{u <= t} Delta_u, channel c
of the score of query t and key t' is Re(conj(q~_c) k~_c exp(-i omega_c (Phi_t - Phi_{t'}))), with q~_c, k~_c channel c of
query and key written as complex numbers: Howard's propagator with an imaginary rate constant (his s = i omega; appendix E;
the complex-SSM identity is Mamba-3's Prop. 3). Index RoPE is the constant case, CARoPE and CoPE the non-negative case,
MapFormer, Selective RoPE and Mamba-3 the signed case. In the real (decay) slot nothing wraps; in the rotary slot the last
two cases represent the same codes.

**Representability.** Per channel a non-negative step reaches any angle mod 2 pi (-1 = m - 1 in Z_m). Our step is a shared
rank-r map, so we need a scoped version. *P1(a) (proved).* If at most r tokens have non-zero steps, with embeddings
independent of the others', a |.| rank-r model reproduces a signed angle table exactly mod 2 pi. A W_in that annihilates the
observation embeddings leaves the moves as the only stepping tokens (2 on the ring, 4 on the torus; r = 4), so P1(a) covers
every |.| arm here, with or without observation steps (built in the real modules on the ring: observation steps <= 1.7e-6).

**Lemma 1 (relative-sign gauge).** Negate every token's step in a set of channels (equivalently omega_c, or row c of W_out)
and the second coordinate of each such channel's query and key projections (weights and biases, every layer). Every
attention score is unchanged. *Proof:* with J = diag(1, -1), J R(phi) J = R(-phi). (Checked bitwise on trained
checkpoints, a random two-layer model and the MapFormer-EM variant.) The identity is elementary; its consequences are what
we use. (i) As real tables with the same latent, i.e. without wrapping, a |.| model realises a signed table iff each row is
sign-coherent (all tokens stepping in a channel step the same way): |.| fixes no channel's direction, only the *relative*
signs within a channel. (ii) So the barrier excludes no map code where one with sign-coherent rows exists: one stepping
token (parity, mod-m counting) or two moves that step the same way (+1 and +2). That it then costs no learnability is a
prediction, not a consequence; Sec. 4.5 tests it on +1/+2 and does not confirm it. (iii) For an inverse pair, a channel cancelling
through zero has opposite signs and is excluded; the same function remains as the sign-coherent wrapped code (a, 2 pi - a)
(P1). The barrier removes a route to cancellation, not cancellation.

**One channel's geometry (proved; appendix E).** Let sigma = Theta_{c,+} + Theta_{c,-}. *P3 (cut circle):* both classes
attain the same losses on the angle torus. From a start in the open non-negative quadrant with sigma < 2 pi, the nearest
cancelling point (Sec. 2) is min(sigma, 2 pi - sigma) / sqrt 2 away for signed steps and (2 pi - sigma) / sqrt 2 for |.|;
these coincide iff sigma >= pi, and every |.| path to cancellation from sigma < pi crosses sigma = pi. *P4 (sticky wall):*
with a = omega |s|, dL/ds = sign(s) dL/da, so where the loss wants an angle smaller, s = 0 is a strict local minimum and
gradient descent chatters within about one step of it (cf. Woodworth et al. 2020; Amid and Warmuth 2020). Softplus has no
fold, but its slope vanishes toward zero.

**The collective route (hypothesis).** P3-P4 are geometry, not dynamics. Gradient flow on the exact one-channel retrieval
potential (at a fresh model's small attention scale; appendix E) reproduces neither the signed escape (12% of channels cross
zero) nor a sharp past-pi rescue (only channels within ~0.5 rad of 2 pi are captured: 28% of a U(1.7, 4.5) start). But for
signed steps, with z(+1), z(-1) the two moves' latents W_in x, the cancelling set {z(-1) = -z(+1)} is a linear subspace
of the move latents, so one latent move cancels every channel at once; under |.| the same move gives a(-1) = a(+1), a
clock. Sec. 4.3 tests this.

## 4. Results

### 4.1 Barrier parameterisations do not learn the map

On the torus at the training length (T = 1024), |.| costs -0.177 in accuracy (0.821 vs 0.998; permutation p 0.0002) and
solves 0/8 against 8/8, with 3/8 |.| runs still descending at 900 epochs; softplus costs -0.218 (1/8; initialisation
confound). On the ring, |.| solves 2/8 and 1/8 at p_plus 0.5 and 0.75 against 8/8 (Fisher p 0.007, 0.0014); at 0.9 the
contrast is unmeasured (4/8 vs 8/8; accuracy -0.046). Three times the training leaves |.| at 3/8. The ring accuracy gap is
small (-0.03; |.| reaches 0.95-0.97, above the 0.554 floor and the one-layer index model's 0.72-0.73 at p_plus 0.5), so the
ring evidence is the SOLVED criterion; the accuracy evidence is the torus. Post hoc, that criterion matters at p_plus 0.5:
with a loss threshold of 0.1 instead of 0.05, |.| solves 4/8 vs 8/8 (Fisher p 0.077), while at 0.02 or under an accuracy
criterion (>= 0.99) it stays at 2/8; at p_plus 0.75 (1-2/8 vs 8/8) and in every observation-step-free and torus contrast
the reading holds under all four criteria (appendix C). Also post hoc, unsolved ring models lose accuracy with the gap
since the last visit (appendix Fig. C1; sign-flip p 0.0078 in each observation-step-free batch); signed models are at
1.000 in every stratum.

The codes show what is missing (Fig. 1, post hoc): signed models cancel opposite moves mod 2 pi in 64-98% of channels (group
means), |.| and softplus models in 3-11%, including the runs that solve. Observation-step-free versions fail the same way:
ring 0/8 vs 8/8 (two batches), torus 0/8 vs 8/8 (-0.120) and 1/8 vs 8/8 (-0.104).

### 4.2 The non-negative solution exists and survives training

We convert each solved signed observation-step-free model exactly into non-negative form (P1(a)): move latents re-solved
orthogonal, each channel's steps set to (omega Delta mod 2 pi) / omega' with omega' = max(|omega|, 1). Every angle is
unchanged mod 2 pi (error <= 1.4e-6 rad) and frozen accuracy equals the source's on 8/8 seeds; on the ring the weights stay
within ~5x of the signed model's. Keeping the signed latents is also exact but ill-conditioned (cosine -0.987 to -0.998),
and Adam loses it within ~10 epochs: the test must be posed at a trained scale.

On the ring, the constructed model trained 300 more epochs stays solved on 8/8 seeds and never leaves the solved region;
weighted by how strongly attention reads each channel, it keeps its cancelling code (99% of used channels at construction,
98% at the end). On the torus it is knocked out early (peak loss 0.11-0.21 vs <= 0.017 for the signed control; 6/8 leave
the solved region) and re-finds the solution (8/8 SOLVED, 1.000), at lr 1e-4, fixed before launch because in a pilot lr
1e-3 knocked out even the signed control; at 1e-3 (descriptive) it re-solves 6/8 (0.988). From scratch the class solves 0/8
on both tasks. The solution is stable in task terms on the ring and re-found from nearby on the torus.

### 4.3 The mechanism on the ring: a route through zero

We vary only the initial move angles of observation-step-free ring models; every other weight is identical at a seed
(Table 1).

| initial move angles | signed | \|.\| |
|---|---|---|
| default (small, signs independent) | 8/8 | 0/8 |
| \|default\|: both moves rotate the same way (the \|.\| start) | **7/8** | 0/8 |
| large, the two angles summing to less than pi | -- | 2/8 |
| large, summing to more than pi | 8/8 | **8/8** |

*Table 1. SOLVED, ring p_plus 0.5, 300 epochs, n = 8 per cell. Post hoc, each contrast below reads the same at loss
thresholds 0.02 and 0.1 and under accuracy >= 0.99 (appendix C).*

Four registered contrasts read Table 1. (i) From the same start the sign matters: 0/8 vs 7/8 (Fisher p 0.0014; accuracy
-0.092 [-0.135, -0.045]). (ii) A start past pi rescues |.|: 8/8 vs 0/8 (+0.109 [+0.085, +0.135], p 0.0002). (iii) The sum,
not only the size: a large start with sums below pi solves 2/8 (Fisher p 0.007 vs 8/8). (iv) From past pi the sign makes no
difference (8/8 vs 8/8). Post hoc (Fig. 2): signed models from the default or the same-direction start end with 98-99% of
channels opposite-signed (from 0% at the same-direction start) and none of their cancelling channels wrap; from past pi
both signs cancel by wrapping (88% and 100% of cancelling channels).

**The signed escape is collective (registered; fresh seeds).** From the same-direction start, freezing the two move latents
(W_out and omega free per channel) blocks the escape: 1/8 vs 7/8 SOLVED (Fisher p 0.010; -0.075 [-0.130, -0.020]); 15% of
channels end opposite-signed against 82% with free latents. From the default start freezing has no detected effect (7/8 vs
8/8). The signed model moves one move's latent toward the negative of the other's, the route of Sec. 3 that |.| lacks.

**What fits loosely (post hoc, eval-only).**
- *The |.| start is small in most channels, not all:* per-channel sums span 0.01-6.58 rad (5% start past pi). Of the
  channels whose step does not collapse to zero, the 20 (over 8 runs) that start with sums between pi and 3 pi / 2 end with
  none cancelling: being past pi is not sufficient per channel.
- *Rescued models use the wrap sparingly:* from past pi, |.| models solve with 23% of channels cancelling (signed 34%; 7% at
  the start by chance); capture falls gently with distance from 2 pi (0.34 within 0.25 rad, 0.08 at 2-3 rad). The share is
  close to the one-channel flow's 28% (Sec. 3), but channel by channel the flow does not predict which channels cancel: run from
  each channel's own start it predicts 33%, agrees with the trained models barely above chance (Cohen's kappa +0.09), and
  captures 95% of the channels starting within 0.5 rad of 2 pi where the models cancel 28%; 59% of the models' cancelling
  channels are ones it does not capture. So the one-channel flow does not explain the rescue; whether the rescue is
  collective is untested. What the other 77% do is unknown.
- *A competing account, the wall:* 44% of channels in |.| models from the same-direction start end with the forward-move
  angle within 0.05 rad of zero, against 17% for signed models from that start and 0.2% and 1.4% for |.| from the large
  starts. This would fit P4 if the loss pushed those steps toward zero. It does not clearly do so. Counted per move step,
  collapse is common to every |.| arm from a small start (24-42% on the ring, 60-70% on the torus, softplus included) and
  smaller in signed controls from the default start (10-16%, 34-40%). It is partly inherited: 77-81% of the |.| steps that start within 0.05 rad of zero
  end there (signed 29-35%). Attention reads the collapsed |.| channels (usage weight 0.39-1.1 times the others'). But at the end
  of training the loss gradient points into the wall for only 51-57% of the collapsed steps (eval mode), which a rule
  fixed before the check reads as mixed, not as P4.
- *Start size:* a start nearer the wrap also moves 13-22x faster per Adam step; the sub-pi control (~5x) separates sum from
  size only partly.

So the signed step has a collective route through zero that the barrier removes (registered); why the barrier step fails
from a small start (an unreachable wrap, P3; the wall, P4; both; or neither, since |.| also struggles where no zero
crossing is needed, Sec. 4.5) is not separated, and neither the one-channel flow nor the end-of-training gradient decides it.

### 4.4 The start lever does not transfer to the torus

On the torus (observation-step-free, 4 moves), the past-pi start does not rescue |.|: 0/8 vs 1/8 from the default start
(+0.005 [-0.060, +0.070]; no effect detected). It slows the signed model to 2/8 against 8/8 (secondary), with 5/8 runs per
large-start arm still descending at 900 epochs. The default-start sign cost replicates (1/8 vs 8/8, Fisher p 0.0014). An
untested account: with 64 cells per axis and two inverse pairs sharing every channel, a random large-angle code is far from
any map for either sign. The start lever is a ring result, not a design rule.

### 4.5 Does the cost follow the relative sign? (registered)

Lemma 1(ii) says a |.| step can represent a map whose moves step the same way without wrapping or crossing zero. We asked
whether the learning cost then disappears. Ring of 32 cells, observation-step-free signed and |.| models, the Sec. 4.3
recipe, n = 8 fresh seeds per cell; moves +1/-1 (an inverse pair) vs +1/+2 (map code a(+2) = 2 a(+1), sign-coherent).

| moves | signed | \|.\| | difference (95% CI) | label |
|---|---|---|---|---|
| +1/-1 | 1.000, 8/8 | 0.931, 1/8 | -0.069 [-0.095, -0.045], Fisher p 0.0014 | signed learns better |
| +1/+2 | 0.947, 5/8 | 0.883, 2/8 | -0.064 [-0.200, +0.075], Fisher p 0.31 | unmeasured |

*Table 2. Accuracy at T = 128 and SOLVED, ring 32 cells, observation-step-free models, 300 epochs, n = 8 per cell.*

The cost replicates on the inverse pair (a third observation-step-free ring batch). +1/+2 needs integration (index RoPE 0.649, 0/8; floor
~0.50), and the signed model learns it with a sign-coherent code (0.92 of channels end with both move angles of one sign,
against 0.01 on +1/-1): what signed steps learn follows the moves' relative sign, as Lemma 1 allows. But the registered
prediction is not confirmed. |.| still solves +1/+2 on only 2/8, no more than +1/-1 (1/8), although its code there needs no
zero crossing (|.| on +1/+2 vs +1/-1: no difference detected, Fisher p 1.00); the comparison is unmeasured because +1/+2
is harder for signed steps too (5/8), its revisits being all long-gap laps. Descriptively (post hoc reading of a declared
secondary): on laps, the stratum both tasks share, |.| scores 0.893 (+1/-1) and 0.884 (+1/+2), and the point estimates of
the gap are about equal (-0.069, -0.064). Nothing here suggests the cost needs an inverse pair, and the barrier-at-zero account
does not explain |.|'s difficulty on +1/+2. Lemma 1 stands as a statement about representation only.

### 4.6 Published monotone methods

Implemented faithfully, neither learns these maps at one layer. CARoPE (a per-head rate in (0, 1), initialised at index
RoPE) cannot represent cancellation (*P6, proved*: every token advances each channel by (0, 1], so the shortest return
advances it by (0, 4] < 2 pi). It scores 0.667 on the torus, below index RoPE's 0.722 (p 0.040), and 0.741 vs 0.730 on the
ring (unmeasured). CoPE can represent ring cancellation (a periodic position table read at a gated count; constructed),
contrary to MapFormer's statement that it "cannot represent an action and its inverse", but solves 0/8 (0.887 vs signed
1.000, p 0.0002; 1 of 32 runs at the paper's initialisation over three ring batches, pooled post hoc), while its gated count beats the index by +0.157. On
the line CoPE still fails at the registered criterion (1/16 vs 0/16 on the ring) while signed steps solve it 8/8 and the
index does not (0.772), so removing the wrap does not make CoPE learnable. Post hoc, at a loss threshold of 0.1 half the
line runs pass (8/16 vs 2/16 on the ring, Fisher p 0.054; accuracy >= 0.99: 7/16 vs 2/16), so whether the wrap adds to
CoPE's difficulty is unmeasured. CoPE's obstacle is unidentified; the precision its count needs is a plausible account
(appendix H).

## 5. Related work

**Integrators and position.** Howard et al. (2014) drive leaky integrators by a constant, speed or signed-velocity rate,
with the decay real and the rate given. GridPE (Li et al. 2024) builds a position embedding for given coordinates from
grid-cell velocity-controlled-oscillator models and Fourier features, recovering RoPE in one dimension. TEM and TEM-t
(Whittington et al. 2020, 2022) put an action-driven recurrent position code into a memory model and a transformer;
MapFormer (Rambaud et al.) makes it a path-integrated rotary phase.

**Content-dependent rotary position.** Signed: Selective RoPE (Movahedi et al.), Mamba-3 (Lahoti et al.), MapFormer.
Non-negative: CARoPE and, as a gated count read through a table, CoPE (Golovneva et al.). GRAPE (Zhang et al.) unifies
multiplicative and additive slots; its non-negative content phase is an additive bias in the real slot, as in FoX.

**Sign and state tracking.** Sarrof et al. (2024) and Grazzi et al. (2025) prove that recurrences with real non-negative
eigenvalues cannot track parity or count modulo m (Khavari et al. 2025: in diagonal SSMs, input dependence and negative
eigenvalues must occur in the same layer). Rotations by non-negative angles are not covered: Grazzi et al. implement Z_m
with rotations by 2 pi w / m >= 0, and Mamba-3 writes parity as R(pi x_t). We give the same advice, allow a sign, for a
different reason.

**Sign-constrained parameterisations and initialisation.** A reparameterised weight cannot change sign under gradient flow
(Woodworth et al. 2020; Amid and Warmuth 2020: w = |u| is gradient descent on the non-negative orthant). The LRU (Orvieto
et al. 2023) parameterises phases as positive and needs them initialised near zero to learn PathX. For networks obeying Dale's
law the same question, constraint or starting point, has been asked (Cornford et al. 2021; Li et al. 2023 find the
spectrum at initialisation matters more than the sign constraint, though both contribute). Here, with inverse pairs
sharing a channel, the constraint removes only relative signs (Lemma 1); on the ring the useful start lay past pi, on the
wrap's side, and on the torus no start we tried helped. To our knowledge neither the relative-sign gauge nor the loss of the
through-zero route for inverse pairs has been stated; we claim their conjunction and the measurements, not any single
ingredient.

**Depth.** Shallow transformers can represent solvable automata but learn such shortcuts unreliably (Liu et al. 2023; Hahn
and Rofin 2024); Selective RoPE and PaTH (Yang et al.) solve parity or Flip-Flop in one layer. Appendix K compares one
signed path layer with deeper index attention on our torus.

## 6. Limitations

"Does not learn" is scoped to our budgets and one recipe per task; 3x the ring budget did not help, and learning rate,
weight decay from scratch, optimiser and depth were not varied for the sign contrast. Small models (one layer, 2 heads,
d = 128, a rank-4 latent shared across heads), synthetic tasks, n = 8, no recurrent or SSM baselines, no exp(.)-type (LRU)
parameterisation. On the ring the evidence is mainly a loss criterion, and at p_plus 0.5 it does not survive a looser
threshold (0.1; post hoc). On the torus the barrier class has less latent slack: the signed cancelling code needs one
latent dimension per inverse pair, the wrapped code also the 2 pi k / omega offsets, and per-head rank is known to decide
torus learnability for signed steps; this is uncontrolled. Softplus carries an
initialisation confound. The mechanism is shown on the ring only, with its |.| side unresolved; on the torus the large start
also slows the signed model. Lemma 1's learnability corollary is untested at our power (Sec. 4.5: the same-direction
task is harder for both arms), and the barrier account does not explain the |.| difficulty there. The torus sign batch had
no independent pre-launch check. CARoPE and CoPE were tested at one layer, CoPE on the ring and line only. Natural text is not studied.

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
- Li, B., Wu, Y., Huang, N. and Zhang, W. GridPE: a grid cell-inspired unified position embedding for arbitrary-dimensional
  spaces. arXiv:2406.07049 (v3), 2024.
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
- Whittington, J. C. R., Muller, T. H., Mark, S., Chen, G., Barry, C., Burgess, N. and Behrens, T. E. J. The
  Tolman-Eichenbaum Machine: unifying space and relational memory through generalization in the hippocampal formation
  (TEM). Cell 183(5):1249-1263, 2020.
- Whittington, J. C. R., Warren, J. and Behrens, T. E. J. Relating transformers to models and neural representations of the
  hippocampal formation (TEM-t). ICLR 2022; arXiv:2112.04035.
- Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E., Savarese, P., Golan, I., Soudry, D. and Srebro, N. Kernel and
  rich regimes in overparametrized models. COLT 2020*; arXiv:2002.09277.
- Yang, S., Shen, Y., Wen, K., Tan, S., Mishra, M., Ren, L., Panda, R. and Kim, Y. PaTH attention: position encoding via
  accumulating Householder transformations. NeurIPS 2025; arXiv:2505.16381.
- Zhang, Y., Chen, Z., Liu, Y., Qin, Z., Yuan, H., Xu, K., Yuan, Y., Gu, Q. and Yao, A. C.-C. Group representational
  position encoding (GRAPE). ICLR 2026; arXiv:2512.07805.

## Figures and tables

- Table 1 (Sec. 4.3): start x step parameterisation. Table 2 (Sec. 4.5): moves +1/-1 vs +1/+2. (Draft 3's Table 1, the
  three integrators, is folded into Sec. 3.)
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
- **Fig. C1** (appendix; `docs/paper/figs/fig_ring_strata.png`). Ring revisit accuracy by gap and wrap-only stratum (post hoc).
- **Fig. K1** (appendix; `docs/paper/figs/fig_factorial_strata.png`). Torus T = 1024, accuracy by revisit stratum (within 128
  moves, >= 128 moves, wrap-only) for path 1 layer and index 1 layer, 4 layers, looped x4 (n = 4; means, range). Dashed line:
  the best-constant (predict-blank) floor, averaged over strata and runs (overall 0.507); the n-gram floor 0.576 is not drawn.

## Appendix plan

- **A. Models and recipes.** Architectures; step parameterisations (Signed, |.|, softplus, AO) and softplus's initial
  step; learning rates, schedules, epochs, seeds per batch; `--fast-attn`.
- **B. Floors and gates.** Predict-blank and best action-stream n-gram floors per task.
- **C. Per-seed tables** for every registered contrast; Fig. C1 and the ring strata table (post hoc; the gap-matched
  wrap-only deficit varies between batches: -0.063, -0.007, -0.032); SOLVED counts and Fisher p for every contrast at loss
  thresholds 0.02 / 0.05 / 0.1 and accuracy >= 0.99 (post hoc; `docs/audits/2026-10-15/reanalyses/solved_thresholds_out.md`);
  the collapse, start, usage and gradient table for every barrier arm and the channel-by-channel flow comparison (post hoc,
  rules fixed in advance in `docs/audits/2026-10-15/reanalyses/PREDICTIONS.md`; `collapse_wall_out.md`,
  `capture_by_channel_out.txt`).
- **D. Constructions.** The balanced conversion (P1(a)); why the naive one is exact but ill-conditioned; continuation protocol
  and the LOST rule (a continuation is LOST if its training loss leaves the solved region); ring vs torus continuation
  learning rates.
- **E. Theory** (`docs/paper/APPENDIX_theory_v2.md`; checks in `docs/paper/appendix_checks/`): A0; Lemma 1 and corollaries
  with bitwise checks; P1(a), P1(c) proved, P1(d) approximation at large angle scale, P1(e) conjecture; P2 rescaling
  sensitivity as mathematics; P3, P4, P5 with its scope; the collective route with the per-channel flow and the
  frozen-latent test.
- **F. Dyck-2 at matched depth.** +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers.
- **G. Navigation in words.** Text world, floors, step table; fresh-seed numbers; dropout-scale re-score.
- **H. Faithful CARoPE and CoPE.** Implementations; P6; the CoPE ring construction; P7 (gate lattice, proved under its
  setup); the CoPE diagnosis (post hoc), read with the line result; precision margins; the near-wrap table start
  (unmeasured).
- **I. The toy** (three lines, exploratory): reproduces the |.| failure without a transformer, fails often from the signed
  default (6/12 solve) and shows no signed escape.
- **J. The clock side** (moved from the main text; below).
- **K. Depth** (moved from the main text; below).

### Appendix J. The other direction is not established

A non-negative step might learn a content clock faster than a signed one. On k-back recency (n = 16 seeds per arm) it trends
that way and is unmeasured twice: loss AUC -0.197 (paired p 0.053), then -0.106 on fresh seeds (p 0.19). We do not claim a
double dissociation. A consistent but untested account: signed steps contain every non-negative path, so a clock costs them
a delay (moving minority signs through zero), while a map costs |.| a block.

### Appendix K. Where a map is needed, one path layer beats index attention at the depths we tried

On the torus at T = 1024 (n = 4 per cell), one signed path layer scores 1.000 (4/4 SOLVED) against index RoPE at one layer
0.719, four layers 0.860 and a block looped four times 0.864 (0/4 each, all stalled within 900 epochs; complete separations,
p 0.029; floors: blank 0.507, n-gram 0.576). Depth and looping lift index accuracy only on revisits within 128 moves
(0.74-0.79 -> 0.90-0.96); long-gap (>= 128 moves) and wrap-only revisits stay at 0.37-0.53, near or below the blank floor
(Fig. K1). Told in English words (pilot seeds excluded): path 0.979 vs index 0.505, +0.474 (6/6 vs 0/6 SOLVED, p 0.0022); the
eight-seed registered numbers are 0.969 vs 0.505 (p 0.0002), and with the dropout-scale re-score the path arm reads 0.993.

This is a learnability measurement, not a capacity bound. The torus walk is the group Z_64 x Z_64; Liu et al. (2023) show
that one attention layer with an MLP can simulate such a group (their Lemmas 6 and 8), using absolute positions and width and
weights growing with T, so their construction does not settle whether our 128-wide RoPE models can represent the task. Their
experiments are mixed for our purpose: at T = 100 with every state labelled, the median of 20 runs stays below 90% for C_4
at 1-15 layers (and C_2, C_6 are unreliable), while C_5, C_7 and C_8 reach >= 98.5% from 3 layers. Our labels are sparse and
our groups larger. On Dyck-2 at matched depth one path layer is worth about three index layers (appendix F).
