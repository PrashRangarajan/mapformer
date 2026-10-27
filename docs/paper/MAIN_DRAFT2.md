# A map is a phase without a clock: what path-integrated attention learns, and what decides whether training finds it

*Main paper, draft 2 (2026-10-21; revised 2026-10-26 per `docs/audits/2026-10-26/CRITICAL_REVIEW.md` Part B and cut to
about 11k words). Appendix: `docs/paper/MAIN_APPENDIX2.md`. Word counts, the source of every number and what changed:
`docs/paper/MAIN_DRAFT2_notes.md`. Delete this line before submission.*

## Abstract

Path-integrated rotary attention replaces RoPE's token index with a learned phase that sums a content-dependent step per
token. We measure what such models learn and what decides whether training finds a map. On 1024-move torus walks, where
a map-free rule that retraces the last run of moves scores 0.843, one path-integrated layer scores 0.998 (8 of 8 seeds
solved) against 0.731 for index RoPE; at matched depth one path layer is worth about three index layers on Dyck-2 and a
biased 1D walk, and more than four on the torus, where four index layers score what retracing scores; LSTM and GRU
baselines with as many or four times the parameters fit their training map but score 0.57-0.65 on new maps. A map code is a
phase without a clock, the phase an out-and-back adds, and we bound how much a clock erodes one attention pass's
position match. Two choices of the step map decide whether training removes the clock. Per-head rank: rank 2 solves the
torus on at most 2 of 8 seeds against 8 of 8 at rank 4, also at twice the budget and at three recipes, although a rank-2
solution exists and is kept by training; forbidding the clock by construction makes rank 2 solve on every seed.
Parameterisation: steps written |s| or softplus(s), which put a barrier at zero, solve the torus in one layer on 0-1 of
8 seeds against 8 of 8 for signed steps from small starts; the |.| cost holds with each step at its best learning rate
on a ring and the torus and in a one-layer diagonal complex state-space model, although the |.| solution exists and is
kept or re-found. Where the angles start relative to pi matters too: a |.| step started past pi learns the ring map by
wrapping, and the LRU-style exp step shows no cost in the state-space model on the ring at one rate, consistent with its
past-pi start, but pays in attention on the torus. Told in English, the walk is learned by one path layer (0.979 vs 0.505), and the result survives a pixel front
end.

## 1. Introduction

Rotary position encodings are integrators. Index RoPE (Su et al. 2024) adds a constant angle per token; content-dependent
variants let each token set its own step (CARoPE, Veisi et al. 2025; CoPE, Golovneva et al. 2024; Selective RoPE,
Movahedi et al. 2026; Mamba-3, Lahoti et al. 2026; MapFormer, Rambaud et al. 2026). When actions drive the step, the
summed phase is a position on a map. MapFormer built this into a transformer and showed that it helps navigation and
learns opposite latents for opposite actions. We add a measurement of how much of the task needs a map, the exchange rate
between path integration and index depth, and an account of what decides whether training finds the map.

The account turns on one decomposition. Split the phase a channel gains over an interval into a part that depends on net
displacement, the *map*, and a part that grows with the number of moves whatever their direction, the *clock*. A map code
is one whose used channels carry no clock: an out-and-back returns the phase to where it started, and one attention layer
finds the same place at any gap and across the wrap of the torus. Every model we see fail keeps short-gap revisits and
loses long-gap and wrap-only ones (Fig. 2), the signature of a clock (Proposition 2).

Depth does not buy the map (Sec. 6), so it has to come from the step map, and whether training finds it depends on whether
training removes the clock. A signed step can remove it collectively, driving the latents of opposite moves to sum to
zero; above rank D, the map's dimension, each channel can also stop reading the clock on its own. A |.| or softplus step
from a small start has neither route, and at rank D no channel can be both clock-free and a full map while a clock
remains (Proposition 4). On both axes the constrained classes represent the map exactly, keep or re-find an installed
solution, and from scratch rarely find it (Fig. 1).

**Contributions.**

1. *Measuring map use.* Map-free floors and revisit strata. Retrace scores 0.934 on MapFormer's 128-move walks and 0.843
   at 1024 moves, where one path layer scores 0.998 against index RoPE's 0.731 (8/8 vs 0/8 SOLVED, p 0.0002).
2. *What depth buys index attention.* Registered exchange rates at matched length and depth: about three index layers on
   Dyck-2 and a biased ring, more than four on the torus (replicated on fresh seeds); trained index models learn local
   return rules whose reach grows with depth in direction runs (analysis). LSTM and GRU baselines at matched and four
   times the parameters memorise their training map and fall below one index layer on new maps (registered).
3. *What decides whether the map is found.* The clock/map decomposition and a proved route inventory per step class.
   Registered support on both axes: the solution exists, is kept or re-found, and is not found from scratch at rank 2 (at
   most 2/8, also at twice the budget and three recipes) or with a |.| step (0/8 against 8/8 on the torus, at most 2/8
   on the ring, also at each step's best learning rate and in a one-layer diagonal complex SSM, where softplus pays too). Causal
   tests: forbidding the clock lets rank 2 solve (8/8 against 1/8); freezing the move latents blocks the signed step's
   collective route (1/8 against 7/8); a signed increment under a non-negative readout restores it (16/16); a |.| step
   started past pi learns the ring map (8/8).
4. *Readers and codes.* Loops, extra layers and a score rule change how a code is read, never the code family (proved).
   PoPE's polar score pins every kernel's peak at zero phase difference (proved); freeing its offset raises the index
   model by +0.122 and leaves the path model at ceiling (registered).
5. *Maps from words.* One path layer learns navigation told in English (registered), 0.956 against 0.411 for two index
   layers on the revisits retrace cannot reach, and files a word by its role, not its identity.

![Figure 1](figs/fig_summary.png)

*Figure 1 (`fig_summary`). Torus, T = 1024 unless marked. (a) One path-integrated layer learns the map; index attention at
one or four layers or looped four times does not (n = 4 shown; fresh-seed replication in App. F7; dashed: retrace and
always-blank floors). (b) For both constrained classes a solution exists and training keeps it (rank 2: 7/8; |.| on the
ring: 8/8) or re-finds it (|.| on the torus: 8/8), while training from scratch solves 0/8. (c) Every failing one-layer
model keeps short-gap revisits and loses long-gap and wrap-only ones; the model that learns the map holds all three (n = 8
per arm; dashed: always-blank within each stratum).*

## 2. Background

**Rotary position as an integrator.** Write the phase of rotary channel c (a head and a two-dimensional block) at token t
as theta_{c,t} = omega_c Phi_{c,t}, with Phi_{c,t} = sum_{u <= t} Delta_{c,u}. The channel's contribution to a score
depends on position only through exp(-i omega_c (Phi_{c,t} - Phi_{c,t'})) (Mamba-3's Proposition 3; App. E1). Howard et
al. (2014) drive one bank of integrators by a rate alpha(t): a constant rate codes elapsed time (Case I), the speed codes
distance travelled (Case II), and the signed velocity codes position (Case III). With an imaginary rate the rotary
families, as organised by GRAPE (Zhang et al. 2026), Selective RoPE and Mamba-3, fall into these cases: index RoPE adds a
constant angle; CoPE, CARoPE and non-negative content steps a non-negative one; MapFormer, Selective RoPE and Mamba-3 a
signed one. Velocity-modulated integrators have been trained with a real Laplace variable (Alipour et al. 2025;
Mochizuki-Freeman et al. 2023; Kabir et al. 2025), and RoPE derived as a velocity-controlled oscillator (GridPE, Li et
al. 2024). TEM and TEM-t (Whittington et al. 2020, 2022) frame the Case III code as a "where" learned apart from the
"what".

**MapFormer.** MapFormer's working-memory model (MapWM) rotates content queries and keys by
theta_t = omega * cumsum(W_out W_in x_t), a low-rank content-to-angle map; observations learn near-zero steps and opposite
actions opposite steps (their Fig. 3d). In their Table 2, one-layer MapWM with rank 2 solves 2D navigation at 128 steps,
rank 1 does not, and RoPE, TAPE and CoPE with up to four layers fall short. Our implementation reproduces that result
(0.989 for MapWM and 0.987 for MapEM, against 1.00 in v4 and 0.99 / 1.0 in v1; App. A4).

**Representability.** Real non-negative eigenvalues cannot track parity (Sarrof et al. 2024, Thm. 2; Grazzi et al. 2025,
Thm. 1). Rotations escape this: rotations by non-negative angles generate cyclic groups (Grazzi et al., Thm. 3; Mamba-3,
Sec. 3.2). Proposition 3 extends this to our rank-limited step maps, so every difference between signed and non-negative
steps in this paper is a difference in what training finds.

## 3. Setup and how map use is measured

**Models.** MapWM with one layer (up to four where stated), 2 heads, d_model 128 (about 204k parameters), and step
Delta = W_out W_in x through a rank-r latent; unless stated r = 4, shared across heads (MapFormer's latent is per head;
per-head variants are named wherever rank is compared). *Per-head rank* is the rank of the map from token to one head's
angles. Step parameterisations: *signed* (Delta = s = W_out W_in x), *|.|* (Delta = |s|), *softplus* (every step starting
at ln 2) and *exp* (the LRU's form, every step starting at 1). |.| and softplus are *barrier parameterisations*: no
gradient step carries Delta through zero. *Observation-step-free* variants give steps to the move tokens only. Each head
has 32 two-dimensional rotary blocks (*channels*) with frequencies omega_i = 2 pi (1/Delta_max)^(i/31), Delta_max the grid
size, as in MapFormer; omega is free. The index baseline is RoPE at 1-4 layers or one block looped four times. We also
pair PoPE (Gopalakrishnan et al. 2025), a rotary variant with a polar score, with its path-integrated version MapPoPE, and
compare faithful CARoPE, CoPE and Selective RoPE (App. I).

**Training and evaluation.** AdamW (learning rate 1e-3, weight decay 0.05), warm-up then cosine decay, cross-entropy on
revisited observations only (Dyck-2: next-token loss; App. A2). Torus models are scored on a held-out map, so every torus
number is transfer to a new map; ring models on fresh maps.

**Tasks.** (i) *2D torus*: 64 x 64, walked by runs of 1-10 repeated moves interleaved with observations drawn from 16
objects; T = 1024 moves (2,048 tokens), 900 epochs; MapFormer's T = 128 for the floor analysis and the PoPE pair. (ii)
*1D ring* of 32 cells, T = 128, forward-step probability p_plus (0.5 balanced; 1.0 never reverses), 300 epochs. (iii)
*Dyck-2* at length 32 and nesting depth 12, scored by Hewitt et al.'s (2020) closing accuracy (chance 0.5). (iv) *Text
world*: the torus walk rendered as English (58 words), T = 1024 words. (v) 2D and 3D tori of other sizes for rank
against dimension. Loss and accuracy are on *revisits*, observations at cells visited before.

**Floors and strata.** For each task we measure the best map-free predictor: always-blank, the best n-gram over orders
0-5, and *retrace*: while the current run of moves undoes the previous run, copy the observation seen when the walk was
last at the cell it is returning to. *Short-gap* revisits come fewer than 128 moves after the last visit; *long-gap*
revisits later; *wrap-only* revisits are to a cell seen before only from a different unwrapped position, a lap away.
Runs are at most 10 moves, so retrace never reaches a long-gap or wrap-only revisit: those strata separate a map from a
local rule (Fig. 2).

![Figure 2](figs/fig_strata_modes.png)

*Figure 2 (`fig_strata_modes`). Revisit accuracy by stratum, torus, T = 1024; lines: arm means, dots: runs. (a) Index
attention (n = 4): depth and a loop help short-gap revisits only. (b) Per-head rank (n = 8, same initial weights): rank 4
near 1.000 in every stratum; rank 2 loses accuracy with the gap, most at wrap-only revisits. (c) Step parameterisation
(n = 8): signed near 1.000 everywhere; |.| and softplus fall from short gaps to wrap-only revisits; one-layer index RoPE
for reference. Dashed: always-blank within each stratum; dotted: retrace on short gaps (0.91).*

**Run classes and protocol.** A run is *SOLVED* if its mean training loss over the final 5% of epochs is below 0.05;
otherwise *STALLED* if its final-10% mean loss is within 5% of the preceding 10%, and *DESCENDING* otherwise. Every
contrast not labelled "(analysis)" is a registered decision or declared secondary of a pre-registered batch unless marked
"not registered", with n = 8 seeds per arm unless stated, every model trained and tested at the same length and depth, and
exact permutation (accuracy) and Fisher (SOLVED) tests. Numbers marked "(analysis)" were computed post hoc on stored
streams and checkpoints. App. A lists every batch with its seeds, its independent code check and its reproduction status;
App. O lists every analysis script.

## 4. Theory: maps and clocks in the phase

Proofs are in App. E; Table 1 gives the status of each statement. A channel c has angle a_c(x) = omega_c step_c(x) for
token x; for a step map the latent of token x is z_x = W_in e_x and the channel's readout row is w_c, so
step_c(x) = w_c . z_x before any |.| or softplus. The *usage* q_c of a channel is its mean query norm at move tokens
times its mean key norm at observation tokens, normalised to sum to 1. A channel is *blind* to an inverse pair if both its
angles are 0 mod 2 pi.

**Proposition 1 (the prefix sum is in the phase; proved).** The rotary score of channel c is
Re(conj(q~_c) k~_c exp(-i omega_c (Phi_{c,t} - Phi_{c,s}))). If every move token has a fixed step and every observation
step is zero, theta_t = omega P_t, with P_t the unwrapped displacement after move t, and the score depends on position
only through omega (P_t - P_s). One layer then compares two positions at any gap, and wrapping costs nothing in a channel
whose angle per move is a multiple of 2 pi / N.

**Remark 1 (cited).** Index attention sees position only through t - s, so "the moves in (s, t] sum to zero" must be
computed in the residual stream. A global count gives it in two layers in principle (Liu et al. 2023, Lemma 6; Yao et al.
2021), with a normaliser, precision 1/n_t (1/2048 per axis here) and MLP width of order 4nT (64 x 2048 per axis, against
our 512); count-free rules (retrace, a one-lap offset) are available to one layer. Measured exchange rates are therefore
what training finds, not depth bounds.

**Clock and map.** For an inverse pair of moves (x, y) in channel c write the *clock* k_c = a_c(x) + a_c(y) and the *map*
m_c = a_c(y). Over an interval with n_x moves x and n_y moves y, displacement d = n_y - n_x, the phase is
m_c d + k_c n_x exactly. A *map code* has k_c = 0 mod 2 pi in every channel attention uses. The *clock / map ratio* is the
usage-weighted rms of wrap(k_c) over the usage-weighted rms of the map, with the map taken symmetrically,
(wrap(a_c(x))^2 + wrap(a_c(y))^2) / 2, over the inverse pairs and the channels of both heads.

**Proposition 2 (position-factor bound and tick rule; proved).** (a) A zero-displacement revisit after n backward moves
reads phase k_c n in channel c. If a usage share s of the channels has kappa <= |k_c| <= K, the same-cell position factor
sum_c q_c cos(k_c n) is at most 1 - s (1 - cos(kappa n)) for n <= pi / K: the bound says how fast a clock erodes the
position match while n |k| stays below pi, not that every clocked code fails at long gaps. (b) If the target is the key
at the last visit to the same cell and returns come at lags with greatest common divisor g (g = 2 on the even torus), a
channel can serve place-keyed retrieval only if its per-move *tick* is 0 mod 2 pi / g.

**Proposition 3 (representability by wrapping; proved).** If at most r tokens step, a |.| model of rank r reproduces any
signed angle table exactly mod 2 pi; annihilating the observation embeddings in W_in leaves only the moves stepping, so
every |.| model in this paper can represent the map. Softplus and exp can too: with the observations annihilated each steps
ln 2 or 1, and omega' = 2 pi / ln 2 or 2 pi makes that step a full turn.

**Lemma 1 (relative-sign gauge; proved).** Negating a channel's steps (through omega_c) together with the second coordinate
of that channel's query and key projections leaves every score unchanged, since J R(phi) J = R(-phi) for J = diag(1, -1).
A non-negative step therefore fixes only the relative signs of the tokens that step in a channel. For an inverse pair it
removes cancellation through zero, angles (a, -a), while the wrapped code (a, 2 pi - a), the same function, remains. Lemma
1 is a statement about representation; its learnability reading, that a |.| step should learn freely when no step need
cross zero, was tested and not confirmed (Sec. 7.3).

**Proposition 4 (route inventory; proved).** (a) *Signed, collective route.* The clock vector is k = diag(omega) W u with
u = z_x + z_y. The set {u = 0} cancels every channel at once and leaves the map free; gradient flow on a quadratic clock
surrogate converges to it from every start, and for almost every readout it is the only latent set that cancels every
channel. (b) *|.|.* The same latent move ends at a(y) = a(x), a clock. At a generic readout no latent choice cancels more
than 2r channels non-blindly; per channel, an unwrapped |.| pair has |k_c| >= |m_c|, and its non-blind zero-clock points
lie at |k_c| = 2 pi n, across |k_c| = pi from a small start. (c) *Rank.* Let c be the latent step common to all moves (with
the mean observation step added when observations step) and U the map frame. At per-head rank D, c lies in the plane U
spans, so a channel is clock-free only by losing map resolution; above rank D the part of c outside the map plane can be
cancelled channel by channel with the map untouched (*parking*). (d) *One dimension.* At D = 1 only the collective route
acts.

**Further statements (App. E9-E11).** *Proposition 5* (proved to first order): at rank D >= 2 a channel can lower the
drift loss at no first-order cost to the map by turning its resolution off the drift direction (*hiding*), which slows the
collective route; *Conjecture 1*: this keeps the drift in place at rank D >= 2 (in 1D there is nothing to hide along).
That parking is why rank above D helps (the *clock-exit account*) is a conjecture with a PARTIAL registered test. *Remark
2*: "a solution exists and is kept, but is not found from scratch" means a non-empty, stable solution set whose basin the
initialisation rarely hits. *Proposition 6* (proved): theta is computed once from the embeddings, so a loop or extra
layers widen the reader and never change the code family; a score rule acts only on the readout term of target coherence,
not on the code term C0.

**Proposition 7 (peak pinning; proved given delta = 0).** PoPE's score sets beta_c = -delta_c with delta content-free and,
in trained models, in its clamp's dead zone (delta = 0). Then every head's kernel, for every content pair, peaks at
dphi = 0: with a path phase the same place, the revisit target; with an index phase the current token, and with PoPE's
frequencies (at most 1 rad per token) no fixed far offset can be preferred in one layer.

*Table 1. Status of the theory (proofs in App. E).*

| statement | status | checked against |
|---|---|---|
| Prop. 1; Remark 1 | proved; cited | long-gap same-cell attention; no count in torus index models (Sec. 6) |
| Prop. 2, bound and tick rule | proved | consistent with every failing model's strata (Fig. 2); text-world channels (Sec. 8) |
| Prop. 3, Lemma 1 | proved | constructions exact to 1.4e-6 rad (App. D) |
| Lemma 1 read as learnability | registered test UNMEASURED | +1/+2: \|.\| 2/8, no better than on +1/-1 (Sec. 7.3) |
| Prop. 4, route inventory | proved | end-state clocks; frozen latents; hybrid readout; route class (Sec. 7.3) |
| At rank D the clock is the obstacle (torus) | registered causal support | clock forbidden 8/8 against 1/8 (Sec. 7.2) |
| Prop. 5; Conjecture 1; clock-exit account | first order; conjecture; registered PARTIAL | drift axis; clock exits 3/8 against 1/8 (Sec. 7.2) |
| Props. 6-8 | proved (7 given delta = 0) | reader fits; delta-freedom batch; CARoPE models (Secs. 7.3-7.4) |

## 5. One path layer solves the long-walk map

**Short walks mostly do not need a map (analysis).** On MapFormer's T = 128 walks retrace scores 0.934 and no revisit is
wrap-only: a run that reverses the previous one returns, step by step, to cells seen moments earlier. At T = 1024 retrace
falls to 0.843 and 7.9% of revisits are wrap-only (Table 2). At T = 128 the path model still beats index attention
(position effect +0.243 at rank 2), but index RoPE falls below retrace and the rank-2 path model clears it by only 0.04.

*Table 2. Torus floors (analysis, on the models' evaluation walks) and accuracies (held-out map; n = 8 per model).*

| T (moves) | wrap-only revisits | blank | n-gram | retrace | index RoPE, 1 layer | path, rank 2 | path, rank 4 |
|---|---|---|---|---|---|---|---|
| 128 | 0.000 | 0.507 | 0.598 | 0.934 | 0.805 | 0.971 | 1.000 |
| 1024 | 0.079 | 0.506 | 0.575 | 0.843 | 0.731 | 0.894 | 0.998 |

**At T = 1024 one path layer solves the map and index attention does not.** One path layer scores 0.998 +/- 0.004 (8/8
SOLVED) against 0.731 +/- 0.008 for index RoPE (0/8), +0.267 (permutation and Fisher p 0.0002); index RoPE sits below
retrace on every run (analysis). Two further registered batches give 1.000 against 0.722 (n = 8) and 1.000 against 0.719
(n = 4). By stratum (short gap / long gap / wrap-only) the path model scores 1.000 / 0.995 / 0.982 and index RoPE 0.784 /
0.462 / 0.471, against stratum floors of 0.506 / 0.500 / 0.507 and retrace 0.911 on short gaps (analysis; Fig. 2c). Long-gap
and wrap-only revisits require the map, and only the path model has it.

## 6. What index depth buys

We ask how many index layers one path layer is worth, both trained and tested at the same length and nesting depth.

**Dyck-2: about three.** A bracket string is a walk on a stack. One path layer scores 0.996, index attention 0.648 /
0.889 / 0.964 / 0.982 at 1-4 layers, against a stack-free floor of 0.594 (Fig. 3a). The position main effect over two
encodings (MapWM vs RoPE, MapPoPE vs PoPE; App. F) is +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers, positive on 8/8
seeds at each depth (sign-flip p 0.0078). With three times the budget four index layers reach the ceiling (0.997-0.998);
training on depths 4-12 keeps +0.110 at depth 12.

**A biased 1D walk: the same rate.** On fresh seeds (n = 6) one path layer scores 1.000 at every bias, and index attention
with 1 / 2 / 3 layers 0.710 / 0.949 / 0.998 at p_plus 0.5, 0.677 / 0.962 / 0.998 at 0.75 and 0.824 / 0.987 / 0.999 at 0.9
(Fig. 3b; declared secondary). Wherever a step can be undone three index layers come within 0.002 of the path layer; at
p_plus 1.0 one index layer reaches 1.000.

**The 2D torus: more than four.** At T = 1024 (n = 4) one path layer scores 1.000 (4/4) against index attention at one
layer 0.719, four layers 0.860 and one block looped four times 0.864 (each 0/4; complete separations, p 0.029; Fig. 3c).
A registered replication on four fresh seeds gives every reading again (App. F7): 0.999 (4/4) against 0.729 / 0.830 /
0.859 (0/4 each, p 0.029); over both batches (n = 8, descriptive) 0.999 against 0.724 / 0.845 / 0.862. Four index layers
score what retrace scores on the same walks (-0.000 over both batches; 0.913 on short gaps against retrace's 0.911) and
fall to 0.41 on long gaps, below the constant 0.51: depth buys index attention about what a map-free rule gives.

**Recurrent baselines: integration without transfer (registered; App. F8).** An LSTM or GRU given the same stream can
integrate the moves; the question is whether it also stores a new map. Trained like every torus model here (one map per
run, scored on a new map; T = 1024, n = 8), LSTM and GRU at one and two layers with the path model's ~205k parameters,
and a one-layer LSTM with four times as many at two learning rates, score 0.57-0.65 against 0.9998 for one path layer
(0/8 against 8/8 SOLVED; d +0.35 to +0.43, p 0.00016 for each of the six cells). They fit their training map (0.93-0.98
on fresh walks over it) and fall below one index layer (0.724) and below retrace on the new one, even on short-gap
revisits (0.62-0.70 against 0.910); the larger LSTM fits best and transfers worst. Under this protocol the recurrent
networks store the map in their weights, where the path layer stores it in context; recurrent networks trained across
many maps are not tested.

![Figure 3](figs/fig_exchange_rate.png)

*Figure 3 (`fig_exchange_rate`). Accuracy of index RoPE against its number of layers (orange; dots: runs) beside one path
layer (blue), trained and tested at the same length and depth. (a) Dyck-2, length 32, depth 12 (n = 8). (b) Ring of 32
cells, three move biases (n = 6 fresh seeds). (c) Torus, T = 1024 (n = 4; fresh-seed replication in App. F7): four index
layers (0.860) or a loop of four (0.864) stay just above this batch's retrace floor (0.844). Dashed: map-free floors
(analysis).*

**Why the rate differs by task (analysis; App. F).** Of Remark 1's options training finds only the count-free rules, with
a reach that differs by task. One index layer learns retrace and the one-lap offset, so at p_plus 1.0, where every revisit
is a one-lap return, it matches the path layer. Three index layers on the ring stay flat in the number of direction runs
a revisit spans (Fig. 4b) and carry a displacement code over the last ~64 moves (probe R^2 0.72; older displacement R^2
0.01), which covers most revisits of a 128-move walk. On the torus no index model carries a count (R^2 at most 0.21);
four layers or the loop solve retraces (0.996 / 1.000) and decay with the runs spanned, also at a fixed gap (Fig. 4a).
Nearly all long-gap revisits span 17 or more runs (98.5%), beyond that reach; there the last layer puts 0.00-0.01 of its
attention on the remembered cell, where the path layer puts 0.81. So the torus rate is "more than four" because its
long-range revisits lie beyond a local reach and the global count is not found. Heo et al. (2026) show that the error of
constant-depth parallelizable models falls exponentially with depth, an expressivity account; this is what training finds
at each depth. On a rank-2 path code the same four passes lift long-gap revisits from 0.671 to 0.927-0.990 (Sec. 7.4):
extra passes add accuracy where the code carries position information one pass cannot read.

![Figure 4](figs/fig_segments.png)

*Figure 4 (`fig_segments`; analysis of registered batches, eval mode). Revisit accuracy against the direction runs spanned
between the two visits (2 runs = retrace); lines: means of per-seed accuracies, dots: seeds. (a) Torus, T = 1024 (n = 4),
plain revisits under 128 moves (solid) and at gaps of 32-127 moves only (dashed): four index layers fall from 0.996 at 2
runs to 0.493 at 17+, the loop from 1.000 to 0.502; one path layer is at 1.000 in every bin (table in App. F1). (b) Ring,
p_plus 0.5 (n = 6), plain revisits: three index layers stay flat, two decay, one learns retrace only. Floors on each
batch's own walks: retrace exact on 2-run revisits (dotted), the best constant elsewhere (dashed).*

## 7. What decides whether training finds the map

Two choices of the content-to-angle map decide whether the torus map is learned: its per-head rank and how its step is
parameterised. For each, the constrained class fails from scratch, while a constructed solution, installed and trained, is
kept or re-found.

### 7.1 The clock and the signature

From a small start training ends in one of two places on either axis (analysis; table and figure in App. G3). Models that
succeed end with clock-free heads: signed and rank-3/4 arms at clock / map 0.001-0.051 (arm means), every solved rank-2 run
with at least one clean head. Models that fail end with a clock of the order of their map: the barrier-parameterised arms
at 1.25-1.57 (for an unwrapped |.| channel at least 1 by construction, so the informative part is that the channels did
not wrap), signed steps with frozen move latents at 1.18, and the unsolved rank-2 runs at 0.54-0.69; 4 of the 20
unsolved rank-2 runs instead end clock-free with their two map axes collapsed onto one direction. The strata show the
signature Proposition 2 leads one to expect: rank 2 scores 0.950 / 0.671 / 0.557 on short-gap / long-gap / wrap-only
revisits and |.| 0.870 / 0.630 / 0.515, against 1.000 / 0.995 / 0.982 for the signed rank-4 model (Fig. 2b-c). The bound
alone does not predict failure: two signed torus runs started past pi solve with ratios 1.39 and 1.78, carrying their
usage on channels whose per-pair clock is near pi, where the bound says nothing (App. H7).

### 7.2 Per-head rank of the content-to-angle map

**Per-head rank decides it.** On the torus at T = 1024 six bottlenecks built from the same initial weights at each seed
(Table 3, Fig. 5a): every arm whose heads each read a rank-2 map solves 0-2 of 8; rank 3 solves 6/8 and rank 4 8/8,
whether the latent is shared or per head. At matched sharing (per-head rank 4 against a block-diagonal rank-4 latent in
which each head reads rank 2) the split is 8/8 against 2/8 (Fisher and permutation p 0.0070). Neither total latent size
nor the initial angle scale tracks success. With MapFormer's Table 2: rank 1 cannot represent a 2D map; rank 2 can, and
learns 128-step walks; at 1024 moves reliable learning needs per-head rank above the map's dimension.

*Table 3. SOLVED (of 8) and accuracy at T = 1024 by per-head rank; torus, 900 epochs, same initial weights per seed.*

| per-head rank | arm | SOLVED | accuracy |
|---|---|---|---|
| 2 | one rank-2 latent shared by the heads | 0/8 | 0.894 |
| 2 | per-head rank-2 latents | 2/8 | 0.885 |
| 2 | block-diagonal rank 4 (each head reads rank 2) | 2/8 | 0.948 |
| 3 | per-head rank 3 | 6/8 | 0.987 |
| 4 | shared rank 4 | 8/8 | 0.998 |
| 4 | per-head rank 4 | 8/8 | 0.999 |

**Rank 2 still fails at twice the budget and at three recipes.** At 1800 epochs from scratch rank 2 solves 0/8 (0.908)
against rank 4's 7/8 (Fisher p 0.0014; accuracy +0.086, p 0.0003). PoPE's polar score does not rescue it (3/12 under
either score; Sec. 7.4). On fresh seeds rank 2 was trained at the default, at three times the learning rate and at four
times the weight decay (App. G9): it solves 0/8 at each (0/24), while rank 4 solves 8/8 at the default; rank 2 at its
best recipe (3e-3, selected on training loss) scores 0.910 against 1.000 (decision p 0.0002 on SOLVED and accuracy). The
selection cannot hurt rank 2, but the search is limited: neither alternative is a recipe at which rank 4 is known to
learn (at 3e-3 the related signed rank-4 observation-step-free model solves 0/8 on this torus within 900 epochs, App.
H12; the higher decay shrinks the angles most and leaves the most rank-2 runs stalled). The result reads: rank 2 fails at
each of three recipes. Within 900 epochs rank 4, in turn, learned only with weight decay (8/8 against 1/8 without; App.
G8).

**The rank-2 solution exists and is held.** The rank-2 projection of each solved rank-4 model scores 0.9955 frozen.
Trained from there it solves 7/8, as many as the rank-4 control from its own solution (7/8), against 1/8 for rank 2
trained further from its own stalled weights (accuracy p 0.0009; Fig. 6a). A rank-2 map that solves the torus exists, is
kept by training, and is not reached from scratch.

**Rank against the map's dimension.** Rank D is hard in 2D and 3D and not in 1D (Fig. 5b): on a 2D torus of 32 cells per
side rank 2 solves 1/8 against rank 3's 8/8 (Fisher p 0.0014); on a 3D torus of 10 per side rank 3 solves 1/8 (rank 4,
4/8), and rank 4 solves an 18-per-side 3D torus 8/8; on the ring rank 1 solves 7/8 against rank 2's 8/8. Larson (2026)
finds that gradient descent recruits the minimal faithful dimension of a group-composition task, with a fixed decoder and
dense state supervision. Here too the rank-D solution exists and is held, so the two agree on capacity; what differs is
search: with a sparse revisit loss and a shared readout, training reached the map reliably only with one dimension more.

**On the torus the rank-2 obstacle is the clock (registered).** A batch on fresh seeds (shared rank, seven arms from the
same initial weights; App. G7) replicates the split (rank 2 1/8, rank 4 8/8). In the *drift-forbidden* rank-2 arm the
steps of each inverse pair are tied to exact negatives and every observation step is 0, so the clock is zero by
construction: it solves 8/8 at accuracy 1.000, like rank 4. In the *actions-only* rank-2 arm only the observation steps
are 0 and the moves' own clock remains: it solves 1/8 (Fisher p 0.0014 against 8/8). Removing the clock is sufficient,
removing its observation part is not: what stops rank 2 is the clock, not capacity.

**Why the extra rank helps is open.** The same batch's registered primary gives rank 2 an explicit clock exit without
extra map rank (a learned per-channel slot per move axis): it solves 3/8 against rank 2's 1/8 (Fisher p 0.57), as does a
per-token slot, registered PARTIAL with 4 of the 5 unsolved per-axis runs still descending. The checkpoints are consistent
with parking but do not discriminate it from removal through the latent (analysis; App. G4): in solved heads above rank D
the residual clock lies almost entirely off the map plane, and drifting rank-D heads drift along their least-resolved axis
(50 of 55 runs, against 25.1 at random), the geometry Proposition 5 leads to.

![Figure 5](figs/fig_rank_solved.png)

*Figure 5 (`fig_rank_solved`). Runs (of 8) that solve the map, by per-head rank. (a) Torus, T = 1024, six bottlenecks from
the same initial weights: rank-2 arms 0-2/8, rank 3 6/8, rank 4 8/8. (b) Rank equal to the task's dimension D against
D + 1 (separate batches and recipes): ring, rank 1 7/8 (rank 2, 8/8); 2D torus, rank 2 1/8 (rank 3, 8/8); small 3D torus,
rank 3 1/8 and rank 4 4/8; rank 4 solves an 18-per-side 3D torus 8/8 (hatched). Torus budget 900 epochs; at 1800 epochs
rank 2 still solves 0/8.*

![Figure 6](figs/fig_existence.png)

*Figure 6 (`fig_existence`). A solution exists in each constrained class and training keeps or re-finds it, but training
from scratch does not find it (bars: runs SOLVED of 8; hatched: installed and then trained). (a) Rank 2 on the torus: the
rank-2 projection of a solved rank-4 model is held (7/8); from scratch 0/8, stalled runs trained further 1/8. (b) |.| on
the ring: the installed exact solution is held (8/8, lost on 0/8); from scratch 0/8 against signed 8/8. (c) |.| on the
torus: the installed solution is perturbed (loss above 0.05 on 6/8) and re-found (8/8); from scratch 0/8 against 8/8.*

### 7.3 How the step is parameterised, and where its angles start

**Barrier parameterisations rarely learn the map from a small start.** On the torus at T = 1024 the |.| step scores 0.821
against 0.998 for the signed step (-0.177, permutation p 0.0002) and solves 0/8 against 8/8; softplus scores 0.781 and
solves 1/8 (-0.218, p 0.0003). On the ring |.| solves 2/8 and 1/8 against 8/8 at p_plus 0.5 and 0.75 (Fisher p 0.007 and
0.0014), and observation-step-free models give 0/8 against 8/8 on the ring and on the torus (0.880 against 1.000). The
signed median crosses the SOLVED threshold at epochs 57-480 in four batches, while most |.| runs flatten at a loss of
0.3-0.5 (Fig. 7).

**The cost holds within 900 epochs at each step's best learning rate, without weight decay and at 2 layers d_model 256**
(Table 4). With each step at its own best of three rates on the torus (App. H12) and of four on the ring (App. H1), |.|
solves 0/8 and 2/8 against signed 8/8. Higher rates hurt the signed step first (on the torus it solves 8, 0 and 0 of 8 at
1e-3, 3e-3 and 1e-2, every run at the higher rates still descending) and do not help |.|.

*Table 4. The cost of the |.| step by task, model and recipe (observation-step-free unless marked "full"; n = 8 per arm).*

| task and model | recipe | \|.\| accuracy, SOLVED | signed accuracy, SOLVED | test |
|---|---|---|---|---|
| torus, 1 layer, full | default (lr 1e-3, weight decay 0.05) | 0.821, 0/8 | 0.998, 8/8 | perm p 0.0002; Fisher 0.0002 |
| torus, 1 layer | default | 0.880, 0/8 | 1.000, 8/8 | perm p 0.0002; Fisher 0.0002 |
| torus, 1 layer | each at its best of three rates (\|.\| 3e-3, signed 1e-3) | 0.890, 0/8 | 1.000, 8/8 | selection-aware p 0.0002 (SOLVED), 0.0095 (accuracy) |
| torus, 2 layers, d_model 256 | default | 0.967, 2/8 | 0.999, 8/8 | perm p 0.0005; Fisher 0.0070 |
| ring | default | 0.911, 1/8 | 1.000, 8/8 | perm p 0.0002; Fisher 0.0014 |
| ring | weight decay 0 | 0.873, 0/8 | 1.000, 8/8 | perm p 0.0002; Fisher 0.0002 |
| ring | lr / 3 (3.3e-4) | 0.841, 0/8 | 0.950, 6/8 | Fisher 0.0070 |
| ring | each at its best of four rates (\|.\| 3e-3, signed 1e-3) | 0.963, 2/8 | 1.000, 8/8 | selection-aware p 0.0070 (SOLVED), 0.0057 (accuracy) |
| ring, 1-layer diagonal complex SSM | lr 3e-3 | 0.821, 0/8 | 1.000, 8/8 | perm p 0.0002; Fisher 0.0002 |
| ring, 1-layer diagonal complex SSM | lr 1e-3 | 0.756, 0/8 | 0.994, 7/8 | perm p 0.0002; Fisher 0.0014 |

**The |.| solution exists, and training keeps or re-finds it.** Proposition 3 constructs it from each solved signed
observation-step-free model (angles unchanged mod 2 pi to 1.4e-6 rad, frozen accuracy equal to the source's on 8/8
seeds, weights within about 5x of the signed model's; App. D). On the ring the constructed model, trained 300 more epochs,
stays solved on 8/8 seeds, keeping its cancelling code in the channels attention uses (98%, usage-weighted); on the torus
it is perturbed early (peak loss 0.11-0.21) and re-found, ending solved on 8/8 seeds. From scratch the class solves 0/8
on both tasks (Fig. 6b-c).

**Where the angles start matters too (ring, registered).** Varying only the initial move angles, every
other weight identical at a seed (Table 5, Fig. 8): from the |.| start, where both moves rotate the same way, signed still
solves 7/8 and |.| 0/8 (Fisher p 0.0014); a |.| step started with opposite-move angle sums past pi solves 8/8 (against 0/8
from the small start, p 0.0002), while a large start whose sums stay below pi solves 2/8, so size alone does not rescue
it. From past pi both classes cancel by wrapping; from small starts signed models cancel through zero (98-99% of channels
end opposite-signed; analysis). On the ring the variable is the barrier parameterisation together with where the angle
sums start relative to pi. On the torus the start lever does not transfer: a past-pi start leaves |.| at 0/8 and slows the
signed step to 2/8, with 5/8 runs per large-start arm still descending (App. H3); exp's
built-in past-pi start does not help there either (below).

*Table 5. SOLVED (of 8) by start, move set and model (observation-step-free, rank 4; registered; App. H3, H4, H13, H14).*

| task, model | moves | initial move angles | signed | \|.\| | other |
|---|---|---|---|---|---|
| ring, attention | +1/-1 | default (small, signs independent) | 8/8 | 0/8 | |
| ring, attention | +1/-1 | the \|.\| start (both moves the same way) | 7/8 | 0/8 | |
| ring, attention | +1/-1 | large, sums below pi | -- | 2/8 | |
| ring, attention | +1/-1 | large, sums past pi | 8/8 | **8/8** | |
| torus, attention | four moves | default | 8/8 | 1/8 | |
| torus, attention | four moves | large, sums past pi | 2/8 | 0/8 | |
| ring, attention | +1/+2 (no inverse pair) | default | 5/8 | 2/8 | UNMEASURED |
| ring, SSM, lr 1e-3 | +1/-1 | default (exp: 13 of 32 channels per head past pi) | 7/8 | 0/8 | softplus 0/8, exp **6/8** |
| torus, attention, best of lr 1e-3 / 3e-3 | four moves | default (exp: 11 of 32 channels per head past pi) | 8/8 | 0/8 | softplus 0/8, exp 0/8 |

**The signed escape is collective, and lives in the increment.** From the |.| start, freezing the two move latents (W_out
and omega still free) blocks the signed escape: 1/8 against 7/8 (Fisher p 0.010), ending with a |.|-like clock (1.18).
With the latents free, one move's latent turns toward the negative of the other's and every channel cancels together, the
antipodal end state MapFormer described (their Figs. 3d, 9b). A hybrid with the readout confined to the non-negative
orthant, a_c(x) = omega_c |w_c| . z_x, but a signed increment solves 8/8 at both reference rates, against |.| 0/8 and 2/8
(Fisher p 0.0002 and 0.007; App. H8): a non-negative readout costs nothing; the barrier on the increment is what costs.

**Losing the through-zero route does not account for the difficulty (registered).** By Lemma 1 a |.| step represents,
without wrapping or crossing zero, a map whose moves step the same way. With moves +1/+2 (Table 5; App. H4), which need
integration (index RoPE 0.649, 0/8) and which signed models learn with a sign-coherent code (0.92 of channels end with both
move angles of one sign, against 0.01 on +1/-1), |.| still solves only 2/8, no more than on +1/-1 (1/8 in the same batch;
Fisher p 1.00), against signed 5/8. The registered prediction, that the cost disappears without an inverse pair, is not
confirmed; the comparison is unmeasured because +1/+2, where every revisit is a lap, is harder for signed steps too. Lemma
1 is about representation; what makes the barrier classes hard to train from a small start is not settled by it.

**Barrier or route? (registered, half-read).** A 2x2 on the ring from the |.| start (App. H11) gives |.| one learnable
offset per move, a collective route to a point where every channel cancels by wrapping, with the barrier kept: it solves
0/8 against signed 8/8 (Fisher p 0.0002), while the same class started past pi solves 7/8 and ends on that point. The
registered half-reading is that a route to the wrap does not carry |.| past pi from a small start; the cell without a
barrier and without a route (signed, latents frozen) solves 4/8, so the 2x2 as registered is unmeasured.

**The cost is not specific to softmax attention; exp escapes only in the SSM.** Replacing softmax by the causal unnormalised
kernel gives a one-layer diagonal complex SSM without decay or gate, with transition exp(-i omega Delta(x_t)). On the ring
|.| solves 0/8 against signed 8/8 at learning rate 3e-3 and 0/8 against 7/8 at 1e-3, and softplus 1/8 and 0/8 (Table 4;
App. H10, H13); the constructions are exact and the installed |.| solution is re-found. The exp step diverged at 3e-3 but
shows no cost at 1e-3: 6/8 against 7/8 (Fisher p 1.0), at a rate where the signed anchor reached exact codes on only four
seeds. It ends with every step positive, and 90 of its 95 cancelling channels are channels its initialisation, exp(0) =
1, already put past pi (analysis); softplus, which starts 9 of 32 channels per head past pi, does not solve. In attention
exp pays. On the ring exp and softplus solve 1/8 to 3/8 against signed 8/8 at both reference rates (App. H9). On the
T = 1024 torus (registered, App. H14), each at its best of two rates, exp solves 0/8 (0.916) and softplus 0/8 (0.908)
against signed 8/8 (0.993; Holm p 0.0003 on SOLVED; budget-scoped), its cancelling channels again mostly past-pi-start
ones cancelling by a wrap (95 of 107), without a map. So the escape is not exp's alone; between the two batches mixer, task
and start (past pi on 41% of channels in the SSM, 34% on the torus) all change, and which matters is not separated. The
cost is shown for |s|, softplus(s) and, in attention, exp(s).

**Published content phases also miss the torus map.** *Proposition 8 (proved):* faithful CARoPE advances every channel by
(0, 1] per token, so the shortest return's phase lies in (0, 4], never 0 mod 2 pi (App. E5); it scores 0.667 on the torus
(0/8), below index RoPE's 0.722 (p 0.040). Faithful CoPE can represent ring cancellation (App. I) yet solves 0/8 on the
ring (0.887 against 1.000, p 0.0002). Faithful Selective RoPE, signed, builds a phase it relies on (+0.236 over index
RoPE) but solves 0/8 against MapWM's 8/8 within 900 epochs, every run still descending; an observation steps its phase
0.86-0.96 as far as a move (MapWM: 0.027). Its state-tracking configuration keeps every step non-negative and fails like
the barrier steps (0.769, 0/8).

![Figure 7](figs/fig_loss_curves.png)

*Figure 7 (`fig_loss_curves`). Training loss per epoch, signed (blue), |.| (orange) and softplus (green), every run of four
batches (thick: median of 8; log scale). Dashed: the SOLVED threshold; n/8 = runs SOLVED. (a) Torus, T = 1024: signed 8/8,
|.| 0/8, softplus 1/8. (b) Ring, observation-step-free: 8/8 against 0/8. (c) Torus, observation-step-free: 8/8 against
0/8. (d) As (c) at two layers and d_model 256: 8/8 against 2/8.*

![Figure 8](figs/fig_mechanism.png)

*Figure 8 (`fig_mechanism`; ring, 8 runs per panel). Move angles (a(+1), a(-1)) of every channel at the start (grey) and
after training (dots). A channel cancels opposite moves on the blue line (through zero, reached only with opposite signs)
or the red line (by wrapping); grey shading: the mixed-sign quadrants the |.| step cannot reach; dotted: sums of pi. (a)
Signed from the same-sign start: 7/8 solve, one move's angle crosses zero. (b) |.| from the same start: 0/8. (c) |.| from
a large start below pi: 2/8, by wrapping. (d) |.| from a start past pi: 8/8, by wrapping (App. H).*

### 7.4 Readers cannot repair a code

**More computation reads a rank-2 code; it does not repair it.** Looping the single block four times (identical
parameters and initialisation) or using four real layers raises rank-2 accuracy from 0.894 to 0.973 and 0.990
(permutation p 0.0034 and 0.0012; SOLVED 2/8 and 5/8 against 0/8; rank 4 0.998, 8/8). By Proposition 6 neither alters the
code family. A one-pass, position-only reader fitted to each model's own phase recovers the one-layer models' accuracy
but falls 0.32-0.44 short of the looped and deep models on long-gap and wrap-only revisits, and their codes stay far from
rank 4's (p 0.0002); only 1 of the 16 runs finds a two-dimensional code (analysis; App. M). Downstream computation
compensates for the rank-2 deficit and leaves it in place.

**Where the angle comes from and how content uses it are separate choices.** Path integration changes dphi; PoPE's polar
score removes beta and, by Proposition 7, pins every kernel's peak at zero phase difference. On the short-walk torus PoPE's
score raises the path model (MapPoPE 0.999 against MapWM 0.971) and lowers the index model (PoPE 0.679 against RoPE
0.805), an interaction of +0.154 (8/8); for the index model no kernel then prefers offset 3 over offset 1 (RoPE's is a comb
at the retrace offsets; Fig. 9a; analysis). Freeing PoPE's phase offset (registered) raises the index model by +0.122 (8/8,
sign-flip p 0.0078), most on multi-step reversals, and leaves the path model at ceiling (-0.018, p 0.31). At rank 2 on the
long-walk torus the polar score repairs short-gap revisits through the readout term (target coherence 0.617 to 0.960) and
leaves the code term low on long-gap (0.80 / 0.91 under the two scores) and wrap-only revisits (0.40 / 0.48, against 0.97
at rank 4; analysis; Fig. 9c): under either score rank 2 solves 3/12 (registered).

![Figure 9](figs/fig_kernel_peak.png)

*Figure 9 (`fig_kernel_peak`; analysis of registered batches). (a) Index arms, torus T = 128: the token offset at which
each content pair's kernel peaks. PoPE peaks at offset 1 in 98.2% of pairs; RoPE's is a comb on the retrace offsets 3 / 7
/ 11 / 15. (b) Path arms at T = 128: target coherence C against C0 (content phase removed), per seed: MapWM 0.703 against
0.905, MapPoPE 0.995 against 0.996. (c) C0 by stratum, torus T = 1024: rank 2 0.90 / 0.96 on short gaps, 0.80 / 0.91 on
long gaps, 0.40 / 0.48 on wrap-only revisits (RoPE / polar score); rank 4 0.97-1.00 in every stratum.*

## 8. Maps from words

Each move of the torus walk is written as a sentence: a direction word chosen from three synonyms, verbs that do not move
the walker, optional adverbs and fillers, and, after some moves, an aside that names an object without visiting it ("she
thought about a cat ."). Nothing in the input says which words move the walker (App. J). A walk of 1024 words is about
132 moves, so retrace scores 0.934 here (analysis), beside the constant (0.505).

**One path layer learns navigation told in English.** On fresh seeds (n = 6) one path layer scores 0.979 +/- 0.048 (6/6
SOLVED) against 0.505 for one-layer index RoPE (0/6), +0.474 (permutation and Fisher p 0.0022); two index layers reach
0.793 (0/6), below retrace on every seed. The decisive revisits are the 13.4% that retrace cannot reach (Table 6;
analysis, all 200 evaluation walks): there the path layer scores 0.956 over all eight seeds (0.978 on the fresh seeds;
re-scored with the attention-dropout scale, App. C9), while two index layers score 0.411 and one 0.454, both below the
constant (0.506). On the 6.6% whose cell holds an object, the path layer scores 0.925 and two index layers 0.129, against
a best constant of 0.088.

*Table 6. Text world: accuracy by whether retrace can reach a revisit (analysis; 200 registered evaluation walks, 6,381
revisits; mean over 8 seeds, fresh seeds in brackets; path re-scored with the dropout scale unless marked).*

| model (n = 8) | all revisits | retrace applies (n 5,525) | retrace cannot reach (n 856) | ... cell holds an object (n 423) |
|---|---|---|---|---|
| path, 1 layer | 0.993 | 0.999 | 0.956 (0.978) | 0.925 (0.967) |
| path, 1 layer, eval mode | 0.969 | 0.975 | 0.934 (0.958) | 0.884 (0.927) |
| index RoPE, 2 layers | 0.772 | 0.828 | 0.411 | 0.129 |
| index RoPE, 1 layer | 0.505 | 0.513 | 0.454 | 0.004 |
| best constant | 0.505 | -- | 0.506 | 0.088 |

**What the step table holds: a word's offset depends on its role, not its identity.** On every seed opposite direction
words cancel once the component common to the four directions is removed (0.002-0.012), synonyms collapse to one step
(cosine 1.000), and zeroing the direction steps drops accuracy to the constant floor (declared secondaries; Fig. 10).
Adverbs and fillers get steps of 0.004-0.03 of a move; asides get word steps that cancel within the sentence, leaving the
aside's noun 0.04-0.16 of a move off the place (analysis). That offset is why learned steps beat being told: a model whose
steps are restricted to the direction words stops at 0.970-0.974, its errors naming objects told in an aside at that
place, and offsetting its aside words by half a move lifts it to 0.999-1.000 (analysis). "Only actions move" is the wrong
separation in language: what is mentioned must not be filed where it is not. The same rule covers the what-to-where leak
with new objects in every episode: removing the objects' steps, or normalising the step input, lets every run converge
(16/16 SOLVED against 0/8 for MapWM; App. K3).

**Half the models also count moves, in channels they do not use.** On four of eight registered seeds 31-38 of 64 channels
drift between visits to the same cell (declared secondary), a per-move tick (Howard's Case II) beside the map. By
Proposition 2(b) such channels cannot serve retrieval by place, and they do not: 79-99% of retrieval usage sits in
channels whose tick is zero modulo pi, and removing the clock channels costs at most 0.009 on each clock model at ceiling,
while removing as many map channels collapses accuracy (analysis, 24 models; App. J2-J3).

![Figure 10](figs/fig_textworld.png)

*Figure 10 (`fig_textworld`). A one-layer path model on the torus walk told in English (n = 8). (a) Opposition of the
direction-word steps, |A + B| / |A| (0 = cancel): after removing the common component, 0.002-0.012 on every seed. (b) The
common component is small on four seeds (blue) and large on four (orange), where it points along the verbs' step: a
per-move clock beside the map. (c) Accuracy with every step, with non-direction steps zeroed, and with direction steps
zeroed: the last drops every seed to the constant floor (dashed). Eval mode.*

### Pixel observations

In the image torus (registered; App. K4) every observation of the 1024-move walk is a 28 x 28 image of its class (MNIST
digits and six Fashion-MNIST classes), a fresh instance at every visit (test-split images at evaluation), read by a small
CNN trained end to end. One path layer whose step the actions set scores 0.995 (8/8 solved, every seed at or above the
perception ceiling of 0.993, a perfect map voting with a standalone classifier) against 0.701 for index RoPE (0/8, below
retrace), +0.294 (Holm p 0.0003), and holds long-gap and wrap-only revisits at 0.994. Its phase never sees a pixel: what
this shows is that a content-addressed CNN with an action-driven phase reaches the perception ceiling, and that the revisit
loss alone trains the encoder (linear probe 0.985 against the standalone classifier's 0.980; index RoPE's encoder 0.965).
The batch's registered primary compared the default MapWM, whose step reads every token, with index RoPE and reads
unmeasured (0.668 against 0.701); after a pilot stop rule fired the path arm was changed to the action-driven step
(Amendment 2; the headline is for that arm, partly pilot-informed). Why MapWM fails here is a design point (Sec. 9).

*Table 7. Torus from pixels (registered; T = 1024, test-split images, held-out map, n = 8 per cell; App. K4).*

| model (1 layer) | observations | step set by | accuracy | solved | long gap / wrap-only |
|---|---|---|---|---|---|
| path | images | actions | **0.995** | 8/8 | 0.994 / 0.994 |
| path (MapWM), image embedding at gain 1 | images | every token | 0.994 | 8/8 | 0.993 / 0.991 |
| path (MapWM), image embedding at token scale | images | every token | 0.668 | 1/8 (5/8 collapsed) | 0.589 / 0.544 |
| index RoPE | images | -- | 0.701 | 0/8 | 0.482 / 0.483 |
| path | tokens | actions | 1.000 | 8/8 | 1.000 / 0.997 |
| floors | | | retrace 0.843, blank 0.508; ceiling 0.993 | | retrace 0.509 / 0.508 |

## 9. Discussion and related work

**What the two failure axes share.** For per-head rank 2 and for |.| steps a solution in the class exists, training keeps
or re-finds it, and training from scratch does not reach it: in Remark 2's terms the initialisation sits outside the basin
of a non-empty, stable solution set. Both failures end with a clock of the order of the map and strata that keep short-gap
revisits and lose the rest, and four times the computation or parameters reads that code better without finding the
rank-2 solution: the deficit is local to the content-to-angle map. For rank, the registered causal test names the
obstacle: forbid the clock and rank 2 solves. For the step, the evidence points to the increment and its start: a signed
increment under a non-negative readout learns, a |.| step started past pi learns on the ring, and the exp step learns in
the SSM on the ring from a start with many channels past pi but not in attention on the torus; losing the through-zero route does not by itself account for the cost
(+1/+2), and whether the barrier at pi or the missing route binds from a small start is half-read. MapEM's recency
deficit is a contrast: its exact solution is found with four times the budget (0.578 to 0.928; not registered), while rank
2 stays at 0/8 with twice the budget (App. N).

**Design guidance.** Keep the phase increment signed; a non-negative readout is harmless. If a non-negative increment is
needed, |.| and softplus from small starts are the forms that paid everywhere we looked; exp escaped in an SSM at one rate
and paid in attention on the ring and the torus. Give each head rank above the map's dimension (D + 1), with weight decay; where the
inverse pairs of actions are known, tying their steps to exact negatives removes the clock, and rank D then suffices.
Keep observations from stepping the phase. With pixels this is decisive (Table 7): when the image embedding enters the
step at token scale, 5 of 8 runs collapse, the encoder outputting a constant and the phase becoming a clock that ticks at
every observation slot; this is not the token-level leak (no per-instance offsets) but a collapse of encoder and phase
together. Scaling the image embedding down 30x, a total effect on the step input and the residual stream at once,
restores learning (8/8); since the action-driven model is fine at token scale, the cost belongs to the step. Measure map
use where retracing no longer suffices, by stratum, and read a phase's used-channel clock ratio, not its growth with the
token index.

**What and where.** In the long-walk regime the model that learns the map separates where from what (analysis; App. K1):
for the signed rank-4 model one shared position kernel per head times a per-pair content gain reproduces its accuracy
(0.998), and removing token identity within type costs 0.032 (Holm over 24 contrasts 0.08), while the non-negative and
index models lose 0.25-0.31. MapFormer showed the separation in the embeddings (their Fig. 9); the models rely on it where
a map is needed, in the sense TEM proposed.

**The integrator view and text.** A learned, signed rate can realise all three of Howard et al.'s cases, and which one a
channel realises is set by what the readout keys on: for targets keyed to a place the used channels are clock-free (clock
part over content part 0.064-0.079 on the torus, 0.06-0.23 in the text world; analysis), for k-back recency they are
clocks (21.8-22.1). On ordinary text at small scale (n = 3 pilots) the used channels are clock-dominated (9.9-12.3 on
TinyStories), MapWM matches index RoPE on TinyStories (1.5053 against 1.5050 nats per token), and at matched length path
integration costs on code (+0.0056 bits per byte, 3/3 seeds). With egocentric turn-and-forward actions, recoding them
allocentrically restores path integration's advantage (+0.050 to +0.488, n = 8, short walks; not registered; App. A5).

**Related work.** *Optimisation, not expressivity.* Shakerinava et al. (2026) find that input-dependent rotations in
unit-modulus SSMs can represent non-Abelian state tracking but do not learn it from scratch. Howe (2026), also
pre-registered, installs verified-exact solutions in Householder linear RNNs and finds that an additive input pathway
pulls Adam off them, and that deleting it lets training find the exact automaton, as forbidding the clock lets rank 2 find
the torus map here. Larson (2026) gives a rank law for matrix memories on group composition (Sec. 7.2). Grazzi et al.
(2025) and Mamba-3 own the representability of rotations with non-negative angles; ours is the learnability cost of |.|
and softplus steps in the rotary slot. A reparameterised weight w = |u| cannot change sign under gradient flow (Amid and
Warmuth 2020, Prop. 2; Woodworth et al. 2020); scoRNN fixes the +/-1 diagonal of its Cayley transform (Helfrich et al.
2018); the LRU initialises its phases near zero (Orvieto et al. 2023). With fixed frequencies the parameter symmetries of
rotary attention are per-plane complex scalars (Wang and Wang 2025; Tran et al. 2026); a learnable step adds the
reflection of Lemma 1. *Depth and recurrence.* Depth can substitute for recurrence through shortcuts (Liu et al. 2023;
Yao et al. 2021), error falls exponentially with depth in parallelizable models (Heo et al. 2026), shallow transformers
learn sensitive functions unreliably (Hahn and Rofin 2024), and Selective RoPE and PaTH (Yang et al. 2025) solve parity
with one layer; we give the rate at matched depth, the mechanism trained models use, and where it breaks.

## 10. Limitations

- The models are small (1-4 layers, 2 heads, d_model 128-256) and the tasks synthetic; natural text appears only as a
  three-seed pilot. Actions are allocentric translations. The LSTM and GRU baselines are trained on one map per run, as
  every torus model here; recurrent networks trained across many maps, gated SSMs and index attention given the integrated
  position as an absolute code are not tested.
- "Not found" is relative to the stated budgets (300-1800 epochs), and many failing cells have runs still descending.
  Learning rates for the |.| contrast were swept on the ring (four rates) and the torus (three, decay coupled to the rate,
  no decay-matched control; 5 of 8 |.| runs at its best rate still descending). At 2 layers and d_model 256 the accuracy
  gap is a quarter of the one-layer gap, so at scale part of the cost may be speed. No other optimiser was tried.
- The parameterisation result is shown for |s| and softplus(s), and for exp(s) in attention (ring at two fixed rates; torus
  at the best of two, softplus's optimum not bracketed, most runs still descending, the exp batch partly pilot-informed).
  The exp step showed no cost in the SSM on the ring at one rate (1e-3; unstable at 3e-3), where the signed anchor had
  partial codes on four seeds; whether the mixer, the task or the start explains this is not separated (no SSM on the
  torus, no exp started below pi). The start lever is registered on the ring and does not transfer to the torus within
  900 epochs.
- Why the barrier classes fail from a small start is not settled. The registered test of Lemma 1's learnability reading
  (+1/+2) is unmeasured and did not confirm it; the route-versus-barrier 2x2 (ring, one start, one readable rate) has one
  registered half (a route to the wrap does not carry |.| past pi) and is unmeasured as a whole.
- Rank results use two heads and a latent shared across heads where MapFormer's is per head (per-head arms are reported).
  Within 900 epochs rank 4 learned only with weight decay 0.05 on all parameters (8/8 against 1/8 without; 5 of 7 failures
  still descending), credited to the optimiser setting. Rank 2 failed at three recipes, but neither alternative is a
  recipe at which rank 4 is known to learn, there is no lower-decay or combined arm, and 12 of 24 rank-2 runs were still
  descending; the claim is that rank 2 fails at each of three recipes, not that no recipe rescues it. The clock-exit test
  is PARTIAL with most unsolved runs descending.
- The torus exchange rate is a lower bound at n = 4, replicated on four fresh seeds, with 1 and 4 layers only.
- The pixel test uses class images on the discrete torus, one top-down image per visit, one layer, one length and one
  recipe. Its registered primary (default MapWM against index RoPE) was unmeasured; the reported path arm replaced it after
  a pilot stop rule, and the cost of an image-driven step was predicted from pilots and confirmed on fresh seeds, not blind;
  the gain effect is a total effect at two values.
- The text world is scripted (58 words, context-free direction words), and its walks are short enough that retrace scores
  0.934; the map evidence there is the revisits retrace cannot reach and the step table.
- The theory of Sec. 4 was developed after the registered batches, except the clock-route batch, registered to test its
  rank account; the hiding feedback is a conjecture and the per-channel dynamics of |.| are not derived. Checks marked
  (analysis) are post hoc, on one batch per arm, with thresholds listed in App. O; one CPU analysis (the step-map basin) is
  not deterministic and is reported as exploratory.

## References

Titles and author lists checked against arXiv, OpenReview, Crossref, ACL Anthology or PMLR metadata on 2026-10-15/16
(Heo et al., Howe and Larson on 2026-10-26); entries marked * were not re-checked (venue, or the whole entry where stated in
the notes file).

- Alipour, A., James, T., Brown, J. W. and Tiganj, Z. Self-supervised learning of scale-invariant neural representations
  of space and time. J. Comput. Neurosci. 53:131, 2025*.
- Amid, E. and Warmuth, M. K. Reparameterizing mirror descent as gradient descent. NeurIPS 2020*; arXiv:2002.10487.
- Eldan, R. and Li, Y. TinyStories: how small can language models be and still speak coherent English? arXiv:2305.07759,
  2023*.
- Golovneva, O., Wang, T., Weston, J. and Sukhbaatar, S. Contextual position encoding: learning to count what's important
  (CoPE). arXiv:2405.18719, 2024.
- Gopalakrishnan, A., Csordas, R., Schmidhuber, J. and Mozer, M. C. Decoupling the "what" and "where" with polar
  coordinate positional embedding (PoPE). arXiv:2509.10534, 2025*.
- Grazzi, R., Siems, J., Zela, A., Franke, J. K. H., Hutter, F. and Pontil, M. Unlocking state-tracking in linear RNNs
  through negative eigenvalues. ICLR 2025*; arXiv:2411.12537.
- Hahn, M. and Rofin, M. Why are sensitive functions hard for transformers? ACL 2024 (long papers); arXiv:2402.09963.
- Helfrich, K., Willmott, D. and Ye, Q. Orthogonal recurrent neural networks with scaled Cayley transform. ICML 2018*;
  arXiv:1707.09520.
- Heo, G., Ngotiaoco, T., Irie, K., Gershman, S. J. and Sabatini, B. L. Why depth matters in parallelizable sequence
  models: a Lie algebraic view. arXiv:2603.05573, 2026.
- Hewitt, J., Hahn, M., Ganguli, S., Liang, P. and Manning, C. D. RNNs can generate bounded hierarchical languages with
  optimal memory. EMNLP 2020*; arXiv:2010.07515.
- Howard, M. W., MacDonald, C. J., Tiganj, Z., Shankar, K. H., Du, Q., Hasselmo, M. E. and Eichenbaum, H. A unified
  mathematical framework for coding time, space, and sequences in the hippocampal region. J. Neurosci. 34(13):4692-4707,
  2014.
- Howe, G. L. The automaton underneath: the additive input pathway is a parasitic attractor for state tracking in
  Householder linear RNN. arXiv:2609.18966, 2026.
- Kabir, M. R., Mochizuki-Freeman, J. and Tiganj, Z. Deep reinforcement learning with time-scale invariant memory. AAAI
  2025*; arXiv:2412.15292.
- Lahoti, A., Li, K. Y., Chen, B., Wang, C., Bick, A., Kolter, J. Z., Dao, T. and Gu, A. Mamba-3: improved sequence modeling
  using state space principles. ICLR 2026; arXiv:2603.15569.
- Larson, S. The rank the task demands: a causal rank law for matrix memories trained on group composition.
  arXiv:2609.12259, 2026.
- Li, B., Wu, Y., Huang, N. and Zhang, W. GridPE: a grid cell-inspired unified position embedding for
  arbitrary-dimensional spaces. arXiv:2406.07049, 2024*.
- Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A. and Zhang, C. Transformers learn shortcuts to automata. ICLR 2023;
  arXiv:2210.10749.
- Mochizuki-Freeman, J., Maini, S. S. and Tiganj, Z. Characterizing neural activity in cognitively inspired RL agents
  during an evidence accumulation task. IJCNN 2023*.
- Movahedi, S., Carstensen, T., Afzal, A., Hutter, F., Orvieto, A. and Cevher, V. Selective rotary position embedding.
  ICLR 2026; arXiv:2511.17388.
- Orvieto, A., Smith, S. L., Gu, A., Fernando, A., Gulcehre, C., Pascanu, R. and De, S. Resurrecting recurrent neural
  networks for long sequences (LRU). ICML 2023*; arXiv:2303.06349.
- Rambaud, V., Mascarenhas, S. and Lakretz, Y. MapFormer: self-supervised learning of cognitive maps with input-dependent
  positional embeddings. arXiv:2511.19279 (v4, 10 May 2026).
- Sarrof, Y., Veitsman, Y. and Hahn, M. The expressive capacity of state space models: a formal language perspective.
  NeurIPS 2024; arXiv:2405.17394.
- Shakerinava, M., Khavari, B., Ravanbakhsh, S. and Chandar, S. The expressive limits of diagonal SSMs for
  state-tracking. ICLR 2026; arXiv:2603.01959.
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B. and Liu, Y. RoFormer: enhanced transformer with rotary position
  embedding. Neurocomputing 568:127063, 2024*; arXiv:2104.09864.
- Tran, V.-H., Bui, V. K., Trinh, V.-H., Lai Ngoc, T. and Nguyen, T. M. Functional equivalence in attention: a
  comprehensive study with applications to linear mode connectivity. ICML 2026 (PMLR 306); arXiv:2606.17830.
- Veisi, A., Fartoot, D. and Amirzadeh, H. Context-aware rotary position embedding (CARoPE). arXiv:2507.23083, 2025.
- Wang, H. and Wang, K. Complete characterization of gauge symmetries in transformer architectures. Symmetry and Geometry
  in Neural Representations (NeurReps), PMLR 282:622-663, 2025.
- Whittington, J. C. R., Muller, T. H., Mark, S., Chen, G., Barry, C., Burgess, N. and Behrens, T. E. J. The
  Tolman-Eichenbaum Machine: unifying space and relational memory through generalization in the hippocampal formation
  (TEM). Cell 183(5):1249-1263, 2020.
- Whittington, J. C. R., Warren, J. and Behrens, T. E. J. Relating transformers to models and neural representations of the
  hippocampal formation (TEM-t). ICLR 2022; arXiv:2112.04035.
- Woodworth, B., Gunasekar, S., Lee, J. D., Moroshko, E., Savarese, P., Golan, I., Soudry, D. and Srebro, N. Kernel and
  rich regimes in overparametrized models. COLT 2020*; arXiv:2002.09277.
- Yang, S., Shen, Y., Wen, K., Tan, S., Mishra, M., Ren, L., Panda, R. and Kim, Y. PaTH attention: position encoding via
  accumulating Householder transformations. NeurIPS 2025; arXiv:2505.16381.
- Yao, S., Peng, B., Papadimitriou, C. and Narasimhan, K. Self-attention networks can process bounded hierarchical
  languages. ACL 2021*; arXiv:2105.11115.
- Zhang, Y., Chen, Z., Liu, Y., Qin, Z., Yuan, H., Xu, K., Yuan, Y., Gu, Q. and Yao, A. C.-C. Group representational
  position encoding (GRAPE). ICLR 2026; arXiv:2512.07805.
