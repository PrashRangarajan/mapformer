# A monotone rotary phase can represent a map but does not learn one: sign, wrapping and the path through zero

*Draft 1, 2026-10-14. Workshop length (~5 pages + appendix). Every number is from a pre-registered batch unless marked
(post hoc) or (pilot); the source file is named in `docs/paper/CLAIM_MAP_clock_map.md`. Pending: WRAP_COPE (Sec. 5.4).*

## Abstract

Rotary position encodings accumulate a phase, theta_t = omega * sum_{u <= t} Delta_u. The step Delta can be a constant (index
RoPE), a non-negative function of content (CoPE, CARoPE, GRAPE-AP) or a signed function of content (MapFormer, Selective
RoPE, Mamba-3). These are the three cases of Howard et al.'s (2014) neural integrator -- elapsed time, path length and
position -- carried into the imaginary (rotary) slot. We study what the sign of the step buys on tasks that need a map, where
opposite moves must cancel. In the rotary slot a non-negative step can represent any signed code, because angles wrap mod
2 pi. Yet on a 1D ring and a 2D torus, at the training length, non-negative steps learn the map on 0-3 of 8 seeds against 8 of
8 for signed steps, and three times the training does not help. An exact non-negative solution exists at a trained weight
scale and training keeps (ring) or re-finds (torus) it, so the failure is one of search. The mechanism is specific: a signed
step reaches the cancelling code by moving one move's angle through zero, which it does even from the start a non-negative
step would have (7/8); a non-negative step cannot cross zero, so its only cancelling solution is the wrap, which training
reaches when the opposite moves' initial angles sum past pi (8/8) and not otherwise (0/8, and 2/8 for large starts below pi).
Faithful implementations of two published monotone methods (CARoPE, CoPE) also fail to learn these maps. One layer of signed
path integration beats index attention at one layer, four layers and a four-times-looped block on the torus.

## 1. Introduction

A rotary position code is an integrator. Index RoPE adds a constant per token, so the phase difference between two tokens is
their distance in tokens; content-dependent variants let each token set its own step. Two such families exist: steps that
are non-negative by construction -- CoPE's sigmoid gates, CARoPE's (0, 1) rate, GRAPE-AP's non-negative frequencies -- and
steps that may be negative -- MapFormer's path-integrated phase, Selective RoPE, Mamba-3's data-dependent rotation.

Neuroscience has named these three cases. Howard et al. (2014) drive one bank of integrators by a rate alpha(t): a constant
rate yields elapsed time, a speed yields path length, a signed velocity yields position ("like a bead on a wire": different
routes to one place give the same code). Path length is a clock; position is a map. We take the three rotary families to be
these three cases with an imaginary decay (Sec. 2). The identification is a remark, not a theorem, and a reader may find it
obvious; its use here is that it says exactly which tasks should separate the families -- those where opposite steps must
cancel -- and what to measure.

**What we find.**
1. In the rotary slot, sign is a matter of learnability, not representation: a non-negative step can represent any signed
   code by wrapping, but does not learn the map (Sec. 4.2), and an exact non-negative solution exists and survives training
   (Sec. 4.3).
2. The reason is mechanistic: the constraint removes the path through zero; a start whose opposite moves' angles sum past pi
   restores learning completely (Sec. 4.4).
3. Two published monotone methods, implemented faithfully, also fail to learn these maps; CARoPE is no better than the index
   (Sec. 4.5).
4. Where a map is needed, one layer of signed path integration beats one, four and looped-four layers of index attention
   (Sec. 4.1).
The effect is one-sided: on a content-clock task the non-negative step is not established to be better (Sec. 4.6).

## 2. Three integrators in the rotary slot

Howard et al. write dF/dt = alpha(t) (-s F + f(t)). With real s this is a decaying memory; with imaginary s = i omega the
same equation is a rotation, F_t = exp(i omega int alpha), i.e. a rotary phase theta = omega * int alpha. Discretised per token,
alpha_t is the step: the selective-SSM Delta_t, MapFormer's W_out W_in x_t.

| case (Howard) | rate alpha_t | the interval sum measures | rotary family | examples |
|---|---|---|---|---|
| I | constant | elapsed tokens | index RoPE | RoPE |
| II | non-negative, content-dependent | path length (a content clock) | monotone content phase | CoPE, CARoPE, GRAPE-AP |
| III | signed, content-dependent | net displacement (a map) | signed content phase | MapFormer, Selective RoPE, Mamba-3 |

**Remark (wrapping).** For real s, Cases II and III differ in what they can represent: a non-negative rate cannot undo a step.
In the rotary slot they do not: angles live mod 2 pi, so a step 2 pi / omega - Delta reproduces -Delta exactly in every channel
where it is used, and a non-negative rotary code can represent any signed one (constructed in Sec. 4.3). If the sign matters
in practice, it must matter for learning. The cost of wrapping is that the two opposite moves' angles must sum to 2 pi rather
than to 0.

## 3. Setup

**Model.** MapFormer-WM (Rambaud et al.): one transformer layer, 2 heads, d = 128, whose query-key rotation is the
path-integrated phase theta = omega * cumsum(W_out W_in x_t), shared rank 4. The sign arms differ only in the step: Signed
(as is), |.| (Abs: the sign removed and nothing else), softplus (GRAPE-AP-like). For the mechanism we use
observation-step-free versions in which only the move tokens have steps (so a non-negative copy of a signed solution needs no
wrapped observation steps). Index RoPE uses the canonical frequency ladder; deeper and looped index models for Sec. 4.1.
**Tasks.** A 2D torus walk (64 x 64, T = 1024 moves interleaved with observations; the model predicts the observation at
revisited cells; the observation map is redrawn for every evaluation trajectory, so accuracy is transfer). A 1D ring (32
cells, T = 128) whose backward-step probability is a dial (p_plus = 0.5: steps cancel). Navigation told in English words.
k-back recency (a content clock).
**Protocol.** Every contrast below was pre-registered with its decision rule, run on n = 8 fresh seeds (n = 4 in Sec. 4.1),
trained and tested at the same length, and checked by an independent code review before launch. SOLVED = final 5% of the
training loss < 0.05. Floors (best action-stream n-gram; blank prediction) are reported beside accuracies. Pilots used seeds
outside the batch.

## 4. Results

### 4.1 Where a map is needed, one path layer beats several index layers
On the torus at T = 1024, one layer of signed path integration scores 1.000 (4/4 SOLVED) against index RoPE at one layer
0.719, four layers 0.860 and one block looped four times 0.864 (0/4 each; complete separations, p 0.029; floors 0.507 /
0.576). Depth and looping lift the index model only on revisits within 128 moves (0.74-0.79 -> 0.90-0.96); long-gap and
wrap-around revisits stay at the floor at every depth (Fig. 1). The same holds when the walk is told in English: path 0.969
vs index 0.505 (p 0.0002, 7/8 vs 0/8). On Dyck-2 at matched depth one path layer is worth about three index layers (appendix);
on the torus four are not enough.

### 4.2 Non-negative steps do not learn the map
On the torus at the training length, the non-negative step costs -0.177 in accuracy (p 0.0002; SOLVED 0/8 vs 8/8; softplus
-0.218). On the ring, where the map is one-dimensional, |.| solves 2/8 and 1/8 at p_plus 0.5 and 0.75 against 8/8 (Fisher
p 0.007, 0.0014); three times the training leaves it at 3/8. The signed models cancel opposite moves mod 2 pi in 64-98% of
channels; the non-negative models in 3-11%, including the runs that solve (Fig. 2) -- they never find the wrapped code. The
observation-step-free versions show the same (torus 0/8 vs 8/8; ring 0/8 vs 8/8; Secs. 4.3-4.4).

### 4.3 The non-negative solution exists and survives training
To show the failure is search, not capacity, we convert each solved signed observation-step-free model exactly into
non-negative form: the move latents are re-solved orthogonal (a free gauge, since only the moves have steps), and each
channel's steps are set to (omega Delta mod 2 pi) / omega', omega' = max(|omega|, 1), so every angle is unchanged mod 2 pi
(error <= 1.4e-6 rad; frozen accuracy equal to the source on 8/8 seeds, weights within ~5x of the signed model's). A naive
construction that keeps the signed model's move latents is also exact but ill-conditioned (the trained latents are nearly
antiparallel), and Adam loses it within ten epochs -- the right existence test has to be posed at a trained scale.
On the ring the constructed non-negative model trained for another 300 epochs stays solved on 8/8 seeds and never leaves the
solved region; on the torus it is knocked out early (peak loss 0.11-0.21 vs <= 0.017 for a matched signed copy) and re-finds
the solution (8/8 SOLVED, 1.000 at the end). The same classes from scratch solve 0/8 (ring and torus). A solution exists and
is stable in task terms; training does not find it. (Training also moves the held ring model off the exact cancelling code,
87% -> 18% of channels; on the torus 100% -> 69%.)

### 4.4 Why: the constraint removes the path through zero
We vary only the initial move angles of observation-step-free ring models (every other weight identical at a seed; Table 2).

| initial move angles | signed | non-negative |
|---|---|---|
| default (small, signs independent) | 8/8 | 0/8 |
| |default|: both moves rotate the same way, small (the non-negative model's own start) | **7/8** | 0/8 |
| large, but the two moves' angles sum to less than pi | -- | 2/8 |
| large, summing to more than pi | 8/8 | **8/8** |

Three registered contrasts decide it. (i) From the small same-direction start the sign matters: 0/8 vs 7/8 (p 0.0014). The
signed model gets out: at the end 98% of its channels have opposite-signed move angles, and none of its cancelling channels
wrap -- it cancels by moving one move's angle through zero. (ii) A start whose angles sum past pi rescues the non-negative
step completely (8/8 vs 0/8, +0.109 [0.085, 0.135]), and from that start the sign makes no difference (8/8 vs 8/8); both cancel
by wrapping (100% / 88% of their cancelling channels have raw sums past pi). (iii) It is the sum, not the size: a large start
whose sums stay below pi solves 2/8 (vs 8/8; p 0.007). The large start also makes the angles move faster under Adam (13-22x
per step), which the sub-pi control matches only partly (~5x).
So the constraint does not forbid the map; it removes the route to it from small starts. A signed step reaches cancellation
through zero; a non-negative one must reach the wrap, and does so when it starts past the midpoint.
*Design implication (one task):* initialise a Case-II rotary step near the wrap, not near index RoPE.

### 4.5 Published monotone methods
Implemented faithfully (CARoPE: per-head rate in (0, 1), channel i steps f^i, initialised at index RoPE; CoPE: Listing 1 of
the paper), neither learns the maps. CARoPE cannot represent cancellation (its per-token steps are at most 1 rad); it scores
0.667 on the torus, *below* index RoPE's 0.722 (p 0.04), and equals the index on the ring (0.741 vs 0.730). CoPE can represent
ring cancellation (a period-2 position table read at the gated count; constructed) but learns it on 0/8 seeds (0.887 vs signed
1.000, p 0.0002), though its content-gated count beats the index by +0.157. Starting CoPE's position table near the wrap
does not rescue it measurably (8 seeds per arm, bimodal; sinusoidal starts trend better, 0.87-0.94 vs 0.84, all contrasts
unmeasured): in CoPE the wrap sits in a free per-count table, which training moves to period 2 from any start; the monotone
part is the gate, which must become slightly asymmetric between the two moves. The start lever applies to monotone rotary
steps, where the step itself must carry the wrap.

### 4.6 The other direction is not established
If sign is a matter of fit, a non-negative step might learn a content clock faster than a signed one. On k-back recency it
trends that way twice and is unmeasured twice (loss AUC -0.197, p 0.053; on fresh seeds -0.106, p 0.19). We do not claim a
double dissociation. On natural text (TinyStories, code; pilot) the free signed phase learns a clock (phase ~ token index,
R^2 0.994-0.999) and buys nothing over the index at our scale; CoPE reports gains at larger scale.

## 5. Related work
Howard et al. (2014) and the Howard and Tiganj groups' machine-learning work keep the decay real and the rate fixed or given;
they explicitly set aside complex s. GridPE derives RoPE from a velocity-controlled oscillator (Case III with given
velocity). Sign and negative eigenvalues for state tracking: Sarrof et al. (2024), Grazzi et al. (2025), Khavari et al.
(2025) -- our sign cost replicates theirs in a navigation regime; what we add is that in the rotary slot it is a learnability
effect with an identified mechanism. Signed content rotations: Selective RoPE, Mamba-3, GRAPE (which also gives the
taxonomy); monotone: CoPE, CARoPE. MapFormer (Rambaud et al.) and TEM introduced path-integrated position for cognitive maps;
MapFormer argues that CoPE cannot represent an inverse -- true for its counts read linearly, not for a periodic table or a
rotary phase. Depth and prefix sums: Liu et al. (2023), PaTH.

## 6. Limitations
Small models (one layer on the map tasks), synthetic map tasks, one optimiser recipe per task, fixed budgets (learnability is
budget-scoped, although 3x did not help); the mechanism is shown on the ring only; the large start is also a more mobile
start; n = 4 in the depth factorial; no recurrent baselines; CoPE tested on the ring only (its gated count is quadratic and
costs ~8 h per torus run). The clock side is unestablished.

## Figures and tables
- Table 1 (Sec. 2). Table 2 (Sec. 4.4).
- Fig. 1: `docs/paper/figs/fig_factorial_strata.png` (accuracy by revisit stratum, torus).
- Fig. 2: `docs/paper/figs/fig_wrap_share.png` (share of cancelling channels per run).
- Fig. 3 (to make): per channel, initial vs final move angles (a(+1), a(-1)) for the four starts of Table 2 -- signed points
  crossing the axes to the anti-diagonal, non-negative ones reaching the a(+1) + a(-1) = 2 pi line only from beyond pi.

## Appendix plan
A models and recipes; B floors and gates; C per-seed tables; D the constructions and why the naive one fails; E the toy model
(exploratory; reproduces the asymmetry without a transformer; its stronger "start, not constraint" claim did not transfer);
F Dyck matched depth; G navigation in words; H faithful CARoPE / CoPE and the CoPE cancellation construction.
