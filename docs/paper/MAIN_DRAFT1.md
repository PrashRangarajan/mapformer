# Representable but not found: what path-integrated rotary attention learns, and how it fails

*Main paper, draft 1 (2026-10-18), from `docs/paper/OUTLINE_main_v2.md`. Appendix: `docs/paper/MAIN_APPENDIX1.md`;
full theory appendix with check scripts: `docs/paper/APPENDIX_theory_v3.md`. Word counts, the source of every number and
the deviations from the outline: `docs/paper/MAIN_DRAFT1_notes.md`. Delete this line before submission.*

## Abstract

Path-integrated rotary attention (MapFormer) replaces RoPE's token index with a learned phase that sums a
content-dependent step per token. We ask what such models learn and how they fail. On the standard short-walk benchmark a
map-free rule that retraces the last run of moves already scores 0.934, so we train and test on 1024-move torus walks,
where that rule scores 0.843: one path-integrated layer reaches 0.998 (8 of 8 seeds solved) against 0.731 for index RoPE
(0 of 8). At matched depth one path layer is worth about three index layers on Dyck-2 and a biased 1D walk, and more than
four on the torus, where index depth helps only short-gap revisits. Told in English, the walk is learned from words
(0.979 vs 0.505), with opposite direction words cancelling. Where these models fail, they fail by search, not capacity.
With rank 2 per head the content-to-angle map solves the torus on at most 2 of 8 seeds, against 8 of 8 at rank 4,
although a rank-2 solution exists and training keeps it. A non-negative step represents the map exactly and keeps or
re-finds a constructed solution, yet from scratch solves 0 of 8 against 8 of 8 for signed steps, a gap that persists, on
a ring, at each step's best of four learning rates. Both failures are optimisation, not expressivity.

## 1. Introduction

Rotary position encodings are integrators. Index RoPE (Su et al. 2024) adds a constant angle per token; content-dependent
variants let each token set its own step (CARoPE, Veisi et al. 2025; CoPE, Golovneva et al. 2024; Selective RoPE,
Movahedi et al. 2026; Mamba-3, Lahoti et al. 2026; MapFormer, Rambaud et al. 2026). When the step is driven by actions,
the summed phase is a position on a map, and attention compares net displacements instead of token distances. MapFormer
built this into a transformer and showed that it helps navigation. We ask what such a model actually learns, how much it
buys over index attention, and why training sometimes misses a map the model can represent.

MapFormer shows that path-integrated attention beats index RoPE, TAPE and CoPE baselines with up to four layers on 1D and
2D grid navigation with 128-step walks (and PathAtt in 2D), generalises out of distribution to sparser and denser walks,
extrapolates to deeper Dyck-2 strings with one layer, and learns opposite latents for opposite actions (their Table 2,
Figs. 3 and 9). Three questions remain open. How much of the short-walk benchmark needs a map at all? What is one path
layer worth in index layers when both are trained and tested at the same length and depth? And why does the model
sometimes fail on its home task, for which it can represent the solution?

We answer each in turn. Measured against map-free floors, short walks are mostly solvable by retracing; at 1024 moves the
floor drops and a single path layer still solves the map. Trained and tested at matched depth, one path layer is worth
about three index layers on Dyck-2 and on a biased 1D walk, and more than four on the 2D torus. The same mechanism learns a
map from navigation told in words. And its failures share one signature: a solution in the constrained class exists and
training keeps it, but training from scratch does not find it.

**Contributions.**

1. *A sharper measurement of the advantage.* A map-free retrace rule scores 0.934 on MapFormer's short walks, so we
   measure on 1024-move torus walks against measured floors (blank 0.506, n-gram 0.575, retrace 0.843). There one path
   layer scores 0.998 against index RoPE's 0.731 (8/8 vs 0/8 seeds solved, p 0.0002), and index depth or a looped block
   lifts only short-gap revisits (to about 0.94), while long-gap and wrap-only revisits stay at the floor (0.46-0.51).
2. *An exchange rate at matched depth.* One path layer is worth about three index layers on Dyck-2 trained and tested at
   nesting depth 12 (position effect +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers) and on a biased 1D walk (three index
   layers reach 0.998-0.999 against path 1.000), and more than four on the 2D torus (four index layers 0.860 vs 1.000).
3. *Failure by search, not capacity, along two axes.* With per-head rank 2 the content-to-angle map solves the torus on
   0-2 of 8 seeds (rank 3: 6/8; rank 4: 8/8); twice the budget leaves it at 0/8 (rank 4: 7/8); a rank-2 solution exists
   (0.9955) and is held under training. A non-negative (|.|) step represents the map exactly and keeps or re-finds a
   constructed solution, yet from scratch solves 0/8 against 8/8 for signed steps on the torus (-0.177), a gap that holds
   without weight decay, in a 2-layer d 256 model, and at each step's best of four learning rates (2/8 vs 8/8, p 0.007).
4. *Maps from words.* Navigation told in English is learned by one path layer (0.979 vs index 0.505, 6/6 vs 0/6 seeds,
   p 0.0022), and every model encodes opposite direction words as cancelling steps.

## 2. Background

**2.1 Rotary position as an integrator.** Write the phase of rotary channel c (a head and a 2D block) at token t as
theta_{c,t} = omega_c Phi_{c,t}, with Phi_{c,t} = sum_{u <= t} Delta_{c,u} the channel's summed step. The channel's
contribution to the score of query t and key t' then depends on position only through exp(-i omega_c (Phi_{c,t} -
Phi_{c,t'})), the identity behind Mamba-3's Proposition 3 (App. E). Howard et al.'s (2014) rate-driven integrators give
the vocabulary. A constant rate codes elapsed time (index RoPE); a non-negative rate codes path length (CARoPE, CoPE, and
GRAPE's additive phase in the real slot, Zhang et al. 2026); a signed rate codes position on a map (MapFormer, Selective
RoPE). TEM and TEM-t (Whittington et al. 2020, 2022) frame such a map as a "where" learned apart from the "what".

**2.2 MapFormer.** MapFormer's working-memory model (MapWM) rotates content queries and keys by theta_t = omega *
cumsum(W_out W_in x_t), a low-rank content-to-angle map; observations learn near-zero steps and opposite actions opposite
steps (their Fig. 3d). In their Table 2, one-layer MapWM with rank 2 solves 2D navigation at 128 steps, rank 1 does not,
and RoPE, TAPE and CoPE up to four layers fall short; they attribute CoPE's failure to the sign of its gate, which cannot
represent an action and its inverse. We build on exactly this model. Our implementation reproduces their Table 2 2D
result (0.989 for MapWM and 0.987 for MapEM, against 1.00 reported in v4 and 0.99 / 1.0 in v1; App. A).

**2.3 Representability.** Real non-negative eigenvalues cannot track parity (Sarrof et al. 2024, Thm. 2; Grazzi et al.
2025, Thm. 1). Rotations escape this, and rotations by non-negative angles generate cyclic groups (Grazzi et al., Thm. 3;
Mamba-3, Sec. 3.2). In the rotary slot, then, the sign of a channel's step does not limit what that channel can
represent; Proposition 1 below extends this to our rank-limited step maps, so any difference between signed and
non-negative steps is a difference in what training finds.

**2.4 Optimisation, not expressivity.** Shakerinava et al. (2026) find that input-dependent rotations in unit-modulus
SSMs can represent non-Abelian state tracking but do not learn it from scratch, while near-solution starts recover it.
Orientation-fixing parameterisations have a history: scoRNN fixes the +/-1 diagonal of its Cayley transform during
training (Helfrich et al. 2018); a reparameterised weight w = |u| cannot change sign under gradient flow (Amid and Warmuth
2020, Prop. 2; Woodworth et al. 2020); the LRU initialises its phases near zero (Orvieto et al. 2023). Depth can substitute
for recurrence through shortcuts (Liu et al. 2023), though shallow transformers learn sensitive functions unreliably
(Hahn and Rofin 2024), while Selective RoPE and PaTH (Yang et al. 2025) solve parity or Flip-Flop with one layer. We
bring these lines to path-integrated attention, measuring the depth exchange rate and the learnability gap in one model
family.

**2.5 Two results we use** (proofs in App. E). *Proposition 1 (representability by wrapping).* If at most r tokens step,
a |.| model of rank r reproduces any signed angle table exactly mod 2 pi. Annihilating the observation embeddings in W_in
leaves only the moves stepping, so every |.| model in this paper can represent the map. Softplus can too: with the
observations annihilated each steps softplus(0) = ln 2, and omega' = 2 pi / ln 2 makes that step a full turn.
*Lemma 1 (relative-sign gauge).* Negating a channel's steps (through omega_c) together with the second coordinate of
that channel's query and key projections leaves every attention score unchanged, since J R(phi) J = R(-phi) for
J = diag(1, -1). A non-negative step therefore fixes only the relative signs of the tokens that step in a channel. It
excludes no map function, only parameter points: for an inverse pair (two tokens that undo each other, such as
opposite moves) it removes cancellation through zero, angles (a, -a),
while the wrapped code (a, 2 pi - a), the same function, remains. With fixed frequencies the parameter symmetries of
rotary attention are per-plane complex scalars (Wang and Wang 2025; Tran et al. 2026); a learnable step adds this
reflection.

## 3. Setup

**Models.** MapWM with one layer (up to four where stated), 2 heads, d_model 128, and step Delta = W_out W_in x through a
rank-r latent. Unless stated, r = 4 and the latent is shared across heads; MapFormer's latent is per head, and per-head
variants are named wherever rank is compared. *Per-head rank* is the rank of the map from token to the angles of one head.
Three step parameterisations: *signed* (Delta = s = W_out W_in x), *|.|* (Delta = |s|, applied per channel after W_out),
and *softplus* (Delta = softplus(s), so every step starts at ln 2). *Observation-step-free* variants give steps to the
move tokens only. Each head has 32 two-dimensional rotary blocks with frequencies omega_i = 2 pi (1/Delta_max)^(i/31),
Delta_max the grid size, as in MapFormer; omega is a free parameter whose sign is not constrained, so a channel's
direction is never fixed by the parameterisation, only the relative signs of its tokens. The layer is otherwise a
standard pre-norm transformer block (feed-forward width 4 d_model, dropout 0.1; about 204k parameters at one layer). The
index baseline is RoPE with the canonical ladder base^(-2c/d), at 1-4 layers or as one block looped four times. On Dyck-2
we also pair PoPE (Gopalakrishnan et al. 2025), a rotary variant with a polar score, with its path-integrated version
MapPoPE (App. F, L). Faithful CARoPE and CoPE serve as the published monotone comparison.

**Training and evaluation.** AdamW (learning rate 1e-3, weight decay 0.05), 5% linear warm-up then cosine decay to 10%
of the peak, 98 batches per epoch (16 walks per batch on the 1024-move torus, 128 on the ring and the 128-move torus),
cross-entropy on revisited observations only (Dyck-2: learning rate 1e-4, next-token loss; App. A). Torus models are scored on 100 walks over one held-out map; ring models on 400 walks with freshly
drawn maps. A run that is not SOLVED (defined below) is *STALLED* if its final-10% mean loss is within 5% of the
preceding 10%, and *DESCENDING* otherwise.

**Tasks.** (i) *2D torus*: a 64 x 64 torus walked by runs of 1-10 repeated moves, interleaved with observations drawn
from 16 objects; the map is redrawn at evaluation, so every score is transfer. Walks have T = 1024 moves (2,048 tokens;
MapFormer's T = 128 for the floor analysis), 900 epochs. (ii) *1D ring* of 32 cells, T = 128, forward-step probability
p_plus (0.5 balanced; 1.0 never reverses), maps redrawn per walk, 300 epochs. (iii) *Dyck-2* at length 32 and nesting
depth 12, scored by Hewitt et al.'s (2020) distance-averaged closing accuracy (chance 0.5; best map-free floor 0.594).
(iv) *Text world*: the torus walk rendered as English (58 words: three synonyms per direction, movement-free verbs,
fillers, asides), T = 1024 words. (v) 2D and 3D tori of other sizes for the test of rank against dimension. The loss and
the accuracy are on *revisits*: observations at cells visited before, the only predictable targets.

**Floors.** For each task we measure the best map-free predictor: always-blank, the best n-gram over orders 0-5, and
*retrace* (while the current run of moves undoes the previous run, copy the observation seen when the walk was last at
the cell it is returning to). Revisits are split into *strata*: short-gap (fewer than 128 moves since the last visit),
long-gap, and *wrap-only* (the cell was seen before only from a different unwrapped position, a full lap of the torus
away). Floors, strata floors and cancelling shares were computed after the batches on the stored streams and checkpoints;
we mark such numbers "(analysis)" where they first appear. App. B lists every floor.

**Protocol.** Every verdict was pre-registered with its decision rule against the measured noise floor, with n = 8 seeds
per arm unless stated, every model trained and tested at the same length (and nesting depth), *SOLVED* defined as a mean
training loss below 0.05 over the final 5% of epochs, and exact permutation (accuracy) and Fisher (SOLVED) tests. Before
any result was read, 12 of the 21 main-text batches were checked by an independent, results-blind code review, and the analyses of 20
were re-run from stored checkpoints by an independent reproduction audit that reproduced every one; App. A records both
per batch.

## 4. A sharper measurement of the advantage

**Short walks mostly do not need a map (analysis).** On MapFormer's T = 128 torus walks retrace scores 0.934 (blank
0.507, n-gram 0.598), and no revisit is reachable only by wrapping around the torus (wrap-only share 0.000). At T = 1024
retrace falls to 0.843 (blank 0.506, n-gram 0.575) and 7.9% of revisits are wrap-only (Table 1). The long-walk regime is
where map use becomes measurable. At T = 128 the path model still beats index attention (+0.243 position effect at rank 2;
path 0.971 at rank 2 and 1.000 at rank 4, index 0.805; n = 8), but index RoPE falls below retrace there and the rank-2 path
model clears it by only 0.04.

Retrace is strong on short walks for a simple reason. The walk moves in runs of repeated moves, and a run that reverses
the previous one returns, step by step, to cells seen moments earlier. Over 128 moves nearly every revisit is of this
local kind, so a model can score well above 0.9 without integrating anything. Over 1024 moves the walk drifts far from
where it started, revisits arrive after hundreds of moves or across the wrap of the torus, and a rule that only undoes
the last run reaches 0.843. A score counts as evidence of a map when it clears this floor, and most directly when it
holds on the long-gap and wrap-only revisits that no local rule can reach.

*Table 1. Torus floors and accuracies at MapFormer's walk length and at ours (held-out map; floors (analysis) on the same
evaluation walks as the models; n = 8 per model).*

| T (moves) | wrap-only revisits | blank | n-gram | retrace | index RoPE, 1 layer | path, rank 2 | path, rank 4 |
|---|---|---|---|---|---|---|---|
| 128 | 0.000 | 0.507 | 0.598 | 0.934 | 0.805 | 0.971 | 1.000 |
| 1024 | 0.079 | 0.506 | 0.575 | 0.843 | 0.731 | 0.894 | 0.998 |

**One path layer solves the long-walk map; index attention does not.** On the torus at T = 1024, one path layer scores
0.998 +/- 0.004 (8/8 SOLVED) against 0.731 +/- 0.008 for index RoPE (0/8), a difference of +0.267 (permutation and Fisher
p 0.0002). Index RoPE sits below retrace on every run (0.72-0.74 against 0.843; analysis): it does not learn even the
map-free rule. The same contrast appears in two further registered batches: 1.000 against 0.722 (n = 8) and 1.000 against
0.719 (n = 4; drawn with its own floors in Fig. 1c).

**Where the difference lives.** By stratum (short gap / long gap / wrap-only), the path model scores 1.000 / 0.995 /
0.982 and one-layer index RoPE 0.784 / 0.462 / 0.471, against stratum floors of 0.506 / 0.500 / 0.507 and retrace 0.911
on short gaps (analysis; retrace equals the constant on the other two strata) (Fig. 2c). Index attention gains its margin
over the constant on short gaps, where retrace does better still; long-gap and wrap-only revisits require the map, and only
the path model has it.

## 5. An exchange rate at matched depth

Depth is the obvious way to give index attention more computation, so we ask how many index layers one path layer is
worth, with both trained and tested at the same length and nesting depth, on three tasks that need integration.

**Dyck-2: about three index layers.** A bracket string is a walk on a stack: an opening bracket steps forward, its
closing partner steps back, and MapFormer's models encode the two as opposite directions (their Fig. 3f). Trained and
tested at length 32 and nesting depth 12, one path layer scores 0.996
(1.000 at 2-4 layers), and index attention 0.648 / 0.889 / 0.964 / 0.982 at 1-4 layers, against a floor of 0.594
(Fig. 1a). The position main effect, averaged over two encodings (MapWM vs RoPE and MapPoPE vs PoPE; App. F), is +0.353 /
+0.130 / +0.045 / +0.024 at 1-4 layers, positive on 8/8 seeds at each depth (sign-flip p 0.0078). With three times the
budget, four index layers reach the ceiling (0.997-0.998), and training on a mixture of depths 4-12 keeps an effect of
+0.110 at depth 12 (8/8). The index arms are not held back by their frequency ladder (base 32 and base 10000 agree to
0.001 at four layers with three times the budget), so the rate measures what a layer of attention computes, not a mis-set encoding.

**A biased 1D walk: the same rate.** The ring lets us dial how often a step is undone: at p_plus 0.5 the walk is
balanced, at 0.9 one segment in ten runs backwards, and at 1.0 the walk never reverses and position is a count. On the
ring (fresh seeds, n = 6), one path layer scores 1.000 at every bias, and
index attention with 1 / 2 / 3 layers 0.710 / 0.949 / 0.998 at p_plus 0.5, 0.677 / 0.962 / 0.998 at 0.75, and 0.824 /
0.987 / 0.999 at 0.9 (Fig. 1b). Wherever a step can be undone, two index layers stay 0.013-0.051 short of the path layer
and three come within 0.002. At p_plus 1.0, where no step is undone, a single index layer reaches 1.000. One path layer
solves every run in every cell.

**The 2D torus: more than four.** At T = 1024 (n = 4) one path layer scores 1.000 (4/4 SOLVED) against index attention at
one layer 0.719, at four real layers 0.860 and as one block looped four times 0.864 (each 0/4, every run stalled;
complete separations, p 0.029) (Fig. 1c). Depth and the loop each help index attention (+0.140 and +0.145), and the loop
matches four real layers with a quarter of the parameters (203,925 against 798,741). The four-layer index model scores
0.897 on its own training map against 0.860 on the held-out map, so part of what depth adds is map-specific; the path
model scores 1.000 on both.

**What depth buys index attention.** Four layers or a loop of four lift short-gap revisits from 0.769 to 0.943 / 0.938,
just above retrace on this stream (0.912), and leave long-gap (0.458 / 0.507) and wrap-only (0.457 / 0.509) revisits at
or below the constant floor (0.515 / 0.517; analysis) (Fig. 2a; App. F). Depth buys index attention the local part of the
map; only the path code integrates over hundreds of moves.

Depth substitution is known qualitatively, from shortcuts to automata (Liu et al. 2023) and one-layer parity with
Selective RoPE. We give the rate at matched depth on three tasks and show where it breaks: on a 2D map with long gaps and
wrap-around, four index layers recover only the short-gap revisits.

![Figure 1](figs/fig_exchange_rate.png)

*Figure 1 (`fig_exchange_rate`). Accuracy of index-RoPE attention against its number of layers (orange; dots: runs, line:
means) beside one layer of path integration (blue line), each trained and tested at the same length and depth. (a) Dyck-2
at length 32 and depth 12 (Hewitt closing accuracy; n = 8): index attention needs about three layers to approach one path
layer (0.964 at 3, 0.982 at 4, against 0.996). (b) Ring of 32 cells at T = 128 for three move biases (n = 6 fresh seeds):
three index layers reach one path layer (0.998-0.999 against 1.000). (c) Torus at T = 1024 (n = 4): four index layers
(0.860) or one layer looped four times (0.864) stay far below one path layer (1.000), just above the retrace floor
(0.844). Dashed lines: map-free floors (analysis; Dyck: the best stack-free floor; ring: retrace per bias, on another
stream of the same task; torus: always-blank and retrace on this batch's own walks).*

![Figure 2](figs/fig_strata_modes.png)

*Figure 2 (`fig_strata_modes`). Revisit accuracy by stratum on the torus at T = 1024 for the three ways a model fails to
learn the map; lines: arm means, dots: runs. (a) Index attention (n = 4): depth and a loop help short-gap revisits only.
(b) Per-head rank of the content-to-angle map (n = 8, same seeds and initial weights): rank 4 is near 1.000 in every
stratum; rank 2 loses accuracy with the gap and most at wrap-only revisits. (c) Sign of the step (n = 8): signed near
1.000 everywhere; |.| and softplus fall from short gaps to wrap-only revisits; one-layer index RoPE for reference. Dashed:
always-blank within each stratum; dotted: retrace within the short-gap stratum (0.91); floors (analysis) on each batch's
own walks.*

## 6. Failure by search, not capacity

Two design choices of the content-to-angle map decide whether the torus map is learned: its per-head rank and the sign
parameterisation of its step. For each we show the same three facts. The constrained class fails from scratch; a
solution in the class exists, constructed explicitly; and that solution, installed and trained, is kept or re-found.

### 6.1 Per-head rank of the content-to-angle map

**Per-head rank decides it.** On the torus at T = 1024 we compare six bottlenecks, all built from the same initial
weights at each seed (Table 2, Fig. 3a). Every arm whose heads each read a rank-2 map solves 0-2 of 8; rank 3 solves 6/8
and rank 4 8/8, whether the latent is shared or per head. At matched sharing (per-head rank 4 against the block-diagonal
rank-4 latent, in which each head reads rank 2) the split is 8/8 against 2/8 (Fisher and permutation p 0.0070). Per-head
rank 3 improves on per-head rank 2 by +0.102 in accuracy (permutation p 0.027). Total latent size does not track success: the per-head
rank-2 arm and the shared rank-4 arm both have four latent dimensions and land on opposite sides. Nor is it the initial
angle scale: the block-diagonal arm starts with smaller angle increments, but the two other rank-2 arms start at the
same scale as rank 4 and fail the same way. The training curves show the split early: most rank-2 runs flatten at losses
of 0.2-0.5, while the rank-3 and rank-4 medians cross the SOLVED threshold at epochs 478-532 (App. G). Read with MapFormer's
Table 2: rank 1 cannot represent a 2D map; rank 2 can, and learns 128-step walks, but at 1024 moves reliable learning
needs rank above the map's dimension.

*Table 2. SOLVED (of 8) and held-out accuracy at T = 1024 (mean over 8 seeds) by per-head rank of the content-to-angle
map; torus, 900 epochs, every arm built from the same initial weights at each seed.*

| per-head rank | arm | SOLVED | accuracy |
|---|---|---|---|
| 2 | one rank-2 latent shared by the heads | 0/8 | 0.894 |
| 2 | per-head rank-2 latents | 1/6 fresh seeds (2/8 all seeds) | 0.885 |
| 2 | block-diagonal rank 4 (each head reads rank 2) | 2/8 | 0.948 |
| 3 | per-head rank 3 | 6/8 | 0.987 |
| 4 | shared rank 4 | 8/8 | 0.998 |
| 4 | per-head rank 4 | 8/8 | 0.999 |

**Twice the budget does not rescue rank 2.** Trained for 1800 epochs from scratch, rank 2 solves 0/8 (0.908) against
rank 4's 7/8 (Fisher p 0.0014; accuracy +0.086, permutation p 0.0003).

**The rank-2 solution exists and is held.** The rank-2 projection of each solved rank-4 model scores 0.9955 frozen (at
least 0.981 on every seed). Trained from there it solves 7/8, as many as the rank-4 control trained from its own solution
(7/8), against 1/8 for rank 2 trained further from its own stalled weights (accuracy p 0.0009) (Fig. 4a). The restart at
learning rate 1e-3 lifts every projected run's training loss to 0.37-0.46 (control 0.26-0.30), and seven return below
0.05 at epochs 553-654, inside the control's window (497-669). A rank-2 map that solves the torus exists, is kept by
training, and is not reached from scratch: the failure is search.

**Rank against the map's dimension.** Rank equal to the map's dimension D is hard in 2D and 3D and not in 1D (Fig. 3b).
On a 2D torus of 32 cells per side rank 2 solves 1/8 against rank 3's 8/8 (Fisher p 0.0014). On a 3D torus of 10 cells
per side rank 3 solves 1/8 (rank 4, 4/8), and rank 4 solves an 18-per-side 3D torus 8/8 (1.000). On the 1D ring rank 1
solves 7/8 against rank 2's 8/8. Per-head rank D + 1 learns reliably in 2D and on the larger 3D torus.

![Figure 3](figs/fig_rank_solved.png)

*Figure 3 (`fig_rank_solved`). Runs (of 8) that solve the map, by the per-head rank of the content-to-angle map. (a)
Torus, T = 1024, six bottlenecks built from the same initial weights: every arm whose heads each read a rank-2 map solves
0-2/8, rank 3 solves 6/8 and rank 4 8/8, whether the latent is shared or per head. (b) Rank equal to the task's dimension
D against D + 1 (separate batches and recipes): on the 1D ring rank 1 solves 7/8 (rank 2, 8/8); on a 2D torus rank 2
solves 1/8 (rank 3, 8/8); on a small 3D torus (10 per side) rank 3 solves 1/8 and rank 4 4/8, and rank 4 solves 8/8 on an
18-per-side torus with few wrap-only revisits (hatched). Torus budget 900 epochs; at 1800 epochs rank 2 still solves 0/8.*

### 6.2 The sign of the phase step

**Non-negative parameterisations rarely learn the map from scratch.** On the torus at T = 1024 the |.| step scores 0.821
against 0.998 for the signed step, -0.177 (permutation p 0.0002), and solves 0/8 against 8/8; softplus scores 0.781 and
solves 1/8 (-0.218, p 0.0003). On the ring |.| solves 2/8 and 1/8 against 8/8 at p_plus 0.5 and 0.75 (Fisher p 0.007 and
0.0014). Observation-step-free models show the same: on the ring 0/8 against 8/8, on the torus 0/8 against 8/8 (0.880
against 1.000). The training curves separate (Fig. 5): the signed median crosses the SOLVED threshold at epoch 480
/ 57 / 309 / 183 in four batches and keeps descending, while most |.| runs flatten at a loss of 0.3-0.5.

**The cost holds at each step's best learning rate, without weight decay and at 2 layers d 256** (Table 3). With each step at its own best of four learning rates (ring, fresh
seeds; the |.| optimum, 3e-3, is bracketed by the grid), |.| solves 2/8 against signed 8/8 (selection-aware p 0.0070 on
SOLVED and 0.0057 on accuracy). Without weight decay the ring gives 0/8 against 8/8 (-0.127); at a third of the learning
rate, 0/8 against 6/8 (Fisher p 0.007). A 2-layer, d_model 256 torus model with eight times the parameters gives 2/8
against 8/8 (Fisher p 0.007; 0.967 against 0.999, p 0.0005). Across the learning-rate grid, higher rates hurt the signed
step first (8, 8, 4 and 1 of 8 solved at 1e-3, 3e-3, 1e-2 and 3e-2) and do not help |.| past 3e-3 (1, 2, 1 and 0 of 8).
In every recipe the |.| models learn little cancelling code: opposite moves cancel mod 2 pi in 0.04-0.13 of their
channels (arm means), against 0.43-1.00 for signed steps, including the recipes in which some |.| runs solve.

*Table 3. The sign cost by task, model and recipe (observation-step-free models unless marked "full"; n = 8 per arm).*

| task and model | recipe | \|.\| accuracy, SOLVED | signed accuracy, SOLVED | test |
|---|---|---|---|---|
| torus, 1 layer, full | default (lr 1e-3, weight decay 0.05) | 0.821, 0/8 | 0.998, 8/8 | perm p 0.0002; Fisher 0.0002 |
| torus, 1 layer | default | 0.880, 0/8 | 1.000, 8/8 | perm p 0.0002; Fisher 0.0002 |
| torus, 2 layers, d_model 256 | default | 0.967, 2/8 | 0.999, 8/8 | perm p 0.0005; Fisher 0.0070 |
| ring, full, p_plus 0.5 | default | 0.973, 2/8 | 1.000, 8/8 | Fisher 0.0070 |
| ring | default | 0.911, 1/8 | 1.000, 8/8 | perm p 0.0002; Fisher 0.0014 |
| ring | weight decay 0 | 0.873, 0/8 | 1.000, 8/8 | perm p 0.0002; Fisher 0.0002 |
| ring | lr / 3 (3.3e-4) | 0.841, 0/8 | 0.950, 6/8 | Fisher 0.0070 |
| ring | each step at its best of four rates (\|.\| 3e-3, signed 1e-3) | 0.963, 2/8 | 1.000, 8/8 | selection-aware p 0.0070 (SOLVED), 0.0057 (accuracy) |

*Per-rate results, the threefold learning rate and the SOLVED-threshold sensitivity are in App. H.*

**The non-negative solution exists, and training keeps or re-finds it.** Proposition 1 constructs it: each solved signed
observation-step-free model is converted exactly into |.| form: the move latents are re-solved to orthogonal directions,
each channel's frequency is set to max(|omega|, 1), and each step to the signed angle reduced mod 2 pi. Angles are
unchanged mod 2 pi (error at most 1.4e-6 rad), frozen accuracy equals the source's on 8/8 seeds, and the weights stay
within about 5x of the signed model's, so the test is posed at the scale training uses. Softplus is
constructed the same way, with logits equal to the source's to 2.8e-11 (App. D). On the ring the constructed |.| model,
trained for 300 more epochs, stays solved on 8/8 seeds and never leaves the solved region, keeping its cancelling code in
the channels attention uses (98%, usage-weighted). On the torus, trained at learning rate 1e-4, it is perturbed early
(peak loss 0.11-0.21), re-enters the solved region for good between epochs 119 and 204, and ends solved on 8/8 seeds at
accuracy 1.000, equal to its signed control. From scratch the same class solves 0/8 on both tasks (Fig. 4b-c).

**How signed steps reach the map (ring).** Started from the |.| start, with both moves rotating the same way, signed
steps still solve 7/8 (|.| from the same start: 0/8; Fisher p 0.0014). They end with 98-99% of channels opposite-signed
(analysis): they cancel opposite moves through zero, which Lemma 1 shows the non-negative step excludes for an inverse
pair (Fig. 6). The escape is collective. Freezing the two move latents, with W_out and omega still free, blocks it: 1/8
against 7/8 (Fisher p 0.010): with the latents free the model moves one move's latent toward the negative of the other's
and cancels every channel together (82% of channels end opposite-signed, against 15% with the latents frozen). MapFormer
described the antipodal end state (their Figs. 3d and 9b); this is the causal test of how it is reached. A |.| step has
only the wrapped branch, and on the ring it does learn the map when its opposite-move angles start with sums past pi (8/8,
cancelling by wrapping; Fig. 6d); on the torus that start does not help within the budget (App. H).

**Published monotone content phases also fail these maps.** Faithful CARoPE cannot represent cancellation (Proposition 2,
App. E) and scores 0.667 on the torus (0/8), below index RoPE's 0.722 (p 0.040). Faithful CoPE can represent ring
cancellation (we construct the solution; App. I) yet solves 0/8 on the ring (0.887 against signed 1.000, p 0.0002), while
beating index RoPE by +0.157. It also fails a 256-cell line that the walk never wraps (1/16 solved against signed 8/8),
so wrap-around is not its obstacle either. MapFormer's Table 2 reported CoPE failing 1D and 2D navigation and attributed
it to the gate's sign; our construction shows that the obstacle is not what CoPE can represent.

![Figure 4](figs/fig_existence.png)

*Figure 4 (`fig_existence`). A solution exists in each constrained class and training keeps or re-finds it, but training
from scratch does not find it. Bars: runs SOLVED of 8 (hatched: the solution was installed and then trained). (a) Rank 2
on the torus: a rank-2 projection of a solved rank-4 model scores 0.9955 frozen and is held when trained (7/8), while rank
2 from scratch solves 0/8, and 1/8 when the stalled runs are trained further. (b) The |.| step on the ring: an installed
exact non-negative solution is held under training (8/8 SOLVED, lost on 0/8), while |.| from scratch solves 0/8 against
signed 8/8. (c) The |.| step on the torus: the installed solution is perturbed (loss above 0.05 on 6/8) and re-found (8/8
SOLVED); from scratch 0/8 against 8/8. "Lost": the maximum per-epoch training loss after the start exceeds 0.3, or exceeds
10x the signed control's maximum at that seed and 0.05.*

![Figure 5](figs/fig_loss_curves.png)

*Figure 5 (`fig_loss_curves`). Training loss per epoch for the signed step (blue) and the non-negative steps |.| (orange)
and softplus (green), every run of four batches (thin lines; thick: median of 8 runs; log scale; training loss with
dropout on). Dashed line: the SOLVED threshold (final-5% mean training loss below 0.05); n/8 = runs SOLVED. (a) Torus,
T = 1024, one layer: signed 8/8, |.| 0/8, softplus 1/8. (b) Ring of 32 cells, observation-step-free: 8/8 against 0/8.
(c) Torus, observation-step-free: 8/8 against 0/8. (d) As (c) at two layers and d_model 256: 8/8 against 2/8. The signed
median crosses the threshold at epoch 480 / 57 / 309 / 183 of 900 / 300 / 900 / 900 and keeps descending; the |.|
median never does.*

![Figure 6](figs/fig_mechanism.png)

*Figure 6 (`fig_mechanism`; ring, 8 runs per panel, analysis of a registered batch). Move angles (a(+1), a(-1)) of every
rotary channel at the start (grey) and after training (dots). A channel cancels opposite moves when a(+1) + a(-1) = 0 mod
2 pi: on the blue line (through zero, reached only with opposite signs) or the red line (by wrapping). Grey shading marks
the two mixed-sign quadrants, the only region the |.| step cannot reach (omega's sign is free, so both angles may be
negative); dotted: a(+1) + a(-1) = pi; dashed boxes: start ranges. (a) Signed step from the same-sign start: 7/8 runs
solve; one move's angle crosses zero, and 89% of channels cancel at the end (6% at the start). (b) The |.| step from the
same start: 0/8 solve, 4% cancelling. (c) The |.| step from a large start whose sums stay below pi: 2/8 solve, 5% cancelling,
all by wrapping. (d) The |.| step from a large start whose sums exceed pi: 8/8 solve; cancelling rises from 7% to 23% of
channels, all by wrapping (App. H).*

### 6.3 One failure signature

Every failing model keeps short-gap revisits and loses long-gap and wrap-only ones (short / long / wrap-only): one-layer
index RoPE 0.784 / 0.462 / 0.471; shared rank 2 0.950 / 0.671 / 0.557; |.| 0.870 / 0.630 / 0.515. The solving models are
at 0.977-1.000 in every stratum: signed 1.000 / 0.995 / 0.982 and rank 4 1.000 / 0.995 / 0.977 (Fig. 2b-c). The
non-negative codes also do not cancel opposite moves: weighted by how much attention uses each channel, signed steps cancel
in 70-93% of channels and |.| steps in 2-10% (analysis; App. H). Index attention, a rank-2 map and a non-negative step
all end in the same place: a code that serves local revisits and does not carry position over long gaps or around the
torus.

## 7. Maps from words

The text world asks whether the path model can find its actions when nobody marks them. Each step of the torus walk is
written as a sentence: a direction word chosen from three synonyms, verbs that do not move the walker, fillers, and asides
that name objects without visiting them. Nothing in the input says which words move the walker; the model must learn from
the prediction task which words should rotate the phase and by how much (vocabulary and per-seed step tables: App. J).

**One path layer learns navigation told in English.** With the torus walk rendered as 1024 words (fresh seeds, n = 6),
one path layer scores 0.979 +/- 0.048 (6/6 SOLVED) against 0.505 for one-layer index RoPE (0/6), +0.474 (permutation and
Fisher p 0.0022); two index layers reach 0.793 (0/6). The floors are 0.505 for the best constant and 0.597 for a
reversal-copy rule. Over all eight seeds one-layer index RoPE sits exactly at the constant floor and below the
reversal-copy rule on every seed, and two index layers clear both floors but stall on every seed.

**It learns which words move it, and opposite directions cancel.** On every seed (declared secondaries, n = 8), once the
component common to the four direction steps is removed, north + south and west + east cancel to 0.002-0.012 (Fig. 7a).
Zeroing the direction words' steps drops every seed to or below the constant floor (0.39-0.51), and synonyms collapse to
one step (cosine 1.000) (Fig. 7c). Half the seeds add a per-step clock alongside the map, in 31-38 of 64 channels, and
solve as well (Fig. 7b). On those seeds the common component points along the verbs' step (cosine +0.999 to +1.000): the
model counts steps in some channels and integrates directions in the others, and zeroing the non-direction words'
steps costs it more (0.33-0.69 against 0.05-0.30). The cancellation that signed steps build on the ring (Sec. 6.2)
appears here over a vocabulary, learned from the prediction task alone.

![Figure 7](figs/fig_textworld.png)

*Figure 7 (`fig_textworld`). What a one-layer path model learns from a torus walk told in English (n = 8 seeds). (a)
Opposition of the learned direction-word steps, |A + B| / |A| averaged over north/south and west/east (0 = opposites
cancel): after removing the component common to all four directions, opposites cancel to 0.002-0.012 on every seed. (b)
That common component is small on four seeds (blue, 4-6% of a direction step) and large on the other four (orange,
0.48-1.14), where it points along the verbs' step: a per-step clock layered on the map (31-38 of 64 phase channels drift
between visits to the same cell). (c) Held-out accuracy with every step, with the steps of all non-direction words zeroed,
and with the direction words' steps zeroed: removing the direction steps drops every seed to the constant floor (0.505,
dashed) or below; the clock seeds also depend on the other words. Eval mode without the dropout re-scale (App. C).*

## 8. Discussion

**What the two failure axes share.** For per-head rank 2 and for non-negative steps, a solution in the class exists (the
rank-2 projection scores 0.9955; the |.| construction is exact), training keeps it (7/8 held at rank 2; held 8/8 on the
ring and re-found 8/8 on the torus for |.|), and training from scratch does not reach it (0-2/8; 0/8). The obstacle is
optimisation, not expressivity. Shakerinava et al. report the same pattern for non-Abelian rotations; we find it in
single-layer Abelian tasks, produced by one rank or parameterisation choice. Both failures leave the same trace in the
strata: a code that serves short-gap revisits while long-gap and wrap-only revisits fall toward the floor; for the
non-negative step we also see directly that opposite moves do not cancel.
On the ring we also see the route that the successful parameterisation takes: signed steps move one move's latent toward
the negative of the other, cancelling every channel at once, a route Lemma 1 shows is closed to the non-negative step.
In the rotary slot, then, the interesting design variables are not those that change what a model can represent, but
those that change which representations gradient descent reaches from a small random start.

**Design guidance.** Keep the phase step signed. Give each head rank above the map's dimension (D + 1). Measure map use at
walk lengths where retracing no longer suffices, against measured map-free floors and, where possible, by stratum: the
long-gap and wrap-only revisits separate a map from a local rule, and an aggregate accuracy near 0.9 can be either.

**What and where.** The maps these models learn keep position apart from content in the sense TEM proposed. In converged
path models on the short-walk torus, removing object identity from the score leaves accuracy at 0.974-0.989, and
replacing the score by one shared position kernel times a content gain gives 0.988-1.000, while index models fall below
the floor (analysis; App. K). MapFormer showed the separation in the learned embeddings (their Fig. 9); these checks show
that the trained models rely on it.

**The integrator view.** In Howard et al.'s terms signed steps make a map and non-negative steps a path-length clock. In
the rotary slot these coincide in what they can represent (Proposition 1), so the difference we measure lies in what
training finds. The text world shows the other side of the same coin: given a vocabulary in which only some words move,
the signed step learns which ones, and that their opposites cancel.

**Text.** On ordinary text, at small scale (analysis, n = 3), the learned phase behaves as a token clock: it grows
linearly with token index (R^2 0.994-0.999 on TinyStories, Eldan and Li 2023, and on code), and MapWM matches index RoPE
(1.5053 against 1.5050 nats per token on TinyStories), consistent with there being no map to build.

## 9. Limitations

- Small models (1-4 layers, 2 heads, d_model 128-256) on synthetic map tasks; natural text is touched only at small scale.
- "Not found" is relative to the stated budgets (300-1800 epochs). Learning rates were swept on the ring, while the torus
  was run at one rate and two model sizes; in the larger torus model the accuracy gap is a quarter of the one-layer gap
  and most |.| runs were still descending, so at scale part of the sign cost may be speed.
- Rank results use two heads, and rank equal to the map's dimension is hard in 2D and 3D but not in 1D. Our latent is
  shared across heads where MapFormer's is per head; per-head arms are reported alongside.
- We show how signed steps reach the map; why the non-negative parameterisations stall from small starts is not
  identified.
- No recurrent or SSM baselines. The torus exchange rate is a lower bound (more than four layers), measured at n = 4.

## References

Titles and author lists checked against arXiv, OpenReview, Crossref, ACL Anthology or PMLR metadata on 2026-10-15/16;
entries marked * were not re-checked (venue, or the whole entry where stated in the notes file).

- Amid, E. and Warmuth, M. K. Reparameterizing mirror descent as gradient descent. NeurIPS 2020*; arXiv:2002.10487.
- Eldan, R. and Li, Y. TinyStories: how small can language models be and still speak coherent English? arXiv:2305.07759,
  2023*.
- Golovneva, O., Wang, T., Weston, J. and Sukhbaatar, S. Contextual position encoding: learning to count what's important
  (CoPE). arXiv:2405.18719, 2024.
- Grazzi, R., Siems, J., Zela, A., Franke, J. K. H., Hutter, F. and Pontil, M. Unlocking state-tracking in linear RNNs
  through negative eigenvalues. ICLR 2025*; arXiv:2411.12537.
- Gopalakrishnan, A., Csordas, R., Schmidhuber, J. and Mozer, M. C. Decoupling the "what" and "where" with polar
  coordinate positional embedding (PoPE). arXiv:2509.10534, 2025*.
- Hahn, M. and Rofin, M. Why are sensitive functions hard for transformers? ACL 2024 (long papers); arXiv:2402.09963.
- Helfrich, K., Willmott, D. and Ye, Q. Orthogonal recurrent neural networks with scaled Cayley transform. ICML 2018*;
  arXiv:1707.09520.
- Hewitt, J., Hahn, M., Ganguli, S., Liang, P. and Manning, C. D. RNNs can generate bounded hierarchical languages with
  optimal memory. EMNLP 2020*; arXiv:2010.07515.
- Howard, M. W., MacDonald, C. J., Tiganj, Z., Shankar, K. H., Du, Q., Hasselmo, M. E. and Eichenbaum, H. A unified
  mathematical framework for coding time, space, and sequences in the hippocampal region. J. Neurosci. 34(13):4692-4707,
  2014.
- Lahoti, A., Li, K. Y., Chen, B., Wang, C., Bick, A., Kolter, J. Z., Dao, T. and Gu, A. Mamba-3: improved sequence modeling
  using state space principles. ICLR 2026; arXiv:2603.15569.
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
- Zhang, Y., Chen, Z., Liu, Y., Qin, Z., Yuan, H., Xu, K., Yuan, Y., Gu, Q. and Yao, A. C.-C. Group representational
  position encoding (GRAPE). ICLR 2026; arXiv:2512.07805.
