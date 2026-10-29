# Main paper outline, v2 (2026-10-17)

Target: an 8-9 page track (CCN 2027 proceedings, 8 pages; or TMLR). Page budget per section below; figures by file name
in `docs/paper/figs/`. Every number is taken from the file named in brackets (a CORRECTED / AUDIT block at the top of a
results file supersedes its body). Labels: "(analysis)" = computed after the batch on stored checkpoints or streams;
everything unlabelled is a pre-registered verdict or a declared secondary of a pre-registered batch. Where a batch
contained pilot seeds the fresh-seed value is the one quoted. The bracketed sources are for drafting only.

---

## Title

**Representable but not found: what path-integrated rotary attention learns, and how it fails**

Alternative: *Maps that depth cannot buy and training cannot find: a study of path-integrated rotary attention*

## Abstract (~200 words)

Path-integrated rotary attention (MapFormer) replaces RoPE's token index with a learned phase that sums a
content-dependent step per token, and helps transformers navigate. We ask what such models learn and how they fail. On
the standard short-walk benchmark a map-free rule that retraces the last run of moves already scores 0.934, so we train
and test on 1024-move torus walks, where that rule scores 0.843: one path-integrated layer reaches 0.998 (8 of 8 seeds
solved) against 0.731 for index RoPE (0 of 8). At matched depth one path layer is worth about three index layers on
Dyck-2 and a biased 1D walk, and more than four on the torus, where index depth helps only short-gap revisits. Told in
English, the walk is learned from words (0.979 vs 0.505), opposite direction words cancelling. Where these models fail,
they fail by search, not capacity, along two axes. With rank 2 per head the content-to-angle map solves the torus on at
most 2 of 8 seeds, against 8 of 8 at rank 4, although a rank-2 solution exists and training keeps it. Non-negative step
parameterisations represent the map exactly and keep or re-find a constructed solution, yet from scratch solve 0 of 8
against 8 of 8 for signed steps, a gap that persists at each step's best learning rate. Both failures are optimisation,
not expressivity.

---

## 1. Introduction (1.0 page)

**P1, the question (4-5 sentences).** Rotary position encodings are integrators: index RoPE adds a constant angle per
token, and content-dependent variants let each token set its own step (CARoPE, CoPE, Selective RoPE, Mamba-3, MapFormer).
When the step is driven by actions, the summed phase is a position on a map, and attention compares net displacements
instead of token distances. MapFormer (Rambaud et al., v4) built this into a transformer and showed it helps navigation.
We ask what such a model actually learns, how much it buys over index attention, and why training sometimes misses the
map it can represent.

**P2, what MapFormer establishes and what it leaves open (4-5 sentences).** MapFormer shows that path-integrated
attention beats index, TAPE, CoPE and PathAtt baselines (up to 4 layers) on 1D and 2D grid navigation at 128-step walks,
generalises out of distribution to sparser and denser walks, extrapolates to deeper Dyck-2 strings with one layer, and
learns opposite latents for opposite actions (their Table 2, Figs. 3, 9). Three questions remain: how much of the
short-walk benchmark needs a map at all; what one path layer is worth in index layers when both are trained and tested at
the same length and depth; and why the model sometimes fails on its home task, for which it can represent the solution.

**P3, our answer (4-5 sentences).** We measure against map-free floors and find that short walks are mostly solvable by
retracing; at 1024 moves the floor drops and a single path layer still solves the map. Trained and tested at matched
depth, one path layer is worth about three index layers on Dyck-2 and a biased 1D walk and more than four on the torus.
The same mechanism learns a map from navigation told in words. Its failures share one signature: a solution in the
constrained class exists and training keeps it, but training from scratch does not find it.

**P4, contributions (4 numbered sentences).**
1. *A sharper measurement of the advantage.* A map-free retrace rule scores 0.934 on MapFormer's short walks (analysis),
   so we measure on 1024-move torus walks against measured floors (blank 0.506, n-gram 0.575, retrace 0.843), where one
   path layer scores 0.998 against index RoPE's 0.731 (8/8 vs 0/8 seeds solved, p 0.0002), and index depth or a looped
   block lifts only short-gap revisits (to ~0.94) while long-gap and wrap-only revisits stay at the floor (0.46-0.51).
2. *An exchange rate at matched depth.* One path layer is worth about three index layers on Dyck-2 trained and tested at
   nesting depth 12 (+0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers) and on a biased 1D walk (three index layers reach
   0.998-0.999 against path 1.000, fresh seeds), and more than four on the 2D torus (four index layers 0.860 vs 1.000).
3. *Failure by search, not capacity, along two axes.* With per-head rank 2 the content-to-angle map solves the torus on
   0-2 of 8 seeds (rank 3: 6/8; rank 4: 8/8), twice the budget leaves it at 0/8 (vs 7/8), and a rank-2 solution exists
   (0.9955) and is held under training; non-negative step parameterisations represent the map exactly and keep or re-find
   a constructed solution, yet from scratch solve 0/8 against 8/8 for signed steps on the torus (-0.177), a gap that holds
   without weight decay, in a 2-layer d 256 model and at each step's best of four learning rates (2/8 vs 8/8, p 0.007).
4. *Maps from words.* Navigation told in English is learned by one path layer (0.979 vs index 0.505, 6/6 vs 0/6 seeds,
   p 0.0022), and every model encodes opposite direction words as cancelling steps.

---

## 2. Background (0.8 page)

**2.1 Rotary position as an integrator (0.2).** Per rotary channel c, the score depends on position only through
exp(-i omega_c (Phi_{c,t} - Phi_{c,t'})), with Phi the channel's summed step (Mamba-3, Prop. 3). Howard et al.'s (2014)
integrator families give the vocabulary: a constant rate codes elapsed time (index RoPE), a non-negative rate codes path
length (CARoPE, CoPE, GRAPE's additive phase), a signed rate codes position on a map (MapFormer, Selective RoPE). TEM and
TEM-t (Whittington et al. 2020, 2022) frame the map as a "where" learned apart from the "what".

**2.2 MapFormer (0.2).** MapWM rotates content queries and keys by theta_t = omega * cumsum(W_out W_in x_t), a low-rank
content-to-angle map; observations learn near-zero steps and opposite actions opposite steps (their Fig. 3d). Their
Table 2: one-layer MapWM-r2 solves 2D navigation at 128 steps, rank 1 does not, and RoPE, TAPE, CoPE and PathAtt up to
4 layers fall short; they attribute CoPE's failure to its gate's sign (it cannot represent an action and its inverse).
We build on exactly this model; our implementation reproduces their Table 2 2D accuracy (App. A).

**2.3 Representability results (0.2).** Real non-negative eigenvalues cannot track parity (Sarrof et al. 2024, Thm. 2;
Grazzi et al. 2025, Thm. 1); rotations escape this, and rotations by non-negative angles generate cyclic groups (Grazzi
et al., Thm. 3; Mamba-3, Sec. 3.2). So in the rotary slot sign does not limit what can be represented; any difference
between signed and non-negative steps is a difference in what training finds.

**2.4 Optimisation, not expressivity (0.2).** Shakerinava et al. (2026) find that input-dependent rotations in
unit-modulus SSMs can represent non-Abelian state tracking but do not learn it from scratch, while near-solution starts
recover it. Orientation-fixing parameterisations have a history: scoRNN fixes the +/-1 diagonal of its Cayley transform
during training (Helfrich et al. 2018); a reparameterised weight w = |u| cannot change sign under gradient flow (Amid and
Warmuth 2020; Woodworth et al. 2020); LRU initialises its phases near zero (Orvieto et al. 2023). Depth can substitute
for recurrence through shortcuts (Liu et al. 2023). We bring these lines to path-integrated attention.

**2.5 Two results we use (in the Results text; proofs App. E).** *Proposition 1 (representability by wrapping).* If at
most r tokens step, a |.| rank-r model reproduces any signed angle table exactly mod 2 pi; with observation embeddings
annihilated only the moves step, so every |.| arm can represent the map; softplus can too, by making its step at zero
(ln 2) a full turn. *Lemma 1 (relative-sign gauge).* Negating a channel's steps together with the second coordinate of
its query and key leaves every score unchanged, so a non-negative step fixes only relative signs within a channel; for an
inverse pair it removes cancellation through zero, while the wrapped code (a, 2 pi - a) remains.

---

## 3. Setup (0.7 page)

**Models.** MapWM: 1 layer (up to 4 where stated), 2 heads, d 128, step Delta = W_out W_in x through a rank-r latent
(rank 4 shared across heads unless stated; MapFormer's latent is per head, and per-head variants are named where rank is
compared). Step parameterisations: signed (Delta = s), |.| (Delta = |s|), softplus (Delta = softplus(s); every step
starts at ln 2). Observation-step-free variants give steps to move tokens only. Index RoPE baseline: canonical
base^(-2c/d), 1-4 layers or one block looped x4. Faithful CARoPE and CoPE for the published monotone comparison.

**Tasks.** (i) 2D torus 64 x 64, moves interleaved with observations from 16 objects, map redrawn at evaluation so every
score is transfer; T = 1024 moves (and MapFormer's T = 128 for the floor analysis), 900 epochs. (ii) 1D ring of 32 cells,
T = 128, forward-step probability p_plus (0.5 balanced; 1.0 never reverses), 300 epochs. (iii) Dyck-2 at length 32 and
nesting depth 12, Hewitt closing accuracy (chance 0.5, best map-free floor 0.594). (iv) Text world: the torus walk
rendered as English (58 words: three synonyms per direction, movement-free verbs, fillers, asides), T = 1024 words.
(v) 2D and 3D tori of other sizes for the rank-versus-dimension test.

**Floors.** Per task, the best map-free predictor: always-blank, n-gram over orders 0-5, and *retrace* (copy the
observation seen 2j steps back while the current run of moves reverses the previous run).

**Protocol (two sentences).** Every verdict was pre-registered with its decision rule against the measured noise floor,
n = 8 seeds per arm unless stated, every model trained and tested at the same length (and nesting depth), with SOLVED =
final-5% training loss below 0.05 and permutation / Fisher tests. Batches were checked before launch by an independent,
results-blind code review (most batches) and their analyses re-run from stored checkpoints by an independent audit; App. A
records both per batch.

---

## 4. A sharper measurement of the advantage (1.0 page)

**Claim 4.1 (analysis): short walks mostly do not need a map.** On MapFormer's T = 128 torus walks, retrace scores
0.934 (blank 0.507, n-gram 0.598), and no revisit is reachable only by wrapping around the torus (wrap-only share 0.000).
At T = 1024 retrace falls to 0.843 (blank 0.506, n-gram 0.575) and 7.9% of revisits are wrap-only.
[`docs/audits/2026-10-16/paper2x2_T128_floors_out.txt`, `sign_floors_out.txt`]. Present as a benchmark analysis: the
long-walk regime is where map use is measurable. At T = 128 the registered position effect is +0.243 (path r2 0.971,
path r4 1.000, index 0.805; n = 8) [`PAPER2X2_RESULTS.md`]; one sentence, table in App. B.

**Claim 4.2: one path layer solves the long-walk map; index attention does not.** Torus T = 1024: path 0.998 +/- 0.004
(8/8 SOLVED) vs index RoPE 0.731 +/- 0.008 (0/8), +0.267, permutation and Fisher p 0.0002 [`SIGN_MATCHED_RESULTS.md`].
Index RoPE sits below retrace on every run (0.72-0.74 vs 0.843; analysis): it does not learn even the map-free rule.
Same contrast in two further registered batches: 1.000 vs 0.722 (`GENERALITY_RESULTS.md`), 1.000 vs 0.719 (n = 4,
`FACTORIAL_RESULTS.md`).

**Claim 4.3: where the difference lives.** Revisit strata (short gap < 128 moves / long gap / wrap-only): path 1.000 /
0.995 / 0.982 (n = 8); index 1 layer 0.784 / 0.462 / 0.471 against stratum floors 0.506 / 0.500 / 0.507 and retrace 0.911
on short gaps (retrace equals the constant on the other two strata). The map is what long-gap and wrap-only revisits
require. [`fig_strata_modes_out.txt`]

Figure: **Fig. 1 = `fig_exchange_rate`** (panel c carries this section's torus numbers with floors; panels a-b used in
Sec. 5). **Fig. 2 = `fig_strata_modes`** (panel c here; panel a in Sec. 5, panels b-c in Sec. 6.3).
Table 1: floors and accuracies at T = 128 and T = 1024 (blank, n-gram, retrace, index, path).

---

## 5. An exchange rate at matched depth (0.9 page)

**Claim 5.1: Dyck-2, about three index layers.** Trained and tested at length 32, depth 12 (Hewitt closing accuracy;
floor 0.594): path 1 layer 0.996 (1.000 at 2-4 layers); index 1-4 layers 0.648 / 0.889 / 0.964 / 0.982; position effect
+0.353 / +0.130 / +0.045 / +0.024 (8/8 seeds positive at each depth, sign-flip p 0.0078). With three times the budget,
four index layers reach ceiling (0.997-0.998). Mixture training over depths 4-12 keeps +0.110 at depth 12 (8/8).
[`DYCK_MDEPTH_RESULTS.md`; `fig_exchange_rate_out.txt`]

**Claim 5.2: a biased 1D walk, the same rate (fresh seeds, n = 6).** Ring of 32 cells, T = 128: path 1 layer 1.000 at
every bias; index 1 / 2 / 3 layers 0.71 / 0.949 / 0.998 (p_plus 0.5), 0.68 / 0.962 / 0.998 (0.75), 0.82 / 0.987 / 0.999
(0.9). Three index layers are needed wherever a step can be undone; at p_plus 1.0, where no step is undone, one index
layer reaches 1.000. Path solves every cell with one layer (32/32 SOLVED). [`CANCEL_RESULTS.md` CORRECTED block,
`docs/audits/2026-09-27/fresh_seeds.txt`, `fig_exchange_rate_out.txt`]

**Claim 5.3: the 2D torus, more than four.** T = 1024 (n = 4): path 1 layer 1.000 (4/4) vs index 1 layer 0.719, 4 real
layers 0.860, one block looped x4 0.864 (each 0/4, every run STALLED; complete separations, p 0.029). Depth and the loop
each help index (+0.140, +0.145) [`FACTORIAL_RESULTS.md`].

**Claim 5.4: what depth buys index attention.** Four layers or a loop of four lift short-gap revisits from 0.769 to
0.943 / 0.938, just above retrace (0.912 on this stream), and leave long-gap (0.458 / 0.507) and wrap-only (0.457 /
0.509) revisits at or below the constant floor (0.515 / 0.517) [`fig_factorial_strata_out.txt`]. Depth buys index attention the
local part of the map; only the path code integrates over hundreds of moves.

**Positioning (2 sentences).** Depth substitution is known qualitatively (shortcuts to automata, Liu et al. 2023;
one-layer parity with Selective RoPE). We give the matched-depth rate on three tasks and show where it breaks.

Figures: Fig. 1 panels a-c; Fig. 2 panel a.

---

## 6. Failure by search, not capacity (2.6 pages)

Opening paragraph (3 sentences): two design choices of the content-to-angle map -- its per-head rank and the sign
parameterisation of its step -- decide whether the torus map is learned. For each we show the same three facts: (i) the
constrained class fails from scratch; (ii) a solution in the class exists, constructed explicitly; (iii) installed and
trained, that solution is kept (or re-found). Fig. 4 (`fig_existence`) shows all three facts for both axes side by side.

### 6.1 Per-head rank of the content-to-angle map (1.0 page)

**Claim 6.1.1: per-head rank decides it.** Torus T = 1024, 900 epochs, six bottlenecks built from the same initial
weights at each seed [`RANK_SEP_RESULTS.md`, `RANK3_RESULTS.md`]:

| per-head rank | arm | SOLVED | accuracy |
|---|---|---|---|
| 2 | one rank-2 latent shared by the heads | 0/8 | 0.894 |
| 2 | per-head rank-2 latents | 1/6 (fresh seeds) | 0.885 (n = 8) |
| 2 | block-diagonal rank 4 (each head reads rank 2) | 2/8 | 0.948 |
| 3 | per-head rank 3 | 6/8 | 0.987 |
| 4 | shared rank 4 | 8/8 | 0.998 |
| 4 | per-head rank 4 | 8/8 | 0.999 |

Per-head rank at matched sharing (per-head rank 4 vs block-diagonal): 8/8 vs 2/8, Fisher and permutation p 0.0070.
Rank 3 vs rank 2: +0.102 (permutation p 0.027). Total latent size does not track success (two 4-dimensional latents land
on opposite sides). Read with MapFormer's Table 2: rank 1 cannot represent a 2D map; rank 2 can, and solves 128-step
walks, but at 1024 moves reliable learning needs rank above the map's dimension.

**Claim 6.1.2: not a budget effect at 2x.** At 1800 epochs from scratch rank 2 solves 0/8 (0.908) vs rank 4 7/8 (Fisher
p 0.0014; accuracy +0.086, p 0.0003) [`LOOP_RANK_E1800_P1_RESULTS.md`].

**Claim 6.1.3: the rank-2 solution exists and is held.** The rank-2 projection of each solved rank-4 model scores 0.9955
frozen (>= 0.981 on every seed); trained from there it solves 7/8, the same as the rank-4 control (7/8), against 1/8 for
rank 2 trained further from its own stalled weights (accuracy p 0.0009) [`RANK_PROJ_RESULTS.md`]. The failure is search.

**Claim 6.1.4: rank against the map's dimension.** Rank equal to the dimension D is hard in 2D and 3D and not in 1D: 2D
torus (32 per side) rank 2 1/8 vs rank 3 8/8 (Fisher p 0.0014); 3D torus (10 per side) rank 3 1/8; rank 4 solves an
18-per-side 3D torus 8/8 (1.000); 1D ring rank 1 7/8 vs rank 2 8/8 [`RANK_ND_RESULTS.md`, `RANK_WRAP_RESULTS.md`,
`RANK_TESTS_RESULTS.md`]. Practical reading: per-head rank D + 1 learns reliably in 2D and 3D.

Figures: **Fig. 3 = `fig_rank_solved`** (a: six bottlenecks; b: rank D vs D + 1 in 1D / 2D / 3D). Fig. 4 panel a.
`fig_loss_curves_rank` in App. G (rank-2 runs flatten at 0.2-0.5; rank 3 and 4 medians cross the threshold at epochs
478-532).

### 6.2 The sign of the phase step (1.3 pages)

**Claim 6.2.1: non-negative parameterisations rarely learn the map from scratch.** Torus T = 1024: |.| 0.821 vs signed
0.998, -0.177 (permutation p 0.0002), 0/8 vs 8/8 SOLVED; softplus 0.781, 1/8 (-0.218, p 0.0003)
[`SIGN_MATCHED_RESULTS.md`]. Ring: |.| 2/8 and 1/8 vs 8/8 at p_plus 0.5 and 0.75 (Fisher p 0.007, 0.0014)
[`RING_DIAL_RESULTS.md`]. Observation-step-free models: ring 0/8 vs 8/8 [`WRAP_EXISTENCE_RESULTS.md`], torus 0/8 vs 8/8,
0.880 vs 1.000 [`GENERALITY_RESULTS.md`]. Training curves: the signed median crosses the SOLVED threshold at epoch 480 /
57 / 309 / 183 (four batches); most |.| runs flatten at loss 0.3-0.5. **Fig. 5 = `fig_loss_curves`.**

**Claim 6.2.2: the cost holds across recipes and scale.** Each step at its own best of four learning rates (ring, fresh
seeds; |.| optimum 3e-3 bracketed by the grid): |.| 2/8 vs signed 8/8, selection-aware p 0.0070 on SOLVED, 0.0057 on
accuracy [`LRSWEEP_RESULTS.md`]. Weight decay 0 (ring): 0/8 vs 8/8, -0.127. A third of the learning rate (ring): 0/8 vs
6/8 (Fisher p 0.007). Two layers and d 256 on the torus (8x the parameters): 2/8 vs 8/8 (Fisher p 0.007), 0.967 vs 0.999
(p 0.0005) [`ROBUST_RESULTS.md`]. Table 2: the recipe cells.

**Claim 6.2.3: the non-negative solution exists, and training keeps or re-finds it.** Proposition 1 constructs it: each
solved signed observation-step-free model is converted exactly into |.| form (angle error <= 1.4e-6 rad; frozen accuracy
equal to the source on 8/8; weights within ~5x of the signed model's). Softplus is constructed the same way (logits equal
to 2.8e-11; App. D). On the ring the constructed |.| model, trained 300 more epochs, stays solved on 8/8 and never leaves
the solved region, keeping its cancelling code in the channels attention uses (98% usage-weighted)
[`WRAP_EXISTENCE_RESULTS.md`, `ACCOUNTS_RESULTS.md` T1]. On the torus it is perturbed early (peak loss 0.11-0.21) and
re-finds the solution (8/8 SOLVED, 1.000) [`GENERALITY_RESULTS.md`]. From scratch the same class solves 0/8 on both.
Fig. 4 panels b-c.

**Claim 6.2.4: how signed steps reach the map (ring).** Started from the |.| start (both moves rotating the same way),
signed steps still solve 7/8 (|.| 0/8; Fisher p 0.0014), ending with 98-99% of channels opposite-signed: they cancel
opposite moves through zero, which Lemma 1 shows the barrier excludes for an inverse pair (analysis of the end state,
Fig. 6). The escape is collective: freezing the two move latents (W_out and omega still free) blocks it, 1/8 vs 7/8
(Fisher p 0.010) [`MECHANISM_RESULTS.md`, `ACCOUNTS_RESULTS.md` T2]. MapFormer described the antipodal end state (their
Figs. 3d, 9b); this is the causal test of how it is reached. **Fig. 6 = `fig_mechanism`** (move-angle planes).

**Claim 6.2.5: published monotone content phases also fail these maps.** Faithful CARoPE cannot represent cancellation
(Proposition 6, App. E) and scores 0.667 on the torus (0/8), below index RoPE's 0.722 (p 0.040). Faithful CoPE can
represent ring cancellation (constructed) yet solves 0/8 (0.887 vs signed 1.000, p 0.0002) while beating index by +0.157
[`GENERALITY_RESULTS.md`]. MapFormer's Table 2 reported CoPE failing 1D and 2D navigation; our construction shows the
obstacle is not what CoPE can represent.

### 6.3 One failure signature (0.3 page; analysis on registered strata)

Every failing model keeps short-gap revisits and loses long-gap and wrap-only ones (short / long / wrap-only): index
1 layer 0.784 / 0.462 / 0.471; shared rank 2 0.950 / 0.671 / 0.557; |.| 0.870 / 0.630 / 0.515; the solving models are
at 0.98-1.000 in every stratum (signed 1.000 / 0.995 / 0.982; rank 4 1.000 / 0.995 / 0.977) [`fig_strata_modes_out.txt`].
The failing codes do not cancel opposite moves: signed steps cancel in 70-93% of the channels attention uses, |.| in 2-10%
(analysis; `fig_wrap_share` in App. H). Fig. 2 panels b-c.

---

## 7. Maps from words (0.6 page)

**Claim 7.1: one path layer learns navigation told in English.** T = 1024 words, fresh seeds (n = 6): path 0.979 +/-
0.048 (6/6 SOLVED) vs index RoPE 1 layer 0.505 (0/6), +0.474, permutation and Fisher p 0.0022; index 2 layers 0.793
(0/6). Floors: best constant 0.505, reversal-copy 0.597. [`TEXTWORLD_RESULTS.md` CORRECTED block,
`docs/audits/2026-09-27/fresh_seeds.txt`]

**Claim 7.2: it learns which words move it, and opposite directions cancel.** On every seed (declared secondaries,
n = 8): with the component common to the four direction steps removed, north + south and west + east cancel to
0.002-0.012; zeroing the direction words' steps drops every seed to the constant floor (0.39-0.51); synonyms collapse to
one step (cosine 1.000). Half the seeds add a per-step clock alongside the map (31-38 of 64 channels) and solve as well.
**Fig. 7 = `fig_textworld`.** The same cancellation that signed steps build on the ring (Sec. 6.2) appears here over a
vocabulary.

---

## 8. Discussion (0.6 page)

**What the two failure axes share.** For per-head rank 2 and for non-negative steps, a solution in the class exists
(rank-2 projection 0.9955; exact |.| construction), training keeps it (7/8 held; ring 8/8 held, torus re-found 8/8),
and training from scratch does not reach it (0-2/8; 0/8). The obstacle is optimisation, not expressivity, the pattern
Shakerinava et al. report for non-Abelian rotations, here in single-layer Abelian tasks and produced by one rank or
parameterisation choice. Both failures leave the same trace: a code that serves short-gap revisits and does not cancel
opposite moves, so long-gap and wrap-only revisits fall to the floor.

**Design guidance.** Keep the phase step signed; give each head rank above the map's dimension (D + 1); measure map use
at walk lengths where retracing no longer suffices, against measured map-free floors.

**Integrator view.** In Howard's terms signed steps make a map and non-negative steps a path-length clock; in the rotary
slot these coincide in what they can represent (Proposition 1), so the difference we measure is in what training finds.

**Text (one sentence, analysis, small scale).** On ordinary text the learned phase behaves as a token clock: it grows
linearly with token index (R^2 0.994-0.999, TinyStories and code, n = 3), and MapWM matches index RoPE (1.5053 vs 1.5050
nats/token) -- consistent with there being no map to build [`TINYSTORIES_PILOT_RESULTS.md`,
`docs/audits/2026-10-12/alpha_language_out.txt`].

---

## 9. Limitations (0.2 page)

- Small models (1-4 layers, 2 heads, d 128-256) on synthetic map tasks; natural text is touched only at small scale.
- "Not found" is relative to the stated budgets (300-1800 epochs); learning rates were swept on the ring, while the torus
  was run at one rate and two model sizes.
- Rank results use two heads; rank equal to the map's dimension is hard in 2D and 3D but not in 1D; our latent is shared
  across heads where MapFormer's is per head (per-head arms are reported alongside).
- We show how signed steps reach the map; why the barrier parameterisations stall from small starts is not identified.
- No recurrent or SSM baselines; the torus exchange rate is a lower bound (more than four layers) measured at n = 4.

---

## Page budget

| section | pages |
|---|---|
| Title, abstract | 0.3 |
| 1 Introduction | 1.0 |
| 2 Background (incl. related work, Proposition 1, Lemma 1) | 0.8 |
| 3 Setup | 0.7 |
| 4 Sharper measurement (Table 1) | 1.0 |
| 5 Exchange rate (Fig. 1) | 0.9 |
| 6 Failure by search (Figs. 3-6, Table 2) | 2.6 |
| 7 Maps from words (Fig. 7) | 0.6 |
| 8 Discussion | 0.6 |
| 9 Limitations | 0.2 |
| **total** | **8.7** |

For CCN's 8 pages: move Fig. 6 (`fig_mechanism`) and Claim 6.2.5 to the appendix with one sentence left in 6.2, and
shrink Table 1 to two rows (T = 128, T = 1024).

## Figures and tables

| # | file | section | shows |
|---|---|---|---|
| Fig. 1 | `fig_exchange_rate` | 4, 5 | accuracy vs index layers beside one path layer: Dyck-2, ring (fresh seeds), torus; floors drawn |
| Fig. 2 | `fig_strata_modes` | 4, 5, 6.3 | revisit strata for index depth / rank / sign; stratum floors and retrace |
| Fig. 3 | `fig_rank_solved` | 6.1 | SOLVED by per-head rank; rank D vs D + 1 in 1D / 2D / 3D |
| Fig. 4 | `fig_existence` | 6 | from scratch vs installed-and-trained for rank 2, ring |.|, torus |.| |
| Fig. 5 | `fig_loss_curves` | 6.2 | training loss per epoch, signed vs |.| / softplus, four batches |
| Fig. 6 | `fig_mechanism` | 6.2 | move-angle planes on the ring: through zero vs by wrapping |
| Fig. 7 | `fig_textworld` | 7 | direction-word steps cancel; clock vs map seeds; step ablation |
| Table 1 | -- | 4 | floors and accuracies at T = 128 and 1024 |
| Table 2 | -- | 6.2 | sign cost by recipe: default, wd 0, lr / 3, best-of-four lr, 2-layer d 256 |
| App. | `fig_factorial_strata` | App. F | torus strata, index depth and loop, with this batch's own floors |
| App. | `fig_loss_curves_rank` | App. G | training loss by per-head rank |
| App. | `fig_wrap_share` | App. H | cancelling share per run, usage-weighted and unweighted |
| App. | `fig_ring_strata` | App. H | ring strata with per-stratum retrace |

---

## Appendix plan

- **A. Models, recipes, batches.** Per batch: seeds, epochs, lr, schedule, attention path, pre-launch code check
  (yes / no), reproduction audit (yes / no). Reproduction of MapFormer v4 Table 2 2D: ours 0.989 (MapWM) / 0.987 (MapEM)
  vs reported 1.00 / 1.00 (matches v1's 0.99 / 1.0); command `python3 -m mapformer.main --device cuda --epochs 16
  --n-batches 98`.
- **B. Floors.** Blank, n-gram (orders 0-5), retrace per task and stream; T = 128 vs 1024 table with wrap-only share
  (0.000 vs 0.079); ring retrace by bias (0.860 / 0.758 / 0.612); Dyck A2f floor 0.594; text-world constant and
  reversal-copy. Per-stratum floors.
- **C. Per-seed tables** for every main-text contrast; SOLVED-threshold sensitivity (0.02, 0.1, accuracy >= 0.99);
  attention-dropout evaluation scale (text-world path 0.969 -> 0.993 re-scored; no registered verdict changes).
- **D. Constructions.** Rank-2 projection; balanced |.| conversion; softplus full-turn construction; the LOST rule.
- **E. Theory.** Score identity; Proposition 1 (a: |.|, f: softplus); Lemma 1 with proof (J R(phi) J = R(-phi));
  one-channel geometry (P3 cut circle, P4 sticky wall); Proposition 6 (CARoPE cannot cancel).
- **F. Exchange rate details.** Dyck full matched-depth ladder, 3x budget, mixture training, index base 32 vs 10000;
  ring per-bias table (fresh seeds and all seeds); torus factorial with `fig_factorial_strata` and the memorisation check
  (4-layer index 0.897 own map vs 0.860 held out).
- **G. Rank details.** Five-arm table with arm definitions; rank-3 geometry; 1800-epoch table; RANK_PROJ restart curves;
  3D grids and wrap-only shares; 1D ring arms (incl. drift-forbidden rank 1, 8/8); `fig_loss_curves_rank`.
- **H. Sign details.** Ring dial by bias; recipe cells (ROBUST) and the per-rate learning-rate table (LRSWEEP: signed
  8 / 8 / 4 / 1, |.| 1 / 2 / 1 / 0 SOLVED at 1e-3 / 3e-3 / 1e-2 / 3e-2); the start table on the ring (|.| from a start
  whose opposite-move angles sum past pi solves 8/8) with its torus counterpart (0/8 within 900 epochs); the relative-sign
  test (moves +1/-1 replicate the cost, 1/8 vs 8/8); `fig_wrap_share`, `fig_ring_strata`.
- **I. CARoPE and CoPE.** Faithful implementations; the CoPE ring construction; CoPE on a non-wrapping line.
- **J. Text world.** Vocabulary, grammar, step tables per seed, clock-seed probe.
- **K. What and where (analysis).** Converged path models keep 0.974-0.989 accuracy with object identity removed from the
  score and 0.988-1.000 with a shared position kernel times a content gain; index RoPE and PoPE fall below floor
  (`docs/WHAT_WHERE_CHECKS.md`; separation as such is MapFormer's, Fig. 9). Removing the step from object tokens
  (ActOnly) or normalising the step input (NormStep) lets the model converge: 16/16 SOLVED vs 0/8 for MapWM, replicated
  on fresh seeds (8/8 vs 0/8) (`LEAK_RESULTS.md`, `GAIN_PHASE_RESULTS.md`).
- **L. PoPE score (one line).** MapPoPE's gain over MapWM on the short-walk torus comes from PoPE's score rule (+0.0243,
  p 0.012, n = 16), not from its doubled frequency count (+0.0002) (`MAPPOPE_PAIR_RESULTS.md`).
