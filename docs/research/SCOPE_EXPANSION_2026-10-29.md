# Scope expansion, 2026-10-29: when a map is needed, what Selective RoPE lacks, and what a better phase would be

Research note, not a result. CPU only; no GPU job, no code, results file, CLAUDE.md or draft edited; nothing committed.
Sources: CLAUDE.md, `docs/WHERE_THINGS_STAND.md`, `docs/paper/MAIN_DRAFT2.md`, `CLAIM_MAP_clock_map.md`,
`BIG_PICTURE_2026-10-24.md`, `docs/audits/2026-10-28/FULL_REVIEW.md`, the results files named inline, `docs/lit/LIT_*.md`,
`papers/INDEX.md`, `papers/txt/srope.txt`, `model_srope.py`, and five abstracts fetched today (Sec. E). Two CPU analyses
were run on stored outputs and checkpoints (scripts and outputs under `docs/research/scripts/`); both are **post hoc,
unregistered** and labelled so wherever quoted. Status words as in `WHERE_THINGS_STAND.md` (REG, post hoc, pilot).

Terms. *Revisit*: an observation at a cell visited before. *Gap*: moves since the last visit to that cell. *Runs spanned*:
direction runs between the two visits (2 = an out-and-back). *Retrace*: copy the observation 2j steps back while the current
run undoes the previous one (the registered map-free floor). *Wrap-only*: the cell was seen before only from another
unwrapped position (a lap away). *Map index* of an arm: mean over long-gap and wrap-only revisits of accuracy minus the
always-blank floor (defined in Sec. A4). *Clock*: phase that grows with moves whatever their direction (CLAUDE.md).

---

## A. When is a map needed? An account with six mechanisms

### A1. The mechanisms a model can use at a revisit

| mechanism | what it computes | needs | reach | where the repo measured it |
|---|---|---|---|---|
| M1 **retrace** (local undo) | while the current run reverses the previous one, copy the observation 2j steps back | fixed token offsets, runs of repeated moves | 2 runs, <= 2R moves (R = max run length, 10) | registered floors: 0.934 at T = 128, 0.843 at T = 1024 (`SIGN_MATCHED_RESULTS.md`); 4 index layers sit on it (`FACTORIAL_N8_RESULTS.md`) |
| M2 **windowed displacement code** (index attention + depth) | net displacement over the last W moves in the residual stream | depth; W grows with layers and shrinks with run variety | ring 3L: ~64 moves (probe R^2 0.72); torus 4L: decays from 2 runs spanned to the floor at 17+ (`MAIN_DRAFT2.md` Sec. 6, analysis) | `CANCEL_RESULTS.md`, `DYCK_MDEPTH_RESULTS.md`, Fig. 4 of draft 2 |
| M3 **signature match** (view recognition, in-context) | find an earlier position whose recent (observation, action) context matches, copy what followed; an induction head over n-grams | low aliasing, many prior visits per cell, contexts unique to a place; no integration, no geometry | any gap and any wrap, but only on cells visited often enough | MiniWorld: the index arm SOLVES grid 16 at T = 1024 (0.988) and grid 8, fails grid 32 (`VISITS_TEST.md`, `ALIASING_CONTROLLED.md`; n = 3-5) |
| M4 **memorised map** (weights) | the map itself, stored across training | a fixed map during training | any gap; no transfer | RNNs: own map 0.93-0.98, new map 0.57-0.65 (`RNN_TORUS_RESULTS.md`); 3D grid 10 torus: own 0.986, unseen 0.273 (`docs/WHAT_WHERE_CHECKS.md`) |
| M5 **clock** (monotone or token-count phase) | elapsed selected tokens | targets keyed to recency | any length | k-back recency: monotone phases 0.96-1.00, signed 1.000, index 0.234 (`RECENCY_RESULTS.md`) |
| M6 **map** (path-integrated phase) | net displacement at any gap, periodic in the torus | abelian actions in a known frame, signed steps, rank > D per head, clock-free used channels | any gap, any wrap, redrawn map | every torus / ring / text / pixel batch: 0.998-1.000 (8/8) |

Two mechanisms are often conflated in the paper's stratum language: M2 (index attention integrating within a window) and M3
(index attention recognising a place from context). The wrap-only stratum separates M6 from M1 and M2, **not** from M3: on a
small torus wrap-only revisits are the majority (N = 16, T = 1024: 62% of revisits, Table A3) and the index arm still solves
the map there, because the recent context identifies the place. The repo's 64 x 64 torus at T = 1024 (1-3 prior visits per
cell) makes M3 unavailable, which is why the strata read cleanly there.

### A2. Task properties and the predicted winner

| property | regime | predicted winner | evidence (file; status) | untested |
|---|---|---|---|---|
| **revisit-gap distribution / loop structure** | most revisits are 2-run out-and-backs (short walks, long runs) | M1; any model that learns retrace (one index layer learns it partly: short-gap 0.78 at T = 1024) | T = 128 torus 93% retrace; index 1L 0.805 vs path 0.971 (`PAPER2X2_RESULTS.md`, REG) | run length R as a knob (Table A3 predicts retrace share 0.41 at R = 1 to 0.70 at R = 20): no batch varies R |
| | gaps <= ~64 moves, few runs spanned (ring T = 128, Dyck depth <= 12) | M2 with ~3 layers | Dyck +0.353 / +0.130 / +0.045 / +0.024 at 1-4L; ring 3L within 0.002 of path (`DYCK_MDEPTH`, `CANCEL`, REG) | the window W as a function of depth beyond 4 (G-5 / G-10, not run) |
| | gaps >= 128 moves and >= 17 runs spanned (torus T = 1024) | M6 only; depth buys M1 | 4L / loop x4 at the retrace floor, long-gap 0.41 (`FACTORIAL_N8`, REG n = 8) | depth > 4 |
| **wrap / topology** | wrap-only revisits present (laps) | M6 with periodic channels (tick rule, Prop. 2b); M3 if contexts repeat | rank 2 and SRoPE fail most on wrap-only (0.557, 0.659) (`RANK_MI`, `SROPE_TORUS`, secondaries) | walled (non-periodic) large maps: no torus batch; `KNOB_SWEEP.md` wall +0.251 (n = 3, 16 epochs) |
| **observation aliasing / view informativeness** | contexts unique and often repeated (low aliasing, small map) | M3 | index solves grid 8 and 16 in MiniWorld (`VISITS_TEST.md`, n = 3) | a registered torus batch (E4 below) |
| | heavy aliasing (blank 50%, 16 types on 4096 cells) | M6 (needs no informative view); M3 impossible | path 0.998 at 1-3 prior visits per cell | path integration at 100% aliasing (no objects beyond blank / one type): untested |
| **action group** | abelian translations in a fixed frame | M6 (cumsum) | every torus batch | -- |
| | rotations / egocentric turn-forward, heading not given | M6 fails as a cumsum (+0.050); recoding to allocentric restores it (+0.488); a composed heading integrator (Sec. C2) is the candidate | `KNOB_SWEEP_n8.md`, `ALLOCENTRIC_RECODING.md` (n = 8 / 3, 16 epochs); Memory Maze P_raw 0.129 vs P_E 0.331 with heading given (`MEMORY_MAZE_RESULTS.md`, REG) | a learned heading integrator (E3) |
| | non-commutative relational structure (family tree) | commutative M6 does as well as non-commutative variants | +0.005 / +0.014 for 34x cost (`FAMILY_TREE_RESULTS.md`, n = 3) | -- |
| **map fixed vs redrawn** | fixed across training | M4 competes and wins for recurrent nets and small maps; path transfers anyway | RNN own-map 0.93-0.98 vs new 0.57-0.65; path 0.9998 both (`RNN_TORUS`, REG); 100-cell 2D map memorised (`WHAT_WHERE_CHECKS`, post hoc) | many-map RNN training on the torus (open, `RNN_TORUS` scope) |
| | redrawn per sequence | M6 or M3; M4 impossible | Match-Query 0.730 vs index 0.154 (`MATCH_QUERY_SCALE.md`, n = 5, not pre-registered) | -- |
| **stochastic transitions** | p = 0.10 executed-action noise, drift 13 cells | no mechanism in the repo wins: path 0.505 -> 0.215, loop 0.878 -> 0.337; ~45% of the residual score is a local observation prior | `MQ_NOISE_2X2.md`, `MATCH_QUERY_NOISE_ABLATION.md` (n = 8); InEKF / Level 1.5 is stabilisation, not inference (live negative) | a correctable map (Sec. C4) |
| **sequence length vs map extent** | prior visits per cell >= ~4 (T / N^2 large) | M3 (index solves) | grid 16 T = 1024 index 0.988; grid 8 both solve (`VISITS_TEST`, `ALIASING_CONTROLLED`, MiniWorld n = 3-5) | the same threshold on the torus with a registered design (E4) |
| | prior visits <= ~3 (512+ occupied cells at T <= 1024) | M6 | +0.275 / +0.305 at grid 32; +0.27 at grid 64 T = 1024 | T >> N^2 on the 64 x 64 torus (T = 4096: Table A3 predicts retrace share 0.46, wrap 0.33): not run |
| **recency-keyed targets** | k-back | M5; a signed phase learns the clock (alpha 0.97) | `RECENCY_RESULTS.md`, `CLOCK_SIDE_RESULTS.md` (clock-side contrasts UNMEASURED) | a task needing M5 and M6 at once (E2) |
| **navigation told in words** | variable sentence length | M1 breaks (the 2j-back offset is no longer fixed): index 1L 0.505 even on retrace-reachable revisits, 2L 0.772 < retrace 0.934; M6 unaffected | `TEXTWORLD_RESULTS.md` (REG), `tw_strata_200walks_out.txt` (analysis) | -- |
| **first-person 3D, heading given** | blind loop closures, gap >= 128 | M6 beats M1-free index 1L; four index layers unmeasured against it (and above on average) | `MEMORY_MAZE_RESULTS.md` (REG, loss pilot-informed) | whether R4 uses M3 (frame signatures): frame-swap destruction drops every arm, so it is not separated |

**The account in one sentence.** A map is needed exactly when the revisit cannot be served by a local rule (M1), by a
windowed integration of the recent moves (M2), by recognising the place from its recent context (M3), or by a map stored in
the weights (M4): that is, when gaps are long relative to the index model's window, contexts are not repeated (large map,
few prior visits, heavy aliasing), and the map is new each episode. The repo's torus at T = 1024 is in that corner by
construction; MapFormer's T = 128 is not; small maps are not (M3); Dyck and the ring are not (M2 reaches); recency is a
clock task (M5). Depth moves the M2 window and never reaches M6 on the torus within four layers; recurrence moves to M4.

### A3. CPU test 1 (post hoc): the map-need curve over task parameters

`docs/research/scripts/mapneed_curve.py` -> `out/mapneed_curve_out.txt`, `out/mapneed_curve.json`. Model-free: 40 directed
walks per cell (runs of 1..R repeated moves, R = 10 unless stated), every revisit labelled by gap, runs spanned, retrace
reachability (the registered floor's logic) and wrap-only. The predicted accuracy of a map-free reader of reach X is
share(X) + (1 - share(X)) x 0.5 (the blank guess). Validation: N = 64 gives 0.937 at T = 128 and 0.843 at T = 1024 against
the registered floors 0.934 and 0.843.

| N | T | revisits / walk | retrace share | wrap-only share | gap <= 64 | runs <= 4 | runs >= 17 | pred. retrace | pred. window 64 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1024 | 960 | 0.23 | 0.70 | 0.72 | 0.54 | 0.23 | 0.61 | 0.86 |
| 16 | 1024 | 780 | 0.29 | 0.62 | 0.48 | 0.35 | 0.48 | 0.64 | 0.74 |
| 32 | 512 | 177 | 0.63 | 0.21 | 0.73 | 0.63 | 0.25 | 0.82 | 0.86 |
| 32 | 1024 | 470 | 0.47 | 0.38 | 0.55 | 0.47 | 0.42 | 0.74 | 0.78 |
| 64 | 128 | 30 | 0.88 | 0.00 | 0.98 | 0.88 | 0.02 | **0.94** | 0.99 |
| 64 | 1024 | 334 | 0.69 | 0.09 | 0.78 | 0.68 | 0.20 | **0.84** | 0.89 |
| 64 | 4096 | 1936 | 0.46 | 0.33 | 0.53 | 0.46 | 0.46 | 0.73 | 0.77 |
| 128 | 1024 | 286 | 0.74 | 0.00 | 0.87 | 0.75 | 0.12 | 0.87 | 0.93 |
| 64 | 1024, R = 1 | 653 | 0.41 | 0.00 | 0.87 | 0.39 | 0.30 | 0.71 | 0.93 |
| 64 | 1024, R = 20 | 291 | 0.71 | 0.17 | 0.75 | 0.71 | 0.20 | 0.85 | 0.88 |
| ring 32 | 128 | 97 | 0.70 | 0.13 | 0.95 | 0.63 | 0.02 | 0.85 | 0.97 |

Readings (post hoc). (i) **Map need on the torus is set by T / N^2 and by the run length**: the retrace share falls from
0.88 (N = 64, T = 128) to 0.46 (T = 4096), the wrap-only share rises from 0 to 0.33; short runs (R = 1) remove retrace but
keep most gaps within a 64-move window, long runs (R = 20) feed retrace. The repo's T = 1024 walks (retrace 0.69, wrap 0.09,
runs >= 17 on 20%) are a moderate regime; T = 4096 at N = 64 would roughly double the map-only share and is the cheapest
way to make "depth buys only retrace" bite harder (G-7-like, ~4x the run cost). (ii) **Small maps invert the reading of the
wrap stratum**: at N = 16, 62% of revisits are wrap-only and only 29% retrace-reachable, yet the index arm solves grid 16 in
MiniWorld, so there the index model is using M3, not M1 or M2. The wrap-only stratum is a map test only where contexts do
not repeat. (iii) **The ring at T = 128 is window-sized**: 95% of revisits lie within 64 moves, which is why three index
layers reach the path layer there (M2) and why the ring is the wrong task to show "depth cannot buy the map".

### A4. CPU test 2 (post hoc): every stored stratum table, scored against the local-rule model

`mapneed_curve.py` part B -> `out/strata_crossarm.json` (58 arm means from 37 `*_STRATA.json` files at T = 1024, 2-8 seeds
each). For each arm: local-rule prediction = (n_short x acc_short + n_long x floor + n_wrap x floor) / n_all; map index =
mean(acc_long - floor, acc_wrap - floor). The correlation of (acc - prediction) with the map index is 1.000 by arithmetic
(the residual is the map index times the beyond-short share), so the usable readout is the classification:

| class | map index | arms | pattern |
|---|---|---|---|
| no map | <= 0 (19 arms: every index RoPE 1L / 4L / loop, CARoPE, all six RNN cells, the OOD rank-4 model trained at T = 128) | long-gap and wrap-only at or **below** the blank floor (-0.01 to -0.19); overall accuracy = local-rule prediction within 0.03 | these arms predict a specific wrong object beyond their reach rather than abstain; depth and the loop lift only the short-gap column (0.77 -> 0.91-0.94) |
| partial map | 0.05-0.40 (19 arms: rank 2 shared / per head / block-diagonal, \|.\|, softplus, SRoPE, SRoPE_NoSiLU, SRoPE_ST, rank-2 + loop, rank-2 + 4 layers, rank 3) | short > long > wrap in every case; e.g. rank 2 0.950 / 0.671 / 0.557, SRoPE 0.992 / 0.937 / 0.659 | a graded middle that no index or recurrent arm occupies: a path phase that fails, fails by degree and by gap, the clock signature |
| map | >= 0.40 (20 arms: every signed rank-4 arm, held / re-found constructions, SRoPEGen, the balanced \|.\| constructions) | all strata >= 0.86 | -- |

Two post hoc observations worth a sentence in the paper. (a) **Index attention and RNNs are binary on this task** (no map,
by this index), while the constrained path classes are graded: a reviewer who asks "is rank 2 just slower" can be shown that
its failure mode is unlike index attention's at every stratum. (b) **A path model trained at T = 128 carries no map at
T = 1024** (`RANK_SWEEP` rank 4: long-gap 0.625, wrap-only 0.413, map index +0.016): its code is a map for the gaps it was
trained on, not a periodic code; a map and a torus map are different learned objects (the tick rule), and the T = 128
training regime (no wrap-only revisits) never asks for the second.

### A5. Relation to the literature

- **TEM / TEM-t** (Whittington et al. 2020, 2022): "where" from path integration, "what" from sensory input, bound in
  memory; the repo's M6 and the what/where checks are this. TEM never had an index baseline at long gaps; the floors here say
  how much of a navigation benchmark TEM-style integration is needed for.
- **CSCG** (George et al. 2021, Nat. Commun. 12:2392; gradCSCG, Nikzad et al. 2607.12382): localisation by clone-structured
  sequence context under aliasing, with no coordinates: that is M3 made explicit, and its known limit (identical contexts at
  different places) is where M6 is needed. The repo's `STITCH_ATTENTION.md` (+0.131 vs index -0.005) is the one CSCG-style
  task run.
- **In-context map formation with index positions**: Dedieu et al. 2024 (2401.05946) extract cognitive maps from a
  discrete-bottleneck transformer's latents on aliased gridworlds; Fang & Rajan 2025 (2506.19686, abstract read today) train
  a transformer to in-context RL on rodent-style planning tasks and attribute its representations to in-context structure
  learning and cross-context alignment, with intermediate computations cached in memory tokens; Wang et al. 2025 (2509.22353,
  ICLR 2026) split in-context world-model learning into environment recognition and environment learning with error bounds.
  All three are M3 / M4-ish accounts with index positions; the account here says where they stop (contexts that do not
  repeat) and what a phase adds there. Cite, as FULL_REVIEW M2 asks.
- **Shortcuts and depth**: Liu et al. 2023 (2210.10749; global count in two layers in principle), Merrill et al. 2024
  (2404.08819; SSMs and transformers in TC^0 cannot track permutation composition), Heo et al. 2026 (2603.05573; error falls
  exponentially with depth in parallelizable models). M2's window is what training finds of Liu's construction; the torus
  needs the abelian count at precision 1/2048 that Remark 1 of draft 2 already says training does not find.
- **Representability of the step**: Sarrof 2024, Grazzi 2025, Shakerinava 2026, Khavari 2025 (CLAUDE.md sign row); Shin et
  al. 2026 (2608.30692, abstract read today) reach the sign conclusion for video world models on a Shell-Game task, with
  Memory Maze as a secondary setting: linear attention tracks hidden state only with negative transition eigenvalues; no
  positional encoding is discussed.
- **Rank**: Larson 2026a/b (2609.12259, 2609.17594: rank caps give a sharp recovery transition at the number of bindings;
  abstract read today) and Sarkar, Mukherjee & Sato 2026 (2610.34945, abstract read today: with orthogonal transitions a
  readout composes a finite group exactly only at or above the minimal faithful dimension, and empirically the minimal width
  learns MORE reliably than twice it, 17/17 vs 7/18). Ours is the other direction on a sparse revisit loss with a shared
  readout and decay; the reconciliation sentence of FULL_REVIEW 3a stands.

---

## B. Selective RoPE against MapFormer: mechanism decomposition

### B1. What the faithful module does (from `model_srope.py`, authors' code reproduced to 6e-6)

Per layer, per head: input = l2-normalised per-head chunk of the layer's normed input (64-d); phi = per-head weight-normed
full-rank projection to 32 channels; depthwise causal conv (width 4) then SiLU; optional bias exp(b) (state-tracking config
only, +1 rad per token per channel); gate sigmoid(Linear(x)) per head and position; angle = temperature x cumsum(phi) with
temperature = 500000^(-2c/64) **fixed** (c = 0..31: 1, 0.66, ..., 3e-6). MapFormer: angle = omega x cumsum(W_out W_in e),
omega **learnable**, initialised 2 pi (1/Delta_max)^(i/31) with Delta_max = grid size, one rank-4 latent shared by heads,
computed once from the token embedding before any block.

### B2. CPU test 3 (post hoc): per-channel readout on the stored SROPE_TORUS checkpoints

`docs/research/scripts/srope_channels.py` -> `out/srope_channels_out.txt`, `out/srope_channels.json`. Seeds 1700-1707, four
held-out-map eval walks each, layer-1 angle. Per channel: resolution (rms wrapped phase difference over random observation
pairs), same-cell phase error by stratum (rms wrapped difference between a revisit and the last visit to the same cell; a map
channel has ~0 everywhere, a clock channel grows with the gap, a non-periodic channel fails on wrap-only revisits only), and
the lap-periodicity residual |wrap(64 a_move)| of Prop. 2b. "Clean" = error / resolution < 0.25; "resolving" = resolution
> 1 rad. Unweighted by attention usage (a usage-weighted version is a CPU follow-up).

| arm (n = 8) | resolving channels of 64 | clean on short / long / wrap-only, share of resolving | periodicity residual (rad) | clean-on-wrap and resolving (count) | accuracy, SOLVED (REG) |
|---|---|---|---|---|---|
| MapWM (signed r4) | 54 | 1.00 / 0.99 / 0.55 | 0.79 | 30 | 1.000, 8/8 |
| SRoPEGen (our swap: full rank, conv, gate, one learnable temperature) | 46 | 0.64 / 0.31 / 0.13 | 1.16 | 6 | 0.985, 5/8 |
| SRoPE_NoSiLU | 17 | 0.74 / 0.43 / 0.23 | 1.05 | 4 | 0.969, 2/8 |
| **SRoPE (faithful, best LM config)** | **15** | 0.69 / 0.27 / **0.06** | 0.96 | **0.8** | 0.961, 0/8 |
| SRoPE_ST (state-tracking config) | 64 | 0.00 / 0.00 / 0.00 | 1.60 | 0 | 0.769, 0/8 |

Readings (post hoc, descriptive). (i) **Three quarters of the faithful module's channels never move** (median resolution 0.01
rad): with |phi| of order 1 and the fixed base-500000 ladder, only the top ~15 rungs can turn far enough to resolve a 64-cell
axis; the model works with a quarter of its phase capacity. (ii) **Of the channels that resolve, most carry a clock**: clean
on short gaps (0.69) but not long gaps (0.27); the observation steps (0.86-0.96 of a move, `SROPE_TORUS_RESULTS.md`) are a
per-token tick. (iii) **Almost none is periodic** (0.8 channel per model clean on wrap-only revisits, periodicity residual
0.96 rad vs 0.79 for MapWM, 1.57 for random angles): the fixed ladder is incommensurate with the torus, and the wrap-only
stratum (0.659) is where SRoPE loses most. MapWM's learnable ladder, initialised on the grid size, ends with ~30 periodic
resolving channels. (iv) **The state-tracking config is a pure clock** in every channel: the +1 rad bias at temperature 1
dominates the content step on every token, so it is index RoPE at base 1 with a small content modulation, and fails like the
non-negative classes (matches the registered reading). (v) SiLU removal and our one-temperature swap each recover part of the
long-gap and wrap columns, in the order of their SOLVED counts.

### B3. The differences, one by one

| difference (SRoPE vs MapFormer) | identified as a cause of SRoPE's torus failure by our data? | evidence | minimal tweak and predicted outcome |
|---|---|---|---|
| **Who steps the phase**: every token through the layer input vs a learned step that observations drive to ~0 | **yes, for the long-gap (clock) part**, by analogy, not by a registered SRoPE arm | observations step 0.9x moves (REG secondary); B2 (ii); ActOnly / NormStep remove the leak and converge 16/16 vs 0/8 (`LEAK_RESULTS.md`, REG); image-driven steps collapse training 5/8 (`IMAGE_TORUS`, REG) | **T1 action-only source**: phi = 0 at observation tokens (one mask; the gate's job done by construction) or NormStep-style input. Predicted: long-gap clean share 0.27 -> ~1, SOLVED 0/8 -> several, wrap-only still low |
| **Frequency ladder**: fixed 500000-base temperature vs learnable omega from the grid size | **yes, for the wrap-only part and for capacity** (B2 i, iii); post hoc | B2 | **T2 MapFormer ladder**: temperature -> learnable omega_c = 2 pi (1/64)^(c/31). Predicted: resolving channels 15 -> ~50, periodic channels 1 -> ~20-30, wrap-only 0.66 -> > 0.95. (A base-500000 ladder is a language-model choice: RoPE's long-wavelength rungs; on a 64-cell axis they are dead weight) |
| **Causal conv (width 4) + SiLU**: context-dependent, late-by-one, non-negative-biased step | partly: SiLU no difference shown (2/8 vs 0/8, p 0.57, REG secondary); conv not isolated in a clean arm (`SELECTIVE_ROPE.md` confound) | context CV 0.11-0.13 (REG secondary); SiLU makes the step distribution asymmetric (SiLU(x) >= -0.28), a soft version of the non-negative bias | **T3 skip_conv** (the authors' own branch) at the embedding: predicted small gain; keep the conv if T1 + T2 solve (it is the property that lets a deeper layer's step read context, Sec. C4) |
| **Bias and sign**: exp(bias) +1 rad per token (ST config) | **yes** for the ST config (REG: negative-mass share 0.00; fails like \|.\|) | B2 (iv) | drop the bias or initialise it at 0 / negative; the best-LM config has no bias |
| **Per-head rank / shared latent**: full rank per head (32 independent functionals), no shared latent | not identified; rank > D is fine (rank 4 / 8 / 16 / 32 all 8/8 OOD, `RANK_SWEEP`); but no collective cancellation route (Prop. 4a needs a shared latent) | partial cancellation 0.75-0.80 vs 1.00 (REG secondary); SRoPEGen with full rank solves 5/8 | **T4 shared rank-4 latent** before the per-head projection (their low-rank option exists, rank 32 shared, ST config). Predicted: faster cancellation; not needed if T1 + T2 solve |
| **Gate**: sigmoid per head and position from x | not a suppressor of observations (1.35x on actions vs observations, `GATE_PROBE.md`, SRoPEGen) | GATE_PROBE; faithful SRoPE's obs ratio 0.86-0.96 with the gate on | a gate is the right shape for T1 but does not learn it from a 0.5 start; init its bias at -4 (tokens must earn a step; Mamba's dt-bias trick) if a learned source is wanted (Sec. C3) |
| **Placement**: per layer from the layer input vs once from the embedding | not a cause at 1 layer (norm1(emb) is a fixed function of the token); it is the property MapFormer lacks for correction (C4) | `LIT_CONTEXT_STEPS.md`; CTXSTEP HSR pilot (hidden-state step learns far cues 4/4, pilot) | keep |
| **Real vs complex / rotation vs decay**: both rotate; SRoPE composes with decay gates in GLA, not here | not applicable at 1 softmax layer | G3 SSM: the sign cost holds without decay (`G3_SSM`, REG) | -- |
| **Normalisation**: l2-normalised per-head input + weight norm | not a cause; it is a NormStep-like protection (the leak is a norm effect, `WHAT_WHERE_CHECKS` 3) | -- | keep |
| **Budget**: every faithful run still DESCENDING at 900 epochs | the failure is budget-scoped (REG) | `SROPE_TORUS_RESULTS.md` | any repair claim must be at the same budget with the anchor |

**Minimal repair, predicted**: T1 + T2 (two knobs, +0 parameters, the authors' module otherwise intact) solve the T = 1024
torus 8/8 within 900 epochs; T1 alone fixes long gaps and leaves wrap-only revisits near 0.7; T2 alone fixes wrap-only
revisits on the channels that resolve and leaves the clock. Risk: T2 changes the optimiser's relative step size on omega
(rule 15); the MapWM anchor shows the ladder trains at this recipe.

---

## C. Superior properties: designs

Each with parameterisation, why, what supports or contradicts it in the repo, the cheapest decisive experiment (costs from
measured timings: one-layer torus run at T = 1024 / 900 epochs 0.3-0.6 GPU-h on a 4090 at two jobs per GPU (`EXPTORUS`,
`TORUS_LRSWEEP`, `RNN_TORUS`), 1.3 GPU-h for the SRoPE module on a shared GPU (`SROPE_TORUS`), 4 layers ~3x; ring T = 128 /
300 epochs 0.1-0.4 GPU-h; Memory Maze ~0.55 A40-h per run), and the risk. Costs are per batch of n = 8 unless stated.

### C1. Map heads and clock heads: one model that is a map where actions move and a clock elsewhere

*Parameterisation.* Per head h a step source s_h in {actions-only latent, all-token latent, constant}, so head A's phase is a
path phase and head B's a content clock or the index; or, without a per-head switch, one signed rank-(D+1) step with no
constraint and a task that rewards both. *Why.* Draft 2 Sec. 8 shows 4/8 text-world seeds park a per-move clock in 31-38
unused channels beside the map; recency models learn a clock (alpha 0.97); Thm 1 of `THEORY_KERNEL.md` (C9) says map and
clock coexist in separate channels given rank. The open question is whether training finds both when both are *required*,
and whether that needs the spare dimension (parking) the rank result points to. *Supports.* C9 (REG secondary), RECENCY,
RANKCLOCK (clock removed -> rank 2 solves). *Contradicts.* Nothing; no mixed-demand task exists in the repo.
*Experiment E2 (mixed demands).* Torus T = 1024 stream with two target types: object at revisits (map) and, at query tokens,
the object seen k observations ago (k-back, k <= 64; clock). Arms: signed shared rank 4; per-head rank 2; per-head rank 2
with one head's step replaced by the index (the explicit map-head / clock-head split); per-head rank 3; index RoPE 1L. n = 8,
900 epochs, ~40 runs x 0.5 GPU-h = 20 GPU-h (~10 h wall on kalman). Registered contrasts: (i) rank 2 vs rank 4 on the map
targets with the clock demand present (prediction: rank 2 falls further than on the pure torus, 0/8, because the clock is
now required rather than removable); (ii) the explicit split vs rank 4 (prediction: equal, 8/8 on both target types);
(iii) rank 3 vs rank 2 (prediction: rank 3 solves). Branch rules against the noise floor as in RANKCLOCK. *Risk.* The two
demands may be served by different heads spontaneously at rank 4 (then (ii) is uninformative but (i) still reads). This is
the cheapest registered test of the parking account that FULL_REVIEW calls untested, with a new lever.

### C2. Egocentric actions without a given heading: a composed heading-then-position integrator

*Parameterisation.* Two cumsums. Heading phase eta_t = kappa x cumsum(W_h e_t) over turn tokens (a 1D ring, kappa = 2 pi /
n_headings); position step Delta_t = f(e_t) x [cos eta_t, sin eta_t] with f the learned forward magnitude, theta_t = omega x
cumsum(Delta_t). The step of a forward token depends on the integrated heading: the SE(2) composition R^2 x| SO(2) written
as an abelian integrator feeding a second one. O(T), no attention, drop-in for `ActionToLieAlgebra`. With momentum (Memory
Maze) the turn step reads (a_{t-1}, a_t). *Why.* The repo showed the collapse is representational: rotate actions +0.050,
the allocentric recoding (heading given) +0.488 (`KNOB_SWEEP_n8.md`, `ALLOCENTRIC_RECODING.md`); Memory Maze P_raw 0.129 vs
P_E 0.331 with the compass; the class-E step map (prev action, action) x heading is this design with eta given. *Supports.*
The above; Prop. 1 (position in the phase) extends unchanged once Delta_t is the true displacement. *Contradicts.* Family
tree: non-commutative models buy nothing there (`FAMILY_TREE_RESULTS.md`), but that structure has no heading; H12's budget
curve is bimodal (12 headings). *Experiment E3a (rotate torus).* `action_mode rotate`, `score_moves_only`, T = 128, converged
recipe (400 epochs cosine), n = 8: arms commanded-cumsum (fails, +0.05), composed integrator (new), allocentric oracle
(ceiling 0.99), index 1L. 32 runs x ~0.3 GPU-h = ~10 GPU-h. Registered contrast: composed vs commanded on SOLVED and accuracy;
prediction composed within 0.02 of the oracle. Gate first (rule 11): the rotate stream's n-gram shortcut (`KNOB_SWEEP.md`
VOID row) is removed by `score_moves_only`; re-measure. *Experiment E3b (Memory Maze without the compass).* One new arm
P_comp in the existing pipeline (heading integrated from turn tokens with momentum), n = 8, ~5 A40-h; contrast vs P_raw
(0.129) and P_E (0.331). Prediction: between them (momentum and collisions make the integration lossy; P_odo's 0.54 is the
ceiling). This is the arm FULL_REVIEW says would license "navigates from pixels". *Risk.* E3a is a representational fix
with a known oracle; low. E3b: heading from momentum-affected turns may drift within 1000 steps; medium.

### C3. Learning the step source instead of ActOnly

*Parameterisation.* NormStep (LN before the step; exists) plus a per-token gate g_t = sigmoid(w . LN(e_t) + b) with b = -4 at
init and an L1 penalty on g at observation tokens, so tokens earn a step; or SRoPE's gate with that init. *Why.* ActOnly needs
an oracle action label; in language "only actions move" is wrong (aside words must move their noun off the cell: TW post
hoc). *Supports.* NormStep removes the leak without the label (`LEAK`, `GAIN_PHASE`, REG); the text world learns steps for
asides that beat the direction-word oracle (post hoc). *Contradicts.* A sigmoid gate from a 0.5 start does not learn to
suppress observations (`GATE_PROBE.md`: 1.35x; faithful SRoPE 0.9x); an explicit content gate "separates 4.16x and buys
nothing" (`GATED_RESULTS.md`, live negative). So the gate must start closed. *Experiment.* Fold into E1 as one arm (SRoPE
with gate bias -4) rather than a batch of its own; on the token torus it would only re-measure LEAK. *Risk.* Medium: a
closed gate at init is a parameterisation change (rule 15); the leak is a 0.5-1% effect on tokens, so power needs pixels or
words (IMAGE_TORUS block B).

### C4. A correctable map (noise, landmarks)

*What not to propose.* Level 1.5 / InEKF (stabilisation, not inference; benefit does not grow with drift), refining theta
across depth, filter x loop, TW_LANDMARK (a cumsum cannot be corrected by a name): all live negatives.
*Parameterisation that is not one of them.* SRoPE's per-layer placement used on purpose: layer 1 a plain path phase; layer
2's step reads layer 1's attention output (CTXSTEP's HSR form, Delta = W(e + alpha LN(h1)), alpha = 0 at init), so a
landmark observation that layer 1 retrieves can add a corrective step to layer 2's phase. *Why.* Under stochastic
transitions the cumsum drifts (13 cells at p = 0.10) and every arm collapses (`MQ_NOISE_2X2.md`); the only integrator that
can re-anchor is one whose step sees context, which MapFormer's token-level step cannot and SRoPE's layer-2 step can.
*Supports.* HSR pilot learned a far-cue step 4/4 (pilot, n = 1-2). *Contradicts.* The premise failed in TW_LANDMARK because
there was nothing to correct; it holds only with transition noise. *Experiment E5.* Torus T = 1024 with p_transition 0.05
and 0.10 and landmark observations (unique objects at fixed cells), arms: 1L path, 2L path, 2L path with HSR step, 2L index,
n = 8, 900 epochs: 64 runs x ~0.8 GPU-h = ~50 GPU-h. Registered contrast: HSR vs 2L path on long-gap revisits after a
landmark. *Risk.* High: Match-Query under noise showed a non-positional prior worth 45% of the score; the floor must be
measured with a landmark-free control first (rule 1).

### C5. Language

MapWM equals RoPE on TinyStories and costs +0.0056 bpc on code (n = 3; clock-dominated channels). The design that follows
from C1 is a hybrid in which half the channels keep the index phase and half a signed content step with the SRoPE placement;
SRoPE's own language gains are at 370M in GLA / FoX, and HGRN Table 11 reports a data-dependent phase as worse. A follow-up,
not this paper: ~20-40 GPU-h at the TinyStories size, n = 3-4, and the reading would be "no worse" at best. Keep the
TinyStories sentence as a scope statement.

### C6. Design guidance the data already support (no new runs)

Keep the increment signed; keep observations from stepping the phase; give the ladder the task's period (learnable omega
from the grid size); give each head rank D + 1 with decay; where inverse actions are known, tie their steps. The new sentence
from Sec. B: a content-dependent rotary phase built for language (base-500000 ladder, all-token source) is a quarter-capacity
clock on a torus, and two knobs make it MapFormer.

---

## D. Ranked experiment plan

| rank | id | what | cost | decisive contrast | serves | prediction (and what would overturn it) |
|---|---|---|---|---|---|---|
| 1 | **E1 SRoPE repair** | torus T = 1024, 900 ep, n = 8: SRoPE anchor; +T1 action-only source; +T2 MapFormer ladder; +T1+T2; +T1+T2 with gate bias -4 (C3); MapWM anchor; index RoPE anchor. 56 runs x ~1.3 GPU-h = ~70 GPU-h (~20 h wall on kalman's two 4090s at two jobs each; or one A40 night) | SOLVED and accuracy vs the SRoPE anchor, Fisher + permutation, Holm over 4 contrasts; secondary: long-gap and wrap-only strata; the B2 per-channel readout registered as a secondary (clean-on-wrap resolving count) | **main paper** (Sec. 7.3's Selective RoPE paragraph becomes a mechanism: "two knobs"; also design guidance) | T1+T2 8/8; T1 alone long-gap > 0.95, wrap-only < 0.8; T2 alone wrap-only > 0.9 on resolving channels, long-gap unchanged. Overturned if T1+T2 stays 0/8 (then the conv / placement is the cause) |
| 2 | **E3a composed heading integrator** | rotate torus T = 128, 400 ep, n = 8, four arms, ~10 GPU-h; gate the stream first | composed vs commanded; composed vs allocentric oracle | **main paper** (turns the egocentric Limitation into a result; cheap) | composed within 0.02 of the oracle, 8/8 |
| 3 | **E4 map-need validation on the torus** | N in {16, 32, 64}, T = 1024, index 4L and path 1L, n = 4 (fresh seeds), plus the registered observation-permutation destruction on every arm: 24 runs, 4L x 3 -> ~40 GPU-h | index 4L at N = 16 vs N = 64 on long-gap AND wrap-only revisits; destruction drop | **main paper** (Sec. 5-6's account: signature matching vs map; replaces the MiniWorld n = 3 threshold with a registered torus result) | index 4L >= 0.95 at N = 16 including wrap-only (M3), at the retrace floor at N = 64; destruction collapses N = 16 index to the floor. Overturned if 4L fails at N = 16 (then the MiniWorld threshold was about views, not visits) |
| 4 | **E2 mixed demands** | torus + k-back queries, 5 arms, n = 8, ~20 GPU-h | rank 2 vs 4 with a clock required; explicit split vs rank 4 | main paper if it lands before submission (a registered parking test), else follow-up | rank 2 0/8, rank 3 and 4 and the split 8/8 on both target types |
| 5 | **E3b Memory Maze without the compass** | one arm P_comp, n = 8, ~5 A40-h (+ R4 and GRU to 1200 epochs if run anyway, 20-30 A40-h) | P_comp vs P_raw and P_E | follow-up (new design; licenses "from pixels") | 0.2-0.3 |
| 6 | **E5 correctable map** | noisy torus with landmarks, 64 runs, ~50 GPU-h | HSR vs 2L path after a landmark | follow-up (premise and floor must be gated first) | uncertain |
| 7 | **CPU** | usage-weighted B2 readout; the same readout on the GENERALITY (\|.\|, CARoPE) and RANK checkpoints; the T = 4096 floor on stored walks; registered-style strata for Memory Maze R4 by frame-signature availability | 0 GPU | -- | main paper appendix |
| 8 | language hybrid (C5) | TinyStories size | 20-40 GPU-h | follow-up | "no worse" |

Not proposed (live negatives or already answered): InEKF variants, a content gate as a separate axis, theta refinement,
TW_LANDMARK, more clock-exit slot variants, FACTORIAL beyond n = 8.

---

## E. Novelty and positioning (Q3)

| item | new relative to 2025-26 work? | nearest work | belongs in |
|---|---|---|---|
| The six-mechanism account with measured floors (retrace, window, signature, memorised, clock, map) and the T / N^2 x run-length map-need curve | the *account* is new as a measurement frame; its pieces are known (TEM, CSCG, Liu shortcuts, in-context map papers) | Dedieu 2024, Fang & Rajan 2025, Wang 2025 (index-position map formation = M3/M4); Merrill 2024, Liu 2023, Heo 2026 (depth); George 2021 / Nikzad 2026 (M3 explicit) | **main paper** Sec. 3 / 5 framing: one table and the curve; cite the three in-context papers (FULL_REVIEW M2) |
| Index attention and RNNs are binary, constrained path phases graded (A4) | small, new, post hoc | -- | main paper, one sentence + appendix table |
| Selective RoPE decomposition: quarter-capacity (fixed ladder), clock (all-token source), incommensurate ladder (wrap) | **new**: nobody has measured SRoPE on navigation or by channel; the ladder-period point is not in SRoPE, GRAPE or Mamba-3 (which inherit RoPE's base) | Selective RoPE 2511.17388 (tests parity, MQAR, LM); Mamba-3 2603.15569 (data-dependent RoPE, Prop. 3); GRAPE 2512.07805 (non-negative contextual variant); CARoPE (rank 1, bounded) | **main paper** once E1 runs; post hoc readout as an appendix now |
| Composed heading-then-position rotary integrator (C2) | new as a positional encoding; classical as dead reckoning; PaTH / DeltaProduct do general non-commutative products with Householders (`path.txt`, 2502.10297), ViewRope / PRoPE / MeRoPE take camera pose as given; DriveGATr (2604.01466) is SE(2)-equivariant attention over multivectors, not a phase integrator | PaTH 2505.16381, DeltaProduct 2502.10297, ViewRope 2602.07854, PRoPE 2507.10496 | main paper if E3a lands (one paragraph); E3b a follow-up |
| Map heads + clock heads, mixed demands as a test of parking (C1) | new as an experiment; "positional vs semantic heads" in trained RoPE models is prior art (Song & Zhong 2024, Barbero 2025, cited in BIG_PICTURE) | Howard 2014 Cases I-III (frame); CoPE (a content clock) | follow-up unless it lands early; it strengthens the rank mechanism, which the review calls the weakest claim |
| Learned step source with a closed gate (C3) | incremental (Mamba dt-bias, ReZero, RWKV-7 LoRA inits are the same trick; `LIT_CONTEXT_STEPS.md`) | Mamba 2312.00752, Selective RoPE's gate | arm inside E1 |
| Correctable map via a per-layer step (C4) | the placement is SRoPE's; using it for landmark correction under transition noise is new in this line but risky | CTXSTEP (pilot), SRoPE placement | follow-up |
| Language hybrid (C5) | not new (SRoPE's own LM result; HGRN's negative) | -- | follow-up |

Scoop check today (web, standard searches): no follow-up to Selective RoPE on state tracking or navigation indexed; no
paper combining CSCG-style localisation with transformers in context beyond Swaminathan 2023; no SE(2) rotary integrator
with a learned heading. Sarkar et al. 2610.34945 and Larson 2609.17594 remain the two concurrent rank papers to reconcile.

---

## F. References (ids; those marked * were read first-hand today as abstracts; the rest are in `papers/` or `docs/lit/`)

Rambaud et al., MapFormer, 2511.19279. Movahedi et al., Selective RoPE, ICLR 2026, 2511.17388. Lahoti et al., Mamba-3, ICLR
2026, 2603.15569. Zhang et al., GRAPE, ICLR 2026, 2512.07805. Yang et al., PaTH, NeurIPS 2025, 2505.16381. Siems et al.,
DeltaProduct, NeurIPS 2025, 2502.10297. Veisi et al., CARoPE, 2507.23083. Golovneva et al., CoPE, 2405.18719. Grazzi et al.,
ICLR 2025, 2411.12537. Sarrof et al., NeurIPS 2024, 2405.17394. Shakerinava et al., ICLR 2026, 2603.01959. Merrill, Petty &
Sabharwal, ICML 2024, 2404.08819. Liu et al., ICLR 2023, 2210.10749. Heo et al. 2026, 2603.05573. Howe 2026, 2609.18966.
Larson 2026a, 2609.12259; Larson 2026b*, 2609.17594. Sarkar, Mukherjee & Sato 2026*, 2610.34945. Dedieu et al., ICML 2024,
2401.05946. Fang & Rajan 2025*, 2506.19686. Wang et al. 2025*, 2509.22353 (ICLR 2026). Shin et al. 2026*, 2608.30692.
George et al. 2021, Nat. Commun. 12:2392 (CSCG). Nikzad et al. 2026, gradCSCG, 2607.12382. Swaminathan et al. 2023,
2307.01201. Whittington et al. 2020 (TEM, Cell 183:1249); Whittington, Warren & Behrens 2022 (TEM-t, 2112.04035). Howard et
al. 2014, J. Neurosci. 34:4692. Pasukonis et al. 2022, Memory Maze, 2210.13383. Laird & Clark 2026, 2512.06983. ViewRope
2602.07854; PRoPE 2507.10496; DriveGATr 2604.01466. Kim et al. 2024, 2406.15275; Martorell 2025, 2502.16690. Li et al.,
GridPE, 2406.07049. Mamba, 2312.00752. HGRN, 2311.04823. Hahn & Rofin 2024, 2402.09963.
