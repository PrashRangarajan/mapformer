# Scope expansion (independent opinion, Opus), 2026-10-29

Written without reading `docs/research/SCOPE_EXPANSION_2026-10-29.md` or anything else under `docs/research/` by the
second researcher. Inputs: CLAUDE.md, `docs/WHERE_THINGS_STAND.md`, `docs/paper/MAIN_DRAFT2.md`, `CLAIM_MAP_clock_map.md`,
`BIG_PICTURE_2026-10-24.md`, `docs/audits/2026-10-28/FULL_REVIEW.md`, the results files named in each section,
`docs/lit/LIT_*.md`, `papers/INDEX.md`, a few web searches (Sec. 6), and **new CPU analyses of stored checkpoints, all
post hoc and eval-only** (scripts and outputs in `docs/research/scripts_opus/`; `out/q1_revisit_rows.json.gz` must be
gunzipped before re-running `q1_cover_strata.py`). No GPU job ran; no code, results file, draft or CLAUDE.md was edited;
nothing was committed.

Status words: **REG** = a registered result of the repo; **post hoc** = my CPU analysis today on stored checkpoints of
registered batches (one batch per arm, thresholds chosen today, not registered); **model-free** = walk statistics only;
**prediction** = not tested.

---

## 0. Summary

1. **A principled "when is a map needed" (Sec. 1).** Classify each revisit by the *covering space* on which it is a
   revisit: the Cayley **tree** of the free group (the walk returns by cancelling moves against their immediate inverses:
   retrace and nested retrace), the **plane** Z^D but not the tree (the return needs moves to *commute*: a loop such as
   E N W S), or the **torus** only (the return needs the *wrap*). Post hoc on the 8 FACTORIAL / FACTORIAL_N8 seeds (51,132
   revisits at T = 1024): 70.3% are tree revisits, 22.1% plane-only, 7.6% torus-only. Four index layers and the loop solve
   tree revisits (0.987 / 0.994, including nested non-retrace ones up to ~64 moves old: 0.80-0.98) and sit at the blank
   level on plane-only (0.511 / 0.540; blank 0.507) and torus-only revisits (0.434 / 0.501), except plane loops closed
   within ~16 moves (0.87-0.89). One path layer: 1.000 / 1.000 / 0.992. Age-matched, the deep index models are 0.17-0.24
   better on nested tree revisits than on plane-only ones (8/8 seeds each). The cover shares alone predict the deep-index
   ceiling (0.703 + 0.297 x 0.51 = 0.854 against 0.840 / 0.856 measured). This one distinction also explains why depth
   catches up on Dyck-2 and the 1D ring (no plane-only stratum exists there: Dyck is a stack, and in 1D the free group is
   Z) and not on the torus. Two other map-free routes complete the account: **sequence recognition** (transition recall,
   CSCG-like), which becomes strong in small, densely revisited maps (model-free surrogate: in-context induction floor
   0.86 in a 6x6 room vs 0.47 in 32x32) but which trained index models do not use on the 64-torus (it would add 0.011
   over retrace there); and **memorisation** of a fixed map (RNN_TORUS). A map is needed for revisits outside all three
   routes; stochastic transitions bound its useful horizon (post hoc: at 1% transition noise the path model, trained
   noise-free, falls to 0.43 at gaps 64-256 and 0.18 beyond, below four index layers).
2. **Selective RoPE vs MapFormer (Sec. 2).** Post hoc on the 48 SROPE_TORUS checkpoints: (a) the registered readout
   "observations step 0.86-0.96 as far as a move" is mostly an artefact of the causal convolution: 70% of each token's step
   lands one to three tokens later (|tap| share by lag 0/1/2/3 = 0.29 / 0.31 / 0.22 / 0.18), so the first difference at an
   observation carries the previous move. Removing one observation's projection changes the phase by **0.34x** (SiLU) /
   **0.14x** (no SiLU) of a move's effect: still 5-13x MapWM's 0.027, but not "the leak at full size"; the gate stays open
   on every token class (0.94-0.99). (b) **Faithful Selective RoPE starts with essentially no phase**: per-move angle at
   initialisation median ~0, 90th percentile 1e-4 rad (MapWM: median 0.086, 62% of channels above pi/64); after 900
   epochs only 7-9% of its channels exceed pi/64 per move. This is consistent with every run still descending. (c) Its
   deficit is concentrated at **wrap-only revisits** (same-place phase coherence C0 0.29 vs MapWM 0.91; long gap 0.89 vs
   0.995): its per-move angles are not commensurate with the 64-cell torus (usage-weighted cos(64 m) 0.56 vs MapWM 0.93;
   r +0.64 with wrap-only accuracy over seeds), while out-and-back cancellation is fine (0.98). It learns a *plane* map
   without the torus quotient. Minimal tweak: a map-scale initialisation and a learnable per-channel temperature
   (Selective RoPE already has `temp_grad`), predicted to solve.
3. **Ranked plan (Sec. 4).** For the current paper: fold in the CPU results (0 GPU-h); **E1** Selective RoPE minimal-tweak
   factorial (~16 GPU-h); **E2** a registered *commutation dial* that makes the cover account causal (~6-8 GPU-h);
   **E3** a Memory Maze blind-run-length dial (~18 A40-h). Best follow-up-paper designs: an SE(2) two-level integrator for
   egocentric actions without heading, and a conv-driven step for self-motion inferred from views (where Selective RoPE's
   convolution should help rather than hurt).

---

## 1. When is a map truly needed?

### 1.1 Setup and the three covers

A walk applies group elements g_{a_t} to a state x_t; the observation is o_t = M(x_t) with the map M redrawn per
episode (torus, ring, text world, image torus, Memory Maze) or fixed (MiniWorld fixed-map; the torus *training* map in
RNN_TORUS). The target is o_t at a revisit, so a predictor must find an earlier token at the same state. The loop word
w = a_{s+1} ... a_t between the two visits is trivial in the state group. The question is *in which group it is already
trivial*:

| cover | the loop word is trivial in | example on the torus | what suffices |
|---|---|---|---|
| tree | the free group F_k (every move cancels its immediate inverse, stack order) | E^3 W^3 (retrace); E^2 N S W^2 (nested) | a stack / local reversal rule; no count |
| plane | Z^D but not F_k | E^2 N^3 W^2 S^3 | commutation: a signed count per axis (a metric map) |
| torus | Z_N^D only | E^64 | the count modulo N (wrap-commensurate frequencies) |

Operationally (`q1_cover_strata.py`): the tree node at step t is the freely reduced prefix word (a stack: push a move or pop
its inverse), the plane node is the unwrapped displacement, the torus node the cell; a revisit is classed by the most free
cover on which its node was seen before. Tree ⊂ plane ⊂ torus. The repo's existing strata are a coarser version: retrace
is the depth-1 tree case, "wrap-only" is the torus-only class, and "long gap" mixes tree and plane revisits.

Plus three routes that do not integrate at all:

- **V, sequence recognition / transition recall.** The recent (action, observation) window identifies the predecessor
  cell and the entering edge was traversed before; an induction head copies what followed it (CSCG's clone graph is the
  Bayesian version). Needs informative views and re-traversed edges, i.e. small maps relative to T.
- **A, anchoring.** A unique view (a landmark) or a wall contact resets the state (Liu et al.'s bounded gridworld is an
  aperiodic, reset-like automaton with a learnable depth-2 shortcut, their Thm. 3); only the loop since the last anchor
  needs integrating.
- **W, weights.** With a fixed map, the map can be stored in the weights and localisation needs only the observation
  stream (RNN_TORUS: own map 0.93-0.98, new map 0.57-0.65).

**Claim (semi-formal).** For a model class C, the revisits it can serve without a map are those whose loop is
tree-trivial within C's local reach (stack depth / age), or plane-trivial within its counting reach, or served by V, A or
W. A path-integrated phase with signed, wrap-commensurate steps serves every abelian loop in one layer at any gap
(Prop. 1 of draft 2). Its advantage over the best index model is therefore about P(revisit outside the index model's
routes) x (1 - floor within that set). The account makes no complexity claim: the general free-group word problem is
*not* easier than the abelian one (F_2 maps onto S_5, so its word problem is NC1-hard, while Z^D needs only counting);
what the data show is which structures *training* finds: bounded local cancellation, not a global count (Liu et al.'s
cyclic-group runs; Heo et al. 2026: transformers failed even C_2; Remark 1 of draft 2).

### 1.2 Evidence on the torus (post hoc; `q1_revisit_taxonomy.py`, `q1_cover_strata.py`)

FACTORIAL (s500-503) and FACTORIAL_N8 (s2800-2803) checkpoints, 20 evaluation walks per seed on the registered stream
(held-out map, np seed 1234 + s), T = 1024, eval mode. Accuracies reproduce the registered strata (e.g. L4 short / long /
wrap 0.925 / 0.433 / 0.434).

| cover | share | blank | RoPE 1L | RoPE 4L | RoPE loop x4 | path 1L |
|---|---|---|---|---|---|---|
| tree (95.6% of it retrace) | 0.703 | 0.513 | 0.827 | **0.987** | **0.994** | 1.000 |
| plane-only | 0.221 | 0.507 | 0.471 | 0.511 | 0.540 | 1.000 |
| torus-only | 0.076 | 0.518 | 0.480 | 0.434 | 0.501 | 0.992 |

By age (moves since the tree node was last visited) and kind:

| subset | n | RoPE 1L | RoPE 4L | loop | path |
|---|---|---|---|---|---|
| tree, retrace, age < 16 | 31,001 | 0.885 | 0.999 | 1.000 | 1.000 |
| tree, retrace, age 16-64 | 3,362 | **0.450** | 0.955 | 0.997 | 1.000 |
| tree, nested (not retrace), age < 16 | 254 | 0.504 | 0.953 | 0.980 | 1.000 |
| tree, nested, age 16-64 | 1,316 | 0.478 | **0.799** | **0.856** | 1.000 |
| plane-only, gap < 16 | 251 | 0.518 | 0.869 | 0.892 | 1.000 |
| plane-only, gap 16-64 | 3,466 | 0.480 | **0.627** | **0.617** | 1.000 |
| plane-only, gap 64-128 | 2,555 | 0.471 | 0.474 | 0.501 | 1.000 |
| plane-only, gap >= 128 | 5,019 | 0.46-0.50 | 0.42-0.47 | 0.48-0.53 | 0.996-1.000 |

Per-seed contrast (nested tree above its blank) minus (plane-only short-gap above its blank): RoPE 4L +0.228 and loop
+0.274, 8/8 seeds each; RoPE 1L -0.014 and path -0.014. Age-matched (16-64): 0.799 vs 0.627 (4L), 0.856 vs 0.617 (loop).

Reading: one index layer learns immediate retrace only (it fails retrace once the reversal is 16-64 moves old, 0.450);
depth extends retrace in age and to nested reversals (stack-like), and adds commutation only for loops closed within ~16
moves. Beyond that, index attention at 4 layers is at the blank level and the path layer at ceiling. This sharpens draft 2's
Fig. 4 ("decays with the direction runs spanned"): runs spanned is a proxy for whether the loop is tree-trivial.

**Sequence recognition is not what depth buys here.** An in-context longest-suffix-match predictor ("induction": copy what
followed the most recent earlier occurrence of the longest matching token window) scores 0.405 overall; combined with
retrace (retrace where it applies, else a match of >= 8 tokens) it reaches 0.851 against retrace's 0.840. On non-retrace
revisits where a >= 8-token same-cell match exists and predicts perfectly, RoPE 4L scores 0.574 (short gap) against 0.630
where it does not (over all non-retrace revisits, lower with a match on 7/8 seeds, d -0.045): trained index models do not use transition recall on this torus, where it would
pay +0.011. *For the paper:* the retrace floor is robust to adding sequence recognition (0.851 vs 0.840), a cheap answer to
"your floor ignores induction".

### 1.3 The account against the repo's other regimes

Model-free shares (`q1_regime_sweep.py`, 40 walks per regime, GridWorld's walk policy; rooms are a surrogate, not MiniWorld):

| regime | tree | plane-only | torus-only | need-map at gap >= 16 | predicted deep-index ceiling (tree + need-map x 0.5) |
|---|---|---|---|---|---|
| torus 64, T = 128 (MapFormer's) | 0.876 | 0.124 | 0.000 | 0.116 | 0.94 (retrace 0.934) |
| torus 64, T = 132 moves (text world) | 0.885 | 0.115 | 0.000 | 0.107 | 0.94 (retrace 0.934; "retrace cannot reach" 13.4%) |
| torus 64, T = 1024 | 0.711 | 0.209 | 0.080 | 0.285 | 0.86 (measured 4L 0.840, loop 0.856) |
| torus 64, T = 4096 | 0.472 | 0.203 | 0.325 | 0.524 | **0.74 (prediction)** |
| torus 32 / 16, T = 1024 | 0.489 / 0.289 | 0.144 / 0.085 | 0.367 / 0.626 | 0.51 / 0.71 | 0.75 / 0.65 (prediction) |
| 1D ring (CANCEL), Dyck-2 | all non-wrap revisits are tree revisits (in 1D, F_1 = Z) | 0 | ring: wrap | -- | depth catches up (REG: ~3 layers) |

What each regime of the repo shows, and whether the account predicts it:

| regime (repo result) | dominant revisit class | predicted winner | outcome |
|---|---|---|---|
| torus T = 128 (paper torus; retrace 0.934) | tree / retrace | index depth nearly suffices; path gain small | consistent (index 1L 0.805 is below retrace; MapFormer's Table 2 index arms at <= 4 layers fall short; we have no T = 128 4L arm) |
| torus T = 1024 (FACTORIAL, n = 8) | 30% plane + torus | path; depth plateaus at the tree share | **yes** (4L at retrace; post hoc strata above) |
| Dyck-2 matched depth; biased ring (CANCEL) | tree only (stack; 1D) | depth catches up (stack depth / count reach) | **yes** (~3 layers); ring p_plus 1.0: every revisit is a one-lap fixed offset -> 1 index layer (REG) |
| text world (T = 1024 words, ~132 moves) | like torus T = 132 | retrace-like floor 0.93; map evidence in the 11-13% plane stratum | **yes** (0.956 vs 0.411 on the retrace-unreachable revisits; analysis) |
| image torus | same covers as tokens | path (if the step reads actions) | yes (0.995 vs 0.701) |
| Match-Query blind continuation (0.730 vs 0.154) | observations withheld in the query phase: no retrace from recent views, long gaps | path | yes |
| map extent 32 / 128 / 512 cells (MiniWorld OneRoom, post hoc pooled) | small rooms: high re-traversal (V) | index wins small maps, path large | qualitatively yes: surrogate induction floor 0.86 / 0.79 / 0.56 / 0.47 at 18 / 32 / 128 / 512 occupied cells (`q1_room_induction.py`); the cover shares alone do NOT predict it (plane-only 0.34-0.49 in every room). Quantitatively untested; MiniWorld's dynamics differ |
| Memory Maze (heading given, blind runs of 16) | informative first-person views + short blind runs: anchoring (A) and re-traversal (V) | index depth competitive; path's edge grows with blind length | consistent: R4 0.389 vs P_E 0.331 (unmeasured), R1 0.118. **Prediction**: P_E - R4 grows with the blind run (E3) |
| fixed map (RNN_TORUS training map; MiniWorld fixed) | W | memorising models win on the training map, fail on transfer | yes (RNN own map 0.93-0.98, new 0.57-0.65) |
| k-back recency (RECENCY) | not a revisit task: a content count | a clock (monotone or signed) beats index | yes (index 0.234; monotone ~ signed) |
| rotation actions (KNOB_SWEEP, MM P_raw) | the state group is SE(2), not abelian | an abelian cumsum fails unless actions are recoded allocentrically / heading given | yes (+0.050 -> +0.488; MM P_raw 0.129 vs P_E 0.331) |
| stochastic transitions | commanded loops are no longer trivial | the map serves gaps up to ~1/(p x const), then needs re-localisation | post hoc below; MQ_NOISE (trained with noise): path 0.505 -> 0.215 at p = 0.10 |

**Stochastic transitions (post hoc, robustness: models trained at p = 0, rule 10; `q1_transition_noise.py`).** FACTORIAL_N8
path L1 / RoPE 4L, 4 seeds x 6 walks, accuracy by gap:

| p | path gap < 16 / 16-64 / 64-256 / >= 256 | RoPE 4L same bins |
|---|---|---|
| 0.00 | 1.000 / 1.000 / 1.000 / 0.994 | 0.997 / 0.743 / 0.409 / 0.400 |
| 0.01 | 0.938 / 0.824 / **0.430** / **0.184** | 0.976 / 0.705 / 0.416 / 0.380 |
| 0.03 | 0.863 / 0.621 / 0.199 / 0.133 | 0.928 / 0.638 / 0.397 / 0.424 |
| 0.10 | 0.629 / 0.347 / 0.127 / 0.126 | 0.773 / 0.535 / 0.423 / 0.413 |

A dead-reckoned map has a horizon of order 1/p moves and, past it, is *confidently wrong* (well below the blank level).
Where transitions are stochastic the map must be re-anchored by observations; the repo's two attempts (InEKF / Level 1.5;
TW_LANDMARK) are live negatives, so this is stated as a boundary, not a proposal (Sec. 3.5).

### 1.4 The resulting map of regimes (predictions)

| axis | map needed when | best alternative otherwise |
|---|---|---|
| revisit structure | loops that need commutation (plane-only) or the wrap, beyond the local reach of index depth (~16 moves in 2D at 4 layers; ~64 in 1D at 3) | retrace / nested retrace (index depth) |
| map extent vs T | large maps relative to T (edges rarely re-traversed) | small maps: transition recall (index >= 2 layers) |
| view informativeness | aliased views, or long intervals between unique views | rich views + short blind intervals: anchoring + local integration (index depth) |
| fixed vs redrawn | redrawn per episode (or transfer) | fixed: memorise in weights (any RNN) |
| action group | abelian (translations, or heading known) -> rotary path integration | non-abelian (egocentric without heading): needs an SE(2)-aware integrator (Sec. 3.4) |
| transition noise | gaps below ~1/p | beyond: observation-based localisation (no working design here) |
| what the readout keys on | place (map) | time or content count (clock; RECENCY) |

**Relation to theory.** Liu et al. (2023): bounded gridworlds are aperiodic (reset-like, Thm. 3, learned easily at 2
layers), cyclic groups need counting and learn unreliably even with dense labels; our torus is the group case, our small
rooms closer to the reset case. Merrill, Petty and Sabharwal (2024) and Grazzi / Sarrof: representational state-tracking
limits of SSMs; in the rotary slot those limits do not bind (Prop. 3), so our boundary is about what training finds. TEM
(Whittington et al. 2020) frames path integration as the structure that generalises across maps; CSCG (George et al.
2021) learns transition graphs from sequences: in our terms TEM is the plane/torus route and CSCG the V route, and the
regime table says when each is enough. Dedieu et al. (2024) and Fang & Rajan (2025) build maps in context with index
positions; the account predicts their success where re-traversal is dense (not checked against their settings).

---

## 2. When does Selective RoPE have MapFormer's properties?

### 2.1 Decomposition

| mechanism | MapFormer (MapWM) | Selective RoPE (faithful best-LM config, `model_srope.py`) | what our data say |
|---|---|---|---|
| who steps | every token's embedding through W_out W_in; observations learn ~0.02-0.03x (ActOnly / NormStep remove it) | every token, from the l2-normalised per-head input (scale-invariant, like NormStep) | effective observation step 0.34x (SiLU) / 0.14x (no SiLU) of a move (post hoc, below), 5-13x MapWM's 0.027; gate open on every class (0.94-0.99) |
| context dependence | none (context CV 0.00) | depthwise causal conv, kernel 4, over the projected input; per-position sigmoid gate | taps spread over lags (0.29 / 0.31 / 0.22 / 0.18): a move's step lands over 4 tokens; partial cancellation 0.75-0.80 (REG descriptive); counterfactual surgery (conv collapsed, gate class-averaged) breaks the co-adapted readout (accuracy 0.39-0.85) and does not isolate a cause |
| nonlinearity | none | SiLU after the conv (negative outputs bounded at -0.28) | NoSiLU vs SiLU: no difference shown (2/8 vs 0/8, p 0.47; REG secondary) |
| bias / sign | signed | no bias (LM config): signed; state-tracking config has +1 rad exp-bias: every learned step non-negative | ST config 0.769, 0/8, fails like |.| (REG descriptive): the sign result applies |
| per-head rank | shared rank 4 (D + 1 = 3 per head needed) | full rank per head (64 -> 32) | rank is not the obstacle |
| frequency ladder | omega learnable, init 2 pi x 64^(-i/31) (grid-matched: lowest = one turn per 64 cells) | temperatures fixed (temp_grad false), 500000^(-c/32) (an LM ladder: 1 down to 3e-6) | wrap-commensurability fails (below) |
| step scale at init | per-move angle median 0.086 rad; 62% of channels above pi/64 | **median ~0, p90 1e-4 rad** (conv taps N(0, 0.02), unit-norm input, gate 0.5) | after 900 epochs 7-9% of channels above pi/64; all runs DESCENDING (REG) |
| rotation vs decay | rotation only | rotation only here (their paper argues decay is needed for recall) | not a factor in this comparison |
| placement | once, from embeddings | per layer from the normed residual (equal at 1 layer) | -- |

### 2.2 What the checkpoints identify (post hoc; `q2_srope_decomp.py`, `q2_srope_wrap.py`, `q2_init_scale_out.txt`, `q2_conv_taps_out.txt`)

1. **Where it fails: the wrap.** Same-place phase coherence C0 (usage-weighted cos of the query-key phase difference at
   revisits), 6 walks x 8 seeds:

   | arm | short / long / wrap-only C0 | accuracy short / long / wrap-only |
   |---|---|---|
   | MapWM | 1.000 / 0.995 / 0.905 | 1.000 / 1.000 / 1.000 |
   | SRoPEGen (our earlier swap) | 0.962 / 0.906 / 0.590 | 0.998 / 0.977 / 0.857 |
   | SRoPE no SiLU | 0.971 / 0.931 / 0.673 | 0.984 / 0.949 / 0.834 |
   | SRoPE faithful | 0.980 / 0.887 / **0.292** | 0.991 / 0.932 / **0.602** |

2. **Why the wrap: angles not commensurate with the torus, not a clock.** From mean angle per move: usage-weighted
   cos(64 m) (what a wrap-only revisit along one axis would read) is 0.564 for SRoPE, 0.653 no-SiLU, 0.590 SRoPEGen, against
   0.928 for MapWM; per seed it tracks wrap-only accuracy (r +0.64 / +0.31 / +0.44; MapWM at ceiling). The out-and-back
   coherence cos(8 k) is 0.98-0.99 for every arm: Selective RoPE's opposite moves do cancel. It has learned a *plane* map
   (tree and plane revisits largely fine) without the torus quotient. Only 5-13% of its usage sits on channels whose
   fixed temperature is below 0.01, so the LM ladder's deep channels are not the issue; the scale and fine-tuning of the
   per-move angle are.
3. **Why so slowly: it starts without a phase.** Faithful Selective RoPE's initial per-move angle is ~1e-4 rad (90th
   percentile), three orders below MapWM's; it must grow its conv taps by ~50x under weight decay 0.05 (rule 15: a
   parameterisation is an optimiser choice). This is the most parsimonious reading of "every SRoPE run still descending".
   SRoPEGen (weight-norm projection with a per-channel gain, embeddings not l2-normalised, single learnable temperature)
   starts at MapWM's scale (median 0.052) and solves 5/8.
4. **The registered "observations step 0.86-0.96 as far as a move" should be restated.** The readout (first difference of
   theta at the token) charges the observation with the previous move's delayed conv contribution. Measured by removing
   one token's projection (gates untouched) and summing the phase change over the window: 0.34x (SRoPE), 0.14x (no SiLU).
   **Suggested correction** for draft 2 Sec. 7.3 and `SROPE_TORUS_RESULTS.md` (I have not edited either): "an observation
   moves the phase about a third (no SiLU: a seventh) as far as a move, against 0.027 for MapWM; the first-difference
   readout (0.86-0.96) includes the convolution's lag".

Not identified: SiLU (no difference), the conv vs the gate separately (surgery breaks a co-adapted model; needs training).

### 2.3 Minimal tweaks, with predictions (all one-line changes inside Selective RoPE's own option space)

| tweak | change | predicted outcome on the T = 1024 torus | rationale |
|---|---|---|---|
| T1 map-scale init | conv lag-0 tap initialised so the median per-move angle is ~0.1 rad (others 0) | solves faster; >= 4/8 within 900 epochs; wrap still partly off | removes the start-from-zero handicap |
| T2 learnable temperature | `temp_grad = True` (their flag), per channel | modest alone (scale still ~0 at init) | lets each channel tune its frequency to the quotient |
| T1 + T2 | both | **>= 6/8 SOLVED, wrap-only >= 0.95, cos(64 m) > 0.9** | scale + commensurability, the two identified deficits |
| T3 no conv (kernel 1) | on top of T1 + T2 | small further gain (cancellation 0.75 -> ~1.0); conv not needed when actions are given | removes context dependence and the lagged step |
| T4 token-class gate | gate reads only the token type (move / observation), init open | small (effective observation step already 0.14-0.34x) | ActOnly-like |
| ST config with signed bias | bias without exp, init 0 | rescues ST from 0/8 | the sign result |

### 2.4 When Selective RoPE *should* beat MapFormer

Its two departures are context dependence (conv) and per-layer placement. Both are useless when the displacement is
written in one token (an action) and essential when it must be *inferred from a change between inputs*: visual odometry,
where no action is given and consecutive views shift. A per-token step cannot read a shift from one view; a kernel-2 conv
over view embeddings can (Selective RoPE's own derivation is a first difference of queries). This predicts a crossover
(design D5 below): on an action-free local-view torus, conv-step models learn the map and per-token MapWM cannot.

---

## 3. Superior designs

Each: parameterisation; why; repo evidence for / against; cheapest decisive experiment; risk. GPU-h are for two RTX 4090s
from the measured costs: one-layer torus run, T = 1024, 900 epochs ~0.33 GPU-h (SROPE_TORUS: 48 runs, 15.9 h on one GPU at
4 concurrent); 4-layer ~0.5 GPU-h; LSTM/GRU ~0.64 GPU-h; Memory Maze run ~0.55 A40-h.

### 3.1 D1. Map heads + clock heads

- *Parameterisation.* Head 1: MapWM actions-only signed step, per-head rank D + 1; head 2: index RoPE (or a monotone
  content clock). Concatenated, shared FFN.
- *Why.* Place-keyed retrieval needs clock-free channels (Prop. 2b); recency and local order need a clock. The text world
  already grows a clock beside the map in unused channels (4/8 seeds, REG secondary); on code, path integration costs
  +0.0056 bpc (n = 3); on TinyStories MapWM = RoPE.
- *Against.* MapWM's class contains index RoPE (a constant step on every token is RoPE), and on the torus extra clock
  routes did not help rank 2 (RANKCLOCK slots 3/8). A gain needs a task that rewards both at once.
- *Experiment.* Mixed torus: the T = 1024 walk plus interleaved queries "the observation k tokens ago" (k 1-32). Arms
  MapWM, RoPE 1L, hybrid 1+1 heads (equal params), n = 8, 900 epochs: 24 runs, ~8 GPU-h. Registered: hybrid vs best
  single arm on the per-query-type minimum accuracy. Branch: hybrid >= both on both query types -> "heads specialise";
  MapWM = hybrid -> "the path phase already hosts the clock" (also informative).
- *Risk.* Low information if MapWM learns both (likely on this task). Follow-up paper.

### 3.2 D2. Learned gating between path and index phase

theta = g_h path + (1 - g_h) omega t per head (or per token). Subsumed by D1 (a per-head gate that converges to 0/1 is
D1); a per-token mixture puts a clock into map channels, which is the failure mode of Sec. 7.1. **Not recommended**
beyond D1.

### 3.3 D3. A learned self-motion gate (instead of ActOnly)

- *Parameterisation.* step = sigma(u . LN(x) + b) . W_out W_in LN(x), b initialised at -2 (mostly closed), NormStep input.
- *Why.* ActOnly needs the action labels; NormStep removes the code-norm leak; neither handles an input whose relevance to
  self-motion must be learned (images, words). Selective RoPE's l2-normalised input is NormStep in another form.
- *For.* IMAGE_TORUS P2: an image-driven step at token scale collapses 5/8, at gain 1 learns 8/8 (pilot-informed); the
  text world's learned steps beat "only direction words move" (asides need an offset). *Against.* Selective RoPE's gate
  stayed open on every token class (0.94-0.99): gates are not learned closed unless something pushes them.
- *Experiment.* IMAGE_TORUS with the image in the step: MapWM token scale (collapse anchor), NormStep, gated NormStep,
  ActOnly; n = 8: 32 runs, ~12 GPU-h (this overlaps the optional "block B"). Registered: gated NormStep vs MapWM on
  SOLVED. Prediction: gated and NormStep 8/8.
- *Risk.* Design guidance only; low novelty (Mamba's selection, CoPE's gate).

### 3.4 D4. SE(2) two-level integrator for egocentric actions without heading

- *Parameterisation.* A heading phase phi_t = cumsum(alpha . turn steps) (signed, learned turn quantum); a displacement
  step d_t = R(phi_t) v . [a_t = forward]; position phases theta_c,t = k_c . cumsum(d)_t with learned wave vectors k_c.
  Two prefix sums (heading, then rotated displacement): the semidirect product R^2 x| SO(2), still parallel. Reduces to
  MapWM when no turn occurs.
- *Why.* Turn-and-forward makes the state group non-abelian; a single cumsum cannot integrate it (KNOB_SWEEP rotate +0.050;
  allocentric recoding +0.488; Memory Maze P_raw 0.129 vs P_E 0.331 with heading given). This is the minimal
  non-commutative extension that the navigation group needs, not a general product of matrices.
- *Against / caution.* "Non-commutativity +0.005-0.014 for 34x cost" (FAMILY_TREE) is a live negative, on a different task
  with a general non-commuting product; D4 costs two scans and is tied to SE(2). Learning the turn quantum may inherit the
  sign/start issues of Sec. 7.3 (a 90-degree turn must be learned exactly).
- *Experiment.* Torus with `action_mode=rotate` (4 headings), T = 1024, 900 epochs: MapWM-raw, MapWM-allocentric (oracle
  ceiling), SE2-raw, RoPE 1L, RoPE 4L; n = 8: 40 runs, ~16 GPU-h (+ ~1 day implementation, results-blind code check).
  Registered: SE2-raw vs MapWM-raw (SOLVED and accuracy); non-inferiority of SE2-raw to the allocentric oracle at margin
  0.02. Prediction: SE2-raw >= 0.95, MapWM-raw ~0.56, RoPE at floor. Second stage: Memory Maze without heading (P_raw vs
  P_SE2 vs P_E), 24 runs, ~13 A40-h; prediction P_SE2 within 0.05 of P_E.
- *Novelty.* GTA (geometric transform attention) and LieRE use *given* poses; PaTH is a general data-dependent
  non-abelian product; MapFormer has no egocentric variant. A learned egocentric SE(2) path phase in rotary attention
  appears new (search Sec. 6; not exhaustive). Follow-up paper, or a strong section of a second paper.

### 3.5 D5. Self-motion from views: a conv-driven step

- *Parameterisation.* step_t = W_out W_in [conv_2(e(o_{t-1}), e(o_t))], signed, rank D + 1; no action tokens.
- *Task.* Torus where each observation is the 3 x 3 neighbourhood of objects (aliased), actions withheld: displacement is
  visible only as a shift between consecutive views.
- *Prediction.* MapWM (per-token step) cannot integrate; conv-step and Selective RoPE with T1 + T2 can; index RoPE at
  depth learns local shifts but not long-gap / wrap revisits. This is the regime where Selective RoPE's design is the
  right one, the counterpart of Sec. 2's failure.
- *Experiment.* 5 arms (MapWM, conv-step, SRoPE-T1T2, RoPE 1L, RoPE 4L) x 8 seeds, ~45 runs, ~16 GPU-h plus a new
  environment and its gate (n-gram on views, a "shift-decoder" floor). Follow-up.
- *Risk.* The step must be learned from content, which is exactly where the leak and the pixel collapse live
  (IMAGE_TORUS P2); likely needs NormStep-style inputs.

### 3.6 Landmark correction: not proposed now

Two attempts are live negatives (InEKF / Level 1.5: no component load-bearing, benefit does not grow with drift;
TW_LANDMARK: a cumsum phase cannot be corrected by a name). The premise exists (Sec. 1.3: at p = 0.01 the noise-free path
model is confidently wrong beyond ~64 moves), but any design must leave the parallel prefix sum: a phase reset by retrieval
is sequential. If revisited, the cheapest coherent form is *positional values* (values carry cos/sin of theta, so a second
layer can read "where I was when I saw this landmark" and correct the query in the residual), gated first on a CPU premise
check that trained-with-noise path models fail at long gaps but succeed when a unique landmark was seen recently. I do not
rank it.

### 3.7 Language

The account predicts little for ordinary text: its revisits are not loops in a group, the used channels are clocks (9.9-12.3
on TinyStories), and HGRN's Table 11 and our code result point the same way. Where language *describes* navigation (the
text world), the cover account carries over unchanged (Sec. 1.3). The language test worth running is the text world at
T where the plane/torus share is large (the 4096-word walk G-7 in the repo's list); the cover table predicts the deep index
ceiling falls from 0.94 (132 moves) to ~0.86 (1024 moves), so a 1024-move text walk would separate path from 2-4 index
layers by ~0.14. Expensive (20-35 h) and incremental.

---

## 4. Ranked plan

Ordering by value to the current paper per GPU-hour; every batch follows the house method (pre-registration with branches
against measured floors, n = 8 fresh seeds, pilots outside the batch, a results-blind code check before launch).

| rank | item | serves | cost | registered contrast and branches | predicted |
|---|---|---|---|---|---|
| 0 | **Fold in the CPU results** (no GPU): (a) the cover strata as the mechanism of "what depth buys" (Sec. 6 of draft 2; replaces "runs spanned"); (b) the retrace + induction floor (0.851 vs 0.840) as a robustness line for the floor; (c) correct the SRoPE observation-step sentence (0.14-0.34x, conv lag); (d) the SRoPE wrap / init-scale finding as the reason "SRoPE fails" (labelled post hoc) | main paper Secs. 5-7 | ~1 day writing; re-run the scripts once read-only | -- (post hoc, labelled) | -- |
| 1 | **E1 Selective RoPE minimal tweaks** | Sec. 7.3 "published content phases"; design guidance; turns a budget-scoped negative into a mechanism | 6 arms (SRoPE faithful anchor; T1; T2; T1 + T2; T1 + T2 no conv; MapWM anchor) x 8 = 48 runs, **~16 GPU-h (~8 h wall)** | primary: T1 + T2 vs SRoPE, Fisher on SOLVED and permutation on accuracy; secondaries T1, T2 alone; no-conv vs T1 + T2; wrap-only accuracy and cos(64 m). Branches: T1 + T2 >= 6/8 -> "init scale and frequency learnability explain the failure"; T1 alone >= 6/8 -> scale; only no-conv solves -> context dependence; all fail -> observation steps / gate (then T4) | T1 + T2 6-8/8, wrap >= 0.95 |
| 2 | **E2 commutation dial** (makes Sec. 1 causal) | contribution 2 ("what depth buys") and the answer to "when is a map needed" | torus 64, T = 256, two walk policies at matched gap distribution: *tree-closing* (loops close by nested reversal) vs *plane-closing* (loops close only by commutation, closing gaps >= 16); arms path 1L, RoPE 1L, RoPE 4L; n = 8: 48 short runs, **~6-8 GPU-h** + CPU gate | primary: interaction (path - RoPE4L)[plane] - (path - RoPE4L)[tree] > 0 (permutation over seeds); branch "MAP NEEDED EXACTLY FOR COMMUTATION" if RoPE 4L is within 0.02 of path on tree and >= 0.15 below on plane; "DEPTH COMMUTES" if within 0.05 on both | 4L ~1.0 on tree, ~blank on plane |
| 3 | **E3 Memory Maze blind-run dial** | the Memory Maze row (S1 unmeasured) and the anchoring route | blind runs of 64 (vs registered 16), arms P_E, R4, R1, P_odo; n = 8: 32 runs, **~18 A40-h**; scored set and floors re-gated (rule 11) | primary: (P_E - R4)[B = 64] - (P_E - R4)[B = 16] > 0, with the B = 16 cells re-run in the same batch (rule 12: +32 runs, ~18 A40-h, total ~36 if done properly) | R4 loses more than P_E at B = 64 |
| 4 | **E4 many-map RNN** (fixed vs redrawn) | RNN_TORUS's scope gap (one map per run) | LSTM / GRU matched size trained on a fresh map per batch; n = 8; with path anchor: 24 runs, ~12 GPU-h | primary: path vs best RNN on held-out maps; secondary: RNN tree vs plane strata | RNN learns the tree part at best (<= retrace), path wins |
| 5 | D4 SE(2) integrator | follow-up paper (egocentric) | ~16 GPU-h + ~13 A40-h + implementation | Sec. 3.4 | SE2-raw >= 0.95 |
| 6 | D5 visual odometry torus | follow-up (where SRoPE's conv should win) | ~16 GPU-h + environment | Sec. 3.5 | crossover |
| 7 | room-size dial (V route) | follow-up (map-extent threshold mechanism) | bounded rooms 8 / 16 / 32, fresh maps, T = 512, RoPE 2L / 4L vs path; ~48 runs, ~10 GPU-h | index - path vs the model-free induction floor | index wins small rooms |
| 8 | D1 map + clock heads; D3 gated step | follow-up / design notes | ~8 and ~12 GPU-h | Secs. 3.1, 3.3 | -- |

Stopping point for the current paper: items 0-2 (about 24 GPU-h and two days of writing). Item 3 only if a reviewer
presses the Memory Maze depth comparison; it then needs its own B = 16 re-run (36 A40-h in total).

Design risks to settle before registering:
- E1: a large init may destabilise Selective RoPE (their paper reports instability at higher rates); pilot two outside
  seeds and pre-register the init value. Keep weight decay on the conv as in SROPE_TORUS (do not change two things).
- E2: a plane-closing policy with fixed rectangles is learnable as a fixed offset; draw rectangle sides and orientations
  at random, compose several before closing, and gate with retrace / induction / n-gram floors (CPU) so both policies have
  the same map-free floor.
- E3: at B = 64 the one-layer readout (given odometry 0.54) may floor every arm; P_odo is the monitor (branch: if P_odo <
  0.3 the batch is unreadable).

---

## 5. Novelty vs 2025-2026 work, and where each expansion goes

| expansion | nearest prior / concurrent work | novelty | goes into |
|---|---|---|---|
| cover-space account of "when a map is needed" (tree / plane / torus + V, A, W) | Liu et al. 2023 (aperiodic vs group automata; learnability of cyclic groups); Heo et al. 2026 (depth, Lie-algebraic; transformers fail C_2); Merrill et al. 2024; TEM; CSCG; Dedieu 2024 / Fang & Rajan 2025 (in-context maps with index positions) | the revisit-level decomposition and its measurement on trained models appear new; the ingredients are classical | **current paper** (Sec. 6 mechanism, zero GPU; E2 makes it causal) |
| induction floor (0.851 vs retrace 0.840) | -- | small robustness check | current paper, one line |
| Selective RoPE: lagged-step artefact, zero-scale init, plane map without the torus quotient | Selective RoPE (2511.17388) tests LM and state tracking, not navigation; Wu, Liu & Jadbabaie 2026 (2607.07678: RoPE frequency scale matched to dependency width; abstract read); LeRoPE 2026 (2607.10134, learnable RoPE frequencies; search snippet only) | new, modest; converts a negative into design guidance | **current paper** (post hoc now, E1 registered) |
| stochastic-transition horizon of a dead-reckoned map | classical in robotics / neuroscience (boundary error correction, Hardcastle et al. 2015); MQ_NOISE in repo | not new as an idea; the measurement is cheap context | current paper Limitations, one sentence |
| SE(2) two-level integrator | GTA (given poses), LieRE (given positions), PaTH (general data-dependent products), FAMILY_TREE negative | likely new for learned egocentric rotary phases | follow-up |
| conv-driven step / visual odometry | VO transformers regress poses (e.g. VoT 2510.03348, search snippet); Selective RoPE's derivation | the crossover prediction is new | follow-up |
| map + clock heads | hybrids with partial RoPE / NoPE heads are common in LMs; Mamba-3 | low | follow-up note |

Concurrent rank work flagged by FULL_REVIEW (Larson 2609.12259 / 2609.17594; Sarkar et al. 2610.34945) is unaffected by
these expansions.

---

## 6. References

Read first-hand in this session or in the repo's corpus (`papers/txt/`, `docs/lit/`) unless marked [abstract] (abstract
page only, this session) or [snippet] (search result only; must be read before citing).

- Rambaud, Mascarenhas, Lakretz. MapFormer. arXiv:2511.19279 (v4).
- Movahedi et al. Selective rotary position embedding. ICLR 2026, arXiv:2511.17388 (method via `docs/lit/LIT_CONTEXT_STEPS.md`
  and the authors' code as reproduced in `model_srope.py`).
- Zhang et al. GRAPE. arXiv:2512.07805. Lahoti et al. Mamba-3. arXiv:2603.15569. Yang et al. PaTH. arXiv:2505.16381.
- Golovneva et al. CoPE. arXiv:2405.18719. Veisi et al. CARoPE. arXiv:2507.23083. Gopalakrishnan et al. PoPE. arXiv:2509.10534.
- Liu, Ash, Goel, Krishnamurthy, Zhang. Transformers learn shortcuts to automata. ICLR 2023, arXiv:2210.10749 (Thm. 3,
  Lemmas 6/8, Sec. 5 via `docs/lit/LIT_REPRESENTABILITY_2026-10-14.md`).
- Grazzi et al. Unlocking state-tracking in linear RNNs through negative eigenvalues. ICLR 2025, arXiv:2411.12537.
  Sarrof, Veitsman, Hahn. The expressive capacity of SSMs. NeurIPS 2024, arXiv:2405.17394.
- Merrill, Petty, Sabharwal. The illusion of state in state-space models. ICML 2024, arXiv:2404.08819 [not re-read today].
- Heo, Ngotiaoco, Irie, Gershman, Sabatini. Why depth matters in parallelizable sequence models: a Lie algebraic view.
  arXiv:2603.05573 [abstract-level via search; cited in draft 2].
- Shakerinava et al. arXiv:2603.01959; Howe arXiv:2609.18966; Larson arXiv:2609.12259, 2609.17594; Sarkar et al.
  arXiv:2610.34945 (via FULL_REVIEW).
- Whittington et al. TEM, Cell 183:1249 (2020); TEM-t, ICLR 2022, arXiv:2112.04035. George et al. Clone-structured graph
  representations enable flexible learning and vicarious evaluation of cognitive maps. Nat. Commun. 12:2392 (2021)
  [not re-read today].
- Dedieu et al. Learning cognitive maps from transformer representations for efficient planning in partially observed
  environments. ICML 2024, arXiv:2401.05946 [abstract]. Fang & Rajan. From memories to maps: mechanisms of in-context
  reinforcement learning in transformers. arXiv:2506.19686 [abstract].
- Pasukonis, Lillicrap, Hafner. Memory Maze. arXiv:2210.13383.
- Wu, Liu, Jadbabaie. How data shapes RoPE frequency usage. arXiv:2607.07678 [abstract]. LeRoPE: learnable RoPE frequencies.
  arXiv:2607.10134 [snippet]. VoT, arXiv:2510.03348 [snippet].
- Hardcastle, Ganguli, Giocomo. Environmental boundaries as an error correction mechanism for grid cells. Neuron 86:827
  (2015) [from memory; verify].
- Word problem of free groups: Lipton & Zalcstein, J. ACM 24 (1977), logspace; NC1-hardness via the surjection F_2 -> S_5
  (Barrington 1989) [from memory; verify before citing].
- HGRN, arXiv:2311.04823 (Table 11) via `papers/INDEX.md`.

## Appendix: scripts and outputs (`docs/research/scripts_opus/`)

| script | what | output |
|---|---|---|
| `q1_revisit_taxonomy.py` | per-revisit labels (stratum, retrace, induction) and four FACTORIAL arms' correctness, 8 seeds x 20 walks | `out/q1_revisit_taxonomy_out.txt`, `out/q1_revisit_rows.json.gz` |
| `q1_cover_strata.py` | tree / plane / torus classes joined to the rows | `out/q1_cover_strata_out.txt` |
| `q1_regime_sweep.py` | model-free cover shares across tori and bounded-room surrogates | `out/q1_regime_sweep_out.txt` |
| `q1_room_induction.py` | model-free floors (retrace, induction, combined) in rooms vs the torus | `out/q1_room_induction_out.txt` |
| `q1_transition_noise.py` | eval-only robustness to transition noise, path 1L vs RoPE 4L | `out/q1_transition_noise_out.txt` |
| `q2_srope_decomp.py` | SRoPE phase coherence by stratum, counterfactual phases, effective observation step | `out/q2_srope_decomp_out.txt` |
| `q2_srope_wrap.py` | torus commensurability cos(64 m), out-and-back cos(8 k), vs wrap accuracy | `out/q2_srope_wrap_out.txt` |
| (inline, logged) | init vs trained per-move angle scale; conv taps and gates | `out/q2_init_scale_out.txt`, `out/q2_conv_taps_out.txt` |

Caveats on all of these: post hoc, one batch per arm, thresholds and bins chosen today, eval mode (the attention-dropout
rescale matters only for one-layer path models below ceiling, which these are not), 6-20 walks per seed; the room results
are a surrogate with GridWorld's policy, not MiniWorld; the counterfactual phases in `q2_srope_decomp.py` alter a
co-adapted model and are reported only as coherence, not as causes.
