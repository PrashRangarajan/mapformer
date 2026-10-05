# Synthesis: how a path-integrating transformer builds a cognitive map (2026-10-04)

Theory note, one of five. Read-only over the repo except this file. Every number is from the file named beside it;
CORRECTED / AUDIT blocks were respected; nothing withdrawn in CLAUDE.md is used. New measurements in this note are
**POST HOC** (eval-only CPU on committed checkpoints, thresholds chosen after seeing the data); their scripts are in
the appendix. Status words: REG (pre-registered, 8 seeds), REG-sec (a declared secondary of a registered batch),
NOT-PRE (older batch, no pre-registration), POST HOC, PILOT, CONJ (conjecture: no measurement behind it).

Terms. **step** Delta(tok): the per-token increment the model cumsums; here always omega-scaled, per head h and
frequency block i. **axis** u_d = (Delta(+d) - Delta(-d))/2, the step along world dimension d. **drift** m: the
per-move common step, mean action step + expected observation step (blank p_empty, objects 1 - p_empty), wrapped to
(-pi, pi] per block. A non-zero drift is a **clock**: phase that grows with elapsed moves, not with displacement.
**channel weight** w_hi = mean over action queries |q_pair| x mean over observation keys |k_pair| (exact at layer 1,
where Q and K are functions of the token). **kappa_h** = ||m_h||_w / mean_d ||u_d,h||_w. **indep_h** = s_min/s_max of
the w-weighted D x n_b axis matrix (0 = axes collinear). A head is **clean** if kappa <= 0.01 and indep >= 0.2.
A run is CLEAN (some head clean), COLLAPSE (some head drift-free but every drift-free head has indep < 0.2) or CLOCK
(no drift-free head). SOLVED = final-5% training loss < 0.05, the project's rule.

## 0. The answer in one paragraph

A one-layer path-integrating transformer has a map exactly when **at least one head's step code has zero per-move
drift and D independent axes**; the code tells which runs solved in 79/80 rank-line runs and 94/96 with the 1800-epoch
runs (POST HOC, section 1). Training ends in one of three places: CLEAN (a map), CLOCK (drift left in), COLLAPSE (drift
removed, one axis lost). Elementary linear algebra (the **aliasing lemma**, section 1.2) shows why rank = D is the hard
case: with exactly D latent directions per head, any drift is a combination of the axes, so the drift-free channels
can only read a (D-1)-dimensional projection of position. A clock and a full map cannot share a rank-D head; with one
spare direction they can, and in the text world they do (3/8 seeds: a clock outside the axes' span, parked in channels
carrying 3-11% of the attention weight, a full 2-D map in the rest). On this reading sign, rank, leak and clock-vs-map are four views of one
quantity, the per-move drift. Sign forces drift; rank D ties it to the axes; leak adds an object-dependent part; the
task decides whether drift is wanted (recency) or fatal (navigation). Depth is a second, separate route: multi-layer
rank-2 models solve **without** a clean head (7/7 solved runs), which fits one path layer being worth about three
attention layers. Around this, which solution training ends up with (relational map, in-weights memorisation, local lookup) depends both on what
else is cheap in the data and on how findable the map is.

## 1. New post hoc measurement: the code criterion and the aliasing lemma

### 1.1 Code basin vs SOLVED (POST HOC; `basins.py`, appendix A)

Weighted, drift defined from the per-move common step. Checkpoints: `runs/rank_mi/p0`, `runs/rank_sep/p0`,
`runs/rank3/p0`, `runs/rank_nd/D{2,3}`, `runs/loop_rank_e1800/p0`, `runs/loop_rank/p0`, `runs/sign_matched/p0`.

| set (all T=1024, 900 ep unless noted) | SOLVED | CLEAN | CLOCK | COLLAPSE | solved but not CLEAN |
|---|---|---|---|---|---|
| torus, shared r=2 (`Vanilla`) | 0/8 | 0 | 8 | 0 | 0 |
| torus, per-head r=2 (`Vanilla_r2ph`) | 2/8 | 2 (both solved) | 4 | 2 | 0 |
| torus, block-diag r=4 (2 per head) | 2/8 | 2 (both solved) | 2 | 4 | 0 |
| torus, per-head r=3 | 6/8 | 6 (all solved) | 2 | 0 | 0 |
| torus, shared r=4 / per-head r=4 | 16/16 | 16 | 0 | 0 | 0 |
| N-D 2D grid 32: rank 2 / rank 3 | 1/8 / 8/8 | 1 / 8 | 7 / 0 | 0 | 0 |
| N-D 3D grid 10: rank 3 / rank 4 | 1/8 / 4/8 | 2 / 4 | 6 / 3 | 0 / 1 | 0 |
| torus 1800 ep: shared r=2 / shared r=4 | 0/8 / 7/8 | 0 / 8 | 6 / 0 | 2 / 0 | 0 |
| **1-layer total, 96 runs** | 47 | 49 (47 solved) | 38 | 9 | **0** |
| sign, T=1024: Signed / Abs / Pos (r=4) | 8 / 0 / 1 | 8 / 0 / 0 | 0 / 8 / 8 | 0 | **1** (Pos s7, final loss 0.049) |
| **depth aids at shared r=2** (`runs/loop_rank`): loop x4 / 4 real layers | 2/8 / 5/8 | 0 / 0 | 4 / 6 | 4 / 2 | **2 / 5** |

- Necessity at one layer: 0 of 47 solved rank-line runs lack a clean head; solved runs' best-head kappa <= 0.0008,
  indep >= 0.388. Not sufficient: 2 CLEAN runs did not solve (3D grid 10 rank 3 s7, loss 0.376; 1800-ep r=4 s5,
  STALLED at 0.180). Unweighted channels give the same split (40 CLEAN/S, 1 CLEAN/U, 39 non-clean/U on the first 80).
- A per-axis retrace definition of drift (wrap(Delta(+d) + Delta(-d) + 2 E[obs step])), which allows monotone codes
  to wind mod 2 pi, gives the same picture over 120 one-layer runs: 55 CLEAN/S, 4 CLEAN/U, 1 CLOCK/S (Pos s7), 60
  non-clean/U.
- The registered geometry fits this. The unsolved rank-3 seeds had opposition 1.83 / 1.58 (`RANK3_RESULTS.md`), and here they are CLOCK. The r=2
  stalls at "opposition 0.87-1.78" (`RANK_MATCHED_RESULTS.md`) are CLOCK; the "cancelling but |cos(N,E)| 0.99" r=2
  seeds (`RANK_MATCHED_GEOMETRY.md`) are the COLLAPSE basin.
- **Depth breaks necessity.** Every solved multi-layer rank-2 run (7/7) is CLOCK or COLLAPSE. Extra attention layers
  get around an aliased code. (Weights there use the first block's Q/K: approximate.)

### 1.2 The aliasing lemma (algebra; applies to the trained models by construction)

Per head, step = omega (.) W_out W_in x, latent rank r. Channel i reads position through the covector
p_i = (omega_i W_out,i . u_d)_d (phase per unit displacement) and drifts by delta_i = omega_i W_out,i . m_lat.
If r = D and the latent axes {u_d} are a basis, then m_lat = sum_d c_d u_d, so **delta_i = c . p_i**. A channel is
drift-free (mod 2 pi aside) only if p_i is orthogonal to c, so **all drift-free channels read the same
(D-1)-dimensional projection of position**: displacements along c are invisible to them. With r >= D + 1, m_lat can
have a part m_perp outside span(u). Then delta_i = c . p_i + omega_i W_out,i . m_perp, and that can be zeroed with
p_i unconstrained. So:
- rank D: clock + full map in one head is impossible. Every trained state is CLOCK (drift in the map channels),
  COLLAPSE (drift-free channels, map projected to D-1 dims) or exactly drift-free. These are the three basins seen.
- rank D + 1: a clock can coexist with a full drift-free map if it is placed off the axes' span and its channels
  are down-weighted.
- monotone steps (Abs, Pos): every channel with a non-zero action step drifts, except through winding.

The lemma is a fact about states. It is NOT a theorem about training dynamics. That rank D is hard *because*
the clock + map intermediate is closed to it is **CONJ** (testable, section 3).

### 1.3 The text world shows the D + 1 route (POST HOC; `tw_latent.py`, `tw_weights.py`, appendix A)

Text world, shared r=4, D=2 (`runs/textworld/p0`). Clock latent m_lat = mean direction-word latent + mean verb latent.

| seed | type (`TEXTWORLD_RESULTS.md`) | ||m_lat|| / ||u_lat|| | share of m_lat outside span(u,v) | drifting blocks | weight on drifting blocks | weight on clean blocks; their 2-D independence |
|---|---|---|---|---|---|---|
| s1, s5, s6, s7 | map | 0.07-0.24 | (irrelevant: m ~ 0) | 2-4/64 | 0.011-0.059 | 0.93-1.00; 0.67-0.87 |
| s2, s3, s4 | clock, solved | 1.37-1.71 | **0.65-0.96** | 31-38/64 | **0.034-0.105** | 0.88-0.91; **0.54-0.88** |
| s0 | clock, the **one unsolved seed** (still descending) | 4.23 | **0.007** | 31/64 | 0.178 | 0.74; **0.002** |

The three solved clock seeds put the clock outside the axes' span and give 3-11% of attention weight to the
drifting half of the channels (an even split would give about half). Their clean channels carry a full 2-D map. The one seed whose clock lies
inside the axes' span (s0) has a collapsed clean map and is the batch's only unsolved path seed: the lemma's
rank-D horn, inside a rank-4 model whose clock happens to lie in the 2-D axis span. n=8, post hoc. It was not predicted
before the data were read, but it was derived before this table was computed.

## 2. The principles, tested

### P1. Finding the map is search, not capacity. ACCEPT, sharpened

Statement (sharpened): every arm tested here can represent a map. Whether training reaches one is decided by
which code basin it enters. The search target is concrete: a head with zero per-move drift and D independent axes.
- For: rank-2 projection of solved r=4 scores 0.9955 frozen and is held under training 7/8 (`RANK_PROJ_RESULTS.md`,
  REG). 2x budget solves 0/8 rank-2 runs (`LOOP_RANK_E1800_P1_RESULTS.md`, REG; 5/8 still descending). In this note, the
  1800-ep rank-2 runs are all CLOCK/COLLAPSE. EM's recency deficit: the installed rewind holds at 1.000 but is found
  per token for about half the k (`SEARCH_RESULTS.md`, REG). The leak slows convergence: remedies 16/16 SOLVED vs MapWM 8/8
  descending, r(loss, acc) -0.944 (`LEAK_RESULTS.md`, REG-sec). Recipe: LinearLR from step one cannot leave a
  plateau (rule 3).
- Against / limits: "search" was so far defined by exclusion (exists, held, not found). The basin criterion is a
  positive definition, but post hoc. Two CLEAN runs did not solve, so the code is not all of the search.
- Sharpest test: pre-register the criterion. In the next rank batch (H1 part 2, ~6 h, already costed), predict
  before reading: 1-layer runs SOLVED iff CLEAN (thresholds as above), multi-layer solved runs may be non-clean.
  Save snapshots every 10 epochs to see when the basin is entered (no extra GPU cost).

### P2. The hard rank is the world's dimension D; D + 1 suffices unless wrap-around dominates. REFINE

Statement: per-head rank = D is hard because the per-move drift must be annihilated exactly before any
channel carries a full D-dim map (lemma). One spare direction lets the drift be parked off-span while the map is built.
The wrap/grid-size effect is a separate obstacle, which the lemma does not explain.
- For (REG): paper torus per-head rank 2: 0-2/8; rank 3: 6/8; rank 4: 8/8 (`RANK_SEP_RESULTS.md`,
  `RANK3_RESULTS.md`). N-D 2D grid 32: rank 2 1/8 vs rank 3 8/8. 3D grid 10: rank 3 1/8 (`RANK_ND_RESULTS.md`).
  3D grid 18: rank 4 8/8 (`RANK_WRAP_RESULTS.md`). This note (POST HOC): the rank-D failures are CLOCK or COLLAPSE, which
  are the lemma's two horns. COLLAPSE appears only with a separate per-head rank-2 latent. The shared rank-2 runs
  are 8/8 CLOCK. The text world shows the off-span clock route at r=4.
- Why failures sit on wrap-only and long-gap revisits (rank-2 wrap-only 0.557 vs r=4 0.970; gap >= 128 0.671 vs
  0.988, `RANK_MATCHED_RESULTS.md`): a clock error grows with elapsed moves, and those strata have the longest
  lags. CONJ, testable on stored checkpoints: error vs lag at fixed net displacement.
- Not explained: rank D + 1 at 3D grid 10 still ends in CLOCK on 3/8 (s1, s2, s7) and COLLAPSE on 1/8. Grid 18 fixes
  it (`RANK_WRAP_RESULTS.md`, with cell count, revisit rate and omega's range co-varying). Small, wrap-heavy tori also
  invite in-weights memorisation (P6). Do not attribute the 3D-grid-10 shortfall to the lemma.
- Reject the stronger form "rank must reach 4". Rank 3 vs 4 is UNMEASURED (`RANK3_RESULTS.md`).
- Sharpest test (cheapest): **1D ring, per-head rank 1 vs 2** on the H3 task (section 3, E1). At D = 1 the lemma
  leaves no collapse basin (D-1 = 0), so rank 1 must be either exactly drift-free or a clock. The H3 path arm is
  shared r=2 = D + 1 and solved 32/32 (`CANCEL_RESULTS.md`).

### P3. Clock and map are competing solutions. REFINE: drift is the order parameter; they compete for dimensions, not for the task

Statement: whether the accumulator is a map or a clock is set by one quantity, the per-move drift (signed + zero
drift = map; non-zero drift = clock). The task picks which is useful. The step's rank decides whether both fit in
one head.
- For: the same free signed arm learns alpha 0.591 (map) on the torus and 0.967 (clock) on recency
  (`RECENCY_RESULTS.md`, REG). A constant step on counted content recovers +0.594 over a constant on every token
  (`RECENCY_GATE_ABLATION.md`). Forcing monotone costs -0.280 on the torus vs -0.004 on recency: a task x mechanism
  interaction. Text world (REG secondaries + POST HOC above): 4/8 seeds carry a real per-move clock (31-38 of 64
  blocks drift) and solve as well as map seeds (criterion x SOLVED Fisher p 1.00). They coexist at r=4 only because the clock sits off-span in
  down-weighted channels.
- "Clocks are harmless to accuracy": true when the clock is off-span (s2, s3, s4); false when it is in-span (s0,
  unsolved). This is post hoc, n=1 for the harmful case.
- "Clock seeds rely on attention dropout": **REJECT** the strong form. `TW_NORMSTEP_RESULTS.md` shows the
  train-mode minus eval-mode gap on the runs below ceiling, clock and map types alike (MapWM s12, map-type, 0.828
  -> 0.987). Dependence on dropout goes with not being at the clean solution, not with being a clock. Unchecked on
  other batches.
- Why language makes clocks: a variable number of tokens per move and shared verb/direction-word steps give a
  common component for free. Prediction (CONJ): at r = D = 2 on the text world, clock seeds cannot solve. Each would
  collapse like s0.
- Sharpest test: text world at per-head rank 2 vs 3 (16 runs, ~2-3 h by the `TW_NORMSTEP` cost). Prediction: rank
  2 has no solved clock seeds, and its failures are in-span clocks or collapse; rank 3 reproduces the off-span clock seeds.

### P4. Signed steps are a capability. ACCEPT, with a winding caveat

Statement: a map needs per-channel steps that can cancel. Monotone steps cancel only by winding (a positive step of
2 pi / omega_i - a acts as -a in channel i alone), which no single step can do in every channel at once.
- For (REG): at matched length Abs - Signed -0.177, 0/8 vs 8/8 solved; opposition 1.92-1.97 vs 0.06
  (`SIGN_MATCHED_RESULTS.md`). This note: Abs and Pos 16/16 CLOCK, Signed 8/8 CLEAN.
- Against / caveat (POST HOC): with a winding-aware retrace drift, monotone runs reach 0.016-0.1 of an axis in
  their best head, against Signed 0.001-0.002, so they partly wind. Pos s7 solved (loss 0.049) without a clean head:
  the one exception to necessity over 120 one-layer runs. The arms are budget-scoped (stalled, not converged).
  It is a replication (Sarrof, Grazzi, Selective RoPE).
- Sharpest test: none needed for the headline. The open point is whether winding improves with budget (Pos s7) —
  low value.

### P5. One path layer ~ three attention layers: path integration hands attention a group prefix sum. ACCEPT; depth also repairs aliased codes

Statement: path integration gives one layer the prefix sum of the task's (abelian) group. Index attention must
build it with depth, at ~3 layers whenever the prefix sum is not a function of the token index (any cancellation),
and at 1 layer when it is.
- For (REG): Dyck at matched depth +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers (`DYCK_MDEPTH_RESULTS.md`). H3:
  k = 3 at p_plus 0.5 / 0.75 / 0.9 (knife-edge at 0.9) and k = 1 at p_plus 1.0, where displacement = time
  (`CANCEL_RESULTS.md`). Text world RoPE 2L 0.772 vs path 1L 0.969 (`TEXTWORLD_RESULTS.md`). Match-Query loop x4 ~ 3
  real layers (NOT-PRE). This note: at rank 2 the depth aids solve without a clean code (loop 2/8, 4 layers 5/8,
  all 7 solved runs non-clean; `LOOP_RANK_RESULTS.md` gives 2/8, 5/8), so depth stands in for a missing map dimension
  as well as for the integration itself.
- Boundary that fits: the cumsum is abelian and token-local. Egocentric rotation actions (non-abelian, the step
  depends on heading) are the documented failure, +0.050 -> +0.488 after allocentric recoding (`KNOB_SWEEP_n8.md`,
  NOT-PRE, 16-epoch recipe). Context-dependent words need a context step (`CTXSTEP_*`, PILOT). Paying 34x for
  non-commutativity buys +0.005-0.014 on abelian tasks (`FAMILY_TREE_RESULTS.md`).
- Against: the "3" is budget- and threshold-scoped (H3 knife-edge, Dyck index arms still climbing at 1x).
- Prior art: depth vs automaton prefix sums (Liu et al. 2022, "shortcuts to automata"; Hahn 2020) are not in
  `papers/INDEX.md`. Check them before any claim about the exchange rate's form.
- Sharpest test: hold the walk fixed and vary the group. Run Z_N (ring), Z_N^2 (torus), and a non-abelian group of
  similar size (dihedral D_N actions, as tokens) with a path 1L arm and index 1-4L arms. Prediction: k ~ 3 for both
  abelian groups, and path 1L loses its advantage on the dihedral group unless the step is made state-dependent.
  ~4-6 h at H3's cost (12 min/run).

### P6. Small maps are memorised, large maps are mapped. REFINE: three solutions compete, and findability moves the boundary

Statement: revisit prediction has three solutions: in-weights memorisation of a fixed map, local in-context lookup
(short revisit structure), and a relational map. Training lands on whichever it reaches first among those that fit.
The map wins when the others are expensive (large maps, long lags) and when the map is findable (rank > D, signed,
little leak).
- For: 2D grid 10 (100 cells, fixed training map) is memorised (own map 0.986, unseen 0.273) with no relational where
  (leak 1.13, attention ~3 steps back) (`RANK_WRAP_RESULTS.md` REG-sec; `docs/WHAT_WHERE_CHECKS.md` POST HOC).
  Failing rank-D runs show an own-map minus held-out gap of +0.093 to +0.169, solved runs ~0 (`RANK_ND_RESULTS.md`
  S3, REG-sec): when map search fails, the model partly memorises **even at 1024 cells**. Within one data condition
  separation tracks success across seeds (Spearman -0.90; POST HOC). MiniWorld: index attention solves maps of <= 128
  occupied cells (0.987), and the path effect appears at 512 (+0.29) (`VISITS_TEST.md`, post hoc pooling, NOT-PRE,
  noise floor 0.150).
- Against: the 2D and 3D wrap cells co-vary grid size, revisit rate, cell count and omega range. MiniWorld's
  threshold is n=3-5 and not pre-registered.
- Prior art: the in-weights vs in-context transition with task diversity (Chan 2022, Kirsch 2022, Raventos 2023,
  Reddy 2023; `docs/lit/LIT_NEW_OBJECTS.md`). Map size plays the diversity role. What may be ours: the threshold
  depends on how findable the in-context solution is, not only on the data.
- Sharpest test: a fixed-map size sweep x findability (E3).

### P7. Separation emerges; the what-to-where boundary is learned, and in language it is semantic, not "only actions move". ACCEPT (post hoc), with a mechanism

Statement: trained path models score with one shared position kernel x a content gain, and the where-variable is
built only from tokens that should move it. Which tokens should is a property of the task's semantics. In language
the step works as an address that codes **role as well as place**: words that frame a mention as not-here (asides)
move it off the cell's phase.
- For: object identity removed from the score costs 0.011-0.025 (unmeasured), shared kernel x gain 0.988-1.000;
  index models collapse below floor (`docs/WHAT_WHERE_CHECKS.md`, POST HOC). Leak = the object-dependent part of the
  drift, from the pre-LayerNorm step; removing it buys +0.0107 (`LEAK_RESULTS.md`, REG-sec). The direction-words-only
  oracle caps at 0.972: 78/78 and 89/89 of its errors name an aside object at that cell, and the learned models make
  none (`TW_NORMSTEP_RESULTS.md`, POST HOC, 2 seeds). Step norms (`TEXTWORLD_PROBE.json`, map seed s1, units where a
  direction step ~2.8): "remembered" 0.99, "thought" 0.53, "about" 0.52, "story" 0.43, against "saw" 0.25 and
  objects 0.07-0.18. The aside frame carries the largest non-direction steps, consistent with displacement-tagging
  (descriptive; directions not checked).
- Against: the aside mechanism rests on 2 seeds and norms. "Separation is learned" is prior art (MapFormer Fig. 9).
  Whittington's "separation needs factorised data" is not refuted, only confounded with map size (`LIT_WHAT_WHERE.md`).
- Link to the lemma (CONJ): an aside excursion that returns (net zero over the clause) is drift-free per clause.
  So role-tagging costs no map dimension if it closes. A non-closing aside step would be a clock.
- Sharpest test (CPU first, minutes): on all 8 seeds x {MapWM, NormStep, DirOnly} of `runs/tw_normstep`, measure
  the phase offset of aside nouns from the cell phase and the net phase change over each aside clause. Prediction:
  learned arms offset > 1 rad in weighted channels and net change ~0. DirOnly offset exactly 0. Then GPU (~3 h): a
  grammar variant where aside objects never coincide with the cell's object. Prediction: aside steps shrink and DirOnly
  reaches the learned arms.

### P8 (context, already the thesis). Robustness is not capability

Kept as stated in `docs/WHERE_THINGS_STAND.md`. Every principle above is drawn from matched-length (and, for Dyck,
matched-depth) evidence. None uses an OOD-only number.

## 3. Highest-value experiments (ranked by information per GPU-hour)

Costs come from comparable logs: torus T=1024 runs take ~5.4 s/epoch, so 900 ep ~ 81 min (`runs/rank_sep/*.log`), and
4 concurrent slots give wall ~ runs x 1.35 h / 4. H3 ring runs take 2.4 s/epoch x 300 ep ~ 12 min (`runs/cancel/p0/*.log`).
Every batch needs rule 29: pre-registration, an independent code audit, and pilots on outside seeds.

**E1. Rank = D on the 1D ring, with and without drift (~1.2 h wall, 24 runs).** H3 task, p_plus 0.5, T=128, 300
ep. Arms: per-head r=1 (= D), per-head r=2 (= D+1; the shared-r2 H3 path arm is the reference), and per-head r=1 with drift
forbidden by construction (antisymmetric tied action steps, zero observation and blank steps). Pre-registered
primary: SOLVED counts and the basin of every run. Predictions:
- r=1 <= 2/8 with CLOCK failures. A collapse basin cannot exist at D-1 = 0.
- drift-forbidden r=1 >= 6/8.

Falsifier: r=1 >= 6/8. Then "rank = D is hard" needs D >= 2, and the dynamic reading of the lemma fails. Cheapest
test that can break the synthesis.

**E2. Forbid drift at the hard rank on the 2D torus (~6 h wall, 16 runs + 1 repro).** T=1024, 900 ep. Arms:
- per-head r=2 drift-forbidden (as in E1);
- per-head r=2 ActOnly-only (observation drift removed, action common mode free).

Compare with stored B and E after an exact repro of one seed (as `RANK3` did). Snapshots every 10 epochs in all
arms. Predictions:
- drift-forbidden r=2 >= 6/8 (vs 2/8), with any failures COLLAPSE (the lemma's other horn);
- ActOnly-only in between;
- in snapshots, solved r=3 runs pass through an off-span clock state (perp share > 0.5 with kappa > 0.01) before
  cleaning, and failed r=2 runs enter their basin early.

Falsifier: drift-forbidden r=2 <= 2/8 with clean codes. Then drift is not the obstacle. Add 3D grid 18 rank 3
drift-forbidden (+8 runs, ~3 h) for generality.

**E3. Memorisation vs mapping, as a function of findability (~12 h wall, 36 runs).** One fixed training map per
seed (the N-D trainer), 2D grids side 10 / 14 / 18, per-head rank 2 vs 3, 6 seeds, T=1024, 900 ep. Primary:
own-map minus unseen-map accuracy (memorisation index) and unseen accuracy per grid. Prediction: rank 2 memorises up
to larger grids than rank 3, so the in-weights/in-context boundary moves with how findable the map is. Falsifier:
identical curves, which leaves the data-diversity account (Kirsch / Raventos) sufficient.

Free, do first: commit `basins.py` as a readout, and pre-register "1-layer SOLVED iff CLEAN" on the next rank batch
(P1). Run the P7 aside-offset check (CPU, minutes).

## 4. Measured vs conjectured, at a glance

| claim | status |
|---|---|
| rank-D failure, sign failure, rank-D+1 success | REG |
| 1-layer SOLVED requires a clean head (0/47 rank-line exceptions; 1/120 with sign arms) | POST HOC, thresholds post hoc |
| aliasing lemma (states) | algebra |
| text-world clock seeds use the off-span route; the in-span seed is the unsolved one | POST HOC, n=8 (1 harmful case) |
| depth aids solve without a clean code | POST HOC, n=16 |
| rank D is hard *because* the clock + map intermediate is closed (dynamics) | CONJ (E1, E2) |
| wrap-only / long-lag failures are clock error growing with lag | CONJ (CPU-testable) |
| memorisation boundary moves with findability | CONJ (E3); partial support REG-sec (own-map gap in failing rank-D runs) |
| asides are role-tagged by closing excursions | CONJ (CPU-testable) |

Prior art: rank slack removing spurious minima (Burer-Monteiro low-rank SDP analyses) and lifting from the circle to
higher spheres removing twisted states in phase synchronisation are outside the corpus. They are analogies only;
verify before citing. MEC grid cells also integrating elapsed time (Kraus et al. 2015) is likewise outside the corpus.
The lemma itself is elementary; nothing in `docs/lit/` or `papers/INDEX.md` states it. Check before calling it new.

## Appendix A. Scripts (run from /home/prashr with PYTHONPATH=/home/prashr; CPU, 4 threads, < 2 min each)

Copies at `/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/` (`basins.py`,
`basins_rt.py` = retrace variant, `basins_loop.py`, `tw_weights.py`, `tw_latent.py`, `per_head.py`).

`basins.py` (core):
```python
import json, numpy as np, torch
torch.set_num_threads(4)
from pathlib import Path
from mapformer.train_variant import VARIANT_MAP
R = Path("/home/prashr/mapformer")

@torch.no_grad()
def heads(cp, weighted=True):
    b = torch.load(cp, map_location="cpu", weights_only=False); c = b["config"]; arm = b["variant"]
    m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                         n_layers=c["n_layers"], grid_size=c["grid_size"]).eval()
    m.load_state_dict(b["model_state_dict"])
    V = c["vocab_size"]; D = c.get("n_dims", 2); K = c.get("n_obs_types", 16); pe = c.get("p_empty", 0.5); nA = 2 * D
    e = m.token_emb(torch.arange(V)); a = (m.action_to_lie(e[None])[0] * m.path_integrator.omega[None]).numpy()
    L = m.layers[0]; H = c["n_heads"]; dh = c["d_model"] // H; h = L.norm1(e)
    Q = L.q_proj(h).view(V, H, dh); Kk = L.k_proj(h).view(V, H, dh)
    qa = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); ka = torch.sqrt(Kk[..., 0::2] ** 2 + Kk[..., 1::2] ** 2)
    w = (qa[:nA].mean(0) * ka[nA:].mean(0)).numpy() if weighted else np.ones_like(a[0])
    A = a[:nA]; O = a[nA:]                               # actions (+d,-d pairs), then K objects, then blank
    U = np.stack([(A[2 * d] - A[2 * d + 1]) / 2 for d in range(D)])
    mm = np.angle(np.exp(1j * (A.mean(0) + pe * O[K] + (1 - pe) * O[:K].mean(0))))   # per-move drift, wrapped
    out = []
    for hh in range(H):
        sw = np.sqrt(w[hh] / w[hh].sum()); Uw = U[:, hh] * sw
        sv = np.linalg.svd(Uw, compute_uv=False)
        out.append((np.linalg.norm(mm[hh] * sw) / np.linalg.norm(Uw, axis=1).mean(), sv[-1] / sv[0]))
    fl = np.asarray(b["losses"], float); fl = fl[-max(1, len(fl) // 20):].mean()
    return out, fl

def basin(o):
    ok = [ind for k, ind in o if k <= 0.01]
    if not ok: return "CLOCK"
    return "CLEAN" if max(ok) >= 0.2 else "COLLAPSE"
# loop over runs/<dir>/<arm>_s<seed>/<arm>.pt; tabulate basin(o) x (fl < 0.05)
```
Note: the paper torus and the N-D env share the layout [actions as (+d, -d) pairs][K objects][blank] (checked in
`environment.py` l.44 and `environment_nd.py` l.91-94).

`tw_latent.py` (text world; `load` from `analyze_textworld_secondary`):
```python
m = load(f"{R}/Vanilla_r4_L1_s{s}/Vanilla_r4.pt", "cpu")
Z = m.action_to_lie.w_in(m.token_emb.weight).numpy(); Wo = m.action_to_lie.w_out.weight.numpy()
om = m.path_integrator.omega.detach().numpy().reshape(-1)
Dz = {a: Z[te.dir_ids[a]].mean(0) for a in range(4)}; u = (Dz[0]-Dz[1])/2; v = (Dz[2]-Dz[3])/2
mlat = np.mean([Dz[a] for a in range(4)], 0) + Z[[te.idx[x] for x in VERBS]].mean(0)
B, _ = np.linalg.qr(np.stack([u, v], 1)); perp = norm(mlat - B @ (B.T @ mlat)) / norm(mlat)
clock = wrap(om * (Wo @ mlat)); P = np.stack([om * (Wo @ u), om * (Wo @ v)], 1); clean = abs(clock) < 0.05
# independence: s_min/s_max of P[clean] * sqrt(w[clean]); w as in basins.py with object-word keys
```
