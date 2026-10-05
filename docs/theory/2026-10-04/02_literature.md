# Cognitive-map literature vs our measurements (2026-10-04)

Theory note, not a result. One of five reviews of the repo; this one's angle is the neuroscience and ML theory of
cognitive maps. Nothing here was run. Every number is copied from the results file named beside it; a correction
block at the top of a results file supersedes its body. Prior-art reviews already in the repo
(`docs/lit/LIT_WHAT_WHERE.md`, `LIT_NEW_OBJECTS.md`, `LIT_CONTEXT_STEPS.md`) cover attention-side what/where,
new-object transfer and context steps. This note does not repeat them and covers what they leave out: grid-cell
theory, time cells and temporal context, remapping, event cells, non-local representation.

Status words. **REG** = pre-registered, one batch, n=8, verdict as registered. **REG-sec** = a declared secondary of
a registered batch. **POST HOC** = eval-only analysis of stored checkpoints. **PILOT** = n=1-2. **not pre-reg.** = an
older batch with no registration. **LIT** = no measurement of ours bears on it.

Terms. **Rank** = per-head rank of the content-to-angle map `W_out W_in` (how many independent directions a token can
move one head's phase). **D** = dimension of the task torus. **Wrap-only revisit** = a return to a cell seen before
only at a different unwrapped position (a loop closure through the torus topology). **Leak** = a nonzero phase step
from a non-action token. **Clock** = an accumulator whose increments cannot cancel, so it measures path length or time.
**Map** = one whose increments cancel, so it measures net displacement.

## 1. Table: literature claim -> our measurement

Verdicts: **confirms**, **extends** (adds a measurement or a boundary the source lacks), **contradicts**, **untested**.

| # | literature claim (source) | our measurement | verdict | status | file |
|---|---|---|---|---|---|
| 1 | Structure (where) must be factorised from sensory content (what) to generalise to new environments (TEM, Whittington et al. 2020, Cell 183:1249; "How to build a cognitive map", Whittington, McCaffary, Bakermans, Behrens 2022, Nat Neurosci 25:1257) | Path models transfer to redrawn maps (torus +0.243 over index at training length); converged path models keep 0.974-0.989 with object identity removed from the score, index RoPE/PoPE fall below the 0.598 floor | confirms; extends (the separation is measured causally at the score, in a model where it is not architected) | REG (torus), POST HOC (score) | `PAPER2X2_RESULTS.md`, `docs/WHAT_WHERE_CHECKS.md` |
| 2 | Generalisation needs inference on first visit, by path integration (TEM's loop-closure claim) | In-context maps need path integration: Match-Query 0.730 +/- 0.247 vs index 0.154 (chance 0.0625); destroying the context drops 0.918 -> 0.074 | confirms | not pre-reg., n=5 | `MATCH_QUERY_SCALE.md` |
| 3 | Loop closure is part of the structural code (TEM) | Wrap-only revisits (loop closures) are where learned path integrators fail when rank is tight: 0.53-0.80 vs 0.91-0.97 on other revisits | extends (loop closure is the hard stratum for a *learned* integrator) | REG-sec | `RANK_ND_RESULTS.md`, `RANK_WRAP_RESULTS.md` |
| 4 | TEM-t: transformer = TEM if Q,K come from position only and V from content, with position updated by actions only (Whittington, Warren, Behrens 2022, arXiv 2112.04035) | Position-only score costs 0.011-0.025 in converged path models (below MDE: the interaction is not needed). The action-only update is the right target on the grid task (ActOnly +0.0107, p 0.0002) but the wrong one in language: the direction-word-only oracle caps at 0.970-0.974, every checked error an object from an aside | confirms (score half); **contradicts as a design for language** (update half) | POST HOC (score, aside errors 2/8 seeds); REG (ActOnly) | `docs/WHAT_WHERE_CHECKS.md`, `LEAK_RESULTS.md`, `TW_NORMSTEP_RESULTS.md` |
| 5 | MapFormer Fig. 9 (Rambaud et al. 2511.19279): generalisation arrives when observation steps -> 0 and opposite actions cancel; energy constraints "could be added to force disentanglement" | Residual leak measured: removing it gains exactly its cost, +0.0107 (p 0.0002) and speeds convergence (16/16 SOLVED vs 8/8 descending). Leak = code projection onto the step map's 4-d row space, scaled by pre-LayerNorm norm. Energy constraint never run | confirms the cancellation; extends with the size, mechanism and cost of the residual; energy constraint untested | REG + POST HOC | `LEAK_RESULTS.md`, `docs/WHAT_WHERE_CHECKS.md` |
| 6 | Factorised codes emerge only when task factors are independent in the data; entangled tasks warp grid fields to objects (Whittington et al. 2023, arXiv 2210.01768) | Small fixed map (2D, 100 cells): no relational where at all (observation steps as large as actions, leak 1.13; attention ~3 steps back; own map 0.986, unseen 0.273). Large fixed map (1024 cells): separated as well as redrawn maps (inter/pos 0.034) | consistent; not a test (fixed 1024-cell map is still nearly factorised, 0.21 bits vs 0.87 for 100 cells) | POST HOC | `docs/WHAT_WHERE_CHECKS.md` sec. 2, `RANK_WRAP_RESULTS.md` |
| 7 | Hippocampal maps arise from sequence learning; clones disambiguate aliased observations (CSCG, George et al. 2021, Nat Commun 12:2392; Raju et al., arXiv 2212.01508) | Path integration's advantage is set by map extent, not aliasing: -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells; "effect scales with aliasing" is withdrawn. Heavy aliasing (4 observation types) does break Match-Query's separation (seed overlap) | extends (a size threshold for when a relational map beats sequence memory); partly confirms (heavy aliasing hurts) | not pre-reg., n=3-5 | `VISITS_TEST.md`, `ALIASING_CONTROLLED.md`, `MATCH_QUERY_SCALE.md` |
| 8 | EM and WM give the same solution; EM learns faster except on N-back (Whittington et al. 2025, Neuron 113:321) | EM's recency deficit is search: EM - WM -0.375 (0/8); an installed rewind frozen gives 1.000 | confirms the N-back exception; extends with its cause (optimisation, not representation) | REG | `EM_WM_STATE.md`, `SEARCH_RESULTS.md` |
| 9 | Path integration + nonnegativity + center-surround targets give hexagonal grids (Sorscher et al. 2023, Neuron 111:121; Banino et al. 2018, Nature 557:429; Cueva & Wei 2018, ICLR); grids depend on non-fundamental choices (Schaeffer, Khona, Fiete 2022, NeurIPS) | No hexagonal units in any variant (max grid score 0.258 < 0.3; `Grid_Free` 0/22 modules); our phase code is per-channel 1D bands, imposed by the architecture | confirms Sorscher's conditions and Schaeffer's caution (CE on aliased tokens, no nonnegativity, no DoG target) | not pre-reg., small n; a live negative | `HIPPOCAMPAL_ANALYSIS.md`, `docs/LOG.md` |
| 10 | Grid modules: each module's population lies on a 2-torus (Gardner et al. 2022, Nature 602:123); multiple modules for range and error correction (Fiete, Burak, Brookings 2008, J Neurosci 28:6858; Wei, Prentice, Balasubramanian 2015, eLife 4:e08362); multiple modules are optimal for actionable codes (Dorrell et al. 2023, arXiv 2209.15563) | Per-head rank = D is hard (2D: 1/8; 3D: 1/8); D+1 solves (2D 8/8; 3D 8/8 on grid 18). Solved rank-3 heads still put actions in a plane (2-plane energy 0.9967): the solution is D-dimensional, the search needs D+1. A rank-2 solution exists and is held when installed | **extends**: a learnability constraint the normative theories lack (they give the optimum, not its reachability) | REG | `RANK_ND_RESULTS.md`, `RANK_WRAP_RESULTS.md`, `RANK3_RESULTS.md`, `RANK3_GEOMETRY.md`, `RANK_PROJ_RESULTS.md` |
| 11 | Higher-dimensional variables are encoded by many 2D modules through low-dimensional (random, fixed) projections (Klukas, Lewis, Fiete 2020, PLoS Comput Biol 16:e1007796) | Total rank spread across heads does NOT rescue: block-diagonal rank 4 made of two rank-2 heads 2/8 vs per-head rank 4 8/8 (p 0.0070). Fixed random per-head projections never run | tension (learned projections, 2 heads) -- untested in Klukas's regime (fixed projections, many modules) | REG | `RANK_SEP_RESULTS.md` |
| 12 | 3D grid codes in bats and rats have local, not global, order (Ginosar et al. 2021, Nature 596:404; Grieves et al. 2021, Nat Commun) | A learned 3D path integrator with rank D+1 is exactly periodic on a large 3D torus (8/8, 1.000); on a small wrap-heavy one it partly fails (4/8) | not comparable (our periodicity is architected per channel); untested | REG | `RANK_WRAP_RESULTS.md` |
| 13 | Time, path length and allocentric position are one computation with different rates: alpha = const (time), alpha = speed (path), alpha = signed velocity (position) (Howard et al. 2014, J Neurosci 34:4692) | Index RoPE = constant rate; monotone content-dependent (Abs, softplus) = speed-driven; signed = velocity-driven. At matched length the signed arm solves 8/8, the monotone arms 0/8 and 1/8 (-0.177, p 0.0002); index 0/8 | **confirms and extends**: the first matched-length learning test of Howard's Case II vs Case III, in a learned system that must choose its rate | REG | `SIGN_MATCHED_RESULTS.md` |
| 14 | Grid cells integrate both elapsed time and distance when running in place (Kraus et al. 2015, Neuron 88:578); CA1 time vs distance cells (Kraus et al. 2013, Neuron 78:1090) | Text world: 4/8 seeds learn a per-move clock on top of the map (31-38 of 64 phase channels drift between visits; clock vector parallel to the verb step, cos +1.000); 4/8 learn a pure map; accuracy equal (Fisher p 1.00) | confirms the time+distance mixture can coexist with the map at no cost; extends: in one architecture and task, whether it appears is a seed (basin) property | REG-sec + POST HOC | `TEXTWORLD_RESULTS.md` |
| 15 | Hippocampal cells count events invariant to event duration (event-specific rate remapping, Sun, Yang, Martin, Tonegawa 2020, Nat Neurosci 23:651); boundary and event cells (Zheng, Schjetnan, Rutishauser 2022, Nat Neurosci 25:358) | The text-world clock ticks per move clause, not per word: per-word drift 0.077 rad (MapWM) against a random level ~1.57; NormStep adds a small per-word drift (+0.057 rad, p 0.027) | consistent (an event-indexed counter, nearly duration-invariant); untested as an event-boundary claim | REG (drift), REG-sec (clock) | `TW_NORMSTEP_RESULTS.md`, `TEXTWORLD_RESULTS.md` |
| 16 | LEC carries an experience-driven time code; MEC a spatial code (Tsao et al. 2018, Nature 561:57) | Monotone content-driven accumulators measure elapsed experience (growth exponent 0.94, ballistic); signed ones measure position (0.52, diffusive); r(opposition, exponent) +0.9995 | consistent; the stability argument in sec. 2(b) predicts the split | not pre-reg. (exponent) + REG (sign) | `.claude-memory/project_clock_vs_map.md`, `SIGN_MATCHED_RESULTS.md` |
| 17 | Temporal context drifts monotonically with experience and supports episodic retrieval (TCM, Howard & Kahana 2002, J Math Psychol 46:269) | A monotone drift cannot be a map (sign result); recency needs no clock (a signed rewind solves k-back) | contradicts "temporal context alone suffices" for spatial inference; confirms TCM's own scope | REG (sign), REG (recency) | `SIGN_MATCHED_RESULTS.md`, `RECENCY_RESULTS.md` |
| 18 | Successor representation: predictive maps skew against the travel direction under biased policies (Stachenfeld et al. 2017, Nat Neurosci 20:1643) | Biased 1D walk: 1 path layer solves every p_plus; index needs 3 layers when steps cancel. Attention kernels never inspected for skew | untested (cheap: CPU on `runs/cancel`) | REG (accuracy only) | `CANCEL_RESULTS.md` |
| 19 | Remapping = inferring a new hidden context; the structural code is conserved across remapping (Sanders, Wilson, Gershman 2020, eLife 9:e51140; TEM) | Every eval redraws the observation map (global remapping per sequence); the phase code is shared by construction | not a test (architected) | -- | -- |
| 20 | Path-integration gain is plastic and set by landmarks (Jayakumar et al. 2019, Nature 566:533); speed cells are context-invariant (Kropff et al. 2015, Nature 523:419) | Content sets a small gain on the where-update in MapWM (leak scales linearly with code norm); normalising the step input removes it (NormStep, +0.0107, faster convergence) | extends (an unnormalised velocity read-in leaks content into the where-update and slows learning; a normalised one does not) | REG + POST HOC | `LEAK_RESULTS.md`, `docs/NORMSTEP_NOTES.md` |
| 21 | Non-local representation: remote places are activated volitionally (Lai, Tanaka, Harris, Lee 2023, Science 382:566); goal-directed replay (Pfeiffer & Foster 2013, Nature 497:74); object-vector cells code objects at an offset (Hoydal et al. 2019, Nature 568:400) | Mentioned-but-absent objects (asides) must be bound away from the current cell; learned steps on non-movement words do this (0 aside errors on 2 solved seeds), action-only steps cannot (100% of the oracle's errors) | extends into language (a "not here" binding is needed and is learned); untested in biology | POST HOC, 2/8 seeds | `TW_NORMSTEP_RESULTS.md` |
| 22 | MapFormer v4 Table 6: MapWM collapses in 5D (0.75/0.50/0.35) with inner rank set to the world's dimension and a 5-cell grid | Rank = D is the hard case in 2D and 3D; small wrap-heavy grids make even D+1 partial | untested account (our "packing geometry" account of Table 6 is withdrawn; this is a different one) | REG (our side) | `RANK_ND_RESULTS.md`; paper App. D.1 |
| 23 | Depth/recurrence in hippocampal models (TEM-t recurrence; MapFormer's claim that one layer emulates a stack) | Path integration is worth ~3 attention layers at matched depth (Dyck +0.353/+0.130/+0.045/+0.024 at 1-4 layers) and on a biased 1D walk | extends (an exchange rate); at 4 layers and 3x budget every arm hits ceiling | REG | `DYCK_MDEPTH_RESULTS.md`, `CANCEL_RESULTS.md` |
| 24 | Grid scale ratios follow a geometric progression ~1.4-1.7 (Stensola et al. 2012, Nature 492:72; Wei et al. 2015) | Our omega ladder is geometric by construction (paper eq. 17); trained ratios not analysed | untested | -- | `HIPPOCAMPAL_GRID_FREE.md` (no conclusion) |

What the table says in one line: the literature's **representational** claims (factorisation, path integration for
inference, time and space as one rate-modulated computation) are confirmed; what we add are **learnability and
boundary** results the normative and architected models cannot produce: rank D vs D+1, loop closure as the hard
stratum, a size threshold for map vs memorisation, the cost of a leak, and a "not here" binding in language.

## 2. Where our system can say something the literature cannot yet

Each item: what the literature has, what we have, the theoretical statement, novelty, and predictions. "Possibly new"
means not found in `docs/lit/`, `papers/txt/` or the searches listed in sec. 4; the searches were not exhaustive.

### (a) Rank D is a search bottleneck; D+1 is a learnability margin, not a capacity requirement

**Literature.** Normative grid theory explains *why* codes are multi-scale and multi-module: range and error
correction (Fiete 2008, residue number system), neuron economy (Wei 2015), optimal actionable codes (Dorrell 2023).
Each module's population lies on a 2-torus (Gardner 2022), i.e. exactly D=2 dimensions per module. Klukas 2020 shows
D>2 variables can be encoded by many 2D modules via fixed random projections. All of these characterise the optimum.
None asks whether gradient descent can reach it when the velocity read-in is learned. In the ML models
(Banino 2018, Cueva & Wei 2018, Sorscher 2023) velocity arrives already in Cartesian or speed/heading form. The
read-in is given, not learned from tokens.

**Ours.** A rotating unit (one head) whose learned input has exactly D degrees of freedom fails 7/8 times in 2D and in
3D. With D+1 it solves 8/8 (2D grid 32; 3D grid 18). The spare direction is not used at the end: solved rank-3 heads
put actions in a plane (energy 0.9967), and a rank-2 solution exists and is held (frozen 0.9955). Failures fall into a
non-cancelling basin (opposition 1.58-1.83 vs 0.017-0.076 solved). Overcompleteness *across* heads does not substitute
(two rank-2 heads 2/8 vs one rank-4 head 8/8, p 0.0070). The failures concentrate on loop closures (wrap-only
revisits), and small wrap-heavy tori make even D+1 partial.

**Theoretical statement.** The final code is D-dimensional per module, as in biology. *Finding* it by gradient
descent needs a margin of at least one extra dimension inside each jointly rotating unit. Without the margin the
optimiser falls into a non-cancelling basin, and the error shows up first on loop closures. So the slack sits within a
module, not across modules.

**Novelty.** Possibly new as a statement about path integrators. That overparameterisation eases optimisation is
generic (e.g. implicit acceleration by depth, Arora et al. 2018), so what is specific is the location (within a unit),
the threshold (exactly D+1) and the stratum (loops). LieRE's interior optimum is on the output side (`papers/INDEX.md`).
The scope is n_heads=2, one recipe and 900 epochs. 2x the budget does not rescue rank 2 (0/8 vs 7/8), but 5/8 rank-2
runs were still descending.

**Predictions.**
- ML, cheap: give each head a FIXED random rank-2 `W_out` (Klukas's regime) with many heads (nh = 8) on the 3D torus.
  If the hardness is search over a learned projection, fixed projections across many heads should solve where learned
  rank D per head fails. If they fail too, the per-unit rank itself is binding. About 16 runs.
- ML, cheap: track the effective rank of the action code over training in rank-3 runs. The prediction is a transient
  use of the third direction that collapses to a plane as the model solves. It needs per-epoch checkpoints, which the
  stored runs lack, so re-run 2-3 seeds with snapshots.
- ML: MapFormer v4 Table 6's 5D collapse used inner rank = D and a 5-cell grid. Our account predicts per-head rank
  D+1 on a large 5D torus recovers it, and on a 5-cell grid does not fully recover it.
- Biology: an immature grid module should have population activity of intrinsic dimension > 2 that contracts to the
  2-torus as path integration matures (developmental recordings P16-P30 plus the topological analysis of Gardner
  2022). Separately, loop-closure accuracy should lag retrace accuracy in animals learning a novel looped or toroidal
  VR world without landmarks, and the lag should shrink with experience.

### (b) Clocks vs maps: Howard's three cases, the stability of signed integration, and LEC vs MEC

**Literature.** Howard et al. 2014 unify time cells, path-history cells and landmark/border codes as Laplace
(leaky-integrator) transforms whose rate alpha(t) is constant (Case I: time), speed (Case II: path), or signed velocity
(Case III: allocentric position). In Case III negative velocity makes the real exponential *grow*. They implement it
with pairs of rectified cells (velocity along theta and theta+pi) read out as a ratio, "to avoid numerical errors due
to exponential growth" (Methods). Separately, Tsao 2018 finds an experience-driven time code in LEC. Kraus 2013 and
2015 find time and distance mixed in CA1 and MEC. In ML, Puranik and GRAPE classify encodings as decay (real) vs
rotation (imaginary) slots, and GRAPE's content-dependent additive slot needs g(x) >= 0 (`reference_positional_landscape.md`).

**Ours.** At matched length, signed content-dependent rotation solves the torus 8/8. Monotone content-dependent
rotation (Howard's Case II) solves 0/8 and 1/8, and index rotation (Case I) 0/8. Monotone accumulators grow
ballistically (exponent 0.94) and signed ones diffusively (0.52). In prose, half the seeds add a per-move clock (Case
II) on top of the map (Case III) in the same population, with no accuracy cost. NormStep adds a faint per-word drift
(Case I-like). The clock ticks per move, nearly independent of how many words the move takes.

**Theoretical statement.** Howard's alpha is the same object as the selective-SSM step Delta_t and MapFormer's
content-dependent angle increment ("learning how far to advance time", Puranik). Seen that way, our sign result
is Howard's Case II vs Case III decided by learning, at matched length. The stability side gives a prediction about
anatomy. In a decay (real-exponent) slot a signed rate is unstable: past keys can be amplified without bound. Such a
slot can therefore hold only a clock, unless it uses push-pull pairs as Howard did by hand. In a phase (imaginary) slot
a signed rate is free and bounded. So: **monotone, magnitude-like codes should carry time, and phase-like codes should
carry space.** That is LEC's drifting experience code vs MEC's grids. It is also why GRAPE's contextual additive
slot is forced non-negative.

**Novelty.** The sign result is a replication (Sarrof, Grazzi, Selective RoPE). The "two slots" frame is Puranik's and
GRAPE's. Identifying Howard's alpha with Delta_t and the sign axis with his Case II/III split is possibly new as a
cross-field statement. No source in `docs/lit/`, `papers/txt/` or our searches makes it. The LEC/MEC reading is a
hypothesis.

**Predictions.**
- Biology: decompose a population's step-to-step change during back-and-forth running on a linear track into an odd
  part (flips with direction, cancels on return) and an even part (common to both directions). Prediction: the even
  part points along the slow within-session drift (Mankin et al. 2012, PNAS 109:19462) and accumulates per traversal
  (event), not per second. This is the population form of our clock seeds' common component, parallel to the verb
  step at cos +1.000. Existing Neuropixels linear-track data could test it.
- Biology: LEC's time code should not reverse when the animal retraces a path. MEC's phase should. A rate-coded
  (magnitude) population that does reverse with signed displacement should show push-pull pairing: cells tuned to
  opposite directions with anti-correlated drift.
- ML, cheap: on `runs/textworld`, the clock seeds' extra drifting channels should lie in the spare latent directions
  (r=4, D=2). A rank-2 shared text-world model should be unable to carry both clock and map. Its clock seeds should
  lose accuracy, or it should never produce a clock.

### (c) Asides: what a language cognitive map must exclude ("not here" binding)

**Literature.** Hippocampal-formation models bind content to the *current* position: TEM's `p = x (x) g`, TEM-t's
V = content with Q = K = position, CSCG's emissions. The position update is driven by actions only (TEM-t, Vector-HaSH,
CSCG schemas). Non-local content exists in biology: remote place activation under volition (Lai 2023), replay and
theta sequences (Pfeiffer & Foster 2013), object-vector cells coding an object at an offset (Hoydal 2019). But no
model of the map says how content that is *mentioned without being present* (language's "displacement", Hockett 1960)
should be kept off the current location.

**Ours.** In the text world, an oracle that steps only on direction words (DirOnly, TEM-t's update) sits at
0.970-0.974 on 8/8 seeds with sd 0.001. On the two seeds checked, 100% of its errors (78/78, 89/89) name an object
from an aside ("she thought about a cat .") made at an earlier visit to the same cell. Solved learned-step models make
none of these errors: their non-movement words move the aside off the cell. On the grid task, by contrast, action-only
stepping is optimal (+0.0107).

**Theoretical statement.** In language, "only actions move the map" is the wrong separation. The right one is "only
*presence* binds to here". A content-dependent step is one way to implement it: a mental-state frame ("thought about")
shifts the binding phase, so the mentioned noun lands at an "elsewhere" address. The architected action-only update of
TEM-t, Vector-HaSH and CSCG cannot express that. A content-dependent path integrator can, because the what->where path
that leaks on the grid task is the same path that does the work here.

**Novelty.** Possibly new. The evidence is POST HOC on 2 of 8 seeds, and the grammar is scripted (asides always sit in
their own clause). Not checked: whether LM entity-binding studies (e.g. Feng & Steinhardt 2023, "How do language
models bind entities in context?") test modal or hypothetical mentions.

**Predictions.**
- ML, CPU only: on existing text-world checkpoints, the phase step summed over an aside clause should be a consistent
  vector across asides (an "elsewhere" displacement), not noise. Its size should exceed the attention kernel's width
  at the scored frequencies. If it is random, then "moves the aside off the cell" means scrambling, not addressing.
- ML: add "she saw a cat here" (presence) vs "she thought about a cat" (absence) clauses with identical nouns. A
  learned step should bind the first and displace the second. DirOnly cannot separate them.
- Biology / human fMRI: in a virtual walk narrated in words, an object mentioned modally ("thought about") at a
  location should NOT later be reinstated by returning to that location, while an object encountered there should
  be. Representational similarity between location and object patterns at return is the readout. Prediction from
  the model: the absence of binding is active (a displaced code), not merely a weaker encoding, so mentioned objects
  should be reinstated at a *consistent other* location or context code.

### (d) The what->where leak, normalisation, and gain in entorhinal cortex

**Literature.** CoRelNet and the Abstractor show sensory content entering a relational path costs OOD accuracy
(`LIT_NEW_OBJECTS.md`). In biology the path-integration gain is plastic and landmark-calibrated (Jayakumar 2019), and
MEC speed cells are context-invariant (Kropff 2015). Divisive normalisation is a canonical computation (Carandini &
Heeger 2012, Nat Rev Neurosci 13:51). Whittington 2023 shows grid fields warp toward objects in entangled tasks.

**Ours.** MapWM's where-update reads the content embedding before LayerNorm. Content leaks in as the projection onto
a 4-d row space scaled by the code norm: x2/x4 code norm costs 0.04/0.11-0.14, and codes orthogonal to the row space
leak exactly 0. Normalising the read-in (NormStep) or removing non-action steps (ActOnly) gains +0.0107 in
distribution and converges 16/16 where MapWM is still descending 8/8. Normalisation has its own failure: a shared
bias becomes a per-token tick. On text it is tiny (+0.057 rad) and not caused by the bias.

**Theoretical statement.** A velocity read-in that is not normalised against non-motion input lets *how much*
content there is set a gain on *where*. The cost shows in learning speed more than in the final code. Normalisation
removes the gain channel but leaves a common-mode tick, which is a clock. The literature attributes context-invariant
speed coding to sensing. Our result suggests it is also an optimisation requirement.

**Novelty.** The mechanism (pre-LN step, row space, norm scaling) and its in-distribution cost are possibly ours
(`LIT_NEW_OBJECTS.md` found no prior measurement). That content in the relational path costs accuracy is prior art.
The link to biological gain is an analogy.

**Predictions.**
- Biology: with self-motion held fixed (head-fixed VR, matched optic flow), a brief high-salience non-spatial stimulus
  (odour, object, sound) in darkness should cause at most a grid phase shift independent of stimulus intensity, if
  the velocity read-in is normalised (as Kropff's context invariance suggests). An intensity-proportional shift would
  indicate an unnormalised read-in, MapWM-like.
- Biology, developmental: in rearing or learning conditions where non-spatial input is strong and variable, grid
  regularity should mature more slowly. The model analogue is the leak slowing convergence (r(loss, acc) -0.944;
  partly speed).
- ML: the leak test that can fail (anisotropic codes with a SWAP shift, `LIT_NEW_OBJECTS.md` E2) is the right next
  measurement. Add an energy-penalised arm (MapFormer Fig. 9's suggestion, Whittington 2023) to compare a soft remedy
  against normalisation.

### (e) Memorisation vs mapping as a function of environment size

**Literature.** CSCG and "space is a latent sequence" explain place fields as sequence context that disambiguates
aliased observations. TEM explains transfer by factorisation. Whittington 2023 makes factorisation depend on
independence in the data. In-context vs in-weights learning has diversity thresholds (Chan 2022, Reddy 2023,
Raventos 2023, Kirsch 2022; `LIT_NEW_OBJECTS.md`). Remapping is hidden-context inference (Sanders 2020).

**Ours.** With ONE fixed training map, a 100-cell 2D torus is memorised: own map 0.986, unseen 0.273, no relational
where (observation steps as large as actions; attention a short n-gram-like lookback). A 1024-cell map is not
memorised: it generalises (0.999 unseen) and is as separated as models trained on redrawn maps. With redrawn maps,
path integration's advantage is -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells. The index arm solves small
maps but not large ones. Within one data condition, separation tracks success across seeds (Spearman -0.90).

**Theoretical statement.** Environment size is the diversity knob that switches a learner between an in-weights
lookup (sequence memory of one map) and an in-context relational map. Between 100 and 1024 cells, at this budget, a
single environment is too big to memorise and the factorised solution wins without any redraw. This refines
Whittington 2023: the relevant independence is per-cell content-position mutual information (0.87 bits at 100 cells,
0.21 at 1024), which a single large map already supplies. It also bounds CSCG's story: aliasing does not set when
path integration helps, map extent does.

**Novelty.** Possibly new for navigation. The size-threshold idea parallels known ICL/IWL diversity thresholds. The
evidence is POST HOC (probe) and not pre-reg. (extent table, n=3-5), and the 100-vs-1024 contrast is confounded with
wrap share and revisit rate (`RANK_WRAP_RESULTS.md` caveats).

**Predictions.**
- ML, registered and cheap: one fixed map at 100 / 225 / 400 / 1024 cells (2D, rank 3), n=8, scored on unseen maps.
  Prediction: a sharp transition in unseen-map accuracy and in observation-step size (the leak) at the same size.
  The leak readout is free and is the mechanistic marker. This is a cleaner version of `LIT_WHAT_WHERE.md` P2.
- Biology: animals with extensive experience in a single small enclosure should transfer less structure to a novel
  environment (slower map formation, weaker preservation of grid-place relationships across remapping, TEM's
  signature) than animals with experience in a single large one. Hippocampal codes in small, heavily revisited
  enclosures should look like sequence or lookup codes: predictable from the last few steps, not from integrated
  position.

## 3. What not to claim

- "MapFormer learns grid cells": no hexagonal units; our bands are architected (`HIPPOCAMPAL_ANALYSIS.md`).
- "Separation is learned" as a general finding: MapFormer Fig. 9 and trained-LM studies have it (`LIT_WHAT_WHERE.md`).
- "Our withdrawn lap-counting mechanism" as support for (b): `LAP_TRANSFER_NOREWARD.md` is withdrawn; the event-count
  link rests on the text-world clock only.
- The rank result as general: n_heads=2, one recipe, 900 epochs, torus tasks only.
- Any biology prediction above as supported by our data: they are consequences of the ML results, untested in
  animals.
- "Position effect scales with aliasing": withdrawn (`ALIASING_CONTROLLED.md`, `VISITS_TEST.md`).

## 4. Sources

Local: `papers/txt/mapformer.txt` (Fig. 9, App. A.7, A.8, D.1), `papers/txt/tale_two_algorithms.txt`,
`papers/INDEX.md`, `docs/lit/LIT_*.md` (TEM, TEM-t, CSCG, Raju, Whittington 2023, Dorrell 2023, SR, Vector-HaSH,
CoRelNet, Abstractor, ICL thresholds read there first-hand or at abstract level as marked).

Read for this note: Howard et al. 2014 full text (Cases I-III and Methods, https://www.bu.edu/hasselmo/HowardEtAl2014.pdf).
Abstract or summary level, via search:
- Whittington, McCaffary, Bakermans, Behrens 2022: https://www.nature.com/articles/s41593-022-01153-y
- Tsao et al. 2018: https://www.nature.com/articles/s41586-018-0459-6
- Kraus et al. 2015: https://www.cell.com/neuron/fulltext/S0896-6273(15)00820-X
- Sun et al. 2020: https://www.nature.com/articles/s41593-020-0614-x
- Zheng et al. 2022: https://www.nature.com/articles/s41593-022-01020-w
- Gardner et al. 2022: https://www.nature.com/articles/s41586-021-04268-7
- Ginosar et al. 2021: https://www.nature.com/articles/s41586-021-03783-x
- Klukas, Lewis, Fiete 2020: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007796
- Fiete, Burak, Brookings 2008: https://www.researchgate.net/publication/5255557_What_Grid_Cells_Convey_about_Rat_Location
- Wei, Prentice, Balasubramanian 2015: https://elifesciences.org/articles/08362
- Sorscher et al. 2023: https://par.nsf.gov/biblio/10513491-unified-theory-computational-mechanistic-origins-grid-cells
- Schaeffer, Khona, Fiete 2022: https://openreview.net/forum?id=syU-XvinTI1 ; Schaeffer et al. 2023 (NeurIPS,
  arXiv 2311.02316): https://papers.nips.cc/paper_files/paper/2023/hash/4846257e355f6923fc2a1fbe35099e91-Abstract-Conference.html
- Banino et al. 2018: https://www.nature.com/articles/s41586-018-0102-6 ; Cueva & Wei 2018 (ICLR, arXiv 1803.07770)
- Dorrell et al. 2023: https://arxiv.org/abs/2209.15563
- Jayakumar et al. 2019: https://www.nature.com/articles/s41586-019-0939-3
- Hoydal et al. 2019: https://www.nature.com/articles/s41586-019-1077-7
- Lai et al. 2023: https://www.science.org/doi/10.1126/science.adh5206
- Sanders, Wilson, Gershman 2020: https://elifesciences.org/articles/51140
- Jacob et al. 2019 (grid firing on a circular track set by path integration; not used above, relevant to (a)'s loop
  closure): https://www.nature.com/articles/s41467-019-08795-w
- Bordelon et al. 2025, learning dynamics of integration in linear RNNs (adjacent to (a), no rank result): https://arxiv.org/abs/2503.18754
Cited from memory, not re-read: Howard & Kahana 2002; Stachenfeld et al. 2017; Kraus et al. 2013; Mankin et al. 2012;
Pfeiffer & Foster 2013; Carandini & Heeger 2012; Kropff et al. 2015 (context-invariant speed cells); Stensola et al.
2012; Grieves et al. 2021; Hockett 1960; Arora et al. 2018; Feng & Steinhardt 2023. Check before external use.

Searched and not found: a learned path integrator whose velocity read-in rank is varied around D; a cross-field
statement identifying Howard's alpha(t) with selective-SSM / content-dependent rotary increments; a model of
cognitive maps from language that handles mentioned-but-absent content.
