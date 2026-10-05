# Positional encodings as models of the hippocampal formation (2026-10-05)

Theory note, not a result. Nothing was trained or run. Every number of ours is copied from the results file named
beside it (a CORRECTED / AUDIT block supersedes its body). Builds on `docs/theory/2026-10-04/02_literature.md` (grid
theory, Howard's alpha(t), clocks vs maps, asides, leak and gain, memorisation vs map size) and `docs/lit/LIT_*.md`
(attention-side what/where, TEM / TEM-t / CSCG / Vector-HaSH); those are cited, not repeated.

Question: MapFormer (path-integrated phase) and PoPE (content sets non-negative magnitudes, position alone sets phase)
both appeal to a what/where separation the brain is thought to implement. Does MapPoPE, or another encoding tried here,
make sense as a brain model; which is most plausible; what does each correspond to; what would discriminate them?

**Levels of support** (used in every table). **STRUCT** = the same variable or the same equation (a mathematical
identity at the computational level, substrate-neutral). **ANALOGY** = same role, different mechanism or variable.
**EVID-ours** = one of our measurements bears on it (status as in its file). **LIT** = a published neural finding,
abstract (at least) read for this note or for `02_literature.md`. Nothing here is evidence that the brain runs any of
these models.

Terms. **phase** theta_t,c: the rotary angle of channel c at token t. **step** Delta_t: the per-token increment,
theta_t = omega (*) sum_{u<=t} W_out W_in emb(x_u). **magnitude** mu: PoPE's softplus(q), softplus(k) >= 0.
**score** a_ts: pre-softmax logit. **kernel**: the score as a function of displacement for a fixed content pair.
**leak**: step size of non-action tokens relative to action tokens. **MEC / LEC**: medial / lateral entorhinal cortex.
**VCO**: velocity-controlled oscillator (oscillatory-interference models). **CAN**: continuous attractor network.
**rate remapping**: place cells keep their field locations and change their peak rates across contexts; **global
remapping**: field locations change.

## 0. Verdict

1. **The path-integrated phase is the strongest structural counterpart.** MapFormer's theta_c =
   omega_c * sum u_c . v is, term for term, the phase difference that "essentially integrates velocity" in the
   oscillatory-interference model (Burgess, Barry & O'Keefe 2007), i.e. a band cell (Krupic, Burgess & O'Keefe 2012).
   The same variable is the coordinate on a grid module's population torus (Gardner et al. 2022) whatever maintains it,
   so the identity survives the field's move from interference to attractors (section 1).
2. **The attention score is a population-vector overlap.** For one 2-D rotary block, q^T R(dtheta) k is (up to a
   constant) the overlap between the current and a stored activity vector of a ring of cosine-tuned cells; summed
   over channels it is Solstad, Moser & Einevoll's (2006) grid-to-place summation, with the place field centred where
   the key was stored. STRUCT, given cosine tuning (section 1b).
3. **MapPoPE is the best computational-level model of the encodings tried, of one specific thing**: MEC-like phase
   and LEC-like gain converging on a retrieval step, with content allowed to change *how strongly* a stored place is
   recalled but not *which* place, the pattern of rate remapping without field shift (Leutgeb et al. 2005; Fyhn et al.
   2007; Lu et al. 2013). ANALOGY with LIT support, not STRUCT: PoPE's "phase" is grid phase, not theta spike timing.
4. **Its non-negativity is not the biological non-negativity.** In polar form RoPE's amplitudes are also >= 0; what
   softplus removes is content's *phase* (signed mu = content phase in {0, pi}; RoPE = any content phase). So
   Whittington et al. 2023 (non-negative firing + energy -> factorised cells) is not what MapPoPE implements; the
   constraint it implements is **"sensory input acts as a gain on the spatial phase code, never as a phase shift"** --
   a gain-modulation claim (Salinas & Abbott), testable in our models and in data (sections 3.2, 4).
5. **Least plausible as brain models**: index RoPE as space (time, not place; content shifts the kernel), the InEKF
   correction as implemented (content -> absolute phase is impossible on a redrawn map: it has no memory to localise
   with), hierarchy / hourglass (temporal pooling, not multi-scale space).
6. **What a faithful model would add**, in order of expected consequence: noise in the integrator plus correction from
   retrieved memories (hippocampus -> MEC feedback), a velocity read-in segregated or normalised against content,
   a few fixed modules at ratio ~1.4 instead of a dense learned ladder. Our integrator is exact (float cumsum), so the
   reason biology needs attractors and landmark correction is absent in silico. That is a candidate explanation
   (untested; M2) of why the correction lines (Level 1.5, PC, InEKF) were negatives here (section 3.4).

## 1. Two identities that anchor the comparison

### 1a. Path phase = interference phase = coordinate on the module torus (STRUCT)

Burgess et al. 2007 (Hippocampus 17:801): a dendritic oscillation whose frequency rises above somatic theta in
proportion to running speed along a preferred direction d (named a velocity-controlled oscillator, VCO, in Burgess
2008 and Bush & Burgess 2014); "the phase difference between this oscillation and a somatic input at
theta-frequency essentially integrates velocity":
`phi_d(t) - phi_b(t) = 2 pi beta * integral_0^t v(tau) . d dtau`.
MapFormer, per channel c (`model.py:83,120-121`): `theta_t,c = omega_c * sum_{u<=t} (W_out W_in emb(x_u))_c`.
For action tokens, `(W_out W_in emb(a))_c = u_c . v(a)` with u_c a learned direction (a row of the step map restricted
to actions), so `omega_c <-> 2 pi beta`, `u_c <-> d`, `Delta <-> v dt`. Each rotary channel is one VCO-minus-baseline
phase: a band cell, periodic along u_c (Krupic et al. 2012, Science 337:853: MEC/parasubiculum cells "composed of plane
waves (or bands) drawn from a discrete set of orientations and wavelengths"; "path integration is performed by
integrating displacement along a restricted set of directions"). Our own rate maps found exactly this: single channels
are stripes, not hexagons (`HIPPOCAMPAL_ANALYSIS.md`).

Consequences.
- **Index RoPE is the baseline oscillator** (theta_t = omega t): the velocity-independent reference against which the
  VCO phase is measured. As a code it measures elapsed time, as in the striatal beat-frequency model of interval
  timing (Matell & Meck 2004, Cogn Brain Res 21:139: striatal neurons detect coincidences of cortical oscillator
  phases). ANALOGY.
- **Signed vs monotone steps.** A VCO's frequency is f_theta + beta v.d: always positive, yet the *phase difference*
  from baseline is signed, because the baseline is subtracted and the drive is velocity along d. Two ways to lose the
  sign, and the repo has both: (a) drive the oscillator by speed |v.d| instead of velocity -- a monotone step (Abs,
  softplus, CARoPE), whose phase advances with distance travelled in any direction, a path-length clock (Howard Case
  II); (b) read the VCO without subtracting the baseline -- elapsed time (here: moves) plus the map, i.e. RoPE + MapWM, the
  mixture our text-world clock seeds learned (common component parallel to the verb step; `TEXTWORLD_RESULTS.md`).
  Our matched-length sign result (signed 8/8, monotone 0/8 and 1/8, -0.177, p 0.0002; `SIGN_MATCHED_RESULTS.md`, REG)
  says a map needs (a)'s velocity drive. `02_literature.md` sec. 2(b) reads sign as Howard's Case II vs III; this is
  the oscillator-level version (possibly new as a written statement; implicit in Burgess 2007/2008).
- **Interference vs attractor does not discriminate our models.** The field moved from interference to attractors
  because (i) biological oscillators are too noisy for the required phase stability (Zilli et al. 2009, PLoS Comput
  Biol 5:e1000573: "expected stability times far below those seen in experimental recordings of grid cells",
  though small noise is tolerated); (ii) bat grid cells exist "in the absence of continuous theta-band oscillations,
  and with almost no theta modulation" (Yartsev, Witter & Ulanovsky 2011, Nature 479:103); (iii)
  intracellular ramps during field crossings fit attractor models (Domnisoru, Kinkhabwala & Tank 2013, Nature 495:199;
  Schmidt-Hieber & Hausser 2013, Nat Neurosci 16:325 -- who also found intracellular phase precession best fit by
  interference); (iv) module population activity lies on a low-dimensional torus preserved across environments and
  sleep (Yoon et al. 2013, Nat Neurosci 16:1077; Gardner et al. 2022, Nature 602:123). Hybrids exist (Bush & Burgess
  2014, J Neurosci 34:5065). Pulling the other way, septal inactivation (muscimol, Brandon et al. 2011; lidocaine,
  Koenig et al. 2011, Science 332) abolishes grid periodicity, so theta matters in rodents. All of these concern **how the phase is
  maintained against noise**. MapFormer's phase is maintained by exact floating-point summation: it is the noiseless
  limit of both models, and its theta_c is the angular coordinate of the attractor torus as much as it is a VCO phase.
  What MapFormer lacks is the *reason* biology needs either mechanism (noise), not the variable.
- **Prior art for the identity.** GridPE (Li, Wu, Huang, Zhang, arXiv 2406.07049, preprint) grounds positional
  embeddings in VCO theory (`dPhi = (omega_s + beta v . omega) dt`) and notes the construction reduces to RoPE in 1-D;
  it encodes *given* coordinates and does not path-integrate content. Lie-group grid-cell models (Gao et al. 2021,
  NeurIPS; Xu et al. 2025, ICLR, arXiv 2405.16865) use the same rotation algebra without attention. Dorrell et al.
  2023 (ICLR, arXiv 2209.15563) define an "actionable" code, `g(x + dx) = T(dx) g(x)` with T independent of x: MapFormer's
  phase code is actionable with T a block rotation. MapFormer itself cites grid cells only as an analogy for its
  multiple frequencies (App. A.8) and never cites interference models. Not found: a published link between a
  *content-dependent, path-integrated* rotary phase and VCOs or grid modules.

### 1b. The score is a population overlap; summed over channels it is grid-to-place summation (STRUCT given cosine tuning)

A ring of N cells with preferred phases phi_j uniform on [0, 2pi) and rates r_j(theta) = mu (1 + cos(theta - phi_j))
(non-negative) has population overlap
`sum_j r_j(theta_t) r_j(theta_s) = mu^2 N (1 + cos(theta_t - theta_s) / 2)`.
So one rotary block's `q^T R(theta_s - theta_t) k` with content fixed is the current-vs-stored population overlap of a
cosine-tuned ring, up to a constant and the content's own phase offset (RoPE, MapWM) -- and with no content offset
(PoPE, MapPoPE, MapEM's A_P). Summed over channels with weights w_c,
`F_s(x_t) = sum_c w_c cos(omega_c u_c . (x_t - x_s))`
is a field over the query's position centred on the key's stored position x_s: a single place-like field when the
omega_c span several scales (aliases only where every channel realigns, the residue-code range of Fiete, Burak &
Brookings 2008, J Neurosci 28:6858; Mathis, Herz & Stemmler 2012), a periodic grid-like field when one omega
dominates. This is Solstad, Moser & Einevoll 2006 (Hippocampus 16:1026): place fields "formed by linear summation of
appropriately weighted inputs from entorhinal grid cells" of similar phase and diverse spacing.

Reading: **every stored token is a place cell whose field is where it was stored; the query evaluates all of them at
the current position; softmax is their competition; the value read out is what was bound there.** This is TEM-t's
mapping (Whittington, Warren & Behrens 2022, arXiv 2112.04035; `docs/lit/LIT_WHAT_WHERE.md`) derived from the score
rather than asserted (TEM-t sec. 6, read first-hand by a sub-agent: "K and V can simply be seen as weight matrices
between feature neuron (representing the query) and memory neurons (computing the softmax) ... they appear to have a
spatial tuning for each environment resembling hippocampal place cells"; Q = K = position code, MEC-like; V =
stimulus, LEC-like). Two caveats that keep it from being a circuit model: biological place cells are a fixed
population with fixed grid-to-place weights (so a grid realignment remaps them, Fyhn et al. 2007), whereas attention
creates one "place cell" per stored token and reads only *relative* phase (a global phase shift cancels exactly); and
place fields persist in a weakened form when grid periodicity is lost (Koenig et al. 2011), so place coding is not
grid summation alone.

A side reading (possibly new, cheap to check): grid structure in these models should be looked for in the kernel
F(d) restricted to channels of one frequency, not in single channels (which are bands by construction). Three equal-
omega bands 60 deg apart (`model_grid.py`) make a hexagonal kernel by construction; for learned MapWM / MapPoPE the
kernel's symmetry is set by the learned u_c and has never been measured (`HIPPOCAMPAL_ANALYSIS.md` scored channels).

## 2. Each encoding -> its closest biological counterpart

Grades. **A**: STRUCT correspondence for the variable it computes, with direct neural evidence for that variable.
**B**: STRUCT for one half, or ANALOGY with LIT support. **C**: ANALOGY only. **D**: conflicts with established biology
in the respect the encoding is about. A grade is about plausibility as a brain model, not about task performance.

| encoding | computes | closest counterpart | holds | breaks | level | grade | ours |
|---|---|---|---|---|---|---|---|
| RoPE (index) | phase omega*t; content Q,K rotated (content sets amplitude AND phase offset) | baseline theta oscillator; beat-frequency interval timing (Matell & Meck 2004) | a bank of fixed-frequency oscillators encodes elapsed time | position = token count, not elapsed time or movement; content phase psi lets *what* shift *where* per pair; signed real components are not rates (only the polar amplitude is) | ANALOGY (time) | C time; D space | torus 0.805 vs path 0.971 (`PAPER2X2_RESULTS.md`, REG) |
| NoPE | content only, order via causal mask | content-cued autoassociation (CA3-like) without an MEC position input | recall by content alone | no structure code at all; place fields persist without grid periodicity (Koenig 2011), so "no where" is not even the lesioned state | ANALOGY | C | -- |
| PoPE (index) | content -> non-negative magnitudes mu, index -> phase | dual rate/phase coding (O'Keefe & Burgess 2005, Hippocampus 15:853; Huxter, Burgess & O'Keefe 2003) with a temporal phase | rate and phase carry separate information | phase from token count, not movement (in hippocampus phase tracks distance through the field); no map | ANALOGY | C | hurts index navigation -0.126 (`WHAT_WHERE_ANALYSIS.md` sec. 5) |
| MapWM | path phase (sec. 1a) rotating content Q,K | VCO / band cells feeding a conjunctive (content x place) code | the integrated variable (STRUCT) | content sets a per-pair phase offset in the score: content can move where retrieval lands; signed steps read the pre-LayerNorm embedding (leak) | STRUCT (phase) + ANALOGY (score) | B | trained models separate anyway: inter/pos 0.083, peak0 1.000 at r=4 (POST HOC) |
| MapEM | softmax(A_X (*) A_P): content score times position kernel | gain field (multiplicative conjunction, Salinas & Abbott); TEM's p = x (x) g | one shared kernel scaled by content (L3, shape1 1.000) | the gain is signed: A_X < 0 turns the peak into a trough (no firing-rate analogue); product inside the softmax makes content an inverse temperature; search cost | STRUCT (conjunction) | B | EM - WM -0.375 on recency, search (`EM_WM_STATE.md`, REG) |
| MapEM PosOnly (TEM-t-like) | score = A_P only; content in V | TEM-t: Q,K ~ MEC position, V ~ LEC content, softmax ~ hippocampal memory neurons | the published mapping; strictest separation (L5) | no content in retrieval: cannot express rate remapping or content-cued recall ("where did I see X"); TEM-t's own update is recurrent and non-negative, ours is a signed cumsum | STRUCT (mapping) | B | new-object pilot 0.976 (PILOT, n=1) |
| **MapPoPE** | path phase; content -> softplus magnitudes per channel; score sum_c mu^q mu^k cos(dtheta_c - delta_c) | MEC phase + LEC gain converging on CA1 retrieval; rate remapping without field shift | content changes recall strength and kernel sharpness, never the peak (delta at 0 on 91-100% of channels) | magnitudes per frequency also change field WIDTH (not pure rate); step reads content (leak); dense learned omega ladder (ratio 1.07-1.14 per channel) vs few modules at ~1.4 (Stensola 2012); exact cumsum, no noise, no correction; one layer; "phase" is grid phase, not theta spike timing | STRUCT (phase, overlap) + ANALOGY (gain) | **B+** | SCORE +0.024, 16/16 vs 10/16 SOLVED at rank 2 (`MAPPOPE_PAIR_RESULTS.md`, REG) |
| signed step | velocity-driven VCO phase minus baseline | Howard Case III; Burgess VCO | sign = direction along u_c | -- | STRUCT | A | 8/8 vs 0/8 (`SIGN_MATCHED_RESULTS.md`, REG) |
| monotone step (Abs, Pos, softplus) | phase grows with path length | speed-driven (not velocity-driven) oscillator; Howard Case II; time+distance in MEC (Kraus 2015); LEC experience time (Tsao 2018) | an experience clock | cannot cancel: not a map | ANALOGY | B (as clock) | 0/8, 1/8 at matched length |
| rank of W_out W_in | dimension of the velocity subspace a head reads | velocity input to a module (HD x speed, rank 2 in 2-D) | the solved code is D-dim per head, like a 2-torus module | biology's read-in is wired (speed cells, HD cells), not learned from tokens; D+1 is a learnability margin (`02_literature.md` 2a) | ANALOGY | C | rank 2 0/8, 3 6/8, 4 8/8 (REG) |
| Selective-RoPE generator, causal-conv gate | step gated by a 1-4 token window | gating of path integration by active self-motion (efference copy) | passive transport abolishes grid patterns and velocity modulation of theta (Winter et al. 2015, Curr Biol 25:2493; Terrazas et al. 2005) | window limit (fails with cue 6-13 tokens back) has no biological counterpart | ANALOGY | B- | PILOT only (`CTXSTEP_*`) |
| hidden-state step (HS, HSR) | step computed from an earlier attention layer's state | velocity inferred from context by a separate system, then integrated (e.g. language comprehension feeding the map) | right for language ("did not go north") | no identified circuit; two-stage cortex -> MEC is a hypothesis | ANALOGY | C+ | HSR 4/4 learned a step (PILOT) |
| NormStep | step reads LN(emb): scale-invariant | divisive normalisation of the velocity input (Carandini & Heeger 2012) | content intensity cannot set a gain on "where"; speed cells are context-invariant (Kropff 2015) | normalising the input also removes magnitude, i.e. speed, for continuous velocity; shared LN bias is a per-token tick | ANALOGY | B | +0.0107, p 0.0002 (`LEAK_RESULTS.md`, REG; partly speed) |
| ActOnly | steps only on action tokens | anatomical segregation: MEC spatial / self-motion, LEC object input (Hargreaves et al. 2005) | grid task: optimal (+0.0107) | language: asides need non-motion steps (DirOnly 0.970-0.974, 100% of checked errors are aside nouns) | STRUCT (segregation) | A- space; C language | REG; POST HOC (2/8 seeds) |
| forget gate / decay envelope (ALiBi-like) | logit penalty growing with index or path distance | leaky integration: time cells / Laplace transform (Howard 2014), LEC ramping (Tsao 2018) | recency as a real-exponent decay | path-distance version collapses 2-D to a 1-D average; the forget gate's +0.086 is OOD and its mechanism unidentified (anti-correlated with lambda) | ANALOGY | B (time); C (space) | `FORGET_CONTROL.md` (never matched-length) |
| InEKF / Level 1.5 | phase corrected toward z_t = f(token content) with gain K_t | landmark / boundary error correction of grid phase (Hardcastle, Ganguli & Giocomo 2015; Campbell et al. 2018; Ocko et al. 2018) | the idea (fuse self-motion with landmarks, per-cue reliability) | z_t is a fixed function of token identity; on a redrawn map a token carries no absolute location, so it cannot localise; biological correction uses environment-specific learned associations and hippocampal feedback (Bonnevie et al. 2013: hippocampal inactivation extinguishes grid patterns); our integrator has no noise to correct | concept STRUCT, implementation D | D | stabilisation, not inference (`L15_ABLATION.md`) |
| grid variants (3 bands at 60 deg per omega; DoG head) | Burgess's three VCOs per module | 2-D interference model | architecture matches | hexagons never emerged (max grid score 0.258, 0/22 modules): the code is signed, so neither Sorscher's (non-negativity + centre-surround readout) nor Dorrell's (non-negative actionable code) conditions hold; the DoG head added a target but not a non-negative phase population | STRUCT (arch) | B arch / no evidence | live negative |
| hierarchy / hourglass | temporal pooling of tokens, coarse map over pooled positions | multi-scale modules (dorsoventral gradient, Giocomo et al. 2007; Stensola 2012) | "coarse and fine" | biology's scales are parallel modules on the same input, not pooled time; the multi-omega ladder already is the module code | ANALOGY | C/D | efficiency only (`ENWIK8_HIERARCHY.md`) |
| looped block (x4 shared) | recurrent settling within a token | recurrent attractor dynamics / iterative inference (CA3 recurrence; TEM's inference) | weight-shared recurrence | recurrence over iterations, not over time; no noise to settle | ANALOGY | B- | Match-Query +0.346 unpaired (not pre-reg.) |
| MapPoPE pairwise vs 64 angles | angle count per head | number of phase channels per module | count does not matter: +0.0002, CI [-0.0004, +0.0008] | -- | -- | -- | REG |

What the table says. The **phase** half of every path model is STRUCT; the **score** halves differ in exactly the
respect biology has an opinion on: whether content may move where recall lands (RoPE, MapWM: yes; PoPE, MapPoPE,
MapEM: no, up to PoPE's delta and EM's sign) and whether content gates recall at all (PosOnly: no). Biology says
content gates recall (rate remapping, Leutgeb 2005) and mostly does not move fields (rate remapping without grid
realignment, Fyhn 2007; grid cells redistribute rates across fixed fields, Diehl 2017), but sometimes does (grid
fields drawn toward goals, Boccara et al. 2019, Science 363:1443; reward-driven restructuring, Butler, Hardcastle &
Giocomo 2019, Science 363:1447), and content-anchored position is carried by separate populations (object-vector cells
in mouse MEC, Hoydal et al. 2019, Nature 568:400; LEC object and object-trace cells, Deshmukh & Knierim 2011, Tsao,
Moser & Moser 2013). So biology looks like **MapPoPE plus a separate,
content-anchored channel**, not like either pure scheme.

## 3. MapPoPE as a brain model

### 3.1 What maps onto what

| MapPoPE part | counterpart | level |
|---|---|---|
| theta_t,c = omega_c sum u_c . v | MEC grid/band phase; VCO-minus-baseline | STRUCT (sec. 1a) |
| cos(theta_t,c - theta_s,c) | overlap of current and stored ring activity; coherence of the current path state with the state at storage | STRUCT given cosine tuning (sec. 1b) |
| sum_c mu_c cos(.) | grid-to-place summation (Solstad 2006) | STRUCT |
| mu^k_c = softplus(W_k emb(x_s))_c | strength with which the memory at s was stored, per scale: an LEC-set gain bound at encoding | ANALOGY |
| mu^q_c = softplus(W_q emb(x_t))_c | current sensory/context gain on each scale at retrieval | ANALOGY |
| softmax over s | competition among memory-place cells | ANALOGY |
| V(x_s) | the content bound at that place (LEC side) | ANALOGY (TEM-t) |

**Rate remapping.** In MapPoPE, change the content at fixed positions and every key's kernel keeps its peak (with
delta = 0, any non-negative mixture of cosines peaking at 0 peaks at 0; `WHAT_WHERE_ANALYSIS.md` sec. 2) while its
height changes: the definition of rate remapping. Biology: when cues change in a constant place, "the firing rates
of active cells varied, often over more than an order of magnitude, whereas the location of firing remained constant"
(Leutgeb et al. 2005, Science 309:619); "rate remapping is associated with stable grid fields, global remapping is
always accompanied by a coordinate shift" of the grids (Fyhn et al. 2007, Nature 446:190); CA3 rate variation
"depended on inputs from the lateral entorhinal cortex" and LEC lesions impaired rate remapping (Lu et al. 2013, Nat
Neurosci 16:1085; preserved spatial firing is a body result, not checked here). The same split appears inside MEC:
under box shape/colour changes "grid cells retained their spatial alignment and predominantly responded with
redistributed firing rates across their grid fields" (Diehl et al. 2017, Neuron 94:83). MEC is weakly influenced by
objects, LEC strongly (Deshmukh & Knierim 2011, Front Behav Neurosci 5:69; Keene et al. 2016, J Neurosci 36:3660;
Hargreaves et al. 2005, Science 308:1792). So "content -> rate via LEC; place -> phase via MEC" is the textbook
reading, and MapPoPE is its minimal attention form. Where it diverges: (i) mu is per frequency, so content reweights *scales* -- this changes field width
and shape, not only peak rate (MapEM's single gain would change only amplitude before the softmax); (ii) the gain is a
product of query and key gains, so both "what is here now" and "what was here then" gate recall; (iii) MapWM, the
model PoPE is meant to fix, ends at the same separation after training (inter/pos 0.083 vs 0.128; PoPE imposes it at
init, 0.05-0.06 untrained vs 0.98; `WHAT_WHERE_ANALYSIS.md` sec. 6, POST HOC). So at the trained endpoint on the torus
**both** models implement rate-remapping-like retrieval; MapPoPE starts there.

**Rate vs phase coding.** The hippocampal "dual coding" hypothesis (O'Keefe & Burgess 2005) is about spike *timing*
relative to theta (phase precession, O'Keefe & Recce 1993) carrying position within a field, while rate carries other
variables (Huxter et al. 2003, Nature 425:828: firing time codes location in the field, rate codes running speed, "two
independent variables"; contested by Mehta, Lee & Wilson 2002, who find the two "highly correlated", phase derived
from rate). MapPoPE's phase is not spike timing; it is the grid phase
(position on the module torus). The two are linked only through interference models, in which precession arises from
the same VCO phase (Burgess 2007), and MEC grid cells precess independently of the hippocampus (Hafting et al. 2008).
So "content -> magnitude, path -> phase" matches dual coding as a *division of labour*, ANALOGY; it is STRUCT only for
grid phase.

### 3.2 What softplus non-negativity buys, biologically and in our data

- **Polar form.** Any 2-D block is amplitude >= 0 times a phase. RoPE's content sets amplitude AND phase psi; signed
  magnitudes (PoPE "NoSigma") set amplitude and a phase in {0, pi}; softplus sets amplitude only. So softplus is
  **"content contributes no phase"**, not "firing rates are non-negative". A ring of non-negative cells carries a
  non-negative amplitude whatever the model does.
- **Therefore Whittington et al. 2023 (ICLR, arXiv 2210.01768) does not transfer directly.** Their Theorem 1 needs
  independent bounded factors, non-negative *activity*, fixed total variance and minimal activity energy; Theorem 2
  adds weight energy for an exact linear readout. Then neurons become selective for single factors; when objects are
  fixed across contexts the task is entangled and grid fields warp toward them (their account of goal attraction,
  Boccara 2019). In a 1-layer MapPoPE the magnitudes are functions of the token alone, so they are single-factor by
  construction, not by that mechanism. The theorem would bite in a deeper model, where the residual stream carries
  position: there it predicts that non-negative magnitudes plus an activity penalty stay position-free, and that
  without the penalty non-negativity alone is not enough. Unmeasured (no multi-layer probe; sec. 4, M4).
- **What it does correspond to: gain modulation.** "Sensory content scales the contribution of each spatial scale and
  never shifts its phase" is a multiplicative gain field in the sense of Salinas & Abbott (1995, J Neurosci 15:6461,
  gain-modulated sensory arrays for coordinate transformation; 1996, PNAS 93:11956; review title: Salinas & Thier
  2000, Neuron 27:15, "Gain modulation: a major computational principle of the central nervous system"), applied per
  module.
- **Where biological non-negativity does matter: emergence.** Dorrell et al. 2023: non-negative firing + bounded
  activity + precise coding on an actionable code give lattice (hexagonal) tuning within a module, and multiple modules
  from the conflict between non-negativity (harmonics within a module) and the coding objective. Sorscher et al.
  (2019 NeurIPS; 2023 Neuron 111:121): hexagons need non-negativity AND a centre-surround place readout. MapFormer's
  phase code is actionable but signed (complex), consistent with no lattice emerging here (`HIPPOCAMPAL_ANALYSIS.md`);
  its module structure is architected (the dense omega ladder). A MapFormer whose phase is read through rectified ring populations would be the
  model in which Dorrell's result could apply.
- **What it buys in our data**: SCORE +0.0243 (perm p 0.012), 16/16 vs 10/16 SOLVED at rank 2, T=128; the angle count
  adds nothing (+0.0002, CI [-0.0004, +0.0008]) (`MAPPOPE_PAIR_RESULTS.md`, REG). The SCORE bundle (non-negative
  magnitudes, no content phase except via untied deltas, positive score mean) is **not separated**: whether the
  biological ingredient (gain-only content) is the operative one is open (M1 below). Out of distribution the lead
  grows (T=1024 +0.099), robustness, not capability (rule 10).
- Reading: the constraint acts as a prior that puts the model in the separated regime from the start; the end state
  is reachable without it (MapWM r=4). Biologically: anatomy that forces content to act as gain would buy faster or
  more reliable learning of a map, not a different map. ANALOGY; the speed part is consistent with the leak result
  (remedies 16/16 SOLVED vs MapWM still descending; r(loss, acc) -0.944).

### 3.3 What the score corresponds to

`cos(theta_t,c - theta_s,c - delta_c)` is the coherence between the path state now and the path state when s was
stored -- equivalently the overlap of the two population vectors of a cosine-tuned ring (sec. 1b). It is not LFP
phase coherence between regions, and not interference in the Burgess sense (that compares a VCO with a baseline at
one moment; here two moments of the same integrated phase are compared). The weighted sum over channels is
grid-to-place summation; with several incommensurate omega it is the residue code's decoder (Fiete 2008): a stored
place is recalled when every module's phase matches. delta_c would be a per-scale phase offset between storage and
recall (an object-vector-like displacement if it were content-dependent; in PoPE it is a content-free constant and sits
at its bound 0 on 91-100% of torus channels).

### 3.4 A more faithful variant, and what each change predicts

| change | biological motivation | prediction in our setting | cost |
|---|---|---|---|
| noise in the integrator + correction from retrieval (attention output -> phase correction, via a second pass or the loop) | path-integration drift (Burak & Fiete 2009; Zilli 2009); boundary/landmark error correction (Hardcastle 2015; Campbell 2018; Ocko 2018); hippocampal drive needed for grids (Bonnevie 2013) | with sigma = 0: nothing (consistent with every correction negative here); with sigma > 0: retrieval correction beats no correction and beats Level 1.5, gap growing with sigma | new module + 24-32 runs, ~5 h |
| velocity read-in segregated (ActOnly) or normalised (NormStep) | MEC / LEC segregation (Hargreaves 2005); context-invariant speed cells (Kropff 2015); divisive normalisation | measured: +0.0107 on the new-object task; nothing on the text world | done |
| few fixed modules (M = 4-6, ratio ~1.4, 2-3 bands per module at 60 deg) instead of 32-64 learned omega | discrete modules (Stensola 2012); oscillation frequency gradient (Giocomo 2007); residue capacity (Fiete 2008; Mathis 2012) | works iff the residue range covers the revisit distances; the angle-count null says the dense ladder is not needed | `model_grid.py` + `model_fixed_omega.py` exist; ~16-24 runs |
| per-module non-negative gain instead of per-channel | one gain per module input | narrower class than PoPE; if it matches PoPE, the scale-reweighting freedom is unused | ~16 runs |
| recurrent, non-negative position update (TEM-t's e_{t+1} = ReLU(e_t W_a)) instead of cumsum | CAN / recurrent MEC | no gain without noise; loses the parallel scan (TEMFaithful 120x slower, `TIMING_BENCHMARK.md`) | expensive; only after the noise arm fires |
| a separate content-anchored offset channel (delta_c made content-dependent, in its own heads) | object-vector cells (Hoydal 2019); goal attraction (Boccara 2019; Butler 2019) | helps tasks that need "the cell east of the key", hurts nothing else if confined to its heads | ~16 runs, needs a task that asks for it |

## 4. Predictions

### Neural (what each candidate predicts that the others do not)

**N1. Shape of rate remapping (MapPoPE vs MapEM vs MapWM).** Under MapPoPE-type retrieval, a content change reweights
spatial *scales*: rate remapping should come with field-width/shape changes, and the changes should covary across
place cells receiving from the same grid modules. Under MapEM-type (one content gain on a shared kernel) only the
amplitude changes (field shape preserved before the output nonlinearity). Under MapWM-type (content phase offset) field
centroids shift with content -- partial remapping. Readout: in rate-remapping data (Leutgeb-type colour/odour context
changes in a fixed enclosure), regress field-width change and centroid shift on peak-rate change. We did not find a
paper that reports field width under rate remapping specifically; possibly new.

**N2. Immediate phase shift from a non-spatial event (the leak).** `02_literature.md` 2(d) predicted the intensity
dependence; the three read-in designs separate further: ActOnly / TEM-t -> no shift; NormStep -> a small shift that
depends on the stimulus identity but not its intensity; MapWM -> a shift proportional to intensity, along a fixed
low-dimensional set of directions (the step map's row space). The discriminating feature against known biology:
goal/reward distortions of grids (Boccara 2019; Butler 2019) build up with learning over sessions; a leak is
per-encounter and immediate. Head-fixed VR with matched optic flow, a salient odour or sound pulse in darkness,
grid-phase decoded from a module population (Gardner-style torus coordinates) before vs after the pulse. Current
evidence constrains but does not decide it: MEC cells are "weakly influenced by the objects" (Deshmukh & Knierim 2011;
Keene 2016), consistent with a segregated or normalised read-in; no single-unit study adding or moving an object and
measuring grid phase was found (sub-agent search; one low-profile device paper, Xu et al. 2022, Microsyst Nanoeng 8:104,
reports landmark-object-induced grid rotation/scaling, not used).

**N3. Non-local content does not move the current phase (asides).** Our text-world result: words about absent things
must be bound at an "elsewhere" phase, and learned steps do this (0 aside errors on 2 solved seeds; DirOnly 100% of
78/78 and 89/89 errors are aside nouns; POST HOC). Prediction: during imagined or remote representation (grid-like
signals during imagined navigation, Bellmund et al. 2016, Horner et al. 2016; remote place activation, Lai et al. 2023)
the population phase of the *current-location* module code is unchanged, and an object imagined at a remote place is
later reinstated with that remote place's code, not the current one. ActOnly-type models predict no binding of
imagined content at all; MapWM-type predicts binding at the current phase plus a consistent offset.

**N4. A common-mode velocity component must cancel exactly at rank D (speculative, possibly new).** The rank theory
(`docs/theory/2026-10-04/00_PLAN.md` T1, theorem about end states, not causal) says a D-dimensional unit cannot carry a
step component shared by every move without drifting (CLOCK) or losing a dimension (COLLAPSE). In interference models
the shared component is real: theta frequency rises with running speed in every direction (Burgess 2008, Hippocampus
18:1157, Type I theta), and the VCO-minus-baseline subtraction must cancel it. Since a module's population lies on a
2-torus (Gardner 2022), the brain must cancel it exactly. Prediction: a manipulation that changes the speed-theta
frequency slope without changing speed coding should produce grid drift proportional to distance run (a clock
failure), not loss of one axis. Weak: depends on interference being part of the mechanism.

### ML (cheap, in this repo)

**M1. Split MapPoPE's SCORE bundle by the biological ingredient.** MapPoPE r=2 on the paper torus, T=128, 16 seeds per
arm: softplus (current), signed magnitudes (NoSigma: content phase {0, pi}), ReLU, delta frozen at 0. Decides whether
"content as gain only" carries SCORE's +0.024 and 16/16 vs 10/16. `model_pope_ablate.py` has the arms for index PoPE;
port to the path phase. ~48 runs, ~3-3.5 h (pair batch: 72 runs, 3.5-4.5 h).

**M2. Noise + retrieval correction** (sec. 3.4 row 1). Gaussian noise on each step at train and eval, sigma in 3 levels;
arms: MapPoPE plain, Level 1.5 (content -> phase), retrieval-corrected (phase nudged toward the phase of the
attention-weighted keys whose content matches the current observation). Matched length. Decides whether the brain's
reason for attractors + landmark feedback is the missing ingredient behind all our correction negatives. ~5 h.

**M3. Remapping-type probe on stored checkpoints (CPU, minutes).** Extend `docs/audits/2026-09-27/probe_whatwhere.py`:
for each of the 68 content pairs, kernel peak location, peak height, half-width; report how much of the content x
position interaction (8-13% of score variance) is amplitude, width, or peak shift, for MapWM r2/r4, MapPoPE r2/r4,
MapEM r4 (`runs/paper2x2/p0`, `runs/dof/torus`). Gives N1 its model-side numbers. Cannot be null (a decomposition).

**M4. Non-negativity at depth (CPU if checkpoints exist).** Probe position dependence of layer-2 magnitudes in
2-layer MapPoPE vs MapWM's layer-2 Q/K content; with and without an activity L1 penalty (training needed for the
penalty arm). Tests whether Whittington 2023's mechanism operates once the residual stream carries position.

**M5. Kernel symmetry (CPU).** Grid score of per-frequency-group kernels F(d) (sec. 1b) on stored 2-D checkpoints.
Reframes the hexagon negative; on a square torus with four actions a square kernel is expected, so this is
descriptive.

### Ranked: the three most informative

1. **M3 (CPU, minutes) feeding N1.** Says which remapping type each trained model implements and turns N1 into a
   quantitative neural prediction. Cheapest; cannot fail to produce a number.
2. **M1 (~3 h).** The only test of whether the biologically motivated ingredient (content may gate but not shift) is
   what makes MapPoPE better. If NoSigma matches softplus, "non-negativity" is not the operative word and the
   gain-only claim must be stated as "no continuous content phase".
3. **M2 (~5 h).** The one feature every brain model has and none of ours does (noisy integration corrected by memory).
   A positive result would also re-explain the Level 1.5 / PC / InEKF negatives as a property of the noiseless regime.

Neural, ranked: N1 (needs only existing rate-remapping data), N2 (a clean intensity x identity design), N3.

## 5. Honesty: what kind of claim each is, and novelty

| claim | level | novelty |
|---|---|---|
| path phase = VCO-minus-baseline phase = band cell = torus coordinate | STRUCT | VCO <-> RoPE for given coordinates is published (GridPE, 2406.07049); MapFormer only says "like grid cells that fire at different frequencies" (`papers/txt/mapformer.txt:1556`). For a content-dependent path-integrated rotary phase: possibly new as a written statement, trivial once seen |
| index RoPE = baseline oscillator / beat-frequency timer | ANALOGY | possibly new as a statement; trivial once 1a is accepted |
| score = population overlap; channel sum = Solstad summation; tokens = place cells | STRUCT given cosine tuning | the token-as-memory-neuron mapping is TEM-t's; the derivation from the rotary score is possibly new |
| MapPoPE = rate remapping without field shift (LEC gain x MEC phase) | ANALOGY + LIT | possibly new; PoPE's paper uses "what and where" only as attention vocabulary (`papers/txt/pope.txt:15-25`), no neuroscience |
| softplus = "no content phase", not biological non-negativity | STRUCT (algebra) | possibly new; corrects the obvious reading |
| sign = velocity (not speed) drive of a baseline-subtracted oscillator; text-world clock seeds = VCO read without baseline | STRUCT (first), ANALOGY (second) | sign result is a replication (Sarrof, Grazzi, SRoPE); the oscillator reading possibly new |
| Level 1.5 cannot localise on redrawn maps by construction | STRUCT (argument) | ours; consistent with `L15_ABLATION.md` |
| N1-N4 | predictions | untested in animals; N1, N4 possibly new; N2, N3 extend `02_literature.md` |

Not claimed: that MapFormer learns grid cells (live negative); that any model here is a circuit model; that the
interference/attractor debate bears on our results; any biological prediction as supported by our data.

## Sources

Read for this note (abstract at least; PubMed / publisher records), in addition to `02_literature.md`'s list:
Burgess, Barry & O'Keefe 2007, Hippocampus 17:801 (10.1002/hipo.20327); Hasselmo, Giocomo & Zilli 2007, Hippocampus
17:1252; Blair, Welday & Zhang 2007, J Neurosci 27:3211; Welday et al. 2011, J Neurosci 31:16157; Burgess 2008,
Hippocampus 18:1157 (10.1002/hipo.20518); Burak & Fiete 2009, PLoS Comput Biol 5:e1000291; Zilli et al. 2009, PLoS
Comput Biol 5:e1000573; Yartsev, Witter & Ulanovsky 2011, Nature 479:103; Domnisoru, Kinkhabwala & Tank 2013, Nature
495:199; Schmidt-Hieber & Hausser 2013, Nat Neurosci 16:325; Bush & Burgess 2014, J Neurosci 34:5065; Yoon et al.
2013, Nat Neurosci 16:1077; Gardner et al. 2022, Nature 602:123; Stensola et al. 2012, Nature 492:72 (the ~1.42 ratio
is a body result, not in the abstract); Giocomo et al. 2007, Science 315:1719; Fiete, Burak & Brookings 2008, J
Neurosci 28:6858; Mathis, Herz & Stemmler 2012, Neural Comput 24:2280 and Phys Rev Lett 109:018103; Sreenivasan &
Fiete 2011, Nat Neurosci 14:1330; Hardcastle, Ganguli & Giocomo 2015, Neuron 86:827; Campbell et al. 2018, Nat
Neurosci 21:1096; Ocko et al. 2018, PNAS 115:E11798; Brandon et al. 2011, Science 332:595; Koenig et al. 2011, Science
332:592; Hafting et al. 2008, Nature 453:1248; Jayakumar et al. 2019, Nature 566:533; Kropff et al. 2015, Nature
523:419; Winter et al. 2015, Curr Biol 25:2493; Krupic, Burgess & O'Keefe 2012, Science 337:853; Solstad, Moser &
Einevoll 2006, Hippocampus 16:1026; O'Keefe & Burgess 2005, Hippocampus 15:853; Matell & Meck 2004, Cogn Brain Res
21:139; Bonnevie et al. 2013, Nat Neurosci 16:309; Leutgeb et al. 2005, Science 309:619; Fyhn et al. 2007, Nature
446:190; Lu et al. 2013, Nat Neurosci 16:1085; Diehl et al. 2017, Neuron 94:83; Hargreaves et al. 2005, Science
308:1792; Deshmukh & Knierim 2011, Front Behav Neurosci 5:69; Keene et al. 2016, J Neurosci 36:3660; Tsao, Moser &
Moser 2013, Curr Biol 23:399; Tsao et al. 2018, Nature 561:57; O'Keefe & Recce 1993, Hippocampus 3:317; Skaggs et al.
1996, Hippocampus 6:149; Huxter, Burgess & O'Keefe 2003, Nature 425:828; Mehta, Lee & Wilson 2002, Nature 417:741;
Salinas & Abbott 1995, J Neurosci 15:6461; Salinas & Abbott 1996, PNAS 93:11956; Salinas & Thier 2000, Neuron 27:15
(no abstract; title only); Carandini & Heeger 2012, Nat Rev Neurosci 13:51; Boccara et al. 2019, Science 363:1443;
Butler, Hardcastle & Giocomo 2019, Science 363:1447; Hoydal et al. 2019, Nature 568:400; Lai et al. 2023, Science
382:566; Bellmund et al. 2016, eLife 5:e17089; Horner et al. 2016, Curr Biol 26:842; Bicanski & Burgess 2018, eLife
7:e33752; Terrazas et al. 2005, J Neurosci 25:8085; MacDonald et al. 2011, Neuron 71:737. Full text (sub-agent, key
passages quoted above): Whittington, Warren & Behrens 2022 (arXiv 2112.04035); Whittington, Dorrell, Ganguli & Behrens
2023 (arXiv 2210.01768); Dorrell et al. 2023 (arXiv 2209.15563); Sorscher et al. 2019 (NeurIPS) and 2023 (Neuron
111:121); GridPE (arXiv 2406.07049). Abstract: Gao et al. 2021 (NeurIPS); Xu et al. 2025 (arXiv 2405.16865); TEM
(Whittington et al. 2020, Cell 183:1249); Howard et al. 2014 (J Neurosci 34:4692, body). Local: `papers/txt/mapformer.txt`
(v4), `papers/txt/pope.txt` (no neuroscience motivation: "what" and "where" are attention vocabulary only).
