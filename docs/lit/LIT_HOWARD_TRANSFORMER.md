# Howard's alpha(t) and rotary position: targeted prior-art search (2026-10-12)

Literature review, not a result. Nothing was run. Follow-up to `LIT_CLOCK_MAP.md` (do not repeat it; its
Howard 2014 / TEM-t / Sarrof / Grazzi / SRoPE / Mamba-3 / CoPE / CARoPE entries stand).

**Candidate identification under test.** Howard et al. 2014 drive one bank of leaky integrators
dF/dt = alpha(t) (-s F + f) by a rate alpha(t): constant = time (Case I), speed |v| = path length (Case II),
signed velocity v = position (Case III). Claim: the three rotary families are these three cases with
imaginary s -- index RoPE = Case I, monotone content-dependent phase (CoPE, CARoPE, GRAPE-AP, softplus) =
Case II, signed content-dependent phase (MapFormer, Selective RoPE, Mamba-3 data-dependent RoPE) = Case III
-- and Howard's alpha(t) is the same quantity as the selective-SSM step Delta_t and MapFormer's step.

Read status: **first-hand** = PDF text downloaded and the relevant passages read this session;
**abstract** = abstract / search summary only; **not accessible** = could not be fetched.

## 1. Verdict (blunt)

**Not found stated anywhere, in full or in the specific form.** The pieces exist separately and some are
close:

- The Howard lab's own ML work (DeepSITH, SITHCon, SITH-RNN, CogRNN, Time-Local Transformer, Gradual
  Forgetting) keeps s **real** and alpha **fixed or externally given**. Two of their papers explicitly reject
  complex s: "Instead of s being a complex value composed of real and imaginary parts, we restrict s to a
  positive real value" (Kabir et al. 2025; same sentence in Dickson et al. 2025), and "The Laplace transform
  defines s as a complex variable. This choice would result in exponentially growing and oscillatory neural
  activity, causing numerical instabilities" (Mochizuki-Freeman et al. 2023, fn. 1). So the lab connected
  Laplace memory to transformers and SSMs only on the decay (real) side, and set aside the imaginary
  (rotary) side the identification needs.
- The Tiganj lab states the general principle "alpha = dz/dt from the environment represents z" (velocity ->
  distance, count change -> numerosity) and trains networks with velocity-modulated decay (Alipour et al.
  2025; Mochizuki-Freeman et al. 2023). That is Howard's framework applied in deep nets, with alpha given, not
  learned from token content, and real s. No rotary, no attention position, no sign axis.
- Sarkar & Howard 2026 (SITH-RNN) state that SITH-RNN's diagonal recurrence is "formally analogous to
  discretized SSMs" (S4, HiPPO, Mamba). That is the closest published Howard-to-SSM bridge. It maps the
  decay spectrum s to the SSM A, not alpha(t) to Delta_t, and does not touch selectivity, phase or sign.
- GridPE (Li et al. 2024/2026) derives RoPE as the 1-D case of a velocity-controlled oscillator,
  Phi(t) = omega_s t + beta omega^T x(t): a baseline clock plus a velocity (Case III) term. Closest
  RoPE-to-neuroscience bridge. Velocity and position are given coordinates, not content-dependent; no Case II;
  no Howard; no SSM.
- RoMem (Li et al. 2026, EMNLP) cites Howard 2014 to motivate continuous phase rotation for time and makes
  the rotation speed content-dependent (alpha_r = sigma(MLP(text embedding)) in (0,1)). That is a monotone
  content-gated phase rate motivated by Howard -- a Case II instance in a temporal-KG model -- without the
  Case I/II/III mapping, without attention position encoding, and without the sign axis.
- TIDES (Soydan et al. 2026) discusses Mamba's Delta as a learned stand-in for the physical time step and
  suggests Delta "encodes any meaningful continuous parameter along which the sequence evolves ... arc
  length" -- Case II in SSM words, with no Howard citation and no signed case.

**What remains novel**: the explicit three-way identification (constant / non-negative / signed
content-dependent rate = Howard Cases I / II / III) for rotary attention; alpha(t) = Delta_t = MapFormer's
content-dependent angle step; and the observation that the rotary (imaginary-s) slot is where a signed rate
is stable, which is exactly the move the Howard lab declined. It is a short theoretical remark. Once Howard
2014, Mamba and Selective RoPE are side by side, a reviewer can call it obvious; it is not a result.
Cite Howard 2014 for the framing, GridPE for VCO -> RoPE, Sarkar & Howard 2026 for SITH ~ SSM, and the
Tiganj-lab papers for the real-s choice.

## 2. Entries

### (A) Howard / Tiganj / Shankar / Jacques lab work touching ML

| work | id / venue | read | what it says | states the identification? |
|---|---|---|---|---|
| Jacques, Tiganj, Howard, Sederberg 2021, DeepSITH | 2104.04646, NeurIPS 2021 | abstract (+ grep of text: no transformer/positional/SSM/velocity hits) | stacked SITH (log-spaced real-decay time cells) layers vs LSTM on long-delay tasks | not |
| Jacques, Tiganj, Sarkar, Howard, Sederberg 2022, SITHCon | 2107.04616, ICML 2022 | abstract | conv over a log-time axis; invariance to time rescaling | not |
| Sarkar & Howard 2026, "Hierarchical temporal receptive windows and zero-shot timescale generalization in biologically constrained scale-invariant deep networks" (SITH-RNN) | 2601.02618 | first-hand (grep + Discussion via fetch) | SITH-RNN's diagonal recurrence "formally analogous" to discretized SSMs (Fig. 8, S4); SSMs use polynomial (HiPPO) bases, SITH a geometric spectrum; Mamba lacks the scale-invariant prior. Eigenvalues "tile the real axis"; no complex rates | partly (Laplace bank ~ SSM state); not alpha = Delta_t, not rotary, not sign |
| Daniels & Howard 2025, "Continuous attractor networks for Laplace Neural Manifolds" | 2406.04545, Comput Brain Behav 8:392 | first-hand (grep) | edge/bump attractor whose speed sets the Laplace variable; notes the same circuit driven by spatial velocity gives border cells; alpha(t) can be evidence | not (neuroscience only) |
| Howard & Hasselmo 2020, "Cognitive computation using neural representations of time and space in the Laplace domain" | 2003.11668 | first-hand (sections on space, alpha, operators) | restates Cases: alpha = signed velocity -> border cells / BVCs; alpha = speed -> trajectory / splitter cells; "alpha(t) can be externally set" for flow control; data-independent **translation operator R_delta is diagonal** in the Laplace domain (cf. RoPE's relative-position property, but real s); generalising beyond time "require[s] ... gain control of decay by factors such as running velocity" | not (no ML; the RoPE parallel of R_delta is unstated) |
| Howard, Esfahani, Le, Sederberg 2025, "Learning temporal relationships between symbols with Laplace Neural Manifolds" | 2302.10163, Comput Brain Behav 8:211 | first-hand (grep) | associative learning over Laplace-coded time; mentions TEM; no transformer/positional/SSM | not |
| Sarkar, Wang, Zuo, Howard 2024/25, "What x When working memory representations using Laplace Neural Manifolds" | 2409.20484, eLife RP 108804 | first-hand (grep) | conjunctive what x when; edge/bump speed sets time code | not |
| Mochizuki-Freeman, Maini, Tiganj 2023, "Characterizing neural activity in cognitively inspired RL agents during an evidence accumulation task" | IJCNN 2023 | first-hand (method section) | Laplace RNN with alpha = rate of change of evidence; "if alpha = dx/dt is velocity, then the network would represent traveled distance x"; footnote rejects complex s (instability) | partly (alpha as a learned-from-environment rate driving one integrator bank); real s, no attention/rotary |
| Alipour, James, Brown, Tiganj 2025, "Self-supervised learning of scale-invariant neural representations of space and time" | J Comput Neurosci 53:131 | first-hand (sec. 2.2.1) | dF/dt = v(t)(-sF + f): v = 1 gives time cells, signed v gives place cells; generalises to alpha = dz/dt; velocity as modulator (not input) is what yields scale-invariant place fields. "real-value transform since s is real, not complex" | partly (Cases I and III in a trained net; no Case II contrast, no ML-position-encoding link) |
| Kabir, Mochizuki-Freeman, Tiganj 2025, "Deep RL with time-scale invariant memory" (CogRNN) | 2412.15292, AAAI 2025 | first-hand (grep) | real-s Laplace memory + attention readout; explicitly restricts s to positive reals | not |
| Dickson, Mochizuki-Freeman, Kabir, Tiganj 2025, "Time-Local Transformer" | Comput Brain Behav 9:78; PMC13262383 | first-hand (via fetch) | Laplace + inverse timeline fed to multi-head attention instead of a fixed window; s fixed, real; S4/Mamba cited only as "recent success"; Howard 2014 cited for time/place cells | not |
| Dickson & Tiganj 2025, "Gradual forgetting: logarithmic compression for extending transformer context windows" | 2510.22109, NeurIPS 2025 CogInterp workshop | first-hand | SITH filter bank compresses distant history, concatenated before a standard transformer (learned PE); cites Howard 2014 | not |
| Bajaj, Mistry, Maini, Aggarwal, (Dickson), Tiganj 2025/26, temporal biases / induction heads | 2510.22752 (EACL 2026); 2604.01094 | first-hand (grep: no RoPE/Laplace/Howard-2014 hits) | free-recall (TCM/CMR) signatures in transformers and SSMs | not |
| Howard 2023, "Formal models of memory based on temporally-varying representations" | 2201.01796, handbook chapter | first-hand (grep) | Laplace memory, TCM review | not |
| Howard, Fotedar, Datey, Hasselmo 2005, TCM in spatial navigation | Psychol Rev 112:75 | secondary (search summary) | TCM's integrator given velocity describes entorhinal spatial cells: earliest velocity-driven version | not (precursor of Case III) |

### (B) ML papers that cite Howard 2014 / Shankar & Howard 2012, or relate SSM state/step to time cells

Citation sweep: OpenAlex citers of Howard 2014 (202) and of Shankar & Howard 2011/2012 + Howard et al.
2014 "A distributed representation of internal time" (233, deduplicated); Semantic Scholar citers of Howard
2014 (189). Titles filtered for transformer / attention / position / SSM / Mamba / RNN / deep / embedding;
every ML hit grepped first-hand. The only ML citers are the Howard/Tiganj-lab papers in (A) plus:

| work | id / venue | read | what it says | states the identification? |
|---|---|---|---|---|
| Li, Zhang, Yang, Ma, Guo 2026, "Time is not a label: continuous phase rotation for temporal knowledge graphs and agentic memory" (RoMem) | 2604.11544, EMNLP 2026 | first-hand (sec. 3.2-3.4) | time as continuous phase rotation, citing Howard 2014 ("hippocampus encodes time as continuous geometric trajectories"); angle theta_r(tau) = s alpha_r tau omega with RoPE-like omega_i = b^(-i/kd); per-relation **content-dependent speed** alpha_r = sigma(MLP(text emb.)) in (0,1) | partly: a Howard-motivated, content-gated, monotone (Case II-like) phase rate. Not attention position, no Case mapping, no sign, no Delta_t |
| Momennejad et al. 2024, "Memory, space, and planning: multiscale predictive representations" | 2401.09491 | first-hand (grep) | multiscale SR ~ Laplace transform of future (Momennejad & Howard 2018) | not |
| Ji-An, Zhou, Benna, Mattar 2024, "Linking in-context learning in transformers to human episodic memory" (not a citer; found by topic search) | 2405.14992, NeurIPS 2024 | first-hand (grep) | induction heads ~ CMR (Howard-Kahana TCM family); positional encoding mentioned only as needed for K-composition | not |
| Lu, Zhang, Hu, Lee, Kim, Lu 2025, "State space models naturally produce time cell and oscillatory behaviors ..." | 2507.13638 | first-hand (grep) | trained S5: complex-diagonal rotation during delays projects to time-cell-like units | not (no Howard/Laplace citation; no step-size or position link) |
| Soydan, Bessa, Mohr, Barreira 2026, TIDES | 2605.09742 | first-hand (abstract, Discussion) | input-dependent Delta breaks Delta = physical time; moves selectivity to Re(Lambda); Delta could be "arc length" or any continuum | partly (Case I vs Case II for Delta, in SSM language); no Howard, no sign, no rotary position |
| Cao 2026, "Optimal decay spectra for linear recurrences" | 2604.07658 | first-hand (grep) | power-law kernel via Laplace integral -> optimal decay spectrum (real s), independently of Howard | not (parallel to Shankar & Howard's spectrum argument; not cited) |
| Mamba-3 citers checked (Semantic Scholar, 106 titles; grepped: Shared Phase and Retention Control 2609.39082, Input-Dependent Readout Phase 2610.12496, Length-Independent State Tracking 2609.32447, Prospective Coding 2609.04134, Voltic 2610.05700, ReCAT 2609.35200) | -- | first-hand (grep) | no Howard / Laplace-time-cell / hippocampus / path-integration link; 2609.39082 has an input-dependent "phase clock" in an SSM (engineering, no neuroscience) | not |

### (C) Time cells / grid cells / path integration connected to RoPE

| work | id / venue | read | what it says | states the identification? |
|---|---|---|---|---|
| Li, Wu, Huang et al., GridPE | 2406.07049 (v3 Jun 2026) | first-hand (secs. 1, 3) | VCO: d Phi/dt = omega_s + beta v^T omega, Phi = omega_s t + beta omega^T x(t); grid code = sum of Fourier terms; "in the one-dimensional scenario, this expression naturally reduces to RoPE" | partly: RoPE = phase of a velocity-controlled oscillator (Case III with given coordinates; omega_s t is a Case I clock term). No Howard, no Case II, no content-dependent rate, no SSM |
| Whittington, Warren, Behrens 2022 (TEM-t) | 2112.04035, ICLR 2022 | first-hand (corpus, see LIT_CLOCK_MAP) | recurrent (path-integrated) position encodings in a transformer = TEM; index PE "linear structure" | partly (index = clock, PI = space); no Laplace/Howard cases, no monotone case |
| Rambaud, Mascarenhas, Lakretz, MapFormer | 2511.19279 v4 | first-hand (corpus) | content-dependent cumsum angle; fixed angle = RoPE; grid-cell analogy; CoPE cannot represent inverse actions | partly (Case III vs Case I, Case II's failure); no Howard citation (its only "Howard" is L. R. Howard 2014 Curr Biol), no Delta_t |
| Gao, Xie, Wei, Zhu, Wu 2021; Xu et al. 2025 (conformal isometry) | 2006.10259; 2405.16865 | abstract | grid cells as rotation-group path integration of given velocity | not (no attention/RoPE, no Laplace) |
| Zhao, Xu, Kong, Zhang, Wu 2025, "Place cells as multi-scale position embeddings" | 2505.14806 | first-hand (grep) | place cells as random-walk transition-kernel embeddings | not |
| Phase-coding / theta-precession <-> RoPE | -- | searched | nothing found connecting hippocampal phase coding to RoPE beyond GridPE | -- |
| Dorrell et al. 2023, actionable representations | 2209.15563 | first-hand earlier (LIT_WHAT_WHERE) | g(x+dx) = T(dx) g(x): an actionable code; MapFormer's phase code is one | not (no time/Laplace axis) |
| Yu, Wang, Balasubramanian 2026, "When and where: a model hippocampal network unifies formation of time cells and place cells" | 2604.00036; bioRxiv 2026.03.22.713480 | abstract (bioRxiv full text 429) | time and place cells as two regimes of one CA3 RNN; contrasts with leaky-integrator (Laplace) time-cell models | not (neuroscience; argues against the single-integrator account) |

### (D) Taxonomies of positional encodings

| work | id | read | what it says | states the identification? |
|---|---|---|---|---|
| GRAPE (Zhang et al.) | 2512.07805 | first-hand (corpus) | multiplicative (rotation) vs additive (decay) slots; contextual additive slot needs g(x) >= 0 | not (algebraic, no constant/monotone/signed taxonomy in neuroscience terms) |
| Puranik (Jane Street blog) | 22 Apr 2026 | first-hand (corpus) | decay vs rotation slots; "learn how far to advance time" | partly (Delta as a learned time rate); no Howard, no sign |
| CoPE survey-level groupings (GRAPE's "content-adaptive"; EMNLP 2024 PE survey) | 2405.18719; 2024.findings-emnlp.582 | abstract | content-adaptive / contextual counting as one bucket | not |
| Zhang et al. 2025 survey (content-dependent phase) | 2503.17407 | first-hand earlier (memory note) | covers content-dependent phase | not (no Case split) |
| "Position encoding in transformers ... RoPE ... long-context scaling" | 2608.10021 | abstract | absolute/relative/RoPE/scaling survey | not |

No taxonomy found that sorts encodings by constant / non-negative / signed increment in neuroscience terms.

## 3. What a paper can and cannot say

- Can say: "Howard et al. (2014) set one integrator's rate to a constant, a speed or a signed velocity to get
  time, path length or position. Index RoPE, monotone content-dependent phase (CoPE, CARoPE, softplus
  increments) and signed content-dependent phase (MapFormer, Selective RoPE, Mamba-3) are these three cases
  with imaginary rates; the rate is the selective-SSM step." Present it as a cross-field identification we
  have not seen stated, with GridPE (VCO -> RoPE), Sarkar & Howard 2026 (SITH ~ SSM), Alipour et al. 2025
  (velocity-modulated decay in a trained net) and RoMem (Howard-motivated content-gated phase rate) cited as
  the nearest partial statements.
- Can add: the Howard lab explicitly kept s real for stability; with imaginary s the signed rate is bounded
  (rotation), which is why the map case lives naturally in the rotary slot and the decay slot stays monotone
  (GRAPE's g(x) >= 0; Howard 2014's push-pull rectified pairs). This stability argument is ours
  (`docs/theory/2026-10-04/02_literature.md` sec. 2(b)); not found published.
- Cannot say: "first to connect time cells / Laplace memory to transformers or SSMs" (Tiganj lab, Sarkar &
  Howard, Lu et al. 2025 did); "first to connect RoPE to grid-cell path integration" (GridPE, MapFormer,
  TEM-t); "first content-dependent phase rate motivated by Howard" (RoMem).

## 4. Coverage and gaps

Searched (WebSearch standard + extended, 2026-10-12): DeepSITH, SITHCon, SITH-RNN, Laplace Neural Manifold,
Time-Local Transformer, Howard/Tiganj + transformer/attention/positional/SSM/Mamba/S4; SSM/Mamba + time cells
/ Laplace / hippocampus; RoPE + grid cells / path integration / hippocampus / phase coding / theta
precession; Whittington-Behrens-Dorrell + rotary; complex Laplace + RoPE equivalence; positional-encoding
taxonomies (constant / monotone / signed). Citation sweeps as listed in (B).

Not accessible / not done:
- Semantic Scholar citers of MapFormer (API returned 0 entries), Selective RoPE and Shankar & Howard 2012
  (rate-limited, 429). OpenAlex covered the Shankar & Howard citers but under-indexes arXiv preprints, so an
  arXiv-only paper citing Shankar & Howard 2012 could be missed.
- Google Scholar "cited by" (not queried; no API).
- bioRxiv full text of Yu, Wang, Balasubramanian 2026 (HTTP 429); abstract only.
- Howard 2005 (Psychol Rev) not re-read; known only via search summary.
- DeepSITH and SITHCon read at abstract level plus text grep, not section by section.

Sources (first-hand PDFs this session): arxiv.org/pdf/{2003.11668, 2104.04646, 2201.01796, 2302.10163,
2401.09491, 2405.14992, 2406.04545, 2406.07049, 2409.20484, 2412.15292, 2505.14806, 2507.13638, 2510.22109,
2510.22752, 2601.02618, 2604.01094, 2604.07658, 2604.11544, 2605.09742, 2609.04134, 2609.32447, 2609.39082,
2610.05700, 2610.12496}; homes.luddy.indiana.edu/ztiganj/papers/EvidenceAccumulationRL_IJCNN2023.pdf;
homes.luddy.indiana.edu/ztiganj/LearningScaleInvariantRepresentations.pdf; pmc.ncbi.nlm.nih.gov/articles/PMC13262383
(Time-Local Transformer); sites.bu.edu/tcn/files/2026/01/MarcHowardCV.pdf (publication list checked: no
transformer / RoPE / SSM paper). Citation lists: api.openalex.org (cites:W2009513819, W2101782048,
W2150301186); api.semanticscholar.org (Howard 2014, Mamba-3 2603.15569).
