# Prior and concurrent art for Draft 5 (2026-10-16)

Literature check of `docs/paper/DRAFT5_workshop.md` (abstract, Sec. 1, Sec. 3, Sec. 5, references) against the
local corpus (`papers/txt/`, `papers/new_2026-10-14/`, `papers/new_2026-10-15/`, `docs/lit/LIT_*.md`) and a web search
(WebSearch standard and extended; Semantic Scholar citations of Selective RoPE, CARoPE and MapFormer; the
reference lists of the papers found). Nothing in the draft was edited; nothing was committed. New downloads are in
`papers/new_2026-10-16/<id>/paper.txt` (index: `papers/new_2026-10-16/SOURCES.txt`).

Read status: **[read]** = the cited passage read as text this session (file:line given); **[corpus]** = read first-hand
in an earlier note and the passage re-grepped here; **[abstract]** = abstract or search summary only.

The five claims checked: (a) a learnability, not representational, cost of a non-negative parameterisation (|s|,
softplus) of a rotary phase step on synthetic maps; (b) existence: exact non-negative solutions are held (ring) or
re-found (torus) under training but not found from scratch; (c) Lemma 1, the per-channel reflection identity;
(d) the ring mechanism (through-zero route, collective latent escape, past-pi start); (e) CARoPE and CoPE fail these maps.

---

## 1. Summary

**No paper found states or shows (a) or (c).** No source isolates the sign of a content-dependent rotary step, or
any rotation angle, as a learnability variable, and none states the relative-sign gauge for content-dependent steps.
The one novelty sentence (Sec. 5) survives, with two scoping changes (Sec. 4 below).

**The most threatening overlaps, in order.**

1. **MapFormer v4 itself (already cited, but only for one sentence).** (i) Its Table 2 already reports CoPE failing
   its 1D and 2D navigation tasks in distribution (1 layer: 0.63 +/- 0.21 in 1D, 0.65 in 2D; 4 layers 0.88 / 0.76;
   MapWM 1.00), and the text attributes the failure to the sign of CoPE's gate. So the CoPE half of (e) is a
   replication, and Sec. 4.6 cites only the representational sentence, to contradict it. (ii) Its Figs. 3d / 9b show
   that trained signed models map opposite actions to antipodal latents (cos(Delta_left, Delta_right) = -1) and
   observations to zero steps. That is the end state of the draft's "collective route" (Sec. 3, {z(-1) = -z(+1)}) and
   of the opposite-signed channels in Fig. 2 / Sec. 4.3, described but not tested. The draft's causal test (freezing the
   move latents, 1/8 vs 7/8) and the |.| contrast are new; the code itself is not. Cite both.
2. **Shakerinava, Khavari, Ravanbakhsh, Chandar, ICLR 2026 (not cited).** In input-dependent unit-modulus complex
   diagonal SSMs (AUSSM: transitions exp(i Delta(x) Lambda(x)), i.e. content-dependent rotations), they name an
   "expressivity vs learnability" gap: "the solutions exist in weight space, yet standard training does not reliably
   reach them", and near-solution initialisation recovers them ("solutions lie within a basin of attraction"). Same
   framing as (b), in the rotary slot's recurrent twin, but for multi-layer non-Abelian groups; their single-layer
   signed-angle AUSSM learns every Abelian group tried (C2 ... C60) from scratch. Must be cited as the nearest
   learnability-gap result for rotation-based recurrences.
3. **Helfrich, Willmott, Ye, ICML 2018 (scoRNN; not cited).** A published case where the orientation (sign) part of a
   rotation parameterisation cannot be moved by gradient descent and must be fixed at initialisation: the scaled
   Cayley transform W = (I + A)^-1 (I - A) D keeps the +/-1 diagonal D fixed during training, its count of -1s is a
   tuned hyperparameter, and the plain Cayley map learns near -1 eigenvalues "very slowly, if at all". The closest
   precedent for "the sign of a rotary parameterisation is a start condition, not a learnable quantity". Cite in the
   "Sign-constrained parameterisations and initialisation" paragraph.
4. **Howe 2026 (arXiv 2609.18966, concurrent, single author; not cited).** Pre-registered within-architecture ablation in
   Householder linear RNNs; initialises at a verified-exact solution and shows Adam pulls the model off it. Same
   methodological move as (b) (warm start at an exact solution, compared with from-scratch), different variable (an
   additive input pathway, not sign). Optional, as concurrent work.

**Missing citations (recommended).** MapFormer Table 2 and Figs. 3d / 9b (above); Shakerinava et al. 2026; Helfrich
et al. 2018. Lower priority: Karuvally et al. 2025 (AUSSM) and Ebrahimi and Memisevic 2025 (further constructions of
Z_m from non-negative angles; the latter shows learned input-dependent angles settle at harmonics of 2 pi / m);
Wang and Wang 2026 and Tran et al. 2026 (the parameter-symmetry group of RoPE attention, for positioning Lemma 1);
Lezcano-Casado 2019 and Hayes et al. 2023 (pi boundaries and initialisation basins when angles or frequencies are
learned by gradient descent, for P3 and the past-pi start).

**Draft sentences that must change.**
- Sec. 5, novelty sentence: keep, but scope it to content-dependent steps and set it beside the fixed-frequency
  symmetry results and scoRNN (Sec. 4).
- Sec. 4.6, "contrary to MapFormer's statement ...": add that MapFormer also *measured* CoPE failing navigation
  (Table 2), so "CoPE fails these maps" is a replication; what is new is that CoPE can represent ring cancellation
  and still fails, and the line and CARoPE results.
- Sec. 1, "transitions with real non-negative eigenvalues cannot track parity or count modulo m (Sarrof et al.
  2024; Grazzi et al. 2025)": Grazzi's Thm. 2 says more: for m not a power of two, real eigenvalues *of either sign*
  cannot count modulo m; a non-real eigenvalue is needed (`papers/txt/grazzi.txt:327-332`). Suggested: "real
  non-negative eigenvalues cannot track parity, and real eigenvalues cannot count modulo m (m not a power of two)".
  This strengthens the paper's point that the theorems are about the real slot.
- Sec. 5, "Khavari et al. (2025) add that ... none of these theorems covers rotations by non-negative angles":
  (i) "add" has no antecedent in Sec. 5 (Sarrof and Grazzi are only in Sec. 1); (ii) Khavari's Thm. 4.4 does cover a
  model containing rotations (S4D, time-invariant complex eigenvalues); it fails because those layers are input-
  independent. Write "none of these theorems restricts an input-dependent rotation by a non-negative angle".

---

## 2. Claim-by-claim

| claim | closest prior / concurrent work | verdict |
|---|---|---|
| (a) learnability, not representation, cost of a non-negative rotary step | Representational sign costs in the real slot: Sarrof 2024, Grazzi 2025, Khavari 2025 (cited). Non-negative angles represent Z_m: Grazzi Thm. 3, Mamba-3 Sec. 3.2 (cited), AUSSM Lemma 1, Ebrahimi-Memisevic Prop. 2 (not cited). Learnability gaps in rotation recurrences: Shakerinava 2026 (not cited). Sign of a rotation parameterisation fixed at init: Helfrich 2018 (not cited). MapFormer attributes CoPE's navigation failure to its positive gate (representational reading). | **Not found.** No paper ablates the sign of a content-dependent rotary step. Position as "the same advice as Sarrof / Grazzi, for a different reason", as the draft does. |
| (b) exact |.| solution held / re-found, not found from scratch | Shakerinava 2026 (near-solution init recovers solutions training does not find, DCD/AUSSM SSMs); Howe 2026 (exact-init arms, pulled off); Hahn and Rofin 2024 (isolated sharp minima, cited); Liu 2023 (cited). | **Framing published; this instance new.** Cite Shakerinava as precedent; the rule-14 design (frozen and trainable, at a trained scale) is a methodological refinement. |
| (c) Lemma 1, relative-sign gauge | Conjugate-pair convention for fixed phases (LRU App.; S4D); RoPE attention's symmetry group with fixed frequencies is the commutant aI + bJ (complex scalars), no reflections (Wang and Wang 2026; Tran et al. 2026). Sign preservation under reparameterisation (Woodworth 2020; Amid and Warmuth 2020, cited). | **Not found** for content-dependent steps. J R(phi) J = R(-phi) is textbook; the paper's use (a parameter symmetry that appears only when the step is a parameter, fixing relative signs within a channel) was not seen. Keep "to our knowledge". |
| (d) through-zero route; collective latent escape; past-pi start | MapFormer Figs. 3d / 9b: antipodal latents for opposite actions (descriptive). Initialisation decides learnability of phases: LRU (cited), Helfrich 2018, Hayes 2023. pi as a chart boundary with spurious minima: Lezcano-Casado 2019. | **Partly anticipated.** The end-state code is MapFormer's; the causal freezing test, the |.| side and the past-pi rescue were not found. Cite MapFormer for the code. |
| (e) CARoPE, CoPE fail | MapFormer v4 Table 2 (CoPE fails 1D / 2D navigation, 1 and 4 layers). No navigation test of CARoPE found. | **CoPE: replication; CARoPE: new.** The CoPE representability construction (contradicting MapFormer's sentence) is new. |

---

## 3. Paper by paper

### 3.1 Already cited; is the draft's description right?

**Rambaud, Mascarenhas, Lakretz. MapFormer. arXiv:2511.19279 v4 (10 May 2026).** [read]
- Shows: navigation needs "an action (moving forward) and its inverse (moving backwards), preventing CoPE from solving
  the task, even in 1D" (`papers/txt/mapformer.txt:584-586`); App.: "Since the sigmoid gate can only take positive
  values, the model cannot represent an action and its inverse" (`:1701-1703`). Table 2 (`:436-560`): CoPE 1 layer
  0.63 +/- 0.21 (1D IID), 0.65 (2D IID); 4 layers 0.88 / 0.76; MapWM-r2 1.00 / 1.00. Opposite actions map to opposite
  directions and observations to near-zero steps (`:650-655`, Fig. 3d); Fig. 9b reports cos(Delta_left, Delta_right) =
  -1 and Fig. 9a that accuracy rises when observation tokens learn zero angles (`:2120-2155`).
- Overlap: (e) for CoPE; (d) and Fig. 1 / Fig. 2 for the signed end-state code; the observation-step-free arms are the
  constraint MapFormer's models learn on their own.
- Draft: cites the sentence correctly (Sec. 4.6) but omits Table 2 and Figs. 3d / 9b. Add: "MapFormer already reports
  CoPE failing 1D and 2D navigation and attributes it to the gate's sign (their Table 2); CoPE can in fact represent
  the ring map, so the obstacle is not representational." And in Sec. 3 / 4.3: "Trained signed models end with
  antipodal move latents and near-zero observation steps (MapFormer, Figs. 3d, 9); we test whether the latents are the
  route."

**Grazzi, Siems, Zela, Franke, Hutter, Pontil. ICLR 2025; arXiv:2411.12537.** [corpus]
- Thm. 1: parity needs an eigenvalue outside [0, inf) in some layer (`papers/txt/grazzi.txt:316-318`). Thm. 2: counting
  modulo m (m not a power of two) needs an eigenvalue with non-zero imaginary part (`:327-332`): real eigenvalues of
  either sign fail. Thm. 3 / App. C.3 builds Z_m from rotations by 2 pi w / m >= 0 (`:485-494`, `:2272-2279`).
- Overlap: (a) representational side. Draft's Sec. 1 attribution is slightly loose on Thm. 2 (see Sec. 1 above);
  "Grazzi et al. ... build cyclic groups from such rotations" is correct.

**Sarrof, Veitsman, Hahn. NeurIPS 2024; arXiv:2405.17394.** [corpus] Non-negative gates "due to exponential or sigmoid
parameterizations" cannot recognise PARITY (`papers/txt/sarrof.txt:147-148`, `:311-314`). Note `:311`: "these design
choices necessitated by optimization, limit the power": Sarrof already frames non-negativity as an optimisation-driven
parameterisation choice. Draft citation correct.

**Khavari et al. ICML 2025 workshop (MOSS); arXiv:2508.07395.** [read] Thm. 4.4: S4D (time-invariant, complex or
negative eigenvalues) plus Mamba (input-dependent, non-negative) layers cannot solve parity
(`papers/new_2026-10-15/2508.07395/paper.txt:217-224`). The draft's summary ("input dependence and negative
eigenvalues must occur in the same layer") is right; "none of these theorems covers rotations by non-negative angles"
needs "input-dependent" (S4D's fixed rotations are covered; they fail for lack of input dependence).

**Lahoti et al. Mamba-3. ICLR 2026; arXiv:2603.15569.** [corpus] Parity as h_t = R(pi x_t) h_{t-1}
(`papers/txt/mamba3.txt:576-579`); angle Delta_t theta_t with Delta_t >= 0 and theta_t a signed projection
(`:637`, `:3026`); no sign ablation (Table 5b compares data-dependent vs standard RoPE). Draft correct.

**Movahedi et al. Selective RoPE. ICLR 2026; arXiv:2511.17388.** [corpus] Signed content rotations; one-layer S2
(parity) (`papers/txt/srope.txt:926-935`); "flips" (`:931`). No sign ablation. Draft correct ("solve parity ... in one
layer").

**Yang et al. PaTH. NeurIPS 2025; arXiv:2505.16381.** [corpus] beta in (0, 2) "allows negative eigenvalues"
(`papers/txt/path.txt:136`); one-layer two-head Flip-Flop theorem (`:202`) and experiments (`:791`). Draft correct.

**Zhang et al. GRAPE. ICLR 2026; arXiv:2512.07805.** [corpus] Phase-modulated Additive GRAPE uses omega_l = g(x_l) >= 0
in a logit bias (`papers/txt/grape.txt:682-684`); softplus gates in the additive slot (`:466-488`). Draft's "its
non-negative content phase is a real-slot bias, as in FoX" is correct.

**Veisi et al. CARoPE, arXiv:2507.23083; Golovneva et al. CoPE, arXiv:2405.18719.** [corpus] CARoPE rate
1/(softplus(xW) + 1) in (0, 1), one scalar per head (`papers/txt/carope.txt:152-155`); CoPE sigmoid-gated counts.
Draft correct.

**Orvieto et al. LRU. ICML 2023; arXiv:2303.06349.** [read, LIT_REPRESENTABILITY] Phases exp(theta_log) > 0; PathX needs
phases initialised in [0, pi/10]. Draft's description correct. Note for positioning (and the mock review's point): LRU
phases are input-independent, so LRU's positive parameterisation is a conjugate-pair convention and costs nothing
representationally (Lemma 1(ii), one driver per channel). Same for **LeRoPE** (Karypis et al., arXiv:2607.10134,
learned index-RoPE frequencies parameterised as exp(alpha), "avoids undesirable gradient dynamics near frequencies of
zero", `papers/new_2026-10-16/2607.10134/paper.txt:198-201`) [read]: a positive parameterisation that is harmless
because one "token" (the index) drives every channel. Optional one-clause example of Lemma 1(ii).

**Woodworth et al. COLT 2020; Amid and Warmuth NeurIPS 2020.** [read, LIT_REPRESENTABILITY] Sign cannot change under
gradient flow of a reparameterised weight. Draft's use ("cf." for P4) correct.

**Cornford et al. ICLR 2021; Li, Cornford, Ghosh, Richards NeurIPS 2023.** [abstract / read] Draft's "spectrum at
initialisation matters more than the sign constraint, though both contribute" matches Li et al.
(`papers/new_2026-10-15/li2023_neurips/paper.txt:29-32`).

**Liu et al. ICLR 2023; Hahn and Rofin ACL 2024.** [read, LIT_REPRESENTABILITY] Draft's one-line uses are correct.

**Howard et al. 2014; Whittington et al. TEM-t ICLR 2022.** [corpus, LIT_CLOCK_MAP] Correct. Optional: Howard's
Methods build a signed (map) code from two rectified (monotone) integrators with opposite preferred directions read as
a difference; in the rotary score no readout subtracts channels, which is why that real-slot remedy does not transfer.
One clause would pre-empt a reviewer who knows Howard.

### 3.2 Not cited; recommended

**Shakerinava, Khavari, Ravanbakhsh, Chandar. The expressive limits of diagonal SSMs for state-tracking. ICLR 2026;
arXiv:2603.01959.** [read] `papers/new_2026-10-16/2603.01959/paper.txt`
- Shows: k-layer input-dependent complex diagonal SSMs track a group iff it has an Abelian subnormal series of length k.
  AUSSM: A(x) = exp(i Delta(x) Lambda(x)), Delta >= 0, Lambda(x) signed (`:179-183`). Single-layer AUSSM learns all six
  Abelian groups from scratch (`:708-712`); two-layer models fail on S3 / A4 though they can represent them: "The main
  bottleneck of non-Abelian tracking is not expressivity but optimization ... the solutions exist in weight space, yet
  standard training does not reliably reach them" (`:724-726`); "sufficiently close initialization improves training
  ... solutions lie within a basin of attraction" (`:727-731`).
- Overlap: (a) and (b) as framing (learnability gap with representable solutions, near-solution init), in the
  recurrent form of content-dependent rotation (Mamba-3 Prop. 3 makes complex diagonal SSMs equivalent to
  data-dependent RoPE). Not the sign axis.
- Cite: Sec. 5 "Sign and state tracking" or a new sentence: "Learnability gaps between representable and learned
  solutions are documented for input-dependent rotation recurrences (Shakerinava et al. 2026: near-solution starts
  recover non-Abelian solutions that training from scratch misses); ours is a gap created by one parameterisation
  choice within a single-layer Abelian task." Also useful in Sec. 4.2 (existence).

**Helfrich, Willmott, Ye. Orthogonal recurrent neural networks with scaled Cayley transform. ICML 2018;
arXiv:1707.09520.** [read] `papers/new_2026-10-16/1707.09520/paper.txt`
- Shows: the Cayley parameterisation cannot represent -1 eigenvalues; near them "Gradient descent algorithms will learn
  this A matrix very slowly, if at all" (`:118-128`); scoRNN multiplies by a fixed +/-1 diagonal D, "D is fixed throughout
  the training process. The number of -1s on the diagonal of D ... is considered a hyperparameter" (`:160-169`). The
  follow-up scuRNN (Maduranga, Helfrich, Ye, AAAI 2019) replaces D by learnable unit-circle phases because the discrete
  D "can not be optimized by gradient descent" [abstract / search summary].
- Overlap: (a) and (d) conceptually: an orientation (sign) component of a rotation parameterisation that gradient
  descent cannot change, so it becomes a start condition. Different object (a fixed recurrent matrix, not a
  content-dependent step).
- Cite: Sec. 5 "Sign-constrained parameterisations and initialisation", beside LRU: "In orthogonal RNNs the +/-1
  orientation part of the scaled Cayley parameterisation cannot be learned by gradient descent and is set at
  initialisation (Helfrich et al. 2018)."

**Howe. The automaton underneath: the additive input pathway is a parasitic attractor for state tracking in Householder
linear RNNs. arXiv:2609.18966 (Jul 2026), unreviewed.** [read, abstract and Secs. 1, 5-6]
- Shows: pre-registered (202 runs) ablation in a DeltaProduct-style model; "Initialized at a verified-exact solution
  with W_b = 0, Adam grows the additive path ... and pulls the model off the exact solution ... while -b controls stay
  at 1.00" (`:27-30`). Cites Shakerinava as naming the gap (`:46-48`).
- Overlap: (b) method (exact init vs scratch; held vs lost), and the pre-registration style. Concurrent.
- Cite: optional, one clause in Sec. 4.2 or Sec. 5 ("cf. concurrent exact-initialisation tests in Householder linear
  RNNs, Howe 2026"). Single-author unreviewed preprint; cite as concurrent only.

**Karuvally, Nowak, Keller, Amo Alonso, Sejnowski, Siegelmann. Bridging expressivity and scalability with adaptive
unitary SSMs. arXiv:2507.05238 (v3, Jan 2026).** [read] Lemma 1: a single-layer AUSSM counts modulo k with
A(e(a)) = exp(2 pi i / k) and each other symbol exp(2 pi i n / k), n in [0, k-1]
(`papers/new_2026-10-16/2507.05238/paper.txt:1400-1410`): another non-negative-angle construction of Z_k. Optional
co-citation with Grazzi Thm. 3 in Sec. 1 / Sec. 3 ("-1 = m - 1").

**Ebrahimi, Memisevic. Revisiting bi-linear state transitions in recurrent neural networks. NeurIPS 2025;
arXiv:2505.21749.** [read] Prop. 2: block-diagonal rotations with input-dependent angles simulate any Abelian group; the
construction uses theta(x_sigma) = 2 pi sigma / m (`:955-985`). Learned R2 models put input 1's angle at harmonics of
2 pi / m, other inputs at multiples (`:656-665`). Overlap: representational side of Sec. 3, plus evidence that signed
input-dependent angles learn wrapped cyclic codes from scratch on modular addition. Optional co-citation.

**Wang and Wang. Complete characterization of gauge symmetries in transformer architectures. PMLR 282 (NeurIPS 2024/2025
Workshops on Symmetry and Geometry in Neural Representations), 2026.** [read] For standard RoPE with distinct
frequencies the query-key gauge reduces to block-diagonal aI_2 + bJ (J the 90-degree rotation), i.e. GL(1, C) per
plane (`papers/new_2026-10-16/wang2026_gauge/paper.txt:277-280`); "No additional continuous or discrete symmetries"
(`:1440`). **Tran et al. Functional equivalence in attention. ICML 2026; arXiv:2606.17830** [read, abstract and Sec. 3]:
RoPE shrinks the symmetry group relative to sinusoidal encodings (`:128-137`, `:352-400`).
- Overlap: (c). Both take frequencies as fixed, so the reflection diag(1, -1) is not a symmetry there; it becomes one
  only when the step (or omega_c) is a parameter and is negated with it. That is exactly Lemma 1's content.
- Cite: one clause in Sec. 3 or Sec. 5: "With fixed frequencies the parameter symmetries of rotary attention are the
  per-plane complex scalars (Wang and Wang 2026; Tran et al. 2026); a learnable step adds the reflection of Lemma 1."
  This supports, and scopes, the novelty sentence.

**Lezcano-Casado. Trivializations for gradient-based optimization on manifolds. NeurIPS 2019; arXiv:1909.09501.**
[read] The Lie exponential is a diffeomorphism on {|Im lambda_i(A)| < pi}; its boundary behaves like a cut locus and
"creates local minima and saddle points" as optimisation leaves the start (`:256-268`); dynamic trivialisations re-centre
the chart. Also Lezcano-Casado and Martinez-Rubio, ICML 2019 (expRNN; Henaff vs Cayley initialisations of the rotation
angles, `papers/new_2026-10-16/1901.08428/paper.txt:365-378`). Overlap: (d) and P3 loosely (pi as the boundary where
angle parameterisations become hard to optimise). Optional, one clause beside P3.

**Hayes, Saitis, Fazekas. Sinusoidal frequency estimation by gradient descent. ICASSP 2023; arXiv:2210.14476.** [read,
abstract and Sec. 2] Gradient descent on an oscillator's frequency faces an objective "non-convex and densely populated
with local minima" (`:18-20`); "unless parameter estimates are initialized in the main basin ... optimization will
converge on an incorrect solution" (`:111`). Overlap: (d) (start decides which angle code gradient descent reaches).
Optional.

**Ran-Milo et al. Provable benefits of complex parameterizations for structured SSMs. NeurIPS 2024; arXiv:2410.14067.**
[read, abstract] Real diagonal SSMs need exponentially large parameters to express complex-SSM mappings, "which cannot
be learned in practice" (`:30-37`). Real vs complex slot, a learnability-flavoured separation; context for Sec. 1's
real/rotary distinction. Optional.

### 3.3 Context only (read enough to rule out overlap)

- **Siems et al. DeltaProduct, NeurIPS 2025 (2502.10297)**; **Terzic et al. AAAI 2025 (2412.19350)**; **Siems et al.
  2026, state tracking from code (2602.14814)**; **Chung et al. 2026, error control (2605.07755)**; **Merrill, Petty,
  Sabharwal ICML 2024** [abstract]: real-eigenvalue sign, non-diagonal transitions, length generalisation, error
  accumulation. None varies the sign of a rotation angle. Not needed in a workshop paper.
- **Hoedt and Klambauer, NeurIPS 2023 (ICNN initialisation)** [abstract]: non-negative weights need a non-centred
  initialisation. Another "constraint vs start" case like Dale's law; optional.
- **Cueva and Wei 2018; Banino et al. 2018** (RNN path integration from signed velocities) [abstract, via the mock
  review]: velocity is given and signed; no sign parameterisation. Not needed.

### 3.4 Searched, nothing found

- A sign or |.| / softplus ablation of a content-dependent rotation angle (RoPE variants, complex SSMs, unitary RNNs).
- "Through zero" vs "by wrapping" cancellation, or a pi-sum start rule, for learned rotation steps.
- A reflection / complex-conjugation symmetry stated for content-dependent rotary steps.
- A navigation test of CARoPE. Semantic Scholar lists 12 citers of Selective RoPE and 7 of CARoPE (2026-10-16); the
  only candidates (DaRoPE 2609.34556, CRoPE 2601.02728) do not touch sign. MapFormer's citation list came back empty
  from the API, so follow-ups to MapFormer could not be enumerated: say "to our knowledge".
- Checked and irrelevant: 2605.11007 (polar state estimator), 2610.14030 (looped-transformer residual rotations).

---

## 4. The novelty sentence (Sec. 5)

> "To our knowledge neither the relative-sign gauge nor the loss of the through-zero route for inverse pairs has been
> stated; we claim their conjunction and the measurements, not any single ingredient."

**Not contradicted.** Nearest statements: the conjugate-pair convention for fixed phases (LRU App.; S4D); the RoPE
symmetry group with fixed frequencies (Wang and Wang 2026; Tran et al. 2026), which contains no reflection; sign
preservation under reparameterisation (Woodworth; Amid and Warmuth); an orientation component fixed at initialisation
(Helfrich et al. 2018); MapFormer's antipodal latents (the through-zero code, observed, not framed as a route).

Suggested replacement (same length): "With fixed frequencies the symmetries of rotary attention are per-plane complex
scalars (Wang and Wang 2026); to our knowledge the reflection that a learnable step adds (Lemma 1), and the loss of the
through-zero route for inverse pairs that follows, have not been stated. We claim their conjunction and the
measurements, not any single ingredient."

Also check, outside Sec. 5: Sec. 1's "We identify a route to the map that the barrier removes" should credit MapFormer
for the end-state code (Figs. 3d, 9b); the route as a causal claim (freezing test) is the paper's.

---

## 5. Reference entries to add (verified against arXiv metadata, `papers/new_2026-10-16/meta/arxiv_batch.xml`, unless marked *)

- Shakerinava, M., Khavari, B., Ravanbakhsh, S. and Chandar, S. The expressive limits of diagonal SSMs for
  state-tracking. ICLR 2026; arXiv:2603.01959.
- Helfrich, K., Willmott, D. and Ye, Q. Orthogonal recurrent neural networks with scaled Cayley transform. ICML 2018*;
  arXiv:1707.09520.
- Wang, H. and Wang, K. Complete characterization of gauge symmetries in transformer architectures. Proceedings of the
  NeurIPS Workshops on Symmetry and Geometry in Neural Representations, PMLR 282:622-663, 2026 (no arXiv id).
- Tran, V.-H., Bui, V. K., Trinh, V.-H., Lai Ngoc, T. and Nguyen, T. M. Functional equivalence in attention: a
  comprehensive study with applications to linear mode connectivity. ICML 2026; arXiv:2606.17830.
- Optional: Karuvally, A., Nowak, F., Keller, A. T., Amo Alonso, C., Sejnowski, T. J. and Siegelmann, H. T. Bridging
  expressivity and scalability with adaptive unitary SSMs. arXiv:2507.05238, 2025. Ebrahimi, M. R. and Memisevic, R.
  Revisiting bi-linear state transitions in recurrent neural networks. NeurIPS 2025; arXiv:2505.21749. Howe, G. L. The
  automaton underneath: the additive input pathway is a parasitic attractor for state tracking in Householder linear
  RNNs. arXiv:2609.18966, 2026. Lezcano-Casado, M. Trivializations for gradient-based optimization on manifolds.
  NeurIPS 2019*; arXiv:1909.09501. Hayes, B., Saitis, C. and Fazekas, G. Sinusoidal frequency estimation by gradient
  descent. ICASSP 2023*; arXiv:2210.14476.
