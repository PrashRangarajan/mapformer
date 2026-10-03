# Literature: separating "what" from "where" in attention and in cognitive-map models (2026-10-03)

Literature review, not a result. Situates the descriptive, post-hoc probe in `docs/WHAT_WHERE_ANALYSIS.md`
sec. 6 (trained 1-layer torus models put ~87-92% of score variance into a shared displacement kernel;
PoPE imposes separation at initialisation but its trained endpoint is not cleaner than MapWM's).
Every source below was read by this session; each note says how deeply ("first-hand" = the method
and result sections were read; "abstract" = abstract/intro only; "secondary" = known through another
paper that was read). Local corpus (`papers/txt/`) was grepped first; everything else was downloaded
from arXiv / Europe PMC and read as text.

Terms, defined once.
- **what**: token identity (object, word). **where**: a relational position variable.
- **score**: pre-softmax attention logit. **interaction share** (`inter/pos` in our probe): fraction of the
  position-dependent score variance that depends on the content pair (0 = one kernel for every pair).
- **architected** separation: the parameterisation forbids (or removes) content-position mixing.
  **learned** separation: the parameterisation allows mixing and training drives it down.
- **L1-L6**: the separation levels of `WHAT_WHERE_ANALYSIS.md` sec. 3 (L1 source purity, L2 content cannot
  move the kernel peak, L3 one shared kernel, L4 additive, L5 TEM-t form, L6 transfer to new content).
- **factorised task**: content and position vary independently across training sequences (our torus:
  every sequence redraws the map). **entangled task**: content is tied to position (a fixed map).

## 1. Summary

1. **"Separation is learned rather than imposed" is not new as a general statement.** It has been shown
   (a) in our exact models at the source level by MapFormer itself (Fig. 9: generalisation arrives when
   observation steps go to ~0 and value norms split by token type), (b) in trained language models with
   architecturally entangled encodings (Song & Zhong 2024: position and context components of hidden
   states nearly orthogonal; TUPE 2021: word-position cross terms in trained BERT look flat; Barbero et al.
   2025 and Urrutia et al. 2025/2026: trained RoPE models split frequencies and heads into positional and
   semantic/symbolic roles, and pure heads emerge with successful learning), and (c) as theory
   (Whittington et al. 2023: nonnegativity + energy efficiency yield factorised codes when the task
   factors are independent; Cui et al. 2024: whether attention learns a positional or a semantic solution
   is a phase transition in sample complexity).
2. **The hippocampal models architect the split; they learn only what fills it.** TEM separates `g`
   (structure) from `x` (content) by construction and learns the structural code; TEM-t sets Q = K =
   position and V = content (our L5); CSCG separates transitions from emissions and transfers by
   freezing the transition graph; Vector-HaSH uses a pre-structured, fixed grid scaffold.
3. **Counter-evidence: content x position mixing is used where the task needs it.** Transformer-XL's
   own term analysis attributes the attention trend to the content-dependent positional bias; removing
   either cross term from DeBERTa hurts every benchmark; Whittington 2023 shows grid codes warp to
   objects when the task is entangled; our own `PAIRSPLIT_RESULTS.md` (per-pair freedom +0.091, n=48)
   agrees on recency.
4. **What remains ours** (no paper found that does it): the score-level, within-head decomposition
   (content pair x displacement, exact for 1 layer, logit-verified) on path-integrated phase models, with
   untrained controls, and the matched comparison of an architected decoupling (PoPE magnitude/phase)
   against its unconstrained counterpart (MapWM) at the trained endpoint. That last contrast is the only
   candidate-novel claim, and it is the weakest: post hoc, n=8 per arm, unpaired, one task, one layer.
5. **Biggest alternative reading, from the theory:** our torus task is factorised by construction, which
   is exactly the regime where Whittington 2023 predicts learned disentanglement for ANY model with the
   right biases. Our own table already shows trained index RoPE dropping from 0.98 to 0.33 interaction
   share. So "path-integrating models learn the separation" is not yet distinguishable from "the data are
   factorised, so most models move toward it". Proposal P2 tests this directly.

## 2. Table

| paper | id | mechanism (score or circuit) | architected vs learned | measured how | result | relevance |
|---|---|---|---|---|---|---|
| TEM, Whittington et al. 2020 | Cell 183:1249; PMC7707106 | `g_{t+1}=sigma(g_t W_a)` (MEC, structure); `x` (LEC, content); `p = flatten(x^T g)` stored Hebbian, retrieved by `g` | architected split; `g`'s code (grids) learned | next-observation prediction on held-out stimulus arrangements; cell rate maps; place-grid remapping in rodent data | first-presentation inference; structure preserved over remapping (r=0.322, 115 pairs, p<.01) | the factorisation claim we test; new environments = new arrangement of a FIXED stimulus set (as ours) |
| TEM-t, Whittington, Warren, Behrens 2022 | 2112.04035 (ICLR) | `y_t = softmax(e_t E^T/sqrt d) X W_x`, Q = K = `E W_e` (position only), V = content; `e_{t+1}=sigma(e_t W_a)` | architected, strict (L5) | sample efficiency vs TEM; rate maps | faster learning than TEM; grid/band cells in `e` | the strict-separation reference; its triple-conjunction rule `softmax((gG^T) * (xX^T))` is MapEM's form |
| Tale of two algorithms, Whittington et al. 2025 | Neuron 113:321 (local `tale_two_algorithms`) | EM (separate position pop. + memory) vs WM (conjunctive slots) | architected (two algorithms) | capacity, learning speed | same solution; EM scales better at fixed budget | MapFormer's EM/WM framing |
| MapFormer v4, Rambaud et al. | 2511.19279 (local) | WM: RoPE on content Q,K with path phase; EM: `A_X * A_P`; EM-s: `A_P` alone | source (step) learned; EM score split architected; EM-s structure-only | Fig. 9: step norms vs accuracy over training; value norms; action cosines | generalisation coincides with observation steps -> 0; opposite actions cancel; EM-s at/near ceiling on navigation, more robust in size/length/vocab (Fig. 11) | closest prior art: learned L1 separation in our models; EM-s is the content-free arm |
| PoPE, Gopalakrishnan et al. | 2509.10534 (local) | `sum_c mu^q_c mu^k_c cos((s-t)theta_c - delta_c)`, `mu=softplus>=0` | architected (deletes RoPE's `phi_k - phi_q`) | Indirect Indexing; LM perplexity; frequency-usage plots | Indirect 94.82 +/- 2.91 vs RoPE 11.16 +/- 2.45; better extrapolation | the imposed scheme we compare; never measures how entangled trained RoPE actually is |
| Shaw et al. 2018 | 1803.02155 | `e_ij = x_i W^Q (x_j W^K + a^K_{clip(j-i)})^T` | architected content-to-position term only | WMT BLEU ablation | removing both a^K, a^V: 12.5 vs 25.8; a^K alone suffices | first relative scheme; keeps a query-content x position term |
| Transformer-XL, Dai et al. 2019 | 1901.02860 | 4 terms: content, content-dependent positional bias, global content bias `u`, global positional bias `v` | architected decomposition | App.: softmax of each term averaged over WikiText-103 | the content-dependent positional term (b) carries the overall trend; pure position (d) is flatter | language USES the content x position term |
| TUPE, Ke, He, Liu 2021 | 2006.15595 (ICLR) | `alpha = (xW^Q)(xW^K)^T/sqrt(2d) + (pU^Q)(pU^K)^T/sqrt(2d) [+ b_{j-i}]` | architected additive (L4) | visualised the 4 expansion terms in trained BERT; GLUE | word-position terms "look uniform" in trained BERT; TUPE-R +1.38 GLUE avg over BERT-R; beats the baselines with 30% of the pre-training steps | trained model already near-zero cross terms (absolute PE) -> then imposed |
| DeBERTa, He et al. 2021 | 2006.03654 (ICLR) | `Qc Kc^T + Qc Kr_{d(i,j)}^T + Kc Qr_{d(j,i)}^T` (p2p dropped) | architected two-vector representation, cross terms kept | ablation | RACE 71.7 -> 69.3 (-c2p), 69.6 (-p2c) | removing content x position terms HURTS on language |
| NoPE, Kazemnejad et al. 2023; Haviv et al. 2022 | 2305.19466 (local); 2203.16634 | no position code; causal mask only | position learned from scratch | probes for absolute position; length generalisation | NoPE models learn position; NoPE resembles T5 relative patterns when trained | "where" can be entirely learned |
| Round and round, Barbero et al. 2025 | 2410.06205 (ICLR) | RoPE; per-frequency q/k norms | learned partition of frequencies | per-2D-chunk q/k norm in Gemma 7B; p-RoPE ablation | high frequencies build positional heads, low frequencies act as semantic channels; 0.75-RoPE ppl 4.441 vs RoPE 4.463 (Gemma 2B) | learned what/where split ACROSS channels |
| Urrutia et al. 2025 | 2511.11579 | definitions: positional = logits invariant to permuting key contents; symbolic = depend only on key content | measures learned behaviour | permutation-based positional/symbolic scores per head and per frequency; frequency control | exclusion theorem (both at once only with uniform attention); early layers positional, late symbolic; performance controllable by restricting frequencies | a standard, depth-agnostic probe we lack |
| Urrutia et al. 2026 | 2605.31558 | as above, over training | learned | scores across training on positional vs symbolic multi-hop tasks | successful learning coincides with emergence of PURE heads | learned separation tied to success, over time |
| Song & Zhong 2024 | 2310.04861 | `h_{c,t} = mu + pos_t + ctx_c + resid` (two-way ANOVA on hidden states); QK split into pos/ctx terms | learned | GPT-2, BERT, BLOOM, Llama 2: rank, smoothness, incoherence; QK constituent plots | pos and ctx nearly orthogonal (random baseline ~0.1); neighbour attention driven by pos-pos, induction by ctx-ctx | the nearest prior to our probe, on hidden states |
| Gu et al. 2026 | 2505.13027 (ICLR) | additive (Toeplitz bias) vs multiplicative (Hadamard with Toeplitz kernel, RoPE) | analysis | synthetic task needing integration of position and content | multiplicative forms win; positional processing concentrates in one shallow head | WM vs EM/PoPE-like taxonomy, language side |
| Give it Space!, 2026 | 2605.30022 | three explicit streams (semantic, absolute, relative position) | architected | probing benchmark | improves 49/65 phenomena | architected follow-up to Song & Zhong |
| CSCG, George et al. 2021 | Nat Commun 12:2392; PMC8062558 | cloned action-HMM: emissions fixed by clone structure, transitions `T(z'|z,a)` learned by EM | architected split; topology learned | new room: freeze T, re-learn emissions | shortcut paths in an unseen room | transfer by freezing structure = TEM's claim, done by fiat |
| Space is a latent sequence, Raju et al. 2022 | 2212.01508 | CSCG as hippocampal theory | as CSCG | reproduces >12 place-cell phenomena | spatial maps emerge from sequence learning | where from sequence; place fields as an interpretive overlay |
| Graph schemas, Guntupalli et al. 2023 | 2302.07350 | learned latent graph as template, slots bound to observations | architected | transfer to novel environments | fast transfer by rebinding | structure reuse with new content (our L6) |
| TDB, Dedieu et al. 2024 | 2401.05946 | transformer with discrete bottleneck -> extracted map | learned latent | in-context accuracy, planning in new aliased rooms | near-perfect in-context accuracy and maps | transformer-side cognitive map, no what/where analysis |
| Vector-HaSH, Chandra, Sharma, Chaudhry, Fiete 2025 | Nature, 10.1038/s41586-024-08392-y | fixed modular grid scaffold <-> hippocampal layer (fixed random), Hebbian sensory<->hippocampus | architected (pre-structured scaffold) | capacity vs detail; sequence memory | no memory cliff; scaffold essential even for non-spatial episodes | the strongest "imposed" position code |
| Successor representation, Stachenfeld et al. 2017 | Nat Neurosci 20:1643; bioRxiv 097170 | `V = M R`, `M = sum gamma^k T^k` | architected split of dynamics from reward | place/grid fits | predictive maps | separates structure from value, not from content |
| Disentanglement with biological constraints, Whittington et al. 2023 | 2210.01768 (ICLR) | nonneg + energy-efficient activity and weights | learned, under constraints | theorems; MI matrices; grid/OVC modules; entangled vs factorised task | factorised codes iff factors independent; entangled task -> grid fields warp to objects | theory of WHEN learned separation should appear |
| Actionable representations, Dorrell et al. 2023 | 2209.15563 (ICLR) | representation updated linearly by action matrices + nonneg + bounded | normative | analytic + simulation | multi-module hexagonal grids optimal | theory of the where code itself |
| Phase transition, Cui et al. 2024 | 2402.03902 | tied low-rank dot-product attention, teacher mixes positional and semantic attention | learned | exact asymptotics of the global minimum | positional -> semantic solution switches sharply with sample complexity | which mechanism training finds depends on data amount |
| Johnston & Fusi 2023 | Nat Commun 14:1040; PMC9950464 | feedforward multi-task nets | learned | abstraction (cross-condition generalisation) metrics | abstract/disentangled codes emerge with the number of tasks | emergence driver: task diversity |
| Lindsey & Issa 2024 | eLife; PMC11226229 | factorisation = fraction of parameter variance outside the other-parameter subspace | learned | V4/IT and DNN libraries; "statistical lesion" rotating subspaces | factorisation of identity from position/background rises V4 -> IT and predicts brain match | gives a causal-style lesion we can copy |
| Locatello et al. 2019; Higgins et al. 2018 | 1811.12359; 1812.02230 | unsupervised disentanglement; symmetry-based definition | -- | 12,000 models; group-theoretic definition | impossible without inductive biases; disentangling defined by group actions | frames "imposed" vs "learned": learning needs a bias somewhere |
| Key-value memory in the brain, Gershman, Fiete, Irie 2025 | 2501.02950 (local) | TEM and Vector-HaSH as key-value memories | review | -- | position as key/address, content as value | where = address, what = value |

## 3. Per-paper notes

**TEM** (first-hand, PMC full text). Separation is the design: "we separate variables of abstract location
that generalize across maps (g ...) from those that are grounded in sensory experience". The memory is
the outer product `p = flatten(x^T g)`; retrieval can be cued by `g` alone ("what did I see the last time
I was here"). Learned: the transition weights `W_a` and hence the grid/band/OVC structure of `g`.
Generalisation test = new environments with "each vertex randomly assigned a stimulus" from a fixed
one-hot set; the sensory input is compressed to two-hot and temporally smoothed. So TEM, like us, tests
new arrangements of known objects, not new objects. Remapping prediction confirmed in Barry et al. 2012
and Chen et al. 2018 recordings.

**TEM-t** (first-hand). Three modifications: Q and K tied and computed from position encodings only
(`Q, K = E W_e`), V from stimulus only (`V = X W_x`); causal memory; recurrent position
`e_{t+1} = sigma(e_t W_a)` with an action-dependent matrix. Softmax temperature scaled by
log(number of memories). The authors call Q/K-from-position "an extreme version of the realisation that
... best performance is when position encodings are used to compute keys and queries, but not values"
(no citation given). Proposed three-population rule `softmax((g G^T) * (x X^T)) C` is the elementwise
product MapEM uses. Nothing is measured about separation; it is imposed. Results are qualitative
(sample efficiency vs TEM, rate maps).

**MapFormer** (first-hand, local). EM "factorizes content and position in two separate neural spaces";
WM "directly update[s] the conjunction of what and where". Fig. 9 is the learned-separation measurement:
"perfect generalization is achieved as soon as the model learns that action tokens trigger state
rotations while observation tokens leave it untouched"; trained value norms are large for observations,
small for actions. Fig. 9 caption: "Other constraints, such as bounded energy [42 = Whittington 2023],
could be added to force disentanglement" -- the authors point at the same theory. MapEM-s ("relying on
structure alone") is a content-free score; App. A.7: "This forced separation is primarily used to
highlight the benefits of focusing on different modalities ... the model should learn to balance ...
we decided to leave it for future work." Table 2 shows EM-s at or near 1.00 on navigation (pdftotext
layout of Table 2 is ambiguous: check the PDF before citing a cell).

**PoPE** (first-hand, local). Argues by algebra that RoPE's `phi_k - phi_q` "confound[s] information about
the presence or absence of features (the 'what') and relative positions (the 'where')" and removes it.
Evidence is behavioural (Indirect Indexing 94.82 vs 11.16; language and music perplexity; extrapolation).
It does not measure how much trained RoPE models actually use the confound -- our probe does (trained
RoPE interaction share 0.334 on the torus, `WHAT_WHERE_ANALYSIS.md`).

**Shaw / Transformer-XL / TUPE / DeBERTa** (first-hand). The four-term expansion
`xWqWk^T x + xWqWk^T p + pWqWk^T x + pWqWk^T p` (written out in TUPE eq. 6 and RoPE eq. 6-7) is the
language-side what/where decomposition. Shaw keeps content-content and content-to-position; T-XL adds
global content and global position biases and reports (App.) that the content-dependent positional
term carries the trend; TUPE inspects trained BERT, finds the cross terms flat, and deletes them
(additive, L4), noting relative-position x word correlations are a different matter; DeBERTa keeps both
cross terms, drops position-position, and its ablation shows both cross terms help. Net: in language
the evidence is split -- absolute-position x word mixing is noise, relative-position x word mixing is
useful.

**NoPE** (first-hand abstract and theorem statements, local; Haviv first-hand abstract and probe
section). Position is learned entirely from the causal mask; trained NoPE resembles T5 relative-bias
patterns.

**Barbero et al. 2025** (first-hand). Mean 2-norm of each RoPE 2D chunk of q and k per head in Gemma 7B:
most usage at the lowest frequencies; positional (diagonal, previous-token) heads identified by
high-frequency use; proves NoPE cannot form diagonal/off-diagonal patterns and RoPE can. p-RoPE
(drop the lowest 25% of frequencies) slightly improves perplexity on Gemma 2B. Learned separation
across channels, not within a channel.

**Urrutia et al. 2025, 2026** (first-hand defs/theorem/method; 2026 abstract+intro). Positional head:
logits invariant to permuting which content sits at which key position; symbolic: invariant to
moving content. Exclusion principle: a head can be both only if attention is near-uniform. Scores
from block swaps; applied per frequency. 2026: pure heads emerge as training succeeds; positional
mechanisms extrapolate worse than symbolic ones. Our L2/L3 are within-head, per content pair; their
scores are per head and permutation-based, so they work at any depth.

**Song & Zhong 2024** (first-hand). Mean decomposition of hidden states into position, context and
residual (explicitly "similar to two-way ANOVA"); positional basis low-rank and smooth, context basis
clustered, the two nearly orthogonal; QK split into pos-pos, pos-ctx, ctx-pos, ctx-ctx constituents.
"It is surprising that incoherence arises from automatic feature learning." This is the closest
published analogue to our probe; it works on residual-stream vectors, not on scores as a function of a
path-integrated phase, and has no untrained or imposed-separation control.

**Gu et al. 2026** (abstract + intro). Additive vs multiplicative positional forms; multiplicative (RoPE)
wins a task requiring "strong integration of positional and semantic cues"; a "single-head deposit
pattern" concentrates shallow positional processing.

**CSCG / latent sequence / schemas / TDB** (CSCG and latent-sequence first-hand, schemas and TDB
abstract). CSCG transfer: "We kept the transition matrix of the CSCG fixed, and re-initialized the
emission matrix to random values"; shortcuts in the new room follow. Separation of structure from
content is imposed at transfer time; within one room a clone is a conjunction (observation x context),
like TEM's `p`.

**Vector-HaSH** (abstract from Nature/Europe PMC; architecture from Gershman, Fiete, Irie 2025 Box 1,
local, secondary). "A pre-structured internal scaffold based on grid cell states is essential";
attractor-to-dense weights fixed and random, sensory weights Hebbian. Position is imposed; content is
attached.

**Successor representation** (abstract). Predictive map; value factorises into dynamics and reward.
Mechanism quoted from standard SR definition. Separation of structure from reward, not from content.

**Whittington et al. 2023** (first-hand). Theorem 1/2: with nonnegative activity, a variance (or exact
prediction) constraint and minimal activity energy, each neuron becomes selective for at most one
independent factor. Removing any constraint gives entangled codes; sparsity alone does not. Fig. 7:
"Only when the task is entangled are fields consistently warped towards objects" (objects fixed vs
moving across tasks). Prediction for us: learned separation should require the map to be redrawn
across training sequences.

**Cui et al. 2024** (abstract + intro). Tied, low-rank dot-product attention, teacher mixing positional
and semantic attention; the global minimum is positional below a sample-complexity threshold and
semantic above it -- a phase transition. Which separation is learned depends on data quantity, not
only architecture.

**Johnston & Fusi 2023; Lindsey & Issa 2024** (first-hand, partial). Abstract (factorised) codes emerge
from multi-task training. Lindsey & Issa define factorisation as the fraction of one parameter's
variance outside the other parameter's subspace, and use a "statistical lesion" (rotate subspaces to
overlap, all other statistics fixed) to show factorisation is what supports decoding.

**Locatello 2019; Higgins 2018** (abstracts). Unsupervised disentanglement is impossible without
inductive biases on model and data; disentanglement is best defined through group actions. Our
"learned" separation still rides on biases (path integration, rotation, the redrawn map).

## 4. Prior-art assessment (blunt)

- **"Separation is learned rather than imposed" is prior art.** MapFormer's own Fig. 9 shows it for these
  architectures at L1 (leak -> 0) and for values; Song & Zhong, TUPE, Barbero, Urrutia show learned
  position/content separation in trained LMs; Whittington 2023 and Cui 2024 give theory for when it
  happens. Stated generally, our conclusion would be a replication in a new regime, at best.
- **Our L1 numbers replicate MapFormer** (learned leak 0.025-0.05; opposite actions cancel). Say so.
- **What looks new**, after this search (no paper found that does any of these):
  1. Score-level, within-head variance decomposition (content pair x displacement) for content-dependent
     phase models, exact for 1 layer and verified against the model's logits.
  2. Untrained controls separating "imposed at init" from "learned": PoPE and MapPoPE ~0.12-0.14 untrained,
     MapWM 0.98 untrained, all 0.08-0.13 trained.
  3. The specific contrast "an architected decoupling (PoPE) does not give a cleaner trained endpoint than
     the unconstrained model (MapWM r=4 0.083 vs MapPoPE r=4 0.128)". PoPE's paper argues RoPE's
     entanglement hurts but never measures it in trained models. This is the one candidate-novel claim.
- **Weaknesses that a reviewer with this literature would raise.**
  1. Factorised-by-construction data (Whittington 2023): the separation may be a property of the task.
     Our own trained index RoPE also drops from 0.98 to 0.33; separation is partly learned without path
     integration.
  2. Language evidence (T-XL, DeBERTa) and our recency result say content x position interaction is
     useful when the task needs it; "separated is better" is not established anywhere, including here
     (no intervention).
  3. One layer, one task, post hoc, n=8, unpaired; the PoPE-vs-WM contrast points the "wrong" way and
     has no registered test.
  4. Our L2-L3 are not the field's standard metrics; reviewers will expect Urrutia's positional/symbolic
     scores or a Song & Zhong decomposition for comparability, and those work at depth.
- **TEM's own transfer test is the same as ours** (new arrangement of a fixed object set). New-object
  transfer goes beyond TEM, TEM-t and MapFormer -- if registered and powered. The content-free arm is
  not new: MapFormer's MapEM-s is one, and TEM-t is one by construction.

## 5. Proposals (cheapest first; all respect matched length, n=8, pre-registration)

**P1. Causal test of the measured separation, on stored checkpoints (CPU/GPU minutes, no training).**
Motivated by Lindsey & Issa's statistical lesion and Urrutia's frequency control. For each 1-layer
checkpoint (`runs/paper2x2/p0`, 56 checkpoints), evaluate with the score replaced by (a) its
position-only part (content-averaged `q_bar^T R(dtheta) k_bar`: the TEM-t / L5 surrogate) and (b) an
additive surrogate (position-only + a per-content-pair constant: L4), against (c) the unmodified score.
Converse lesion: add a controlled per-pair phase offset (RoPE-style `psi_k - psi_q`) of increasing size
to move kernel peaks, and read the accuracy dose-response. Registered primary: held-out-map accuracy at
T=128 (matched length), floor beside it. Verify first that the hooked forward reproduces the logits
exactly at zero intervention (rule 19). Turns "8-13% interaction" from descriptive into
necessary/unnecessary. Cost: an afternoon of code, minutes of compute.

**P2. Factorised vs entangled training data (GPU; the theory test).** Whittington 2023 predicts learned
separation only when content and position are independent across training. Arms: MapWM r=4 (learned)
and MapPoPE r=4 (imposed); conditions: fresh map per sequence (current) vs one fixed map vs a pool of 4
maps. Registered primary: the probe's interaction share and peak0 on trained checkpoints; secondary:
held-out-map accuracy (expected to collapse in the fixed-map arm -- that is the point, not a failure).
Branches: (i) MapWM separation rises under entangled data while MapPoPE's does not -> separation is
data-driven and PoPE's decoupling is what holds when data do not factorise (a real, new reason to
combine them); (ii) MapWM stays separated under a fixed map -> path integration itself drives it.
Free pilot first: the probe on the `RANK_WRAP` 2D high-wrap checkpoints (`runs/rank_wrap/N10/D2`), which
memorised their training map (0.986 on it, 0.273 on an unseen map; `RANK_WRAP_RESULTS.md`) -- a natural
entangled case; the probe's +/-16 displacement window must be cut to the 10-cell torus.
Cost: 2 arms x 3 conditions x 8 seeds = 48 runs, about one `PAPER2X2` batch (~3.5 h wall on two GPUs).

**P3. The literature's standard probes, at depth (CPU, hours).** Urrutia positional/symbolic scores
(block-swap permutations; per head and per frequency), Song & Zhong two-way decomposition with
incoherence on the residual stream, and Barbero per-frequency q/k norms, on the 1-layer torus
checkpoints (calibrate against our L2/L3) and on multi-layer ones (`runs/dyck_mdepth`, text world). Gives
comparability with published numbers and the depth > 1 readout our exact probe cannot. Cost: CPU, a
day including validation that the scores reproduce on a known positional head (rule 9).

**P4. Register the new-object test with the content-free arm, and probe its checkpoints (GPU).** The
pilot (`runs/newobj_pilot`, commit f35150e, n=1 per arm, NOT a result) reaches 0.976-0.983 on unseen
object codes for MapWM, MapEM, MapPoPE and PosOnly vs 0.39-0.44 for index RoPE/PoPE. Register it at
n=8 with PosOnly as the TEM-t-like reference (it is MapFormer's MapEM-s; add a tied `q0 = k0` variant to
match TEM-t's tied Q = K, and say that `model_tem_faithful.py` is told which tokens are actions, so its
source purity is given, not learned). Then run the score probe with unseen codes as keys: does L2/L3
separation measured on training objects hold on new ones? That links score-level separation to TEM's
L6 claim, which TEM itself never tested with new objects. Cost: ~6-7 arms x 8 seeds at 900 epochs,
~2 h per run at 2 jobs/GPU -> ~14 h wall on two GPUs.

Not proposed: new DeBERTa/TUPE-style architectures (the additive and two-stream forms are published and
answer a different question); re-deriving PoPE's algebra.

## 6. Sources

Local corpus (`papers/txt/`): `mapformer`, `pope`, `nope`, `rope` (eq. 5-7 for Shaw/T-XL), `kv_brain`,
`tale_two_algorithms`, `cope`, `grape`.

Downloaded and read for this note:
- TEM: https://europepmc.org/article/PMC/PMC7707106
- TEM-t: https://arxiv.org/abs/2112.04035
- Shaw: https://arxiv.org/abs/1803.02155 ; Transformer-XL: https://arxiv.org/abs/1901.02860
- TUPE: https://arxiv.org/abs/2006.15595 ; DeBERTa: https://arxiv.org/abs/2006.03654
- Haviv NoPE: https://arxiv.org/abs/2203.16634
- Round and round: https://arxiv.org/abs/2410.06205
- Urrutia 2025: https://arxiv.org/abs/2511.11579 ; Urrutia 2026: https://arxiv.org/abs/2605.31558
- Song & Zhong: https://arxiv.org/abs/2310.04861 ; Gu et al.: https://arxiv.org/abs/2505.13027 ;
  Give it Space!: https://arxiv.org/abs/2605.30022
- CSCG: https://europepmc.org/article/PMC/PMC8062558 ; Space is a latent sequence:
  https://arxiv.org/abs/2212.01508 ; Graph schemas: https://arxiv.org/abs/2302.07350 ; TDB:
  https://arxiv.org/abs/2401.05946
- Vector-HaSH: https://www.nature.com/articles/s41586-024-08392-y (abstract; preprint
  10.1101/2023.11.28.568960)
- SR: https://doi.org/10.1101/097170 (abstract)
- Whittington 2023: https://arxiv.org/abs/2210.01768 ; Dorrell 2023: https://arxiv.org/abs/2209.15563
- Cui 2024: https://arxiv.org/abs/2402.03902
- Johnston & Fusi: https://europepmc.org/article/PMC/PMC9950464 ; Lindsey & Issa:
  https://europepmc.org/article/PMC/PMC11226229
- Locatello: https://arxiv.org/abs/1811.12359 ; Higgins: https://arxiv.org/abs/1812.02230
