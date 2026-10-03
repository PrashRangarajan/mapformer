# What and where: does combining MapFormer and PoPE separate them? (2026-10-01)

Analysis, not a registered result. Equations are read from the code (file:line), not the papers.
One CPU probe on stored checkpoints (section 6); it is descriptive and post hoc, n=8 per arm, one
batch, no pre-registration. Probe: `docs/audits/2026-09-27/probe_whatwhere.py` (output `probe_whatwhere_out.txt`,
`probe_whatwhere.json`), re-run from the committed copy 2026-10-01 with byte-identical JSON.

Terms, defined once.
- **what**: the identity of a token (an object, a word, a bracket). **where**: a relational
  position variable (a cell on the torus, a stack depth) that should be the same for every
  object found there.
- **score** `a_ts`: the pre-softmax attention logit of query position t over key position s.
- **phase** `theta_t`: the rotary angle at position t. **step** `Delta_t`: the per-token increment
  that MapFormer sums into the phase, `theta_t = omega * sum_{u<=t} Delta_u`.
- **magnitude** `mu`: the non-negative per-frequency weight PoPE gets from content.
- **kernel**: the score as a function of the phase difference, for a fixed query/key content pair.
- **leak**: step size of observation tokens relative to action tokens (0 = only actions move "where").
- **index** models put the token count t into the phase; **path** models put the cumsum of steps.

## 1. Plain-language summary

**RoPE (index).** Position is the token count. Content and position meet in one cosine per
frequency pair, and the content vector carries its own phase, so what a token is can shift where
its attention lands ("attend 3 back" can become "attend 5 back" for some token pairs). Entangled
by construction; and "where" is a clock, not a map.

**NoPE.** No position code. Only the causal mask leaks order. Nothing to separate.

**PoPE (index).** Content sets only how strongly each frequency counts (magnitude, never negative);
position alone sets the phase, plus a learned constant offset per frequency. Content can make the
position kernel sharper or flatter but cannot move its peak (exactly true when the offsets are zero,
which on the torus they almost all are). That is a real what/where decoupling of the *score*, of a
specific kind: a content-weighted sum of fixed position kernels, not "content term times position
term". Its "where" is still the token count.

**MapWM (MapFormer working memory).** "Where" becomes a path integral of learned per-token steps,
and the model learns to give movement tokens a step and objects ~0. So the *variable* is
separated: where is computed from the action words. But then that variable is used RoPE-style on
content Q/K, so in the score what and where are re-entangled (content sets each frequency's
amplitude AND phase offset). The architecture allows full entanglement; trained 1-layer torus
models nonetheless converge to a nearly shared kernel (section 6).

**MapEM (MapFormer episodic memory).** Score = content attention times position attention,
elementwise, before the softmax. The position factor depends only on the phase difference; the
content factor only on the tokens. This is the closest of all to a literal factorisation, and it
is TEM's conjunction `g (x) x` written as one bilinear form. Two caveats: the product sits inside
the softmax (content acts as a signed gain on the position kernel, so a negative content score
turns "attend to the peak" into "attend to the trough"), and EM pays a search cost for its rigidity
(recency: `EM_WM_STATE.md`).

**MapPoPE (path phase + PoPE encoding).** Where = phase from the path integral; what = magnitudes
from content. On paper this is the cleanest *combination*: where is built from actions, and in
the score content cannot move the kernel's peak. It is still not a full separation: (i) the step
is computed from the token embedding, so "where" is built from "what" -- purity depends on the
learned leak, which is small after training but not zero; (ii) content still reshapes the kernel
(which frequencies count), so the score is a content-weighted kernel, not a product; (iii) with
more than one layer, the magnitudes come from a residual stream that already contains
position-mixed information.

**Selective RoPE generator / context gate / hidden-state step.** The step now depends on context
(a 4-token window, or a whole earlier attention layer). "Where" is no longer a function of the
action token alone. This is not a violation of separation if the context dependence is about
*whether a word is an action* ("she did not go north" should not move you); it is the right
generalisation of "where is computed from actions" when actions are not marked. It becomes a
violation only if context makes the step depend on *which objects* are around.

**Decay envelope.** An additive penalty `-lambda * distance`, content-free. Adds a pure-where term.
In the path version the distance is the absolute difference of a 1-D average of the step channels,
which on a 2-D torus is a projection, not a displacement norm.

**The answer.** Combining MapFormer and PoPE does the cognitive-map separation *in part*, at two
different places, and it is not the full TEM separation:
- (a) **Architecture.** MapFormer separates the *source* of where (a path integral of steps that
  training can confine to action tokens). PoPE separates the *use* of where in the score (content
  cannot shift the kernel's peak). MapPoPE has both; MapWM has only the first; PoPE only the
  second; MapEM has the first and a stricter score factorisation than PoPE. None has TEM-t's
  strict form, in which content is absent from the score and enters only through the values
  (`positional_review.tex:891-895`).
- (b) **Trained models (torus, 1 layer, probe below).** Every trained path model ends with a
  score that is ~87-92% a shared function of the displacement, ~0-2% content alone, 8-13%
  content x position interaction, and (rank 4, or MapPoPE) peaks at the correct cell for every
  content pair. Trained MapWM r=4 is as separated as MapPoPE (interaction share of the
  position-dependent variance 0.083 vs 0.128): **the separation is something training finds; PoPE
  imposes it at initialisation (0.14 untrained vs 0.98 for WM) but does not make the trained
  endpoint cleaner.** Learned leak is 0.025-0.05 (observations step at 2.5-5% of actions).
- (c) **Unmeasured.** Whether the separation is what *buys* anything (no intervention that forces
  or breaks it has been run); generalisation to new object sets; any multi-layer model; any
  language task. The project's held-out-map evaluations are a transfer test of where across a new
  arrangement of known objects, not across new objects.

## 2. The equations, from the code

All models: `h = LN(x)`, `q = W_q h + b_q`, `k = W_k h + b_k` (bias on: `nn.Linear` default,
`model.py:213`), `d_h = 64`, 2 heads, scores divided by `sqrt(d_h)`. In a 1-layer model `x` is the
token embedding alone, so `q` and `k` are functions of token identity only. Per 2-D rotary block b
write `q_b = |q_b| e^{i psi^q_b}` (the content's own phase).

| scheme | score `a_ts` (per head) | code | depends on |
|---|---|---|---|
| RoPE (index) | `sum_b |q_b||k_b| cos(omega_b (s - t) + psi^k_b - psi^q_b)`, `omega_b = 10000^{-b/32}`, 32 blocks | `model_baseline_rope.py:49,58-66` -> `model.py:238-239,264` | content of t and s (amplitude AND phase); index lag |
| NoPE | `q_t . k_s` | `model_baseline_nope.py:30-42` | content only; order only via the causal mask |
| PoPE (index) | `sum_{c=1}^{64} softplus(q_c) softplus(k_c) cos(omega_c (t - s) - delta_c)`, `omega_c = 10000^{-c/64}`, `delta_c in [-2pi, 0]` | `model_pope.py:115,121` (phase), `:61-70` (score) | content -> magnitudes only; index lag -> phase; `delta_c` content-free constant |
| MapWM | RoPE's form with `theta_t = omega (*) sum_{u<=t} W_out W_in emb(x_u)` in place of `omega t` | `model.py:83` (step), `:120-121` (cumsum), `:238-239` (rotate content Q,K), `:264` | content of t, s (amplitude and phase); path phase driven by ALL tokens' embeddings |
| MapEM | `A_X * A_P`, `A_X = q_t . k_s` (no rotation), `A_P = q0^T R(theta_s - theta_t) k0` | `model.py:346-349` (rotate q0, k0), `:397-401` | `A_X`: content only; `A_P`: phase difference only (separate `q0`, `k0`) |
| MapPoPE | PoPE's form with per-element phase `phi_t = omega (*) sum_{u<=t} W_out W_in emb(x_u)`, 64 channels per head | `model_pope.py:80-86` (`_widen_to_d`), `:94-100`, `:61-70` | content -> magnitudes; path phase (from all tokens' embeddings) -> phase; `delta_c` |
| Selective-RoPE generator (our port) | MapWM's score; `theta_t = e^tau sum_{u<=t} sigmoid(W_g x_u) (*) sum_{j<4} a_j W_omega x_{u-j}` | `model_selective.py:84-92,108-116` | step depends on a 4-token window (the gate multiplies lagged terms) |
| Context gate (CG) | MapWM r=4 score; `Delta_t = g_t(x_{t-3..t}) W_out W_in emb(x_t)`, one scalar gate per head | `model_context_step.py:35-58` (`:48`) | step = word's own step x a window-dependent on/off |
| Hidden-state step (HS) | layer 2 = MapWM with `Delta_t = W_out W_in LN(h1_t)`, `h1` = an index-RoPE layer's output | `model_context_step.py:60-91` (`:82`) | step depends on the whole prefix through attention |
| HS residual (HSR) | `Delta_t = W_out W_in (emb(x_t) + alpha LN(h1_t))`, `alpha` init 0 | `model_context_step.py:93-112` | word's own step + a learned context correction |
| Decay envelope | PoPE score `- softplus(lambda_h) * dist`; `dist = |t - s|` (index) or `|S_t - S_s| / mean|step|`, `S = cumsum(mean_blocks Delta)` (path) | `model_pope_decay.py:42-66` (`:54`), `:76-89` (`:83`), `:99-110` (`:106`) | adds a content-free term in index or 1-D-projected path distance |

Not in the code, from the corpus (`papers/INDEX.md`): GRAPE (2512.07805) -- the two-slot frame
(multiplicative rotation + additive bias), and it calls its own multiplicative extension
"non-contextual"; Mamba-3 (2603.15569) -- `Diag(A(t) + i theta(t))`, both decay and rotation
data-dependent (Prop. 3: data-dependent RoPE equivalence); CARoPE (2507.23083) -- positive bounded
accumulated step, rank 1 per head. All three set the phase from content like MapWM; none adds a
PoPE-style magnitude/phase split, so for this question they sit in MapWM's row.

Algebra worth stating (all exact for 1 layer):
- MapWM: write `q = q^c + b_q` (content part plus bias). Then
  `a_ts = q^c R k^c + q^c R b_k + b_q R k^c + b_q R b_k`, `R = R(theta_s - theta_t)`. The last term is a
  content-free position kernel (an EM-like `A_P` with `q0 = b_q, k0 = b_k`), the first is fully
  entangled. WM is not additive (`AUDIT_2026-09-10.md` #1); this expansion just says it *contains*
  a pure-where term and an entangled term, and training decides their balance.
- PoPE: `a_ts = sum_c w_c(content) kappa_c(dphi)` with `w_c = mu^q_c mu^k_c >= 0` and
  `kappa_c = cos(dphi_c - delta_c)`. With every `delta_c = 0`, every `kappa_c` peaks at `dphi = 0`, so
  any non-negative mixture peaks there: **content cannot move the peak**. With unequal non-zero
  `delta_c` the peak of the mixture moves with the weights, so content can move it again, by at most
  the spread of the `delta_c`. On the torus checkpoints 91-100% of the clamped `delta_c` sit at the
  bound 0 (max |delta| 0.11-1.59; `ABLATE_RESULTS.md` found 80.7% in the clamp on its task).
- PoPE is not `f(content) x g(position)`. It is a product inside a sum over 64 frequencies: content
  chooses a non-negative weighting over a fixed bank of position kernels.
- MapEM: `A_X (*) A_P = (q^x (x) q^p) . (k^x (x) k^p)`, one bilinear form on the tensor product
  (`positional_review.tex:868-877`). Softmax of a product is not the product of two softmaxes:
  `exp(A_X A_P)` makes `A_X` an inverse temperature (with sign) on the position kernel. Sign is a
  gauge per head (`A_X (*) (-kappa) = (-A_X) (*) kappa`, `EM_WM_STATE.md` Sec 2).

## 3. What "separating what from where" means for an attention score

TEM's claim (MapFormer's framing, `papers/txt/mapformer.txt:57-69`): a cognitive map "factorizes
content from position" so a learned structure applies to new observations. MapFormer's own App.
reading (`mapformer.txt:1182-1185`): EM "factorizes content and position in two separate neural
spaces"; WM "represent[s] position implicitly ... directly update[s] the conjunction of what and
where". So by MapFormer's own account **MapWM does not factorise in the score**; its factorisation
is in the path integrator, where trained observation steps go to ~0 (`mapformer.txt:2135-2140`).
PoPE's claim (`papers/txt/pope.txt:785-795`): RoPE lets "aspects of the key and query ...
dynamically shift the position tuning of a component"; PoPE removes that.

Operational levels, from weakest to strongest. A scheme can satisfy one and fail another.

| level | name | test on a score | what passes by construction |
|---|---|---|---|
| L1 | **source purity** | the where-variable depends only on action tokens: leak = 0 | index models (trivially: no content at all); path models only if trained to leak 0 |
| L2 | **peak invariance** | for every content pair, the kernel peaks at the same displacement (content cannot move where attention goes) | PoPE with `delta = 0`; MapEM up to the per-head sign gauge |
| L3 | **shape sharing** | every content pair uses the same kernel up to a (signed) scale: rank 1 after centring rows | MapEM (exactly) |
| L4 | **additive separation** | `a = f(content) + g(position)`: zero interaction | ALiBi/decay term alone; nothing here as a whole |
| L5 | **TEM-t form** | the score depends on where only; what enters through values | none of the implemented models |
| L6 | **behavioural transfer** | a where learned with one content set applies to another (new map, new objects) | -- (an outcome, not a property) |

L2-L4 matter for retrieval; L1 is the "where is computed from actions" claim; L6 is what TEM says
the factorisation is *for*.

## 4. Each scheme against the definition

| scheme | L1 source | L2 peak | L3 shape | L4 additive | notes |
|---|---|---|---|---|---|
| RoPE | n/a (index) | no | no | no | content phase `psi` shifts the lag kernel per pair |
| NoPE | n/a | n/a | n/a | n/a | no where |
| PoPE | n/a (index) | **yes** if `delta = 0` | no (content reweights frequencies) | no | decouples the score, but where is a clock |
| MapWM | learned (leak -> ~0) | no (allowed); learned | no (allowed); learned | no | where separated at the source, re-entangled in the score |
| MapEM | learned | yes up to sign | **yes** | no (multiplicative) | strictest factorisation; rigid (search cost) |
| MapPoPE | learned | **yes** if `delta = 0` | no | no | source + peak; content still sets sharpness |
| Selective / CG / HS | learned, now context-dependent | as MapWM | as MapWM | no | relaxes L1 from "token" to "context" |
| + decay | -- | adds a content-free term | -- | the decay term is additive | adds an L4 component |

Intuitions:
- **RoPE.** Each query/key pair is a dial pair; the score is high when the dials line up after
  rotating by the lag. Content sets where each dial starts, so different words "line up" at
  different lags. What can push where.
- **PoPE.** The dials are fixed to start at zero; content only sets how loud each dial is. All dials
  line up at lag 0 (shifted by the shared `delta`), so loudness changes how sharp the peak is, not where
  it is. What can change how *confidently* you look somewhere, not *where* you look.
- **MapWM.** The dials are turned by movement instead of by time, and the model learns that only
  movement words turn them. Where is built from the action words. But it then reads the dials with
  content-set starting points, exactly as RoPE does, so the conjunction of what and where is formed
  directly in the score (MapFormer's own description of WM).
- **MapEM.** Two separate questions -- "is this the right kind of thing?" and "is this the right
  place?" -- multiplied. The position answer is the same function for every pair. A negative answer to
  the first flips the second (peak becomes trough); on the torus this happens per head, uniformly over
  content (a gauge), not per pair (section 6).
- **MapPoPE.** Where from movement (MapFormer) + content cannot move the peak (PoPE). "The step comes
  from token content" means the purity of where is not guaranteed: an object token with a non-zero
  step would move the map every time it is seen. Training drives that to 2.5-5% on the torus. In a
  real vocabulary, where the same word is sometimes a move and sometimes not, a token-only step
  cannot be pure -- which is the context-step line.
- **Context steps.** If "north" moves you only in "walked north", then a pure where *must* depend on
  context. A context-dependent step is the correct generalisation of L1 (where depends on what
  *happened*, read from context), and the pilots show its limits: window-limited steps (CG,
  Selective) suppress decoys only within their window; HS reaches far cues when it learns a step at
  all (`CTXSTEP_PILOT3.md`, `CTXSTEP_HS_RECIPE.md`; pilots, n=1-2 per cell, no result). It would
  violate separation if the step came to depend on object identity; nothing measures that yet.

## 5. Evidence from the project (non-withdrawn results only)

- **Path integration helps where the task is a map; PoPE's encoding helps on top.** Torus, training
  length, `PAPER2X2_RESULTS.md`: RoPE 0.805, PoPE-Flat 0.679, MapWM r=2 0.971, MapPoPE r=2 0.999
  (n=8). Position main effect +0.243 (8/8); r=2 interaction +0.154 (8/8, raw). At r=4 MapWM is at
  1.000, so the encoding cannot show at T=128. PoPE *hurts* the index model on navigation (-0.126):
  decoupling a clock from content does not give a map.
- **PoPE's encoding helps the path row; path integration hurts PoPE on clock tasks**
  (`.claude-memory/project_mappope_asymmetry.md`, report "How the two ideas combine"): MapPoPE -
  MapWM Bach -0.0165 NLL (5/5), Dyck 1L/2L +0.073 / +0.050; MapPoPE - PoPE on code +0.0033 (0/3).
  Whether the path-row gain is larger (the 2x2 interaction) is NOT established on these tasks.
  Reading for this question: when where is genuinely relational, stopping content from shifting it
  (PoPE) helps; when where is a clock, replacing the exact clock with a learned one costs a little.
- **WM is not additive; EM's single kernel is real; per-pair freedom helps on recency.**
  `EM_WM_THEORY.md` 1b (after its bug fixes): kernel phase spread across content pairs, recency
  checkpoints -- EM 0.000 (one kernel for every pair), WM trained 1.947, WM untrained 2.722, uniform
  null 3.267. So WM's training *reduces* per-pair spread. `PAIRSPLIT_RESULTS.md` (n=48): giving EM
  per-pair position origins (a content-dependent kernel) buys +0.215, of which freedom +0.091. On a
  task whose retrieval offset varies by query, **strict L3 separation costs accuracy** (via search:
  EM - WM -0.375, the frozen installed solution 1.000; `EM_WM_STATE.md`).
- **Where is built from the movement words, learned from prediction alone.** `TEXTWORLD_RESULTS.md`:
  on 8/8 seeds opposite directions cancel after removing a common component, and zeroing the
  direction words' steps drops every seed to the floor. On 4/8 seeds the common component is a real
  step clock (31-38 of 64 phase channels drift between visits to the same cell): there the where
  variable mixes position with elapsed time. That is an L1 failure of a different kind -- not what
  leaking into where, but *when* leaking into where -- and it did not cost accuracy.
- **Match-Query (in-context maps)**: path 0.730 vs index 0.154, context destruction 0.918 -> 0.074
  (`MATCH_QUERY_SCALE.md`, `MATCH_QUERY_RESULTS.md`): the map is built in context from the action
  stream, as TEM requires.
- **Transfer.** Every navigation number here is measured on a held-out observation map (evals redraw
  the map; CLAUDE.md "The project"). That is L6 for a *new arrangement of the same 16 objects*; new
  objects are untested.

Not used as support (withdrawn): the T2 / accumulator + kernel account of the MapPoPE collapse and
the PoPE-decoupling corollary (`T2_RESULTS.md`), "MapWM is additive / OR-gate" and Thm 3, the
WM-vs-EM regime table, "r=2 loses because its basis is skewed".

## 6. Probe: how separated are trained scores? (CPU, stored checkpoints)

**What it measures.** For a 1-layer model the layer-1 score is exactly a function `S(a, o, d)` of
the query token, the key token and the phase difference. Grid: query = the 4 action tokens (the
loss is on the observation following an action), key = the 17 observation tokens (16 objects +
blank) -> 68 content pairs; position = torus displacement `d = (dx, dy)`, |dx|,|dy| <= 16 (1089 cells),
phase built from the model's OWN action steps along a minimal path, plus the key token's own step
(as in the cumsum). Index models: position = token lag 1, 3, ..., 255. Readouts per head on the
68 x P matrix: shares of variance (position main, content main, interaction = residual of the
additive fit), `inter/pos` = interaction / (position + interaction) (0 = content cannot change the
kernel at all), `shape1` = top singular value energy of the row-centred matrix (1 = one shared kernel
up to signed scale, L3), `peak0` = fraction of content pairs whose argmax is d = 0 (L2), `leak` =
mean omega-scaled step norm of observation tokens / action tokens (L1), `opp` = |step(N)+step(S)| /
|step(N)|. Untrained controls: 3 random inits of each architecture.

**Verified (rule 9).** The score function re-implements the layer code; the full forward rebuilt
from it reproduces `model(tokens)` logits with max abs difference 0.0 on all 56 checkpoints and 21
inits; deleting the rotation from it gives 17.7-25.7 (so the check can fail). Positive control:
MapEM's `shape1` is 1.000 when trained (leak ~0), as the algebra requires.

Checkpoints: `runs/paper2x2/p0/*` (the `PAPER2X2_RESULTS.md` batch, torus paper task, 300 ep cosine,
n=8 per arm); MapEM r=4 separate `q0/k0` from `runs/dof/torus/VanillaEM_r4_s*` (a different batch:
no cross-batch accuracy comparison is made). Head = the head with the larger position-dependent
variance (both-head means are within 0.07 of these on every row).

| arm | pos | content | inter | inter/pos | shape1 (L3) | peak0 (L2) | leak (L1) | opp N/S |
|---|---|---|---|---|---|---|---|---|
| RoPE (index lag) | 0.661 | 0.008 | 0.331 | 0.334 +/- 0.190 | 0.735 | -- | -- | -- |
| RoPE untrained | 0.010 | 0.427 | 0.563 | 0.983 | 0.190 | -- | -- | -- |
| PoPE-Flat (index lag) | 0.867 | 0.014 | 0.119 | 0.120 +/- 0.045 | 0.889 | -- | -- | -- |
| PoPE-Flat untrained | 0.829 | 0.105 | 0.066 | 0.074 | 0.939 | -- | -- | -- |
| MapWM r=2 | 0.861 | 0.018 | 0.120 | 0.125 +/- 0.138 | 0.993 | 0.432 | 0.167 | 0.509 |
| MapWM r=2 untrained | 0.014 | 0.257 | 0.728 | 0.981 | 0.175 | 0.000 | 1.038 | 8.35 |
| MapWM r=4 | 0.899 | 0.019 | 0.082 | **0.083 +/- 0.022** | 0.982 | **1.000** | 0.045 | 0.083 |
| MapWM r=4 untrained | 0.014 | 0.266 | 0.720 | 0.981 | 0.179 | 0.000 | 0.999 | 3.17 |
| MapPoPE r=2 | 0.918 | 0.000 | 0.082 | 0.082 +/- 0.075 | 0.971 | **1.000** | 0.046 | 0.333 |
| MapPoPE r=2 untrained | 0.824 | 0.044 | 0.132 | 0.138 | 0.874 | 0.255 | 0.975 | 0.970 |
| MapPoPE r=4 | 0.872 | 0.000 | 0.128 | **0.128 +/- 0.029** | 0.934 | **1.000** | 0.025 | 0.055 |
| MapPoPE r=4 untrained | 0.846 | 0.040 | 0.115 | 0.120 | 0.891 | 0.382 | 1.080 | 1.54 |
| MapEM r=4 (dof batch) | 0.640 | 0.287 | 0.074 | 0.104 +/- 0.012 | **1.000** | **1.000** | 0.051 | 0.112 |
| MapEM r=4 untrained | 0.017 | 0.366 | 0.617 | 0.977 | 0.649 | 0.000 | 0.999 | 3.17 |

Readings (descriptive; no registered test, so none is a "result"):
1. **Every converged path model ends close to separated.** About 87-92% of the score's variance is
   a shared function of displacement; content alone ~0-2% (except EM, where content gates the
   whole kernel, 29%); 8-13% interaction. At rank 4 (and MapPoPE r=2) every content pair's kernel
   peaks at the correct cell (L2 holds 68/68 on 8/8 seeds). Leak 2.5-5%: observation tokens barely
   move the map (L1 nearly holds), matching MapFormer's own Fig. 4 claim.
2. **PoPE imposes it; WM learns it.** Untrained, MapWM's position-dependent variance is 98%
   interaction (content owns the kernel); MapPoPE's is 12-14%. Trained, both sit at 8-13%. MapPoPE r=4
   is not cleaner than MapWM r=4 (0.128 vs 0.083; unpaired, post hoc, if anything the other way).
   So "combine with PoPE to get separation" is true at initialisation and not at the trained
   endpoint on this task.
3. **MapEM's sign "flip" is a per-head gauge here.** The fraction of pairs with `A_X < 0` is exactly
   0 or 1 per head on every seed (4 of 16 heads all-negative), and those heads have the trough of
   `A_P` at d = 0; peak0 is 1.000 counting that. Content does not invert the kernel pair by pair on the
   torus (on recency it does: 55% of solved large-k cells take the trough route, `EM_WM_THEORY.md`).
4. **Residual interaction is not obviously bad.** Within MapPoPE r=2 the interaction is bimodal:
   0.000-0.016 on seeds 1, 4, 5, 7 (final loss 0.002-0.023) and 0.14-0.19 on seeds 0, 2, 3, 6 (final
   loss 0.0001-0.0002). All eight are at the accuracy ceiling (0.999), so this is fit, not accuracy;
   n=8, post hoc. In MapWM r=2 the two unconverged seeds (loss 0.16, 0.38) carry the high interaction
   (0.31-0.82); there entanglement tracks failure. Content modulating kernel *sharpness* (allowed by
   PoPE) is not the same thing as content moving the *target* (forbidden by PoPE), and only the latter
   breaks retrieval.
5. **MapWM r=2's low peak0 (0.43) is a where-code property, not a what/where one.** Its kernel is
   shared (shape1 0.993) but on 6/8 seeds peaks off d = 0 within the window: the learned rank-2 steps
   are near-collinear (seed 2: cos(N, E) -0.995 / -0.989 per head), so other displacements alias the
   origin. Not an explanation of r=2's accuracy (that claim is withdrawn); just what the probe sees.

Scope: torus only, 1 layer, T=128 recipe, queries = action tokens, keys = observation tokens (action
keys and the "attend to observations, not actions" gating are outside the grid), window +/-16 cells.
Untrained controls n=3.

## 7. What is unmeasured

- **Whether separation causes anything.** No arm forces or breaks it at fixed everything else. The
  only intervention is EM's per-pair origins on recency (more freedom, +0.091), which goes *against*
  strict separation there.
- **New objects (L6 proper).** All transfer here redraws the map over the same 16 object tokens.
- **Depth > 1.** The probe's exactness and PoPE's "content sets only magnitudes" both rely on layer-1
  Q/K being token functions; at depth the magnitudes come from a residual stream that carries position.
- **Language.** Bach, code and Dyck checkpoints were not probed (multi-layer or no map structure).
- **Context steps.** Whether CG/HS steps stay object-independent; no batch has reported (pilots only).
- **The decay envelope's path metric** collapses a 2-D displacement to a 1-D average; untested on 2-D.
- **delta.** Nearly inert on the torus (91-100% at the bound 0); its effect on L2 elsewhere unmeasured.

## 8. Proposed tests (cheapest first)

1. **Commit and extend this probe (CPU, minutes).** Add action keys; add the text world (per-word
   step table already exists, `TEXTWORLD_PROBE.json`) and the 1-layer Dyck checkpoints
   (`runs/dyck_mdepth`) with position = stack depth.
2. **Swap tests on trained models (CPU).** (a) Fixed positions, permute object identities across
   cells (a new map): the attention *pattern over positions* should be unchanged if L2-L3 hold --
   this is already implicit in held-out-map evals. (b) Fixed content, shift all positions by a
   constant displacement: scores unchanged for a relational where (trivially true for path models
   by translation invariance; the useful version is shifting by a displacement *never seen in
   training*).
3. **New-object transfer (GPU, needs registration).** Observation embeddings drawn fresh per sequence
   (frozen random codes, or a held-out set of object tokens with embeddings tied to a random code
   book); test on unseen codes. [Corrected 2026-10-03: this is NOT TEM's tested claim -- TEM and TEM-t
   generalise to new ARRANGEMENTS of a fixed object set, never to unseen objects (`docs/lit/LIT_NEW_OBJECTS.md`).] Arms: RoPE, PoPE, MapWM, MapPoPE, MapEM, plus a
   TEM-t-style arm whose score has no content term (Q = K = position only, content in V) as the
   strict-separation reference.
4. **Enforce vs relax separation (GPU).** MapPoPE with `delta` frozen at 0 (L2 exact) vs free; MapWM
   with the content part of Q/K unrotated (an additive content + position-kernel score, L4) vs
   standard. Registered primary: held-out-map accuracy at matched length (rule 10).
