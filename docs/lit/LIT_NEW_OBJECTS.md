# New objects and content leaking into structure: literature (2026-10-03)

Context: the new-object pilot (`runs/newobj_pilot/*/eval.json`, n=1 per arm, seed 100). Objects are
fixed random 64-d Gaussian codes (`model_codes.codebook`, seed 12345) entering through a learned
linear encoder `A` and a learned linear readout; each sequence draws 16 of a 1000-code pool and a
fresh map; test codes come from a disjoint 1000-code pool from the same Gaussian
(`environment_newobj.py`). Terms: **leak** = a nonzero path-integration step from an object token
(content moving the "where"); **content floor** = accuracy with object-token steps zeroed at eval
(what is left when the where-channel is made exact; the audit zeroed object steps only, the
redesign also zeroes blank steps); **leak cost** = content floor minus accuracy
as trained.

## Summary

1. **The pilot's transfer result is known and was engineered in.** A copy/binding mechanism
   trained on many fresh random-vector fillers transfers to unseen vectors from the same
   distribution: Chen et al. 2019 (N(0,1) 50-d unit-norm fillers; "unlimited filler training"
   generalises, a 6-filler pool scores 0%), Reddy 2023 (novel Gaussian classes, D=63, via an
   induction head, above a class-count threshold), Boix-Adsera et al. 2024 (unseen symbols need data
   diversity), Lazic et al. 2026 (freezing (un)embeddings plus diversity fixes unseen tokens),
   Kirsch et al. 2022 and Raventos et al. 2023 (diversity thresholds). Fixed codes through a shared
   linear encoder ARE the published remedy (frozen embeddings): with 1000 iid codes spanning R^64, a
   test code is a new point in an already-trained linear map. The pilot's object-identity transfer
   could not fail.
2. **The test can be made to fail only by moving the code distribution, not its identity.** With a
   linear encoder and readout, an object's identity is irrelevant; what matters is where the test
   codes sit relative to the training codes' span, covariance and norm. Held-out feature combinations
   do not change that for additively or tensor-composed codes whose span is already covered. Tests in
   the literature that do fail: disjoint held-out entities with few training entities (ESBN m=95,
   Chen limited fillers, Reddy low K), corrupted or rescaled objects (Abstractor random linear maps
   and additive noise; TCN scale and translation regions; positional attention value range x c),
   held-out primitives in composition (SCAN add-jump 1.2%, COGS 16-35%).
3. **Separating structure from content as the remedy is the standard idea**: TEM / TEM-t, ESBN,
   CoRelNet, Abstractor, DAT, Vector-HaSH, CSCG schemas, Syntactic Attention, the relational
   bottleneck review. Where that separation is complete, as in TEM-t, Vector-HaSH and CSCG, the
   position update is **driven by actions only** and content cannot reach it by construction. Our
   PosOnly arm copies only half of TEM-t's remedy (the position-only score) and keeps MapFormer's
   learned, all-token step, so the leak is expected to survive. It does.
4. **What looks new, and is modest:** a measurement of content -> structure leakage in a *learned*
   content-dependent path-integration step. MapFormer reports observation steps of about 0 only
   qualitatively (its Fig. 3d and Fig. 9) and suggests energy constraints. I found no paper that
   quantifies the leftover step or tests it under a shift in content statistics. Our audit gives a
   step gain of 0.08 per object versus 20.3 per action (MapPoPE). The leak grows linearly with code
   norm because `ActionToLieAlgebra` reads the pre-LayerNorm embedding, while LayerNorm shields the
   score and value paths. Zeroing object steps restores about 1.000. TEM and TEM-t never tested
   unseen stimuli (both use one-hot codes over a fixed set of 45 objects). So "novel-object
   transfer in a path-integrating transformer" is untested in that family, but the literature
   predicts its outcome.
5. **Redesign:** keep the codes, drop "new identity" as the manipulation, and register **code-
   statistics shifts** (anisotropic-covariance swap, norm, random linear map, sparse codes). Report
   the leak cost and the content floor separately. Compare against arms whose step is action-only
   by construction (the TEM-t step) or norm-invariant or energy-penalised. Then the "where" claim
   can fail on its own terms. Costs are in the final section.

## Table

"Can fail": whether the paper's novel-entity test has an outcome under which the model fails, and
whether some model in the paper actually fails it.

| paper | id | novel-entity construction | can the test fail | result | leak remedy | relevance |
|---|---|---|---|---|---|---|
| MapFormer (Rambaud et al.) | 2511.19279 | none: 16 fixed objects, new maps | n/a for objects; the map test fails for RoPE | obs. steps about 0 after training (Fig. 3d, 9); qualitative only | none enforced; learns the partition; suggests energy constraints | the architecture whose step leaks |
| TEM (Whittington et al., Cell 2020) | doi 10.1016/j.cell.2020.10.024 | new environments = new random arrangement (with replacement) of the SAME 45 one-hot stimuli | yes for structure (loop closure on first visit); unseen stimuli never tested | transfers structure to new arrangements | g updated by action-dependent W_a; sensory enters g only through memory retrieval in inference | the "TEM claim" is new arrangements, not new objects |
| TEM-t (Whittington, Warren, Behrens) | 2112.04035 | same as TEM, one-hot | as TEM | spatial codes like TEM | e_{t+1} = sigma(e_t W_a), action-only; Q=K from position only, V from stimulus only | full two-part remedy; PosOnly has only the score half |
| Vector-HaSH (Chandra, Sharma, Chaudhuri, Fiete) | bioRxiv 2023.11.28.568960; Nature 638 (2025) | new rooms with new sensory cues; random or mini-ImageNet patterns | capacity and recall fail for learned scaffolds (Fig. 2d) | zero-shot recall on novel paths; 11 rooms in sequence without forgetting | fixed (unlearned) grid scaffold, updated by velocity only; content bound by Hebbian heteroassociation | leak impossible by construction; learned keys hurt |
| CSCG graph schemas (Guntupalli et al.) | 2302.07350 | new rooms = same latent graph relabelled with new observations | yes (schema mismatch) | fast transfer on MPG and One-Shot StreetLearn | action-conditioned latent transitions; observations only through a re-learned emission matrix | structure frozen, content re-bound |
| ESBN (Webb, Sinha, Cohen) | 2012.14601 | m of 100 Unicode images withheld (m in 0,50,85,95); test only on withheld | yes: LSTM, NTM, RN, Transformer fail at m=95 | ESBN >= 95% on all 4 tasks and regimes; Transformer hurt by a random-projection encoder, ESBN not | controller sees only keys; image embeddings only as values | key/value split = where/what split |
| CoRelNet (Kerg et al.) | 2206.05056 | held-out shapes (pentominoes -> hexominoes, stripes); unseen relations | yes | similarity-matrix-only models generalise; adding sensory features to the decoder input DEGRADES OOD | partition: decoder sees only the T x T similarity matrix | direct evidence that content leak costs OOD |
| Abstractor (Altabaa, Webb, Cohen, Lafferty) | 2304.00195 | iid Gaussian objects; test objects corrupted by a random linear map Phi or additive noise | yes: Transformer degrades faster | Abstractor more robust over sigma | relational cross-attention: Q,K from objects, V = learned input-independent symbols | the corruption sweep is a ready-made code-statistics shift |
| DAT (Altabaa, Lafferty) | 2405.16727 | n/a (sample efficiency) | n/a | n/a | separate sensory and relational heads; positional or position-relative symbols | position as a content-free pointer |
| Relational bottleneck review (Webb et al.) | 2309.06629 | review | n/a | n/a | restrict downstream flow to relations; calls for a graded version | framing; cites TEM's MEC/LEC split |
| TCN (Webb et al.) | 2007.05059 | analogies tested in regions graded by distance from the training region, by translation and by scale | yes, graded | context normalisation improves extrapolation | normalise embeddings within a sequence | scale shift = our norm stress |
| Chen et al. (role-filler binding) | 1902.09006 | fillers = N(0,1) 50-d vectors, unit norm; limited pool (6 per role) vs a fresh vector per story | yes: limited-pool training -> 0% on unseen fillers (chance 2.3%) | unlimited-filler training generalises (memory-augmented nets) | external memory | **closest prior art to the pilot's design** |
| Boix-Adsera et al. | 2310.09753 | unseen tokens (unseen symbols) in template tasks | yes: MLPs fail; transformers fail to COPY unseen symbols at large d_emb | transformers generalise with enough diversity; +aI on W_K W_Q^T and +bI on W_V W_O^T fix it | copy-friendly reparametrisation | explains why per-token learned embeddings fail and shared codes do not |
| Lazic et al. (To See the Unseen) | 2604.21632 | unseen variable names | yes | unseen-token unembeddings collapse to one vector | copy architecture + diversity + frozen or reset (un)embeddings | our fixed codebook is this remedy |
| Reddy | 2312.03002 | novel classes: means redrawn from the same Gaussian, D=63 | yes: below a class-count / burstiness threshold, in-weights learning wins | induction head forms abruptly; generalises to novel items above the threshold | none | same "new = fresh iid Gaussian" design |
| Chan et al. | 2205.05055 | Omniglot holdout classes assigned to labels in context | yes | ICL needs burstiness plus many rare classes | none | diversity drives transfer |
| Kirsch et al. (GPICL) | 2212.04458 | tasks = random projections + label permutations; test on unseen tasks AND other datasets | yes: memorisation regime below a task count | generalises past a transition; transfers MNIST -> FashionMNIST/CIFAR10 | none | separates "unseen draw" from "different dataset" |
| Raventos et al. | 2306.15063 | regression tasks; finite pretraining pool vs Gaussian prior | yes: below the diversity threshold it acts as the discrete-prior Bayes estimator | above the threshold it matches ridge on new tasks | none | diversity threshold |
| Olsson et al. (induction heads) | 2209.11895 | repeated sequences of random tokens | yes (prefix-matching, copying scores) | induction heads copy arbitrary tokens | none | copying arbitrary content is a generic attention capability |
| Song, Xu, Zhong | 2408.09503 | copying under a shifted token distribution and longer length | yes: one layer fails, two layers succeed | OOD via composition; shared "bridge" subspace | none | |
| Positional attention (de Luca et al.) | 2410.01686 | values extended to [-2c,2c]; k-hop induction on an unused alphabet | yes: positional attention fails k-hop on an unused alphabet | positional-only attention generalises in value scale where the algorithm is positional | attention weights from positions only, values carry content | = our PosOnly; norm OOD is its test |
| SCAN (Lake, Baroni) | 1711.00350 | add-primitive: "jump" seen only in isolation | yes | best 1.2%, overall-best 0.08% | none | held-out primitive |
| Syntactic Attention (Russin et al.) | 1904.09708 | SCAN add-jump | yes | 78.4 mean / 91.0 median (+/-27.4) vs GRU+attn 12.5 | separate syntax (attention) and semantics (context-free word -> action) streams; sequential semantics hurts | separating streams helps; the leak direction matters |
| Li et al. (primitive substitution) | 1910.02612 | SCAN jump, turn-left | yes | 14.0 -> 98.8% jump | two representations (attention map vs symbol mapping) + entropy regularisation | |
| Lake, meta seq2seq | 1906.05381 | primitive -> meaning assignment permuted every episode; test assignment withheld | yes | 99.95% on add-jump | episodic memory; arbitrary bindings per episode | = fresh bindings per sequence |
| COGS (Kim, Linzen) | 2010.05465 | lexical (known primitive in a new role) vs structural | yes | 16-35% gen.; lexical easier, structural near 0 | none | |
| Kim, Linzen 2022 | 2212.10769 | context-controlled words replaced by novel strings or novel embeddings | yes | pretrained T5 overestimated; novel embeddings worst | none | "unseen" must really be unseen |
| TUPE (Ke, He, Liu) | 2006.15595 | n/a | n/a | n/a | untie word and position correlations in the score | score-level leak only |
| Song, Zhong (hidden geometry) | 2310.04861 | n/a | n/a | position and context means nearly orthogonal in trained LMs | none (measurement) | a decomposition to measure leak in hidden states |
| Give it Space (Lequeu et al.) | 2605.30022 | n/a | n/a | n/a | three explicit streams (semantic, AP, RP); loss on semantic only | separate streams in an encoder |
| Barbero et al. (Round and round) | 2410.06205 | n/a | n/a | low RoPE frequencies act as semantic channels | p-RoPE removes them | content and position share RoPE channels |
| Whittington et al. (biological constraints) | 2210.01768 | n/a | n/a | nonnegativity + activity/weight energy -> single-factor neurons | soft constraints | the learned-remedy arm |

## Per-paper notes

Labels: **first-hand** = PDF text read in the relevant sections. **abstract** = abstract or
intro only. **local** = in `papers/txt/`.

**MapFormer** (local, first-hand). Lists "disentangle actions from observations" as requirement 1
of the navigation task: the model "isn't told which tokens should update the cognitive map".
Fig. 9a: perfect accuracy arrives "as soon as" observation steps approach 0. Fig. 9 caption:
"other constraints, such as bounded energy [42], could be added to force disentanglement". No
quantification of the leftover observation step, and no content shift.

**TEM** (first-hand, UCL open-access PDF, methods). Its stimuli: "ns = 45 (the number of different
sensory objects)", one-hot, compressed to two-hot; "sensory stimuli are chosen randomly, with
replacement, at each node". A new environment changes size and arrangement, not the object set.
Hebbian M resets between environments. The transition of g is action-dependent (W_a). In the
inference model, g is corrected from a memory retrieved by the sensory observation. That is a
content -> structure path, but an error-correcting one: retrieval indexes a stored g, not the
object's features. **TEM never tests unseen stimuli.** `docs/WHAT_WHERE_ANALYSIS.md` section 8.3
calls new-object transfer "TEM's actual claim". It is not: TEM's claim is structure transfer
across arrangements, which our redraw-map evals already test.

**TEM-t** (first-hand, section 2). Quote: "keys and queries only focus on position encodings.
Meanwhile, values are exclusively dependent on the stimulus"; "e_{t+1} = sigma(e_t W_a), where W_a
is a learnable action-dependent weight matrix". Both halves of the separation are architectural.

**Vector-HaSH** (first-hand: spatial-memory section; plus Box 1 of `kv_brain`, local). "In a novel
room, we randomly initialize grid module phases, and velocity inputs to each module then update the
grid phases through path integration". Sensory cues reach the scaffold only through the
hippocampal heteroassociation (bidirectional recall). The scaffold is fixed: "learned to make the
grid-hippocampal pattern pairs self-consistent, the scaffold capacity collapses" (Fig. 2d). A
structure that is not learned cannot absorb content.

**CSCG graph schemas** (first-hand, abstract and section 3). "The goal of transfer is to reuse the
topology of a previously learned graph to model a new environment by relabeling the nodes with new
observations." Transitions are conditioned on actions; observations enter only through emissions,
which are re-learned per room. George et al. 2021 (Nat. Commun.) not read.

**ESBN** (first-hand). n=100 Unicode images, m in {0,50,85,95} withheld, test only on withheld.
"These two pathways only interact indirectly via a key/value memory." With a random-projection
encoder, ESBN still generalises; the Transformer is "significantly impaired". Related work: Chen et
al. 2019 found NTM / fast weights generalise to novel entities "when allowed a sufficiently dense
sampling of the space of potential objects (the 'objects' in their study were randomly sampled
50-dimensional vectors)". That sentence describes our pilot.

**CoRelNet** (first-hand, sections 3-5). Fig. 4 left: concatenating sensory information to the
relational input "degrades the OoD generalization capacity of the models". This is the clearest
published case of content leaking into the relational path and costing OOD accuracy. Random
(frozen) encoders do as well as learned ones.

**Abstractor** (first-hand, section 2 and Appendix robustness). Relational cross-attention:
Attention(Q <- X, K <- X, V <- S), with S learned symbols that do not depend on the input. OOD
test: objects corrupted at test by Phi_ij ~ N(0, sigma^2) or by additive N(0, sigma^2 I). The
Abstractor degrades more slowly than a Transformer. Justification: <Phi x, Phi y> ~ <x, y> in high
dimension. Its objects are iid N(0, I) in R^32 (order task) or Cartesian products of attribute
codes (48 objects).

**DAT** (abstract and the symbol-assignment section). Position or position-relative symbols serve
as content-free pointers.

**Relational bottleneck review** (first-hand). Definition: "any mechanism that restricts the flow
of information from perceptual to downstream reasoning systems to consist only of relations". It
names the hippocampal binding of MEC (structure) and LEC (content) as a candidate substrate. Open
question it raises: a "graded formulation that controls the amount of non-relational information
allowed to pass through the bottleneck". Our leak is one such graded quantity.

**TCN** (first-hand, sections 1-3). The VAEC benchmark tests extrapolation by distance from the
training region, along translation and scale. The remedy, temporal context normalisation,
normalises across items within a sequence.

**Chen et al. 2019** (first-hand). Fillers: "Each index of the vector is independently drawn from a
N(0, 1) distribution, and then the vectors are normalized to have unit Euclidean norm". Limited
pool (six fillers per role), tested on a disjoint pool: "the test accuracy of each network
remained at 0%", because networks "always predict fillers from the training set". Unlimited
(fresh per story): generalises, for networks with external memory.

**Boix-Adsera et al.** (first-hand, sections 1, 3, 4). Unseen symbols are tokens absent from
training. Theorem 4.1: with tied learned embeddings, early-training updates to W_V W_O^T lie in the
span of the training tokens' embeddings, nearly orthogonal to an unseen token's direction at large
d_emb, so copying fails. Adding bI to W_V W_O^T fixes it. Our design side-steps this failure: codes
pass through one shared encoder, so a test code has no per-token parameters to be left untrained.

**Lazic et al. 2026** (abstract and intro). The unembeddings of unseen tokens collapse to nearly
one vector. The fix combines a copy architecture, diversity, and frozen or periodically reset
(un)embeddings. Our readout scores (B h) . c with fixed codes c, which is the frozen-unembedding
version.

**Reddy 2023** (first-hand, setup). Items: x = (mu_k + eps eta)/sqrt(1+eps^2), with mu_k Gaussian,
D=63. "Novel classes (the mu_k's are drawn anew)". Above a threshold in K, B and eps, the
induction head forms and transfers. Below it, in-weights learning wins. Our task removes the
in-weights competitor (fresh map and fresh objects per sequence), so no threshold should appear at
P=1000.

**Chan et al. 2022** (grep-level). Holdout Omniglot classes "never encountered in training". ICL
needs burstiness and a large set of rare classes.

**Kirsch et al. 2022** (first-hand, sections 1-3). Tasks are random projections of inputs plus
label permutations. Transitions run memorisation -> task identification -> general learning, with
task count. Meta-test sets include different datasets (FashionMNIST, CIFAR10). That distinction,
"unseen draw from the same generator" versus "different generator", is the one the pilot lacks.

**Raventos et al. 2023** (abstract). Below a diversity threshold, the model is Bayes-optimal for the
discrete pretraining prior and fails on new tasks.

**Olsson et al.** (grep-level). Induction heads are defined and measured on "repeated random
sequence[s] of tokens". The tokens are in the vocabulary; the sequences are random.

**Song, Xu, Zhong 2024** (abstract). OOD copying requires two-layer composition. One layer only
"weak-learns".

**Positional attention** (first-hand, section 7). Attention weights come only from positional
encodings; values carry content. OOD means values extended by a scale factor c, or an alphabet
unused in training (k-hop induction). It generalises in scale where the algorithm is positional.
It fails k-hop induction on an unused alphabet, a task needing content-dependent routing.

**SCAN** (first-hand, numbers). Add-jump: "The best performance was 1.2% ... The overall-best model
reached 0.08%".

**Syntactic Attention** (first-hand). Add-jump 78.4 mean / 91.0 median vs GRU+attn 12.5, GRU+attn-dep
0.7, CNN 69.2. Ablation: "performance on the jump-split test set was worse when the strict
separation ... was violated by allowing sequential information to be processed in the semantic
stream"; letting syntax reach the output directly did not hurt. **Separation helps against leaks
in one direction only**: structure into content hurt, content alongside structure did not. That
matches CoRelNet's direction (sensory into the relational decoder hurts).

**Li et al. 2019** (abstract). Two representations plus entropy regularisation: jump 14.0 -> 98.8%.

**Lake 2019 meta seq2seq** (first-hand, section 4). "Each meta-training episode provides a
different random assignment of the primitive instructions ... to their meanings". 99.95% on
add-jump. This is the per-sequence fresh binding our task already uses.

**COGS** (first-hand, numbers). In-distribution about 99%, generalisation 16-35%. Lexical
generalisation (a known primitive in a new position) is easier than structural.

**Kim, Linzen 2022** (abstract). Replacing controlled words with novel embeddings degrades T5 more
than novel character strings do. An "unseen" item must be unseen in every channel.

**TUPE** (abstract). Untied word-word and position-position correlations in the score. This is a
score-level separation and leaves untouched a step-level leak like ours.

**Song, Zhong 2023** (abstract). h = mu + pos_t + ctx_c + resid; position and context components
are nearly orthogonal. Applied to MapFormer, the same ANOVA decomposition would measure leak in the
residual stream rather than in the step.

**Give it Space** (abstract). Three explicit streams (semantic, AP, RP), with the loss on the
semantic stream only. An architectural separation in a language encoder.

**Barbero et al.** (grep-level). Gemma uses low RoPE frequencies as "semantic channels"; p-RoPE
truncates them. Content and position share the rotary channels in index RoPE too, in the score
rather than in the step.

**Whittington et al. 2023** (abstract). Nonnegativity plus activity/weight energy yield
single-factor neurons. This is the soft, learned alternative to an action-only mask, and the
reference MapFormer cites.

Searched and not found: a paper that quantifies the observation-token step in MapFormer-like or
other content-dependent path-integration encodings (CoPE 2405.18719, Selective RoPE 2511.17388,
CARoPE 2507.23083, Mamba-3 2603.15569 are content-dependent by design; none of the local copies
analyses a leak under content shift). Searched and not found: a TEM-family model tested on unseen
stimuli. The search was not exhaustive. Say "not found", not "does not exist".

## Prior-art assessment of the pilot (blunt)

| pilot finding | status |
|---|---|
| Path arms name unseen objects (about 0.98-0.99 object-identity on both pools) | **Known; not a result.** Chen 2019 is the same design. Reddy, Boix-Adsera and Lazic explain why: fixed codes through a shared encoder are the frozen-embedding remedy. iid test codes are in distribution. Usable only as a sanity check. |
| RoPE / PoPE fail | **Not about new objects.** They fail on TRAIN objects too (PoPE object acc 0.10 / 0.13 train / test per the audit; overall 0.51-0.58 against the always-blank floor of about 0.5 on revisits). This is MapFormer's Table 2 and our `PAPER2X2_RESULTS.md`. |
| Train-test gap is blank calibration (test codes copied with higher gain) | A finite-codebook artefact. The encoder and readout fit the sample covariance of 1000 codes in 64 dims (relative error about sqrt(64/1000), roughly 0.25). Fresh codes per sequence (Chen's "unlimited" regime) would remove it. Minor; not citable. |
| Residual error is "what" leaking into "where" (object steps; zeroing them -> about 1.000) | **Not found in prior work as a measurement.** MapFormer asserts steps of about 0 without quantifying them. The direction of the effect is predicted by CoRelNet (content in the relational path hurts OOD). |
| Norm stress x2/x4 drops path arms (object acc 0.95 / 0.85-0.87 for MapPoPE, MapEM; about 0.93 / 0.81 in the audit summary), index arms invariant | Mechanism (step linear in the pre-LN embedding; LN shields score and value) **plausibly new as a statement about MapFormer**. The test itself is standard (Abstractor noise and linear maps, TCN scale, positional attention value range). Counts as robustness, not capability (rule 10): no matched control yet. |
| A content-free score (PosOnly) does not remove the leak | **Near-tautological.** TEM-t's separation has two parts, an action-only position update and a position-only score. PosOnly has only the second, and the leak lives in the first. One sentence, not a headline. |

Net: the transfer headline must not be claimed. What is worth registering is the leak: its size,
its scaling, the content shifts under which it bites, and whether an enforced or regularised
separation removes it at matched length.

## Redesign: a new-object test that can fail

Design principle, from the table: with a linear encoder and readout, "new object" means **new code
statistics** (span, covariance, norm, sparsity), not a new identity. Every condition reports two
numbers, so a failure can be attributed:

- **content floor C** = object-identity accuracy (objects only, no blank competitor) with
  object-token AND blank steps zeroed at eval: the "what" channel alone;
- **leak cost L** = C minus object-identity accuracy as trained: the "where" cost of content.

Reference arms, all retrained in one batch (rule 12):
- `ActOnly`: steps computed only for action tokens (ids 0-3); Delta := 0 for blank and objects by
  construction (TEM-t's step). L = 0 by construction. It hands over the token partition that
  MapFormer learns, so it is a reference, not a competitor.
- `NormStep`: `ActionToLieAlgebra` reads LayerNorm(x) rather than x. This removes norm sensitivity
  only.
- `DeltaL1`: the MapFormer step plus lambda * mean ||Delta_t||_1 (the energy constraint of
  Whittington 2023 and MapFormer Fig. 9). A learned remedy; lambda is set by a pilot.

Code-statistics shifts (test codes always from a disjoint pool; train with fresh codes per sequence
so no codebook covariance is fitted):
- `ID`: same distribution as training (the control; must give L about 0 on remedied arms).
- `SWAP`: train on anisotropic N(0, Sigma), with 16 strong dims (var 1) and 48 weak dims
  (var eps^2, eps = 0.25), rescaled to E||c||^2 = 64. Test with strong and weak swapped (matched
  E||c||^2). Weak directions get little gradient pressure to be suppressed in the step, so a
  learned separation may not cover them. This is the shift that can fail for "where" while the
  encoder still carries the content (eps > 0). Gate on CPU before launch: the ideal-copy accuracy
  given the encoder's gain.
- `NORM` x2, x4 (exists; robustness).
- `RANDMAP`: c -> Phi c, Phi_ij ~ N(0, sigma^2 / 64), sigma in {0.5, 1, 2} (the Abstractor sweep).
- `SPARSE`: k-sparse +/-1 codes, k = 8, norm-matched (a heavy-tailed shift).

Rejected: held-out feature combinations (Abstractor-style A x B objects, or outer-product codes).
With a linear encoder and readout, any held-out combination whose span is covered by the training
combinations is in distribution again. That test cannot fail, for the same reason as the pilot.

Registered primary (per experiment below): L on the named shift, at training length, n=8; the
exact-t MDE and a permutation p from `stats_core.py`; floors C and the always-blank predictor
reported beside it.

## Experiments, cheapest first

Cost calibration: the pilot ran 6 runs (900 ep, T=1024, batch 16) as one wave on 2 GPUs in about
2.5 h, i.e. about 2.4 runs/h at 3 jobs per GPU.

**E0. Leak decomposition on the pilot checkpoints (CPU, under 1 h, no training).**
Eval-only on `runs/newobj_pilot/*/<arm>.pt`. NORM sweep x {0.5, 1, 2, 4, 8}; RANDMAP sigma sweep;
SPARSE; plus an attribution probe: test codes at matched norm placed in the top-r right-singular
subspace of the trained object-step map M = W_out W_in A versus its complement. Report C and L per
cell. n=1, so this is descriptive only. It decides which shifts give a nonzero L and sizes the
effect for the MDE. Expected: L tracks the component of c along M's row space; complement-placed
codes give L about 0. If L about 0 for every non-norm shift, the leak is purely a norm effect, and
NormStep is the whole remedy.

**E1. Enforced vs learned separation at matched distribution (GPU, registered; 48 runs, about 20 h).**
Arms: MapWM, MapWM-ActOnly, MapWM-NormStep, MapWM-DeltaL1, PosOnly, PosOnly-ActOnly (= TEM-t exactly),
with 8 seeds each, trained on the iid pool. Primary: test-pool object accuracy at training length,
ActOnly minus MapWM. Expected about +0.01 to +0.02, near the MDE: the floor C is about 0.9995, so
headroom is small. Report it as unmeasured if it falls below the MDE. Secondary: NORM x4, labelled
robustness. This is WHAT_WHERE section 8.4 in the new-object setting.

**E2. Code-statistics shift, the test that can fail (GPU, registered; 24 runs, about 10 h).**
Anisotropic training as specified above. Arms: MapWM, MapWM-ActOnly, MapWM-NormStep; 8 seeds.
Conditions: ID, SWAP, RANDMAP sigma=1, SPARSE. Primary: L on SWAP, MapWM vs ActOnly. Branches are
fixed before launch. L(SWAP) above the MDE and L(ID) about 0 means the learned separation is
statistics-specific, and "where" depends on "what" outside the training content distribution.
L(SWAP) about 0 means the learned step suppresses content generically. NormStep shows whether the
dependence is norm only.

**E3. Diversity sweep, the TEM regime (GPU; 24 runs at n=4, about 10 h; extend the boundary to n=8).**
P in {16, 64, 1000} train codes, test on a disjoint pool; arms MapWM and MapWM-ActOnly. P=16 is the
paper's torus regime (the same 16 objects in every sequence), i.e. "does a MapFormer trained the
TEM way name an object it never saw". At P=16 < 64 the learned encoder sees a 16-d span. AdamW
decay shrinks A on the unseen 48 dims by about exp(-lr * wd * steps), roughly 0.01 here. So C is
expected to collapse (Chen's 0%, Raventos' threshold). L tells whether the step on those dims also
decayed, giving no leak, or kept its init, giving a leak. A content-channel failure here is the
prior art's result. Only the L column is ours.

Order: E0 first (free). It decides whether E2's SWAP is worth its 10 h, or whether NORM plus
NormStep in E1 already covers the leak.

## Sources

- MapFormer: https://arxiv.org/abs/2511.19279 (local `papers/txt/mapformer.txt`)
- TEM: https://discovery.ucl.ac.uk/10115119/7/1-s2.0-S009286742031388X-main.pdf ; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7707106/
- TEM-t: https://arxiv.org/abs/2112.04035
- Vector-HaSH: https://www.biorxiv.org/content/10.1101/2023.11.28.568960v1.full.pdf ; https://github.com/FieteLab/VectorHaSH ; key-value memory review https://arxiv.org/abs/2501.02950 (local `kv_brain.txt`)
- Graph schemas (CSCG): https://arxiv.org/abs/2302.07350
- ESBN: https://arxiv.org/abs/2012.14601
- CoRelNet: https://arxiv.org/abs/2206.05056
- Abstractor: https://arxiv.org/abs/2304.00195 ; DAT: https://arxiv.org/abs/2405.16727
- Relational bottleneck: https://arxiv.org/abs/2309.06629
- TCN: https://arxiv.org/abs/2007.05059
- Chen et al. 2019: https://arxiv.org/abs/1902.09006
- Boix-Adsera et al.: https://arxiv.org/abs/2310.09753
- Lazic et al.: https://arxiv.org/abs/2604.21632
- Reddy: https://arxiv.org/abs/2312.03002 ; Chan et al.: https://arxiv.org/abs/2205.05055
- Kirsch et al.: https://arxiv.org/abs/2212.04458 ; Raventos et al.: https://arxiv.org/abs/2306.15063
- Olsson et al.: https://arxiv.org/abs/2209.11895 ; Song, Xu, Zhong: https://arxiv.org/abs/2408.09503 (PNAS: https://www.pnas.org/doi/10.1073/pnas.2417182122)
- Positional attention: https://arxiv.org/abs/2410.01686
- SCAN: https://arxiv.org/abs/1711.00350 ; Syntactic Attention: https://arxiv.org/abs/1904.09708 ; Li et al.: https://arxiv.org/abs/1910.02612 ; meta seq2seq: https://arxiv.org/abs/1906.05381
- COGS: https://arxiv.org/abs/2010.05465 ; Kim, Linzen 2022: https://arxiv.org/abs/2212.10769
- TUPE: https://arxiv.org/abs/2006.15595 ; Song, Zhong: https://arxiv.org/abs/2310.04861 ; Give it Space: https://arxiv.org/abs/2605.30022 ; Barbero et al.: https://arxiv.org/abs/2410.06205
- Whittington et al. 2023: https://arxiv.org/abs/2210.01768

Pilot numbers quoted here: `runs/newobj_pilot/*/eval.json`, plus the 2026-10-02 audit probe
(object/blank split, norm stress, zeroed steps, step gains; scratchpad output, not committed). All
n=1.
