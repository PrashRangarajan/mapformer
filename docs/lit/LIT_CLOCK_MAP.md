# Clock vs map: the literature (2026-10-12)

Literature review, not a result. Prior-art check for a candidate paper framing: "a learned additive
(path-integrated) phase is a MAP when increments cancel and a CLOCK when they do not; it beats the token
index exactly when the task needs a map; on clock-like data the model learns a clock that does no better
than counting tokens; a signed phase can emulate a clock, a monotone one cannot be a map." Nothing was run.

Read before searching: `papers/INDEX.md` (corpus, grepped, not re-searched), `docs/lit/LIT_*.md`,
`.claude-memory/reference_positional_landscape.md`, `.claude-memory/project_clock_vs_map.md`,
`docs/theory/2026-10-04/02_literature.md`, `docs/theory/2026-10-05/neuro_positional.md`,
`mapformer_math.tex` (sign prior-art paragraph), `positional_review.tex` sec. clockmap.

Read status per entry: **first-hand** = method / result passages read as text in this session (local
corpus or downloaded); **first-hand (corpus)** = read first-hand in an earlier session per `papers/INDEX.md`
or `LIT_*.md`, specific passages re-grepped here; **abstract** = abstract or search summary only;
**secondary** = known only through another source.

Terms. **step** Delta_t: per-token increment of the phase, theta_t = omega (*) sum_{u<=t} W_out W_in emb(x_u).
**signed** = steps of either sign (can cancel). **monotone** = steps of one sign (|Delta|, softplus,
sigmoid, CoPE's gate): cannot cancel. **interval sum** = theta_s - theta_t, the only positional quantity a
rotary score sees. **map** = interval sum is net displacement (same place, different times collide).
**clock** = interval sum is path length / elapsed (selected) token count. **matched length** = trained and
tested at the same T. **Case I/II/III** = Howard et al. 2014's three settings of the integrator rate alpha(t).

## 1. Summary (blunt)

1. **The dichotomy itself is published, in neuroscience, 12 years ago.** Howard et al. 2014 (J Neurosci)
   set one integrator's rate alpha(t) to a constant (Case I: time), to speed |v| (Case II: "the total
   distance traveled along the path"; route-dependent, retrospective coding), or to signed velocity
   (Case III: "although the states of F(s,tau) change over time, they are always the same for a
   particular position x. Like a bead on a wire ..."). That is index / monotone / signed, clock / path
   length / map, with the collision property stated in words. Burgess-style VCO models have the same
   split (phase relative to baseline tracks signed displacement). The repo already knew this
   (`02_literature.md` sec. 2(b)); a paper must cite it as the source of the framing, not as an analogy.
2. **"Monotone cannot cancel, signed can" is published in ML as a theorem** (Sarrof 2024: nonnegative
   gates cannot do PARITY; Grazzi 2025: positive eigenvalues cannot; Khavari 2025: the sign must sit in
   the same input-dependent layer), carried into content-dependent rotations by Selective RoPE (sec.
   4.2, "flips") and Mamba-3 (data-dependent RoPE solves parity, standard RoPE does not), and stated for
   navigation by MapFormer itself: CoPE fails "even in 1D" because it cannot "represent an action (moving
   forward) and its inverse (moving backwards)"; MapFormer's Fig. 9 shows opposite actions cancel. Signed
   counters for Dyck depth are textbook (counter machines; Weiss 2018; Bhattamishra 2020; Sarrof Thm 5-6;
   Yao 2021).
3. **"Index position is a clock; structured tasks want a task-structured position" is TEM-t's
   Discussion** (Whittington, Warren, Behrens 2022): sinusoids "represent a linear structure"; positional
   encodings "should reflect the abstract underlying properties of the task"; learned by path
   integration in space. NoPE models learn a token count from the causal mask (Haviv 2022). Gates as
   learned clock rates: chrono init (Tallec & Ollivier 2018), Puranik 2026.
4. **"Content-dependent phase buys little on language at training length" is consistent with published
   numbers but not stated by anyone as a clock claim.** HGRN Table 11 (data-dependent phase worse),
   HGRN2 Table 1, Selective RoPE Table 3 (perplexity/accuracy trade "in a model-dependent way"),
   CARoPE Table 1 (at the training length 512: 21.31 -> 21.23 and 29.33 -> 28.99 ppl, single runs; the
   large gains are at 1024, i.e. OOD). **Counter-evidence:** CoPE reports -5% code perplexity and gains on
   Wikitext vs RoPE (20M-scale, in distribution) with a monotone, selective clock. So "a learned clock does
   no better than counting tokens" is contradicted for a *selective* clock at CoPE's scale.
5. **What is left that we did not find published:** (i) the sign axis isolated as one operation
   (|Delta| vs Delta, identical parameters, matched length) on navigation; (ii) one sign constraint run on
   a map task and a recency task in the same architecture (crossover); (iii) a growth-exponent readout
   (range ~ T^alpha, 0.59 vs 0.97) of an unconstrained learned accumulator; (iv) learned step tables on
   real text/code/music decomposed into a shared (clock) component; (v) a matched-depth exchange rate
   (1 path layer ~ 3 attention layers on Dyck). All are empirical, narrow (small models, one recipe), and
   each sits next to a published qualitative version.
6. **The cross-field identification** (Howard's alpha(t) = selective-SSM Delta_t = content-dependent
   rotary step; sign axis = Case II vs Case III) was not found stated anywhere. It is one sentence once
   Howard, Mamba and Puranik are on the table; a reviewer can call it obvious.

## 2. Table: what each source covers

Columns: framing pieces. **F1** cancellation decides map vs clock; **F2** monotone cannot be a map /
signed needed for state; **F3** signed can emulate a clock; **F4** content position beats index on
map-like tasks (in-distribution); **F5** on clock-like data learned position ~ index; **F6** path
integration substitutes for depth; **F7** cross-field (time cells / place cells).

| source | id / venue | read | F1 | F2 | F3 | F4 | F5 | F6 | F7 |
|---|---|---|---|---|---|---|---|---|---|
| Howard, MacDonald, Tiganj, Shankar, Du, Hasselmo, Eichenbaum 2014, "A unified mathematical framework for coding time, space, and sequences in the hippocampal region" | J Neurosci 34:4692; PMC3965792 | first-hand | **yes** (Cases I-III) | yes (speed codes path, not place) | implicit | -- | -- | -- | **yes** |
| Howard & Hasselmo 2020, "Cognitive computation using neural representations of time and space in the Laplace domain" | 2003.11668 | abstract | yes (review) | -- | -- | -- | -- | -- | yes |
| Whittington, Warren, Behrens 2022 (TEM-t), "Relating transformers to models and neural representations of the hippocampal formation" | 2112.04035, ICLR 2022 | first-hand (sec. 2-3, Discussion) | partial (index = "linear structure"; PI = space) | -- | -- | yes (spatial tasks, architected) | conjecture the opposite for language ("grammar") | -- | **yes** |
| Rambaud, Mascarenhas, Lakretz, MapFormer | 2511.19279 v4 | first-hand (corpus) | partial | **yes** (CoPE cannot represent inverse; opposite actions cancel, Fig. 9) | -- | yes (Table 2, architecture-level) | -- | yes ("emulate a stack", 1 layer) | yes (grid cells) |
| Sarrof, Veitsman, Hahn 2024, "The expressive capacity of SSMs: a formal language perspective" | 2405.17394, NeurIPS 2024 | first-hand (corpus) | -- | **yes** (Thm 2; gates nonnegative "due to exponential or sigmoid parameterizations") | -- | -- | -- | partial (Dyck: layer 1 counts up/down, layer 2 flip-flop) | -- |
| Grazzi, Siems, Zela, Franke, Hutter, Pontil 2025, "Unlocking state-tracking in linear RNNs through negative eigenvalues" | 2411.12537, ICLR 2025 | first-hand (corpus: abstract, theorems) | -- | **yes** (proof; [0,1] -> [-1,1]) | yes ([-1,1] contains [0,1]) | -- | LM "comparable" at 1.3B, **but** DeltaNet [-1,1] improves code and math perplexity | -- | -- |
| Khavari et al. 2025, "Parity requires unified input dependence and negative eigenvalues in SSMs" | 2508.07395 | abstract | -- | yes (same layer) | -- | -- | -- | no (stacking does not substitute) | -- |
| Merrill, Petty, Sabharwal 2024, "The illusion of state in state-space models" | 2404.08819, ICML 2024 | abstract | -- | adjacent (TC0; input-dependent transitions fix) | -- | -- | -- | -- | -- |
| Movahedi et al., Selective RoPE | 2511.17388, ICLR 2026 | first-hand (corpus: sec. 4.2, Table 3) | -- | **yes, in our exact cell** (signed content rotation; parity with 1 layer) | -- | partial (parity, MQAR; near train length) | yes (LM mixed) | partial (1 layer parity) | -- |
| Lahoti et al., Mamba-3 | 2603.15569 | first-hand (corpus: Prop. 3, Table 5b) | -- | yes (data-dependent RoPE solves parity; std RoPE 1.56%) | -- | OOD only (train <=160, test 256) | no RoPE ablation on LM | -- | -- |
| Zhang et al., GRAPE | 2512.07805, ICLR 2026 | first-hand (corpus, partial) | -- | implicit (contextual additive slot needs g(x) >= 0) | -- | -- | LM only | -- | -- |
| Veisi et al., CARoPE | 2507.23083 | first-hand (corpus: eq., Table 1) | -- | (monotone, in (0,1); cannot reverse or stay) | -- | -- | **yes** (+0.4%/+1.2% ppl at train length, single runs; large OOD gain) | -- | -- |
| Golovneva et al., CoPE | 2405.18719 | first-hand (corpus: tasks, Table 5) | partial (a selective counter) | -- | -- | yes, but on clock tasks (counting, selective copy, Flip-Flop in-dist) | **against** (code -5% ppl vs RoPE) | -- | -- |
| Yang et al., PaTH | 2505.16381 | first-hand (corpus) | -- | yes (beta in (0,2) "allows negative eigenvalues", cites Grazzi) | -- | yes (Flip-Flop in-dist 0% vs RoPE 6.9%) | LM gain at 760M | yes (1 layer Flip-Flop) | -- |
| Qin et al., HGRN / HGRN2 | 2311.04823 / 2404.07904 | first-hand (corpus) | -- | -- | -- | -- | **yes** (data-dependent phase worse; complex gain was state size) | -- | -- |
| Puranik, Jane Street blog | 22 Apr 2026 | first-hand (corpus) | partial ("learn how far to advance time") | -- | -- | -- | -- | -- | -- |
| Tallec & Ollivier 2018, "Can recurrent neural networks warp time?" | 1804.11188 | first-hand (corpus, via LIT_CONTEXT_STEPS) | partial (gate = learned clock rate) | -- | -- | -- | -- | -- | -- |
| Haviv et al. 2022, "Transformer LMs without positional encodings still learn positional information" | 2203.16634, Findings EMNLP | abstract (+ probe section per LIT_WHAT_WHERE) | -- | -- | -- | -- | partial (a count is learnable from the mask) | -- | -- |
| Liu, Ash, Goel, Krishnamurthy, Zhang 2023, "Transformers learn shortcuts to automata" | 2210.10749, ICLR 2023 | abstract | partial (gridworld via prefix sums of signed moves) | -- | -- | -- | -- | **yes, qualitatively** (depth vs recurrence; gridworld in 2 layers) | -- |
| Bhattamishra, Ahuja, Goyal 2020 | 2009.11264, EMNLP | abstract | -- | partial (Dyck-1 by attention-averaged signed count) | -- | -- | -- | partial (1 layer for counter languages) | -- |
| Yao, Peng, Papadimitriou, Narasimhan 2021 | 2105.11115, ACL | abstract | -- | partial (layer 1 computes depth by attending to all previous tokens) | -- | -- | -- | **yes** (D+1 layers hard; 2-layer generator) | -- |
| Weiss, Goldberg, Yahav 2018 | 1805.04908, ACL | abstract | -- | yes (squashing/GRU cannot count; LSTM/ReLU can) | -- | -- | -- | -- | -- |
| Heo, Ngotiaoco, Irie, Gershman, Sabatini 2026, "Why depth matters in parallelizable sequence models: a Lie algebraic view" | 2603.05573 | abstract | -- | -- | -- | -- | -- | yes (error falls exponentially with depth) | -- |
| Vanjani et al. 2026, ReCAT | 2609.35200 | abstract | adjacent (additive updates best for counting/timing, delta rule best for spatial recall) | -- | -- | -- | -- | -- | -- |
| Kraus et al. 2013 / 2015; Abramson et al. 2023 (eLife 83930) | Neuron 78:1090; Neuron 88:578 | abstract / secondary | time vs distance cells; task decides which dominates | -- | -- | -- | -- | -- | yes |
| Tsao et al. 2018, "Integrating time from experience in the LEC" | Nature 561:57 | abstract | time from experience (LEC) vs space (MEC) | -- | -- | -- | -- | -- | yes |
| Buzsaki & Tingley 2018, "Space and time: the hippocampus as a sequence generator" | TICS 22:853; PMC6166479 | secondary (full text blocked) | argues the hippocampus is blind to whether a sequence is spatial or temporal | -- | -- | -- | -- | -- | yes |
| Burgess 2008 (VCO); Gao, Xie, Wei, Zhu, Wu 2021 (rotation path integration) | Hippocampus 18:1157; 2006.10259 | secondary / abstract | signed velocity -> phase = displacement (given velocity, not learned from content) | -- | -- | -- | -- | -- | yes |
| GridPE | 2406.07049 | abstract | -- | -- | -- | -- | -- | -- | grid-cell PE, fixed, index-driven |

Searched and not found (2026-10-12, WebSearch standard/extended + Semantic Scholar citations of
2411.12537, 111 papers by title): a paper that (a) uses "clock" vs "map" (or time vs space, count vs
displacement) as a named dichotomy for positional encodings; (b) ablates only the sign of a learned
positional increment at matched length; (c) runs one sign constraint on both a map task and a
recency/clock task; (d) fits a growth exponent to a learned positional accumulator; (e) decomposes learned
content-dependent step tables on language/code/music into a shared component. Citations of MapFormer could
not be listed (Semantic Scholar rate limit); a general search found no MapFormer follow-up on sign.
Naming collisions to avoid: "clock" algorithm in modular-arithmetic interpretability (Nanda et al. 2023;
Zhong et al. 2023, "The clock and the pizza"), and Gurnee & Tegmark 2023 "Language models represent space
and time" (probing world coordinates) -- known, not re-read; neither is about positional increments.

## 3. Notes on the sources that matter most

**Howard et al. 2014** (first-hand, PDF text). Leaky integrators dF/dtau = -alpha(tau) s F + alpha(tau) f.
Case I alpha = const: time cells. Case II alpha = |v|, "like the speedometer reading on a car, alpha(tau) is
always greater than or equal to zero"; F is the Laplace transform with respect to "the total distance
traveled along the path"; same place reached by different routes gives different codes (W-maze
retrospective coding, Fig. 10). Case III alpha = signed velocity along a direction: the return journey
recapitulates the outbound states, "always the same for a particular position x". Methods: negative alpha
makes the real exponential grow, so they use pairs of rectified (monotone) cells with opposite preferred
directions and read the ratio -- **a published construction of a map from two monotone accumulators.**
Covers F1 and F7 completely, F2 for a scalar channel. Does not cover learning, attention or sign ablation.

**TEM-t** (first-hand). Index encodings "uniquely identif[y] each time-step"; TEM's path-integrated g
"plays the role of position encodings"; Discussion: "positional encoding should reflect the abstract
underlying properties of the task ... In language ... this structure is grammar. This contrasts to the
typical positional encodings ... which represent a linear structure." So TEM-t frames index = linear/time,
PI = space, and *predicts* task-structured positions should help language. Our TinyStories/code/Bach
step tables (a mostly shared clock, no gain) are evidence against that conjecture at our scale -- a
point TEM-t invites but does not test.

**MapFormer** (first-hand, corpus). Table 2 discussion: navigation needs to "represent an action (moving
forward) and its inverse (moving backwards), preventing CoPE from solving the task, even in 1D." Sec. 4.1
lists "opposite actions cancel each other" as a requirement; Fig. 9b shows it learned. Dyck: "MapFormers
can use their positional encoding mechanism to emulate a stack (by moving forward or backwards)", trained
at 1 layer vs a 2-layer baseline. So the map half of F2 and the qualitative F6 are in the paper we
reproduce. What it does not do: isolate sign inside one architecture, control for length, or measure
the Dyck effect at matched depth (ours closed at 4 layers; `DYCK_MDEPTH_RESULTS.md`).

**Sarrof / Grazzi / Khavari / Selective RoPE / Mamba-3 / PaTH** (corpus, first-hand except Khavari).
The non-negativity observation, the impossibility proofs, the [-1,1] fix, the "same layer" requirement and
the transfer into content-dependent rotary position are all published. Sarrof's Dyck construction is
literally "the first layer tracks the depth by counting up for each opening bracket, and down for each
closing bracket" -- note that this signed counter lives in the additive input pathway (B = +/-1), allowed
even with nonnegative gates. Selective RoPE: "the input dependent rotations allow it to model flips".
Mamba-3 Table 5b: parity 100.00 (data-dependent RoPE) vs 1.56 (standard RoPE) vs 2.27 (none) -- but
trained up to length 160, scored at 256, so OOD. CoPE and PaTH report **in-distribution** wins over RoPE on
Flip-Flop / selective copy / counting; those are clock (selective count) or state tasks, not map tasks.

**Language/code numbers at training length.** CARoPE (monotone, rank 1/head): 512-token ppl RoPE 21.31 vs
CARoPE 21.23 (GPT-Small), 29.33 vs 28.99 (GPT-Tiny); at 1024 56.61 vs 21.39 and 81.27 vs 36.74 -- the
headline gain is extrapolation. HGRN Table 11: "the phase argument theta should not be data-dependent".
Selective RoPE Table 3 (FoX/GLA 370M): mixed. CoPE Table 5: code ppl -5% vs RoPE, -17% vs absolute (20M,
one run as far as stated); Wikitext gains too. CoPE's own motivating example (count sentences, attend to
position 0 of the current sentence) is what our TinyStories pilot found learned in a signed integrator
(extra ticks at `.`/`!`/`?`, `<eos>`, newline; `TINYSTORIES_PILOT_RESULTS.md`, CORRECTED block) -- a
spontaneous CoPE-like discourse clock, with no gain over RoPE (n=3, pilot).

## 4. Verdict on the candidate framing

### (a) Already published -- cite, do not claim
- **Cancellation decides map vs clock** (signed rate -> position, same place collides; nonnegative rate ->
  path length, route-dependent; constant -> time): Howard et al. 2014, explicitly, with words and figures.
  Also VCO grid models. Our table in `positional_review.tex` sec. clockmap is a restatement of Howard's
  Cases I-III for a rotary phase.
- **Monotone increments cannot track state; signed ones can**: Sarrof 2024, Grazzi 2025 (proofs), Khavari
  2025; in content-dependent rotary position: Selective RoPE, Mamba-3; for navigation specifically:
  MapFormer's CoPE argument and Fig. 9. Signed counters for Dyck depth: Weiss 2018, Bhattamishra 2020,
  Yao 2021, Sarrof Thm 5-6.
- **A signed phase can emulate a clock**: trivial containment ([-1,1] contains [0,1]; Grazzi). Not a
  contribution; one clause.
- **Index position is a linear/time structure; structured tasks want structured positions**: TEM-t.
- **Path integration / prefix sums substitute for attention depth (qualitatively)**: Liu 2023 (gridworld,
  2 layers), Yao 2021, Sarrof (2-layer Dyck), PaTH and Selective RoPE (1-layer Flip-Flop / parity),
  MapFormer (1-layer stack), Heo 2026 (depth theory).
- **Content-dependent phase gives little or nothing on language at the training length**: implied by
  HGRN Table 11, HGRN2, Selective RoPE Table 3, CARoPE's in-distribution column. Nobody says "because it
  learns a clock", but the end-metric fact is there.

### (b) Plausibly new as an empirical contribution (narrow; say "replication in a new regime" where so)
- **Sign isolated as one operation at matched length, on navigation** (|Delta| vs Delta, identical params,
  12 seeds; 0/8 vs 8/8, -0.177, p 0.0002; `SIGN_MATCHED_RESULTS.md`). The repo's own framing is right:
  a replication of Sarrof/Grazzi/SRoPE in a new regime with a cleaner isolation. Not a discovery.
- **The crossover**: the same sign constraint costs -0.280 on the torus and -0.004 on k-back recency
  (`RECENCY_RESULTS.md`). No paper runs one constraint on a map task and a recency task. Caveat (repo's
  own, `project_clock_vs_map.md`): recency does not *need* a clock (a signed rewind solves it), so this is
  "harmless there", not a double dissociation. Do not call it one.
- **Growth exponent of the unconstrained accumulator** (0.59 torus vs 0.97 recency; constrained arms ~1.0).
  Not found in the literature (searched once; weak evidence of absence). The repo's own caveats stand: it is
  collinear with opposition (r = +0.9995), not independently controllable, a re-description rather than a
  cause. Its one distinct use: it needs only trajectories, so it can be read on text. Present it as a
  diagnostic.
- **Learned step tables on code / Bach / TinyStories are mostly a shared clock** (75-92% shared component).
  Not found as a measurement anywhere. But: TinyStories is PILOT (n=3, not registered; CORRECTED block
  withdraws "nothing cancels" -- `mr`/`mrs` cancel `.`), the step latent was rank 4 shared over 8 heads
  ("a map had little room beside a clock"), and the clock is token frequency plus punctuation ticks, i.e.
  a CoPE-like selective clock.
- **Matched-depth exchange rate on Dyck** (+0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers; ceiling at 4L
  and 3x budget) and the H3 biased-walk analogue. The qualitative claim is published (above); a measured
  rate at matched depth with an index control trained to convergence is not. Budget-scoped.
- **Navigation told in words** (path 0.969 vs RoPE 0.505 at T=1024) as a map task embedded in language:
  no published counterpart found; TEM-t conjectures structured positions for language but tests none.

### (c) New as a statement, needs care
- **"Beats the token index exactly when the task needs a map."** "Exactly" is not supported and is
  contradicted in the literature: CoPE (monotone, selective) beats index position in distribution on
  counting, selective copy, Flip-Flop and code -- clock tasks. Content-dependent position can beat the
  index *without* a map, by counting the right tokens. And signed state helps code in Grazzi 2025
  (DeltaNet [-1,1] improves code and math perplexity at 340M/1.3B), so "code is clock-like" is itself
  contestable. Defensible form: "the *sign* matters exactly when
  the task needs cancellation; on our clock-like data a learned step does not beat the index at our scale".
  Also: within-task tests of the clock/map boundary failed for the PoPE corollary (T2/T2b; withdrawn), and
  code at matched length shows a small *cost* of path integration (+0.0056 bpc, n=3,
  `CODE_FULLVAL_RESULTS.md`). The "no gain" side is a set of nulls, mostly UNMEASURED below MDE.
- **"On clock-like data the model learns a clock that does no better than counting tokens."** Needs an a
  priori definition of clock-like data (e.g. no token pairs whose effects should cancel; no revisits),
  otherwise it is circular (data is clock-like because the model learned a clock). "No better than
  counting tokens" conflicts with CoPE's code/Wikitext gains for a selective clock; restrict to "our
  models, our scale, our recipes", and report the CoPE counterexample.
- **"A monotone one cannot be a map."** True per channel and for a rotary score that reads each channel's
  phase difference separately. Not true for a population: Howard 2014's Methods build a signed (map)
  code from two monotone rectified cells read out as a ratio, and two monotone channels counting "left"
  and "right" carry net displacement in their difference if any downstream readout can subtract them.
  State the scope: a monotone *rotary* channel cannot cancel inside the score; whether depth/MLP can
  recombine monotone channels into a map is untested here.
- **The cross-field identification** (Howard's alpha(t) = Mamba Delta_t = MapFormer's content-dependent
  angle step; sign axis = Case II vs III; the real/decay slot must be monotone for stability, the phase
  slot can be signed -- hence GRAPE's g(x) >= 0 and Howard's push-pull pairs). Not found stated. It is a
  short theoretical remark, plausibly new, and close to obvious once the sources are side by side. The
  LEC (time) / MEC (space) anatomical reading is a hypothesis (`02_literature.md`), not a result.

### Bottom line
The framing (clock vs map by cancellation) is Howard et al. 2014's, transposed to rotary attention; the
sign result is a replication (Sarrof, Grazzi, Selective RoPE, MapFormer's own CoPE argument); TEM-t
already says index position is a linear clock and task structure should set position. What a paper could
honestly claim is the measurement package: the isolated matched-length sign ablation on navigation, the
map/recency crossover, the matched-depth exchange rate, navigation-in-words, and the learned-step-table
diagnostics on real sequences -- each narrow, small-scale, and with the published qualitative version
cited beside it. The "exactly when" and "no better than counting" phrasings must go.

## 5. Sources

Local corpus (`papers/txt/`): mapformer, sarrof, grazzi, srope, mamba3, grape, carope, cope, path, hgrn,
hgrn2, puranik_janestreet, fox (grepped only). Downloaded and read as text this session: Howard et al.
2014 (https://people.bu.edu/marc777/docs/HowardEtal14-JNsci.pdf), TEM-t (https://arxiv.org/pdf/2112.04035).
Abstract / search summary only:
- Merrill, Petty, Sabharwal 2024: https://arxiv.org/abs/2404.08819
- Liu et al. 2023: https://arxiv.org/abs/2210.10749
- Bhattamishra et al. 2020: https://arxiv.org/abs/2009.11264
- Yao et al. 2021: https://arxiv.org/abs/2105.11115
- Weiss et al. 2018: https://arxiv.org/abs/1805.04908
- Khavari et al. 2025: https://arxiv.org/abs/2508.07395
- Heo et al. 2026: https://arxiv.org/abs/2603.05573
- Vanjani et al. 2026 (ReCAT): https://arxiv.org/abs/2609.35200
- Haviv et al. 2022: https://arxiv.org/abs/2203.16634
- Howard & Hasselmo 2020: https://arxiv.org/abs/2003.11668
- Nevermann & Gros 2026, distance generalisation (fixed context; no content-dependent position): https://arxiv.org/abs/2609.11913
- Mozer, Siddiqui, Liu 2026, "The topological trouble with transformers" (recurrence advocacy; no position): https://arxiv.org/abs/2604.17121
- Zhao et al. 2026, linear recurrent memory in POMDPs (no sign/position): https://arxiv.org/abs/2605.31261
- Kraus et al. 2013/2015, Abramson et al. 2023: https://elifesciences.org/articles/83930
- Tsao et al. 2018: https://www.nature.com/articles/s41586-018-0459-6
- GridPE: https://arxiv.org/abs/2406.07049
- Gao et al. 2021, path integration of grid cells: https://arxiv.org/abs/2006.10259
Secondary only: Buzsaki & Tingley 2018 (PMC6166479; full text blocked by a browser check), Burgess 2008.
Citation list scanned: Semantic Scholar citations of 2411.12537 (111 titles, 2026-10-12).
