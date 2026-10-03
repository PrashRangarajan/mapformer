# Context-dependent steps: the literature (2026-10-03)

Scope: where published data-dependent position / decay / rotation generators get their data from (token,
short window, hidden state, query-key pair), what that buys in context range, how they are made to train,
and whether anyone has tested words that sometimes count. Situates our pilots: `CONTEXT_STEP_DESIGN.md`,
`CTXSTEP_PILOT1..3.md`, `CTXSTEP_HS_RECIPE.md`, `CTXSTEP_HSR_PILOT.md`, `TEXTWORLD_RESULTS.md`;
models `model_context_step.py`, `model_selective.py`. Our side is PILOT level throughout (n = 1-2 per cell).

## Summary

1. **Where the data comes from.** Every published generator reads the *input of its own layer*. At layer
   1 that is the token (Mamba-2, Mamba-3, Gated DeltaNet, FoX, CARoPE, MapFormer), the token through a
   short causal conv (Mamba-1: width 4; PaTH: width 3; Selective RoPE: width unstated) or a token shift
   (RWKV-7: width 2). At layer >= 2 it is a hidden state, so range comes from **depth**, not from the
   generator. Pair-based schemes (CoPE, stick-breaking, Selective Attention, DAPE) gate a query-key pair
   with per-layer hidden states and have unlimited range. Our three arms map exactly onto this:
   CF = Mamba-2/3 at layer 1, CG and SR = Mamba-1/PaTH-style window at layer 1, HS = the per-layer
   placement at layer 2.
2. **Range.** Nobody we read measures a window-limited generator against cue distance. The window limit
   is true by construction for a single layer; the only explicit statement is a footnote in Allen-Zhu's
   Canon-layers paper (a task "designed so a 4-token window cannot resolve" 10-14-token dependencies).
   Mamba-3 drops the short conv with no loss on language, consistent with range coming from depth.
3. **Training tricks.** "Start as the static model" is standard and well documented: Mamba's dt bias
   (inverse-softplus of log-uniform [0.001, 0.1]), RWKV-7's decay LoRA with its first factor zero-initialised
   (decay = static per-channel schedule at init), TAPE and CARoPE initialised to RoPE, DaRoPE reducing to
   RoPE at w=0, alpha=1, ReZero (alpha = 0 residual), Flamingo (tanh(alpha), alpha = 0, and instability
   without it), Canon (residual "ensures stability"), chrono init for LSTM gates. Selective RoPE reports
   training instability of its data-dependent angle at higher LR, fixed by l2-normalised input, weight
   norm or a rank-8/16 bottleneck. Our HSR fix (Delta = W(emb + alpha LN(h1)), alpha = 0) is this trick.
4. **Words that sometimes count.** The closest prior tasks are CoPE's counting task (variables
   incremented among "pass" distractors) and selective copy, Flip-Flop ("i" = ignore the next bit),
   Mamba's selective copying, contextual counting (count 1s only inside brackets), and bAbI task 9
   (simple negation). All are scalar / monotone (count or ignore), none has a signed vector step that a
   cue must cancel, and none varies the cue's distance from the word it modifies.
5. **Verdict in one line.** The mechanisms (gated step, window generator, hidden-state generator,
   identity-at-init) are all published; what is not, as far as this reading goes, is a controlled
   cue-distance x generator-source dissociation on a signed, path-integrated phase, read out
   mechanistically (swap test). That is an empirical, task-level contribution, and at pilot level.

## Table

Range = how far a cue may sit from the word whose step it must change, for ONE layer of the generator.
"depth" = unlimited once the generator reads a hidden state of an earlier attention/recurrent layer.

| paper | arXiv | data-dependence source (layer 1 / deeper) | range at layer 1 | training tricks | relevance |
|---|---|---|---|---|---|
| MapFormer | 2511.19279 | token embedding, once, before all blocks | 0 (token only) | low rank r=2 W_out W_in | our baseline CF |
| Selective RoPE | 2511.17388 | per layer, per head, from q (or l2-normed x); depthwise conv; sigmoid phase gate | conv width (unstated) / depth | weight norm, rank-8/16 MLP, l2-normed input against LR instability; SiLU after conv | our SR arm = its generator at layer 0 only |
| Mamba (S6) | 2312.00752 | Delta = softplus(b + low-rank Linear(conv4(x))), per layer | 3 tokens / depth | dt bias = inv-softplus(logU[1e-3,0.1]); low-rank dt_proj (rank D/16) | Delta -> 0 "ignores the input" = decoy suppression; conv-fed gate = our CG |
| Mamba-2 (SSD) | 2405.21060 | dt from in_proj(u) in parallel; conv only on x,B,C | 0 / depth | same dt-bias init; extra norm for stability | gate from token at layer 1 = CF-like |
| Mamba-3 | 2603.15569 | angle step = Delta_t * theta_t, both linear in layer input; theta shared across heads (code) | 0 / depth | dt-bias init; BC-norm; conv removed (optional) | **structurally our CG without the window: gate x signed angle**; solves parity with RoPE-trick |
| GRAPE | 2512.07805 | phase-modulated variant Phi_t = sum g(x_l), g >= 0 | token | -- | unifies; contextual variant non-negative; LM experiments only |
| CARoPE | 2507.23083 | per-head scalar from token embedding, 1/(softplus(xW)+1) in (0,1) | 0 | initialised as RoPE | bounded, positive, rank 1 per head |
| CoPE | 2405.18719 | gate sigma(q_i . k_j) per pair, per layer; position = sum of gates | unlimited (pair) | gates = 1 recovers relative PE; sep-keys option | **counts only some tokens**; selective copy, counting, Flip-Flop; monotone, scalar |
| PaTH | 2505.16381 | H_t = I - beta_t w_t w_t^T; w_t low-rank + conv(3) + l2; beta = 2 sigma(.) | 2 tokens / depth | RoPE->PaTH conversion by distillation | data-dependent multiplicative position; Flip-Flop solved with 1 layer, 2 heads |
| FoX | 2503.02130 | f_t = sigma(w^T x_t + b), scalar per head, per layer; Pro adds KV-shift | 0 (Pro: 1) / depth | data-independent gates need careful init | data-dependent decay beats fixed (ALiBi-like) gates |
| Stick-breaking attn | 2410.17980 | beta_ij = sigma(q_j . k_i); A_ij = beta_ij prod (1 - beta_kj) | unlimited (pair) | no PE at all | position emerges from content; recency |
| Selective Attention | 2410.02703 | tokens mask earlier tokens for all later queries; F = cumsum(relu(S)) | unlimited (pair) | parameter-free (reuses a head) | **trailing retraction primitive** for attention |
| DeltaNet | 2406.06484 | beta from x; short conv(4) on q,k,v | conv / depth | L2 norm on q,k | short conv standard in linear models |
| Gated DeltaNet | 2412.06464 | alpha, beta linear in layer input; conv(4) on q,k,v only | 0 / depth | Mamba dt-bias init on alpha (code) | w/o short conv: ppl 27.35 -> 28.95 |
| RWKV-7 | 2503.14456 | decay w = w0 + tanh(x_w W1) W2 on token-shifted x (lerp x_t, x_{t-1}) | 1 token / depth | **W1 = 0 at init: decay starts static** (code); "init crucial" | zero-init data dependence on a static schedule |
| Grazzi et al. | (corpus `grazzi`) | eigenvalues in [-1, 1] | -- | -- | sign of the increment (our sign axis) |
| HGRN | 2311.04823 | data-dependent forget gate with learned lower bound; phase data-independent | 0 / depth | lower bound (static floor + data part) | data-dependent phase WORSE on LM (Table 11) |
| DAPE / DAPE v2 | 2405.14722 / 2410.04798 | MLP over (QK^T, static bias) per entry; v2 adds conv over the score map | pair | -- | additive, no accumulation |
| TAPE | 2501.00712 | positional tensor updated per layer by attention/MLP | depth | RoPE init; W2 = 0 (LoRA-style) | contextual positions across layers |
| DaRoPE | 2609.34556 | slow-band coordinate L sigma(w^T x_m + alpha beta_m) from hidden state, per layer | per layer, not accumulated | reduces to RoPE at w=0, alpha=1; w std 0.1, alpha 0.5 | "positional prior + content correction" = our HSR shape, without a cumsum |
| Canon layers | 2512.17351 | h' = h + conv4(h), anywhere in the block | 3 tokens | residual "ensures stability" | **explicit window-limit statement** (Depo2 footnote) |
| ReZero / Flamingo | 2003.04887 / 2204.14198 | -- | -- | alpha = 0 residual; tanh(alpha) with alpha = 0, instability without | our alpha = 0 trick |
| chrono init | 1804.11188 | gates as learned time warps | -- | gate biases from the time scale | "a gate is a learned clock rate" (cf. Puranik) |
| Flip-Flop | 2306.00946 | task | -- | -- | ignore instruction = 1-token leading cue |
| Contextual counting | 2406.02585 | task | -- | -- | count only inside a delimited region; NoPE best |
| bAbI / StepGame | 1502.05698 / 2204.08292 | task | -- | -- | task 9 negation ("no longer in"), task 19 path finding; StepGame adds distractor sentences |

## Per-paper notes

Status tags: **[first-hand]** = method (and the cited result) read in the text or source code;
**[partial]** = specific sections read; **[abstract]** = abstract / intro only.

### Selective RoPE (Movahedi et al., 2511.17388, ICLR 2026) [first-hand, local `srope`]
- Computes `omega = conv1d(W_omega q)`; `theta = temp * cumsum(omega)`; plus a sigmoid gate on the phase
  and an optional bias (Fig. 4). Derived from the softmax kernel's shift `q_t - q_{t-1}` (a width-2
  difference) but implemented as a "short convolution"; width not stated (our 4 is the Mamba convention,
  noted in `model_selective.py`).
- Placement: inside every layer, per head, from the queries -> at depth the angle reads hidden states.
- Tricks: instability at higher LR, fixed by using l2-normalised x instead of q, weight norm on the
  projection, or a rank-8/16 MLP; "the most significant improvements come from adding a SiLU activation
  after the convolution" and the phase gate (Table 2, 370M GLA).
- Experiments: MQAR, MAD, copying (length-extrapolates), S2/A3 state tracking, 370M-1.3B LM.
- Relative to us: our SR arm is its generator at MapFormer's placement (layer 0, from x, full rank,
  **no SiLU after the conv**). Pilot 2's finding that the gate x lagged-conv interaction uses cues on both
  sides within the window is not discussed by them. They never test decoys or cue distance.

### Mamba (Gu & Dao, 2312.00752) [first-hand, local `mamba` + `mamba_simple.py`]
- `Delta = softplus(Parameter + Linear_D(Linear_R(x)))`; in the code `x = SiLU(conv1d_4(x))` is computed
  first and `x_proj` reads it, so **Delta is a function of a 4-token causal window at layer 1**. dt bias
  initialised to inverse-softplus of log-uniform [0.001, 0.1]; dt_proj low rank (`dt_rank = ceil(D/16)`).
- Sec. 3.5: "a large Delta resets the state and focuses on the current input, a small Delta persists the
  state and ignores the current input"; motivating tasks are selective copying and induction heads.
  Table 7: selective Delta is the most important selective parameter (ppl 10.93 -> 9.81 alone).
- Relative to us: Delta -> 0 on a decoy is exactly our gate g_t -> 0. A conv-fed gate is CG's design.
  Anticipates the principle "content must be able to switch the step off"; does not test cue distance.

### Mamba-2 (Dao & Gu, 2405.21060) [partial, local `mamba2` + `mamba2.py`]
- Parallel projections: `zxBCdt = in_proj(u)`; conv applies to `xBC` only; `dt = softplus(dt + dt_bias)`
  reads the block input directly -> **token-only at layer 1**. Extra norm added for stability.

### Mamba-3 (2603.15569) [first-hand on the complex SSM / RoPE sections, local `mamba3` + `mamba3.py`]
- Complex SSM `Diag(A_t + i theta_t)`; discretised rotation angle per step is `Delta_t * theta_t`
  (Prop. 2-4, App. eq. on `R(Delta_t theta_t)`); Prop. 3 = data-dependent RoPE on B, C (= K, Q).
- Code: `angles` is a slice of `in_proj(u)` (linear, signed), **broadcast to all heads**; `DT =
  softplus(dd_dt + dt_bias)` per head; the kernel takes both. No short conv by default ("Mamba-3 + conv"
  15.85 vs 15.72 ppl, Table 5a); BC bias + trapezoid make it "optional".
- Results: Parity 100 and modular arithmetic near-solved; without RoPE or with standard RoPE ~chance.
- Relative to us: **the angle increment is a non-negative data-dependent gate times a signed
  data-dependent angle**, i.e. our CG form (g_t x step) with the gate and the step both from the layer
  input. At layer 1 it is token-only (no window), so a 1-layer Mamba-3 is CF-like for our decoys; at
  depth it is HS-like. The head-shared angle is relevant to our open "sharing" cell in the rank line.

### CoPE (Golovneva et al., 2405.18719) [first-hand, local `cope`]
- `g_ij = sigma(q_i . k_j)`, `p_ij = sum_{k=j..i} g_ik`, interpolated position embeddings, per layer, per
  head; sep-keys variant. Gates all 1 recovers relative PE.
- Tasks (error %): Flip-Flop (in-dist 0.0 vs RoPE 1.8; OOD 4.9 vs RoPE 20.3 / abs 21.7), selective copy (0.0 vs 16.9-40.1 in-dist),
  counting with 1/3/5 variables among "pass" (1.2 vs 17.8 relative at 3 vars), plus LM / code.
- Relative to us: the closest prior statement of "positions should count only the tokens that matter".
  Its gate is pair-based and per layer, so it reaches any cue the key's hidden state encodes. But the
  count is scalar and non-negative: it cannot express a signed, direction-specific step or a cancellation.
  Anticipates the *goal*, not the signed path-integration setting.

### GRAPE (Zhang et al., 2512.07805) [partial, local `grape`]
- Contextual "phase-modulated" variants: `Phi_t = sum_{l<t} omega_l`, `omega_l = g(x_l) >= 0`; App. D
  calls its multiplicative extension non-contextual. Experiments: FineWeb-Edu LM only.
- Relative to us: taxonomy owner; the contextual phase is non-negative (a clock), token-sourced.

### CARoPE (2507.23083) [first-hand, local `carope`]
- Per-head scalar frequency from the token embedding, `1/(softplus(xW)+1) in (0,1)`, accumulated;
  initialised as RoPE. Token-only, positive, rank 1 per head: cannot stay put on a decoy except by
  driving the step toward 0 from the token alone.

### PaTH (Yang et al., 2505.16381) [first-hand, local `path`]
- `H_t = I - beta_t w_t w_t^T`, `beta_t = 2 sigma(u^T x_t + b)` (negative eigenvalues allowed), `w_t` from a
  low-rank linear, a **short conv of width 3**, then l2 norm; cumulative product, per layer.
- Flip-Flop error (Table 1), one layer and two heads: in-distribution PaTH 0% vs RoPE 6.9 / SBA 9.6 /
  FoX 8.3%; OOD sparse (98% ignore) 0.0001% vs 40.3 / 38.9 / 36.3%.
  A5 word problems, LM at 760M, RoPE -> PaTH conversion by distillation ("start the conversion before the
  model ossifies").
- Relative to us: a non-commutative cousin; its window at layer 1 is 3 tokens. Flip-Flop's ignore cue
  is adjacent (distance 1), so its 1-layer success is inside any window.

### FoX (Lin et al., 2503.02130) [partial, local `fox`]
- `f_t = sigma(w_f^T x_t + b_f)` per head, from the layer input; Pro adds KV-shift (a data-dependent
  2-token lerp), QK-norm, output gate/norm. Data-dependent beats data-independent and fixed gates
  (Fig. 7); for data-independent gates "we find it crucial to initialise b properly" (ALiBi-matched).
- Relative to us: decay, not phase; token-sourced at layer 1.

### Stick-breaking attention (Tan et al., 2410.17980, ICLR 2025) [first-hand, method + MQRAR]
- `A_ij = beta_ij prod_{i<k<j}(1 - beta_kj)`, `beta = sigma(q . k)`, no position embedding. Most-recent
  match wins; MQRAR (repeated assignment) shows the recency inductive bias.
- Relative to us: position as a content-gated product over intervening tokens, pair-based, unlimited
  range; no signed step.

### Selective Attention (Leviathan et al., 2410.02703, ICLR 2025) [first-hand, method]
- `softmax(QK^T/sqrt(d) - F)`, `F_ij = sum_{k<i} relu(S_kj)` with S = a reused head's logits: a token can
  mask an EARLIER token for all LATER queries. Variable assignment, parity, copy; LM gains ~ 2x heads.
- Relative to us: the attention-side primitive for a **trailing cue that retracts an earlier word**.
  Path integration cannot mask: a retraction must be a later cancelling step (what HS does on trailing
  cues, attention from cue to word). Worth one sentence when we write the trailing case up.

### DeltaNet / Gated DeltaNet (2406.06484 / 2412.06464) [first-hand on conv and block design; GDN code]
- Short conv (width 4) + SiLU on q, k, v; GDN: "alpha, beta use linear projection only" (so token-only at
  layer 1); code initialises alpha's dt bias exactly as Mamba. GDN ablation: without short conv avg ppl
  27.35 -> 28.95.

### RWKV-7 (Peng et al., 2503.14456) [first-hand eq. 12 + `RWKV-v7/train_temp/src/model.py`]
- `w = w0 + tanh(x_w W1) W2`, `x_w = lerp(x_t, x_{t-1}, mu)` (token shift, window 2), soft-clamped and
  exponentiated; **W1 is zero-initialised** so the decay begins as the static per-channel schedule w0.
  The paper: "proper parameter initialisation is crucial ... deviations may lead to degradation".
- Relative to us: the cleanest published instance of "data dependence starts at zero on top of a static
  schedule". HSR is the same move (word step static at the token, context correction at zero).

### HGRN (2311.04823) [partial, local `hgrn`]
- Data-dependent forget gate with a learned, layer-increasing lower bound; Table 11: a data-dependent
  phase is worse on language ("theta should not be data-dependent"). Published negative for
  content-dependent phase on LM; not on navigation.

### DAPE / DAPE v2 (2405.14722 / 2410.04798) [DAPE partial (local); v2 abstract]
- Additive bias = MLP over the attention logits and a static bias per (i, j); v2 convolves the score map
  across neighbouring entries and heads (kernels like 1x3). Pair-sourced, additive, no accumulation.

### TAPE (2501.00712) [partial, local `tape`]
- Positional tensor updated by attention and MLP in every layer under equivariance; RoPE as init;
  `W2 = 0` (LoRA-style) so "the augmented model remains identical to the original at initialisation".

### DaRoPE (Levy, Videau et al., 2609.34556, Sep 2026, ICLR 2027) [first-hand, method]
- Fast RoPE bands kept; on slow bands the index is replaced by `L sigma(w_h^T x_m + alpha_h beta_m)` with
  `beta_m = sigma^{-1}((m+0.5)/L)`, x_m the layer's hidden state. Falls back to RoPE at w=0, alpha=1
  (initialised w std 0.1, alpha 0.5). 15 synthetic tasks, music, genomics, EEG, LM 124M-50B.
- Relative to us: newest data-aware rotary; a per-token coordinate, NOT an accumulated step, so a decoy
  is not a problem it faces. Its "prior + content correction" shape mirrors HSR.

### Canon layers (Allen-Zhu, 2512.17351) [partial]
- `h'_t = h_t + conv1d_4([h_t .. h_{t-3}])`; residual design "ensures stability and never hurts";
  Mamba-2's built-in conv1d "drives most of its gains". Footnote 13: Depo2 "is designed so a 4-token
  window cannot resolve key-value pairs spanning 10-14 tokens".
- Relative to us: the one explicit statement of the window limit we found, used as a task design rule,
  not measured as a curve.

### ReZero (2003.04887), Flamingo (2204.14198), chrono init (1804.11188) [first-hand on the cited parts]
- ReZero: `x_{i+1} = x_i + alpha_i F(x_i)`, alpha = 0 at init. Flamingo: new layers enter as
  `tanh(alpha) * f(.)`, alpha = 0, so the model starts as the frozen LM; removing it costs 4.2% and
  "leads to training instabilities". Chrono: gates are learned time warps; initialise gate biases from
  the expected time scale; large gains on copy / adding at long T.
- Relative to us: HSR's alpha = 0 is ReZero/Flamingo applied to the step generator.

### Tasks [Flip-Flop first-hand abstract + CoPE/PaTH descriptions; contextual counting abstract + task; bAbI task list; StepGame abstract]
- Flip-Flop (2306.00946): w/i/r instructions, "i" = ignore the following bit; cue adjacent.
- Contextual counting (2406.02585): count 1s only inside the bracketed region (cue = delimiter at any
  distance); causal attention, NoPE best, RoPE competitive.
- bAbI (1502.05698): task 9 "Fred is no longer in the office", task 19 path finding with direction words;
  StepGame (2204.08292): multi-hop spatial relations with distractor sentences.
- CLAUDE.md rule 16: borrowed benchmarks usually do not test our axis (Flip-Flop was such a null). None of
  these has a signed step that a distant cue must cancel.

### Peripheral, abstract only
CAT (2407.05591, conv-augmented attention, 1-layer AR and copying), Zoology (2312.04927, input-dependent
mixing for MQAR), GAPE (2605.10414, query/key gates on RoPE logits), Dynamic PE (2204.08142, context-refined
positions for NMT). None bears on step generators' range.

## Answers to the four questions

**(a) Token, window or hidden state, and what it buys.** At layer 1: token (Mamba-2, Mamba-3, GDN, FoX,
CARoPE, MapFormer), window (Mamba-1 4, PaTH 3, RWKV-7 token shift 2, Selective RoPE unstated), pair
(CoPE, stick-breaking, Selective Attention, DAPE). At depth all become hidden-state generators. The
field's working assumption is that the short window supplies local mixing (Canon, Zoology, GDN ablation)
and depth supplies range; Mamba-3 removing the conv without loss is consistent with that. Nobody isolates
the generator's source as the variable and measures range, which is what our pilots do.

**(b) Tricks that make data-dependent steps train.** (i) Start at the static model: dt bias init
(Mamba, GDN, Mamba-3), zero-init data-dependent factor (RWKV-7 W1, TAPE W2), init-as-RoPE (CARoPE, TAPE,
DaRoPE), zero residual scale (ReZero, Flamingo, Canon's residual). (ii) Bound and normalise the input to
the generator: l2-normalised x, weight norm, low rank (Selective RoPE), BC/QK norm (Mamba-3), RMSNorm (FoX
Pro). (iii) Nonlinearity after the conv (Selective RoPE's largest gain; Mamba's SiLU). (iv) Static floor
plus data part (HGRN lower bound, DaRoPE prior). Our HS failure (7/14 learn no step) and HSR fix (4/4)
are an instance of (i); the observation that a step computed ONLY from a fresh layer's output often never
starts is in line with Flamingo's "instability without zero-init gating" and Selective RoPE's instability.

**(c) Words that sometimes count.** CoPE (counting among pass, selective copy), Flip-Flop, Mamba's
selective copying, contextual counting, bAbI 9. All scalar/monotone; cues adjacent or delimiter-based; no
cue-distance manipulation; no signed cancellation. Our decoy task (direction word used without moving,
cue 1-3 vs 6-13 tokens away, both sides) has no published equivalent that we found.

**(d) Evidence of the window limit.** None measured. Canon's Depo2 footnote states it as a design rule.
Mamba's selective-copy argument (LTI convolutions cannot handle variable spacing) is about static
kernels, not windowed gates. Our pilot 3 (n = 1 per cell) is the only measurement we know of.

## Prior-art assessment (blunt)

Already known, do not claim:
- That a token-only step moves on every use of a word and content must be able to switch it off: this is
  Mamba's selection argument (Delta -> 0 ignores an input), CoPE's "count what's important" and the
  Flip-Flop ignore instruction.
- The gated step form `g_t x step_t`: Mamba-3's angle `Delta_t theta_t` is exactly a non-negative
  data-dependent gate on a signed data-dependent angle. CG = Mamba-1's conv-fed Delta applied to a
  MapFormer step. Present CG as an instance, not a design.
- The hidden-state step: it is the standard per-layer placement (Selective RoPE, Mamba-3, PaTH, FoX,
  TAPE). HS is "the generator at layer 2".
- HSR's alpha = 0 correction: ReZero / Flamingo / RWKV-7 / TAPE / CARoPE / DaRoPE. The trick is not ours;
  that it fixes this failure is a small engineering note.
- "A k-window generator cannot use a cue more than k-1 tokens away" is true by construction for one
  layer; a reviewer will say so. On its own it is not a finding.
- "Linear context cannot cancel a direction-specific step": an elementary argument; and our own first
  version of it (Selective RoPE is additive) was wrong (pilot 2).

Possibly new (as far as this reading goes; pilot-level evidence):
- A controlled dissociation on ONE task: generator source (token / window / hidden state) x cue distance
  (near / far) x cue side (lead / trail), with a mechanistic readout (north <-> south swap: decoy/move
  ratio). None of the papers above varies cue distance or reads the step's response to a decoy.
- Within the window, both sides work through nonlinear gate x lag interactions (CG reads trailing cues
  via the gates of following tokens; SR via its per-channel gate on lagged terms). Not discussed by
  Selective RoPE or Mamba; derivable, but the lead/trail double dissociation one would naively predict
  does not hold, and that is worth a sentence.
- Signed, vector-valued cancellation in a text-navigation setting (trailing cue -> a later cancelling
  step via attention), as opposed to the scalar/monotone counting in CoPE and the masking in Selective
  Attention.
- The reliability observation (hidden-state-only step fails to start in about half the runs at 2 layers;
  keeping the word's own step fixes it) is a modest but specific result about data-dependent phase;
  related instabilities are reported (Selective RoPE, Flamingo) but not this one.

Risks: all of ours is n = 1-2 per cell; SR is not a one-knob arm (full rank, no omega, temperature; and
it lacks Selective RoPE's SiLU after the conv); the HSR pilot did not cross the SOLVED line; HGRN's
negative for data-dependent phase on language stands untouched (we are on a scripted grammar).

## Proposals

1. **The literature-standard multiplicative gate at depth, against HSR, on far cues.** Arms: (a) HSR;
   (b) Mamba-3 form: `Delta_t = softplus(b + u . LN(h1_t)) * W_out W_in emb(x_t)`, b initialised so the gate
   is ~1 (starts as CF, Mamba dt-bias style); (c) Mamba-1 form: (b) with the gate reading a causal conv-4
   over h1. Prediction: (b)/(c) suppress far LEADING decoys (the gate closes at the direction word); far
   TRAILING decoys need a later cancelling step, which a non-negative gate on the cue's own
   direction-independent step should not be able to produce -- but CG solved near trailing cues anyway
   (pilot 2), so this is open and is the informative cell. This is the comparison a reviewer will ask for,
   since Mamba-3's angle is gate x angle. Cost: 3 arms x 2 sides x 4 fresh seeds = 24 runs at 1800 epochs,
   T=2048; the 8-run HSR pilot took ~4.3 h wall on both GPUs, so ~13 h. New classes in
   `model_context_step.py`-style new file; no edits to running code.
2. **Decompose the HSR fix.** alpha init 0 (current) vs alpha init 1 (word path kept, no zero-init) vs a
   static-floor form `Delta = W(emb) + W'(alpha LN(h1))` with separate low-rank maps for the correction
   (RWKV-7 style, W' first factor zero-init). Tells whether the word skip path or the zero init does the
   work (Flamingo/ReZero say the zero init; Canon says the residual). Cost: 2 new arms x 2 sides x 4
   seeds = 16 runs, ~9 h. Fold HS and HSR controls of the same seeds into the batch (rule 12).
3. **Turn the window claim into a curve, cheaply.** New task subclass with the pad length drawn per decoy
   from 0..12 (cue distance 1..13), trained once per arm; bin the swap-test decoy/move ratio and accuracy by
   distance at eval (no retrain per distance). Arms: CG at k = 2, 4, 8; SR rank-4 one-knob (with SiLU after
   the conv, per Selective RoPE Table 2); HSR. Prediction: each window arm's ratio steps from ~0 to ~1
   between d = k-1 and d = k; HSR flat near 0. This replaces a "by construction" claim with a measured
   transition and checks the k-1 boundary that the nonlinear-interaction account predicts. Cost: 5 arms x
   2 sides x 3 seeds = 30 runs; window arms at 900 epochs (they converged near-cue within 900), ~10-12 h.
4. **Fidelity of the Selective-RoPE arm before any registration (cheap).** Add SiLU after the conv,
   l2-normalised input, and a rank-4 bottleneck as single knobs (the audit already flagged rank/omega).
   Near and far, 2 sides, 2 seeds, 900 epochs: ~16 runs, ~4-5 h; can share the batch with (3).

Not proposed: Flip-Flop / CoPE counting / bAbI on GPU (rule 16; they do not test a signed cancellation
at controlled distance). bAbI 9 / StepGame-style negation templates are a candidate for a later natural
text sanity check only.

## Sources read (local corpus `papers/txt/` unless a URL is given)
srope, mamba, mamba2, mamba3, cope, grape, carope, path, fox, deltanet, rwkv7, hgrn, dape, tape, grazzi
(summary only, from `papers/INDEX.md`). Code: github.com/state-spaces/mamba (`mamba_simple.py`,
`mamba2.py`, `mamba3.py`), github.com/NVlabs/GatedDeltaNet (`gated_delta_net.py`),
github.com/BlinkDL/RWKV-LM (`RWKV-v7/train_temp/src/model.py`). arXiv PDFs: 2410.17980, 2412.06464,
2410.02703, 2306.00946, 2406.02585, 2003.04887, 2204.14198, 1804.11188, 2410.04798, 2312.04927, 2609.34556,
2512.17351, 2407.05591, 2605.10414, 2204.08142, 1502.05698, 2204.08292.
