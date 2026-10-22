# Language side and the integrator framing: expansion for Section 7 and the Discussion (2026-10-20)

*CORRECTED 2026-10-21: NEEDS-MAP is "no retrace AND an object target", not all places reached another way, and the strata are re-run on all 200 walks (`scripts/language/tw_strata_200walks_out.txt`; `docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md` B1, M-LG1). The body below is unchanged.*

Scope: what a one-layer path model learns from words, when a learned phase becomes a map and when a clock, the
integrator (Howard) framing, and whether what/where and the leak belong in the main paper. For `docs/paper/MAIN_DRAFT1.md`
Sections 2.1, 7 and 8. No training was run; every new number is a post hoc, eval-only CPU analysis of stored checkpoints
(scripts and outputs in `docs/paper/expansion/scripts/language/`, section 8). Results files are authoritative; where this
note and a results file differ, the results file wins.

Labels. **ESTABLISHED** = a registered verdict or a declared secondary in a results file (with its file). **ANALYSIS** =
post hoc, on stored checkpoints, not pre-registered (new here unless a file is named). **THEORY** = an argument, with the
check that tests it. **SPECULATION** = untested. Terms: *step* = the per-token increment Delta_w = W_out W_in e_w of a
word w (one per word type; the model's step is context-free); *move* = one step of the walk (one movement clause in the
text world); *tick* tau_c = the phase a channel gains per move whatever its direction; *channel* = one rotary 2D block of
one head (64 per model); *used channel* = weighted by how much retrieval attention uses it (mean ||q_c|| x mean ||k_c||
over the query words and the object words); *re-scored* = read with the x1.111 attention-dropout scale (the registered
correction, `docs/audits/2026-10-04/DROPOUT_RESCORE.md`).

---

## 0. Summary: the key additions

1. **A complete step account of the text-world models (ANALYSIS, 32 models).** One path layer learns: direction words
   as signed steps whose differentials cancel (opposition 0.001-0.013 on all 24 learned-step models, wrap-aware);
   optional words (adverbs, fillers) with steps of 0.004-0.03 of a move on every model; aside sentences whose words have
   sizeable steps that cancel within the sentence, leaving a net offset of 0.04-0.16 of a move (usage-weighted); and, on
   10 of 24 models, a per-move tick in 26-38 channels.
2. **The aside offset is causal, both ways (ANALYSIS).** Putting each aside's words at its clause's object-slot phase,
   with the phase after the aside unchanged, drops the 19 learned-step models at ceiling in eval mode from 0.989-1.000 to
   0.952-0.975, and 71-100% of their errors then name an object told in an aside at that cell: the direction-words-only
   oracle's ceiling and error signature (`TW_NORMSTEP_RESULTS.md`). Conversely, offsetting the oracle's aside words by half
   a move (phase after the aside unchanged) lifts it from 0.962-0.981 (100% aside errors, 8/8 seeds) to 0.999-1.000 on
   8/8 seeds.
3. **The text-world clock counts moves, so it is path length, Howard's Case II (ANALYSIS).** An aside sentence adds
   at most 0.06 of a tick, a filler or adverb word at most 0.007: the clock is neither a word nor a sentence counter. On
   the clock seeds one phase runs Case III (position) in some channels and Case II (path length) in others.
4. **The clock is idle (ANALYSIS).** The per-move tick alone predicts which channels drift between visits (agreement
   0.98-1.00 per model, 32 models). 79-99% of retrieval usage sits in channels whose tick is 0 mod pi (pi: on the even
   64-torus every return takes an even number of moves). Removing the 26-38 clock channels from the score costs <= 0.009
   on all 7 clock models at ceiling in eval mode, while the 3 below ceiling still need them (cost 0.09-0.26); removing as
   many map channels collapses accuracy (0.02-0.65).
5. **A usage-weighted drift/content ratio separates map from clock where the growth exponent cannot (THEORY + ANALYSIS).**
   The pooled growth exponent alpha reads 0.98-1.02 on the text-world clock seeds (which carry a perfect map) and on
   TinyStories alike. The usage-weighted median of rho_c = |mu_c| L / range(content_c, L) (clock part over content part,
   per channel, at the training length) reads: torus 0.064-0.079, text world 0.06-0.23 (converged seeds), code 4.5-6.6,
   TinyStories 9.9-12.3, recency 21.8-22.1. Same architecture, same signed step: the data decide whether the used
   channels are a map or a clock.
6. **The text world is a short-walk task, and the paper must say so (ANALYSIS).** Its walks are ~132 moves, and the
   paper's own retrace floor scores **0.934** on its registered eval stream (0.937 on TW_NORMSTEP's), not the 0.597 of
   the one-step reversal-copy rule the results file reports. Path still clears it, and on the revisits retrace gets wrong
   (5.9%; best constant 0.112) path models score 0.944-1.000 on
   the six seeds at ceiling (0.961 on the fresh seeds 2-7) against 0.10-0.17 for two index layers and 0.00-0.01 for one
   (section 2.6).
7. **Integrator paragraph** (section 4): a positive statement of what is ours (the three-way identification for rotary
   attention; that in the rotary slot the Case II / III distinction is one of learnability, not expressivity; and the
   measurement that a learned rate realises Case I, II or III per channel as the data demand), with precise prior work.
8. **What/where and the leak** (section 5): one paragraph in Section 7, as a single principle -- a token's phase offset
   from its place must depend on its role, not its identity -- with the leak registered result (and its failed accuracy
   replication) and the what/where checks in the appendix.

---

## 1. Established facts (from the results files)

| fact | numbers | label | file |
|---|---|---|---|
| One path layer learns navigation told in English | fresh seeds (n = 6): 0.979 +/- 0.048, 6/6 SOLVED vs RoPE 1 layer 0.505, 0/6; +0.474, permutation and Fisher p 0.0022; RoPE 2 layers 0.793 (0/6). All 8 seeds: 0.969 (eval mode), **0.993 re-scored** vs 0.506 / 0.772 | ESTABLISHED (A; the re-score is registered in DROPOUT_RESCORE) | `TEXTWORLD_RESULTS.md` |
| Floors on that eval stream | best constant 0.505, one-step reversal-copy 0.597 | ESTABLISHED | `TEXTWORLD_SECONDARY.txt` |
| Opposite directions cancel on every seed | after removing the direction words' common component: 0.002-0.012 (8/8); synonyms one step (cosine 1.000, vs 0.36-0.61 across directions); zeroing direction steps -> 0.39-0.51 | ESTABLISHED (declared secondaries; registered B branch did not fire, 4/8) | `TEXTWORLD_RESULTS.md` |
| Half the seeds also count | 4/8 seeds: common component parallel to the verbs' step (cos +0.999 to +1.000); 31-38 of 64 channels drift between visits (2-4 in map seeds) | ESTABLISHED (secondary + CORRECTED block) | `TEXTWORLD_RESULTS.md` |
| Learned steps beat "only direction words move" | DirOnly 0.970-0.974 on 8/8 seeds; on 2 seeds 100% of its errors name an aside object told at that cell; learned arms make none. Re-scored DirOnly - MapWM -0.024 (p 0.0002), a post hoc correction (eval mode: no difference) | ESTABLISHED (post hoc in the file) | `TW_NORMSTEP_RESULTS.md` |
| NormStep on words | accuracy +0.006, UNMEASURED below 0.096; the registered "per-word clock" is the aside offset (0.126 of 0.133 rad) | ESTABLISHED | `TW_NORMSTEP_RESULTS.md` |
| State sentences | MapWM 0.994 vs RoPE 0.516 (sanity check); C = location + recency among nouns at a place, NOT verb semantics (a verb-agnostic rule scores 1.000 on T2drop); state sentences displace the phase 0.028 moves, like asides; DirOnly worse only under the re-score | ESTABLISHED (CORRECTED 2026-10-11) | `TW_STATECHANGE_RESULTS.md` |
| What/where, causal | converged path models keep 0.974-0.989 with object identity removed from the score, 0.988-1.000 with a shared kernel x content gain; index models fall below the floor | ANALYSIS (post hoc) | `docs/WHAT_WHERE_CHECKS.md` |
| The what-to-where leak | in distribution, removing it (ActOnly / NormStep) gains +0.0107 (p 0.0002, 8/8), equal to MapWM's leak; both remedies SOLVED 16/16 vs MapWM descending. Fresh seeds: SOLVED replicates (8/8 vs 0/8), accuracy does not (+0.0045, p 0.46) | ESTABLISHED (LEAK registered; GAIN_PHASE D1r) | `LEAK_RESULTS.md`, `GAIN_PHASE_RESULTS.md` |
| TinyStories | MapWM 1.5053 vs RoPE 1.5050 nats/token (n = 3); clock share 0.79-0.87; `mr`/`mrs` cancel '.'; a uniform clock costs 0.07-0.09 nats | PILOT (CORRECTED 2026-10-11) | `TINYSTORIES_PILOT_RESULTS.md` |
| Growth exponent on text and code | alpha 0.999-1.006 (TinyStories), 0.990-1.016 (code); R^2(S ~ t) 0.994-0.999 | ANALYSIS | `docs/audits/2026-10-12/alpha_language_out.txt` |
| Code at matched length | position main +0.0056 bpc (3/3, t p 0.024): path integration costs on code | ESTABLISHED (n = 3) | `CODE_FULLVAL_RESULTS.md` |

Not to be revived (CLAUDE.md, withdrawn): "the model tracks state changes"; TinyStories "nothing cancels", "spatial 0%",
"20x the MapWM - RoPE difference"; "two defects, two fixes".

---

## 2. What a one-layer path model learns from words (question 1)

Models: the 8 registered text-world path runs (`runs/textworld/p0`, seeds 0-7) and the 8 MapWM, 8 NormStep and 8
DirOnly runs of TW_NORMSTEP (`runs/tw_normstep/p0`, seeds 10-17). Eval: held-out map, np seed 424242 (used by no
registered script), 40 walks x 1024 words, unless stated. The manual forward used for manipulations reproduces every
model's logits exactly (max |diff| 0.0) and the role-annotated generator reproduces `TextWorld.generate_trajectory`
token for token (`tw_lib.py` checks).

### 2.1 The step table, by word class (ANALYSIS; `tw_step_account_out.txt`)

Steps omega-scaled (radians), relative to the clause-level direction differential |D_N| (one move):

| word class | map-type models (14 learned-step) | clock-type models (10) | what it means |
|---|---|---|---|
| direction words (differential) | 1.00 by normalisation; opposites cancel 0.001-0.013 | same | signed, cancelling steps |
| adverbs, fillers | 0.004-0.03 | 0.004-0.025 | words whose count per move varies get no step |
| `and` / `a` / `she` (shared by fillers, seeing phrases, asides) | 0.02-0.07 | 0.02-0.08 | near zero |
| verbs, seeing words, object words, `.` | 0.02-0.14 | 0.24-1.49 | one per move: free to carry a per-move tick |
| aside-only words (`thought`, `about`, `remembered`, `story`) | 0.07-0.16 per word | 0.41-1.33 per word | large, but cancel within the sentence |
| aside sentence, net (usage-weighted) | 0.05-0.12 of a move | 0.04-0.16 of a move | a small offset off the place |
| DirOnly (oracle): every non-direction word | 0 by construction | -- | asides sit at the place's phase |

The clause tick tau (what every move adds whatever its direction) is 0.03-0.23 of a move (map type) and 3.3-6.8 moves
(clock type, raw norm, all channels), parallel to the verbs' step (cos +0.999 to +1.000 on all 10 clock models).

### 2.2 The tick predicts the drift; the clock counts moves (ANALYSIS)

- **The tick alone predicts which channels drift.** Predicted drift per channel = mean |wrap(lag x tau_c)| over the
  consecutive-visit pairs (median lag 8 moves, 1,221 pairs); observed = mean |wrap(dtheta_c)| at those pairs. Classified at
  1 rad, they agree on 0.98-1.00 of channels on every one of the 32 models (r 0.96-1.00 on the 24 learned-step models).
  The clock seeds drift in 26-38 channels (tw s0/s2/s3/s4: 26/29/35/36; MapWM s11/13/14/15: 38/27/28/30; NormStep s11/16:
  27/28), map seeds in 0-3. (The CORRECTED block's 31-38 counted drift from the FIRST visit; consecutive visits give the
  numbers here.) `tw_step_account_out.txt`.
- **It counts moves, not words or sentences.** Regressing each move's net phase on direction, aside present, filler
  count and adverb present, in the drifting channels: an aside sentence adds 0.009-0.062 of a tick, a filler word
  0.001-0.007, an adverb word 0.001-0.007 (10 clock models; `tw_clock_unit_out.txt`). A word counter would give
  fillers and adverbs ~1/8 of a tick each; a sentence counter would give asides ~1 tick. Every move is one cell, so the
  move count is the path length: **these channels integrate Howard's Case II (distance travelled) while the others
  integrate Case III (position)**.
- **Pi ticks are map features.** On all 10 clock models 2-6 channels tick by exactly pi per move (e.g. MapWM s11:
  -3.141, -3.140 rad). On the even 64-torus every return to a cell takes an even number of moves, so a pi tick is invisible
  to revisits: it encodes the checkerboard parity of the cell. These channels carry up to 0.24 of retrieval usage
  (MapWM s11).
- **Why clock seeds lean on non-direction words.** The registered secondary found that zeroing every non-direction
  step costs the clock seeds 0.33-0.69 and the map seeds 0.05-0.30. In the channels retrieval uses (tick 0 mod pi), the
  direction words' common component is 0.06-0.34 of a move on the clock seeds and 0.01-0.03 on the map seeds (registered
  text-world batch), and the
  other words cancel it within the move. Zeroing them leaves it as a tick in exactly the used channels: across the 8
  seeds the cost tracks that component (Spearman +0.905; `tw_common_used_out.txt`).

### 2.3 The clock is idle: usage and channel ablation (ANALYSIS)

- **Usage.** 79-99% of retrieval usage sits in channels whose tick is 0 mod pi (lowest: tw s0 0.79, unconverged in
  eval mode; NormStep s16 0.82); channels where the clock dominates the phase (rho_c > 1 and more than half a turn per
  window, section 3) carry 0.13-0.30 of usage on clock models and 0.01-0.15 on map models (`usage_tick_out.txt`).
- **Ablation.** Removing the drifting channels' q and k components from the score (re-scored accuracy):

| model | intact | clock channels removed | same number of map channels removed (5 draws) | only clock channels kept |
|---|---|---|---|---|
| tw s0 (eval mode 0.875, unconverged) | 0.973 | **0.717** | 0.43 [0.24, 0.58] | 0.07 |
| tw s2 | 0.999 | 0.990 | 0.25 [0.03, 0.63] | 0.03 |
| tw s3 | 0.998 | 0.993 | 0.06 (all 29 map channels) | 0.06 |
| tw s4 | 0.999 | 0.999 | 0.03 (all 28) | 0.03 |
| MapWM s11 | 0.989 | 0.985 | 0.48 (all 26) | 0.48 |
| MapWM s13 | 0.997 | 0.996 | 0.65 [0.50, 0.84] | 0.48 |
| MapWM s14 (eval mode 0.960) | 0.995 | **0.902** | 0.25 [0.03, 0.72] | 0.03 |
| MapWM s15 | 0.995 | 0.994 | 0.07 [0.02, 0.23] | 0.02 |
| NormStep s11 | 0.998 | 0.996 | 0.46 [0.20, 0.60] | 0.44 |
| NormStep s16 (eval mode 0.835) | 0.982 | **0.809** | 0.22 [0.04, 0.48] | 0.04 |

On the 7 clock models at ceiling in eval mode the clock costs at most 0.009 to remove; the 3 that need it (tw s0, MapWM
s14, NormStep s16: 0.09-0.26) are exactly the ones below ceiling in eval mode, i.e. not converged to the clean solution.
A clock in the channels retrieval does not use is how half the seeds and a perfect map coexist. That unconverged models
still lean on it suggests the clock is used early and abandoned later (SPECULATION; E1 tests it). Why training builds it
at all is not identified.

### 2.4 Asides: the offset is what beats the oracle (ANALYSIS; `tw_attention_asides_out.txt`)

`TW_NORMSTEP_RESULTS.md` found that DirOnly stops at 0.970-0.974 because an aside's noun sits at the place's phase, and
that learned-step models make none of these errors. Two manipulations at eval, each keeping the phase after the aside
unchanged (so only the within-sentence offset changes):

- **Collapse** (learned models): every aside word is put at its clause's object-slot phase. The 19 learned-step models
  at ceiling in eval mode (>= 0.98) fall from 0.989-1.000 to **0.952-0.975**, and **71-100%** of their errors then name an object told in an
  aside at that cell (tw s1-s6 0.94-1.00; MapWM s10, s11, s13, s15-s17 0.71-1.00; NormStep s10-s15, s17 0.86-1.00); before
  the manipulation they make 0-13 errors, none of that kind. That is DirOnly's ceiling and its error signature, produced
  in the models that had escaped it. (The 5 models below ceiling in eval mode respond partly: 0.966-0.994, 0-85% aside
  errors.)
- **Lift** (DirOnly, `tw_attention_asides_dironly_out.txt`): intact, DirOnly scores 0.962-0.981 and **100%** of its
  errors are aside errors on **8/8** seeds (TW_NORMSTEP checked 2). Offsetting its aside words by half a move in-plane,
  with the phase after the aside unchanged, gives **0.999-1.000 on 8/8**; by one move off the map plane (a random
  direction orthogonal to the direction steps) 1.000 on 7/8 (s16 0.972); by a quarter move off-plane 0.963-0.996 (the
  kernel is not sharp at that distance). The oracle's ceiling is exactly the missing aside offset.

So the learned non-direction steps do two jobs: they cancel within every move (so the place's phase is the same
whatever the wording) and they leave a small net offset inside aside sentences (so a mentioned-but-absent object is not
filed at the place).

### 2.5 Attention at revisits (ANALYSIS)

At each revisit query (head mean), the 19 learned-step models at ceiling put 0.55-0.76 of their attention on the object
slot of the most recent visit to that cell, 0.07-0.14 on older visits' object slots, 0.004-0.035 (mean 0.015) on aside
nouns told at the cell, 0.07-0.30 on other tokens of earlier clauses at the cell, and 0.01-0.06 elsewhere. Retrieval is
by place: 0.94-0.99 of the attention lands on tokens told at the current cell. DirOnly puts 0.041-0.077 (mean 0.047,
about 3x) on aside nouns told at the cell and 0.10-0.23 elsewhere.

### 2.6 The text world is a short-walk task: the retrace floor (ANALYSIS; `tw_retrace_floor_out.txt`, `tw_strata_out.txt`)

The text world renders ~132 moves per 1024 words, the walk length at which the main paper's Section 4 shows a map-free
retrace rule scoring 0.934 on the torus. On the text world's own registered eval stream the same rule (port of
`docs/audits/2026-10-16/sign_floors.py`) scores **0.9337** (TW_NORMSTEP stream: 0.9365), while the constant and the
one-step reversal-copy rule reproduce the registered 0.5054 / 0.5965 exactly. Consequences:

- The text world's accuracy margin over the best map-free rule is small: path 0.993 re-scored (0.969 eval mode) vs
  retrace 0.934; DirOnly's 0.972 is barely above it. Index RoPE (0.505 / 0.772) does not reach retrace.
- The map claim must rest on the revisits retrace gets wrong and on the mechanism (cancelling steps; direction steps
  zeroed -> floor). On the first 100 walks of the registered stream, 179 of 3,055 revisit targets (5.9%) are ones retrace
  gets wrong (best constant there 0.112: 'cat'; 'nothing' never):

| model (registered batch, n = 8) | all | RETRACE-OK | NEEDS-MAP (n 179) |
|---|---|---|---|
| path, 1 layer, re-scored | 0.994 | 0.998 | **0.920** (seeds 1-6: 0.944-1.000; s0 0.603, s7 0.832, the two below ceiling in eval mode); fresh seeds 2-7: 0.961 |
| path, 1 layer, eval mode | 0.970 | 0.975 | 0.885 |
| RoPE, 2 layers | 0.779 | 0.819 | **0.138** (0.10-0.17) |
| RoPE, 1 layer | 0.507 | 0.538 | 0.002 |

  Two-layer index RoPE learns part of retrace (two seeds reach 0.93, just under the 0.934 floor; RETRACE-OK 0.98) and
  stays at the constant floor where a map is needed. Path solves those revisits. This is the text-world analogue of
  Section 4's strata and the strongest single piece of evidence that the words are integrated into a map.
- Section 7 of the draft should report retrace beside the constant floor. This does not change any registered verdict.

---

## 3. Theory: when a learned phase is a map and when it is a clock (question 2)

### 3.1 Setting and decomposition (THEORY)

Channel c's score between query t and key t' depends on position only through omega_c (Phi_c(t) - Phi_c(t')), the
summed step over the interval. With token frequencies f_w, split every step into a common part and a content part:
Delta_c(w) = mu_c + d_c(w), mu_c = sum_w f_w Delta_c(w), so sum_w f_w d_c(w) = 0. Over an interval of L tokens

    Phi_c(t) - Phi_c(t') = mu_c L + sum of d_c over the interval.

The first term is a clock (Howard Case I if mu is carried by every token; Case II if only by selected tokens, as in the
text-world move counter); the second is content-driven and can be a map (Case III) only if it cancels over closed
loops. "The common component tracks the frequency-weighted mean step" is true by definition of mu; the substance is in
which channels mu_c survives and why.

### 3.2 T1: the tick rule for a retrieval channel (THEORY, checked)

If the target at t is the key at the last visit to the same latent state z (a cell), matched pairs come at lags
spread over a distribution, and for a channel to score matched pairs consistently its phase difference over them must be
constant mod 2 pi. With the map part cancelling over the closed loop, the remainder is lag x tau_c. Hence: **a channel
can serve latent-keyed retrieval only if its tick is 0 mod 2 pi / g**, g the gcd of the possible return lags (g = 2 on a
bipartite lattice such as the even torus), and otherwise it decorrelates once |wrap(tau_c)| x sd(lag) exceeds ~1 rad.

Checks (section 2.2-2.3): the tick alone predicts each channel's revisit drift (agreement 0.98-1.00, 32 models); used
channels have tick 0 mod pi (79-99% of usage), including exact-pi parity channels; channels that violate T1 are not used,
and removing them costs <= 0.01 on converged models.

### 3.3 T2: the readout decides, channel by channel (THEORY + ANALYSIS)

T1 constrains mu_c only where the readout looks. Elsewhere a tick is free, and the text world shows one surviving in
26-38 channels on 10 of 24 models without cost. Conversely, when targets are keyed to a count back (k-back recency) or
to recent context (language), a nonzero mu_c in the used channels is what the readout needs. So the prediction is about
**used** channels: their drift/content ratio rho_c = |mu_c| L / range(content_c, L) at the training length L should be
<< 1 when targets are keyed to a latent state reached at variable lags, and >> 1 when they are keyed to a token or event
count.

Check (`crossover_usage_out.txt`, `usage_tick_out.txt`; usage-weighted median rho over channels, all-token usage for the
multi-layer language models):

| data (all MapWM, signed step) | used-channel rho (median) | usage in clock-dominated channels | seeds |
|---|---|---|---|
| 2D torus, 1 layer, matched length (sign_matched Signed_r4) | 0.064-0.079 | 0.000 (0/64 channels) | 8 |
| text world, 1 layer (24 learned-step models; tw s0 unconverged 2.5) | 0.06-0.23 | 0.01-0.30 | 24 |
| code, bytes, 9 layers (trained at 2048) | 4.5-6.6 | 0.50-0.55 | 3 |
| TinyStories, words, 6 layers (trained at 512) | 9.9-12.3 | 0.42-0.43 | 3 |
| k-back recency with fillers, 1 layer (recency Signed_r4) | 21.8-22.1 | 0.93-1.00 | 8 |

The same architecture and parameterisation (the torus and recency rows are the identical `Signed_r4` class) put its
used channels at opposite ends; language and code sit on the recency side. (Clock-dominated: rho_c > 1 and more than half
a turn per training window. On language part of the usage is in slow channels with rho > 1 that turn less than half a
turn per window -- slow recency kernels -- which is why the share is 0.42-0.55 while the median rho is 4.5-12.)

### 3.4 What the growth exponent can and cannot show (ANALYSIS; `tw_alpha_drift_out.txt`)

The pooled exponent alpha (range of the raw summed step ~ T^alpha, the readout of `alpha_language.py`) weights channels
by their raw range, so a large tick in channels attention ignores dominates it:

| models | alpha | pooled rho(1024) | used-channel rho | accuracy |
|---|---|---|---|---|
| text-world map seeds (14) | 0.61-0.74 (the true walk: 0.668) | 0.12-0.86 | 0.06-0.13 | 0.985-1.000 (re-scored) |
| text-world clock seeds (10) | 0.98-1.02 | 9.3-13.9 | 0.07-0.23 (tw s0, unconverged: 2.5) | 0.989-0.999 at ceiling (7) |
| DirOnly (8) | 0.60-0.65 | 0.05-0.07 | -- | 0.962-0.981 (this stream) |
| TinyStories (3) | 1.00-1.01 | 11.9-12.5 | 9.9-12.3 | MapWM = RoPE |
| code (3) | 0.99-1.01 | 2.9-5.0 | 4.5-6.6 | path costs +0.0056 bpc |

The clock seeds and TinyStories are indistinguishable by alpha and pooled rho, yet one carries a perfect map. So the
draft's "it grows linearly with token index (R^2 0.994-0.999)" does not by itself show the absence of a map. The
used-channel rho does separate them (0.07-0.23 vs 9.9-12.3): on TinyStories the channels attention uses are dominated by
the clock part. The two-part model accounts for alpha: the exponent of |mu| T + range(content, T) is within 0.01-0.07 of
the measured alpha on every model (it over-predicts by 0.01-0.03 on map-type models and under-predicts by 0.03-0.07 on
clock-type models, TinyStories and code, since the range of a sum is not the sum of the ranges).

### 3.5 Howard's cases in one model (THEORY + ANALYSIS)

A signed, content-dependent rate can realise all three of Howard et al.'s cases: Case I (mu carried by every token),
Case II (a non-negative tick on selected tokens: the text-world move counter; the TinyStories punctuation and boundary
ticks read as an event counter of the same kind, an interpretation), Case III (signed steps that cancel). Which one a channel ends in is set by what the
readout keys on, not by the parameterisation: torus and text world put their used channels in Case III, recency and
language in Case I/II. The sign result of Section 6.2 is the complement: a non-negative step can represent Case III in
the rotary slot (Proposition 1) but does not learn it from a small start.

SPECULATION, not tested: language would put used channels in Case III wherever prediction pays for a latent state
reached by variable routes (dialogue about places, procedural text). The text world says a one-layer model finds it when
the words that move are context-free; TW_AMBIG (section 7, E2) would test the context-dependent case.

---

## 4. The integrator framing (question 3)

### 4.1 What is prior art (from `docs/lit/LIT_HOWARD_TRANSFORMER.md`, `LIT_CLOCK_MAP.md`)

- Howard et al. 2014 (J Neurosci 34:4692): one integrator bank driven by a rate alpha(t) codes time (alpha constant,
  Case I), distance travelled (alpha = speed, Case II) or position (alpha = signed velocity, Case III: "like a bead on a
  wire"). The dichotomy, with the collision property, is theirs.
- Velocity-modulated integrators in trained networks: Alipour et al. 2025 (J Comput Neurosci 53:131; Cases I and III,
  alpha given, real s); Mochizuki-Freeman et al. 2023 (IJCNN; alpha = dz/dt from the environment). The Howard and Tiganj
  labs keep s real and explicitly reject complex s for stability (Mochizuki-Freeman 2023 fn. 1; Kabir et al. 2025, AAAI;
  Dickson et al. 2025).
- RoPE as a velocity-controlled oscillator: GridPE (Li et al., arXiv:2406.07049): Phi(t) = omega_s t + beta omega^T x(t),
  reducing to RoPE in 1D; coordinates given, not learned from content.
- Laplace memory and SSMs: Sarkar and Howard 2026 (arXiv:2601.02618) relate SITH's diagonal recurrence to discretised
  SSMs (decay side, not the step Delta_t, not phase).
- Content-gated phase rate motivated by Howard: RoMem (Li et al. 2026, EMNLP; arXiv:2604.11544), monotone (Case II-like),
  for temporal knowledge graphs.
- Delta_t as a learned stand-in for time or arc length: TIDES (Soydan et al. 2026, arXiv:2605.09742), no signed case.
- Index position as "linear structure" and path-integrated position as a learned map in a transformer: TEM-t (Whittington
  et al. 2022). Sign and state tracking: Sarrof et al. 2024, Grazzi et al. 2025 (Thm 3 for rotations), Mamba-3 Sec. 3.2.

### 4.2 What is ours

1. The explicit identification, for rotary attention: index RoPE, monotone content-dependent phase (CoPE, CARoPE,
   softplus or |.| steps) and signed content-dependent phase (MapFormer, Selective RoPE, Mamba-3's data-dependent RoPE)
   are Howard's Cases I, II and III with an imaginary rate, and Howard's alpha(t) is the selective-SSM step Delta_t and
   MapFormer's content-dependent angle step. Not found stated (LIT_HOWARD_TRANSFORMER sec. 1); one sentence once the
   pieces are side by side, so present it as an organising remark, not a result.
2. The rotary slot is where the signed rate is bounded (a rotation), the move the Howard lab declined for real s; and in
   that slot Cases II and III coincide in what they can represent (Proposition 1, extending Grazzi Thm 3 / Mamba-3 to
   rank-limited step maps). The distinction becomes one of learnability (Section 6.2).
3. Measured: which case a learned signed rate realises is decided per channel by the data (section 3.3), and one model
   can run Case III and Case II in different channels at once (section 2.2).

---

## 5. What/where and the leak (question 4)

Recommendation: keep both in the main paper only as one paragraph closing Section 7, stated as a single principle with
the text world, and move the detail to App. K.

- **The principle (THEORY, light).** Retrieval by place needs every token that should be retrieved from a place to sit
  at the same phase offset from that place, and every token that should not be retrieved there to sit elsewhere. So a
  token's offset must depend on its role, not its identity. The leak violates the first half (an object's step depends on
  its identity, so objects at one place spread over identity-dependent offsets: LEAK, registered); DirOnly violates the
  second (an aside's noun sits at the place's phase); learned steps satisfy both in the text world (asides offset
  0.04-0.16 of a move, observed objects at a fixed offset).
- **The leak, as registered** (`LEAK_RESULTS.md`): +0.0107 in distribution (p 0.0002) for removing it, equal to MapWM's
  own leak, and the remedies converge where MapWM is still descending. Must be said with its replication: on fresh seeds
  (GAIN_PHASE D1r) SOLVED replicates (8/8 vs 0/8) but accuracy does not (+0.0045, p 0.46). Effect size ~1%, one layer,
  partly convergence speed (r(loss, acc) -0.94). That is a supporting result, not a contribution line.
- **What/where checks** (post hoc, `docs/WHAT_WHERE_CHECKS.md`): already in the draft's Discussion with the right
  status; keep it there and add that separation was shown in embeddings by MapFormer (Fig. 9), so our addition is that the
  trained models rely on it.
- Do not include: GAIN_PHASE's gain score (no difference, D2), NormStep on words (unmeasured), the state-change C as
  anything beyond location + recency.

---

## 6. Proposed paper text

### 6.1 Section 7, replacement (about 0.9 page plus figure)

> **7. Maps from words**
>
> The text world asks whether the path model can find its actions when nobody marks them. Each move of the torus walk is
> written as a sentence: a direction word chosen from three synonyms, verbs that do not move the walker, optional
> adverbs and fillers, and, after some moves, an aside that names an object without visiting it ("she thought about a
> cat ."). Nothing in the input says which words move the walker (vocabulary and step tables: App. J). The walks are
> about 132 moves long, so the map-free retrace rule of Section 4 already scores 0.934 here; we report it with the
> constant (0.505) and reversal-copy (0.597) floors.
>
> **One path layer learns navigation told in English.** On fresh seeds (n = 6) one path layer scores 0.979 +/- 0.048
> (6/6 SOLVED) against 0.505 for one-layer index RoPE (0/6), +0.474 (permutation and Fisher p 0.0022); two index layers
> reach 0.793 (0/6). Both index models stay below retrace on every seed; path clears it (0.993 on all eight seeds with the
> attention-dropout re-score, App. C). On the revisits retrace gets wrong -- places reached by a route other than undoing
> the last run, 5.9% of targets, best constant 0.112 -- path scores 0.961 on the fresh seeds (0.944-1.000 on every seed
> at ceiling) and two index layers 0.10-0.17 (analysis).
>
> **What the step table holds.** On every seed, opposite direction words cancel once the component common to the four
> directions is removed (0.002-0.012), synonyms collapse to one step, and zeroing the direction steps drops accuracy to the
> constant floor (declared secondaries). The rest of the vocabulary is organised by how often a word occurs per move
> (analysis, 32 models, Fig. 7d): adverbs and fillers, whose count varies, get steps of 0.004-0.03 of a move; words that
> occur once per move (verb, seeing words, object, full stop) may carry a common tick; aside sentences get sizeable word
> steps that cancel within the sentence, leaving the aside's noun 0.04-0.16 of a move off the place.
>
> **The offset is why learned steps beat being told.** A model whose steps are restricted to the twelve direction words
> stops at 0.970-0.974 on every seed, and its errors name objects told in an aside at that place: with no other word
> stepping, the aside's noun sits at the place's phase (Sec. 6 of `TW_NORMSTEP`). Moving each aside back onto its
> place's phase at evaluation, with the phase after it unchanged, drops the learned-step models from 0.989-1.000 to
> 0.952-0.975, with 71-100% of their errors now of the same kind; offsetting the oracle's aside words by half a move
> lifts it to 0.999-1.000 on every seed (analysis). "Only actions move" is the wrong separation in language: some words that are not actions need steps, so
> that what is mentioned is not filed where it is not.
>
> **Half the models also count moves, in channels they do not use.** On 10 of 24 learned-step models a tick per move
> rides on every move, parallel to the verbs' step. It counts moves -- an aside sentence adds at most 0.06 of a tick, a
> filler word at most 0.007 -- so in Howard et al.'s terms these channels integrate distance travelled (Case II) beside
> the others' position (Case III). The tick alone predicts which channels drift between visits (agreement 0.98-1.00), and
> those channels are idle: 79-99% of retrieval attention uses channels whose tick is zero modulo pi (on the even torus a
> return always takes an even number of moves), and removing the 26-38 clock channels from the score costs at most 0.009
> on every clock model at ceiling (7), while removing as many map channels collapses accuracy (analysis).
>
> **What must and must not move.** The same rule covers the what-to-where leak on a task with new objects at every
> episode: a step that depends on an object's identity spreads objects at one place over identity-dependent offsets, and
> removing it gains +0.011 in distribution (registered, p 0.0002; on fresh seeds the convergence gain replicated and the
> accuracy gain did not, App. K). A token's offset from its place should depend on its role, not on its identity.

Figure 7: keep panels (a)-(c); add (d) the step table by word class (bars: direction differential, once-per-move words,
optional words, aside words, aside net) for map and clock seeds, and (e) the aside collapse/lift test (accuracy and
aside-error share). Caption must say "eval mode" for (c) as now and "re-scored" for (e).

### 6.2 Background 2.1, replacement of the Howard sentences

> **2.1 Rotary position as an integrator.** [first two sentences unchanged] Howard et al. (2014) drive one bank of
> integrators by a rate alpha(t): a constant rate codes elapsed time (Case I), the speed codes distance travelled (Case
> II), and the signed velocity codes position, the same for every visit to a place "like a bead on a wire" (Case III).
> The rotary families are these three cases with an imaginary rate: index RoPE adds a constant angle per token; CoPE,
> CARoPE and non-negative content steps advance by a non-negative, content-dependent amount; MapFormer, Selective RoPE and
> Mamba-3's data-dependent RoPE add a signed one, and alpha(t) is the selective-SSM step Delta_t. Velocity-modulated
> integrators have been trained in networks with a real Laplace variable (Alipour et al. 2025), and RoPE has been derived
> as a velocity-controlled oscillator on given coordinates (GridPE, Li et al. 2024); the machine-learning models of the
> Howard and Tiganj labs keep the Laplace variable real and set the complex (oscillatory) case aside as unstable
> (Mochizuki-Freeman et al. 2023; Kabir et al. 2025). In the rotary slot a signed rate is a bounded rotation, and the three cases differ not in what a
> channel can represent (Sec. 2.3) but in what the data and the optimiser make of it. TEM and TEM-t (Whittington et al.
> 2020, 2022) frame the Case III code as a "where" learned apart from the "what".

### 6.3 Discussion, replacement for "The integrator view" and "Text"

> **The integrator view.** A learned, signed, content-dependent rate can realise all three of Howard et al.'s cases, and
> which one a channel realises is set by what the readout keys on. Split each step into the frequency-weighted mean (a
> clock) and a content part. For targets keyed to a place reached at variable lags, a channel is usable only if its tick
> is zero modulo the lattice's return period, and the models we train put their used channels there (torus: clock part /
> content part 0.06-0.08; text world 0.06-0.23). For targets keyed to a count back, the used channels are clocks (k-back
> recency, the same one-layer architecture: 22). One text-world model can do both at once, with distance travelled in
> idle channels and position in used ones. In the rotary slot the cases coincide in what they can represent (Proposition
> 1), so the sign result of Section 6.2 is about learnability: a non-negative step could hold a Case III code and does
> not find one.
>
> **Text.** On ordinary text at small scale (analysis, n = 3), the channels attention uses are clock-dominated (clock part
> / content part 9.9-12.3 on TinyStories, 4.5-6.6 on code, against 0.06-0.23 in the text world), MapWM matches index RoPE
> on TinyStories (1.5053 against 1.5050 nats per token), and at matched length path integration costs on code (+0.0056
> bits per byte, 3/3 seeds, t-test p 0.024). The phase grows linearly with the token index (R^2 0.994-0.999), but that
> alone does not exclude a map: text-world models that carry a perfect map show the same growth, carried by channels
> attention does not use. Where a map would pay in natural language, and whether a step that reads context would find it,
> is open.

### 6.4 Limitations, add

> - The text world is scripted (58 words, one movement clause per move, context-free direction words) and its walks are
>   short enough that retrace scores 0.934; the map evidence there is the mechanism and the revisits retrace gets wrong.

### 6.5 Corrections to the current draft (Sections 7-8)

- Section 7 reports only the constant and reversal-copy floors; add retrace (0.934).
- "Half the seeds add a per-step clock alongside the map, in 31-38 of 64 channels, and solve as well": add that the clock
  counts moves (not words) and sits in channels retrieval does not use (section 2.2-2.3), else a reader takes it as a
  failure mode.
- "zeroing the non-direction words' steps costs it more (0.33-0.69 against 0.05-0.30)": the reason is the direction words'
  common component in the used channels, which the other words cancel (Spearman +0.905, section 2.2), not that the clock
  is used.
- Discussion "Text": "grows linearly with token index ... consistent with there being no map to build" -- replace by
  6.3 (the linear growth is not diagnostic).
- References to add: Alipour et al. 2025; GridPE (Li et al., arXiv:2406.07049); Mochizuki-Freeman et al. 2023; Kabir et
  al. 2025 (arXiv:2412.15292); optionally Sarkar and Howard 2026 (arXiv:2601.02618), RoMem (arXiv:2604.11544).

---

## 7. Proposed GPU experiments (with cost)

Costs from TW_STATECHANGE (32 text-world runs, 900 epochs, T = 1024, 1 layer, on one GPU in 5.7 h at 2 jobs/GPU, about
0.18 GPU-h per run) and TW_AMBIG's own estimate (64 runs at 1800 epochs, ~15 h). Each needs the rule-29 audit and a
pre-registration before launch; pilots on outside seeds.

| id | question | design | cost | priority |
|---|---|---|---|---|
| E1 | Why do half the seeds build an idle move counter? Hypothesis: it appears early (when recency -- reversal copying, floor 0.597 -- is the cheapest cue) and attention later leaves its channels rather than unlearning it | 8 fresh MapWM text-world seeds, 900 epochs, step table and per-channel usage saved every 25 epochs (no change to training). Registered readouts: epoch at which the tick exceeds 1 move and the map's opposition falls below 0.05; usage of later-clock channels over time | 8 runs, ~1.5 GPU-h | medium (explains a figure panel; not a claim) |
| E2 | Does a context-dependent step learn which uses of a word move (TW_AMBIG, prepared, STOPPED 2026-10-11) | direction words also used as non-moving content; MapWM vs oracle vs HSR (hidden-state step), cue near/far | 64 runs at 1800 epochs, ~15 h (prereg's estimate) | low for this paper (a separate question; mechanisms published, LIT_CONTEXT_STEPS) |
| E3 | Does the used-channel ratio move with the readout inside one stream? Double dissociation | text world where a fraction of targets ask "what did she see k moves ago" (count-keyed) and the rest are revisits (place-keyed); 8 seeds. Prediction: used channels split into rho << 1 and rho >> 1, and ablating each set removes only its query type | 16 runs (with a revisit-only control), ~3 GPU-h | high if the theory paragraph is kept: it turns section 3.3's across-task comparison into a within-model test |
| E4 | Is the text-world advantage over retrace a map at long walks? | text world at 4096 words (~530 moves), where retrace should fall as on the torus (0.843 at 1024 moves); path vs RoPE 1-2 layers, 8 seeds | 24 runs at 4x tokens per run (attention cost grows faster than 4x), ~20-35 GPU-h; --fast-attn valid (no MapEM) | medium: removes the short-walk caveat of 2.6 |
| E5 | Retrace floor and strata re-scored on TW_NORMSTEP / TW_STATECHANGE batches | CPU, no training | 0 GPU, ~1 h CPU | do before any text-world number goes in the paper |

---

## 8. Analyses run (scripts and outputs)

All in `docs/paper/expansion/scripts/language/`; run from `/home/prashr` with `PYTHONPATH=/home/prashr python3
<script>`; CPU only, eval only, nothing written outside this directory.

| script | output | what |
|---|---|---|
| `tw_lib.py` | -- | loaders, role-annotated generator (checked token-identical), manual one-layer forward (checked logit-identical, max diff 0.0) |
| `tw_step_account.py` | `tw_step_account_out.txt`, `.json` | step table by class, clause regression (tick, differentials, opposition), predicted vs observed drift, usage, clock share, aside net offset (32 models) |
| `tw_attention_asides.py` | `tw_attention_asides_out.txt` (no JSON: the run was stopped after the 24 learned-step models, its DirOnly part being run separately) | attention at revisits, aside collapse / lift, clock-channel ablation with random-channel controls, lag strata |
| `tw_attention_asides_dironly.py` | `tw_attention_asides_dironly_out.txt`, `.json` | the same for the 8 DirOnly models, with a quarter-move off-plane lift added |
| `tw_clock_unit.py` | `tw_clock_unit_out.txt` | what the clock counts (moves vs sentences vs words) |
| `tw_common_used.py` | `tw_common_used_out.txt` | why clock seeds depend on non-direction words (Spearman +0.905) |
| `tw_alpha_drift.py` | `tw_alpha_drift_out.txt`, `.json` | alpha, pooled rho, two-part prediction, true-walk reference; text world, TinyStories, code |
| `usage_tick.py` | `usage_tick_out.txt`, `.json` | used-channel rho and clock-channel usage; text world, TinyStories, code |
| `crossover_usage.py` | `crossover_usage_out.txt`, `.json` | the same for the torus (sign_matched Signed_r4) and recency (Signed_r4) |
| `tw_retrace_floor.py` | `tw_retrace_floor_out.txt` | retrace on the text world's eval streams |
| `tw_strata.py` | `tw_strata_out.txt` | accuracy on RETRACE-OK vs NEEDS-MAP revisits, path / RoPE 1L / RoPE 2L, registered batch |

Caveats on all of the above: post hoc, not pre-registered, no independent code audit (rule 29 applies before any of it
is cited as more than analysis); one eval stream per analysis (40-100 walks); thresholds (1 rad drift; 0.05 rad for
"0 mod pi"; rho > 1 and half a turn for "clock-dominated") were chosen by me before looking at the outputs except the
0 mod pi rule, which was added after seeing the pi channels; the usage weight is a proxy (q and k norms), not a causal
measure (the channel ablation is the causal check). TinyStories and code are n = 3 pilots/analyses, small models.
