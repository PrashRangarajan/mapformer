# Cognitive maps beyond navigation: what a path-integrated "where" is for text (2026-10-04)

Status: THEORY note plus one POST HOC eval-only probe (section 2.1; script in the appendix, not committed; copy it
to `docs/audits/` before citing, rule 7). No GPU used. Inputs: `TEXTWORLD_RESULTS.md`, `TW_NORMSTEP_RESULTS.md`,
`CONTEXT_STEP_DESIGN.md`, `CTXSTEP_PILOT1-3.md`, `CTXSTEP_HS_RECIPE.md`, `CTXSTEP_HSR_PILOT.md`, `CTXSTEP_PREREG.md`,
`environment_textworld*.py`, `docs/lit/LIT_*.md`, Jericho feasibility (`/home/prashr/jericho_data/feasibility/`),
`DYCK_MDEPTH_RESULTS.md`, `CODE_FULLVAL_RESULTS.md`, `JSB_RESULTS.md`, memory notes on PoPE and language.
"Possibly new" = not found in `docs/lit/` or in the web search below; not a novelty claim.

## Key claim

**For text, MapFormer's "where" between two tokens is a linear image of the bag of (context-resolved) events
between them, read modulo the phase periods; it beats the index exactly when four conditions hold together:
(R) the target is decided by RETURNING to the same group element (an equality / revisit query), (U) the updates
are relative and the coordinate is UNNAMED (no landmark in the content identifies it), (N) the NARRATION order
is the walk order, and (A) the event algebra on the relevant dimension is ABELIAN (translations, signed time
shifts, nesting depth, counts).** The scripted text world meets all four (+0.464). Natural corpora rarely do:
places are named (Jericho), most dependencies are clock-like (code, Bach, enwik8), relations are stated out of
order (StepGame-type descriptions, the human "map making" of Park et al. 2020), and state changes are often
resets or permutations (bAbI, Kim & Schuster boxes). That accounts for every language-side null we have, and it
says where to look for a positive: unnamed relative dimensions narrated in order -- route directions, and above
all **narrative time** ("two days later", "a year earlier").

## 1. "Where" for a text stream, precisely

### 1.1 Definition
Context-free step (`model.py`): `theta_t = omega (.) sum_{u<=t} Delta(x_u)`, `Delta(x) = W_out W_in emb(x)`. The
score between query t and key s depends on position only through

    dtheta(s,t) = omega (.) M c(s,t),   M = W_out W_in E^T  (rank <= r),   c(s,t) in Z^V = word counts in (s, t].

So the "where" is the **interval's bag of words, projected to r dimensions and wrapped mod 2 pi / omega_i**.
With a context-dependent step (CG, HS, HSR) replace word counts by counts of context-resolved events
(e.g. "north, negated" vs "north, moved"). The family, in this language:

| scheme | dtheta(s,t) is a function of | group |
|---|---|---|
| index RoPE | `|t - s|` (every token +1) | Z, uniform clock |
| CoPE | number of gated tokens in the interval | monotone scalar (N) |
| CARoPE / GRAPE contextual / our "Abs" | non-negative weighted count | monotone, a content clock |
| MapFormer CF | signed r-dim linear image of word counts | Z^r -> torus |
| CG / HS / HSR | signed image of counts of context-resolved events | same, after a parser |

**Definition (exploitable "where").** A stream has one for target y if there is a monoid homomorphism `phi` from
token (or event) sequences into an abelian group G with `y_t = y_s` whenever `phi(s..t) = 0` and the content at
s is the thing to retrieve. MapFormer realises `phi` linearly in counts and needs per-head rank >= dim G (our
rank line: = D is hard, D+1 is easy on large tori).

### 1.2 Consequences
1. **Only equality queries.** The phase enters only through `Q^T R(dtheta) K`; it is never a value. MapFormer can
   answer "what was here before?" and "what happened on that day?", not "how many coins does she have?" or
   "which day is it?". A text task must be phrased as retrieval at a matched coordinate (R).
2. **Abelianisation.** A commutative phase sees any event algebra only through its abelianisation. For swaps
   of n boxes (S_n under transpositions) every one-dimensional unitary representation is trivial or the sign,
   so the only group-consistent quantity the phase can carry is the **parity** of the swap count. Diagonal /
   commutative recurrences being limited to abelian state tracking is known (Grazzi et al.; Merrill et al.;
   `papers/INDEX.md` `path`, `grazzi`). Possibly new: the link to Li, Guo & Andreas (ICML 2025), whose trained
   transformers solve S_n with a "parity-associative" algorithm that first applies a parity heuristic -- the
   exact quantity a MapFormer phase could supply at layer 0. Our rotation-action boundary (+0.050, allocentric
   recoding +0.488) is the same fact in navigation: egocentric "turn left, walk on" is SE(2), non-abelian;
   recoding makes it abelian.
3. **Resets are not group elements.** "She went to the kitchen", "On Monday", a scene cut: constant maps. A
   cumsum cannot express them; attention can, by recency or by matching the name (a landmark). By Krohn-Rhodes
   an event automaton is a cascade of groups and resets (used for transformers by Liu et al. 2023, "shortcuts
   to automata"): the path phase is a fast channel for the abelian group part only.
4. **Index is right** when the target depends on text distance (recency, n-back, copying, local syntax) or the
   update is the same for every token. Consistent with our data: recency monotone cost -0.004; code at matched
   length +0.0056 bpc against path integration (t p 0.024, n=3); Bach nothing at best-val; enwik8 underpowered.
5. **Narration order (N).** The phase integrates along TEXT order. Relational descriptions given out of walk
   order ("A is left of B. C is above A. B is ...") define positions by a constraint graph; no prefix sum computes
   them. MapFormer is a path integrator, not a map maker.

### 1.3 Which text has the structure (A + U + N + R)

| text kind | event algebra | landmark? | prediction for path vs index |
|---|---|---|---|
| route directions with unnamed places ("two blocks north, then east") | Z^2 translations | no | path wins (text world: +0.464, 1L vs 1L) -- **if** magnitude x direction is handled (2.3) |
| egocentric directions ("turn left, go on") | SE(2), non-abelian | no | needs heading recoding (boundary row) |
| named-place narratives (bAbI 1-3, Jericho, Talk the Walk) | resets + names | yes | no path benefit: Jericho rooms are named (first line exact on revisit 0.68-1.00), maps are fixed (memorised table 0.93-1.00) and near-trees (cycle rank <= 3 in 28/56 games); in-context edge recall + constant >= 0.90 in 45/54 games |
| narrative time ("the next day", "three years earlier") | Z, signed; dates are resets | rarely | **path's best natural target**: story time decoupled from text time; flashbacks are negative steps |
| nesting / discourse depth (Dyck, parentheticals, quote depth) | Z (depth) | no | depth is abelian, matching is a stack: depth substitution (1 path layer ~ 3 attention layers, `DYCK_MDEPTH_RESULTS.md`) |
| quantities ("gained 3, lost 2") | Z^k | no | only via equality queries (1.2.1); contrived in text |
| entity state with swaps (boxes) | S_n | -- | no gain beyond parity (1.2.2) |
| relational descriptions (StepGame, social hierarchies, CLUTRR) | Z^2, Z (generation) | no | helps only in chain order; shuffled order needs graph inference |
| conceptual spaces (Constantinescu bird morphs) | Z^2 in feature space | the content IS the coordinate | path unnecessary when the morph is observable; needed only when only relative change is narrated |

Situation models (Zwaan & Radvansky's event-indexing model) index events on time, space, causation,
intentionality and protagonist. Two of the five -- **space and time** -- have the abelian, relative structure a
path phase exploits; protagonist is a pointer (reset / name), causation and goals are graphs. Possibly new as a
statement; it predicts that a path-integrated phase is a module for the spatiotemporal part of a situation model,
not a situation model.

Mechanistic contrast with LMs (Tang et al. 2026, "Do LMs track entities across state changes?"): trained LMs do
NOT update state incrementally; they aggregate the relevant operations in parallel at the query token. That is
the index model's strategy and costs depth with the number of operations; path integration is the incremental
alternative at O(1) depth. Our exchange rate (1 path layer ~ 3 attention layers, on Dyck and H3) is the
small-scale measurement of that difference.

## 2. Constraints on a language step, from our results

### 2.1 Asides: "only the action words move" is the wrong target
Registered/post hoc (`TW_NORMSTEP_RESULTS.md`): DirOnly (steps only on direction words) caps at 0.970-0.974; on
2 seeds 100% of its errors are an aside object ("she thought about a cat .") heard at the same cell; learned-step
models make none. New post hoc probe (appendix; held-out map, 15 walks, eval mode, head-averaged attention from
the revisit query to keys at the same cell, aside noun different from the true object, 381 pairs per run):

| run | attention to first-visit object | to aside noun at that cell |
|---|---|---|
| MapWM s16 / s10 / s13 | 0.679 / 0.478 / 0.574 | **0.020 / 0.005 / 0.046** |
| NormStep s12 / s14 | 0.717 / 0.695 | **0.013 / 0.015** |
| DirOnly s10 | 0.542 | 0.070 |

Learned steps cut attention to aside nouns 1.5-14x relative to DirOnly, with tiny displacements: the aside's
net phase offset is 0.013-0.078 rad mean over channels (0 channels > 1 rad), against a direction step of
0.09-0.23 rad (16 runs, `aside_loop` in the appendix). In this grammar the aside is NOT a closed detour: only "."
follows its noun, so the noun's offset is exactly the displacement the walker keeps (it is part of the per-word
drift NormStep's readout B measured). Reading: **the phase is a two-level address, place + clause role.** The
words since the last move ("thought about a" vs "and saw") put a small role-specific offset on the noun, enough in
the sharp channels to separate "mentioned here" from "located here". DirOnly n=1 for attention; post hoc.
Constraint: a language step map must give WHAT-side words small, role-specific steps, and pays for them in drift
unless a later word cancels them. Cheap eval-only follow-up: raise `p_aside` at eval on the stored 32 runs and
read accuracy vs aside count per walk (drift accumulates linearly if nothing cancels).

### 2.2 The window limit: what the step must read
Pilots (n=1-2, `CTXSTEP_*`): steps reading a 4-token window suppress decoys only with the cue 1-3 tokens away, on
either side (via gate x lag interactions); a hidden-state step (attention, then the step) reaches 6-13 tokens when
it learns a step; HSR (word step + alpha LN(h1), alpha=0) learns one 4/4. As constraints on a language map:

| phenomenon | what the step must depend on | mechanism class | expectation |
|---|---|---|---|
| lexical direction ("north") | the token | CF | solved (text world) |
| magnitude x direction ("two days later") | 1-3 neighbours, **multiplicatively** | window gate (CG) | CF provably cannot (2.3) |
| argument order ("A is left of B" vs "B is right of A") | 1-3 neighbours, sign flip | window | near-cue pilots say yes |
| local negation ("did not go north") | the negator, usually 1-3 tokens | window | fine in typical English (dependency distance short; unmeasured on a corpus) |
| long-scope negation / hedges ("never, after a long pause, went north") | a cue 6-13 tokens away | hidden-state step (depth) | pilots: window fails, HS/HSR reach |
| **frames**: hypothetical, dream, plan, reported speech, flashback | a discrete FRAME STATE over clauses | attention computes it (a flip-flop: last of {open, close}); step at depth >= 1 | possibly new framing |
| retraction of a span ("... -- no, that was a dream") | the NET displacement of the whole span | needs a phase-restore primitive or attention summing the span's step contents | predicted hard; one-word retraction worked in pilots |
| implicit flashback return ("back in the present") | restore theta to its value at the frame opening | not a group element; no current mechanism | possibly new boundary |
| reported belief / false belief (Sally-Anne, ToMi) | a second trajectory gated by who observes | per-frame (per-head) integrators with separate gates | possibly new framing; heads share an angle in Mamba-3 |

Two solution classes for frames, distinguishable post hoc: **gate-inside** (g_t = 0 while the frame is open;
needs the frame state at each word, which a single attention layer gets by recency) and **cancel-at-close** (a
step equal to minus the span's displacement at the closing word; needs the span's sum). Prediction: with a
leading frame marker models learn gate-inside; with only a trailing retraction over >= 2 moves, every single-path
design fails or learns slowly. The trailing-cue success in pilot 3 / HSR is the one-move special case.

### 2.3 A step is a product, not a sum (possibly new as a test)
English separates magnitude and direction into words. Suppose additive token steps had to realise
"k days later/earlier", k in {1,2,3}: `D(k)+D(days)+D(later) = k u` and `... + D(earlier) = -k u` give
`D(later) - D(earlier) = 2k u` for every k, so `2u = 0` modulo every channel period: the phase can then track only
parity. A context gate within 3 tokens can (`g = k/3` on the direction word), as can HS/HSR. Our text world never
posed this: each direction word is a whole step. Real route directions ("three blocks north") and narrative time
both do.

## 3. Proposed tasks, ranked by information per GPU-hour

Cost basis: the text-world batch (32 runs, T=1024, 1 layer, 900 ep) ~3 h wall on two 4090s, i.e. ~0.19 GPU-h per
1-layer run, ~0.38 for 2 layers; context-step runs at T=2048, 1800 ep ~0.56 (1L) / ~1.05 (2L) GPU-h. Every task
below is a `TextWorld` subclass in the style of `environment_textworld_ctx3.py`, gated on CPU first (rule 11:
word n-grams 1-5, constant, reversal-copy, plus the task's own trivial rule named below), pilots on outside
seeds, one batch, 6-8 seeds.

### 1. Landmarks vs path integration in words (cue competition) -- ~14 GPU-h, ~7 h wall
- **Task.** Text world + place names: per sequence, an injective random naming of cells from a pool of 512 name
  tokens (fresh per walk, so names must be bound in context; the object map stays the run's map). On arrival
  the clause adds "and reached <name>" with probability p_name in {0, 0.5, 1}. Eval-only knobs: names stripped
  at test; accuracy split by named / unnamed revisits.
- **Arms.** Path 1L (r=4), RoPE 2L (name copying needs an induction circuit), 8 seeds x 3 p_name = 48 runs.
- **Floor / trivial predictor.** Name-copy oracle (object last seen with the same name; = named-revisit share at
  p_name=1), plus reversal-copy and constant; n-grams cannot use per-walk names.
- **Predictions.** RoPE 2L climbs toward the name-copy oracle with p_name; path 1L stays near 0.97 at all
  p_name; the path advantage shrinks to roughly the unnamed-revisit share. The informative cell: path 1L trained
  at p_name=1, evaluated with names stripped. Map kept = path integration is learned despite landmarks; map lost
  = **landmarks overshadow path integration** (classic cue competition; Zhao & Warren 2015 in humans).
- **Publishable.** Either branch: a quantitative account of why natural text (named places; Jericho) shows no
  path benefit, and the first cue-competition measurement between landmark retrieval and path integration in a
  transformer (possibly new). Decides whether Talk the Walk (landmark-rich) is worth building.

### 2. Narrative time with compositional steps (fabula vs discourse) -- ~9 GPU-h, ~4.5 h wall
- **Task.** A 1D walk over 64 story days (reflecting ends), steps of 1-3 days, backward with p_back = 0.3
  (flashbacks); each scene states the day's event object ("..., she baked a [obj] ."); 0-3 filler sentences
  per scene decouple text distance from story time. Two renderings: LEXICAL (one token per signed magnitude,
  six nonce or fixed words) and COMPOSITIONAL ("two days later", "a day earlier", "three days before").
- **Arms.** CF 1L (MapWM r=4), CG 1L (window 4), RoPE 2L; x 2 renderings x 6 seeds = 36 runs.
- **Floor.** Constant; "same object as k scenes ago" for k=1-3; reversal-copy in scene steps; n-grams.
- **Predictions.** LEXICAL: CF and CG solve, RoPE 2L does not (H3: index needs ~3 layers). COMPOSITIONAL: CF
  fails by the 2.3 argument (a sign-only solution leaves |k|-errors), CG solves. Readout: CF's learned step for
  "later" vs "earlier" and the number words (parity-only structure predicted).
- **Publishable.** Path integration tracks a non-spatial cognitive dimension (narrative time; Bellmund et al.
  2018's "cognitive spaces") in words, and a provable limit of token-additive steps with its fix -- the step of a
  phrase must be a product. Small, clean, motivated by real English.

### 3. Frames: hypotheticals and retractions over spans -- ~15-35 GPU-h; pilot first (~4 GPU-h)
- **Task.** Text world + frames of m in {1..4} moves (drawn per frame; bin by m at eval) that must not move the
  walker: LEADING ("she imagined walking north, then east, and seeing a cat .") vs TRAILING ("she walked north
  and east and saw a cat -- no, that was a dream ."). Objects inside a frame are not scored.
- **Arms.** CF 1L, HSR 2L, RoPE 2L (+ CF 2L depth control); 2 conditions x 6 seeds; T=2048, 1800 ep as HSR.
- **Floor.** As the decoy tasks; frame/real must be unpredictable from the 4 tokens before and 3 after each
  direction word (gate as `gate_ctxstep3.py`).
- **Predictions.** CF fails in proportion to frame moves; HSR solves LEADING for every m by gate-inside (read:
  per-word phase change inside frames ~0); TRAILING solved at m=1 and degrading with m (cancel-at-close needs the
  span sum). A clean TRAILING success at m >= 2 would refute 2.2's account.
- **Publishable.** The first frame-scoped (not distance-scoped) test of context-dependent steps, with a
  mechanistic readout separating gate-inside from cancel-at-close (possibly new). Reviewers will want the
  Mamba-3-style gate at depth as an arm (`LIT_CONTEXT_STEPS.md` P1).

### 4. Abelianisation check: swaps of boxes -- ~3 GPU-h
- **Task.** Kim & Schuster-style boxes, generated: 5 boxes, 5 objects, "swap box A and box C" operations,
  queries "box C holds the [obj]"; T ~256.
- **Arms.** MapWM 1L/2L, RoPE 1L/2L, 4 seeds (16 runs).
- **Floor.** Initial content of the queried box; most-recent-mention rule.
- **Prediction.** MapWM - RoPE within the MDE at each depth; if the phase is used at all, swap-word steps sit at
  multiples of pi in some channel (a parity code). A finding would be the parity channel, not accuracy.
- **Publishable.** Only as a short boundary paragraph: the phase is an abelian channel, as theory says.
  Low information per GPU-h because the main outcome is near-guaranteed; run only bundled with 1 or 2.

### 5. Relational order: chain vs shuffled descriptions -- ~3 GPU-h + CPU audit
- **Task.** StepGame-style relations among unnamed entities ("the lamp is two steps north of the cat"),
  presented in CHAIN order (each sentence relates the next entity to the previous: a walk) or SHUFFLED; target:
  the relation of a queried pair, scored as retrieval at matched position.
- **Prediction.** Path 1L wins in CHAIN only; both fail SHUFFLED at 1 layer. Mostly true by construction (N);
  its value is the scope statement "path integrator, not map maker". CPU-only first: n-gram / constant / index
  floors on StepGame and bAbI 17/19 (check whether StepGame sentences are shuffled before using it).

Not proposed for GPU: Jericho (floor 0.90+ on 45/54 games from in-context edge recall; named rooms); bAbI 1-3
(resets, recency solves them); CLUTRR (kinship is a groupoid; only generation depth is abelian; ordering
unclear); CogEval (an LLM evaluation, not trainable at our scale); Othello-GPT-style world models (board updates
are resets with captures, not a group).

## 4. What would be publishable, overall
A short paper: "A path-integrated 'where' for language: four conditions." Evidence: text world (registered),
task 1 (landmarks), task 2 (time + product steps), the aside role-offset analysis, and the frame result if task 3
is run. Its honest headline is a scope result: path integration is a specialist module for the spatiotemporal,
relative, unnamed, in-order part of a situation model; natural corpora rarely isolate it, which is why LM-scale
tests (ours on enwik8/code/Bach; HGRN's negative for data-dependent phase on LM) see little.

## Sources (web, read for this note; abstracts unless stated)
- Kim & Schuster 2023, Entity Tracking in Language Models, ACL: https://aclanthology.org/2023.acl-long.213/
- Tang et al. 2026, Do Language Models Track Entities Across State Changes?: https://arxiv.org/abs/2605.30233
- Li, Guo & Andreas 2025, (How) Do Language Models Track State?, ICML: https://arxiv.org/abs/2503.02854
- Yang 2026, A Calibrated Test of Internal Action Maps: https://arxiv.org/abs/2608.13626 (LM hidden states support
  one-step transitions without global composition/closure; relevant to 1.2.2)
- Momennejad et al. 2023, CogEval: https://arxiv.org/abs/2309.15129
- Kim et al. 2024, Textualized Gridworld / cognitive-map CoT: https://arxiv.org/abs/2406.15275
- Shi, Zhang & Lipani 2022, StepGame: https://arxiv.org/abs/2204.08292
- Constantinescu, O'Reilly & Behrens 2016, Science 352:1464: https://www.science.org/doi/abs/10.1126/science.aaf0941
- Bellmund, Gardenfors, Moser & Doeller 2018, Science 362:eaat6766
- Park, Miller & Boorman 2021, Nat Neurosci: https://www.nature.com/articles/s41593-021-00916-3 ; Park et al. 2020,
  Neuron, "Map making"
- Zhao & Warren 2015, Psychol Sci, landmark vs path integration: https://doi.org/10.1177/0956797615574952
- Local corpus: `cope`, `srope`, `mamba3`, `path`, `grazzi`, `hgrn` (`papers/txt/`); `docs/lit/LIT_CONTEXT_STEPS.md`
  (bAbI, StepGame, Flip-Flop, contextual counting already covered there).
- Not re-read here (cited from general knowledge, check before citing): Zwaan & Radvansky 1998 event-indexing
  model; Liu et al. 2023 "Transformers learn shortcuts to automata"; Merrill et al. 2024 "The illusion of state";
  Li et al. 2023 Othello-GPT; Cote et al. 2018 TextWorld; Hausknecht et al. 2020 Jericho.

## Appendix: the post hoc probe (run from /home/prashr with PYTHONPATH=/home/prashr, CPU)

```python
# aside_attn: attention from revisit queries to the first-visit object vs an aside noun at the same cell
import numpy as np, torch, torch.nn.functional as F
from mapformer.environment_textworld import TextWorld
from mapformer.tw_normstep_readouts import load, wrap, HELDOUT
from mapformer.model_textstep import step_of
from mapformer.model import _apply_rope
R = "/home/prashr/mapformer/runs/tw_normstep/p0"
for arm, s in [("MapWM", 16), ("MapWM", 10), ("MapWM", 13), ("NormStep", 12), ("NormStep", 14), ("DirOnly", 10)]:
    m, _ = load(f"{R}/{arm}_s{s}/{arm}.pt")
    te = TextWorld(size=64, seed=HELDOUT); np.random.seed(0); objs = set(te.obj_ids); dot = te.idx["."]
    A_obj, A_as = [], []
    with torch.no_grad():
        for _ in range(15):
            tok, obs, rev = te.generate_trajectory(1024); locs = te.visited_locations; t = tok.numpy()
            x = m.token_emb(tok[None]); d = step_of(m, tok[None]); cos_a, sin_a = m.path_integrator(d)
            L0 = m.layers[0]; h = L0.norm1(x); B, T, _ = h.shape
            Q = _apply_rope(L0.q_proj(h).view(B, T, L0.n_heads, L0.d_head).transpose(1, 2), cos_a, sin_a)
            K = _apply_rope(L0.k_proj(h).view(B, T, L0.n_heads, L0.d_head).transpose(1, 2), cos_a, sin_a)
            sc = (Q @ K.transpose(-1, -2)) / L0.d_head ** 0.5
            sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
            P = F.softmax(sc, -1)[0].sum(0).numpy() / L0.n_heads
            slots = np.nonzero(obs.numpy())[0][:len(locs)]; first = {}; aside_at = {}
            for k, i in enumerate(slots):
                L = tuple(locs[k])
                if L in first and rev[i] and L in aside_at:
                    for a in aside_at[L]:
                        if t[a] != t[i]:
                            A_obj.append(P[i - 1, first[L]]); A_as.append(P[i - 1, a])
                first.setdefault(L, i)
                if i + 2 < len(t) and t[i + 1] == dot:
                    j = i + 2
                    while j < len(t) - 1 and t[j] != dot and j < i + 8: j += 1
                    if t[j] == dot and t[j - 1] in objs and j - 1 > i + 2:
                        aside_at.setdefault(L, []).append(j - 1)
    print(arm, s, len(A_obj), np.mean(A_obj), np.mean(A_as))
# aside_loop (same loading): per aside, mean_channels |wrap(theta[noun] - theta[obj slot])| with
# theta = cumsum(step_of) * omega; MapWM s10-17 0.013-0.048 rad, NormStep s10-17 0.016-0.078 rad, 0 channels > 1 rad;
# direction-word step 0.093-0.228 rad. Net displacement over the aside equals the noun offset in this grammar.
```
