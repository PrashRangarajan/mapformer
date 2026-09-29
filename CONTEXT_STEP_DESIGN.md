# A context-dependent step -- design (2026-09-28; not yet pre-registered)

> **CORRECTED 2026-09-28 23:45 (from the pilot's training losses, before any eval was read).** The
> claim below that Selective RoPE's generator is additive is WRONG. It computes
> `w_t = sigmoid(W_g x_t) * sum_k a_k u_{t-k}`: the gate at t multiplies the conv's LAGGED terms, so
> a word's step can be carried one to three tokens late and switched off by a gate reading the words
> that FOLLOW it. It cannot use a cue before the word; it can use one after. The pilot's decoy grammar
> gave it exactly that: after a real move the direction word is always followed by "and", "there"
> or a filler, after a decoy always by "." (checked in the task code), and SR's loss fell to 0.009
> like the context gate's. The design is revised into a double dissociation (section "Revision")
> below; the pilot runs to completion and its SR gates are read to confirm the mechanism.

## The problem
MapFormer's step is a function of the token alone: `Delta_t = W_out W_in emb(x_t)` (`model.py:171`).
In the text world (`TEXTWORLD_RESULTS.md`) that sufficed because a direction word only ever appeared
inside a movement clause. Real language uses the same word both ways: "she walked north" moves you,
"she thought about the north" and "she did not go north" do not. A context-free step moves the phase
on every "north", so it must drift on every non-movement use, and nothing downstream can undo it
(the accumulated angle is wrong from that point on).

## Which kinds of context can fix it -- an argument, checked before building
What is needed is `Delta_t = 0` for "north" after "about the" or "not go", and `Delta_t = Delta(north)`
after "walked". That is a MULTIPLICATIVE dependence on context: the cue must switch the word's step
off.

- **Additive / linear context cannot.** If `Delta_t = sum_k A_k emb(x_{t-k})` (a linear causal conv
  before the bottleneck), then `Delta("about the north") = f(about) + f(the) + f(north)`. Cancelling it
  needs `f(about) + f(the) = -f(north)` for north, south, east and west at once, which is impossible
  for a direction-independent cue.
- **Selective RoPE's generator, at MapFormer's placement, is additive too.** Its conv is depthwise over
  the ALREADY-PROJECTED angle channels (`model_selective.py:43-64`), so it is a linear mix of per-token
  steps, and its gate `sigmoid(W_g x_t)` reads only the current token (`model_selective.py:87`). Neither
  can see "about the" when deciding the step for "north". (The published Selective RoPE computes the
  angle per layer from queries, so at depth > 1 it sees context through attention; that is the
  hidden-state arm below, not this generator.) Prediction: it fails like the context-free step.
- **A context gate can.** `Delta_t = g_t * W_out W_in emb(x_t)` with `g_t = sigmoid(v . phi(conv_k(emb))_t)`,
  a causal window of k tokens mixed across channels, a nonlinearity, then one gate per head. The cue
  switches the step off without changing what the step is when it is on, so the word-level step
  table stays readable (the text world's probe still applies, times the gate).
- **A hidden-state step can.** Two layers: layer 1 is an ordinary index-RoPE attention layer; the step
  is computed from its output, `Delta_t = W_out W_in h1_t`; layer 2 uses the path-integrated phase. This
  is the Mamba-3 / per-layer Selective RoPE placement. The most general, the least interpretable, and
  it costs a second layer, so it needs depth-matched controls.

Both working designs keep the cumsum, so the parallel scan is untouched: `g_t` and `h1_t` depend only on
tokens <= t.

## Task: the text world with decoys (a subclass of `TextWorld`, new file; nothing running is edited)
Between steps, with probability `p_decoy`, insert a sentence that uses a direction word WITHOUT
moving: "she thought about the north .", "a sign pointed left .", "she did not go east .". The
walker's position is unchanged, and the object clause is absent, so there is no target in a decoy.
Cues sit BEFORE the direction word, within 3 tokens, so a causal window of k=4 can see them. (A cue
after the word, e.g. "the north wind", is impossible for any causal step and is left out.)
Gate before any GPU (rule 11): floors on the eval set (constant, reversal-copy, word n-grams), revisit
rate, and a check that decoys do not change what an n-gram can predict.

## Arms (8 seeds each, one batch, trained and tested at T=1024, the text-world recipe)
| arm | step | layers | prediction with decoys |
|---|---|---|---|
| CF | context-free (current `Vanilla_r4`) | 1 | fails in proportion to the decoy rate |
| SR | Selective-RoPE generator at our placement (`MapFormerWM_SRoPEGen`) | 1 | fails like CF |
| **CG** | **context gate** (new) | 1 | recovers the decoy-free accuracy |
| **HS** | **hidden-state step** (new) | 2 | recovers it |
| CF2 | context-free | 2 | fails (depth control for HS) |
| RoPE1, RoPE2 | index | 1, 2 | floor / 0.77-level, as in the text world |

Plus a **no-decoy control** (`p_decoy = 0`) for CF, CG and HS: context must not cost anything when it is
not needed.

## Readouts (to be fixed in the pre-registration after a pilot)
- Primary: held-out accuracy at T=1024 with decoys, CG - CF and HS - CF2 (permutation and Fisher on
  SOLVED).
- Mechanism: the effective step at direction words by context class (movement / decoy / negation):
  `||Delta_decoy|| / ||Delta_move||`, which should go to 0 for CG and HS and stay at 1 for CF by
  construction; the gate's values by class for CG; opposition and synonym agreement on movement uses,
  after the common-component correction learned in the text world.
- Cost: CG and HS vs CF with no decoys (within the MDE = no cost).
- Dose (optional, second batch): `p_decoy` in {0.1, 0.3, 0.5}.

## Risks and design checks
- Existence (rule 14): the gate solution is plainly in the class (a 4-token window detecting "about
  the", "pointed", "not go"), but check it trains from the context-free init by starting the gate
  bias high (gate ~1, the CF model) rather than assuming.
- A decoy with no object clause changes tokens per step; keep T in words, and report steps per
  sequence per arm so no arm sees a shorter walk.
- Parameter counts: CG adds a conv (d x d_g x k) and a gate head; report them. HS doubles the layers;
  CF2 and RoPE2 are its controls.
- Nothing here may edit `train_variant.py`, `model.py` or `environment_textworld.py` while H1 runs (md5
  guards, rule 22): new classes go in `model_context_step.py`, the task in
  `environment_textworld_ctx.py`, and the trainer builds its own variant map.

## Cost
Main batch 7 arms x 8 seeds = 56 runs, plus the no-decoy control 3 x 8 = 24: 80 runs at ~25-35 min
each, ~8-9 h on both GPUs alone. A 2-seed pilot of CF / CG / HS first (~1 h).

## Revision (2026-09-28): two cue conditions, a double dissociation
The cue's POSITION decides which mechanism can use it:

| condition | example | what is matched | CF | CG (reads before) | SR (gates lagged steps by what follows) | HS (attention) |
|---|---|---|---|---|---|---|
| **leading cue** | "she did not go north and saw a lamp ." | the tokens AFTER the direction word are drawn exactly as after a real move | fail | solve | fail | solve |
| **trailing cue** | "walked north -- no , she stayed ." | the tokens BEFORE the direction word are drawn exactly as before a real move | fail | fail | solve | solve (cancel at the cue) |

Task: `environment_textworld_ctx2.py`, one `cue` argument. In the leading condition a decoy is a full
clause whose observation is the CURRENT (unchanged) cell, not scored. In the trailing condition the
decoy starts exactly like a movement clause and is retracted right after the direction word. Gate:
the class (move / decoy) must be unpredictable from the 3 tokens after the direction word (leading)
or the 4 tokens before it (trailing), measured with a frequency table on held-out data.

## Second revision (2026-09-29, from pilot 2): cue DISTANCE, not cue side
Pilot 2 (`CTXSTEP_PILOT2.md`) refuted the lead/trail dissociation: the context gate used trailing cues
and the Selective-RoPE generator used leading ones, each through a nonlinear interaction inside its
4-token window (the gates of neighbouring tokens read the direction word, or the direction word's
per-channel gate multiplies the cue's lagged terms). The distinction that should hold is the window:
CG and SR can only use a cue within k-1 = 3 tokens of the direction word; the hidden-state step reads
context through attention. Next task: insert movement-free padding between cue and direction word
("she did not, after a long pause and some thought, go north"), distance d in {1-3, 6-10}, both sides.
Predictions: CF fails at every d; CG and SR solve at d <= 3 and fail at d >= 6; HS solves at every d.
A window of 4 is a design choice, so the claim will be "finite-window steps fail past their window",
tested at one window size.
