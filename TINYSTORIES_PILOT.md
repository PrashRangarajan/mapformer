# Which words move a path-integrated language model? TinyStories, word level -- pilot design (2026-10-09, before any run)

Exploratory pilot, NOT pre-registered: no branch is decided here and no verdict will be claimed from it. The readouts
below are declared before training so they cannot be chosen after seeing the models.

## Question
On navigation, and on navigation told in scripted English, MapFormer's learned step `Delta = omega * W_out W_in emb(token)`
moves on the action words, barely moves on the rest, and opposite moves cancel. On real code it became a clock (75-92%
of every byte's step shared; `docs/audits/2026-10-05/code_step_table_out.txt`). On natural narrative text, with real
motion words ("went", "came back", "ran into the house"), which tokens does a path-integrated LM choose to move on,
does anything cancel, and does the content-dependent part of the step matter for prediction?

## Data (`tinystories_data.py`; gate `docs/audits/2026-10-09/tinystories_gate.py` / `_out.txt`)
TinyStories V2 (GPT-4 split), downloaded 2026-10-09 from huggingface.co/datasets/roneneldan/TinyStories. Lowercase words,
punctuation as separate tokens, '\n' and '<eos>' tokens; vocabulary 8192 (top train tokens), <unk> 0.09%. Train 535M
tokens (2.72M stories), val 5.4M (the official valid file, 27,630 stories). Round trip PASS. Floors on val (nats/token):
unigram 5.574, interpolated bigram 3.603, **trigram 2.935**.

## Arms and recipe (`train_tinystories.py`, `run_tinystories.sh`)
MapWM (`Vanilla`, rank 4 shared, as on code) and index RoPE (`RoPE`, canonical base 10000), seeds 0-2. 6 layers, d 256,
8 heads, context 512 tokens, batch 32, 12,000 iterations (~197M tokens, ~0.37 epoch), AdamW lr 1e-3 wd 0.05, 500 warmup
+ cosine to 0.1x, clip 1.0, grid_size = 512. Every arm at a seed sees the same training batches. ~8.9M parameters.

## Declared readouts (post hoc script, CPU, on the best checkpoint by val; no verdicts)
1. **Validation loss** (nats/token, 200 fixed batches) per arm beside the trigram floor; MapWM - RoPE with all three
   seeds shown (n = 3: no test is claimed; rule 5).
2. **Step table** (MapWM; from the weights, as for code): per token `step = omega * W_out W_in emb`. Clock share
   (|| frequency-weighted mean step || / frequency-weighted mean || step ||); top 30 tokens by step norm among tokens with
   frequency > 1e-5; share of the frequency-weighted step carried by token classes: punctuation, '\n' / '<eos>', quote,
   function words (a fixed list of the 100 most frequent closed-class words), motion verbs (a fixed list: go/goes/went/
   gone/going, come/came/comes/coming, walk(ed), run/ran, jump(ed), climb(ed), fly/flew, swim/swam, move(d), return(ed),
   leave/left, arrive(d), enter(ed)), spatial words (up, down, in, out, into, outside, inside, back, away, home, there,
   here, over, under), and the rest.
3. **Opposition** after removing the common (clock) component, `|| d(a) + d(b) || / mean(|| d(a) ||, || d(b) ||)` (0 =
   cancel, 2 = same): up/down, in/out, inside/outside, came/went, come/go, open/close, opened/closed, start/stop,
   forward/back, left/right, push/pull, give/take, on/off; against random pairs of tokens with frequency > 1e-4 (median,
   5th percentile).
4. **Causal**: val loss with every token's step replaced by the frequency-weighted mean step (a pure clock at the learned
   average rate) minus intact; the same with steps zeroed (no position at all); and with only content words (tokens
   outside punctuation, newline / eos and the function-word list) replaced by the mean step.
5. **Story boundaries**: whether '<eos>' carries an outlier step (a phase reset or jump between stories).

## What it can and cannot show
- Can: whether natural narrative makes the step content-dependent (low clock share, spatial or motion words on top,
  cancelling pairs), and whether that content dependence is used (readout 4).
- Cannot: claim path integration helps language (n = 3, one size, one budget, a pilot); separate "motion words move
  because they are motion" from "they are frequent sentence-level words" without a matched control.
- Expected (stated now): mostly a clock, punctuation and newline on top, as on code.

## Cost
6 runs concurrently on one GPU (GPU 0; another user holds GPU 1): ~1.5-2 h. Probes CPU, minutes.
