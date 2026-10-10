# Which words move a path-integrated language model? TinyStories pilot -- results (2026-10-09)

PILOT, not pre-registered; readouts declared before training in `TINYSTORIES_PILOT.md`. Runs `runs/tinystories/p0`
(MapWM = Vanilla rank 4, and index RoPE; seeds 0-2; 6 layers, d 256, 8 heads, context 512 words, 12k iterations,
~197M tokens). Readouts `docs/audits/2026-10-09/tinystories_probe.py` / `_out.txt`, `runs/tinystories/PROBE.json`.
Seeds 1-2 were first killed by GPU out-of-memory (6 runs on one GPU) and rerun 2 at a time with the same recipe;
seed 0 is from the first launch (logs of the killed runs in `runs/tinystories/p0/oom_2026-10-09/`).

## Prediction is the same with and without path integration
| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| MapWM val (nats/token) | 1.5070 | 1.5057 | 1.5033 | 1.5053 |
| RoPE val | 1.5041 | 1.5093 | 1.5015 | 1.5050 |
| MapWM - RoPE | +0.0029 | -0.0036 | +0.0018 | +0.0004 |

Floors: trigram 2.935, bigram 3.603, unigram 5.574. Both arms are far past the word-statistics floors, and do not differ
(n = 3; no test claimed).

## What moves the phase: a discourse clock, not a map
- **Mostly a clock.** 79-87% of the frequency-weighted step is shared by every token (clock share 0.86 / 0.87 / 0.79;
  on code 0.75-0.92).
- **Which tokens tick it faster** (step relative to the mean step; the same on all three seeds): end of story `<eos>`
  5.4-6.8x; sentence ends `.` `!` `?` 2.2-3.3x; quotes and newline 1.6-2.5x; `mr` / `mrs` ~2x; story openers `once`,
  `upon`; clause links `and`, `but`, `because`, `until`, `suddenly`; speech and reaction verbs `replied`, `agreed`,
  `nodded`, `asked`, `smiled`. Punctuation carries 32-35% of all step, function words 29-32%, newline / `<eos>` 6-7%.
- **Motion words do not move it.** Motion verbs (go / went / came / ran / walked / climbed ...) carry 1% of the step;
  spatial words (up, down, in, out, inside, back, home ...) 0%.
- **Nothing cancels.** After removing the shared component, opposite pairs score 1.5-2.0 (0 = cancel, 2 = same), at
  or above the median of random pairs (1.82-1.89); the lowest, came/went on two seeds (1.50, 1.63), is not below the random 5th percentile
  (1.10-1.14) on any seed.
- **Story boundaries are jumps.** `<eos>` has the largest step on every seed (5-7x a mean token): the phase leaps
  between stories.

## Is the content dependence used? Yes, a little
Validation loss change when the step table is replaced at test time (the swap itself changes the loss by < 1e-8):

| substitution | s0 | s1 | s2 |
|---|---|---|---|
| every token steps the average step (a clock at the learned mean rate) | +0.081 | +0.093 | +0.067 |
| only content words (not punctuation, newline / `<eos>`, function words) step the average | +0.030 | +0.025 | +0.018 |
| no steps at all (no position) | +2.82 | +2.86 | +2.88 |

The model needs position (removing it costs 2.8 nats per token) and uses the variable rate: making it a uniform clock
costs 0.07-0.09 nats per token, about 20x the MapWM - RoPE difference. Most of that is punctuation, line and story ends;
content words account for a quarter to a third.

## What it means
- On natural narrative a path-integrated LM learns a **discourse clock**: its position advances by sentences, speech
  turns, clauses and stories, not by words and not by the places the story describes. Count-of-events time, not a map.
  Index RoPE, which counts tokens, does as well.
- The motion vocabulary of the stories is not used as actions. In the scripted text world, direction words were the
  only tokens that predicted where an object would be; in TinyStories, where things are is rarely needed to predict the
  next word, so nothing pushes the step toward a map. The step goes where prediction pays: segment boundaries.
- The `<eos>` jump is a phase discontinuity between unrelated contexts, structurally like global remapping between
  environments; speculative, not tested.

## Caveats
- Pilot: n = 3, one model size (8.9M parameters), one budget (0.37 epoch), one corpus; context 512 words, so most
  windows span several stories.
- `mr` / `mrs` get large steps; the tokenizer splits "mr." into `mr` `.`, so this may be compensation for a full stop
  that does not end a sentence. Unchecked.
- The class lists (function words, motion, spatial) were fixed before training; a different list would move the
  shares a little, not the ordering.
- The step is context-free (one step per word type); a context-reading step could behave differently.
