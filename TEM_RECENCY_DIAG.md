# TEM recency diagnostics

Written 2026-09-14, before training. Follows `TEM_RECENCY_PILOT.md`: the installed rewind scored 1.000,
but TEM from scratch stayed at chance (4/4). These pilots locate where from-scratch learning fails. Same
recipe as the pilot; 2 seeds each; `TEMRecency_Query` (non-committing query) throughout.

| id | arm | change | reading rule |
|---|---|---|---|
| D1 | `TEMRecency_Query --k-fixed 1` | every query asks for the most recent symbol | at chance again: the adaptation does not start learning at all (optimisation, not recency) |
| D2 | `TEMRecency_Query_CounterInstalled` | counter installed and frozen; per-query rewinds learned | learns: the counter was the blocker. Stays near the pilot: the per-query rewind search is |
| D3 | `TEMRecency_Query_Init1` | transitions initialised at scale 1.0, not 0.05 | learns: the near-identity start was the blocker |

These are diagnostics about TEM as adapted here, not comparisons with MapFormer.

## Results (2 seeds each; same recipe; chance 0.0625, uniform loss 2.77)

| id | arm | seed | acc T=1024 | acc T=2048 | final loss | first epoch with loss < 0.5 | acc by k: 1-16 / 17-32 / 33-48 / 49-64 |
|---|---|---|---|---|---|---|---|
| D1 | k fixed at 1, from scratch | 0 | **1.000** | **1.000** | 0.003 | 13 | (k = 1 only) |
| D1 | | 1 | **1.000** | **0.999** | 0.003 | 14 | |
| D2 | counter installed, rewinds learned | 0 | 0.320 | 0.266 | 2.440 | never | **0.89** / 0.08 / 0.15 / 0.21 |
| D2 | | 1 | 0.335 | 0.198 | 2.519 | never | **0.80** / 0.20 / 0.13 / 0.10 |
| D3 | init scale 1.0, from scratch | 0 | 0.112 | 0.083 | 2.729 | never | 0.09 / 0.16 / 0.12 / 0.10 |
| D3 | | 1 | 0.088 | 0.097 | 2.735 | never | 0.07 / 0.14 / 0.09 / 0.05 |

## Verdicts by the rules written first

- **D1: the adaptation does learn.** With every query at k=1, TEM learns the counter and retrieval from
  scratch within about 13 epochs, and holds at twice the length. The pilot's failure is not a broken
  model or optimiser.
- **D2: the per-query rewind search is a blocker.** With a perfect counter installed, TEM learns
  rewinds only for the smallest offsets: 0.80-0.89 for k up to 16, 0.08-0.21 beyond. Training loss
  is still falling slowly at epoch 300 (2.44 and 2.52), so the budget is not exhausted.
- **D3: a larger initial scale does not help.** Still at chance.

## Reading (pilot; not a registered result)

- **TEM's failure on full recency looks like MapEM's:** rewinds are found for short offsets and not
  long ones. MapEM's own SEARCH anatomy showed rewinds found per token for about half the k, and a k
  curriculum helped only small k.
- **TEM's per-query transform is not a rank-4 bottleneck.** It is a full orthogonal matrix per query
  token, able to rotate each frequency block independently. So with the counter given and that
  freedom available, long rewinds were still not found within this budget. That weakens the
  explanation I offered earlier: that MapEM struggles because its rewind passes through a rank-4
  bottleneck, while MapWM gets a free dial per block. Free per-block dials on the position side were
  not enough here.
- **What still separates TEM and MapEM from MapWM:**
  - where the offset lives: the query's structural code, rather than content phase in the key-query
    comparison;
  - how retrieval is gated: TEM queries by structure alone.

  Neither has been isolated.
- **Caveats:**
  - n=2 per arm, one recipe.
  - D2 was still improving, and a longer budget could change the long-k picture.
  - TEM here has more parameters than MapEM.
  - The MapEM and MapWM references come from another batch.

## Candidate next steps

1. D2 at 3-4x the budget, to check whether long rewinds arrive late or never.
2. D2 with block-diagonal query transforms (one angle per block exactly), to test the "free per-block
   dial" idea directly.
3. A same-batch MapWM with the same counter installed, to check whether content phase finds long
   offsets quickly under identical conditions.
