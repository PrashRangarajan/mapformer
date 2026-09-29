# Navigation told in words -- results (2026-09-28)

Pre-registration `TEXTWORLD_PREREG.md` (+ Amendment 1, committed before any result was read); runs
`runs/textworld` (24 runs, one batch); registered output `TEXTWORLD_ANALYSIS.txt`
(`analyze_textworld.py`), step table `TEXTWORLD_PROBE.json`, declared secondaries
`TEXTWORLD_SECONDARY.txt` / `.json` (`analyze_textworld_secondary.py`). The torus walk rendered as
English (58 words, three synonyms per direction, movement-free verbs, fillers and asides), T = 1024
words (~132 steps), held-out map, 8 seeds, 900 epochs.

## Verdict A: PATH WINS IN WORDS

| model | T=1024 accuracy | SOLVED | T=2048 (no verdict) |
|---|---|---|---|
| **path integration, 1 layer (r=4)** | **0.969 +/- 0.054** | 7/8 (1 descending) | 0.956 |
| index RoPE, 1 layer | 0.505 +/- 0.001 | 0/8 | 0.512 |
| index RoPE, 2 layers | 0.772 +/- 0.097 | 0/8 | 0.656 |
| floors on the eval set: best constant / reversal-copy rule | 0.505 / 0.597 | | |

Path - RoPE 1L **+0.464** (exact permutation p 0.0002; SOLVED 7/8 vs 0/8, Fisher p 0.0014). Path -
RoPE 2L +0.197 (p 0.0009). RoPE 1L sits exactly at the constant floor and below the reversal-copy
rule on all 8 seeds; RoPE 2L clears both but is STALLED on 8/8. r(final loss, acc) = -0.989. Scope:
900 epochs; the RoPE arms were still creeping, so "cannot" is budget-scoped.

## Verdict B: no registered branch -- but the map is there on 8/8

The registered criterion (move ratio < 0.2, opposition < 0.3, synonym cosine > 0.9) holds on 4/8
seeds (FINDS THE ACTION WORDS needed 6/8; FINDS SYNONYMS ONLY needed <= 2/8). Synonym cosine is
1.000 on 8/8. The declared secondaries (Amendment 1) show what the other four are:

| seed | raw opposition | **minus common component** | common / north | cos(common, verb step) | ablate non-direction steps | ablate direction steps |
|---|---|---|---|---|---|---|
| s1, s5, s6, s7 (criterion met) | 0.080-0.120 | **0.003-0.012** | 0.04-0.06 | 0.45-0.89 | 0.998->0.893, 1.000->0.898, 1.000->0.950, 0.882->0.586 | -> 0.505 / 0.505 / 0.505 / 0.498 |
| s0, s2, s3, s4 (criterion missed) | 1.43-1.77 | **0.002-0.007** | 0.48-1.14 | **+0.999 to +1.000** | 0.881->0.191, 0.999->0.665, 0.995->0.673, 0.999->0.539 | -> 0.388 / 0.482 / 0.463 / 0.445 |

- **Every seed learned a cancelling map of the direction words.** With the common component of the
  four direction steps removed, north + south and west + east cancel to 0.002-0.012 on all 8 seeds
  (0.002-0.010 omega-scaled). Zeroing the direction words' steps drops every seed to the constant
  floor (0.39-0.51): the movement words carry the position.
- **Two solutions, split 4/4.** Half the seeds put essentially all the phase on the direction words
  (common component 4-6% of a direction step). The other half add a **per-step clock**: a shared
  vector, identical in direction to the verbs' step (cosine 1.000), carried by every verb and every
  direction word -- a step counter layered on the map. The two are equivalent in what they encode
  about position (a verb and a direction word occur once per clause), which is why the registered
  opposition and move-ratio readouts, which are not invariant to that gauge, split them. The clock
  seeds depend more on the non-direction words (zeroing them costs 0.33-0.69 vs 0.05-0.30).
- Synonym agreement (1.000 on 8/8) is real but weak evidence: every interchangeable class collapses
  (verbs 0.96-1.00), while different directions sit at |cos| 0.36-0.61.
- Criterion x SOLVED is unrelated (Fisher p 1.00): the clock seeds solve as well as the pure-map
  seeds.

## What it means
With the actions buried in prose, a one-layer path-integrated model learns which words move it, that
opposite directions cancel and that synonyms are one move, from the prediction task alone, and it
beats an index model with twice the layers. The one thing the registered readout did not anticipate
is that half the models also count steps; Amendment 1 predicted that from the pilot before the batch
was read.

## Caveats
- Scripted grammar, 58 words, context-free steps: a direction word never appears outside a movement
  clause. A word used both ways ("she thought about the north") would move the phase by construction;
  that needs a context-dependent step, untested.
- The registered B branch did not fire; the 8/8 map reading rests on declared secondaries.
- 900-epoch budget; RoPE arms stalled, not converged. One path seed (s0) still descending.
- n=8, one recipe, r=4 shared, 1 layer for path.
