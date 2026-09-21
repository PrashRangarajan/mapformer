# CROSS_PREREG -- does the decay envelope inherit the position variable's metric?

## The claim under test

`DYCK_DECAY_RESULTS.md` found that one ALiBi-style envelope helps the path-integrated row on Dyck-2
(+0.048 at distance 33+) and drives the index row to the no-stack floor beyond distance 8. The
proposed reading: **a decay envelope is a proximity prior in whatever metric the position variable
defines** -- token distance for an index code, learned state (here, stack depth) for a path-integrated
one. That contrast varied the position mechanism AND the decay metric together, so it cannot
distinguish the reading from the simpler "path integration is what helps".

## The crossing

|  | decay over token distance | decay over learned state distance |
|---|---|---|
| **index phase** | `PoPE_decay` (run: 0.879 F1, 0.511 at d 33+) | **`PoPE_decay_statemetric`** (new) |
| **path-integrated phase** | **`MapPoPE_decay_idxmetric`** (new) | `MapPoPE_decay` (run: 0.956, 0.778) |

The new arms keep their phase mechanism and swap only the distance the envelope decays over. For the
index arm a learned increment map is added that feeds the METRIC ONLY -- the attention phase stays
`t * theta_c`. 8 seeds each, `DYCK_PREREG.md` recipe verbatim.

## Registered verdicts, at L128 D12, closer accuracy at distance >= 9 (chance 0.500)

- **C1 (the metric reading)** predicts the outcome follows the COLUMN: `MapPoPE_decay_idxmetric`
  should lose long-range retrieval, falling toward the 0.503 that `PoPE_decay` shows, while
  `PoPE_decay_statemetric` should retain it, rising toward `MapPoPE_decay`'s 0.906.
- **C2 (the position reading)** predicts the outcome follows the ROW: the path-integrated arm stays
  high and the index arm stays at chance, whatever metric the envelope uses.
- These are mutually exclusive on the two new cells, which is the point of the design.
- **Falsification of C1** is any result where swapping the metric does not move the arm toward the
  other row -- including a null, which would say the metric is irrelevant and the position mechanism
  carries everything.
- Also reported: F1 for comparability with the existing table, and the learned lambda per head (if the
  index arm drives its envelope to zero, it has declined the prior rather than been hurt by it).

**Stakes**: C1 is the only positive theoretical claim to come out of this line that has not yet been
tested. If it fails, the line is entirely eliminative.

## Amendment 1 (2026-09-20): the same crossing on Bach, where the two metrics nearly coincide

The Dyck crossing fired: swapping the envelope's metric moves the INDEX row by +0.281 (8/8,
detectable) and the path-integrated row not at all. Dyck is the favourable case for the claim,
because its accumulator is a map encoding stack depth, so the learned state and token distance mean
genuinely different things (they correlate 0.755 vs 0.278 with depth and token distance).

Bach is the unfavourable case: its accumulator is a clock (alpha = 1.00), so a learned state is close
to a token count and the two metrics nearly agree. Same 2x2, 5 seeds, the 512-crop setup that
produced the existing decay results. Two cells exist (`PoPE_decay` 0.6262, `MapPoPE_decay` 0.6223 at
2-4x); two are new.

- **X1 (registered prediction)**: swapping the metric should do LITTLE on either row, because on a
  clock the two metrics nearly coincide -- unlike Dyck. Specifically, the index-row swap should be far
  smaller than Dyck's +0.281.
- **X2**: if instead the index row gains substantially here too, then the learned state is doing
  something beyond counting even when its growth exponent says it is a clock, and "the metric is what
  matters" is a stronger claim than the coincidence argument allows.
- **X3 (falsifier for the coincidence argument)**: if swapping the metric HURTS either row detectably
  on Bach, the metric claim does not transfer and the Dyck result is task-specific.
- Reported alongside: the correlation between the learned state distance and token distance on Bach,
  which is the quantity the coincidence argument rests on and has never been measured there.
