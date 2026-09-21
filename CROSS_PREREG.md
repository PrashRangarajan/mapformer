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
