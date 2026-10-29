# Decisive control: is it the lap-counting demand, or just distribution shift?

Phase 2 on the SAME lap circuit with the REWARD REMOVED. Identical data
distribution, token statistics and episode format; the episodes differ at
**exactly one token** (verified). Only the lap-counting demand is deleted.

| arm | MQ before | MQ after | delta | per seed |
|---|---|---|---|---|
| lap **with** reward | 0.377 | 0.083 | **-0.293** | -0.317, -0.308, -0.255 |
| lap **without** reward | 0.377 | 0.085 | **-0.291** | -0.292, -0.321, -0.260 |
| control (same task) | 0.377 | 0.378 | +0.002 | +0.026, -0.015, -0.007 |

## Verdict: the interesting claim is dead

Removing the lap-counting demand changes the degradation by 0.002. The collapse
of Match-Query is **catastrophic forgetting under distribution shift**, and has
nothing to do with laps, position codes, or a conflict between event-specific
representations and cognitive maps.

800 steps on any different distribution wipes this 600K-parameter model's map.
That is a fact about scale and sequential training, not about cognitive maps, and
it should not be written up as one.

## Also retracted: the theta-drift metrics are not diagnostic

| model | obs/act \|Delta\| | theta drift |
|---|---|---|
| Match-Query-trained (HAS a working map) | 0.252 | 4.37 |
| lap-trained from scratch | 0.188 | 3.86 |

The lap model is MORE faithful on both measures than the model with a working
map. So neither metric separates "has a cognitive map" from "does not", and the
earlier claim that the lap-trained model "abandoned path integration" was an
inference from an uncalibrated number.

## What still stands

- The lap task is built, gated, and shortcut-free (`LAP_GATES.md`).
- MapFormer solves it: exact **1.000** at the trained K=4 (random floor 0.250),
  **0.000** at K=6 OOD.
- The mechanism is unknown.
