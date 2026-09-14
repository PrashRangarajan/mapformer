# Revisit accuracy by recurrence interval: why index PoPE trails index RoPE at training length

Eval-only probe on the PAPER2X2 checkpoints (`probe_revisit_2x2.py`); no training. Written before
the probe was run.

In PAPER2X2 the two index arms cross over: RoPE 0.805 vs PoPE-Flat 0.679 at T=128, and 0.374 vs
0.607 at T=1024.

**Hypothesis (not established).** RoPE's content-dependent phase lets a query shift its positional
kernel to a relative offset chosen by content. An index model needs exactly that to answer revisits
from the action tokens in its context. PoPE deletes the term: content sets only non-negative
magnitudes, and the per-(head, frequency) phase offsets are constants. So PoPE cannot build those
lookups. It should be lower at training length, and have nothing to misfire beyond it.

PoPE also differs from RoPE in three other ways: softplus magnitudes, 64 rather than 32
frequencies, and learned offsets. This probe **cannot attribute** the gap to content phase. It
only tests whether the gap has the shape the hypothesis predicts.

**Predictions:**
- **P1.** At T=128, RoPE - PoPE-Flat is concentrated at short recurrence intervals. It must be:
  - detectably positive in at least one bucket <= 16 steps;
  - largest in a bucket <= 16;
  - not detectably positive in any bucket >= 33.
- **Q2 (descriptive, no direction registered).** At T=1024, does RoPE keep its short-interval
  accuracy for revisits late in the sequence (step >= 128)?
  - If yes, the collapse beyond training length comes from long intervals.
  - If no, it comes from position beyond the training range.

**Design.**
- Held-out map (env-seed 10000).
- Every arm at seed s sees the same trajectories, so contrasts pair model seed and evaluation data.
- 256 trajectories per checkpoint at T=128 and 64 at T=1024.
- Buckets are steps since the cell was last visited.

