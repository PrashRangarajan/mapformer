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

## Results

### T=128, revisits at steps < 128 (inside the training length)

| arm | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65-128 |
|---|---|---|---|---|---|---|---|
| RoPE | 0.995 | 0.939 | 0.936 | 0.670 | 0.470 | 0.484 | 0.480 |
| PoPE-Flat | 0.960 | 0.901 | 0.663 | 0.488 | 0.419 | 0.447 | 0.445 |
| Vanilla | 0.985 | 0.984 | 0.985 | 0.981 | 0.945 | 0.891 | 0.836 |
| MapPoPE-Flat | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.994 | 0.977 |
| Vanilla_r4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 |
| MapPoPE_r4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.987 |
| *blank rate (floor)* | *0.507* | *0.500* | *0.503* | *0.511* | *0.496* | *0.511* | *0.509* |
| *events (all seeds)* | 11779 | 9937 | 15080 | 15117 | 5442 | 2682 | 1236 |
| *share of events* | 19.2% | 16.2% | 24.6% | 24.7% | 8.9% | 4.4% | 2.0% |

RoPE - PoPE-Flat, paired by seed (same trajectories):

| bucket | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| 1-2 | +0.035 | 0.048 | 0.048 | 8/8 | unmeasured |
| 3-4 | +0.038 | 0.111 | 0.109 | 5/8 | unmeasured |
| 5-8 | +0.273 | 0.143 | 0.142 | 8/8 | DETECTABLE |
| 9-16 | +0.181 | 0.049 | 0.049 | 8/8 | DETECTABLE |
| 17-32 | +0.051 | 0.015 | 0.015 | 8/8 | DETECTABLE |
| 33-64 | +0.037 | 0.037 | 0.036 | 8/8 | DETECTABLE |
| 65-128 | +0.036 | 0.036 | 0.036 | 7/8 | unmeasured |

### T=1024, revisits at steps < 128 (inside the training length)

| arm | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65-128 |
|---|---|---|---|---|---|---|---|
| RoPE | 0.994 | 0.937 | 0.933 | 0.666 | 0.487 | 0.489 | 0.496 |
| PoPE-Flat | 0.958 | 0.894 | 0.662 | 0.497 | 0.434 | 0.467 | 0.463 |
| Vanilla | 0.989 | 0.986 | 0.984 | 0.982 | 0.968 | 0.915 | 0.850 |
| MapPoPE-Flat | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.991 |
| Vanilla_r4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.994 |
| MapPoPE_r4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.988 |
| *blank rate (floor)* | *0.497* | *0.499* | *0.499* | *0.510* | *0.508* | *0.521* | *0.508* |
| *events (all seeds)* | 2905 | 2499 | 3648 | 3708 | 1393 | 729 | 295 |
| *share of events* | 19.1% | 16.5% | 24.0% | 24.4% | 9.2% | 4.8% | 1.9% |

RoPE - PoPE-Flat, paired by seed (same trajectories):

| bucket | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| 1-2 | +0.036 | 0.051 | 0.050 | 7/8 | unmeasured |
| 3-4 | +0.043 | 0.118 | 0.116 | 5/8 | unmeasured |
| 5-8 | +0.271 | 0.137 | 0.136 | 8/8 | DETECTABLE |
| 9-16 | +0.169 | 0.047 | 0.047 | 8/8 | DETECTABLE |
| 17-32 | +0.054 | 0.017 | 0.017 | 8/8 | DETECTABLE |
| 33-64 | +0.022 | 0.046 | 0.045 | 5/8 | unmeasured |
| 65-128 | +0.034 | 0.038 | 0.038 | 5/8 | unmeasured |

### T=1024, revisits at steps >= 128 (beyond it)

| arm | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65-128 | 129-256 | 257+ |
|---|---|---|---|---|---|---|---|---|---|
| RoPE | 0.334 | 0.327 | 0.351 | 0.340 | 0.340 | 0.334 | 0.331 | 0.320 | 0.308 |
| PoPE-Flat | 0.949 | 0.863 | 0.604 | 0.477 | 0.445 | 0.450 | 0.450 | 0.441 | 0.439 |
| Vanilla | 0.858 | 0.851 | 0.837 | 0.820 | 0.769 | 0.709 | 0.654 | 0.594 | 0.458 |
| MapPoPE-Flat | 0.997 | 0.996 | 0.997 | 0.995 | 0.989 | 0.976 | 0.944 | 0.774 | 0.458 |
| Vanilla_r4 | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 | 0.997 | 0.992 | 0.919 | 0.417 |
| MapPoPE_r4 | 0.998 | 0.998 | 0.998 | 0.998 | 0.997 | 0.996 | 0.974 | 0.852 | 0.559 |
| *blank rate (floor)* | *0.513* | *0.508* | *0.503* | *0.512* | *0.514* | *0.510* | *0.507* | *0.504* | *0.499* |
| *events (all seeds)* | 21028 | 17998 | 27994 | 29399 | 11394 | 7531 | 7488 | 8256 | 18205 |
| *share of events* | 14.1% | 12.1% | 18.8% | 19.7% | 7.6% | 5.0% | 5.0% | 5.5% | 12.2% |

RoPE - PoPE-Flat, paired by seed (same trajectories):

| bucket | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|
| 1-2 | -0.615 | 0.154 | 0.152 | 0/8 | DETECTABLE |
| 3-4 | -0.535 | 0.149 | 0.148 | 0/8 | DETECTABLE |
| 5-8 | -0.252 | 0.147 | 0.146 | 0/8 | DETECTABLE |
| 9-16 | -0.137 | 0.109 | 0.108 | 0/8 | DETECTABLE |
| 17-32 | -0.105 | 0.137 | 0.136 | 1/8 | unmeasured |
| 33-64 | -0.116 | 0.147 | 0.146 | 1/8 | unmeasured |
| 65-128 | -0.120 | 0.155 | 0.154 | 1/8 | unmeasured |
| 129-256 | -0.122 | 0.166 | 0.164 | 1/8 | unmeasured |
| 257+ | -0.131 | 0.171 | 0.170 | 1/8 | unmeasured |


## Verdicts

- **Sanity checks.**
  - The probe's revisit events match the environment's revisit mask exactly: 814/814 at T=128 and
    10,288/10,288 at T=1024 on a test batch.
  - Seed 0 RoPE at T=1024 scores 0.221-0.245 here, against 0.226 in `_PAPER2X2_RAW.json`.
  - Early revisits inside T=1024 sequences reproduce the T=128 numbers (RoPE 0.994 / 0.937 / 0.933 /
    0.666 against 0.995 / 0.939 / 0.936 / 0.670), as causal attention requires.
- **P1: NOT MET as registered, on its third clause.** The first two clauses hold. RoPE - PoPE-Flat is
  detectably positive in buckets <= 16 (5-8: +0.273, MDE 0.142, 8/8; 9-16: +0.181, MDE 0.049, 8/8), and
  its largest value is in the 5-8 bucket. The third clause fails: at 33-64 the contrast is +0.037 against
  an MDE of 0.036 (8/8), i.e. detectably positive by 0.001. At 17-32 it is +0.051 (MDE 0.015). The gap
  is concentrated at 5-16 steps but is not confined there.
- **Q2 (descriptive): the collapse beyond training length is about POSITION, not interval.**
  - For revisits at step >= 128 in T=1024 sequences, RoPE falls to 0.308-0.351 at EVERY interval,
    including 1-2 steps (0.334). That is below the ~0.51 blank floor.
  - The same model scores 0.994 on 1-2-step revisits early in the same sequences.
  - PoPE-Flat keeps its short-range accuracy late (0.949 / 0.863 at 1-2 / 3-4).
  - Late RoPE - PoPE-Flat is -0.615 at 1-2 steps (MDE 0.152, 0/8).

## Reading (post hoc; not registered)

- **Both index models are short-range path integrators working inside attention.**
  - RoPE is near perfect out to 8 steps (0.995 / 0.939 / 0.936), partial at 9-16 (0.670), and below the
    blank floor from 17 steps on (0.470-0.484).
  - PoPE-Flat's range is shorter: 0.960 / 0.901 at 1-4 steps, 0.663 at 5-8, 0.488 at 9-16.
  - Weighted by event share, most of RoPE's +0.126 training-length lead comes from the 5-8 and 9-16
    buckets, which hold about half of all revisit events. That weighting is arithmetic on the table
    above, not a tested contrast.
  - This fits the hypothesis that RoPE can reach content-chosen offsets further back. It does not
    isolate content phase from PoPE's other three differences.
- **The length half of the original hypothesis is wrong as stated.** It said RoPE's lookups "misfire at
  larger offsets". In fact RoPE fails at every interval, the shortest included, once the query is past
  the training length.
  - Candidate account (untested): beyond training length the context contains keys at relative offsets
    never seen in training, and under RoPE some of those score highly enough to pull attention away from
    the correct recent key.
  - PoPE, with non-negative magnitudes and fixed phase offsets, is robust to this. That robustness is
    what the PoPE paper claims.
  - An eval-only test is available: restrict RoPE's attention to a trailing window no longer than the
    training length. If late short-interval accuracy recovers, unseen-offset distractors are the cause.
- **Path-integrated arms fail the other way.** Late in the sequence they hold short and medium intervals
  (r=4: 0.997 up to 64 steps) and degrade only at intervals longer than any seen in training (129-256
  steps: 0.852-0.919; 257+: 0.417-0.559). Their length failure is about recurrence interval, not
  position. The exception is r=2 Vanilla, which degrades late at all intervals (0.858 at 1-2).
