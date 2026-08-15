# Lap task -- pre-flight gates (CPU, no training)

K=4 laps. Exactly one of the K lap boundaries is a REWARD boundary.
Headline metric is **exact** = hit the right boundary AND no false alarms.

| variant | positional-shortcut exact | n-gram boundary acc (o1/o2/o4/o8) | random-boundary exact | oracle |
|---|---|---|---|---|
| loop_len FIXED | **1.000** | 0.250 / 0.344 / 0.746 / 0.750 | 0.250 | 1.000 |
| loop_len VARIABLE | **0.163** | 0.250 / 0.673 / 0.750 / 0.750 | 0.250 | 1.000 |

## Floors that matter

- always-say-no: exact **0.000**, but boundary accuracy **0.750** for free.
  This is why boundary accuracy alone must NOT be the headline.
- always-say-yes: hit 1.000 but false-alarm 1.000, exact 0.000.
- random boundary: exact **0.250**.

## Reading the positional gate

`positional-shortcut exact` fits the single best 'REWARD is at token index i'
rule on half the episodes and tests it on the other half. With a FIXED loop
length the reward always lands at index K*loop_len, so this rule is perfect and
the task measures nothing. Variable loop length is the operating point iff this
collapses.

## Probe result (Vanilla / MapWM-Flat, 60 epochs, seed 0)

| condition | hit | false alarm | **exact** |
|---|---|---|---|
| K=4 (trained) | 1.000 | 0.000 | **1.000** |
| K=6 (OOD lap count) | 0.277 | 0.281 | **0.000** |

The task is learnable and solved perfectly at the trained lap count. The
pre-registered prediction ("MapFormer cannot distinguish same-place-different-lap")
is therefore REFUTED as stated.

### RETRACTED 2026-08-09 -- the mechanism below is NOT supported

Two later measurements kill it:

**1. The theta/Delta metrics are not diagnostic.** A Match-Query-trained model --
one with a WORKING cognitive map -- scores obs/act |Delta| **0.252** and drift
**4.37**. The lap-trained model scores **0.188** and **3.86**, i.e. MORE faithful
on both. Reading 0.188 as "broke path integration" had no baseline behind it.

**2. The degradation is distribution shift, not lap counting.** Training a
Match-Query model on the lap circuit drops MQ by -0.293. Training it on the SAME
circuit with the REWARD REMOVED -- deleting the entire lap-counting demand, a
one-token change per episode -- drops it by **-0.291**. Identical. See
`LAP_TRANSFER.md` / `LAP_TRANSFER_NOREWARD.md`.

WHAT SURVIVES: the lap task is built and gated, and MapFormer solves it (exact
1.000 at K=4, 0.000 at K=6 OOD). HOW it solves it is unknown. The original text
follows for the record.

### (retracted) But the mechanism is the finding

`probe_lap_theta.py`, trained model:

| measurement | value |
|---|---|
| per-lap \|theta\| drift | **3.86 rad** |
| mean \|Delta\| on ACTION tokens | 2.065 |
| mean \|Delta\| on OBSERVATION tokens | **0.389** (19% of an action) |

The circuit has exactly zero net displacement (verified 0/1000 non-zero), so a
model path-integrating faithfully would have theta return every lap and the laps
would alias. Instead the model gives observation tokens real displacement, so
theta accumulates and becomes a lap counter.

**MapFormer solved the lap task by ABANDONING faithful path integration**, not by
counting laps in content attention. So the underlying claim survives in a sharper
form: the position code cannot represent same-place-different-lap *while
remaining a position code*; the model escapes only by ceasing to be one.

Consistent with the OOD failure: a drift-based counter fires at a learned
accumulated-theta threshold, which does not transfer to a different lap count
(exact 0.000 at K=6).

### Follow-up this implies

If theta no longer returns at the same place, the cognitive map should be
degraded. A lap-trained model should be measurably worse on Match-Query
(revisit matching). That is a direct, cheap test and has NOT been run.
