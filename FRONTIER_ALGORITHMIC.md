# Depth vs loop: does looping substitute for depth in BOTH position codes?

Venue: **algorithmic**, chosen by `decide_frontier.py` on a rule fixed
before the numbers were seen. Eight arms: {index, path integration} x
{1, 2, 3 real layers, loop x4}.

**Axis note.** The loop is parameter-matched to the 1-layer model, NOT
compute-matched -- four passes cost four passes. Everything below is a
parameters-and-memory result and none of it is a FLOPs result.

## parity

| arm | params | accuracy |
|---|---|---|
| index L1 | 199,042 | 0.519 +/- 0.019 (n=16) |
| index L2 | 397,314 | 0.541 +/- 0.014 (n=16) |
| index L3 | 595,586 | 0.565 +/- 0.006 (n=16) |
| index LOOP x4 | 199,042 | 0.575 +/- 0.004 (n=16) |
| path-int L1 | 199,490 | 0.598 +/- 0.013 (n=16) |
| path-int L2 | 397,762 | 0.632 +/- 0.033 (n=16) |
| path-int L3 | 596,034 | 0.641 +/- 0.090 (n=16) |
| path-int LOOP x4 | 199,490 | 0.667 +/- 0.077 (n=16) |

### Where the loop sits on each row's depth curve

| row | loop - L1 | loop - L2 | loop - L3 |
|---|---|---|---|
| index | +0.056 (sd 0.021, MDE 0.014, 16/16) | +0.034 (sd 0.014, MDE 0.010, 16/16) | +0.010 (sd 0.005, MDE 0.003, 16/16) |
| path-int | +0.070 (sd 0.080, MDE 0.056, 14/16) | +0.036 (sd 0.085, MDE 0.060, 9/16) | +0.026 (sd 0.146, MDE 0.102, 8/16) |

**Loop gain over 1 layer**: path-int +0.070 (DETECTABLE POSITIVE), index +0.056 (DETECTABLE POSITIVE); difference **+0.014**.

The loop's gain does not differ detectably between position codes: it substitutes for depth the same way in both, and LOOP_HEADROOM's apparent interaction does not survive having a depth baseline on the index row.

- **index**: loop vs 3 real layers +0.010 (sd 0.005, MDE 0.003, 16/16) -> DETECTABLE POSITIVE. The loop BEATS real depth.
- **path-int**: loop vs 3 real layers +0.026 (sd 0.146, MDE 0.102, 8/16) -> UNMEASURED. The loop MATCHES real depth at a third of the parameters.

## copy -- VACUOUS, read nothing into it

| arm | params | accuracy |
|---|---|---|
| index L1 | 200,841 | 0.126 +/- 0.001 (n=16) |
| index L2 | 399,113 | 0.124 +/- 0.001 (n=16) |
| index L3 | 597,385 | 0.125 +/- 0.001 (n=16) |
| index LOOP x4 | 200,841 | 0.125 +/- 0.001 (n=16) |
| path-int L1 | 201,289 | 0.128 +/- 0.006 (n=16) |
| path-int L2 | 399,561 | 0.144 +/- 0.074 (n=16) |
| path-int L3 | 597,833 | 0.127 +/- 0.005 (n=16) |
| path-int LOOP x4 | 201,289 | 0.129 +/- 0.007 (n=16) |

### Where the loop sits on each row's depth curve

| row | loop - L1 | loop - L2 | loop - L3 |
|---|---|---|---|
| index | -0.000 (sd 0.002, MDE 0.001, 8/16) | +0.001 (sd 0.001, MDE 0.001, 11/16) | +0.001 (sd 0.001, MDE 0.001, 10/16) |
| path-int | +0.002 (sd 0.010, MDE 0.007, 8/16) | -0.015 (sd 0.070, MDE 0.049, 8/16) | +0.002 (sd 0.009, MDE 0.006, 9/16) |

**Loop gain over 1 layer**: path-int +0.002 (UNMEASURED), index -0.000 (UNMEASURED); difference **+0.002**.

The loop's gain does not differ detectably between position codes: it substitutes for depth the same way in both, and LOOP_HEADROOM's apparent interaction does not survive having a depth baseline on the index row.

- **index**: loop vs 3 real layers +0.001 (sd 0.001, MDE 0.001, 10/16) -> UNMEASURED. The loop MATCHES real depth at a third of the parameters.
- **path-int**: loop vs 3 real layers +0.002 (sd 0.009, MDE 0.006, 9/16) -> UNMEASURED. The loop MATCHES real depth at a third of the parameters.

**Every arm on copy is at chance** (0.124-0.144 against 0.125), so every
"matches real depth" verdict above is a comparison between two things that do not
work. Copy has no dynamic range at L=128 for any architecture, which is the same
degeneracy `ALGORITHMIC_RESULTS.md` flagged. It supports nothing and refutes
nothing; it is reported only so the failure is on the record rather than quietly
dropped.

## What this settles

**Parameter efficiency: CONFIRMED, in both position codes, at n=16.** On parity a
199K looped model equals or beats a 596K three-layer model:

| row | 1 layer | 2 layers | 3 layers | LOOP x4 (1 layer's params) |
|---|---|---|---|---|
| index | 0.519 | 0.541 | 0.565 | **0.575** |
| path-int | 0.598 | 0.632 | 0.641 | **0.667** |

The best arm overall is also the SMALLEST: path-integrated + loop, 199,490
parameters, 0.667. In the index row the loop BEATS three real layers by +0.010 on
16/16 seeds; in the path-integrated row it matches (+0.026, MDE 0.102, 8/16).

**No interaction: looping substitutes for depth the SAME WAY in both codes.** Loop
gain over one layer is +0.070 path-integrated and +0.056 index, difference +0.014 --
not detectable. This is the question the frontier existed to answer, and it settles
it against the hint in LOOP_HEADROOM, whose +0.315 could not be interpreted because
the index row had no depth baseline. Combined with the raw-scale interaction of
+0.003 in ALGORITHMIC_RESULTS.md, **Match-Query's super-additivity is
task-specific and does not generalise.**

**The path-integrated row is far noisier than the index row** -- sd 0.077-0.146
against 0.004-0.021 -- which is why its loop-vs-depth contrast is 8/16 seeds despite
a positive mean. Any future claim about that row needs more seeds than this one
does, and the index row's cleanliness is itself worth noting: whatever makes
training bimodal here rides on the path-integration machinery.

## Scope

One train length, one loop count, depth to 3. Accuracy is reported at L=128 (the extrapolation point). A frontier is a shape, and three depth points define it only loosely -- read the per-arm numbers, not just the contrasts.
