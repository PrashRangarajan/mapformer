# Why a non-negative rotary step does not learn the map: the start it imposes -- pre-registration (2026-10-14, before the batch)

## Background
WRAP_EXISTENCE (registered): on the ring an exact non-negative solution exists and is held under training (8/8), but the
non-negative class never finds it from scratch (0/8 vs signed 8/8). Why not?

## Exploratory toy (`toy_wrap.py`; CPU; NOT registered -- reported with its n, used to form the hypothesis)
A 64-channel retrieval model with no transformer and no observation tokens: two move angles per channel (rank-4 latents,
signed or |.|, MapWM's frequency ladder), and softmax retrieval of the earlier moves at the same ring cell by
sum_c w_c cos(theta_c(t) - theta_c(t')). Default init matched to SignedAO_r4's initial angle scale (|angle| mean ~0.5 rad).
Results so far (`runs/toy_wrap/{explore,grid,swap}`; seeds 0-3 and 10-17; 2-3 seeds per cell; SOLVED = retrieval acc >= 0.99):
| start (move angles) | signed | non-negative (|.|) |
|---|---|---|
| default (zero-mean, small; opposite moves' signs independent) | 1-2/3 (0.82-1.00) | 0/3 + 0/3 more (0.17-0.70) |
| the seed's default angles, balanced gauge ("same") | 0.79-1.00 | -- |
| |default angles| (the start |.| imposes), balanced gauge ("absmag") | **0.02-0.69 (fails)** | 0.05-0.58 (fails) |
| small, same direction, U(0.05, 1.5) ("low") | 0/3 | 0/2 |
| large, U(1.7, 4.5), sum > pi ("high") | 2/2 | **2/2 (solves, wrapping)** |
The toy reproduces the sign asymmetry without a transformer, and in it the whole cost of the constraint goes through the
start: a signed model started where |.| starts fails like it, and a |.| model started with large angles solves (its
high-weight channels cancel mod 2 pi). The balanced gauge alone changes nothing (signed "same" solves as signed default).

## Hypothesis (to test on the real model)
Opposite moves must end with angles that cancel mod 2 pi. Under the signed parameterisation with a zero-mean init, roughly half
the channels start with opposite moves rotating in opposite directions, and short out-and-back returns pull each channel
toward a(-1) = -a(+1) while its difference -- the position signal -- survives. |.| makes every channel start with both moves
rotating the same way by small angles; from there the pull goes toward a(+1) = a(-1) = 0, which erases the position signal, and
the cancelling configuration that exists for non-negative angles (sum = 2 pi) lies beyond a sum of pi. So the sign constraint
costs learning through the START it imposes, not through the solutions it allows.

## Registered test (`train_wrap_mech.py`; ring 32, p_plus 0.5, RING_DIAL recipe, 300 epochs; observation-step-free models;
seeds 800-807; n = 8 per arm)
| arm | model | start |
|---|---|---|
| SO | SignedAO_r4 | default |
| AO | AbsAO_r4 | default (= |SO's default angles|) |
| SO_same | SignedAO_r4 | its default angles, balanced gauge (gauge control) |
| SO_absmag | SignedAO_r4 | |default angles|, balanced gauge: the start |.| imposes |
| AO_absmag | AbsAO_r4 | |default angles|, balanced gauge (gauge control; same angles as AO) |
| AO_high | AbsAO_r4 | a(+1), a(-1) iid U(1.7, 4.5) rad per channel, balanced gauge |
Balanced gauge = WRAP_EXISTENCE Amendment 1's construction at init (move latents c e1, c e2; omega' = max(|omega|, 1)); every
other weight is the seed's default init, identical across arms. Exactness of the starts is checked (angle error ~2e-7).
Accuracy: `train_cancel.evaluate`, fresh stream, 400 trajectories, T = 128 (eval mode); SOLVED = classify_run (final 5% of
training loss < 0.05). Analysis `analyze_wrap_mech.py`; tests: Fisher on SOLVED, unpaired permutation on accuracy (as RING_DIAL).
- **M1 (does the |.| start hurt a SIGNED model?) SO_absmag vs SO_same.** HURTS: Fisher p < .05 with SO_same ahead, OR perm
  p < .05 with mean SO_absmag - SO_same <= -0.05. NO EFFECT: SO_absmag SOLVED >= SO_same - 1 AND mean >= SO_same - 0.01.
  UNMEASURED otherwise.
- **M2 (does a large start rescue the NON-NEGATIVE model?) AO_high vs AO_absmag.** RESCUES: Fisher p < .05 with AO_high ahead,
  OR perm p < .05 with mean AO_high - AO_absmag >= +0.05. NO EFFECT: AO_high SOLVED <= AO_absmag + 1 AND mean <= AO_absmag
  + 0.01. UNMEASURED otherwise.
- **M3 (gauge controls; a difference qualifies M1 / M2): SO_same vs SO and AO_absmag vs AO**, each labelled DIFFERS (Fisher
  p < .05 or perm p < .05 with |d| >= 0.05) or NO DIFFERENCE DETECTED.
- **M4 (replication inside this batch): AO vs SO** as WRAP_EXISTENCE's E3 (SIGNED LEARNS BETTER / NO SIGN COST / UNMEASURED,
  gated on SO SOLVED >= 4).
Predictions, stated now: M1 HURTS, M2 RESCUES, M3 NO DIFFERENCE DETECTED, M4 SIGNED LEARNS BETTER. If M1 HURTS and M2 RESCUES,
the learnability cost of the non-negative rotary step is a property of the start it imposes; if neither, the toy's account
does not carry to the transformer.
Secondaries (declared, no verdict): wrap share at the end for every arm (AO_high: does it solve by wrapping?); the per-channel
initial angle difference |a(+1) - a(-1)| vs final cancellation (signed arms); accuracy at 4T; loss AUC per arm; the toy grid
at 8 seeds per cell when it finishes (exploratory).

## Process and cost
No pilot beyond the toy (the arms reuse WRAP_EXISTENCE's classes and recipe; the start construction is checked on CPU). An
independent code check (rule 29) before launch. 48 ring runs, ~10-25 min each depending on load; run alongside GENERALITY on
both GPUs: ~1.5-2 h of the shared budget.
