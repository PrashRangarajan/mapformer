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

## Amendment 1 (2026-10-14, after the independent check `docs/audits/2026-10-14/mechanism_audit.md`; before launch)
The check found the code correct (starts exact to <= 1e-6 rad; AO's default angles = |SO's| exactly; every other weight and
the RNG state identical across arms; stats calls in the right order). Changes:
- **A1 (toy table corrected; all toy runs, `runs/toy_wrap/summary.txt`, SOLVED = retrieval acc >= 0.99, exploratory):**
  | start | signed | non-negative |
  |---|---|---|
  | default | 6/12 (median 0.946) | 0/12 (0.516); x3 scale 0/11 (0.626) |
  | its default angles, balanced gauge ("same") | 3/8 (0.854) | -- |
  | |default angles| ("absmag") | **0/8 (0.406)** | 0/8 (0.385) |
  | small, same direction U(0.05, 1.5) | 0/11 | 0/11 |
  | large but sum < pi, U(0.9, 1.5) ("mid", new) | **0/8** | **0/8** |
  | large, sum > pi, U(1.7, 4.5) ("high") | 8/8 | 11/11 (+8/8 without the omega clamp) |
  | exact wrapped / iid U(0.2, 2 pi - 0.2) | -- | 8/8 / 9/9 |
  The toy's signed default solves only half its seeds; the asymmetry is clearer in accuracy. "High" beats the signed default,
  so it carries an advantage beyond undoing |.|. signed_high and abs_high are bit-identical in the toy (the constraint never
  binds from that start). Toy accuracies are cell retrieval, not comparable with the batch's (floor 0.554).
- **A2 (gates; reverse labels; effect sizes).** M1 UNREADABLE if SO_same SOLVED < 4; M2 / M2b UNREADABLE if SO SOLVED < 4.
  Every contrast has both directions (HURTS / HELPS, RESCUES / WORSE, ...) and "NO EFFECT DETECTED" (SOLVED within 1 and
  |d| <= 0.01; not equivalence, rule 5), with the permutation 95% CI of d and a paired sign-flip p (descriptive).
- **A3 (matched-start sign contrasts, new):** **M5 = AO_absmag vs SO_absmag** (identical angles, gauge and weights: does the
  constraint cost anything once the start is matched?) and a new arm **SO_high** (SignedAO_r4 from AO_high's exact angles)
  with **M6 = AO_high vs SO_high**. Prediction: NO EFFECT DETECTED for both (the cost goes through the start).
- **A4 (mobility control, new arm AO_mid):** a(+1), a(-1) iid U(0.9, 1.5): sums below pi, ~5x the default magnitude. The
  check measured that a large start moves angles 13-22x more per Adam step (via W_in and omega), so a RESCUE could be a
  hotter optimiser on the phases. **M2b = AO_high vs AO_mid**; the toy's mid fails for both signs (0/8, 0/8), which is why
  the arm is added. AO_mid's mobility is ~5x, not the full 13-22x, so the separation is partial; stated.
- **Reading, restated.** M1 HURTS + M2 RESCUES + M2b (sum above pi beats mid) + M5 / M6 NO EFFECT DETECTED: the non-negative
  step's learnability cost acts through the start it imposes (same-direction small rotations of opposite moves); report the
  start effects as d with CIs, not "the whole cost". If M5 or M6 shows a sign cost at a matched start, the cost is split
  between start and constraint, by those d. Mixed M1 / M2: report each as falls; no combined claim. A gauge-control DIFFERS
  (M3) means the balanced gauge itself changes learning and M1 / M2 apply to the gauge arms only.
- **Disclosed:** the gauge arms are effectively rank 2 (W_out[:, 2:] and W_in[2:] are 0 with exactly zero gradient); the
  initial opposite-sign share at default init varies 0.34-0.91 across seeds 800-807 (set by the cosine of the two move
  latents), now a secondary vs loss AUC; the initial max |W_out| per arm is printed (AO_high / SO_high up to 7.6, AO_mid up to
  2.5, gauge arms 0.7, default 0.5). Dropout-scale re-score added as a secondary. No GPU pilot of AO_high (the toy and the
  smoke stand in; AO_high's warm-up is visible in its loss curve). The batch shares the machine with GENERALITY; no module
  either batch imports will be edited while they run.
- Arms: 8 (SO, AO, SO_same, SO_absmag, AO_absmag, AO_high, SO_high, AO_mid) x seeds 800-807 = 64 ring runs; artifacts 128.
