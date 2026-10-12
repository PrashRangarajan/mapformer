# RING_DIAL -- independent code audit (rule 29), 2026-10-12

Auditor blind to results: `runs/ring_dial_pilot/` and any pilot amendment were not read. Read-only; CPU tests
only, scripts and outputs in the session scratchpad (`audit_rd/t1.py` init/scale, `t2.py` existence,
`t3.py` wrap fractions, tiny trainer runs, synthetic-layout analysis run). Code at HEAD e208693 (+ 589ec2d, 9f91be0).

## Verified correct (with evidence)
- Arms built as named (t1): `Signed_r4` -> `MapFormerWM_signed_r4` (mode signed), `Abs_r4` -> `MapFormerWM_abs_r4`
  (mode abs), `RoPE` -> `MapFormerWM_RoPE`; 1 layer each; params 204,243 / 204,243 / 203,411.
- Signed and Abs share initial weights at a seed: identical state_dict keys, max |diff| 0.0; the torch RNG after
  construction is identical (next draw 0.47232 both), so dropout streams match too. At init the forward differs
  (max logit diff 0.081) only through `|.|` (`model_sign.py:246-251`).
- Same training stream across arms at a seed: `train.py:109` base_seed = `torch.initial_seed()` = `--seed`, independent
  of model construction; `data_parallel.py:54-74` pickles the env object, so `p_plus` reaches the workers. Batch seeds
  `seed*1_000_003 + i` (seeds 30-35, i < 29,400) never equal the eval seed 10**6; pilot seed 130 is disjoint too.
- Same eval stream across arms at a p: `train_cancel.evaluate` reseeds numpy with 10**6 (`train_ring_dial.py:51`).
- p_plus passed and recorded: tiny CPU runs at p 1.0 record `p_plus: 1.0`; dir name `p1.0` matches the analysis
  f-string; `float("0.9") == 0.9`.
- Driver: seeds 30-35 x p {0.5, 0.9, 1.0} x 3 arms, seed outer; `DRV_GPUS` default 0 honoured by `drv_pick`;
  `flock` lock + argv-exact `running()` guard (trailing-space match, no prefix collisions); done marker only after
  `drv_require` of all 108 artifacts, the second md5 guard, and the analysis outputs.
- Analysis direction: `perm2_p(a, b)` tests mean(b) - mean(a); `contrast(signed, mono)` gives Abs - Signed as
  registered. Smoke passes; a synthetic 54-run layout runs end to end (REPO/R monkeypatched to scratch).
- `rescore_hook` would hook all three arms (`mapformer.model.WMTransformerLayer`, dropout p 0.1).

## BUG

**B1. Three registered secondaries are not implemented** (prereg l.34-35 vs `analyze_ring_dial.py:152-182`).
(a) The dropout-scale re-score with its FLAG line: nothing calls `rescore_hook`, and the trainer saves only eval-mode
accuracy. (b) The opposition score: not computed anywhere. (c) Fisher on SOLVED for the Abs - Signed contrasts: Fisher
is printed only for DI. (a) matters most (see M4).
Fix: in the analysis (or a small `eval_ring_dial.py` run by the driver before the analysis, md5-guarded), reload each
`<arm>.pt`, re-run `evaluate` at T 128 with `rescore_hook.install("auto")`, assert `STATS["hooked"] == 1` per model
and no skipped classes, print DM/DD/DC re-scored with `FLAG` where a label changes; add Fisher(SOLVED) beside DM/DD/DC;
implement the opposition score in its wrap-aware form (M1).

**B2. The DC equivalence CI is a grid artefact at ceiling.** `perm2_ci` (grid=None) walks outward from d0 in 0.001
steps, so s = 0 is never evaluated; with tied 1.000 values the shift test jumps to p = 0.0022 as soon as s separates the
ties. Result: printed CIs exclude 0 while perm p = 1.0, e.g. Signed 6 x 1.0 vs Abs 5 x 1.0 + 0.95 -> "d -0.0083 (perm p
1.0000, 95% CI [-0.0243, -0.0003])"; Abs one run at 0.97 -> CI [-0.0150, -0.0000]; at 0.999 -> [-0.00017, -0.00017].
DC reads MATCHES/UNMEASURED off this CI. With `grid=(-0.3, 0.3)` the same inputs give [-0.025, 0.000] etc. (contains 0,
consistent with p). Fix: call `perm2_ci(..., grid=(-0.5, 0.5))` (report `clipped`), for DC and in all printed CIs.

## DESIGN major

**M1. |W_out W_in x| is not monotone for a rotary phase: Abs can represent the map exactly.** The score reads each
channel's angle mod 2 pi. Setting, per channel, Delta(+1) = a >= 0 and Delta(-1) = 2 pi k / omega - a >= 0 gives
theta = omega a (n_plus - n_minus) mod 2 pi -- the signed code -- for ANY a and any environment period (not only this
ring). Constructed in the real modules (t2): w_in projects onto two directions orthogonal to all 17 observation
embeddings, w_out solved per channel; Abs deltas >= 0 (min 5e-11), observation-token |Delta| <= 1.7e-6; over a 512-step
p 0.5 walk the Abs cos/sin equal the Signed code's to 6.3e-4 (fp32), and 2,000 same-cell revisit pairs have circular
angle mismatch <= 4.6e-4. On the top channel (omega = 2 pi) any increment mod 2 pi is reachable with Delta in [0, 1),
and init |Delta| is 0.18-0.39 (t1), so the modular solution sits at init scale. Consequences:
- Abs is not Howard's Case II. Case II is a real-valued (Laplace, non-periodic) accumulator; a rotary step is
  circle-valued, and "monotone" in the unwrapped angle is not monotone in what the score sees. Any Abs deficit here
  (and on the torus) is search/optimisation, budget-scoped, not representation. The prereg's prediction ("monotone
  costs accuracy at 0.5 and 0.9") and the DIAL label "HOWARD CASE II PATTERN" assert the Case II mechanism without
  this; `model_sign.py` docstring ("a monotone code cannot represent net displacement") and `LIT_CLOCK_MAP.md` l.226-227
  ("a monotone rotary channel cannot cancel inside the score") are false as stated.
- The registered opposition score ||D(+1) + D(-1)|| / mean ||D|| is not invariant to the 2 pi / omega gauge (rule 8):
  the exact wrap solution above reads **1.71** ("no cancellation"). Half-turn channels (omega Delta = pi, Delta(+) =
  Delta(-)) cancel exactly yet read 2.0. SIGN_MATCHED's probe (Abs 1.92) has the same blind spot.
Fix (before launch): restate the hypothesis as "a non-negative rotary step must find a full-turn (modular) solution
to cancel; we predict it fails to within the budget where steps cancel" and drop the Case II identification from the
verdict label (e.g. "NON-NEGATIVE STEP FAILS ON THE MAP"). Register a wrap-aware opposition: per channel
phi_c = wrap(omega_c (Delta_c(+1) + Delta_c(-1))) to (-pi, pi], weighted by |omega_c Delta_c(+1)| (exclude channels
whose step is ~0 mod 2 pi), and report it for both arms; a solved Abs run with small wrap-aware opposition is the
modular solution. Optional (rule 14, cheap): warm-start Abs from the constructed solution, frozen and trainable.
Flag for the caller: SIGN_MATCHED's mechanism sentence ("the constrained arms cannot [use sign]") needs the same
correction; its accuracy result is unaffected.

**M2. DC is true by construction, so DIAL reduces to DM.** At p_plus 1.0 the action token is constant; |c| and c
differ per channel by a reflection R(-phi) = P R(phi) P, P = diag(1, -1), which W_q / W_k absorb exactly; the only
other difference is the sign of observation-token Deltas, whose ideal value is 0. Howard's Cases II and III coincide
by definition when v >= 0. DC "MATCHES" therefore carries no evidence for Howard (a "WORSE" would be an optimiser
artefact), and CANCEL had path and index at 1.000 8/8 there. Fix: label DC a manipulation check; if the question is
"does the cost track how much the task needs a map", register a graded readout, e.g. DIAL holds iff DM fires negative
AND d(DD) is between d(DM) and 0 (or DD - DM tested), with DC as precondition.

**M3. n = 6 cannot detect a bimodal Abs outcome.** With Signed at ceiling and failing Abs runs at 0.80: DM fires only
if >= 5/6 Abs seeds fail (k = 4: p 0.061; k = 3: d -0.100, p 0.18 -> "NOT SHOWN"); with a noisy ceiling, >= 4/6.
Fisher on SOLVED needs <= 1/6 Abs solved (2/6: p 0.061). Rule 5: such cells are UNMEASURED, not "NOT SHOWN".
Fix: n = 8 (+18 runs, ~20-30 min at 6 parallel) or register Fisher(SOLVED) as a co-primary firing rule; print
UNMEASURED when the CI contains both 0 and -0.05.

**M4. The eval-mode dropout bias points at DM/DD firing.** One layer, so the house correction applies; eval mode
under-reports runs below ceiling (up to +0.17, CLAUDE.md) and leaves ceiling runs alone. Signed is expected at ceiling,
Abs below it: the bias inflates exactly the registered contrasts. The prereg makes the re-score a FLAG-only secondary
and does not say what a firing that vanishes under re-score means. Fix: with B1(a), register "a DM/DD firing that does
not fire under re-score is reported as an eval-scale artefact", or make the re-scored accuracy co-primary.

## DESIGN minor

- **m1. The dial also moves the wrap regime.** Wrap-only revisit fraction (no earlier visit at the same unwrapped
  coordinate), T 128: **0.11 / 0.74 / 1.00** at p 0.5 / 0.9 / 1.0 (t3; T 512: 0.12 / 0.79 / 1.00). At 0.9 most
  targets are lap revisits with occasional corrections, so "how much a map is needed" is not the only thing varying;
  equal across arms, but state it. The omega ladder is not a differential: `grid_size=32` puts omega_min = 2 pi / 32
  exactly at the ring fundamental for Delta = 1 (t1: 0.196350), the same relation as 2 pi / 64 on the 64-torus, and
  the omega_max = 2 pi channel makes modular cancellation reachable on both tasks (M1).
- **m2. Provenance.** Asserts cover arm, p_plus, seed, len(losses) only (`analyze_ring_dial.py:163`). Add epochs 300,
  n_steps 128, n_trials 400, size 32, lr 1e-3, batch 128, n_batches 98; eval.json carries no args or eval seed (write
  them). Print the floors (0.554 / 0.520 / 0.498) and all run classes (SOLVED / STALLED / DESCENDING), not only SOLVED
  (rules 1, 3); r(final loss, acc) per p (rule 2).
- **m3. md5 GUARD** (`run_ring_dial.sh:199-200`) omits `data_parallel.py` and `environment.py` (imported by train.py;
  run_cancel guarded data_parallel) and will need `rescore_hook.py` / any eval script added for B1. Amend before launch:
  the second md5 guard (l.217) fails the batch if the analysis is edited afterwards.
- **m4. Shared GPU budget with CLOCK_SIDE.** `drv_ntrain` counts every `mapformer.train_*` on cuda:0. Ring MAXPG 6
  vs clock_side MAXPG 4: while ring jobs run, clock_side cannot launch remaining runs or its eval step until the total
  drops below 4. Launch after clock_side's training finishes, or accept the delay.
- **m5.** A positive firing (Abs better, e.g. d +0.19, p 0.0022) prints "NOT SHOWN" for DM/DD; give it its own label.

## Cosmetic
- c1. Float boundary at |d| = 0.05: mean([0.95]*6) - 1.0 = -0.04999999999999993 does not fire (synthetic run printed
  "NOT SHOWN ... d -0.0500, p 0.0022"). Round d to 1e-9 before the threshold.
- c2. `model_sign.py` calls Signed_r4 "wiring check only"; it is a primary arm here (as in SIGN_MATCHED). Harmless.
