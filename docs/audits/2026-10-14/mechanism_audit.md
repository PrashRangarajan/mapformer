# MECHANISM: independent code check (rule 29), 2026-10-14, before launch

Scope: `MECHANISM_PREREG.md`, `train_wrap_mech.py`, `analyze_wrap_mech.py`, `run_wrap_mech.sh` (HEAD 80bce18), the toy
`toy_wrap.py` and `runs/toy_wrap/{explore,grid,swap}` (read-only; the grid had finished, 80/80 JSONs). Context:
`model_wrap_exist.py`, `model_sign.py`, `model.py`, `train.py`, `train_cancel.py`, `environment_cancel.py`, `stats_core.py`,
`lib_driver.sh`, WRAP_EXISTENCE prereg / results / audit. CPU only. Check scripts: session scratchpad `mech/c1.py`
(starts, RNG, weight identity), `c2.py` (init scale real vs toy), `c3.py` (one-step angle mobility). Smoke: 6 arms x seeds
800-801, `--epochs 2 --n-batches 2 --data-workers 1 --device cpu --n-trials 20`, then `analyze_wrap_mech --seeds 800-801
--no-json`. No repo file edited except this one.

Verdict: the code does what the prereg says. Starts are exact, the arms share every other weight, RNG use is identical, and
the stats calls are in the right order. No BLOCKER. The design has four MAJOR issues. Fix them in an amendment before
launch. Two cost nothing (a gate and an extra registered contrast on arms already in the batch). Two need runs or rewording.

## Verified correct
- **Starts.** `set_move_angles` is exact. Max wrap-aware angle error: 1.2e-7 to 9.5e-7 rad (s800, s801, s803).
  - `ang0` is the raw signed step x omega in both classes, because `w_out` is called directly and bypasses the `|.|`.
  - AO's default angles equal |SO's default angles| exactly (difference 0.0; omega > 0 at init).
  - SO_absmag and AO_absmag have identical angles (0.0). AO_absmag equals AO (1e-7). SO_same equals SO (5e-7).
- **Shared weights.** Every state-dict key except `w_in`, `w_out` and `omega` is bit-equal to SO's at the seed.
- **RNG.** The torch and numpy global RNG states after construction are identical across all six arms. AO_high draws from
  its own `RandomState(seed + 7919)`. Dropout masks and the data stream are therefore the same sequence in every arm.
  - Eval uses one stream for all arms and seeds (np seed 10^6, map seed 10000).
- **Weight scale.** Default: |W_out| max 0.50, |W_in| max 0.088.
  - Gauge arms: |W_out| max 0.63-0.73, |W_in| max 0.017-0.029.
  - AO_high: |W_out| max 3.2-7.6 (mean 0.79-1.81), step mean 2.1-2.2 vs AO_absmag's 0.09-0.23.
- **Stats calls.** `perm2_p(r, a)` tests mean(x) - mean(ref), the same sign as `d`. `fisher_solved` is two-sided, and the
  direction is checked separately (`kr > ka` / `ka > kr`). Thresholds and directions of M1-M4 match the prereg text. M4
  copies E3 with the A5 gate.
- **Secondary code.** The initial-difference vs final-cancellation block computes what its label says: median split of
  |wrap(a+ - a-)| at init, then the cancel share (< 0.1 rad) among channels with |wrap(a+)| > 0.05 at the end. The import
  inside the loop is harmless.
- **Driver.**
  - Knobs are assigned after sourcing.
  - Lock, least-loaded picker, md5 guard (start and pre-analysis), and the running/done skips all work.
  - 96 = 48 x 2 artifacts.
  - The marker is set only after both outputs exist.
  - GUARD covers every module the run path executes (environment_cancel/nd, model, model_sign, model_wrap_exist,
    lie_groups, train, data_parallel, train_cancel, stats_core).
- **Smoke.** All 12 runs and the analysis complete, and every printed branch executes.

## Findings

**1. MAJOR: M1 and M2 read "NO EFFECT" vacuously at the floor. The smoke printed exactly that.**
- Smoke, all arms at 0.5059 and 0/2: `M1: NO EFFECT`, `M2: NO EFFECT`. This is WRAP_EXISTENCE audit finding 5 again.
  The prereg gated M4 (A5) but not M1/M2.
- M1 NO EFFECT is the conclusion "the |.| start does not hurt a signed model". It is meaningless if SO_same itself does
  not solve, for example if the balanced gauge broke the signed model.
- Fix:
  - M1 is UNREADABLE if SO_same SOLVED < 4.
  - M2 NO EFFECT requires AO_high's training to be sane. Proposal: its accuracy above the floor 0.554 + 0.1, or report M2
    NO EFFECT only with the per-seed loss curves showing no warm-up blow-up (see finding 3: AO_high's angles are 13-22x
    more mobile).

**2. MAJOR: M1 HURTS + M2 RESCUES does not establish the stated conclusion ("the cost goes through the start, not
through the solutions it allows").** Both contrasts are within one sign class. Neither compares the classes at a matched
start, yet that is the claim that the constraint adds nothing once the start is equated.
- The batch already contains the needed pair, SO_absmag vs AO_absmag (identical angles, gauge and weights), but registers
  no contrast on it. If SO_absmag escapes partly (the signed model can cross zero; in the 4-step smoke 46-50 of 128 raw
  steps already went negative), the cost is split between start and constraint. The design must be able to say so.
- Toy evidence that the counterpart pair matters: signed_high and abs_high are BIT-IDENTICAL on all 8 grid seeds (angle
  difference 0.0, same loss history). The minimum signed angle in the signed runs is 1.43-1.73 rad, so from the high start
  the constraint never binds and the "signed" model finds a non-negative solution.
- Fix:
  - (a) Register M5: AO_absmag vs SO_absmag, with labels as M4 (prediction: no sign cost at the matched start). Free.
  - (b) Add an SO_high arm (SignedAO_r4, the same U(1.7, 4.5) angles). AO_high vs SO_high tests whether the constraint
    costs anything from a high start (8 runs). Alternatively, log the min raw step of AO_high per epoch and report whether
    the `|.|` ever binds.
  - (c) Reword the conclusion. With n = 8, "no difference" is not equivalence (rule 5). Say "a start effect of size d [CI]"
    and report `perm2_ci` for every contrast. Do not say "the whole cost".

**3. MAJOR: M2 confounds the start's location with the angles' mobility (rule 15).**
Under Adam an update of fixed size per entry moves an angle in proportion to the other factors of omega' x W_out x latent,
so a 6-14x larger start angle is a 6-14x larger effective learning rate on the angles.
- Measured (`c3.py`, s800, one AdamW step at lr 1e-3, one parameter group at a time), mean |d angle| for AO_absmag vs
  AO_high:

  | parameter group | AO_absmag | AO_high | ratio |
  |---|---|---|---|
  | omega | 9.4e-5 | 2.1e-3 | 22x |
  | W_in | 0.050 | 0.66 | 13x |
  | token_emb | 3.6e-5 | 2.8e-4 | 8x |
  | W_out | 1.2e-3 | 1.2e-3 | 1x |

- Relative mobility (d angle / angle) is about equal. Because the task reads angles mod 2 pi, absolute mobility is what
  matters. A RESCUE could therefore be "a hotter optimiser on the phases", not "a start beyond sum = pi".
- 'High' also changes more than the sum:
  - Wrapped, its moves are opposite-signed on 48-59% of channels.
  - Its wrapped magnitudes average 2.4 rad, against 0.2-0.6 rad at default.
  - Its sums are centred on 2 pi (mean 6.0-6.3). Mean |wrap(a+ + a-)| is 0.90-1.00 rad, against 0.34-0.57 for SO default.
- With two non-negative angles, "large" and "sum < pi" cannot both hold above pi/2 each, so no single arm separates them.
  The toy has a partial control: abs default x3 fails 0/11 (0.45-0.76), but in the default gauge.
- Cheap fix:
  - Add AO_mid: a(+1), a(-1) iid U(0.9, 1.5). Sum < pi, same direction, about 5x absmag's mobility. Register M2b
    AO_high vs AO_mid. Run it on the toy first (CPU, minutes) and add it to the batch only if the toy's mid fails.
  - Print each arm's initial max |W_out| and step (rule 15 disclosure).
  - State in the prereg that a RESCUE means "a large start whose sums exceed pi". The start location is not separated from
    mobility.

**4. MAJOR: the toy table in the prereg is stale and partly misreports the JSONs.** Recomputed from all 80 grid + 25
other JSONs (SOLVED = acc >= 0.99):

| start | signed | non-negative |
|---|---|---|
| default | **6/12** (0.63-1.00, mean 0.906) | 0/12 (0.17-0.70, mean 0.482); x3 scale 0/11 (0.45-0.76) |
| same (balanced gauge) | 1/3 (0.79-1.00) | -- |
| absmag | 0/3 (0.02-0.69) | 0/3 (0.04-0.58) |
| low U(0.05, 1.5) | 0/11 (0.38-0.54) | **0/11** (prereg: 0/2) |
| high U(1.7, 4.5) | **8/8** (prereg: 2/2; no explore runs exist) | **11/11** (+8/8 without the omega clamp) |
| band (exact wrapped cancel) / offband (iid U(0.2, 2 pi - 0.2)) | -- | 8/8 / 9/9 |

- The toy's signed default solves only half its seeds. The sign asymmetry is clear in accuracy (0.906 vs 0.482) but
  weaker in SOLVED count than the "1-2/3" row suggests.
- 'High' beats the signed default (8/8 vs 6/12). The high start therefore carries an advantage beyond undoing `|.|`, which
  supports finding 3.
- signed_high is identical to abs_high (finding 2).
- The decisive swap cells (absmag, same) have n = 3 and are not in the 8-seed grid.
- "Its high-weight channels cancel": the top 8 channels by retrieval weight cancel on 100% of high runs, but they carry
  only 16-19% of the total weight. The weight-weighted cancel share is 0.33-0.39 and the unweighted share 0.23-0.33.
- Fix: replace the table in an amendment with these counts and the seeds. Run absmag/same on s13-17 (15 CPU runs). Report
  the weighted share as numbers.

**5. MINOR: missing reverse branches.** M1 labels "NO EFFECT" when SO_absmag is significantly BETTER than SO_same. M2
labels "NO EFFECT" when AO_high is significantly WORSE (a plausible outcome given finding 3: AO_wrapg at |W_out| 8.4 lost
its solution in WRAP_EXISTENCE). Add HELPS / WORSE labels (Fisher or perm p < .05 in the other direction), and rename NO
EFFECT to "NO EFFECT DETECTED" (rule 5).

**6. MINOR: the balanced gauge bundles three changes, and M3 sees them only on the default start.**
- The three changes:
  - omega' = max(|omega|, 1), on 30 of 64 channels.
  - Orthogonal action latents (default cos -0.83 to +0.44 across s800-807).
  - An exact saddle: W_out[:, 2:] and W_in[2:] are 0 with gradient exactly 0, and stay 0 (smoke checkpoints: absmax 0.0).
    The gauge arms are effectively rank 2. With two actions this changes no representation, only the optimisation.
- M1 and M2 are within-gauge, so none of this confounds them. Carrying a gauge-arm result back to the default arms rests
  on M3. M3 cannot see a gauge x start interaction, and at SO's ceiling (8/8 in WRAP_EXISTENCE) "NO DIFFERENCE" is
  uninformative.
- "A difference qualifies M1/M2" is not operationalised. State what a DIFFERS does to the reading.
- State the rank-2 saddle in the prereg.

**7. MINOR: "opposite moves' signs independent" holds only on average.**
- At the real init, the share of channels with opposite-signed moves is 0.34-0.91 across seeds 800-807. It is set by the
  cosine of the two action latents.
- Cheap secondary: per-seed opposite share vs SO / SO_same loss AUC.
- In the declared secondary, the hypothesis's predictor is the initial sign opposition, or |wrap(a+ + a-)|, rather than
  |a+ - a-|. Add it beside the current split.

**8. MINOR: undeclared standard secondaries.**
- The 1/(1-p) dropout re-score is missing. It is standard for one-layer runs below ceiling since 2026-10-04, and the AO
  arms sit at about 0.92.
- The arms are seed-paired (same weights, stream and dropout). A paired sign-flip test is a natural descriptive companion
  to the registered unpaired permutation test.
- Mixed outcomes (only one of M1 / M2 fires) have no pre-stated reading.

**9. MINOR: process.**
- "No pilot beyond the toy" leaves a new weight scale untested on the GPU recipe. AO_high has |W_out| up to 7.6 and
  13-22x mobility. A one-seed GPU pilot of AO_high and SO_absmag on an outside seed (e.g. 890; rule 29) checks the warm-up
  and the timing.
- `train_cancel` imports `train_variant` and through it ~90 model modules. Any edit to one of them during the batch (for
  example by GENERALITY work) can crash spawns or trip the final md5 check. Either freeze those modules for the batch's
  duration or copy `evaluate` into the trainer.

**10. MINOR: wording.**
- Initial |angle| mean is 0.21-0.62 rad across s800-807 (toy 0.33-0.63), not "~0.5"; the match is adequate.
- `toy_wrap`'s walk matches `environment_cancel` exactly: start uniform, p_plus, runs U{1..10}, and the start cell is not
  counted as visited.
- Toy accuracy is exact cell retrieval: argmax over earlier moves, correct if at the same cell, so floors near 0 are
  recency solutions. It is not the real task's observation prediction (floor 0.554), so toy accuracies are not comparable
  to the batch's.

## Minimal amendment before launch
- Gate M1 on SO_same SOLVED >= 4 (finding 1).
- Register M5, AO_absmag vs SO_absmag (finding 2a).
- Add HELPS / WORSE branches and "NO EFFECT DETECTED" with perm CIs (finding 5).
- Correct the toy table (finding 4).
- Restate the conclusion as start effect sizes, not "the whole cost" (finding 2c).
- Recommended, +16 runs (~+35 min): SO_high (finding 2b) and AO_mid if the toy's mid fails (finding 3).
