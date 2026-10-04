# NormStep on the text world -- results (2026-10-04)

Pre-registration `TW_NORMSTEP_PREREG.md` (+ Amendment 1 from the code audit, Amendment 2 from the pilot; committed
302d901 before launch). Runs `runs/tw_normstep/p0` (32 runs, one batch, seeds 10-17, all fresh); registered output
`TW_NORMSTEP_ANALYSIS.txt` / `TW_NORMSTEP.json` (`analyze_tw_normstep.py`, run by the driver after a second md5 check).
Void check OK (DirOnly non-direction steps exactly 0, ablation a no-op on every seed). Post hoc checks:
`docs/audits/2026-10-04/`.

## Registered verdicts
- **A (accuracy): NO DETECTABLE DIFFERENCE.** NormStep - MapWM +0.0059 (perm p 0.47; SOLVED 7/8 vs 8/8, Fisher 1.00);
  unmeasured below 0.096 (accuracy is bimodal per seed, so the sd-based MDE is large). Train mode (all dropout on, 40
  walks): +0.0024, p 0.60.
- **B (per-word clock): WORD-COUNT CLOCK, BIAS NOT SHOWN TO CAUSE IT.** Phase drift integrated over the optional words
  (adverbs, fillers, asides): NormStep 0.133 rad vs MapWM 0.077 rad, +0.057 (perm p 0.027; paired sign-flip p 0.047).
  NormStep - NormStepNB +0.032 (p 0.17): removing beta does not detectably remove it.
- **Composite: NORMSTEP DOES NOT CARRY OVER CLEANLY** -- by the registered rule, because B fired.

| arm | T=1024 acc | SOLVED | per-word drift (rad) | per-move clock seeds (drift > 16 of 64) | train-mode acc | own map |
|---|---|---|---|---|---|---|
| MapWM | 0.973 +/- 0.058 | 8/8 | 0.077 | 4/8 | 0.994 | 0.969 |
| NormStep | 0.979 +/- 0.060 | 7/8 | **0.133** | 2/8 | 0.996 | 0.981 |
| NormStepNB (no beta) | 0.941 +/- 0.082 | 7/8 | 0.102 | 3/8 | 0.992 | 0.941 |
| DirOnly (steps on direction words only) | 0.972 +/- 0.001 | 8/8 | 0 | 0/8 | 0.966 | 0.975 |

Floors on this eval stream: best constant 0.512, reversal-copy 0.602. r(final loss, acc) over 32 runs -0.73.

## How big the word-count clock is
Small. The registered readout fired at +0.057 rad, just past its 0.05 materiality floor, against a random-phase
level of ~1.57 rad. No optional-word channel drifts by more than 1 rad on any of the 32 runs (0/64 everywhere), so
no channel has become a clock; the drift is diffuse. On the injection scale (`docs/audits/2026-10-03/
tw_normstep_sensitivity_out.txt`) it corresponds to a per-word tick of about 0.05-0.1 of a direction step. It costs
nothing measurable: the NormStep seeds with the most word drift (0.16-0.19 rad) score 0.999-1.000.

## Where it comes from (post hoc, `docs/audits/2026-10-04/tw_normstep_wordsteps.py` / `_out.txt`)
- **Not the bias.** The NormStepNB contrast does not fire, and the two NormStep seeds with a large beta step (0.17 and
  0.18 of a direction step, s11 and s16) have the LOWEST word drift (0.072, 0.042): they are the per-move clock seeds,
  where beta's step is part of the clock and is cancelled on optional words. On the six map-type seeds beta's step is
  0.003-0.008 of a direction step: training put beta nearly in W's null space.
- **The step sizes are about the same; the embedding norms are not.** Optional words' omega-scaled steps relative to
  direction steps are similar in map-type seeds (MapWM 0.031-0.042, NormStep 0.024-0.049). MapWM keeps optional-word
  embeddings at 0.36-0.54 of a direction word's norm; NormStep's are 2.1-2.9x (it has no reason to shrink them, since
  the step ignores scale). This is the same scale knob whose removal fixed the leak on the new-object task. It is a
  description, not a demonstrated cause of the extra drift.

## Two further findings
- **An oracle that knows which words move is NOT the ceiling: asides cap it** (post hoc,
  `docs/audits/2026-10-04/dironly_aside_errors.py` / `_out.txt`). DirOnly sits at 0.970-0.974 on all 8 seeds (sd 0.001).
  On two seeds checked, **100% of its errors (78/78, 89/89) name an object mentioned in an aside ("she thought about a
  cat .") at an earlier visit to the same cell.** With no step on any non-direction word, an aside's noun sits at the
  cell's phase. The solved learned-step models make none of these errors (MapWM s16, NormStep s12: 0 errors): learning
  the step map lets non-movement words move the aside off the cell. So "only the action words should move" (TEM-t's
  action-only update, ActOnly on the new-object task) is the wrong target in language: some what-side words need steps.
- **Eval mode under-reports clock-type solutions** (Amendment 2). Train-mode minus eval-mode accuracy on the same
  walks: MapWM +0.023, NormStep +0.017, NormStepNB +0.051, DirOnly +0.002. The gaps sit on the runs below ceiling,
  clock-type and map-type alike (NormStep s16, clock, 0.836 -> 0.974; NormStepNB s12, clock, 0.828 -> 0.993; MapWM
  s12, map-type with 3 drifting channels, 0.828 -> 0.987; NormStepNB s10 0.818 -> 0.961), so "clock-type" in the
  pilot's reading was too narrow: the dependence on dropout goes with not having converged to the clean solution.
  Attention-probability dropout alone accounts for it (`docs/audits/2026-10-03/dropout_mode_check_out.txt`).
  Unchecked on other batches.

## What it means
NormStep neither helps nor hurts accuracy on navigation told in words (A unmeasured below 0.096; it cannot show its
new-object benefit here, since objects are 16 learned words). The predicted word-count clock is real by the registered
test but tiny, not from the LayerNorm bias, and free of accuracy cost at this budget. NormStep is safe to use here
but has no measured benefit. The larger lesson is DirOnly's: the useful separation is not "actions move, everything
else is still", because language puts objects where they are not.

## Caveats
- n = 8, one grammar, 1 layer, r = 4, 900 epochs, T = 1024. Accuracy is bimodal (per-move clock vs map seeds), so A is
  weak; the per-move clock rate (2/8 vs 4/8) is unmeasured.
- B fired just past its materiality floor; its paired test is at p 0.047. Read it as "a small per-word drift exists",
  not as a clock in any channel.
- Both mechanism readings (embedding norms; asides) are post hoc; the aside check covers 2 of 8 DirOnly seeds.
