# The clock side of the clock / map dissociation -- results (2026-10-12)

Pre-registration `CLOCK_SIDE_PREREG.md` + Amendment 1 (calibration, gate), Amendment 2 (independent audit) and Amendment 3
(interpretation: rotary non-negative steps wrap, so every label is about learnability, not representation; written before
results). Batch 01:32-04:21 at 9f91be0 on GPU 0 (another user held GPU 1); 40 runs (5 arms x seeds 200-207), k-back
recency, k_max 128, trained and tested at T = 1024, 300 epochs. Primary readout: the ~1,670-target re-evaluation
(`CLOCK_SIDE_EVAL.json`). Registered output `CLOCK_SIDE_ANALYSIS.txt`, `CLOCK_SIDE_VERDICTS.json`.
**Process note:** the driver's analysis crashed AFTER printing every registered line, in a secondary (`signflip_p` returns a
dict); the secondary was fixed, the analysis re-run (registered lines identical) and the done marker set by hand (logged).

| arm | Howard case (real-rate reading) | accuracy at T = 1024 (re-scored) | min | SOLVED / STALLED / DESCENDING |
|---|---|---|---|---|
| Signed_r4 | III | 0.638 +/- 0.267 (0.638) | 0.177 | 0 / 2 / 6 |
| Abs_r4 | II (sign removed only) | 0.723 +/- 0.171 (0.721) | 0.454 | 0 / 2 / 6 |
| CARoPE_r4 | II | 0.807 +/- 0.136 (0.807) | 0.544 | 0 / 1 / 7 |
| Pos_r4 | II | 0.722 +/- 0.218 (0.723) | 0.228 | 0 / 1 / 7 |
| RoPE (index) | I | 0.188 +/- 0.014 (0.189) | 0.166 | 0 / 8 / 0 |
| floor (best trivial, 8,000-episode gate) | | 0.072 | | |

## Registered
- **D1 (Abs - Signed, the sign alone): UNMEASURED** -- +0.085 (perm p 0.49; 95% CI [-0.17, +0.34]).
- **D1c (CARoPE - Signed): UNMEASURED** -- +0.169 (p 0.16; CI [-0.08, +0.41]).
- **D1p (Pos - Signed): UNMEASURED** -- +0.084 (p 0.53; CI [-0.20, +0.36]).
- **D2 INDEX FAILS** -- RoPE - Signed -0.450 (p 0.0011): a token clock cannot count content (RECENCY's result at k_max 128).
- No label changes under the dropout-scale re-score (every arm moves <= 0.002).

## What limits it
- **Nothing converged.** 0/40 runs SOLVED; 26/32 phase-arm runs still descending at 300 epochs. Accuracy is almost entirely
  training progress: r(final loss, accuracy) -0.987 over 40 runs (-0.98 to -0.99 within each phase arm). The calibration rule
  chose the first setting with the signed pilot run in [0.30, 0.95] -- a level of accuracy, not convergence -- so the batch
  measured how far each parameterisation got in a fixed budget, with very large seed-to-seed spread (Signed 0.18-1.00; two signed seeds reach 0.999-1.000 without meeting the loss threshold).
- **The direction is the monotone arms' on every contrast** (+0.08 to +0.17; CARoPE ahead of Signed on 5/8 paired seeds,
  sign-flip p 0.10), as on the one pilot seed (0.77 vs 0.49), but nothing is detectable at n = 8 with this spread.
- By Amendment 3, any difference here would be about learnability: every phase arm can represent every other arm's solution
  (rotary steps wrap mod 2 pi).

## Secondaries (declared; no verdict)
- Paired by seed (the arms share initial weights, data and eval items): Abs +0.085 (4/8 seeds ahead, sign-flip p 0.40),
  CARoPE +0.169 (5/8, p 0.10), Pos +0.084 (3/8, p 0.47); RoPE -0.450 (1/8, p 0.016).
- Eval targets per run 1,672 (vs ~213 in-training); the in-training eval agrees to within 0.02 per arm.

## What it means
- **The clock side of the dissociation is not shown.** On a content-counting task at this budget, the non-negative
  parameterisations are directionally ahead of the signed one, but not detectably, and no arm converged. The index model
  fails, as RECENCY found.
- A conclusive version needs a budget at which the arms converge (or a convergence-speed readout registered as primary);
  that is a candidate follow-up, not run.

## Caveats
- One task, one difficulty chosen by a level-of-accuracy rule, 300 epochs, 1 layer, n = 8; --fast-attn (TF32) in every arm.
