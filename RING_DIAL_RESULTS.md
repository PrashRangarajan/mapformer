# The ring dial: does a non-negative rotary step learn the map? -- results (2026-10-12)

Pre-registration `RING_DIAL_PREREG.md` + Amendment 1 (independent audit; pilot disclosed): a LEARNABILITY dial. A
non-negative rotary step (Abs_r4 = |W_out W_in x|) can represent the signed code exactly by wrapping (Delta(-1) = 2 pi /
omega - Delta(+1); constructed by the audit), so the question is whether training finds the map as reliably as with a signed
step (Signed_r4), on one task family (the 1D ring, 32 cells, T = 128, 300 epochs, 1 layer) at three values of the
backward-step dial. Batch 04:24-07:17 at 44efc5d on GPU 0 (another user held GPU 1); 48 runs, seeds 30-37. Registered
output `RING_DIAL_ANALYSIS.txt`, `RING_DIAL_VERDICTS.json`.

| p_plus | Signed: acc T=128 (SOLVED) | Abs: acc T=128 (SOLVED) | registered | acc T = 512 (OOD): Signed / Abs |
|---|---|---|---|---|
| 0.5 | 1.000 (8/8) | 0.973, min 0.931 (2/8) | **ABS LEARNS WORSE** (Fisher p 0.0070; acc -0.028, p 0.007) | 0.995 / 0.878 |
| 0.75 | 1.000 (8/8) | 0.968, min 0.954 (1/8) | **ABS LEARNS WORSE** (Fisher p 0.0014; acc -0.032, p 0.0002) | 0.994 / 0.889 |
| 0.9 | 1.000 (8/8) | 0.954, min 0.825 (4/8) | UNMEASURED (acc -0.046, p 0.026, below the 0.05 materiality; Fisher p 0.077) | 0.942 / 0.842 |
Floors (CANCEL): 0.554 / 0.544 / 0.520; one-layer index RoPE (CANCEL, n = 8): 0.717 / 0.676 / 0.825. No label changes under
the dropout-scale re-score (every arm moves <= 0.0001).

## Registered
- **Where steps cancel (p_plus 0.5, 0.75), the non-negative step learns the map worse**: it converges on 2/8 and 1/8 seeds
  against 8/8 for the signed step, at the same parameterisation and initial weights apart from the absolute value. Both
  firings are on SOLVED; the accuracy deficit is small (-0.03, below the 0.05 materiality floor) because Abs gets most of
  the way (0.95-0.97, far above the floor and the one-layer index).
- **At p_plus 0.9 it is UNMEASURED** (4/8 vs 8/8; accuracy -0.046).

## Secondaries (declared; no verdict)
- **Abs does not find the wrap solution.** Wrap-aware opposition, median over channels of |wrap(step(+1) + step(-1))|: Signed
  0.03-0.07 rad with 63-82% of channels cancelling within 0.1 rad; Abs 0.89-0.93 rad with 4-7% cancelling. A non-negative
  step that could reproduce the signed code mod 2 pi learns a different, non-cancelling code that is good but not exact
  (and degrades more past the training length: 0.84-0.89 vs 0.94-1.00 at T = 512). The raw opposition (Signed 0.07-0.17, Abs
  1.83-1.88) agrees here, but is not wrap-aware in general.
- **No graded dial.** The SOLVED counts do not track the amount of cancellation (2/8, 1/8, 4/8) and the accuracy deficit grows
  slightly toward p_plus 0.9 (-0.028, -0.032, -0.046), where wrap-only revisits are the majority (0.74 of revisits at 0.9
  vs 0.11 at 0.5): the dial also changes how much the task needs the periodic code.

## What it means
- **A sign constraint on a rotary step is a learnability handicap, not a representational one, and it shows up within one
  task family and one model.** The non-negative step can represent the map, training does not find it (no channel-wise
  cancellation), and the result is a near-miss code that rarely converges. This is the within-task version of the torus
  sign result (`SIGN_MATCHED_RESULTS.md`, 0/8 vs 8/8), with the representational reading now excluded by construction.
- For the clock / map framing: in the rotary slot, Howard's Cases II and III coincide in what they can represent and
  differ in what training finds. The dial does not show a cost that scales with the need to cancel.

## Caveats
- One task family (the 1D ring), T = 128, 300 epochs, 1 layer, n = 8. SOLVED is a training-loss threshold; Abs's misses are
  small in accuracy. The batch took ~2.9 h on one GPU (estimate was ~1 h).
- The mechanism of Abs's non-cancelling near-solution is not identified.
