# Fresh-seed replication of CLOCK_SPEED's primary contrast -- results (2026-10-13)

Pre-registration `CLOCK_SPEED_REP_PREREG.md` + Amendment 1 (independent check; power and wording pre-committed). Batch
00:01-01:34 at 4a946f3 on GPU 1; k-back recency (k_max 128, T = 1024, 300 epochs), Signed_r4 and Abs_r4 on fresh seeds
230-245 (32 runs). Registered output `CLOCK_SPEED_REP_ANALYSIS.txt`, `CLOCK_SPEED_REP_VERDICTS.json`.

| arm | AUC (mean training loss) | epoch to running loss < 2.0 / < 1.0 (median) | loss at 100 / 200 / 300 | accuracy T (2T) |
|---|---|---|---|---|
| Signed_r4 | 2.271 +/- 0.269 | 213 / 296 | 2.76 / 2.18 / 0.93 | 0.684 (0.660) |
| Abs_r4 | 2.165 +/- 0.231 | 187 / 301 | 2.74 / 1.86 / 1.07 | 0.683 (0.657) |

## Registered
**S1 (Abs - Signed, n = 16 fresh seeds): UNMEASURED** -- paired mean AUC difference -0.106 (exact sign-flip p 0.19; 95% CI
[-0.270, +0.055]); Abs faster on 10/16 seeds. Pre-committed reading (Amendment 1): with ~50% power at the first batch's
effect size, UNMEASURED is not a failed replication; it does not fire on fresh seeds either.

## Post hoc (labelled; does not change any verdict; no third batch, as pre-committed)
Pooled with CLOCK_SPEED's 16 seeds: mean -0.151, sign-flip p 0.018 (Monte Carlo), 22/32 seeds faster with the non-negative
step. The non-negative step leaves the chance plateau earlier in both batches (median epoch 172 and 187 vs 210 and 213) but
is not lower at epoch 300 here (1.07 vs 0.93): the difference is early, not in where training ends up within 300 epochs.

## What it means
- **The clock side stays unestablished.** Two registered tests (CLOCK_SPEED, this replication) both read UNMEASURED; the
  effect on fresh seeds is about half the first batch's (-0.106 vs -0.197). The direction is the same (the non-negative step
  leaves the plateau earlier) and the pooled post hoc reading is p 0.018, but by the pre-committed rule that does not
  upgrade the verdict.
- **The learnability effect is asymmetric:** large and clean where a map is needed (ring: signed faster on 8/8 paired
  seeds at every dial setting; non-negative steps never find the wrap solution), small and unestablished in the other
  direction where a content clock is needed. Do not claim a double dissociation.

## Caveats
- One clock task, 300 epochs, no run converged, 1 layer, n = 16; AUC measures progress within this budget.
