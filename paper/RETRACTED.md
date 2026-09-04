# RETRACTED — do not build on this draft

Superseded on 2026-09-04. Kept as a record of what was believed in April 2026.
The replacement draft is `paper_rank/`.

## Both the thesis and the evidence are gone

**The thesis.** The title claims heteroscedastic Kalman correction as the
contribution. Level 1.5 has since been **withdrawn as inference**. On
Match-Query with stochastic transitions — a task built specifically to supply
the drift the filter exists to correct — the effect does not grow with the
drift: `+0.003` in one recipe and `-0.141` in another, and never detectable at
`p=0.10` (`MQ_NOISE_2X2.md`, `MQ_NOISE_2X2_C2.md`). At n=5, no individual
component is load-bearing: the measurement head, the per-token gate and the
learned `Pi` can each be removed alone at no measurable cost
(`L15_ABLATION.md`). What survives is stabilisation at out-of-distribution
length, roughly `+0.06` to `+0.13` loss-matched — a much smaller claim than the
one the draft makes.

**The evidence.** `sections/04_results.tex` cites landmarks 15 times and `lm200`
7 times. **Every lm200 number was retracted on 2026-07-16**: those checkpoints
never converged (final CE ~1.0 against ~0.005), so the reported leaderboard
tracked training convergence rather than architecture. See the RETRACTION
section of `CLAUDE.md` and `CORRECTED_LM200_LEADERBOARD.md`.

The abstract leans on Level 1.5 four times and landmarks three times, so this is
not a matter of dropping a results table.

## What survives, and where it went

- Level 1.5 as **stabilisation** (not inference): Appendix A of
  `mapformer_math.tex`, stated with its withdrawal.
- The clean and noise checkpoints, which retrain bit-identically and were never
  in scope of the lm200 retraction.
