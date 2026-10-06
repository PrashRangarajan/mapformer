# Gain granularity between MapEM and MapPoPE -- results (2026-10-05)

Pre-registration `GAIN_GRAIN_PREREG.md` (+ Amendment 1 from the code audit, committed dadb3a3 before launch; the driver
logged that commit and a clean tree at launch). Runs `runs/gain_grain/p0` (120 runs, one batch, seeds 26-45, n = 20 per
arm). Registered output `GAIN_GRAIN_ANALYSIS.txt` (`analyze_gain_grain.py`); declared secondaries
`docs/audits/2026-10-05/gain_grain_remap_out.txt` and `gain_grain_rescore_out.txt`. Pilot (all arms solved, seeds
100-101) read after the branches were committed and disclosed in the prereg.

Every arm scores `sum_c G_c(content) * A_c * cos(dtheta_c + delta_c)` on the same path phase (paper torus, T = 128, rank 2,
32 angles per head); they differ in the content gain G:

| arm | content gain | T=128 accuracy | SOLVED | epoch loss first < 0.05 (median) |
|---|---|---|---|---|
| MapWM (W) | per frequency, with content phase | 0.9902 +/- 0.021 | 18/20 | 132 |
| MapPoPE-Pair (P) | per frequency, >= 0 | 0.9995 | 20/20 | 74 |
| **GainScalar (S)** | **one per token per head, >= 0; delta 0** | **0.9999** | **20/20** | **23** |
| GainMod4 (M) | one per band of 8 frequencies, >= 0 | 0.9997 | 20/20 | 21 |
| MapEM (E) | one signed scalar, q.k | 0.9968 | 19/20 | 148 |
| MapEM, softplus(q.k) (N) | one scalar, >= 0 | 0.9816 +/- 0.028 | **12/20** | 117 |

Floors: best n-gram 0.598, always-blank 0.507.

## Registered (a): SCALAR GAIN SUFFICES (per-module also as good) -- with the NO HEADROOM qualifier
S vs P: +0.0004, SOLVED 20/20 vs 20/20, non-inferiority at -0.01 one-sided p < 0.0001. M vs P: +0.0002, 20/20 vs 20/20,
p < 0.0001. One positive gain per token, or one per band of scales, loses nothing against MapPoPE's per-frequency gain.

Qualifier (registered): the headroom check did not pass. MapWM solved 18/20 here (P - W +0.0092, p 0.011, below the
0.01 materiality floor; SOLVED Fisher p 0.49), against 10/16 in MAPPOPE_PAIR at seeds 10-25. On these seeds the task
barely revealed MapWM's cost, so "as good" shows that the coarse gains are not worse, not that they avoid a cost this
task can expose. The score-rule effect of MAPPOPE_PAIR keeps its direction on fresh seeds but shrinks (+0.024 -> +0.009).

## Registered (b): NON-NEGATIVE WORSE
N vs E: -0.0152 (perm p 0.026; 95% CI -0.029 to -0.002), SOLVED 12/20 vs 19/20 (Fisher p 0.020). Making MapEM's content
gain non-negative with softplus(q.k) hurts. The registered label reads "the sign freedom helps"; see the caveats for why
"softplus on the content dot product hurts" is the safer reading.

## Secondaries (no verdict)
- **The form of the gain matters more than its sign.** The best arm is also non-negative: S - N +0.018 (p < 0.0001), 20/20
  vs 12/20 (p 0.003). A product of per-token gains (S) works; softplus of a content dot product (N) does not. S - E +0.0031
  (p 0.009, below materiality), 20 vs 19.
- **Speed (descriptive):** the single positive gain reaches training loss 0.05 by a median epoch 23, MapPoPE 74, MapWM 132,
  MapEM 148: the simplest gain trains about 3x faster than MapPoPE and 6x faster than MapWM on this task.
- **Remap probe:** S is pure gain (gain share 1.000, field width 5 cells, peak at d = 0 on every run), using the fine bands
  most (spectrum share 0.35 / 0.27 / 0.21 / 0.18, fine to coarse). M is gain plus some width (0.88 / 0.12), peaked at 0.
  P gain + width (0.74 / 0.17). E and N: pure gain by construction, but their observation-key fields peak 5-7 cells away
  from d = 0, and E's content gain on (action, observation) pairs is negative in the median head (25/40 heads entirely
  negative): MapEM uses the sign, likely as a gauge (-A_X)(-A_P), which softplus removes. Its retrieval route at rank 2 is
  not explained by this probe (it covers observation keys only); unresolved.
- **Clock / collapse classification:** "SOLVED iff a clean head" does NOT hold at T = 128 (5-16/20 per arm): at short
  walks drifting or collapsed heads still solve, consistent with the formal note that these defects cost at long gaps.
- Out of distribution (rule 10): S - P +0.009 / +0.014 at T = 512 / 1024 (n.s.); N - E -0.067 / -0.112.
- Dropout-scale re-score: every arm moves <= 0.0002; no contrast changes.

## What it means
- **The cleanest gain-field model is enough.** Content setting one non-negative number per token per head -- the pure
  rate-remapping form, with the field always centred on "here" -- matches MapPoPE on this task and trains fastest. The
  per-frequency width freedom that distinguishes MapPoPE is not needed here.
- **Non-negativity is not the active ingredient by itself.** Applied to MapEM's content dot product it hurts; applied as a
  product of per-token gains it is the best arm. What helps is a factorised, content-independent position kernel with a
  per-token gain, not the sign constraint as such.

## Caveats
- One task, T = 128, rank 2, 1 layer; the headroom qualifier above (MapWM 18/20).
- S and M have ~31-33k fewer content-projection parameters than P, no delta and tied pairs; an AS GOOD verdict is
  unaffected (fewer parameters and still as good), but the speed difference is not attributed to any one of these.
- N differs from E by softplus, which also halves the gradient to A_X at init and gives a positive mean gain; "the sign
  freedom helps" is not separated from those.
- The remap and basin readouts are post-registration secondaries with thresholds set at T = 1024.
