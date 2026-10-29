# Level 1.5 on Match-Query -- does the correction survive with no measurements?

Match-Query's query phase feeds MASK back, so the InEKF innovation `z - theta_hat` is computed from an uninformative token at every scored step. If the correction is measurement-driven it should collapse to plain path integration here.

All three arms trained in the SAME batch (standing rule 3). `Vanilla_ExtraHead` has MORE parameters than Level15 (667,478 vs 650,774) and is the capacity control. **Chance 0.0625; never-moved floor 0.0893.**

| variant | params | T_query=256 (train) | T_query=512 (OOD) |
|---|---|---|---|
| Vanilla | 601,174 | 0.8884 ± 0.1401 | 0.9024 ± 0.1172 |
| Vanilla_ExtraHead | 667,478 | 0.7562 ± 0.2160 | 0.7055 ± 0.2561 |
| Level15 | 650,774 | 0.8764 ± 0.2128 | 0.8528 ± 0.2542 |

## Per-seed (T_query=256)

- `Vanilla`: 0.7312, 1.0000, 0.9340
- `Vanilla_ExtraHead`: 0.5886, 1.0000, 0.6801
- `Level15`: 0.6306, 1.0000, 0.9986

## Reproducibility check passes exactly

The Vanilla arm here was retrained from scratch in this batch, and it reproduces
`MATCH_QUERY_RESULTS.md` to four decimals: **0.8884 vs 0.888**, per-seed
**0.7312 / 1.0000 / 0.9340 vs 0.731 / 1.000 / 0.934**. So this batch is directly
comparable to everything previously reported on this task, and the Level15 and
ExtraHead rows can be read against the published Vanilla number as well as
against their own batch-mate.

## Verdict: the correction shows NO advantage without measurements

Level15 0.876 vs Vanilla 0.888 at T_query=256, and 0.853 vs 0.902 at 512. That
is the prediction stated before the run: Match-Query feeds MASK back throughout
the query phase, so the InEKF innovation `z - theta_hat` is computed from an
uninformative token at every scored step, and the filter has nothing to correct
with. It collapses to plain path integration.

This **bounds** the correction result rather than contradicting it. Level 1.5's
+24.8pp over Vanilla (lm200 OOD T=512, `LM200_CORRECTED_MULTISEED.md`) is earned
in a regime with informative observations. It does not transfer to a regime
without them, and nothing in the correction's design says it should. The two
surviving result lines are complementary, not competing -- which is what this
run was launched to determine.

## The comparison is UNDERPOWERED, and that limits every ordering above

Every arm has one seed at 1.000 and one much lower:

| variant | s0 | s1 | s2 | sd |
|---|---|---|---|---|
| Vanilla | 0.7312 | 1.0000 | 0.9340 | 0.140 |
| Vanilla_ExtraHead | 0.5886 | 1.0000 | 0.6801 | 0.216 |
| Level15 | 0.6306 | 1.0000 | 0.9986 | 0.213 |

With n=3 and standard deviations of 0.14-0.26, the -0.012 gap between Level15
and Vanilla is far inside the noise. **This run establishes "no advantage", not
"a deficit", and it cannot order the three arms.** Distinguishing them on this
task would need many more seeds -- the same lesson `MATCH_QUERY_SCALE.md`
recorded when going from n=3 to n=5 moved the base figure from 0.888 +/- 0.140 to
0.730 +/- 0.247.

## Extra capacity is not free here

`Vanilla_ExtraHead` (667,478 params, MORE than Level15's 650,774) scores 0.756 --
the lowest of the three. So generic added capacity costs ~0.13 on this task while
the InEKF costs ~0.01. Read cautiously given the variance above, but it does mean
Level15's near-parity with Vanilla is not simply "extra parameters are harmless".
