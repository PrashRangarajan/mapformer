# The clock side of the clock / map dissociation -- pre-registration (2026-10-12, before any run)

## Question
The paper's frame is Howard et al. 2014's integrator (`docs/lit/LIT_CLOCK_MAP.md`, `docs/lit/LIT_HOWARD_TRANSFORMER.md`):
one accumulator driven by a constant rate (Case I: time), a non-negative content-dependent rate (Case II: a content clock,
path length) or a signed rate (Case III: position, a map). The three families of rotary position are these cases:
index RoPE (I), monotone content-dependent phase (II: CARoPE, GRAPE-AP / softplus, |.|), signed content-dependent phase
(III: MapFormer). On the MAP task the prediction is established at matched length: forcing Case II on the torus fails
(`SIGN_MATCHED_RESULTS.md`, Abs - Signed -0.177, 0/8 vs 8/8). On the CLOCK side it is not: on k-back recency (a
content-counting task, where index RoPE fails at 0.234) every phase arm was at or near ceiling at the training length
(`RECENCY_RESULTS.md`: Signed 1.000, CARoPE 1.000, Pos 0.979, Abs 0.961), and the registered -0.004 was read out of
distribution (T = 2048 after training at 1024). **On a content-clock task made hard enough to stay below ceiling, at the
training length, does a monotone (Case II) phase beat the signed (Case III) one -- completing a double dissociation --
or does the signed phase match it (it can emulate a clock)?**

## Task and recipe (`train_recency.py`, unchanged; = `run_recency.sh` except k_max / T, set by the calibration below)
k-back recency: retrieve the k-th most recent symbol among fillers, 16 symbols, p_query 0.25, min_gap = k_max (the
default, which keeps the answer-stream shortcut closed), 300 epochs x 48 batches, batch 16, cosine, lr 1e-3, 1 layer,
d 128, 2 heads, `--fast-attn` (as RECENCY; every arm shares it). Evaluation at the training length T and at 2T
(secondary, out of distribution).

## Calibration (pilot, seed 100, outside the batch; rule fixed now)
Settings tried in order: (k_max 128, T 1024), (k_max 128, T 2048), (k_max 256, T 2048); arms Signed_r4 and CARoPE_r4 at each.
The batch uses the FIRST setting at which the pilot's Signed_r4 accuracy at the training length is in [0.30, 0.95]
(below ceiling, well above the floor). If none qualifies: no batch; report. The pilot's CARoPE result is disclosed but
does not enter the choice. The gate (`validate_recency.py`: chance, most-recent shortcut, n-gram floors, scored targets
per sequence) is run for the chosen setting before launch.

## Arms (`train_variant.VARIANT_MAP`; n = 8, seeds 200-207, fresh)
| arm | Howard case | step |
|---|---|---|
| Signed_r4 | III (signed) | `W_out W_in x` (rank 4) |
| CARoPE_r4 | II (monotone) | `1 / (softplus(.) + 1)` (CARoPE) |
| Pos_r4 | II (monotone) | `softplus(.)` (GRAPE-AP) |
| RoPE | I (constant) | token index |

## Registered decisions (`analyze_clock_side.py`, to be written with the amendment; two-sided permutation tests, n = 8 v 8)
Primary readout: held-out accuracy at the training length (eval mode). Monotone = mean of CARoPE_r4 and Pos_r4 per seed index
(RECENCY's registered composite).
- **D1 CLOCK SIDE.** contrast mean(monotone) - Signed. **CASE II ADVANTAGE (double dissociation)**: d > 0, perm p < .05
  and d >= 0.02. **SIGNED BETTER**: d < 0, p < .05, |d| >= 0.02. **NO DIFFERENCE (signed matches the clock)**: neither
  fires and the 95% CI lies within [-0.03, +0.03]. **UNMEASURED**: otherwise.
- **D2 INDEX CHECK.** RoPE - Signed: **INDEX FAILS** if d <= -0.10 with p < .05 (a token clock cannot count content);
  otherwise reported as it falls.
Secondary: SOLVED counts (final-5% loss < 0.05) with Fisher tests; per-arm accuracy at 2T (OOD, rule 10); dropout-scale
re-score of the primary (one layer); r(final loss, acc); the growth exponent alpha per arm.

## Process
Pilot (calibration) -> Amendment 1 (setting chosen, gate) -> independent code audit (rule 29) -> Amendment 2 -> batch on
GPU 0 only while another user holds GPU 1 (user's permission 2026-10-12). Cost estimate: 32 runs x 300 epochs at
2-4 s/epoch, 4 concurrent on one GPU: ~1.5-2.5 h.
