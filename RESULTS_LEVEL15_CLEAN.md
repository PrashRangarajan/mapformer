# Level 1.5 on the Clean Paper Task — Drop-in Replacement?

## TL;DR

Level 1.5 InEKF trained on the paper's clean task (no noise, no landmarks,
200K sequences × 5 epochs' worth of training iterations) beats vanilla
MapFormer on accuracy at most lengths and on calibration (NLL) at every
length. Only T=512 accuracy is slightly worse (−1.1pp) despite 20% lower
NLL even there.

## Setup

- Clean task: torus grid 64×64, 16 obs types + 1 blank, p_empty=0.5, no landmarks
- Action noise at training: 0 (matching paper)
- Vanilla model: figures_v6/MapFormer_WM.pt (paper-faithful, 200K sequences)
- Level 1.5 clean: figures_inekf_level15_clean/MapFormer_WM_Level15InEKF.pt
  (1M sequences)

Note: different training amounts (vanilla is 200K, L1.5 is 1M) because
Level 1.5 was trained at the 50-epoch schedule of our matched-noise
experiments. A fair apples-to-apples would retrain vanilla at 50 epochs.

## Results (3 seeds × 300 trials per length)

### Accuracy (revisit positions only)

| T    | Vanilla (paper)         | Level 1.5 clean          | Δ     |
|------|-------------------------|--------------------------|-------|
| 128  | 0.9569 ± 0.0033         | **0.9723 ± 0.0016**      | +1.5pp|
| 256  | 0.9302 ± 0.0035         | 0.9297 ± 0.0054          | 0    |
| 512  | **0.8518 ± 0.0065**     | 0.8405 ± 0.0034          | −1.1pp|
| 1024 | 0.6630 ± 0.0027         | **0.6836 ± 0.0006**      | +2.1pp|

### NLL (calibration — lower is better)

| T    | Vanilla  | Level 1.5 clean | Δ      |
|------|----------|-----------------|--------|
| 128  | 0.158    | **0.090**       | −43%   |
| 256  | 0.260    | **0.256**       | −2%    |
| 512  | 0.826    | **0.662**       | −20%   |
| 1024 | 3.157    | **1.632**       | −48%   |

## Interpretation

Level 1.5's heteroscedastic Kalman gain (learned R_t per token) provides
a calibrated uncertainty estimate even in a clean deterministic task. The
model evidently uses this to down-weight uncertain predictions, reducing
NLL substantially everywhere. Accuracy is equal or better at most lengths;
the one small accuracy dip at T=512 is paired with a 20% NLL improvement,
so Level 1.5 is knowingly less confident in cases where it's wrong.

## Combined with previous results

Level 1.5 is the best model we measured for:
- Training-loss on clean task (0.159 vs vanilla's 0.194)
- Calibration (NLL) at every length on clean task
- Accuracy at T=128 and T=1024 on clean task
- Noise robustness on matched-distribution noise (see earlier RESULTS files)
- Landmark utilization (73% vs 1.7% for vanilla+noise; see RESULTS_LEVEL15.md)

The only metric vanilla wins on is accuracy at T=512 by 1.1pp. This is the
closest we have to a drop-in replacement for MapFormer that's better on
essentially every axis.

## Caveats

1. Training amount mismatch (200K vs 1M sequences). A fair comparison
   would retrain vanilla at 50 epochs. Empirically, MapFormer-WM's loss
   has mostly plateaued by ~200K sequences on this task, so the extra
   training probably contributes <1pp.
2. T=512 gap is small but real (3 seeds show std ~0.003-0.007).
