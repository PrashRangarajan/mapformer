# Does a skewed basis inject drift that accumulates with length?

State residual after removing everything net displacement can explain,
normalised by the action scale. `s_t` is exact (a cumsum of a linear map
of the embeddings), so no forward pass is involved.

| arm | t=128 | t=512 | t=1024 | t=2048 | growth |
|---|---|---|---|---|---|
| `Vanilla` | 15.802 ± 15.471 | 63.500 ± 61.786 | 128.674 ± 124.633 | 254.718 ± 247.239 | **16.12x** |
| `Vanilla_r4` | 1.389 ± 0.734 | 5.560 ± 2.972 | 11.184 ± 5.986 | 22.213 ± 11.919 | **15.99x** |

n = 8 seeds, 24 trajectories each, env seed 10000.

## The pre-registered comparison

- t=128: `Vanilla` 15.802 vs `Vanilla_r4` 1.389 -> **11.38x**
- t=512: `Vanilla` 63.500 vs `Vanilla_r4` 5.560 -> **11.42x**
- t=1024: `Vanilla` 128.674 vs `Vanilla_r4` 11.184 -> **11.51x**
- t=2048: `Vanilla` 254.718 vs `Vanilla_r4` 22.213 -> **11.47x**

Predicted ratio from the opposition errors measured in `ACTION_GEOMETRY.md` (0.495 vs 0.092): **5.4x**.


## Does drift explain the accuracy, seed by seed?

Per-seed drift at t=1024 against per-seed held-out accuracy at T=1024
(`RANK_SWEEP.json`, same checkpoints).

| arm | seed | drift @1024 | acc @T=1024 |
|---|---|---|---|
| `Vanilla` | 0 |   319.46 | 0.869 |
| `Vanilla` | 1 |    29.08 | 0.820 |
| `Vanilla` | 2 |    19.57 | 0.864 |
| `Vanilla` | 3 |    27.23 | 0.912 |
| `Vanilla` | 4 |   291.65 | 0.749 |
| `Vanilla` | 5 |   132.76 | 0.729 |
| `Vanilla` | 6 |    25.75 | 0.874 |
| `Vanilla` | 7 |   183.90 | 0.856 |
| **`Vanilla` within-arm** | | | **r = -0.363** (n=8) |
| `Vanilla_r4` | 0 |    12.95 | 0.906 |
| `Vanilla_r4` | 1 |     3.70 | 0.934 |
| `Vanilla_r4` | 2 |    22.57 | 0.937 |
| `Vanilla_r4` | 3 |     8.66 | 0.921 |
| `Vanilla_r4` | 4 |     5.39 | 0.925 |
| `Vanilla_r4` | 5 |    12.30 | 0.919 |
| `Vanilla_r4` | 6 |    15.09 | 0.912 |
| `Vanilla_r4` | 7 |     8.81 | 0.903 |
| **`Vanilla_r4` within-arm** | | | **r = +0.086** (n=8) |

**Pooled over both arms (n=16): r = -0.614, and r = -0.733 against log(drift).**

## Verdict

**Split the claim, because the two halves came apart.**

**CONFIRMED -- a skewed basis injects a path-length residual that grows linearly
in t.** Growth over a 16x length increase is 16.12x (r=2) and 15.99x (r=4):
linear to two decimals, which is what a constant symmetric component multiplying
n_+ + n_- predicts and what nothing else obviously would. The r=2 code carries
**11.4x** more of it than r=4, and the ratio is flat across every length
measured. r=4 also compresses the *spread* of drift across seeds (relative sd
53% vs 98%).

**NOT SUPPORTED -- that this drift is what makes the rank effect grow with
length.** Within `Vanilla`, drift spans a **16x range across seeds** (19.6 to
319.5) while accuracy spans 0.729-0.912, and the two do not track: r = -0.363
(n=8). Within `Vanilla_r4`, r = +0.086. The concrete counterexample is sharper
than the correlation: seed 0 has the *highest* drift in the arm (319.5) and an
*above-median* accuracy (0.869), while seed 2 has the lowest drift (19.6) and
essentially the same accuracy (0.864).

**The pooled r = -0.614 must not be cited.** It is computed across a two-level
factor, so it says only "r=4 has less drift and more accuracy" -- confounded
with every other difference between the arms. Pooling across the manipulation
and reporting the correlation is the shape of an argument, not one.

**Power caveat, stated so the null is not overread.** At n=8 a within-arm
correlation of -0.363 has a 95% interval running from about -0.83 to +0.42, so
this is UNMEASURED rather than refuted (rule 11). Distinguishing r = -0.6 from
zero needs roughly n=20. The counterexample above is what makes a real null
plausible; the correlation on its own does not.

**My pre-registered point prediction was wrong by 2x.** I predicted the drift
ratio would track the opposition errors alone, 0.495 / 0.092 = 5.4x. Measured
11.4x. The excess is plausibly the near-degeneracy of the r=2 basis (|cos(N,E)|
0.783 vs 0.175), which makes the displacement fit ill-conditioned and amplifies
any residual -- but that is an explanation offered after seeing the number and
has not been tested. The direction and the linearity were predicted; the
magnitude was not.

**What this leaves.** The length-dependence of the rank effect -- the one
signature common to all six condition-pairs measured -- is **still unexplained**.
A mechanism was proposed, made quantitative, and tested at its decisive link,
and the link did not hold. What survives is a cheap exact diagnostic that
separates the two ranks by 11x with no forward pass, and the knowledge that
whatever the accuracy mechanism is, it is not the accumulated state residual.
