# Are the filter and the loop complementary?

The 2x2 that had never been run. Clean torus paper task, 5 arms x 12 seeds,
300 epochs warmup+cosine, held-out map, all arms trained in ONE batch.

`Level15Looped` is verified to be bit-identical to `Level15` at n_loops=1
(max|diff| 0.000e+00) and causal (leak 0.00e+00). The loop adds 0 parameters
on both rows and the filter adds exactly 49,600 on both, so the interaction
is parameter-matched; only the filter MAIN EFFECT carries a capacity gap.

## Accuracy (mean +/- sd)

| arm | params | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| Vanilla | 204,373 | 0.947 +/- 0.062 | 0.876 +/- 0.071 | 0.749 +/- 0.084 |
| Level15 | 253,973 | 0.990 +/- 0.030 | 0.953 +/- 0.056 | 0.878 +/- 0.092 |
| Looped | 204,373 | 0.999 +/- 0.004 | 0.872 +/- 0.119 | 0.730 +/- 0.166 |
| Level15Looped | 253,973 | 0.994 +/- 0.014 | 0.929 +/- 0.089 | 0.830 +/- 0.168 |
| LoopedSampled | 204,373 | 0.997 +/- 0.007 | 0.905 +/- 0.066 | 0.745 +/- 0.120 |

## Convergence (rule 9)

| arm | mean final loss | per-seed range |
|---|---|---|
| Vanilla | 0.1549 | 0.0087 – 0.6099 |
| Level15 | 0.0420 | 0.0001 – 0.4198 |
| Looped | 0.0076 | 0.0000 – 0.0557 |
| Level15Looped | 0.0189 | 0.0000 – 0.1147 |
| LoopedSampled | 0.0180 | 0.0000 – 0.1120 |

- r(final loss, accuracy) at T=128: **-0.956** over 60 runs
- r(final loss, accuracy) at T=512: **-0.471** over 60 runs
- r(final loss, accuracy) at T=1024: **-0.326** over 60 runs

## Contrasts

Positive = first arm higher. DETECTABLE means |mean| > MDE = 2.8*sd/sqrt(n).

### T=128

| contrast | raw | loss-matched | |
|---|---|---|---|
| Looped - Vanilla <br><sub>loop main effect (no filter)</sub> | +0.052 (sd 0.060, t +3.03, MDE 0.048, 12/12) | +0.006 (sd 0.021, t +0.99, MDE 0.017, 6/12) | UNMEASURED |
| Level15 - Vanilla <br><sub>filter main effect (no loop)</sub> | +0.044 (sd 0.075, t +2.03, MDE 0.060, 11/12) | +0.008 (sd 0.022, t +1.34, MDE 0.018, 6/12) | UNMEASURED |
| Level15Looped - Looped <br><sub>filter, INSIDE the loop</sub> | -0.004 (sd 0.010, t -1.53, MDE 0.008, 1/12) | -0.001 (sd 0.003, t -1.11, MDE 0.002, 5/12) | UNMEASURED |
| Level15Looped - Level15 <br><sub>loop, INSIDE the filter</sub> | +0.004 (sd 0.034, t +0.39, MDE 0.027, 4/12) | -0.003 (sd 0.008, t -1.51, MDE 0.006, 3/12) | UNMEASURED |
| Level15Looped - LoopedSampled <br><sub>the filter vs the free fix</sub> | -0.002 (sd 0.008, t -1.06, MDE 0.006, 3/12) | -0.002 (sd 0.005, t -1.31, MDE 0.004, 5/12) | UNMEASURED |
| LoopedSampled - Looped <br><sub>sampling the count (reference)</sub> | -0.002 (sd 0.006, t -1.30, MDE 0.005, 1/12) | +0.001 (sd 0.004, t +1.02, MDE 0.003, 6/12) | UNMEASURED |
| **INTERACTION** <br><sub>(L15Loop-Loop)-(L15-Vanilla)</sub> | **-0.048 (sd 0.079, t -2.12, MDE 0.064, 1/12)** | **-0.009 (sd 0.021, t -1.58, MDE 0.017, 5/12)** | **UNMEASURED** |

### T=512

| contrast | raw | loss-matched | |
|---|---|---|---|
| Looped - Vanilla <br><sub>loop main effect (no filter)</sub> | -0.004 (sd 0.134, t -0.10, MDE 0.108, 7/12) | -0.058 (sd 0.121, t -1.65, MDE 0.098, 6/12) | UNMEASURED |
| Level15 - Vanilla <br><sub>filter main effect (no loop)</sub> | +0.077 (sd 0.096, t +2.79, MDE 0.077, 11/12) | +0.036 (sd 0.053, t +2.33, MDE 0.043, 10/12) | UNMEASURED |
| Level15Looped - Looped <br><sub>filter, INSIDE the loop</sub> | +0.058 (sd 0.140, t +1.42, MDE 0.113, 7/12) | +0.062 (sd 0.135, t +1.58, MDE 0.109, 7/12) | UNMEASURED |
| Level15Looped - Level15 <br><sub>loop, INSIDE the filter</sub> | -0.023 (sd 0.092, t -0.88, MDE 0.074, 4/12) | -0.032 (sd 0.063, t -1.77, MDE 0.051, 4/12) | UNMEASURED |
| Level15Looped - LoopedSampled <br><sub>the filter vs the free fix</sub> | +0.024 (sd 0.101, t +0.83, MDE 0.082, 7/12) | +0.025 (sd 0.099, t +0.86, MDE 0.080, 7/12) | UNMEASURED |
| LoopedSampled - Looped <br><sub>sampling the count (reference)</sub> | +0.033 (sd 0.135, t +0.85, MDE 0.109, 7/12) | +0.037 (sd 0.133, t +0.96, MDE 0.107, 7/12) | UNMEASURED |
| **INTERACTION** <br><sub>(L15Loop-Loop)-(L15-Vanilla)</sub> | **-0.020 (sd 0.169, t -0.40, MDE 0.136, 5/12)** | **+0.026 (sd 0.123, t +0.73, MDE 0.099, 5/12)** | **UNMEASURED** |

### T=1024

| contrast | raw | loss-matched | |
|---|---|---|---|
| Looped - Vanilla <br><sub>loop main effect (no filter)</sub> | -0.020 (sd 0.209, t -0.33, MDE 0.169, 5/12) | -0.080 (sd 0.200, t -1.39, MDE 0.162, 4/12) | UNMEASURED |
| Level15 - Vanilla <br><sub>filter main effect (no loop)</sub> | +0.129 (sd 0.127, t +3.51, MDE 0.103, 10/12) | +0.083 (sd 0.102, t +2.79, MDE 0.083, 9/12) | UNMEASURED |
| Level15Looped - Looped <br><sub>filter, INSIDE the loop</sub> | +0.100 (sd 0.202, t +1.72, MDE 0.163, 7/12) | +0.105 (sd 0.196, t +1.85, MDE 0.159, 7/12) | UNMEASURED |
| Level15Looped - Level15 <br><sub>loop, INSIDE the filter</sub> | -0.048 (sd 0.163, t -1.03, MDE 0.132, 4/12) | -0.058 (sd 0.139, t -1.44, MDE 0.112, 4/12) | UNMEASURED |
| Level15Looped - LoopedSampled <br><sub>the filter vs the free fix</sub> | +0.085 (sd 0.216, t +1.36, MDE 0.175, 8/12) | +0.085 (sd 0.214, t +1.37, MDE 0.173, 8/12) | UNMEASURED |
| LoopedSampled - Looped <br><sub>sampling the count (reference)</sub> | +0.015 (sd 0.182, t +0.29, MDE 0.147, 6/12) | +0.020 (sd 0.180, t +0.38, MDE 0.146, 7/12) | UNMEASURED |
| **INTERACTION** <br><sub>(L15Loop-Loop)-(L15-Vanilla)</sub> | **-0.029 (sd 0.236, t -0.42, MDE 0.190, 7/12)** | **+0.022 (sd 0.182, t +0.42, MDE 0.147, 7/12)** | **UNMEASURED** |

## Verdict against the pre-registration

**T=512. Interaction +0.026 (loss-matched, MDE 0.099); raw -0.020 (MDE 0.136). -> UNMEASURED.**

NOT super-additive. Levels at T=512: Level15Looped 0.929, Level15 0.953, Looped 0.872. The combination is WORSE than the better single mechanism, so they are not complementary in any useful sense. Per rule 11 an interaction inside its MDE is UNMEASURED, not zero.

**T=1024. Interaction +0.022 (loss-matched, MDE 0.147); raw -0.029 (MDE 0.190). -> UNMEASURED.**

NOT super-additive. Levels at T=1024: Level15Looped 0.830, Level15 0.878, Looped 0.730. The combination is WORSE than the better single mechanism, so they are not complementary in any useful sense. Per rule 11 an interaction inside its MDE is UNMEASURED, not zero.

**The filter vs the free fix at T=1024**: Level15Looped - LoopedSampled +0.085 (sd 0.214, t +1.37, MDE 0.173, 8/12) -> UNMEASURED. Sampling the loop count costs 0 parameters; the filter costs 49,600. A filter that only matches sampling is not the answer to the loop's length problem.

## Scope

Clean config only, n=12, one task, one width, one loop count (4). Noise was
rejected as a condition here: at p=0.25 every arm sits within 0.11 of the
0.500 blank floor at T=512, so an OOD interaction would be floor-compressed
exactly where it needs measuring. The loss-matched analysis is a regression
control, not a randomised one.

One caveat on the reference arm: every looped model here is evaluated at 4
passes, which is the consistent choice for the 2x2 but is NOT LoopedSampled's
best count. Its own sweep peaks at 2 passes out of distribution (0.915 vs
0.898 at T=512, 0.736 vs 0.719 at T=1024), so the free fix is understated by
roughly 0.017 in the `Level15Looped - LoopedSampled` row. Small against the
MDEs above, but it runs AGAINST sampling, so read that row as a lower bound
on the free fix rather than a fair point estimate.
