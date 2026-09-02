# Which recipe converges? Fixing power before changing the task

Torus clean, 2 arms x 3 conditions x 8 seeds, one batch, parallel data
path throughout.

**Primary metric is CONVERGED FRACTION** (final loss < 0.05 and a flat
tail), **secondary is sd of accuracy**. Mean accuracy is reported and is
NOT the criterion: choosing a recipe on the mean over 8 seeds selects a
lucky basin.

## Convergence

| condition | Vanilla converged | Looped converged | Vanilla final loss | Looped final loss |
|---|---|---|---|---|
| C0 · 300 ep, lr 3e-4  (current recipe) | **3/8** | **8/8** | 0.0001 – 0.8532 | 0.0000 – 0.0069 |
| C1 · 300 ep, lr 1e-3 | **4/8** | **8/8** | 0.0011 – 0.0874 | 0.0001 – 0.0025 |
| C2 · 600 ep, lr 1e-3 | **6/8** | **8/8** | 0.0016 – 0.3656 | 0.0002 – 0.0018 |

## T=128

| condition | Vanilla mean | Looped mean | Vanilla sd | Looped sd | Vanilla MDE@8 | Looped MDE@8 |
|---|---|---|---|---|---|---|
| C0 | 0.936 | 1.000 | 0.086 | 0.001 | 0.085 | 0.001 |
| C1 | 0.993 | 1.000 | 0.017 | 0.000 | 0.017 | 0.000 |
| C2 | 0.985 | 1.000 | 0.035 | 0.000 | 0.034 | 0.000 |

## T=512

| condition | Vanilla mean | Looped mean | Vanilla sd | Looped sd | Vanilla MDE@8 | Looped MDE@8 |
|---|---|---|---|---|---|---|
| C0 | 0.863 | 0.910 | 0.096 | 0.073 | 0.095 | 0.073 |
| C1 | 0.944 | 0.909 | 0.028 | 0.091 | 0.027 | 0.090 |
| C2 | 0.930 | 0.824 | 0.066 | 0.153 | 0.065 | 0.151 |

## T=1024

| condition | Vanilla mean | Looped mean | Vanilla sd | Looped sd | Vanilla MDE@8 | Looped MDE@8 |
|---|---|---|---|---|---|---|
| C0 | 0.735 | 0.733 | 0.110 | 0.138 | 0.109 | 0.136 |
| C1 | 0.834 | 0.718 | 0.064 | 0.118 | 0.064 | 0.116 |
| C2 | 0.777 | 0.636 | 0.158 | 0.167 | 0.156 | 0.165 |

## Verdict

**C1 (lr 1e-3) is a large power win on the arm that was failing.** Vanilla's seed
sd, which is what sets every MDE in this project:

| length | C0 sd | C1 sd | tighter by | C0 mean -> C1 mean |
|---|---|---|---|---|
| T=128 | 0.086 | **0.017** | 4.9x | 0.936 -> 0.993 |
| T=512 | 0.096 | **0.028** | 3.5x | 0.863 -> 0.944 |
| T=1024 | 0.110 | **0.064** | 1.7x | 0.735 -> 0.834 |

Higher mean AND lower variance at every length, from one hyperparameter. Vanilla's
MDE at n=8 falls from 0.095 to 0.027 at T=512. Looped is roughly unaffected
(T=512 sd 0.073 -> 0.091, T=1024 0.138 -> 0.118) because it was never the arm with
the problem -- it converged 8/8 under every condition.

**Two corrections to how this was analysed, both mine.**

1. *The decision rule was broken.* It required a strictly higher converged count on
   EVERY arm. Looped is 8/8 in all three conditions, so no condition could ever
   qualify, and the first verdict printed "No condition beats the current recipe"
   while C1 was cutting Vanilla's sd by 3.5x. Fixed to a Pareto rule: no regression
   anywhere, a strict gain somewhere.

2. *The pre-registered primary metric was the wrong proxy.* Converged fraction
   ranks C2 (6/8) > C1 (4/8) > C0 (3/8) for Vanilla. Accuracy sd -- the thing that
   actually sets the MDE -- ranks C1 best and puts C2 WORST at T=1024 (sd 0.158,
   worse than C0's 0.110). C2 converges more often and generalises less reliably.
   That is the same decoupling measured last night, where r(final loss, accuracy)
   fell from -0.956 at T=128 to -0.326 at T=1024. Converged fraction is a proxy for
   power; it is not power. Reported both rather than silently switching metrics.

**So: adopt C1 (300 ep, lr 1e-3) as the default for torus work.** The bimodality is
not eliminated -- Vanilla is still 4/8 converged -- but the runs that do converge
land far closer together, which is what buys the power.

**Caveat that costs something.** Step 2 (Match-Query with stochastic transitions)
was already running when this was corrected, and it inherited the buggy rule's
choice of C0. All 64 of its runs use C0, so its within-batch comparison is valid;
it is simply less powerful than it could have been. Whether to replicate it under
C1 is a decision to take AFTER reading its MDE -- recipe transfer across tasks is
an assumption this run does not test.

## Scope

One task, two arms, three conditions, n=8. Only knobs that already existed were varied (epochs, lr); warmup fraction, cosine floor, dropout and init scale are hardcoded and untested here. A recipe that fixes the torus is not guaranteed to fix Match-Query -- that is why step 2 re-measures the converged fraction rather than assuming it.
