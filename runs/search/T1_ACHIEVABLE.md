## T1 -- was a good rewind AVAILABLE? (single-p0 EM, k >= 8)

`Q` = best achievable weighted kernel at the answer, optimising z in the model's own rank-4 subspace over real episode samples. `A` = what the model achieves. 1.0 is a perfect wrapped rewind.

| cells | n | mean Q (available) | mean A (achieved) | gap |
|---|---|---|---|---|
| SOLVED (acc >= 0.9) | 166 | 0.995 | 0.005 | +0.990 |
| FAILED (acc <= 0.3) | 181 | 0.992 | -0.047 | +1.039 |

**Verdict: SEARCH -- the solution was available and not found.** Failed tokens had Q = 0.992 available and achieved A = -0.047.

**P1a (pruning is the strategy)**: r(dead-block fraction, tokens solved) = -0.546 over 8 seeds; dead fraction 0.59 +/- 0.12, solved 20.8 +/- 7.8 of 57.
