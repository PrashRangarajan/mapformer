# Recency (k-back) task -- pre-flight gates (CPU, no training)

Retrieve the k-th most recent symbol. A MATCH on the time axis, not a decode -- `environment_map_query.py` asked for a decode and got 0.121 against chance 0.016.

n_symbols=16, k_max=8, p_query=0.25, 200 episodes per row. **chance = 1/16 = 0.0625**; the most-recent-symbol shortcut floor is 1/k_max + chance*(1-1/k_max) = 0.1797.

| min_gap | T | chance | marginal | o1 | o2 | o3 | o5 | most-recent | oracle | scored/token | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 256 | 0.0625 | 0.0663 | 0.1494 | 0.0964 | 0.0718 | 0.0672 | 0.1791 | 1.0000 | 0.194 | 9922 |
| 0 | 512 | 0.0625 | 0.0661 | 0.1297 | 0.1051 | 0.0721 | 0.0619 | 0.1802 | 1.0000 | 0.196 | 20109 |
| 0 | 1024 | 0.0625 | 0.0647 | 0.1422 | 0.1221 | 0.0775 | 0.0635 | 0.1777 | 1.0000 | 0.199 | 40700 |
| 0 | 2048 | 0.0625 | 0.0644 | 0.1399 | 0.1321 | 0.0858 | 0.0662 | 0.1793 | 1.0000 | 0.198 | 81288 |
| 1 | 256 | 0.0625 | 0.0708 | 0.1233 | 0.0721 | 0.0682 | 0.0763 | 0.1731 | 1.0000 | 0.161 | 8243 |
| 1 | 512 | 0.0625 | 0.0672 | 0.1246 | 0.0847 | 0.0673 | 0.0629 | 0.1765 | 1.0000 | 0.163 | 16642 |
| 1 | 1024 | 0.0625 | 0.0648 | 0.1245 | 0.0988 | 0.0681 | 0.0634 | 0.1770 | 1.0000 | 0.165 | 33874 |
| 1 | 2048 | 0.0625 | 0.0642 | 0.1276 | 0.1154 | 0.0723 | 0.0630 | 0.1803 | 1.0000 | 0.166 | 67933 |
| 4 | 256 | 0.0625 | 0.0701 | 0.0799 | 0.0647 | 0.0705 | 0.0724 | 0.1768 | 1.0000 | 0.108 | 5510 |
| 4 | 512 | 0.0625 | 0.0684 | 0.0766 | 0.0673 | 0.0591 | 0.0643 | 0.1787 | 1.0000 | 0.110 | 11236 |
| 4 | 1024 | 0.0625 | 0.0657 | 0.0770 | 0.0646 | 0.0636 | 0.0670 | 0.1740 | 1.0000 | 0.111 | 22650 |
| 4 | 2048 | 0.0625 | 0.0644 | 0.0885 | 0.0663 | 0.0642 | 0.0629 | 0.1785 | 1.0000 | 0.111 | 45415 |
| 8 | 256 | 0.0625 | 0.0690 | 0.0610 | 0.0636 | 0.0704 | 0.0715 | 0.1845 | 1.0000 | 0.076 | 3870 |
| 8 | 512 | 0.0625 | 0.0677 | 0.0673 | 0.0699 | 0.0669 | 0.0713 | 0.1833 | 1.0000 | 0.076 | 7753 |
| 8 | 1024 | 0.0625 | 0.0657 | 0.0612 | 0.0627 | 0.0588 | 0.0624 | 0.1777 | 1.0000 | 0.077 | 15675 |
| 8 | 2048 | 0.0625 | 0.0641 | 0.0625 | 0.0616 | 0.0621 | 0.0648 | 0.1798 | 1.0000 | 0.077 | 31445 |

## G7 -- does the accumulator difference address a unique key?

The mechanism the task is built to separate. `signed ambiguous` is the fraction of scored queries where some OTHER candidate key carries the same accumulator value AND a different symbol, so the difference `theta_query - theta_key` does not identify the answer.

| min_gap | T | signed ambiguous | mean keys sharing theta | monotone ambiguous |
|---|---|---|---|---|
| 0 | 256 | **0.906** | 6.06 | 0.000 |
| 0 | 512 | **0.926** | 8.07 | 0.000 |
| 0 | 1024 | **0.952** | 11.29 | 0.000 |
| 0 | 2048 | **0.962** | 15.53 | 0.000 |
| 1 | 256 | **0.916** | 6.55 | 0.000 |
| 1 | 512 | **0.937** | 8.86 | 0.000 |
| 1 | 1024 | **0.955** | 12.11 | 0.000 |
| 1 | 2048 | **0.966** | 16.55 | 0.000 |
| 4 | 256 | **0.931** | 7.35 | 0.000 |
| 4 | 512 | **0.950** | 10.12 | 0.000 |
| 4 | 1024 | **0.964** | 13.90 | 0.000 |
| 4 | 2048 | **0.973** | 19.45 | 0.000 |
| 8 | 256 | **0.934** | 7.82 | 0.000 |
| 8 | 512 | **0.952** | 10.86 | 0.000 |
| 8 | 1024 | **0.970** | 15.63 | 0.000 |
| 8 | 2048 | **0.977** | 21.30 | 0.000 |

Monotone is 0.000 by construction (`theta = t` is injective), which is the point and not a measurement. The signed column is simulated as a +/-1 walk per token -- a proxy for *unconstrained and not learning a counter*. It is an upper bound on the difficulty, NOT a prediction about a trained signed model: the unconstrained arm is free to learn a monotone code, and whether it does is the second prediction, measured on trained models with `probe_sign.py`.

**Reading it.** Every baseline column must sit at `chance` except `most-recent`, which sits at its own stated floor, and `oracle`, which must be exactly 1.000. `o1`-`o5` above chance means the answer stream is self-predictable and the `min_gap` for that row is unusable.

## Verdict

A row PASSES when every n-gram order is within 0.01 of chance, `marginal` is within 0.01 of chance, `most-recent` is within 0.02 of its stated floor, and `oracle` is exactly 1.000.

| min_gap | T | n-gram | marginal | most-recent | oracle | verdict |
|---|---|---|---|---|---|---|
| 0 | 256 | **FAIL** o1=0.1494 | ok | ok | ok | fail |
| 0 | 512 | **FAIL** o1=0.1297 | ok | ok | ok | fail |
| 0 | 1024 | **FAIL** o1=0.1422 | ok | ok | ok | fail |
| 0 | 2048 | **FAIL** o1=0.1399 | ok | ok | ok | fail |
| 1 | 256 | **FAIL** o1=0.1233 | ok | ok | ok | fail |
| 1 | 512 | **FAIL** o1=0.1246 | ok | ok | ok | fail |
| 1 | 1024 | **FAIL** o1=0.1245 | ok | ok | ok | fail |
| 1 | 2048 | **FAIL** o1=0.1276 | ok | ok | ok | fail |
| 4 | 256 | **FAIL** o1=0.0799 | ok | ok | ok | fail |
| 4 | 512 | **FAIL** o1=0.0766 | ok | ok | ok | fail |
| 4 | 1024 | **FAIL** o1=0.0770 | ok | ok | ok | fail |
| 4 | 2048 | **FAIL** o1=0.0885 | ok | ok | ok | fail |
| 8 | 256 | ok | ok | ok | ok | **PASS** |
| 8 | 512 | ok | ok | ok | ok | **PASS** |
| 8 | 1024 | ok | ok | ok | ok | **PASS** |
| 8 | 2048 | ok | ok | ok | ok | **PASS** |

4 of 16 rows pass. The failures are the low-`min_gap` diagnostic rows and are EXPECTED -- they are what establishes that the default (`min_gap = k_max`) is necessary rather than decorative.

