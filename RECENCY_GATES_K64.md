# Recency (k-back) task -- pre-flight gates (CPU, no training)

Retrieve the k-th most recent symbol. A MATCH on the time axis, not a decode -- `environment_map_query.py` asked for a decode and got 0.121 against chance 0.016.

n_symbols=16, k_max=64, p_query=0.25, 800 episodes per row. **chance = 1/16 = 0.0625**; the most-recent-symbol shortcut floor is 1/k_max + chance*(1-1/k_max) = 0.0771.

| min_gap | T | chance | marginal | o1 | o2 | o3 | o5 | most-recent | oracle | scored/token | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1024 | 0.0625 | 0.0645 | 0.0708 | 0.0660 | 0.0637 | 0.0632 | 0.0781 | 1.0000 | 0.067 | 55271 |
| 16 | 1024 | 0.0625 | 0.0666 | 0.0634 | 0.0630 | 0.0601 | 0.0668 | 0.0772 | 1.0000 | 0.024 | 19749 |
| 64 | 1024 | 0.0625 | 0.0663 | 0.0605 | 0.0704 | 0.0665 | 0.0641 | 0.0760 | 1.0000 | 0.007 | 5687 |

## G8 -- can a FIXED INDEX code address the answer?

The token distance from a query back to its answer, per offset k. If that distance were constant, `k` would just be a token offset and an index code could address it directly -- measured, and it does: with `p_filler = 0`, RoPE scores **1.000** at every `k_max` tried while signed and monotone TIE at 0.946, i.e. the task was index retrieval in disguise. Filler tokens are emitted into the stream but not counted, so the distance becomes a random variable and `k` becomes a CONTEXTUAL position in CoPE's sense (arXiv:2405.11582: relative PE "can do" no better than "a decaying attention"). With `p_filler = 0.5`, RoPE falls to **0.31-0.37**.

| min_gap | T | k=1 mean+/-sd | k=mid mean+/-sd | k=max mean+/-sd |
|---|---|---|---|---|
| 4 | 1024 | k=1: 2.6+/-0.9 | k=33: 76.9+/-8.8 | k=64: 148.9+/-12.3 |
| 16 | 1024 | k=1: 2.6+/-1.0 | k=33: 68.9+/-8.2 | k=64: 134.2+/-10.8 |
| 64 | 1024 | k=1: 2.5+/-0.9 | k=33: 66.3+/-7.9 | k=64: 128.3+/-11.8 |

## G7 -- does the accumulator difference address a unique key?

The mechanism the task is built to separate. `signed ambiguous` is the fraction of scored queries where some OTHER candidate key carries the same accumulator value AND a different symbol, so the difference `theta_query - theta_key` does not identify the answer.

| min_gap | T | signed ambiguous | mean keys sharing theta | monotone ambiguous |
|---|---|---|---|---|
| 4 | 1024 | **0.974** | 10.31 | 0.000 |
| 16 | 1024 | **0.977** | 11.12 | 0.000 |
| 64 | 1024 | **0.975** | 11.13 | 0.000 |

Monotone is 0.000 by construction (`theta = t` is injective), which is the point and not a measurement. The signed column is simulated as a +/-1 walk per token -- a proxy for *unconstrained and not learning a counter*. It is an upper bound on the difficulty, NOT a prediction about a trained signed model: the unconstrained arm is free to learn a monotone code, and whether it does is the second prediction, measured on trained models with `probe_sign.py`.

**Reading it.** Every baseline column must sit at `chance` except `most-recent`, which sits at its own stated floor, and `oracle`, which must be exactly 1.000. `o1`-`o5` above chance means the answer stream is self-predictable and the `min_gap` for that row is unusable.

## Verdict

A row PASSES when every n-gram order is within 0.01 of chance, `marginal` is within 0.01 of chance, `most-recent` is within 0.02 of its stated floor, and `oracle` is exactly 1.000.

| min_gap | T | n-gram | marginal | most-recent | oracle | verdict |
|---|---|---|---|---|---|---|
| 4 | 1024 | ok | ok | ok | ok | **PASS** |
| 16 | 1024 | ok | ok | ok | ok | **PASS** |
| 64 | 1024 | ok | ok | ok | ok | **PASS** |

3 of 3 rows pass. The failures are the low-`min_gap` diagnostic rows and are EXPECTED -- they are what establishes that the default (`min_gap = k_max`) is necessary rather than decorative.

