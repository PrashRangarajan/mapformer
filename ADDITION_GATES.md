# Addition task gates (`validate_addition.py`, run on the task code, before any training)

Balanced sampling over 1..16 digits per operand; zero-padded; sum reversed. Per-digit accuracy of
predictors that see only the SUM stream, on 4,000 held-out problems (trained on 20,000):

| predictor | per-digit accuracy |
|---|---|
| order-1 n-gram on sum digits | 0.165 |
| order-2 | 0.165 |
| order-3 | 0.161 |
| order-4 | 0.122 |
| order-5 | 0.092 |
| majority digit (always 0) | 0.163 |
| uniform chance | 0.100 |
| reference: (a_j + b_j) mod 10, no carry (a partial algorithm, not a shortcut) | 0.759 per digit, 0.133 exact |

**PASS.** No sum-stream predictor beats the majority-digit rate. The marginal sits above 0.1 because
zero-padding and the top carry digit make 0 common. Exact-match chance is about 10^-(n+1). Note that
per-digit accuracy is a weak metric: the no-carry reference already gets 0.759. Exact match is the
primary metric.
