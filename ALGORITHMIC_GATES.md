# Algorithmic tasks -- pre-flight gates (CPU, no training)

Every baseline column must sit at `chance`. A gate AT chance is a PASS.

| task | L | chance | marginal | ngram1 | ngram2 | ngram3 | ngram5 | echo-input | repeat-prev | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| parity | 16 | 0.5000 | 0.5027 | 0.4942 | 0.5050 | 0.5072 | 0.5018 | 0.5050 | 0.4923 | 6000 |
| parity | 32 | 0.5000 | 0.5001 | 0.5122 | 0.4882 | 0.4896 | 0.4993 | 0.5002 | 0.4923 | 12400 |
| parity | 64 | 0.5000 | 0.5021 | 0.4960 | 0.4960 | 0.5000 | 0.5031 | 0.4990 | 0.4978 | 25200 |
| copy | 16 | 0.1250 | 0.1313 | 0.1297 | 0.1223 | 0.1314 | 0.1224 | 0.1192 | 0.1263 | 6400 |
| copy | 32 | 0.1250 | 0.1301 | 0.1197 | 0.1272 | 0.1346 | 0.1289 | 0.1210 | 0.1241 | 12800 |
| copy | 64 | 0.1250 | 0.1277 | 0.1253 | 0.1200 | 0.1160 | 0.1256 | 0.1214 | 0.1235 | 25600 |

## Verdict

Largest excess of any trivial baseline over its own chance rate: **+0.0122**.

**PASS.** No trivial strategy beats chance by more than 0.02, so a trained model's score is not available for free.

Note on parity specifically: `repeat-prev` is the strategy to watch. The running parity changes only when the bit is 1, so predicting the previous answer is right half the time -- which is chance here, but would NOT be if the bit distribution were skewed.
