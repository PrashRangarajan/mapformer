# Revisit accuracy by recurrence interval

Index-position models beat the marginal in LIKELIHOOD (train loss 1.59-1.68 vs 2.079 nats) while sitting at it in ACCURACY (0.513 vs a 0.506 blank floor). This asks where that likelihood goes.

Recurrence interval = steps since the cell was last visited. The paper's walk is directed with run lengths 1-10, so an out-and-back run retraces cells a few steps later -- detectable from the ACTION TOKENS AS CONTENT, with no position code. If that is the source, index models win only in the leftmost buckets.

| variant | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ |
|---|---|---|---|---|---|---|---|
| RoPE | 0.563 | 0.521 | 0.503 | 0.504 | 0.507 | 0.485 | 0.488 |
| Vanilla | 0.994 | 0.996 | 0.997 | 0.994 | 0.977 | 0.935 | 0.880 |
| *blank rate (floor)* | *0.504* | *0.511* | *0.497* | *0.508* | *0.503* | *0.539* | *0.477* |
| *n per seed* |  |  |  |  |  |  |  |
