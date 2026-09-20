# Revisit accuracy by recurrence interval

Index-position models beat the marginal in LIKELIHOOD (train loss 1.59-1.68 vs 2.079 nats) while sitting at it in ACCURACY (0.513 vs a 0.506 blank floor). This asks where that likelihood goes.

Recurrence interval = steps since the cell was last visited. The paper's walk is directed with run lengths 1-10, so an out-and-back run retraces cells a few steps later -- detectable from the ACTION TOKENS AS CONTENT, with no position code. If that is the source, index models win only in the leftmost buckets.

| variant | 1-2 | 3-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65+ |
|---|---|---|---|---|---|---|---|
| RoPELooped | 1.000 | 1.000 | 0.999 | 0.992 | 0.855 | 0.512 | 0.481 |
| Looped | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.989 |
| *blank rate (floor)* | *0.498* | *0.514* | *0.486* | *0.518* | *0.522* | *0.497* | *0.499* |
| *n per seed* |  |  |  |  |  |  |  |
