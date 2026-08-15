# Match-Query: blind query phase far past training length

Inference only on existing checkpoints. Trained at T_query=256. chance = 0.0625.

| variant | TQ=256 | TQ=512 | TQ=1024 | TQ=2048 |
|---|---|---|---|---|
| Vanilla | 0.904 ± 0.098 | 0.894 ± 0.125 | 0.831 ± 0.197 | 0.693 ± 0.282 |
| PlainFlat | 0.150 ± 0.037 | 0.118 ± 0.007 | 0.103 ± 0.013 | 0.093 ± 0.007 |
