# Match-Query scale-up (n=3, n=5 on base)

Vanilla = path integration; PlainFlat = index position. Gates re-run per config.
**Chance is 0.0625 at n_obs=16 and 0.2500 at n_obs=4.**

## base: 64^2, n_obs=16  (chance 0.0625)

| variant | TQ=256 | TQ=512 | TQ=1024 |
|---|---|---|---|
| Vanilla | 0.730 ± 0.247 | 0.739 ± 0.253 | 0.352 ± 0.077 |
| PlainFlat | 0.154 ± 0.018 | 0.125 ± 0.009 | 0.107 ± 0.006 |

## big: 128^2, n_obs=16 -- 4x the map  (chance 0.0625)

| variant | TQ=256 | TQ=512 | TQ=1024 |
|---|---|---|---|
| Vanilla | 0.823 ± 0.043 | 0.747 ± 0.064 | 0.720 ± 0.075 |
| PlainFlat | 0.192 ± 0.022 | 0.164 ± 0.025 | 0.150 ± 0.011 |

## alias: 64^2, n_obs=4 -- heavy aliasing  (chance 0.2500)

| variant | TQ=256 | TQ=512 | TQ=1024 |
|---|---|---|---|
| Vanilla | 0.510 ± 0.187 | 0.458 ± 0.180 | 0.393 ± 0.152 |
| PlainFlat | 0.332 ± 0.012 | 0.295 ± 0.012 | 0.274 ± 0.007 |

