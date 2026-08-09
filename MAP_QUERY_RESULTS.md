> **UNDERTRAINED -- NOT A RESULT (2026-08-09).** These runs used 25 epochs,
> a budget copied from the hier-goal config without checking it suited a harder
> task. A 200-epoch diagnostic shows the room term sits at chance until ~epoch 30
> at T_explore=256 and only then descends, so training stopped before anything
> happened. Do not read the chance-level rows as a negative result.
>
> Pipeline validity is separately confirmed: at T_explore=16 the SAME code reaches
> held-out direction 0.969 and room 0.994 (chance 0.50 / 0.016).

# Map-Query results (n=3 seeds)

Trained at T_explore=256 -- the minimum operating point the gates validate.
Held-out env (seed=10000). Gates in `MAP_QUERY_GATES.md`.

**Chance: direction ~0.50, room 0.016.** Oracle 1.000 for both.

## Room identity (the cognitive-map metric, 64 classes)

| variant | T=256 | T=512 | T=1024 |
|---|---|---|---|
| MapWM-Flat | 0.027 ± 0.002 | 0.003 ± 0.003 | 0.005 ± 0.001 |
| MapWM-Hier | 0.018 ± 0.007 | 0.008 ± 0.010 | 0.012 ± 0.004 |
| Plain-Flat | 0.021 ± 0.012 | 0.025 ± 0.022 | 0.015 ± 0.007 |
| Plain-Hier | 0.017 ± 0.007 | 0.029 ± 0.009 | 0.014 ± 0.003 |
| PoPE-Flat | 0.020 ± 0.013 | 0.020 ± 0.020 | 0.012 ± 0.006 |
| MapPoPE-Hier | 0.013 ± 0.005 | 0.018 ± 0.011 | 0.010 ± 0.003 |

## Goal direction (chance ~0.50)

| variant | T=256 | T=512 | T=1024 |
|---|---|---|---|
| MapWM-Flat | 0.523 ± 0.007 | 0.515 ± 0.024 | 0.494 ± 0.037 |
| MapWM-Hier | 0.513 ± 0.019 | 0.495 ± 0.010 | 0.504 ± 0.007 |
| Plain-Flat | 0.498 ± 0.012 | 0.499 ± 0.016 | 0.516 ± 0.011 |
| Plain-Hier | 0.506 ± 0.005 | 0.507 ± 0.038 | 0.499 ± 0.010 |
| PoPE-Flat | 0.504 ± 0.016 | 0.504 ± 0.006 | 0.505 ± 0.019 |
| MapPoPE-Hier | 0.514 ± 0.025 | 0.493 ± 0.006 | 0.498 ± 0.004 |
