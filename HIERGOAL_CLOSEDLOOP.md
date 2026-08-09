# Hierarchical goal nav — CLOSED-LOOP success (model drives its own rollout)

Fixed start, held-out env. Success = reached target cell within T_navigate=96 steps. n_trials=200, seeds=[0, 1, 2].

| variant | T_exp=64 | T_exp=128 | T_exp=256 |
|---|---|---|---|
| MapWM-Flat | 0.020 ± 0.000 | 0.022 ± 0.006 | 0.018 ± 0.006 |
| MapWM-Hier | 0.022 ± 0.005 | 0.037 ± 0.006 | 0.020 ± 0.004 |
| Plain-Flat | 0.027 ± 0.006 | 0.020 ± 0.004 | 0.020 ± 0.007 |
| Plain-Hier | 0.025 ± 0.004 | 0.033 ± 0.012 | 0.022 ± 0.002 |
| PoPE-Flat | 0.017 ± 0.006 | 0.013 ± 0.002 | 0.017 ± 0.008 |
| MapPoPE-Hier | 0.018 ± 0.008 | 0.022 ± 0.006 | 0.022 ± 0.005 |
