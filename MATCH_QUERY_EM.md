# EM on Match-Query — does single-p_0 help where the task is MATCHING?

All three arms trained in ONE batch, TE=512 TQ=256, 200 epochs, n=3.
Held-out env (seed=10000). **Chance 0.0625.** Gates: `MATCH_QUERY_GATES.md`.

Match-Query is the only task in this repo whose shortcuts are gated, and it
tests position MATCHING — which is what A_P is for. So it is the most
informative place to ask whether the q0/k0 parameterisation matters.

| variant | T_query=256 | T_query=512 (OOD) |
|---|---|---|
| MapWM-Flat (WM) | 0.888 ± 0.140 | 0.902 ± 0.117 |
| MapEM separate q0/k0 (paper-faithful) | 0.450 ± 0.332 | 0.385 ± 0.323 |
| MapEM single p_0 (ablation) | 0.808 ± 0.168 | 0.789 ± 0.188 |

## Per-seed (T_query=256)

| variant | s0 | s1 | s2 |
|---|---|---|---|
| MapWM-Flat (WM) | 0.731 | 1.000 | 0.934 |
| MapEM separate q0/k0 (paper-faithful) | 0.107 | 0.769 | 0.475 |
| MapEM single p_0 (ablation) | 0.736 | 1.000 | 0.689 |

Reference: the earlier 6-variant sweep gave Vanilla 0.888 ± 0.140 under
identical settings. The Vanilla arm here is the reproducibility check.
