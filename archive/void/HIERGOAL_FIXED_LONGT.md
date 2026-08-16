> **INVALIDATED (2026-08-09)** -- see `HIERGOAL_ABLATION.md`. The task is
> solvable from the navigate action prefix alone: randomising the goal AND
> the entire explore phase leaves accuracy unchanged. An order-3 Markov model
> on actions alone scores 0.971 (interleaved) / 0.969 (raw BFS). No number in
> this file measures navigation.

# Hier-goal at LONG OOD explore length (existing checkpoints, inference only)

Trained at T_explore=64. n_trials=100, seeds=[0, 1, 2].

| variant | T=256 | T=512 | T=1024 | T=2048 |
|---|---|---|---|---|
| MapWM-Flat | 0.892 ± 0.030 | 0.874 ± 0.060 | 0.877 ± 0.048 | 0.822 ± 0.093 |
| MapWM-Hier | 0.780 ± 0.031 | 0.748 ± 0.057 | 0.709 ± 0.053 | 0.626 ± 0.095 |
| Plain-Flat | 0.911 ± 0.015 | 0.902 ± 0.022 | 0.864 ± 0.045 | 0.795 ± 0.099 |
| Plain-Hier | 0.663 ± 0.060 | 0.622 ± 0.044 | 0.632 ± 0.025 | 0.590 ± 0.050 |
| PoPE-Flat | 0.939 ± 0.002 | 0.938 ± 0.004 | 0.940 ± 0.003 | 0.936 ± 0.005 |
| MapPoPE-Hier | 0.899 ± 0.012 | 0.753 ± 0.067 | 0.667 ± 0.003 | 0.642 ± 0.004 |
