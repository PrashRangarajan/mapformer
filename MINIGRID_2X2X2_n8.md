# The torus 2x2 on MiniGrid-DoorKey-16x16 (n=3)

External, published benchmark; egocentric observation (the cell in front of the agent). All four arms trained in ONE batch (rule 3), 50 epochs, 25K cached trajectory buffer.

**Measured floor** (most common scored target) per length: T=128: **0.635**, T=512: **0.536**, T=1024: **0.490**. The original n=1 evaluation reported no floor at all.

| model | position | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| MapWM-Flat (RoPE + path int.) | path-integrated | 0.995 ± 0.004 | 0.902 ± 0.058 | 0.823 ± 0.088 |
| MapWM-Hier (RoPE + path int. + hier) | path-integrated | 0.995 ± 0.004 | 0.945 ± 0.014 | 0.893 ± 0.023 |
| MapPoPE-Flat (PoPE + path int.) | path-integrated | 0.994 ± 0.002 | 0.959 ± 0.012 | 0.919 ± 0.026 |
| MapPoPE-Hier (PoPE + path int. + hier) | path-integrated | 0.993 ± 0.004 | 0.966 ± 0.009 | 0.942 ± 0.017 |
| RoPE-Flat (index) | **index** | 0.978 ± 0.005 | 0.914 ± 0.019 | 0.827 ± 0.044 |
| RoPE-Hier (index + hier) | **index** | 0.986 ± 0.003 | 0.950 ± 0.004 | 0.924 ± 0.006 |
| PoPE-Flat (PoPE + index) | **index** | 0.981 ± 0.003 | 0.963 ± 0.002 | 0.953 ± 0.003 |

| *measured floor* | | *0.635* | *0.536* | *0.490* |

## The torus comparison

For reference, the identical 2x2 on the 64x64 torus paper task at n=8 (`INDEX_BASELINE_PAPER_TASK_n8.md`), floor 0.506:

| | index | path-integrated |
|---|---|---|
| RoPE encoding | 0.530 | **0.967** |
| PoPE encoding | 0.509 | **0.994** |

Both index arms sit ON the floor there. Whether that survives on an egocentric, rotation-actioned environment is what this file measures.
