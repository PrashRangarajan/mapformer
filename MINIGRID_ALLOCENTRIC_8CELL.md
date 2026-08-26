# The torus 2x2 on MiniGrid-DoorKey-16x16 (n=3)

External, published benchmark; egocentric observation (the cell in front of the agent). All four arms trained in ONE batch (rule 3), 50 epochs, 25K cached trajectory buffer.

**Measured floor** (most common scored target) per length: T=128: **0.642**, T=512: **0.536**, T=1024: **0.495**. The original n=1 evaluation reported no floor at all.

| model | position | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| MapWM-Flat (RoPE + path int.) | path-integrated | 0.874 ± 0.008 | 0.833 ± 0.009 | 0.809 ± 0.010 |
| MapWM-Hier (RoPE + path int. + hier) | path-integrated | 0.876 ± 0.009 | 0.835 ± 0.014 | 0.820 ± 0.023 |
| MapPoPE-Flat (PoPE + path int.) | path-integrated | 0.873 ± 0.011 | 0.833 ± 0.013 | 0.818 ± 0.016 |
| MapPoPE-Hier (PoPE + path int. + hier) | path-integrated | 0.877 ± 0.008 | 0.840 ± 0.010 | 0.825 ± 0.013 |
| RoPE-Flat (index) | **index** | 0.873 ± 0.007 | 0.811 ± 0.009 | 0.781 ± 0.024 |
| RoPE-Hier (index + hier) | **index** | 0.874 ± 0.006 | 0.818 ± 0.007 | 0.788 ± 0.023 |
| PoPE-Flat (PoPE + index) | **index** | 0.871 ± 0.006 | 0.828 ± 0.003 | 0.807 ± 0.003 |
| PoPE-Hier (PoPE + index + hier) | **index** | 0.873 ± 0.007 | 0.831 ± 0.005 | 0.817 ± 0.010 |

| *measured floor* | | *0.642* | *0.536* | *0.495* |

## The torus comparison

For reference, the identical 2x2 on the 64x64 torus paper task at n=8 (`INDEX_BASELINE_PAPER_TASK_n8.md`), floor 0.506:

| | index | path-integrated |
|---|---|---|
| RoPE encoding | 0.530 | **0.967** |
| PoPE encoding | 0.509 | **0.994** |

Both index arms sit ON the floor there. Whether that survives on an egocentric, rotation-actioned environment is what this file measures.
