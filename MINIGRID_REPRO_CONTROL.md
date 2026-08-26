# The torus 2x2 on MiniGrid-DoorKey-16x16 (n=3)

External, published benchmark; egocentric observation (the cell in front of the agent). All four arms trained in ONE batch (rule 3), 50 epochs, 25K cached trajectory buffer.

**Measured floor** (most common scored target) per length: T=128: **0.642**, T=512: **0.536**, T=1024: **0.495**. The original n=1 evaluation reported no floor at all.

| model | position | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| PoPE-Flat (PoPE + index) | **index** | 0.871 ± 0.006 | 0.828 ± 0.003 | 0.807 ± 0.003 |

| *measured floor* | | *0.642* | *0.536* | *0.495* |

## The torus comparison

For reference, the identical 2x2 on the 64x64 torus paper task at n=8 (`INDEX_BASELINE_PAPER_TASK_n8.md`), floor 0.506:

| | index | path-integrated |
|---|---|---|
| RoPE encoding | 0.530 | **0.967** |
| PoPE encoding | 0.509 | **0.994** |

Both index arms sit ON the floor there. Whether that survives on an egocentric, rotation-actioned environment is what this file measures.
