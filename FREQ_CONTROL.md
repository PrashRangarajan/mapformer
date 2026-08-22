# The torus 2x2 on MiniGrid-DoorKey-16x16 (n=3)

External, published benchmark; egocentric observation (the cell in front of the agent). All four arms trained in ONE batch (rule 3), 50 epochs, 25K cached trajectory buffer.

**Measured floor** (most common scored target) per length: T=128: **0.642**, T=512: **0.536**, T=1024: **0.495**. The original n=1 evaluation reported no floor at all.

| model | position | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| MapWM-Flat (RoPE + path int.) | path-integrated | 0.996 ± 0.001 | 0.890 ± 0.043 | 0.792 ± 0.077 |
| MapWM-Flat, omega FROZEN | path-integrated | 0.996 ± 0.001 | 0.886 ± 0.036 | 0.800 ± 0.076 |
| RoPE-Flat (index) | **index** | 0.982 ± 0.003 | 0.929 ± 0.004 | 0.860 ± 0.022 |

| *measured floor* | | *0.642* | *0.536* | *0.495* |

## The torus comparison

For reference, the identical 2x2 on the 64x64 torus paper task at n=8 (`INDEX_BASELINE_PAPER_TASK_n8.md`), floor 0.506:

| | index | path-integrated |
|---|---|---|
| RoPE encoding | 0.530 | **0.967** |
| PoPE encoding | 0.509 | **0.994** |

Both index arms sit ON the floor there. Whether that survives on an egocentric, rotation-actioned environment is what this file measures.

## Result: the confound is empirically negligible, and the position effect is real

Every "position effect" in this repo compared path-integrated arms (learnable
`omega`, an `nn.Parameter`) against index arms (fixed frequencies, a
`register_buffer`). Those two properties are perfectly correlated across every
cell of every grid, so "position" has always meant "path-integration AND
frequency learning". `Vanilla_FixedOmega` breaks the correlation: identical
architecture and init, path integration intact, only the 64 `omega` values
frozen.

| effect | T=512 | T=1024 | paired |
|---|---|---|---|
| **frequency learning** (Vanilla − FixedOmega) | **+0.004** | **−0.008** | −0.003/+0.013/+0.001 and −0.006/+0.010/−0.028 |
| **pure position** (FixedOmega − RoPE) | **−0.042** | **−0.060** | −0.087/−0.013/−0.026 and −0.125/−0.033/−0.022 |

**Frequency learning does nothing.** Freezing all 64 angular velocities moves
accuracy by less than a hundredth in either direction, with the paired
differences straddling zero at both lengths. The confound exists in the code but
not in the numbers.

**The position effect survives and grows.** With frequency learning removed it is
−0.042 / −0.060 rather than the −0.012 / −0.037 measured with the confound in
place, and it is negative on **3 of 3 seeds at both lengths**. So on MiniGrid,
path integration is not merely useless — it is mildly harmful, and the earlier
estimate was if anything conservative.

## Side finding: the paper's learnable angular velocities buy nothing here

`omega` is one of MapFormer's stated design choices (App. A.8 gives a geometric
initialisation and makes them learnable). On MiniGrid-DoorKey-16x16 they can be
frozen at that initialisation with no measurable cost — 614,474 trainable
parameters perform the same as 614,538. Whether that also holds on the torus,
where the position code is doing far more work, is untested and should not be
assumed from this.

## What this does NOT establish

`FixedOmega − RoPE` is closer to a pure position comparison but still not exact:
the two use different frequency SCHEDULES (`omega_max*(1/grid)^(i/(nb-1))` vs
`base^(-k/(nb-1))`). This removes the learned-vs-fixed difference; it does not
equalise the schedules. A fully clean position comparison would need one arm
built with the other's schedule, which is a further experiment.
