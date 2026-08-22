# The torus 2x2 on MiniGrid-DoorKey-16x16 (n=3)

External, published benchmark; egocentric observation (the cell in front of the agent). All four arms trained in ONE batch (rule 3), 50 epochs, 25K cached trajectory buffer.

**Measured floor** (most common scored target) per length: T=128: **0.642**, T=512: **0.536**, T=1024: **0.495**. The original n=1 evaluation reported no floor at all.

| model | position | T=128 | T=512 | T=1024 |
|---|---|---|---|---|
| MapWM-Flat (RoPE + path int.) | path-integrated | 0.996 ± 0.001 | 0.890 ± 0.043 | 0.792 ± 0.077 |
| MapWM-Hier (RoPE + path int. + hier) | path-integrated | 0.997 ± 0.000 | 0.956 ± 0.010 | 0.909 ± 0.006 |
| MapPoPE-Flat (PoPE + path int.) | path-integrated | 0.994 ± 0.001 | 0.957 ± 0.012 | 0.919 ± 0.022 |
| MapPoPE-Hier (PoPE + path int. + hier) | path-integrated | 0.995 ± 0.001 | 0.971 ± 0.005 | 0.948 ± 0.014 |
| RoPE-Flat (index) | **index** | 0.982 ± 0.003 | 0.929 ± 0.004 | 0.860 ± 0.022 |
| RoPE-Hier (index + hier) | **index** | 0.987 ± 0.003 | 0.950 ± 0.005 | 0.921 ± 0.005 |
| PoPE-Flat (PoPE + index) | **index** | 0.982 ± 0.002 | 0.961 ± 0.001 | 0.951 ± 0.004 |

| *measured floor* | | *0.642* | *0.536* | *0.495* |

## The torus comparison

For reference, the identical 2x2 on the 64x64 torus paper task at n=8 (`INDEX_BASELINE_PAPER_TASK_n8.md`), floor 0.506:

| | index | path-integrated |
|---|---|---|
| RoPE encoding | 0.530 | **0.967** |
| PoPE encoding | 0.509 | **0.994** |

Both index arms sit ON the floor there. Whether that survives on an egocentric, rotation-actioned environment is what this file measures.

## Hierarchy helps every cell, on every seed

Paired within each (encoding × position) pair, so the only thing changing is
flat vs hierarchical, at exactly matched parameters (614K):

| pair | T=512 paired Δ | T=1024 paired Δ |
|---|---|---|
| RoPE + path-int | +0.115 / +0.038 / +0.044 → **+0.066** | +0.203 / +0.057 / +0.092 → **+0.117** |
| PoPE + path-int | +0.028 / +0.003 / +0.009 → **+0.013** | +0.059 / +0.014 / +0.016 → **+0.029** |
| RoPE + index | +0.017 / +0.028 / +0.019 → **+0.021** | +0.087 / +0.040 / +0.054 → **+0.061** |

**18 of 18 paired comparisons positive.** No seed, in any cell, at either
length, is hurt by hierarchy. That is the most consistent effect measured on this
benchmark, and it contrasts with the torus, where hierarchy's benefit was
structure-dependent — +0.163 on compositional transfer with repeated motifs,
+0.006 without them, and NEGATIVE (−0.015) on precise recall.

Note where it helps most: **+0.117 for the weakest path-integrated arm and
+0.029 for the strongest**. Hierarchy is compensating for a weaker base rather
than adding something the strong models lack.

## The ingredient ranking REORDERS between environments

Each factor averaged over the other two:

| | encoding (PoPE−RoPE) | hierarchy (hier−flat) | position (path-int−index) |
|---|---|---|---|
| **torus paper task** (n=8) | +0.011 | — | **+0.461** |
| MiniGrid T=512 | **+0.038** | +0.033 | −0.012 |
| MiniGrid T=1024 | **+0.085** | +0.069 | **−0.037** |

On the torus, position is worth 40x the encoding. On MiniGrid the order is
**encoding > hierarchy > position**, and position is *negative*. Three
architectural factors, and which one to spend on is decided by the environment,
not by the architecture.

## The inversion survives depth

At `n_layers=1` the index model beat the standard MapFormer at long horizon
(RoPE 0.851 vs MapWM-Flat 0.800 at T=512). At 3 layers both improve, and the
ordering is unchanged: **RoPE-Flat 0.929 vs MapWM-Flat 0.890** at T=512, and
**0.860 vs 0.792** at T=1024. So the paper's own model losing to an index
baseline here is not a capacity artifact — it holds at 1 and 3 layers, and
MapWM-Flat remains the highest-variance arm (±0.043, ±0.077 against ±0.001-0.022
for everything else).

## Honest limits

- **Seven of eight cells.** There is no PoPE+index+hierarchy variant in
  `VARIANT_MAP`, so the hierarchy factor rests on three pairs and the encoding
  and position factors on three each, not four. The factorial is not complete and
  the averages above are over what exists.
- **n=3**, though the paired consistency (18/18) is stronger evidence than the
  means alone.
- **One MiniGrid environment.** DoorKey-16×16 only; the 8×8 version is too small
  to separate anything and MultiRoom was never run.
- **Everything is far above the floor** (0.495-0.642), so this benchmark is not
  discriminative in the way the torus is — the whole spread across seven models
  at T=1024 is 0.792-0.951.
