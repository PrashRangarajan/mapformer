# MiniWorld — non-blank accuracy at T=1024

chance (non-blank) = 1/16 = 0.0625, oracle-acc = 1.0. Path-integrated = {Vanilla, MapPoPE-Flat}; index = {RoPE, PoPE-Flat}. ⚠ marks an arm with final train loss > 1.5.

| variant | position | allocentric (24-bin dir) | oracle (exact cell Δ) |
|---|---|---|---|
| Vanilla | path-int | 0.131 ± 0.017 | 0.289 ± 0.075 |
| MapPoPE-Flat | path-int | 0.115 ± 0.022 ⚠ | 0.217 ± 0.041 |
| RoPE | index | 0.335 ± 0.032 | 0.543 ± 0.041 |
| PoPE-Flat | index | 0.279 ± 0.006 | 0.782 ± 0.010 |

## Position effect (path-integrated − index), paired within seed

| encoding | effect (mean ± std over seeds) | per-seed |
|---|---|---|
| allocentric (24-bin dir) | **-0.184 ± 0.034** (n=3) | s0=-0.221, s1=-0.177, s2=-0.153 |
| oracle (exact cell Δ) | **-0.409 ± 0.072** (n=3) | s0=-0.332, s1=-0.419, s2=-0.476 |

> **INCOMPLETE / SUSPECT — verdict withheld.**
> missing: ['Vanilla_raw_s0', 'Vanilla_raw_s1', 'Vanilla_raw_s2', 'MapPoPE-Flat_raw_s0', 'MapPoPE-Flat_raw_s1', 'MapPoPE-Flat_raw_s2', 'RoPE_raw_s0', 'RoPE_raw_s1', 'RoPE_raw_s2', 'PoPE-Flat_raw_s0', 'PoPE-Flat_raw_s1', 'PoPE-Flat_raw_s2']
> possibly-non-converged (loss>1.5): ['MapPoPE-Flat_allo_s0(loss=1.60)']
> Re-run the flagged arms in the same batch before reading the flip.
