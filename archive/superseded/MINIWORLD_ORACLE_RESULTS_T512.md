# MiniWorld — non-blank accuracy at T=512

chance (non-blank) = 1/16 = 0.0625, oracle-acc = 1.0. Path-integrated = {Vanilla, MapPoPE-Flat}; index = {RoPE, PoPE-Flat}. ⚠ marks an arm with final train loss > 1.5.

| variant | position | allocentric (24-bin dir) | oracle (exact cell Δ) |
|---|---|---|---|
| Vanilla | path-int | 0.284 ± 0.016 | 0.448 ± 0.079 |
| MapPoPE-Flat | path-int | 0.232 ± 0.057 ⚠ | 0.324 ± 0.029 |
| RoPE | index | 0.501 ± 0.014 | 0.977 ± 0.007 |
| PoPE-Flat | index | 0.364 ± 0.008 | 0.938 ± 0.036 |

## Position effect (path-integrated − index), paired within seed

| encoding | effect (mean ± std over seeds) | per-seed |
|---|---|---|
| allocentric (24-bin dir) | **-0.174 ± 0.035** (n=3) | s0=-0.214, s1=-0.159, s2=-0.150 |
| oracle (exact cell Δ) | **-0.571 ± 0.042** (n=3) | s0=-0.524, s1=-0.603, s2=-0.587 |

> **INCOMPLETE / SUSPECT — verdict withheld.**
> missing: ['Vanilla_raw_s0', 'Vanilla_raw_s1', 'Vanilla_raw_s2', 'MapPoPE-Flat_raw_s0', 'MapPoPE-Flat_raw_s1', 'MapPoPE-Flat_raw_s2', 'RoPE_raw_s0', 'RoPE_raw_s1', 'RoPE_raw_s2', 'PoPE-Flat_raw_s0', 'PoPE-Flat_raw_s1', 'PoPE-Flat_raw_s2']
> possibly-non-converged (loss>1.5): ['MapPoPE-Flat_allo_s0(loss=1.60)']
> Re-run the flagged arms in the same batch before reading the flip.
