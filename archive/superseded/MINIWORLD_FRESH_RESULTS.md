# MiniWorld fixed-map — non-blank accuracy at T=512

Path integration on a KNOWN map (novel walk/episode). chance (non-blank) = 1/16 = 0.0625, oracle = 1.0. Path-integrated = {Vanilla, MapPoPE-Flat}; index = {RoPE, PoPE-Flat}. ⚠ marks an arm with final train loss > 1.5 (possibly non-converged; project rule: acc tracks final loss).

| variant | position | raw | allocentric |
|---|---|---|---|
| Vanilla | path-int | 0.303 ± 0.062 ⚠ | 0.284 ± 0.016 |
| MapPoPE-Flat | path-int | 0.308 ± 0.015 | 0.232 ± 0.057 ⚠ |
| RoPE | index | 0.398 ± 0.002 | 0.501 ± 0.014 |
| PoPE-Flat | index | 0.384 ± 0.012 | 0.364 ± 0.008 |

## Position effect (path-integrated − index), paired within seed

| encoding | effect (mean ± std over seeds) | per-seed |
|---|---|---|
| raw (turn/forward) | **-0.086 ± 0.021** (n=3) | s0=-0.065, s1=-0.085, s2=-0.107 |
| allocentric (displacement) | **-0.174 ± 0.035** (n=3) | s0=-0.214, s1=-0.159, s2=-0.150 |

> **INCOMPLETE / SUSPECT — verdict withheld.**
> missing: none
> possibly-non-converged (loss>1.5): ['Vanilla_raw_s2(loss=1.53)', 'MapPoPE-Flat_allo_s0(loss=1.60)']
> Re-run the flagged arms in the same batch before reading the flip.
