# MiniWorld fixed-map — non-blank accuracy at T=1024

Path integration on a KNOWN map (novel walk/episode). chance (non-blank) = 1/16 = 0.0625, oracle = 1.0. Path-integrated = {Vanilla, MapPoPE-Flat}; index = {RoPE, PoPE-Flat}. ⚠ marks an arm with final train loss > 1.5 (possibly non-converged; project rule: acc tracks final loss).

| variant | position | raw | allocentric |
|---|---|---|---|
| Vanilla | path-int | 0.212 ± 0.048 ⚠ | 0.131 ± 0.017 |
| MapPoPE-Flat | path-int | 0.231 ± 0.016 | 0.115 ± 0.022 ⚠ |
| RoPE | index | 0.248 ± 0.008 | 0.335 ± 0.032 |
| PoPE-Flat | index | 0.298 ± 0.013 | 0.279 ± 0.006 |

## Position effect (path-integrated − index), paired within seed

| encoding | effect (mean ± std over seeds) | per-seed |
|---|---|---|
| raw (turn/forward) | **-0.051 ± 0.020** (n=3) | s0=-0.030, s1=-0.057, s2=-0.068 |
| allocentric (displacement) | **-0.184 ± 0.034** (n=3) | s0=-0.221, s1=-0.177, s2=-0.153 |

> **INCOMPLETE / SUSPECT — verdict withheld.**
> missing: none
> possibly-non-converged (loss>1.5): ['Vanilla_raw_s2(loss=1.53)', 'MapPoPE-Flat_allo_s0(loss=1.60)']
> Re-run the flagged arms in the same batch before reading the flip.
