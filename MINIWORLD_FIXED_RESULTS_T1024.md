# MiniWorld fixed-map — non-blank accuracy at T=1024

Path integration on a KNOWN map (novel walk/episode). chance (non-blank) = 1/16 = 0.0625, oracle = 1.0. Path-integrated = {Vanilla, MapPoPE-Flat}; index = {RoPE, PoPE-Flat}. ⚠ marks an arm with final train loss > 0.6 (possibly non-converged; project rule: acc tracks final loss).

| variant | position | raw | allocentric |
|---|---|---|---|
| Vanilla | path-int | 0.499 ± 0.108 | 0.574 ± 0.005 |
| MapPoPE-Flat | path-int | 0.577 ± 0.044 | 0.787 ± 0.030 |
| RoPE | index | 0.445 ± 0.032 | 0.612 ± 0.033 |
| PoPE-Flat | index | 0.611 ± 0.038 | 0.790 ± 0.015 |

## Position effect (path-integrated − index), paired within seed

| encoding | effect (mean ± std over seeds) | per-seed |
|---|---|---|
| raw (turn/forward) | **+0.010 ± 0.017** (n=3) | s0=+0.015, s1=-0.009, s2=+0.024 |
| allocentric (displacement) | **-0.021 ± 0.013** (n=3) | s0=-0.034, s1=-0.007, s2=-0.021 |

**Flip:** raw +0.010 → allocentric -0.021 (per-seed Δ = -0.049, +0.002, -0.045). no clear flip — allocentric does not reliably raise the path-int−index gap.
