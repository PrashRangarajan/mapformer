# MiniWorld fixed-map — non-blank accuracy at T=512

Path integration on a KNOWN map (novel walk/episode). chance (non-blank) = 1/16 = 0.0625, oracle = 1.0. Path-integrated = {Vanilla, MapPoPE-Flat}; index = {RoPE, PoPE-Flat}. ⚠ marks an arm with final train loss > 0.6 (possibly non-converged; project rule: acc tracks final loss).

| variant | position | raw | allocentric |
|---|---|---|---|
| Vanilla | path-int | 0.653 ± 0.035 | 0.801 ± 0.021 |
| MapPoPE-Flat | path-int | 0.662 ± 0.019 | 0.819 ± 0.020 |
| RoPE | index | 0.620 ± 0.024 | 0.798 ± 0.015 |
| PoPE-Flat | index | 0.655 ± 0.027 | 0.807 ± 0.017 |

## Position effect (path-integrated − index), paired within seed

| encoding | effect (mean ± std over seeds) | per-seed |
|---|---|---|
| raw (turn/forward) | **+0.020 ± 0.013** (n=3) | s0=+0.015, s1=+0.034, s2=+0.010 |
| allocentric (displacement) | **+0.008 ± 0.009** (n=3) | s0=+0.017, s1=-0.000, s2=+0.007 |

**Flip:** raw +0.020 → allocentric +0.008 (per-seed Δ = +0.002, -0.035, -0.003). no clear flip — allocentric does not reliably raise the path-int−index gap.
