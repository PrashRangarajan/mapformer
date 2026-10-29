> **INCONCLUSIVE (not a clean negative).** No allocentric flip (raw -0.019 ->
> allo -0.043), BUT every model is stuck at ~0.08-0.23 non-blank accuracy against
> a 1.0 oracle -- nobody learned the task. Path integration does not win in either
> encoding, index models are not at the floor (unlike the torus), and training is
> high-variance/bimodal (one RoPE seed at 0.025). When no model approaches the
> oracle, the raw-vs-allocentric comparison is between weak partial solutions and
> does NOT reflect path-integration capability. The MiniWorld task as configured
> (fresh map/episode, continuous rotation nav, cross-cell revisits at median lag 62)
> is too hard for a 3-layer d=128 model at 40 epochs. This does NOT overturn the
> clean MiniGrid allocentric flip; it means MiniWorld did not converge well enough
> to test the hypothesis. Needs an easier/learnable config or more compute to be
> conclusive.

# MiniWorld continuous-3D — non-blank accuracy at T=512

Held-out fresh obs_map. chance (non-blank) = 1/16 = 0.0625. Path-integrated = {Vanilla, MapPoPE-Flat}; index = {RoPE, PoPE-Flat}.

| variant | position | raw | allocentric |
|---|---|---|---|
| Vanilla | path-int | 0.197 ± 0.026 | 0.077 ± 0.033 |
| MapPoPE-Flat | path-int | 0.139 ± 0.107 | 0.148 ± 0.032 |
| RoPE | index | 0.141 ± 0.102 | 0.154 ± 0.011 |
| PoPE-Flat | index | 0.233 ± 0.017 | 0.157 ± 0.037 |

## Position effect (path-integrated − index)

| encoding | path-int mean | index mean | position effect |
|---|---|---|---|
| raw (turn/forward) | 0.168 | 0.187 | **-0.019** |
| allocentric (displacement) | 0.113 | 0.156 | **-0.043** |

**Flip:** raw -0.019 → allocentric -0.043 (no clear flip).
