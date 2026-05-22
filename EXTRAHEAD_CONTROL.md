# Capacity control: is the Hopfield cross-scale win structure or just an extra head?

`Level15_ExtraHead`: Level15 + a GENERIC extra attention head (content
Q/K projections, position-modulated, all-positions causal KV), added the
same residual way as the Hopfield head, with MORE parameters than
Level15_Hopfield. Conservative capacity control.

Cross-scale held-out T=512 OOD, n=3 seeds.

| Variant | size 32 | size 64 | size 128 |
|---|---|---|---|
| **Level15** | 0.782 ± 0.138 (n=3) | 0.921 ± 0.050 (n=3) | 0.953 ± 0.032 (n=3) |
| **Level15_Hopfield** | 0.919 ± 0.089 (n=3) | 0.972 ± 0.030 (n=3) | 0.985 ± 0.018 (n=3) |
| **Level15_ExtraHead** | 0.934 ± 0.081 (n=3) | 0.979 ± 0.028 (n=3) | 0.984 ± 0.022 (n=3) |
| **TEMFaithful** | 0.936 ± 0.021 (n=3) | 0.973 ± 0.006 (n=3) | 0.981 ± 0.005 (n=3) |

## Per-seed (size 32 T=512) — exposes the seed-instability story

| Variant | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **Level15** | 0.977 | 0.701 | 0.668 |
| **Level15_Hopfield** | 0.986 | 0.793 | 0.977 |
| **Level15_ExtraHead** | 0.987 | 0.819 | 0.996 |

## Verdict: CAPACITY, not structure

`Level15_ExtraHead` (generic extra head, more params, no Hopfield
structure) reaches 0.934 ± 0.081 at size 32 — statistically
indistinguishable from `Level15_Hopfield` (0.919 ± 0.089). A generic
second attention head closes the cross-scale gap exactly as well as the
position-keyed obs-restricted Hopfield head.

**The cross-scale fix is adding a second attention head (capacity /
training stabilisation), NOT the TEM-inspired Hopfield mechanism.**
The per-seed table shows base Level15 collapses on 2 of 3 seeds
(~0.67-0.70, the coupled-ω bad basin). Adding ANY extra attention head
rescues most seeds; the position-only-key / obs-restricted-KV structure
is not necessary. The earlier 'we ported TEM's memory mechanism'
framing is NOT supported — it is an extra-head / capacity effect.

ExtraHead still has one weak seed (seed 1 = 0.819), so it reduces but
does not eliminate the instability. TEMFaithful remains the most
seed-stable (±0.021).

*Auto-generated; n=3 after seed-1 rerun (original seed 1 crashed on a
transient CUDA OOM).*
