# Paper-task held-out revisit ACCURACY

Paper config: 1 layer, 2 heads, d=128, T=128, 200K sequences (16 epochs x 98 batches x 128).
Paper Table 2 (2D columns), IID: MapWM **0.99**, MapEM-os **1.0**. (An earlier version of this file cited 0.955 / 0.999; those numbers appear in no table of the paper and were retracted in CLAUDE.md on 2026-08-09.)

`same-map` = new trajectories on the trained obs_map; `fresh-map` = unseen obs_map (in-context map learning).

| variant | same-map acc | fresh-map acc |
|---|---|---|
| Vanilla | 0.968 ± 0.039 | 0.967 ± 0.039 |
| MapPoPE-Flat | 0.994 ± 0.016 | 0.994 ± 0.017 |
| RoPE | 0.524 ± 0.043 | 0.530 ± 0.043 |
| PlainFlat | 0.529 ± 0.036 | 0.534 ± 0.040 |
| PoPE-Flat | 0.501 ± 0.007 | 0.509 ± 0.001 |
| VanillaEM_P0 | 0.987 ± 0.008 | 0.987 ± 0.009 |
