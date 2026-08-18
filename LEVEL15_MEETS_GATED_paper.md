# Paper-task held-out revisit ACCURACY

Paper config: 1 layer, 2 heads, d=128, T=128, 200K sequences (16 epochs x 98 batches x 128).
Paper Table 2 (2D columns), IID: MapWM **0.99**, MapEM-os **1.0**. (An earlier version of this file cited 0.955 / 0.999; those numbers appear in no table of the paper and were retracted in CLAUDE.md on 2026-08-09.)

`same-map` = new trajectories on the trained obs_map; `fresh-map` = unseen obs_map (in-context map learning).

| variant | same-map acc | fresh-map acc |
|---|---|---|
| Vanilla | 0.989 ± 0.010 | 0.989 ± 0.010 |
| Vanilla_ExtraHead | 0.972 ± 0.038 | 0.972 ± 0.039 |
| Level15 | 0.938 ± 0.081 | 0.938 ± 0.080 |
