# Paper-task held-out revisit ACCURACY

Paper config: 1 layer, 2 heads, d=128, T=128, 200K sequences (16 epochs x 98 batches x 128).
Paper reports MapFormer-WM **0.955**, MapFormer-EM **0.999**.

`same-map` = new trajectories on the trained obs_map; `fresh-map` = unseen obs_map (in-context map learning).

| variant | same-map acc | fresh-map acc |
|---|---|---|
| Vanilla | 0.989 ± 0.010 | 0.989 ± 0.010 |
| VanillaEM | 0.898 ± 0.108 | 0.901 ± 0.102 |
| VanillaEM_P0 | 0.987 ± 0.012 | 0.987 ± 0.012 |
