# Paper-task held-out revisit ACCURACY

Paper config: 1 layer, 2 heads, d=128, T=128, 200K sequences (16 epochs x 98 batches x 128).
CORRECTED 2026-09-11: those figures (0.955 / 0.999) appear in NO table of the paper. Table 2
(1D-2D grid navigation), 2D columns, v1: MapWM 0.99 / 0.99 / 0.96, MapEM-os 1.0 / 0.99 / 0.97;
v4 has 1.00 throughout. See CLAUDE.md 2026-08-09.

`same-map` = new trajectories on the trained obs_map; `fresh-map` = unseen obs_map (in-context map learning).

| variant | same-map acc | fresh-map acc |
|---|---|---|
| Vanilla | 0.989 ± 0.010 | 0.989 ± 0.010 |
| VanillaEM | 0.898 ± 0.108 | 0.901 ± 0.102 |
| VanillaEM_P0 | 0.987 ± 0.012 | 0.987 ± 0.012 |
