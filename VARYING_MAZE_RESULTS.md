# VARYING maze: build a cognitive map, then plan on it

Fresh maze + landmarks every episode -> memorisation impossible by construction.
(Fixed-maze version collapsed 0.94 -> 0.68 on a novel maze: it had memorised.)
Chance = 0.25. Greedy wall-ignoring policy = 0.73, so >0.73 means the model
is USING the map it built during exploration.

| Variant | len 1-5 | len 6-10 | len 11-15 | len 16+ | all |
|---|---|---|---|---|---|
| Level15 (n=3) | 0.446±0.008 | 0.482±0.001 | 0.534±0.007 | 0.532±0.001 | 0.503±0.003 |
| HierAttn (n=3) | 0.454±0.008 | 0.486±0.005 | 0.531±0.006 | 0.541±0.004 | 0.506±0.005 |
| HierAttn_LocalOnly (n=1) | 0.424 | 0.481 | 0.530 | 0.534 | 0.499 |
| HierAttn_CoarseOnly (n=1) | 0.431 | 0.472 | 0.524 | 0.538 | 0.495 |
