# Paper-task validation: does our reimplementation reproduce MapFormer?

Paper task (aliased-obs revisit, torus), paper config: 1 layer, 2 heads,
d=128, batch 128, T=128, 200K sequences. CLAIMED in CLAUDE.md: WM 0.955 / EM 0.999.

| variant | seed | final train loss | held-out revisit acc |
|---|---|---|---|
| Vanilla | 0 | 0.0700 | n/a |
| Vanilla | 1 | 0.1362 | n/a |
| Vanilla | 2 | 0.1379 | n/a |
| VanillaEM | 0 | 0.3908 | n/a |
| VanillaEM | 1 | 1.0234 | n/a |
| VanillaEM | 2 | 0.0832 | n/a |
| VanillaEM_Fixed | 0 | 1.1963 | n/a |
| VanillaEM_Fixed | 1 | 1.2021 | n/a |
| VanillaEM_Fixed | 2 | 1.1840 | n/a |
