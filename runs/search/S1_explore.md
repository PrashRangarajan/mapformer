## Exploratory (not registered): route of each solved cell (acc >= 0.9, k >= 8)

Read in the head with the most attention on the answer. Categories in priority order: peak-rewind (sel - sel0 >= 0.5), trough-rewind (selmin - selmin0 >= 0.5), peak-static (sel0 >= 0.5), trough-static (selmin0 >= 0.5), other.

| arm | solved | peak-rewind | trough-rewind | peak-static | trough-static | other | A_X>0 in peak-* | A_X>0 in trough-* |
|---|---|---|---|---|---|---|---|---|
| VanillaEM_P0_r4 | 595 | 0.427 | 0.541 | 0.000 | 0.032 | 0.000 | 1.000 | 0.000 |
| EMDoF_alignlock | 595 | 0.371 | 0.576 | 0.000 | 0.045 | 0.007 | 1.000 | 0.000 |
| EMDoF_magonly | 600 | 0.398 | 0.557 | 0.000 | 0.045 | 0.000 | 1.000 | 0.000 |
| EMDoF_alignfree | 1009 | 0.424 | 0.515 | 0.028 | 0.033 | 0.000 | 1.000 | 0.000 |
| VanillaEM_r4 | 958 | 0.501 | 0.432 | 0.035 | 0.031 | 0.000 | 1.000 | 0.000 |

## Exploratory: P(solved) against distance from the nearest head's kernel peak to k

n*_h = argmax of head h's Delta(q)=0 profile; d = min_h |k - n*_h|. If the difficulty of search is the size of the shift the query token must make, the arms fall on one curve.

| arm | d 0-3 | d 4-7 | d 8-15 | d 16-31 | d 32-64 |
|---|---|---|---|---|---|
| VanillaEM_P0_r4 | 0.972 (n=72) | 0.740 (n=96) | 0.458 (n=192) | 0.352 (n=384) | 0.470 (n=792) |
| EMDoF_alignlock | 0.931 (n=72) | 0.740 (n=96) | 0.495 (n=192) | 0.323 (n=384) | 0.475 (n=792) |
| EMDoF_magonly | 0.944 (n=72) | 0.833 (n=96) | 0.547 (n=192) | 0.333 (n=384) | 0.463 (n=792) |
| EMDoF_alignfree | 0.812 (n=298) | 0.742 (n=252) | 0.770 (n=365) | 0.753 (n=380) | 0.722 (n=241) |
| VanillaEM_r4 | 0.732 (n=313) | 0.688 (n=324) | 0.733 (n=442) | 0.723 (n=376) | 0.802 (n=81) |
