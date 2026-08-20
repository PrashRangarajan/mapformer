# Dissociation sweep: when does hierarchy pay, and when does path integration pay?

The 2x2 {flat, hierarchical} x {index, path-integrated}, swept over `n_templates` -- the number of distinct room motifs tiled over the grid's 64 rooms. **Low = repetitive, real compositional structure to exploit. High = each room near-unique, so the task degenerates toward pure retrieval.**

Every arm at every sweep point trained in ONE batch (rule 3); each model evaluated on the environment it was TRAINED on.

## Compositional transfer — `cross_nb_acc` @T=256

| n_templates | MapWM-Flat | MapWM-Hier | Plain-Flat | Plain-Hier | **hierarchy effect** | **path-int effect** | floor |
|---|---|---|---|---|---|---|---|
| 2 | 0.365 ± 0.016 | 0.560 ± 0.206 | 0.316 ± 0.006 | 0.446 ± 0.023 | **+0.163** | **+0.081** | 0.072 |
| 4 | 0.260 ± 0.034 | 0.413 ± 0.172 | 0.200 ± 0.003 | 0.316 ± 0.043 | **+0.135** | **+0.078** | 0.072 |
| 8 | 0.360 ± 0.157 | 0.347 ± 0.151 | 0.139 ± 0.007 | 0.220 ± 0.023 | **+0.034** | **+0.174** | 0.071 |
| 16 | 0.291 ± 0.216 | 0.219 ± 0.038 | 0.099 ± 0.012 | 0.183 ± 0.044 | **+0.006** | **+0.114** | 0.081 |

## Precise recall — `exact_acc` @T=1024

| n_templates | MapWM-Flat | MapWM-Hier | Plain-Flat | Plain-Hier | **hierarchy effect** | **path-int effect** |
|---|---|---|---|---|---|---|
| 2 | 0.610 ± 0.020 | 0.761 ± 0.054 | 0.581 ± 0.035 | 0.670 ± 0.018 | **+0.120** | **+0.060** |
| 4 | 0.765 ± 0.042 | 0.864 ± 0.120 | 0.629 ± 0.009 | 0.682 ± 0.040 | **+0.076** | **+0.158** |
| 8 | 0.873 ± 0.066 | 0.856 ± 0.124 | 0.592 ± 0.033 | 0.637 ± 0.030 | **+0.014** | **+0.251** |
| 16 | 0.803 ± 0.150 | 0.758 ± 0.018 | 0.619 ± 0.012 | 0.635 ± 0.043 | **-0.015** | **+0.154** |

## Pre-registered predictions — scored

Written into `sweep_dissociation.py` before the run.

| # | prediction | outcome |
|---|---|---|
| **P1** | hierarchy's `cross_nb` advantage FALLS with n_templates, approaching zero at the high end | **CONFIRMED.** +0.163 → +0.135 → +0.034 → **+0.006**, monotone across all four points. |
| **P2** | path integration's `exact_acc` advantage is FLAT in n_templates | **REFUTED.** +0.060 → +0.158 → +0.251 → +0.154. Not flat, and not monotone either — it peaks at nt=8. |
| **P3** | the two cross: which ingredient matters is a property of the TASK, not the architecture | **CONFIRMED, on both metrics.** |

### P3 is the result

At the structured end hierarchy is the bigger lever; at the unstructured end path
integration is, by a wide margin. The rank reverses on `cross_nb` AND on
`exact_acc`:

| | hierarchy | path-int | winner |
|---|---|---|---|
| `cross_nb`, nt=2 | **+0.163** | +0.081 | hierarchy, 2x |
| `cross_nb`, nt=8 | +0.034 | **+0.174** | path integration, 5x |
| `exact_acc`, nt=2 | **+0.120** | +0.060 | hierarchy, 2x |
| `exact_acc`, nt=8 | +0.014 | **+0.251** | path integration, 18x |

**And hierarchy goes NEGATIVE on precise recall once the structure is gone**
(-0.015 at nt=16). The trade-off that previously had to be assembled from two
different benchmarks — hierarchy buys transfer, costs recall — now appears inside
one task, on one axis, with the sign flip visible.

### P2's failure is NOT explained by the confound I proposed

I predicted the axis confounds compositional structure with aliasing, and that
this would account for P2. `ALIASING_COVARIATE.md` measures it and rules that
out: per-cell aliasing is flat (~1100 cells share an observation, H(obs)
2.91-3.00 bits at every point), and sequence-level disambiguation moves 10x
(run_8: 20.4 → 7.4 → 3.4 → 2.1) but in the WRONG direction — content localises
better at high n_templates, which should make path integration less necessary
there, not more.

The account still standing, read off this table and flagged as untested: the
position-aware ceiling rises with n_templates while the index ceiling does not.
MapWM-Flat `exact_acc` climbs 0.610 → 0.765 → 0.873 while Plain-Flat is pinned at
0.581 → 0.629 → 0.592.

## Caveats that limit how hard this can be pushed

**1. The MapWM arms have enormous seed variance**, growing with n_templates:
MapWM-Hier ±0.206 at nt=2, MapWM-Flat ±0.157 at nt=8 and **±0.216** at nt=16.
The nt=16 column in particular is unreliable on its own; P1's monotone decline is
carried mainly by the nt=2 → 8 range, where the Plain arms (±0.006-0.043) are
tight enough to trust.

**2. At the low-structure end the two backbones DIVERGE, and the average hides
it.** The reported hierarchy effect is a mean over both backbones. At nt=16 that
+0.006 is the average of **-0.072** (MapWM: 0.291 → 0.219) and **+0.084**
(Plain: 0.099 → 0.183). Hierarchy keeps helping a plain transformer while
actively hurting MapFormer. The backbone-independence that holds at nt=2 and
nt=4 breaks down here, and "hierarchy's advantage goes to zero" is the wrong
summary for that column — it goes to zero *on average*, by cancellation.

**3. One environment, n=3, one training recipe.** The curve is four points.
