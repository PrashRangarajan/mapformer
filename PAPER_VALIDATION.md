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

## CORRECTION + accuracy measurement (2026-08-09)

The "CLAIMED in CLAUDE.md: WM 0.955 / EM 0.999" target above is **wrong** — no
such pair appears in the paper. Read from the PDF, Table 2 (1D-2D grid
navigation), 2D columns:

| paper model | 2D IID | 2D OOD-d | 2D OOD-s |
|---|---|---|---|
| MapWM | 0.99 | 0.99 | 0.96 |
| MapEM-os | 1.0 | 0.99 | 0.97 |

Ours is MapEM-**os** (paper sec. C: "three types of EM models: (1) MapEM-os
relying on both observation and structure to compute attention"), since our
A = softmax(A_X o A_P)V uses both.

Held-out revisit accuracy, measured (`eval_paper_task.py`, n=3 seeds, T=128):

| variant | same-map | fresh-map |
|---|---|---|
| Vanilla (WM) | 0.989 ± 0.010 | 0.989 ± 0.010 |
| VanillaEM (separate q0/k0) | 0.898 ± 0.108 | 0.901 ± 0.102 |
| VanillaEM_P0 (single p_0) | 0.987 ± 0.012 | 0.987 ± 0.012 |

same-map == fresh-map to 3 decimals for every variant, so the models are
building the cognitive map **in context**, not memorising obs_map into weights.

CORRECTION (2026-08-09): "correction" is the wrong word for single-p_0. Appendix
A.4 says the paper DOES use separate k0p/q0p and that collapsing them to a single
p_0 is optional ("we could set k0p = q0p = p0 without loss of generality.
However, we suspect this separation to be beneficial"). So separate q0/k0 is
paper-faithful, and single-p_0 is an ABLATION of a conjecture the paper never
tests. Our data refutes the conjecture. Keep BOTH rows in all tables.

Verdict: WM reproduces (0.989 vs paper 0.99). EM reproduces only in the
single-p_0 ablation (0.987, best seed 0.9995, vs paper 1.0); the separate
q0/k0 version does not (0.898, and seed-unstable: 0.778 / 0.931 / 0.986).
NOT yet tested: the paper's own OOD protocol (OOD-d = 64 steps/grid 32,
OOD-s = 256/128), which is a different length+grid change from our T=512 OOD.
