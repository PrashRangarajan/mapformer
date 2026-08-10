# Match-Query results (n=3 seeds)

Blind continuation: explore with observations revealed, then continue with
them withheld and predict the observation at each cell. Scored at cells
visited during explore and non-blank; each cell scored once per episode.

Trained TE=512 TQ=256, 200 epochs. Held-out env (seed=10000).
**Chance 0.0625.** Gates in `MATCH_QUERY_GATES.md` (all at chance).

## Match accuracy

| variant | T_query=256 (train) | T_query=512 (OOD) |
|---|---|---|
| MapWM-Flat | 0.888 ± 0.140 | 0.902 ± 0.117 |
| MapWM-Hier | 0.786 ± 0.227 | 0.771 ± 0.220 |
| Plain-Flat | 0.153 ± 0.012 | 0.127 ± 0.010 |
| Plain-Hier | 0.155 ± 0.020 | 0.131 ± 0.004 |
| PoPE-Flat | 0.117 ± 0.011 | 0.109 ± 0.008 |
| MapPoPE-Hier | 0.847 ± 0.132 | 0.823 ± 0.155 |

## Match NLL (lower better)

| variant | T_query=256 | T_query=512 |
|---|---|---|
| MapWM-Flat | 0.376 ± 0.441 | 0.341 ± 0.382 |
| MapWM-Hier | 0.745 ± 0.778 | 0.817 ± 0.777 |
| Plain-Flat | 2.578 ± 0.042 | 2.635 ± 0.016 |
| Plain-Hier | 2.590 ± 0.057 | 2.636 ± 0.006 |
| PoPE-Flat | 2.668 ± 0.041 | 2.692 ± 0.022 |
| MapPoPE-Hier | 0.510 ± 0.443 | 0.576 ± 0.498 |

## Per-seed (no bimodality; the separation is total)

| variant | path integration? | s0 | s1 | s2 |
|---|---|---|---|---|
| MapWM-Flat | **yes** | 0.731 | 1.000 | 0.934 |
| MapPoPE-Hier | **yes** | 0.775 | 1.000 | 0.768 |
| MapWM-Hier | **yes** | 0.548 | 1.000 | 0.809 |
| Plain-Flat | no (index RoPE) | 0.164 | 0.155 | 0.139 |
| Plain-Hier | no (index RoPE) | 0.176 | 0.153 | 0.136 |
| PoPE-Flat | no (index) | 0.128 | 0.119 | 0.106 |

(T_query=256. Chance 0.0625.)

## Findings

**1. Path integration is necessary, and close to sufficient.** 9/9 runs with
input-dependent path integration beat 9/9 runs with index position. The ranges do
not overlap or even approach each other: worst path-integration seed **0.548**,
best index seed **0.176** -- a 3.1x gap. Several path-integration seeds reach
**1.000**. Index-position models sit at 0.11-0.18 against a chance of 0.0625, so
they learn something, but not a usable map.

**2. The axis is path integration, NOT the positional-encoding scheme.**
MapPoPE-Hier (PoPE *with* MapFormer path integration) scores 0.847, while
PoPE-Flat (PoPE with index position) scores 0.117. Same attention mechanism,
opposite outcome, decided entirely by whether position is path-integrated.

**3. No OOD degradation for path-integration models.** Doubling the blind query
phase from 256 to 512 leaves them flat (MapWM-Flat 0.888 -> 0.902), while the
index models drift slightly down. The map holds over a query phase twice as long
as trained, with observations withheld throughout.

**4. Hierarchy is a NULL result here -- not a harm.** Paired by seed
(Hier - Flat): MapWM -0.183 / +0.000 / -0.125 (mean -0.103), Plain +0.012 /
-0.002 / -0.003 (mean **+0.002**). In the plain backbone hierarchy does nothing
at all; in MapWM it is negative on two seeds and an exact TIE on the third. At
n=3 with a seed spread of 0.548-1.000 that is "no benefit, possibly a small
cost", NOT established harm. An earlier phrasing here ("0/3 wins") overstated it.

This is expected rather than surprising: Match-Query is a flat torus with i.i.d.
observations and the task is per-cell retrieval, so there is no multi-scale
structure to exploit and pooling can only blur the position precision the task
needs. On a task that DOES have structure, hierarchy helps both backbones --
compositional motifs, cross_nb @T=256: MapWM-Flat 0.270 -> MapWM-Hier 0.415
(+0.145), Plain-Flat 0.216 -> Plain-Hier 0.318 (+0.102)
(`COMPOSITIONAL_MULTISEED.md`, unaffected by the 2026-08-09 retractions).

Consistent with the standing framing in CLAUDE.md: hierarchy buys compositional
transfer, backbone-independently, and ~0 on exact recall. Match-Query is an
exact-recall task.

**5. This reverses PoPE-Flat's standing, and that is the point.** On the
invalidated hier-goal task PoPE-Flat looked like the best model in the project
(0.936 held flat to T=2048, +/-0.002). Here, on a task whose shortcuts are gated,
it is **last** at 0.117. That is what a shortcut-driven result looks like once the
shortcut is removed, and it is independent corroboration of the
`HIERGOAL_ABLATION.md` finding.

## Caveats

- n=3, and seed variance among path-integration models is large (0.548-1.000).
  The *separation* is unambiguous; the *ordering within* the path-integration
  group is not resolved at this n.
- One task. The gates pass (all baselines at chance, `MATCH_QUERY_GATES.md`), but
  this is a single new task and has not been replicated elsewhere.
- Chance is 0.0625 by construction (non-blank answers only); the marginal
  baseline is 0.068, so the floor is honest.
