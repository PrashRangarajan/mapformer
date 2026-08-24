# Baseline table — every model on every task, with provenance

A jumping-off reference. **Read the provenance line under each table before
comparing across tables.** Numbers from different training batches are not
directly comparable (standing rule 3), and the lm200 leaderboard is what happens
when that is ignored: it once ranked training convergence rather than
architecture.

Each table states its own **measured floor**. Floors differ by an order of
magnitude between tasks (0.0625 to 0.80), so a raw number means nothing without
its column (standing rule 4).

---

## THE RESULT THE TABLE NOW SHOWS

**No architectural ingredient is best. Which one to spend on is decided by the
environment, and one environment property decides most of it.**

The same 2x2 — {RoPE, PoPE} encoding x {index, path-integrated} position — run on
two environment families, each factor averaged over the other:

| | encoding | hierarchy | position |
|---|---|---|---|
| **torus paper task** (n=8) | +0.011 | — | **+0.461** |
| **MiniGrid DoorKey-16x16** T=512 (n=8) | **+0.035** | +0.022 | −0.005 |
| **MiniGrid DoorKey-16x16** T=1024 (n=8) | **+0.076** | +0.048 | **−0.021** |

*(MiniGrid figures average all four pairs of the complete 8-cell factorial.)*

Position is worth **40x** the encoding on the torus and is *negative* on
MiniGrid. Section I isolates why: of the five properties that differ between
them, **rotation-based actions account for −0.388 of the −0.438 swing** (n=8) —
more than the other four combined. Section I also gives the remedy: recording
the action stream as absolute displacement rather than turn/forward restores the
effect completely at four headings (+0.050 → +0.488), and at Habitat's twelve as
well once the budget is adequate (+0.264 → +0.383 and climbing; the earlier
"partial" reading was undertraining).

**And the ordering it produces is pointed**: on MiniGrid the two best models of
eight are *index-position* models with no path integration at all, and the
paper's own MapFormer-WM is last.

---

## A. Paper task — held-out revisit accuracy, fresh obs_map

n=8 seeds · 1 layer, d=128, T=128, 16 epochs · **floor 0.506** (always-predict-blank)
**One batch, all six arms** (`runs/paper_task_n8`) → cross-model comparison VALID.

| model | position code | fresh-map acc |
|---|---|---|
| **MapPoPE-Flat** | PoPE + path integration | **0.994 ± 0.017** |
| MapEM-os (single p_0) | path integration | 0.987 ± 0.009 |
| MapWM-Flat | path integration | 0.967 ± 0.039 |
| Plain-Flat | index RoPE | 0.534 ± 0.040 |
| RoPE | index (MapFormer arch) | 0.530 ± 0.043 |
| PoPE-Flat | PoPE + index | 0.509 ± 0.001 |

*Paper reports (Table 2, 2D, IID): MapWM 0.99, MapEM-os 1.0.*

---

## B. Paper task under OOD — the paper's own protocol, plus a length extension

n=8 · same checkpoints as A · **floor rises with p_empty**: ~0.50 at IID, ~0.20 at
OOD-d, ~0.80 at OOD-s. A 0.80 on OOD-s is the floor, not a score.

| model | IID | OOD-d | OOD-s l=512 | ext l=1024 | ext l=2048 |
|---|---|---|---|---|---|
| **MapPoPE-Flat** | **0.994** | **0.991** | **0.992** | **0.985** | **0.970 ± 0.028** |
| MapEM-os | 0.987 | 0.983 | 0.978 | 0.963 | 0.939 ± 0.020 |
| MapWM-Flat | 0.969 | 0.958 | 0.943 | 0.893 | 0.854 ± 0.026 |
| *paper reports* | *0.99 / 1.0* | *0.99* | *0.96 / 0.97* | — | — |

At l=2048: MapPoPE-Flat vs MapWM **+0.116** (t=8.6); vs MapEM-os **+0.031**
(t=2.55, p≈0.02, **±1sd bands overlap**). The `ext` columns are ours — no
published counterpart exists.

---

## C. Match-Query — blind continuation, shortcut-gated, ablation-verified

n=3 (n=5 on the base pair) · **chance 0.0625, operative floor 0.0893** (never-moved)
**One batch of six** (`runs/match_query`); the three EM arms are a second batch
whose Vanilla control reproduced 0.888 to three decimals, so they are comparable.

| model | TQ=256 (train) | TQ=512 (OOD) |
|---|---|---|
| MapWM-Flat | **0.888 ± 0.140** | **0.902 ± 0.117** |
| MapPoPE-Hier | 0.847 ± 0.132 | 0.823 ± 0.155 |
| MapEM-os (single p_0) | 0.808 ± 0.168 | 0.789 ± 0.188 |
| MapWM-Hier | 0.786 ± 0.227 | 0.771 ± 0.220 |
| Level15 | 0.876 ± 0.213 | 0.853 ± 0.254 |
| MapEM sep-q0/k0 (paper-faithful) | 0.450 ± 0.332 | 0.385 ± 0.323 |
| Plain-Hier | 0.155 ± 0.020 | 0.131 ± 0.004 |
| Plain-Flat | 0.153 ± 0.012 | 0.127 ± 0.010 |
| PoPE-Flat | 0.117 ± 0.011 | 0.109 ± 0.008 |

At n=5 the base MapWM-Flat figure is **0.730 ± 0.247** — cite that one.
Level15 is from a separate 3-arm batch (`runs/level15_meets`) whose Vanilla
control also reproduced 0.888 exactly.

---

## D. Compositional — motif transfer and precise recall, same models

n=3-8 · **cross_nb floor ~0.072** · from `COMPOSITIONAL_MULTISEED.md`
Accumulated over several batches — **the weakest provenance here.** The fresh
single-batch version is table E.

| model | cross_nb @T=256 | exact_acc @T=256 |
|---|---|---|
| MapWM-Hier | **0.415 ± 0.096** (n=8) | 0.959 ± 0.027 |
| Plain-Hier | 0.318 ± 0.029 (n=8) | 0.918 ± 0.005 |
| MapWM-Flat | 0.270 ± 0.030 (n=3) | 0.924 ± 0.020 |
| Plain-Flat | 0.216 ± 0.004 (n=8) | 0.904 ± 0.002 |
| MapEM-Flat | 0.097 | — |

---

## E. Compositional, swept by structure — ONE fresh batch, 4 arms × 4 points

n=3 · **one batch** (`runs/dissociation`) → the cleanest cross-model comparison
on this task. `n_templates` = distinct room motifs; low = more structure.

`cross_nb` @T=256 (floor ~0.072):

| model | nt=2 | nt=4 | nt=8 | nt=16 |
|---|---|---|---|---|
| MapWM-Hier | **0.560** | **0.413** | 0.347 | 0.219 |
| MapWM-Flat | 0.365 | 0.260 | **0.360** | **0.291** |
| Plain-Hier | 0.446 | 0.316 | 0.220 | 0.183 |
| Plain-Flat | 0.316 | 0.200 | 0.139 | 0.099 |

`exact_acc` @T=1024:

| model | nt=2 | nt=4 | nt=8 | nt=16 |
|---|---|---|---|---|
| MapWM-Hier | **0.761** | **0.864** | 0.856 | 0.758 |
| MapWM-Flat | 0.610 | 0.765 | **0.873** | **0.803** |
| Plain-Hier | 0.670 | 0.682 | 0.637 | 0.635 |
| Plain-Flat | 0.581 | 0.629 | 0.592 | 0.619 |

---

## F. Family tree — non-commutative relational structure

n=3 · **floor 0.163** (hub-node baseline; the 0.125 chance is NOT the floor)
**One batch of five** (`runs/correction_gaps/familytree`, 2026-08-19).
Supersedes `FAMILY_TREE_RESULTS.md`, whose three arms this batch reproduces to
three decimals — see `FAMILY_TREE_WM_GAP.md`.

| model | T=64 (train) | T=128 (OOD) |
|---|---|---|
| **Level15** | **0.843 ± 0.015** | **0.789 ± 0.027** |
| **MapWM-Flat** | 0.805 ± 0.072 | 0.746 ± 0.080 |
| MapEM-NC-NL (non-commutative) | 0.729 ± 0.010 | 0.672 ± 0.012 |
| MapEM-os (commutative control) | 0.715 ± 0.008 | 0.659 ± 0.015 |
| Plain-Flat (index) | 0.601 ± 0.011 | 0.550 ± 0.031 |

Two corrections to the earlier version of this table:
- **The plain-WM arm was missing and is the best of the published set** (+0.076
  over MapEM-NC-NL, five times the margin the non-commutativity comparison turns
  on). Non-commutativity still buys +0.014 for 34x the compute — but it does so
  *below* plain MapWM-Flat.
- **Level15's +0.038 is variance reduction, not a mean gain.** Paired per seed:
  −0.005 / +0.001 / **+0.117**; two exact ties, the whole difference is one
  Vanilla seed collapsing to 0.724. t≈0.89, not significant. The real effect is
  the spread: ±0.015 vs ±0.072.

## G. Landmark regime (lm200) — gated 2026-08-19, interpretation withdrawn

n=3 · fresh retrains under current code · **one batch**

| model | T=128 | T=512 (OOD) |
|---|---|---|
| Level15 | 0.996 ± 0.003 | **0.990 ± 0.005** |
| TEMFaithful | 1.000 ± 0.000 | 0.974 ± 0.008 |
| Level15GSF | 0.982 ± 0.025 | 0.967 ± 0.034 |
| Level15NoDrop | 0.981 ± 0.014 | 0.956 ± 0.029 |
| Level15EM | 0.938 ± 0.045 | 0.823 ± 0.110 |
| MapWM-Flat | 0.814 ± 0.073 | 0.742 ± 0.075 |
| PC | 0.888 ± 0.041 | 0.716 ± 0.012 |
| MapEM (sep q0/k0) | 0.830 ± 0.050 | 0.656 ± 0.130 |
| MambaLike | 0.562 ± 0.010 | 0.549 ± 0.013 |
| RoPE | 0.636 ± 0.042 | 0.482 ± 0.023 |

**Gate: PASSES** (`LM200_ABLATION.md`). Resampling the action stream collapses
Level15 0.985 → 0.155 (−0.830, comparable to Match-Query's −0.842). Not a
shortcut artifact.

**But read the ordering with three caveats, all measured 2026-08-19:**

1. **A capacity control ties Level15 and is absent from this table.**
   `Vanilla_ExtraHead` — generic extra head, no filter, more parameters — scores
   **0.995 ± 0.007** at T=128 against Level15's 0.985 ± 0.021 (t=0.79). The gap
   over plain MapWM-Flat is real; it is not evidence for the Kalman mechanism.
2. **MapWM-Flat's non-convergence reproduces.** One of three fresh seeds finished
   at training loss 0.988 — the failure that voided every April lm200 checkpoint
   — with accuracy 0.722 vs 0.894/0.913. The regime is basin-sensitive under
   current code, so "Level15 beats Vanilla here" is partly "Vanilla sometimes
   fails to converge here".
3. **This regime was built for the correction.** `environment.py`: landmarks are
   "the regime where Kalman/PC corrections have sharp measurements". It is not
   the paper's task, and a win here is close to circular.

## H. MiniGrid DoorKey-16x16 — a published external benchmark

n=3 · **one batch per grid** · measured floors 0.642 / 0.536 / 0.495
Second environment family; egocentric observation, rotation actions, 256 cells.

**Full factorial — all 8 cells, `n_layers=3` (parameter-matched), n=8 seeds,
floors 0.536 / 0.490:**

| model | position | T=512 | T=1024 |
|---|---|---|---|
| **PoPE-Hier** | **index** + hier | 0.964 ± 0.002 | **0.955 ± 0.003** |
| PoPE-Flat | **index** | 0.963 ± 0.002 | 0.953 ± 0.003 |
| MapPoPE-Hier | path-int + hier | 0.966 ± 0.009 | 0.942 ± 0.017 |
| RoPE-Hier | **index** + hier | 0.950 ± 0.004 | 0.924 ± 0.006 |
| MapPoPE-Flat | path-int | 0.959 ± 0.012 | 0.919 ± 0.026 |
| MapWM-Hier | path-int + hier | 0.945 ± 0.014 | 0.893 ± 0.023 |
| RoPE-Flat | **index** | 0.914 ± 0.019 | 0.827 ± 0.044 |
| MapWM-Flat | path-int | 0.902 ± 0.058 | 0.823 ± 0.088 |

**The two best models on this benchmark have no path integration.** `PoPE-Hier`
and `PoPE-Flat` lead at T=1024 (0.955, 0.953) with the tightest spreads of any
arm (±0.003), ahead of the full path-integration + hierarchy stack (0.942). The
paper's own MapFormer-WM is last (0.823).

**Hierarchy helps in inverse proportion to the strength of its base** — the
cleanest reading of the factorial, now that all four pairs exist:

| pair | base score @T=1024 | hierarchy gain |
|---|---|---|
| RoPE + index | 0.827 | **+0.096** (8/8) |
| RoPE + path-int | 0.823 | +0.070 (7/8) |
| PoPE + path-int | 0.919 | +0.023 (7/8) |
| PoPE + index | **0.953** | **+0.002** (5/8) |

It is compensation, not addition: 27/32 paired comparisons positive overall, but
essentially zero for the strongest arm. *The earlier n=3 report of "18/18, every
cell every seed" was small-sample luck.*

`PoPE-Flat` was retrained alongside the new cell as a reproducibility control and
matched its previous n=8 figures exactly (0.963 / 0.953), which licenses reading
the 8th cell against the rest of the grid.

### Frequency control (`FREQ_CONTROL.md`)

Path-integrated arms use learnable `omega` (`nn.Parameter`); index arms use fixed
frequencies (`register_buffer`). Those properties are perfectly correlated across
every cell of every grid, so "position effect" has always meant "position AND
frequency learning". Freezing `omega` breaks the correlation:

| effect | T=512 | T=1024 |
|---|---|---|
| frequency learning | +0.004 | −0.008 |
| **pure position** | **−0.042** | **−0.060** |

The confound is real in the code and **absent from the data**; the position
effect survives and grows. Side finding: MapFormer's learnable angular
velocities (App. A.8) buy nothing here — 614,474 trainable params match 614,538.

## I. What makes path integration decisive or worthless

`KNOB_SWEEP.md`. Five properties differ between the torus and MiniGrid; each
turned one at a time from the torus baseline, Vanilla and RoPE retrained together
at every condition, n=3, floors measured per condition.

| condition | position effect | reduction from baseline | n |
|---|---|---|---|
| baseline (torus paper task) | **+0.438** | — | 8 |
| **rotate** (turn/turn/forward) | **+0.050** | **−0.388** | 8 |
| wall (bounded, bumps are no-ops) | +0.251 | −0.187 | 3 |
| ego (observe the cell ahead) | +0.265 | −0.173 | 3 |
| richobs (64 obs types not 16) | +0.299 | −0.139 | 3 |
| small (16² not 64²) | +0.324 | −0.114 | 3 |
| **all five combined** | **−0.076** | −0.514 | 8 |
| **rotate + allocentric recoding** | **+0.488** | **+0.050** | 8 |

**Rotation actions dominate**, at twice the next knob and 90% of the available
swing. Mechanism: MapFormer path-integrates by cumsumming a *fixed per-token*
delta, and under turn/turn/forward the displacement depends on accumulated
heading — which that form cannot represent.

**And the mechanism is confirmed with a fix** (`ALLOCENTRIC_RECODING.md`).
Changing only what the token stream RECORDS — the absolute displacement instead
of the commanded turn/forward, dynamics byte-identical — restores MapFormer from
0.558 to **0.996** and the position effect from +0.050 to **+0.488**, against
baseline's +0.438. A complete recovery on **8/8 seeds (±0.005)** — it now
*exceeds* the translate baseline. So rotate's
collapse is a representation mismatch, not task difficulty, and the remedy is
available wherever the agent's heading is known — i.e. every simulator.

**The combination reproduces MiniGrid.** All five knobs on gives −0.084 against
MiniGrid's independently measured −0.060, so the five are jointly sufficient and
nothing important is missing from the list.

Two caveats: the decomposition is **not additive** (single-knob reductions sum to
−1.20 against a combined −0.56, so the knobs interact), and the `rotate` row is a
**redo** — the first version was void with a 0.932 order-1 shortcut, caught by
gating after training instead of before.

## Coverage gaps — what is missing and why it matters

| gap | status |
|---|---|
| ~~Level15 absent from the paper task~~ | **CLOSED**: 1.000 ± 0.000 at 50 epochs vs Vanilla 0.993; at the paper's own 16-epoch budget it reads 0.938, a budget artifact |
| ~~Level15 absent from the family tree~~ | **CLOSED** (F): variance reduction, no significant mean gain |
| ~~Level15 absent from compositional~~ | **CLOSED**: worse on 2/3 seeds, better on likelihood 3/3 |
| ~~No plain-WM arm on the family tree~~ | **CLOSED** (F): it was the best of the published set |
| ~~lm200 never gated~~ | **CLOSED** (G): passes, interpretation withdrawn |
| ~~Single environment family~~ | **CLOSED** (H): MiniGrid, full factorial, **n=8** |
| ~~Position/frequency confound~~ | **CLOSED** (H): measured, empirically negligible |
| ~~PoPE + index + hierarchy~~ | **CLOSED**: built, verified index-based, and it is the best arm on the benchmark |
| ~~allocentric action recoding~~ | **CLOSED** (I): complete recovery at 4 headings (+0.050 → +0.488). At Habitat's 12 headings the apparent partial recovery was UNDERTRAINING — a budget sweep takes it +0.264 → +0.383 with the seed spread collapsing ±0.101 → ±0.005 (`CONTINUOUS_ALLOC.md`) |
| **capacity control at lm200 T=512** | the +24.8pp headline has no ExtraHead arm at that length |
| **MapPoPE on C, F, G** | the best model on A/B/H is untested on three tasks |
| **TEM / Mamba / LSTM on A** | they exist only in the lm200 column |
| **scale** | everything is 200K-3.2M params, d=128, 1-4 layers; the horizon grid showed capacity moves conclusions |
| **3D / continuous embodiment** | MiniWorld, Memory Maze, Habitat — all have I's dominant knob (rotation actions) and none has been run |

## Which comparisons are safe

- **Within a table**: yes where the provenance line says one batch (A, B, E, F, G).
- **Table D**: accumulated across batches — prefer table E.
- **Across tables**: no. Different floors, different seeds, different batches.
  Cross-task statements need a within-batch design, which is what E does for
  compositional and what tables A/B do for the paper task.

---

## The correction family across tasks — five within-batch tests

| task | accuracy vs MapWM-Flat | likelihood / stability |
|---|---|---|
| paper task, 50 ep | 1.000 vs 0.993 — ceiling effect | **16x lower loss** |
| Match-Query | 0.876 vs 0.888 — **no advantage** | — |
| family tree | not significant (t≈0.89) | **±0.015 vs ±0.072** |
| lm200 | +0.142, but a filter-free capacity control ties it (t=0.79) | worst-seed loss 0.188 vs 0.988 |
| compositional | **worse on 2/3 seeds** | **better 3/3, both lengths** |

Across five within-batch comparisons Level 1.5 improves **likelihood and
stability** reliably and **accuracy** almost never. That is the "stabilisation,
not inference" reading in CLAUDE.md, measured on five tasks rather than asserted,
and much narrower than the Kalman framing the project was built on.

## Reading this table

- **Within a section**: valid where the provenance line says one batch
  (A, B, E, F, G, H, I).
- **Section D**: accumulated across batches — prefer E.
- **Across sections**: no. Floors range from 0.0625 to 0.80 and batches differ.
  Cross-task statements need a within-batch design, which is what E does for
  compositional, A/B for the paper task, H for MiniGrid and I for the knobs.
- **Every floor here is measured**, not assumed. An index model reading 0.80 on
  OOD-s is sitting *on* its floor; the same model reading 0.27 on OOD-d is doing
  the same thing. The number means nothing without its column.
