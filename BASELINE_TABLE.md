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
**One batch** (`runs/family_tree`). No plain-WM arm exists — a known gap.

| model | T=64 (train) | T=128 (OOD) |
|---|---|---|
| MapEM-NC-NL (non-commutative, MLP) | **0.729 ± 0.010** | **0.672 ± 0.012** |
| MapEM-NC-L (non-commutative, linear) | 0.720 ± 0.011 | 0.671 ± 0.006 |
| MapEM-os (COMMUTATIVE control) | 0.715 ± 0.008 | 0.659 ± 0.014 |
| Plain-Flat (index) | 0.600 ± 0.011 | 0.550 ± 0.031 |

Non-commutativity buys **+0.014** over the commutative control for **34x** the
compute (`TIMING_BENCHMARK.md`).

---

## G. Landmark regime (lm200) — the correction line's home turf

n=3 · fresh retrains under current code · **one batch**
Supersedes every April lm200 table; those ranked training convergence.

| model | T=128 | T=512 (OOD) |
|---|---|---|
| Level15 | **0.996 ± 0.003** | **0.990 ± 0.005** |
| TEMFaithful | 1.000 ± 0.000 | 0.974 ± 0.008 |
| Level15GSF | 0.982 ± 0.025 | 0.967 ± 0.034 |
| Level15NoDrop | 0.981 ± 0.014 | 0.956 ± 0.029 |
| Level15EM | 0.938 ± 0.045 | 0.823 ± 0.110 |
| MapWM-Flat | 0.814 ± 0.073 | 0.742 ± 0.075 |
| PC | 0.888 ± 0.041 | 0.716 ± 0.012 |
| MapEM (sep q0/k0) | 0.830 ± 0.050 | 0.656 ± 0.130 |
| MambaLike | 0.562 ± 0.010 | 0.549 ± 0.013 |
| RoPE | 0.636 ± 0.042 | 0.482 ± 0.023 |

**Never gated**: lm200 has had no context-destruction ablation (rule 2), in the
one regime where an entire leaderboard already turned out to be an artifact.

---

## Coverage gaps — what is missing and why it matters

| gap | consequence |
|---|---|
| Level15 & the correction family absent from A/B/D/E/F | the correction line meets the gated tasks only on Match-Query (table C), where it shows no advantage |
| MapPoPE absent from C (flat form), F, G | the best model on the paper task is untested on 3 of 7 tasks |
| No plain-WM arm on the family tree | the WM/EM axis is untested there |
| TEMFaithful / MambaLike / LSTM only in G | no baseline row on the paper's own task |
| lm200 not gated | rule 2 unapplied to a headline result |

## Which comparisons are safe

- **Within a table**: yes where the provenance line says one batch (A, B, E, F, G).
- **Table D**: accumulated across batches — prefer table E.
- **Across tables**: no. Different floors, different seeds, different batches.
  Cross-task statements need a within-batch design, which is what E does for
  compositional and what tables A/B do for the paper task.
