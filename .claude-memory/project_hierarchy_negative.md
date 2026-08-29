---
name: project_hierarchy_negative
description: "Hierarchy does not help MapFormer — both a Kalman cascade (on theta) and a two-scale attention (on retrieval) are clean negatives, for different mechanistic reasons."
metadata: 
  node_type: memory
  type: project
  originSessionId: d17148dd-e54f-48b7-8a88-096f71aadfc5
  modified: 2026-07-21T04:16:43.472Z
---

Two hierarchy extensions to MapFormer were built and tested (2026-07);
both are honest negatives:

1. **Kalman cascade (Axis 3, `model_inekf_cascade.py`)** — a slow per-chunk
   Kalman filter on top of Level 1.5's fast filter, correcting theta. NO
   benefit: fresh Cascade == NoSlow == Level 1.5 (~0.996 lm200). The slow
   filter is inference-inert (zeroing it: 0.000 change) and collapses to a
   no-op in training. Reason: Level 1.5's theta estimate is already good —
   no exploitable residual structure for a second correction. (The apparent
   win was the stuck-baseline artifact, see
   [[feedback_lm200_stuck_baselines]].)

2. **Two-scale attention (`model_hier_attn.py::MapFormerWM_HierAttn`)** —
   local causal window (W=128) + coarse attention over mean-pooled chunk
   summaries (C=64), same param count as flat Level 1.5. Flat attention
   BEATS it at every length (clean, train n_steps=256, eval to T=4096:
   0.861 vs 0.769; gap does not close). Reason: chunk-pooling (mean K/V over
   64 tokens) destroys the per-token info needed to retrieve a specific
   aliased observation. Full-resolution single-scale retrieval > hierarchical
   pooled retrieval; pooling loss > softmax-dilution cost. The "dilution at
   long T" hypothesis was falsified.

3. **Attention hierarchy at MULTI-ENV TRANSFER** (`HIER_ATTN_MULTIENV.md`) —
   the regime the neuroscience most endorses (reusable structural code should
   pay off at transfer). HierAttn vs flat Level 1.5, 50 train / 50 held-out
   envs. Still no benefit: held-out clean TIED (0.947/0.947 at T=512), held-out
   lm200 flat WINS +5pp (0.993 vs 0.941). HierAttn fit training envs faster
   (lazy shortcut) but transferred equal or worse. Reason: transfer rewards the
   reusable STRUCTURAL code (omega-spectrum position), which MapFormer already
   has and which is already near-ceiling at transfer (existing MULTIENV data:
   even TEMFaithful can't beat Level 1.5). HierAttn pools env-specific CONTENT
   and still destroys per-token retrieval, so it hurts on landmarks either way.

**An aggregate-task "win" appeared, then was RETRACTED as a training-length
confound** (`AGGREGATE_EXTRAS.md`, commit 83e505d). Sequence worth remembering:

- On an AGGREGATE readout (windowed-majority obs-type instead of exact-obs
  retrieval), HierAttn beat flat at length, multi-seed n=3, tight bars
  (T=2048: 0.537±0.004 vs 0.401±0.012). Looked like "hierarchy's value is
  task-determined."
- **Ablation killed the mechanism:** HierAttn_LocalOnly (windowed attention,
  ZERO pooling) ALSO beats flat at length (0.453 vs 0.383 @T=2048). So the
  advantage is a BOUNDED ATTENTION SPAN, not pooling-as-summary.
- **Training-length control killed the claim:** flat trained at n_steps=512
  matches/beats HierAttn trained at 256 at EVERY length (0.746/0.629/0.539 vs
  0.741/0.628/0.531). Flat has NO aggregation deficit — it just needed to
  train nearer the target length.
- **Nested-room (hierarchical SPACE)** showed no real dissociation either:
  flat wins revisit (0.916 vs 0.798 @T=2048); room_novel roughly tied (flat
  better at short T, hier +3pp only at T=2048). Both infer room themes fine.

**Correct claim:** bounded-span attention (windowed OR pooled) extrapolates
beyond its training length; unbounded flat attention does not. That is a
length-EXTRAPOLATION property, NOT an aggregation CAPABILITY. Hierarchy is not
needed for aggregation, does not benefit from hierarchical space, and costs
retrieval accuracy.

**Takeaway:** hierarchy still does not earn its keep in MapFormer on any task
tested (retrieval, aggregation, multi-env transfer, hierarchical space). Its
one real benefit is length-extrapolation from SHORT training — obtainable more
cheaply by just training longer, or by any span-bounding trick. Interpretive
value stands (MapFormer as single-level Lie-group predictive coding;
precision-weighting = attention).

**Bounded-memory test (the brain's actual regime) — also negative.** Gave both
models a hard 128-item READ BUDGET and varied only how it is spent
(`BOUNDED_MEMORY_RESULTS.md`, `model_bounded_mem.py`). BoundedFlat = all budget
on recency; BoundedHier = M/2 recent + M/2 summaries covering ALL history.
n=3, sigma~0.000. Flat WINS at every length past T=256 (T=4096: 0.770 vs
0.752) despite seeing only the last 3% of history. Measured revisit-lag
distribution: median 8 steps at T=256 (95.5% inside flat's window), 40 steps
at T=4096 (52.8% inside, 39.7% >256 back). So old-evidence demand is real at
long T and BoundedHier covers it — but it trades half its high-resolution
recent window for coverage mean-pooled over 32 cells, too lossy to identify a
SPECIFIC old observation. It pays capacity for coverage it cannot exploit.

**THE UNIFYING PRINCIPLE (use this framing):** hierarchy helps only when a
lossy summary is a SUFFICIENT STATISTIC for the task.
  - Exact recall / retrieval -> no summary suffices -> hierarchy is FORCED to
    lose (explains all 5 retrieval negatives; they are not accidents).
  - Aggregation -> summaries suffice -> hierarchy helps, but the same benefit
    is obtainable more cheaply (train longer / any bounded span).
This benchmark is exact-recall, i.e. precisely the one task family where
hierarchical compression is provably useless. Reconciles with the brain: brains
are hierarchical for gist-sufficient problems under inescapable capacity limits,
and keep a separate high-resolution episodic store (hippocampus) exactly where
precision is needed — which is the architecture these results keep implying
(full-res recent store + hierarchy only for gist-sufficient queries).


**Planning-task attempts (2026-07-16) — all invalid or uninformative.** Tried
to find a regime where hierarchy wins, per the literature (options/HRL). Four
task-validity failures in sequence:
  1. open-plan rooms+goals — 100% of BFS-optimal actions are GREEDY, so no
     planning problem exists. Flat==hier exactly. Vacuous.
  2. fixed spanning-tree maze — greedy drops to 0.70, so planning is required,
     but the maze is FIXED: models scored 0.94 on it and collapsed to 0.68
     (below greedy) on a NOVEL maze. Pure memorisation. Flat==hier (0.939),
     no distance trend. Ablations: LocalOnly 0.944 (best), CoarseOnly 0.770
     (barely above greedy — the pooled pathway alone cannot select actions).
  3. varying maze (fresh maze/landmarks per episode, memorisation impossible)
     — ALL variants ~0.50, BELOW the greedy baseline 0.73. Both models failed,
     so the comparison is UNINFORMATIVE, not a negative.
     Diagnosis: 28.1% of maze moves are BLOCKED by walls; MapFormer's
     cumsum(action) path integration assumes every action executes, so theta
     desynchronises on >1/4 of steps with no bump/collision token to correct
     from. MapFormer structurally cannot dead-reckon in a maze — a fact about
     its ACTION MODEL, not about hierarchy.

Stopped here by pre-commitment rather than iterating environments until
hierarchy wins (that would be p-hacking with environments). A bump-token fix
is principled but belongs to a different question: "can MapFormer navigate
mazes at all?" See [[feedback_validate_task_first]].

**Method lesson:** always run the training-length control before claiming a
length-generalization win, and ablate components before claiming a mechanism.
Both confounds fired here. Also: MEASURE the task's information-demand profile
(e.g. revisit-lag distribution) before assuming a memory mechanism is needed.
See [[feedback_seed_ordering]].

Untested variants (lower priority): omega-band-structured heads (structures
attention range by frequency band, no token pooling — avoids the HierAttn
failure mode but has its own uncertainties); depth hierarchy / per-layer
correction (Axis 4, needs multi-layer stability work first).

**TEXT, added 2026-08-28 — same verdict, cleanest version of it.** enwik8 byte-level,
MapWM-Hier 1.4537 vs MapWM-FlatHG 1.4506 bpc at EXACT param parity (28,371,016 both;
identical 3-block scaffold, only the middle block's k=2 pooling differs). Hierarchy is
+0.0032 WORSE, inside the 0.003–0.007 checkpoint sd → a null on quality. The one
measurable effect is **−8.6% wall time**. Reproduces the plain-family direction
(1.4844 vs 1.4727 at −18.75% FLOPs) with parity and deterministic val added.

**Two families now agree: on text, hierarchy is an EFFICIENCY property, not a quality
win.** Exactly what the sufficient-statistic principle predicts — next-byte prediction
is exact-recall, so a lossy summary can only lose. n=1 per arm, so this is consistent
with a null rather than proof of one.

