---
name: project_hierarchy_negative
description: "Hierarchy does not help MapFormer — both a Kalman cascade (on theta) and a two-scale attention (on retrieval) are clean negatives, for different mechanistic reasons."
metadata: 
  node_type: memory
  type: project
  originSessionId: d17148dd-e54f-48b7-8a88-096f71aadfc5
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

**Method lesson:** always run the training-length control before claiming a
length-generalization win, and ablate components before claiming a mechanism.
Both confounds fired here. See [[feedback_seed_ordering]].

Untested variants (lower priority): omega-band-structured heads (structures
attention range by frequency band, no token pooling — avoids the HierAttn
failure mode but has its own uncertainties); depth hierarchy / per-layer
correction (Axis 4, needs multi-layer stability work first).
