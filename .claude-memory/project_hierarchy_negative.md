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

**Takeaway:** MapFormer's single-scale, full-resolution mechanisms are
already near-optimal for this task; adding hierarchy (on theta or on
attention) does not help. The remaining value of the hierarchy line is
interpretive (MapFormer as single-level Lie-group predictive coding;
precision-weighting = attention) and the negative results themselves.

Untested variants (lower priority): omega-band-structured heads (structures
attention range by frequency band, no token pooling — avoids the HierAttn
failure mode but has its own uncertainties); depth hierarchy / per-layer
correction (Axis 4, needs multi-layer stability work first).
