---
name: Minimal sweeps skip GSF, use NoDrop
description: For quick/minimal experiment sets in this project, use Level15NoDrop instead of GSF variants — GSF is accuracy-redundant with NoDrop but ~4x compute.
type: feedback
originSessionId: be30e775-ba9b-48e1-a763-b2488b550411
---
For quick or minimal experimental sweeps (e.g. testing a new architectural idea, scaling claim, or new env regime), include `Level15NoDrop` but **skip** `Level15GSF`, `Level15GSF_NoDrop`, and `Level15GSF_NoDrop_K16` unless the question specifically targets calibration / uncertainty / multi-modal posterior. If a GSF variant was scheduled into a sweep just for "completeness," drop it.

**Why:** From `GSF_NODROP_RESULTS.md` (multi-seed n=3, lm200 OOD T=512): NoDrop and GSF each independently close +13–14pp of the Level15 → TEMFaithful accuracy gap. Stacked, the accuracy gain is essentially the same (+14pp, within seed std). GSF's only distinct contribution is NLL reduction (0.317 → 0.177), which matters for downstream planning / active inference but not for next-token accuracy. GSF (K=8) costs ~3-4× training compute and ~3× inference vs Level15 alone. NoDrop is free (single layer removed).

**How to apply:** When defining variant lists for new run scripts, default to including `Vanilla`, `RoPE`, `Level15`, `Level15NoDrop`, and `TEMFaithful` (when applicable). Add a GSF variant ONLY if the experiment is specifically about (a) calibrated uncertainty, (b) active-inference / world-model planning that consumes the posterior, or (c) explicit multi-modal-hypothesis tracking. For "does this new test discriminate cognitive-map architectures" type sweeps, the GSF row adds compute without adding signal.
