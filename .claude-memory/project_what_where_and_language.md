---
name: project-what-where-and-language
description: The 2026-09-28..10-03 line -- text world, context-dependent step, what/where separation, new objects, leak -- what is ours vs prior art.
metadata:
  type: project
---

Results (files are authoritative): text world PATH WINS IN WORDS (`TEXTWORLD_RESULTS.md`, registered); context-step
pilots (`CTXSTEP_*.md`, pilots only, registered batch stopped); what/where analysis + checks
(`docs/WHAT_WHERE_ANALYSIS.md`, `docs/WHAT_WHERE_CHECKS.md`, post hoc); new-object pilot; leak-remedy batch
(`LEAK_PREREG.md`, running/landed 2026-10-03).

**Prior art (do not claim; `docs/lit/LIT_*.md`):** context-gated steps are Mamba/Selective-RoPE/CoPE forms; alpha=0
residual init is ReZero/Flamingo/RWKV-7; "separation is learned" is in MapFormer's own Fig. 9; transfer to unseen iid
codes is Chen et al. 2019; a content-free score is MapFormer's MapEM-s / TEM-t; TEM never tested unseen objects
(only new arrangements of a fixed set).

**Possibly ours:** step behaviour vs cue distance x generator source (pilot); the causal form of separation in path
models (one shared kernel, content as gain, token type required; RoPE/PoPE depend on the interaction); separation
without map redraw and tracking success across seeds; the leak mechanism (step reads the embedding before LayerNorm,
4-d row space). Context-free MapFormer cannot ignore "did not go north"; window steps reach ~3 tokens either side.

**Why:** the user asked what is new; three literature agents found most design ideas published. **How to apply:**
frame these as measurements in a new regime, not new mechanisms; check `docs/lit/` before claiming novelty.
