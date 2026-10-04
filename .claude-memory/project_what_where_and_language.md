---
name: project-what-where-and-language
description: The 2026-09-28..10-03 line -- text world, context-dependent step, what/where separation, new objects, the what-to-where leak and NormStep -- results with their status, and what is ours vs prior art.
metadata:
  type: project
---

Results (files are authoritative; full account `docs/SESSION_2026-09-27_to_10-03.md`):
- Text world, REGISTERED: PATH WINS IN WORDS (`TEXTWORLD_RESULTS.md`; fresh seeds +0.474, 6/6 vs 0/6).
- Context step, PILOTS only (`CTXSTEP_*.md`); the registered batch was STOPPED before any result. HSR (word step +
  alpha * LN(h1), alpha init 0) learned a step 4/4 on far cues; window steps (CG, SR) work only within ~3 tokens.
- What/where, POST HOC (`docs/WHAT_WHERE_ANALYSIS.md`, `docs/WHAT_WHERE_CHECKS.md`): converged path models do not
  need the content x position interaction (one shared kernel, content as gain; token TYPE needed); RoPE/PoPE do;
  separation without map redraw on a 32x32 map; a memorised 100-cell map has no "where".
- Leak, REGISTERED (`LEAK_RESULTS.md`): MapWM's object tokens move the phase a little (the step reads the embedding
  before LayerNorm). ActOnly and NormStep (step reads LN(emb)) both remove it: +0.0107 unseen-object accuracy in
  distribution (p 0.0002), 16/16 SOLVED vs MapWM 8/8 DESCENDING (partly training speed, r -0.94). The registered
  x4 robustness test could not fail (scale invariance by construction; Amendment 1).
- NormStep, ANALYSIS (`docs/NORMSTEP_NOTES.md`): zeroing its observation steps (-0.19) removes a per-move gauge, not
  leak; its object-identity leak is ~5x smaller than MapWM's (2 seeds). Not provable that it must be smaller.
  Predicted risk: on language the LN bias step is a per-token (word-count) clock -- untested (~3 h, text world
  with a bias-free arm).

**Prior art (do not claim; `docs/lit/LIT_*.md`):** context-gated steps are Mamba/Selective-RoPE/CoPE forms; alpha=0
residual init is ReZero/Flamingo/RWKV-7; "separation is learned" is in MapFormer's own Fig. 9; transfer to unseen iid
codes is Chen et al. 2019; a content-free score is MapFormer's MapEM-s / TEM-t; an action-only step is TEM-t's;
TEM never tested unseen objects (only new arrangements of a fixed set).

**Possibly ours:** step behaviour vs cue distance x generator source (pilot); the causal form of separation in path
models; separation without map redraw, tracking success across seeds; the measured leak mechanism (pre-LN step,
4-d row space) and its in-distribution cost (+0.0107).

**Why:** the user asked what is new; three literature agents found most design ideas published. **How to apply:**
frame these as measurements in a new regime, not new mechanisms; check `docs/lit/` before claiming novelty; say
which rows are registered, pilot or post hoc.
