---
name: project-what-where-and-language
description: [updated 2026-10-29 with TW_STATECHANGE, GAIN_PHASE, the re-score and IMAGE_TORUS] The 2026-09-28..10-03 line -- text world, context-dependent step, what/where separation, new objects, the what-to-where leak and NormStep -- results with their status, and what is ours vs prior art.
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
  Predicted risk tested (`TW_NORMSTEP_RESULTS.md`, REG, 2026-10-04): on the text world NormStep's accuracy is
  unmeasured-equal to MapWM's; the registered "per-word clock" (+0.057 rad) is the ASIDE offset (asides move their
  nouns off the cell's phase), not a word counter (CORRECTED 2026-10-04).
- Text world, post hoc: an oracle stepping only on direction words caps at 0.972 -- every error is an object from an
  aside at the same cell. In language "only actions move" is the wrong target; learned steps move asides away.
- Eval mode (dropout off) under-reports runs below ceiling by up to 0.16: the 1/(1-p) attention-dropout SCALE (x1.111
  at eval recovers them); all batches re-scored 2026-10-04: no registered verdict changes (DROPOUT_RESCORE.md).
- UPDATE 2026-10-29 (registered since; CLAUDE.md rows are current): the dropout re-score moves the text-world path arms
  0.969 -> 0.993 (post hoc). TW_STATECHANGE (`TW_STATECHANGE_RESULTS.md`, CORRECTED 2026-10-11): path 0.994 vs RoPE 0.516; the
  'state bound to place' readout is location + recency among nouns, NOT state semantics -- 'the model tracks state changes'
  is WITHDRAWN. GAIN_PHASE (`GAIN_PHASE_RESULTS.md`, CORRECTED): one defect shown (the leak, in the step map); a score change
  does not touch it; 'two defects, two fixes' WITHDRAWN. LEAK on fresh seeds: SOLVED replicates, the +0.0107 accuracy gain
  does not (+0.0045, p 0.46). With images (IMAGE_TORUS): an observation-driven step at token scale stops training -- keep
  images out of the step map. TW_LANDMARK stopped (live negative); TW_AMBIG and CTXSTEP stopped.

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
