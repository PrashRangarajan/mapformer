---
name: project-robustness-vs-capability
description: Testing past the training length measures graceful degradation, not capability; only matched-length results have survived. Train at the target length, or add a decay envelope.
metadata:
  type: project
---

**The dividing line in the whole project is matched vs mismatched length** (2026-09-23).

- Every positive result that survives is measured at the training length: navigation +0.461 (T=128 ->
  128), and the Dyck depth ladder (`DYCK_LADDER_RESULTS.md`, +0.290 to +0.168 at 1-4 layers, 8/8, fixed width).
- Every "helps at OOD length" claim that got a matched-length control died. Code: -3.694 bpc extrapolating
  from 512 became -0.0030 (unmeasured) at matched 2048, and the composition claim REVERSED
  (`CODE_DECAY_RESULTS.md`, `runs/code2048`). The PoPE paper's "helps OOD" Dyck pattern is produced by
  the F1 metric.
- Never had one: rank (r=4, `RANK_SWEEP.md` -- audit says 94% of it is short-gap revisits at unseen
  absolute positions), InEKF / Level15, forget gate, PoPE-wrapping. All trained T=128, tested T=512/1024.

**Why:** a model evaluated past its training length is being asked to handle accumulator values and
positions it never saw; that is a robustness property, worth having but not "a better model".

**How to apply:**
- Before claiming a design is better, measure it at the length it was trained at. Treat any OOD-only
  effect as robustness until a matched-length control says otherwise.
- Practical recommendation: **train at the target length if you can; if not, add the 48-parameter
  ALiBi-style decay envelope** (`DECAY_RESULTS.md`; RoPE + envelope was the best of eight code arms)
  rather than choosing an encoding for how it extrapolates.
- Related: [[project-mappope-asymmetry]], [[feedback-prelaunch-audit]].
