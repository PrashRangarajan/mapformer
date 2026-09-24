---
name: project-robustness-vs-capability
description: Past the training length (or depth) measures robustness, not capability; every such claim that got a matched control died. Train at the target length, or add a decay envelope.
metadata:
  type: project
---

**The dividing line in the project is matched vs mismatched length -- and depth** (2026-09-23/24).

- **Surviving positives, at their correct scope.** Navigation on the torus at training length, under a
  converged recipe: position **+0.243** (MDE 0.038, 8/8; index RoPE 0.805, path 0.971;
  `PAPER2X2_RESULTS.md`). The often-quoted +0.461 is the 16-epoch recipe where the index arm never
  left the 0.506 floor. Dyck ladder (`DYCK_LADDER_RESULTS.md`, fixed width): training is L32 **D4**,
  where the position effect is +0.293 / +0.081 / +0.048 / +0.019 at 1-4 layers (index 0.979 at 4L);
  the +0.290 / +0.209 / +0.159 / +0.168 at L32 D12 is matched LENGTH but **3x the training nesting
  depth** -- an extrapolation in the variable the claim is about, and owed a matched-depth control.
- **Every "helps at OOD length" claim that got a matched-length control died.** Code: -3.694 bpc
  extrapolating from 512 became -0.0030 (unmeasured) at matched 2048, and the composition claim
  reversed (`CODE_RESULTS.md`, `CODE_DECAY_RESULTS.md`, `runs/code2048`). The MapFormer paper's Dyck
  "helps OOD" pattern is produced by its F1 metric (0.88 no-stack floor).
- **Sign**: the accuracy cost of a monotone increment is extrapolation-only. At training length
  monotone arms score 0.90-0.98 against index 0.80; the matched-length evidence is in the LOSS (12/12
  worse). No matched-length arm yet (`SIGN_ABLATION.md`).
- **Rank at matched length** (T=1024 train and test, 900 epochs): r=4 solves 8/8, r=2 0/8, +0.103 at
  T=1024 (perm p 0.0003) -- registered verdict UNREADABLE (4 r=2 runs still descending;
  `RANK_MATCHED_RESULTS.md`). A rank-2 projection of each solved r=4 scores 0.995 at T=1024 on 8/8
  seeds (`RANK_PROJ_FROZEN.md`), so r=2 can REPRESENT the solution: the gap is search, not capacity.
- **Never had a matched control:** InEKF / Level15, forget gate, PoPE-wrapping, rotate/allocentric.
  All trained T=128 and read at T=512/1024 (allocentric also at the 16-epoch recipe).

**Why:** a model evaluated past its training length or depth meets accumulator values, positions and
depths it never saw; handling them is a robustness property worth having, not "a better model".

**How to apply:**
- Measure a design at the length and depth it was trained at before calling it better; treat an
  OOD-only effect as robustness until a matched control says otherwise (CLAUDE.md rule 10).
- Practical: **train at the target length if you can; if not, add the 48-parameter ALiBi-style decay
  envelope** (`DECAY_RESULTS.md`) rather than choosing an encoding for how it extrapolates. "RoPE +
  envelope is the best of eight code arms" is n=3 and cross-batch (p 0.35 against MapWM-Decay) --
  suggestive, not established.
- Related: [[project-mappope-asymmetry]], [[project-rank-and-selective-rope]],
  [[feedback-validate-task-first]].
