---
name: feedback_lm200_stuck_baselines
description: "The April lm200 baseline checkpoints are stuck (training-convergence artifact); the project's lm200 leaderboard ranked convergence, not architecture. Retrain baselines fresh before any lm200 claim."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d17148dd-e54f-48b7-8a88-096f71aadfc5
---

The stored lm200 checkpoints trained **April 22–24** (Vanilla, Level15,
Level15EM, VanillaEM, RoPE, PC, Level1, CoPE, LSTM, MambaLike) are
**stuck**: they converged to CE loss ~1.0 instead of ~0.005. Checkpoints
trained **May 8+** (TEMFaithful, Level15GSF, Level15_SR, NoDrop,
Vanilla_ExtraHead) converged normally. The reported lm200 "leaderboard"
is monotonic with *training convergence*, not with architecture.

**How discovered (2026-07-15):** the "Level15Cascade wins lm200 +10-25pp"
result turned out to be the fresh cascade compared against a *stuck*
April Level15 baseline. Retraining Level15 fresh under current code gives
0.996 (loss 0.005); the old checkpoint was 0.80 (loss 1.01). A same-seed
reproducibility test (3x each) is perfectly deterministic (0.996±0.000),
so it is NOT nondeterminism — it is a systematic April-condition artifact
(first-epoch loss differs 2.71552 vs 2.71538 → a data/RNG-order shift that
dropped April Level15 into a bad optimization basin). Level15-WM code is
byte-identical April vs now (only change: log_R_init_bias, default 0.0 =
no-op for WM). So Level15 lm200 training is **basin-sensitive**.

**Corrected fresh (current code, seed 0) lm200 T=512 leaderboard:**
Level15 0.996 > TEMFaithful 0.982 > NoDrop 0.915 > Level15EM 0.860 >
Vanilla 0.835 > VanillaEM 0.807 > PC 0.721 > RoPE 0.513.

**What this REVERSES (all artifacts of the stuck Level15 baseline):**
- "TEMFaithful is the lm200 leader" — FALSE, fresh Level15 (0.996) beats
  it (0.982). Supersedes [[feedback_multiple_fixes_match_tem]].
- "NoDrop +13pp over Level15 on lm200" — FALSE, fresh Level15 > NoDrop.
- "GSF/cascade match or beat Level15" — artifacts.
- The Level15Cascade result entirely — no benefit; NoSlow ≡ Level15.

**What SURVIVES (strengthened):** Level15 >> Vanilla on lm200 is real and
larger than reported (0.996 vs 0.835, ~+16pp). Vanilla's code (model.py)
is unchanged and reproduces stuck-ish both times — it genuinely can't
localize landmarks (drift). Correction helps on either backbone
(Level15 > Vanilla, Level15EM > VanillaEM).

**How to apply:** NEVER compare a freshly-trained variant against a
stored baseline checkpoint. Retrain baselines under current code in the
same batch. Any lm200 result where the winner is the more-recently-trained
model is suspect. Re-validate all lm200 tables in RESULTS_PAPER.md /
CLAUDE.md before submission. See also [[feedback_seed_ordering]] (this is
why single-seed lm200 gaps kept inflating) and
[[feedback_minimal_sweep_skip_gsf]].
