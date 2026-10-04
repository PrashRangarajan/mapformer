# Leak remedies on the new-object task -- results (2026-10-03)

Pre-registration `LEAK_PREREG.md` (+ Amendment 1, committed before any result); runs `runs/leak/p0` (24 runs, one
batch, 8 seeds); registered output `LEAK_ANALYSIS.txt` / `LEAK.json` (`analyze_leak.py`); declared secondaries
`LEAK_SECONDARY.txt` (`analyze_leak_secondary.py`). Read-time checks: every guarded file matches
`runs/leak/code_md5.txt`; void check OK (ActOnly leak exactly 0).

## Registered: REMEDY for both ActOnly and NormStep -- but by construction at x4 (Amendment 1)
| arm | unseen-object acc x1 | x2 | x4 | leak L(x1) = steps-zeroed - intact | final loss | class |
|---|---|---|---|---|---|---|
| MapWM | 0.9890 +/- 0.0030 | 0.9513 | 0.8623 | +0.0107 (mean) | 0.070 | 8/8 DESCENDING |
| ActOnly (action-only step) | **0.9997** +/- 0.0006 | 0.9997 | 0.9997 | 0.0000 | 0.016 | 8/8 SOLVED |
| NormStep (step reads LN(emb)) | **0.9998** +/- 0.0002 | 0.9998 | 0.9998 | -0.197 (see below) | 0.020 | 8/8 SOLVED |

The x2/x4 columns of the remedy arms are construction checks (both are invariant to embedding-side code norm), so
the registered REMEDY verdicts say only: each remedy trains to within 0.01 of MapWM in distribution and its leak is
<= 0.01. Robustness to a norm shift follows from the construction, not from training (rule 10 applies regardless).

## What the batch shows (declared secondaries)
- **Removing the leak buys exactly what the leak cost, in distribution, and it is detectable.** ActOnly - MapWM and
  NormStep - MapWM at x1: **+0.0107** (permutation p 0.0002, MDE ~0.0035, 8/8 vs 8/8), equal to MapWM's own leak
  (+0.0107): the remedies reach the leak-free accuracy MapWM only reaches with its object steps zeroed at eval.
- **The leak slows training.** Both remedies converge within 900 epochs (all 16 SOLVED, final loss 0.016 / 0.020);
  MapWM is still descending at 0.070 on every seed. r(final loss, x1 accuracy) = -0.944 over the 24 runs, so the x1
  gain cannot be separated from convergence speed at this budget (rule 2): the leak costs accuracy at least partly
  by slowing optimisation.
- **For NormStep the "leak" readout means something else.** Its object steps are as small as MapWM's (rms object /
  action step 0.0041 vs 0.0044) yet zeroing them costs 0.19: they are systematic, not noise. A LayerNorm output has a
  learned bias common to every input, so every observation gives the same small step -- a per-observation tick the
  model integrates. Removing it at eval shifts the phase. So zeroing is the wrong leak readout for NormStep; its
  object-identity-dependent step is not measured here. (Hypothesis about the mechanism; the LN-bias step was not
  isolated.)
- **The blank step is part of MapWM's code** (as in E0): zeroing object AND blank steps gives 0.9937 vs 0.9998 for
  object steps only. ActOnly also drops the blank step and still solves, so a retrained model does not need it.
- **ActOnly - MapWM bundles three changes** (no object step, no blank step, an oracle action label); NormStep
  reaches the same accuracy without the oracle label, so the label is not what buys the gain.

## Caveats
- One task, 1 layer, r=4, T=1024, 900 epochs (MapWM unconverged), n=8. The x4 robustness is by construction.
- NormStep's mechanism reading is a hypothesis; a NormStep without the LN bias, or a per-token step ablation by
  class, would test it.
