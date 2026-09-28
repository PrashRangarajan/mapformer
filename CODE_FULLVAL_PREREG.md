# Code checkpoints rescored on the full val file -- pre-registration (2026-09-27, before any number)

## Why
Every code contrast now quoted (`CODE_RESULTS.md` C1, `CODE_DECAY_RESULTS.md` C2, the envelope
column) is read on `best_val_bpc`: a MIN over 36 evaluations of 40 random windows (~5.7% of the
val file). That readout carries min-selection bias and window noise, and it is the one that
produced the withdrawn "PoPE Table 5 does not replicate" (`ABLATE_RESULTS.md`). The ledger
(`docs/audits/2026-09-24/experiments/LEDGER.md` B7-B10, B13) marks every one UNDERPOWERED: the
effects are 0.003-0.008 bpc at n=3, paired t-test p 0.04-0.35. Rescoring cannot add power, but it
removes the selection bias; if a sign flips, the claim dies for free.

## What is scored
The `.final.pt` checkpoint (last iteration, not min-selected) of every run, on the WHOLE val file
in non-overlapping windows at the length the run was trained at, with `eval_code_long.py` (the
eval behind `ABLATE_RESULTS.md` A3; it refuses a checkpoint whose stored `val_bpc` disagrees with
its run's JSON). Readout: token-weighted mean bpc over all positions of the window.
- `runs/code2048` (C1): RoPE, PoPE-Flat, MapPoPE-Flat, Vanilla (=MapWM), s0-2, at 2048.
- `runs/code_decay` (C2): RoPE-, PoPE-, MapPoPE-, MapWM-Decay, s0-2, at 512.
- `runs/code` (the no-envelope 512 arms, for the cross-batch envelope column), s0-2, at 512.

## Contrasts (paired by seed, n=3) and how they are read
C1: encoding main ((PoPE-RoPE)+(MapPoPE-MapWM))/2 [was -0.0030]; position main
((MapWM-RoPE)+(MapPoPE-PoPE))/2 [+0.0055]; MapPoPE-PoPE [+0.0033]; MapPoPE-MapWM [-0.0052].
C2: PoPE-Decay - RoPE-Decay [+0.0079]; MapPoPE-Decay - PoPE-Decay [+0.0063]; MapWM-Decay -
RoPE-Decay [RoPE-Decay "best of eight", +0.0034]. Envelope (cross-batch): X-Decay - X for each arm.
For each: mean, house verdict (|t| > 2.8), paired t-test p, exact-t MDE (`stats_core`), sign count.
Registered reading, fixed now:
- A contrast whose full-val SIGN differs from its `best_val_bpc` sign: the claim is withdrawn.
- Same sign, paired t p < 0.05: keep as stated, with the full-val number replacing the old one.
- Same sign, p >= 0.05: UNMEASURED (the current ledger status; the old number is replaced).
No arm is added, no seed is added; this is a readout correction, not a new experiment.
Cost: eval-only, 36 checkpoints.
