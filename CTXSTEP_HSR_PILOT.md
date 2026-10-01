# Hidden-state step fix (HSR) -- pilot (2026-09-30)

Pilot, 2 seeds per cell, not registered. Runs `runs/hsr_pilot` (driver `run.sh`); far cue (leading 7-12
tokens, trailing 6-10), T=2048 words, batch 8, 1800 epochs, seeds 2 and 3 (0, 1, 5 used earlier).
HSR (`model_context_step.HiddenStepResWM`): Delta_t = W_out W_in (emb(x_t) + alpha * LN(h1_t)), alpha a
learned scalar initialised at 0, so it starts as the context-free step (gated: diff 0.0 at init, every
HS weight shared at the seed, causal leak 0). Control: the original HS, same seeds and budget.
Swap test: `docs/audits/2026-09-27/swap_test.py --offset 15`, records in `swap_results_hsr.jsonl`.

| cue | arm | s2 acc (move step, decoy/move) | s3 acc (move step, decoy/move) |
|---|---|---|---|
| leading | **HSR** | **0.966** (0.111, 0.02) | **0.972** (0.164, 0.03) |
| leading | HS | 0.985 (0.576, 0.00) | 0.728 (0.001, no step) |
| trailing | **HSR** | **0.998** (0.103, 0.02) | **0.993** (0.107, 0.28) |
| trailing | HS | 0.955 (0.137, 0.01) | 0.742 (0.001, no step) |

Final training loss: HSR 0.055-0.076 (just above the 0.05 SOLVED line); HS 0.044 / 0.060 on s2, 0.74 /
0.81 on s3. Learned alpha: -0.27, -0.24 (leading), -0.25, +0.19 (trailing).

## Reading
- **HSR learned a movement step in 4 of 4 runs** and ignores far decoys in all 4 (ratio 0.02-0.03, one
  partial at 0.28), reaching 0.966-0.998. The original HS learned a step in 2 of 4 here (pooled over
  all far-cue HS runs: 7 of 14). Keeping the word's own step and adding context as a correction removes
  the failure to start.
- When HS does learn a step it is as good (0.955-0.985, ratio 0.00-0.01); the fix is about reliability.
- For scale: the window-limited steps on far cues (pilot 3, seed 0) 0.615-0.855 and do not ignore decoys;
  on near cues 0.97-1.00.
- n = 2 per cell; HSR has not crossed the SOLVED line within 1800 epochs. Pilot only.
