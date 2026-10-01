# Context-step pilot 3 (cue distance) -- the window prediction holds; the hidden-state step is fragile (2026-09-29)

> **Swap-test numbers re-derived 2026-09-30 (audit B1)** by the committed `docs/audits/2026-09-27/swap_test.py`
> (`run_swap_all.sh`, every record in `swap_results.jsonl`); they match the figures below within rounding.
> Where a run learned no movement step (move change <= 0.002) the ratio is noise.

Pilot, ONE seed per cell, not a registered result. Runs `runs/ctxstep3_pilot`; task
`environment_textworld_ctx3.py` (gated `docs/audits/2026-09-27/gate_ctxstep3.py`: with the cue FAR (7-12 tokens leading, 6-10 trailing),
move vs decoy is at base rate from the 4 tokens before and the 3 after the direction word; NEAR, the cue
side decides it). T=2048 words, batch 8, 900 epochs, held-out map, p_decoy 0.3. Floors on the eval set:
constant 0.51 (lead) / 0.48 (trail), reversal-copy 0.62 / 0.59.

## Accuracy at T=2048, and the swap test (angle change from north <-> south; decoy / move ratio)
| cue | distance | CF context-free | CG context gate | SR Selective-RoPE gen. | HS hidden-state step |
|---|---|---|---|---|---|
| leading | near | -- | **1.000** (0.01) | **0.967** (0.42) | -- |
| leading | far | 0.806 (1.00) | 0.849 (1.00) | 0.844 (1.00) | 0.507, did not train (move step 0.002) |
| trailing | near | -- | **0.992** (0.05) | **1.000** (0.10) | -- |
| trailing | far | 0.608 (1.00) | 0.615 (0.94) | 0.855 (0.96) | 0.842 (**0.10**) |

## Reading
- **Window-limited steps fail past their window, on both sides.** With the cue 1-3 tokens away, the
  context gate and the Selective-RoPE generator suppress decoys (ratio 0.01-0.42) and solve the task
  (0.97-1.00). With the cue 6-13 tokens away they cannot: every decoy moves the angle like a real move
  (0.94-1.00), and accuracy falls to the context-free level. This is the prediction of the second
  revision, confirmed at n=1.
- **The hidden-state step can reach a far cue, but did not finish learning.** Trailing/far: it
  suppresses decoys (ratio 0.10, via attention from the cue back to the direction word) yet reaches only
  0.842 in 900 epochs. Leading/far: it never learned a step at all (move step 0.002, loss 1.69 = the
  floor), an optimisation failure, not a verdict on the design. It was also the weakest arm in pilot 2
  (0.874 / 0.880 on some seeds).
- SR on trailing/far (0.855) beats CF (0.608) without suppressing decoys; unexplained at n=1.

## Before a registered batch
The cheap arms are settled enough to register. HS is not: at n=1 one run failed to train and one did not
converge, so a registered HS-vs-window comparison would currently measure HS's training, not its reach.
Next: an HS recipe pilot on the far conditions (more seeds, 1800 epochs, and/or a warm start of layer 2
from a trained context-free model so only the step has to be learned).

## Qualifications (audit, 2026-09-30)
- **SR is not a one-knob arm.** `MapFormerWM_SRoPEGen` differs from CF in more than context: no rank
  bottleneck (full rank), no omega, and a temperature (`model_selective.py`). CF and CG are r=4 with omega.
  Its window failure is in line with CG's, but attribution to the window alone needs a rank-4 SR arm.
- Cue distances: leading/far 7-12 tokens, trailing/far 6-10 (pads of 5-9 tokens); the task docstring's
  "7-13" and the design's "6-10" are approximate.
- Seeds 0, 1 and 5 have been used by the context-step pilots; a registered batch must exclude or declare them.
