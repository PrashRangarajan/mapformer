# Context-step pilot 1 (decoys with a leaked trailing cue) -- what it showed (2026-09-29)

> **Swap-test numbers re-derived 2026-09-30 (audit B1)** by the committed `docs/audits/2026-09-27/swap_test.py`
> (`run_swap_all.sh`, every record in `swap_results.jsonl`); they match the figures below within rounding.
> Where a run learned no movement step (move change <= 0.002) the ratio is noise.

Pilot only, 2 seeds, NOT a result; the task leaked a trailing cue (a decoy's direction word is always
followed by ".", a real move's never), which is why pilot 2 / the two-condition task exists
(`CONTEXT_STEP_DESIGN.md`, "Revision"). Runs `runs/ctxstep_pilot`, held-out map, T=1024, p_decoy 0.3.

Accuracy (s0 / s1): context-free CF 0.855 / 0.852; context gate CG 1.000 / 1.000; Selective-RoPE
generator SR 0.999 / 1.000; hidden-state step HS 0.958 / 1.000.

Mechanism check (originally an inline script, 2026-09-29 00:57; now `swap_test.py --offset 5`): swap a direction word north <-> south and measure
the change in the accumulated angle 5 tokens later, for real moves and for decoys (60 each, 40
held-out sequences); then the same for decoys with the "." after the direction word replaced by "and".

| arm | move | decoy (ratio to move) | decoy, "." -> "and" (ratio) |
|---|---|---|---|
| CF s0 / s1 | 0.781 / 0.645 | 1.00x / 1.00x | 1.00x / 1.00x |
| CG s0 / s1 | 0.186 / 0.212 | 0.01x / 0.00x | 0.02x / 0.03x |
| SR s0 / s1 | 0.099 / 0.080 | 0.05x / 0.11x | **0.63x / 0.92x** |

- CF moves on every decoy, by construction.
- CG ignores decoys and does not care what follows: it uses the LEADING cue, as designed.
- SR ignores decoys only while the "." follows them: replace it and the decoy moves the angle again
  (0.63-0.92 of a real move). SR uses the TRAILING cue, confirming the correction to the design's
  "Selective RoPE is additive" claim.
