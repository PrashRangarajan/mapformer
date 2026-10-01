# Hidden-state step, recipe pilot on far cues (2026-09-30)

> **Swap-test numbers re-derived 2026-09-30 (audit B1)** by the committed `docs/audits/2026-09-27/swap_test.py`
> (`run_swap_all.sh`, every record in `swap_results.jsonl`); they match the figures below within rounding.
> Where a run learned no movement step (move change <= 0.002) the ratio is noise.

Pilot, not registered. Runs `runs/hs_recipe_pilot` (driver `run.sh`): the hidden-state step (HS) on the
cue-distance task with the cue FAR (6-13 tokens), both sides, seeds 1 and 5, T=2048 words, batch 8.
cold = from scratch, 1800 epochs; warm = warm-started from the solved decoy-free text-world model of
the same seed (embeddings, step map, omega, output head, its layer as HS's layer 2), 900 epochs.
Swap test as in the earlier pilots (angle change from north <-> south at a direction word, 15 tokens
later; decoy / move ratio, 0 = decoys ignored).

| cue | recipe | s1 acc (move step, ratio) | s5 acc (move step, ratio) |
|---|---|---|---|
| leading | cold 1800 | **0.907** (0.057, **0.01**) | **0.867** (0.030, **0.07**) |
| leading | warm 900 | 0.646 (0.109, 0.05) | 0.508 (0.002, no step) |
| trailing | cold 1800 | 0.697 (0.001, no step) | **0.878** (0.064, **0.03**) |
| trailing | warm 900 | 0.572 (0.001, no step) | 0.636 (0.000, no step) |

For scale (pilot 3, 900 epochs, seed 0): context-free 0.806 / 0.608; context gate 0.849 / 0.615;
Selective-RoPE generator 0.844 / 0.855 (leading / trailing); none suppress far decoys.

## Reading
- **When HS learns a step at all, it ignores far decoys, on both sides.** Every run whose movement step
  is non-negligible (>= 0.03) has a decoy/move ratio of 0.01-0.07. Pooled with pilot 3 (trailing/far
  s0: step 1.116, ratio 0.10): 5 of 5 such runs. This is the capability the window-limited steps lack.
- **Whether it learns a step is unreliable.** 5 of the 10 far-cue HS runs across both pilots never
  learned one (movement step 0.000-0.002). The warm runs did worse (3/4 without a step), but warm (900
  epochs) and cold (1800 epochs) differ in budget as well as start, so the two are not separable here.
- **Even with a step, it has not converged:** final losses 0.31-0.40 (cold, with a step), accuracy
  0.87-0.91, below the near-cue window arms (0.97-1.00).
- Unmeasured and at n = 2 per cell: nothing here is a verdict.

## Likely cause and a design fix, untested
HS's step is computed only from LN(h1), the output of a freshly initialised attention layer, so at the
start of training the step carries no reliable information about which word was read, and the path
layer has nothing to integrate. A variant that keeps the word's own step and lets context CORRECT it,
Delta_t = W_out W_in (emb(x_t) + LN(h1_t)), starts as the context-free model (which learns a step
reliably) and adds reach. That is the next thing to try before registering HS.
