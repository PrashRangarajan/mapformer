# Bump-token experiment: can MapFormer dead-reckon in a maze?

MapFormer's theta = cumsum(action_to_lie(token_emb)) is OPEN-LOOP over
COMMANDED actions. 28.1% of maze moves are blocked by walls, so theta
advances while true position does not. Measured dead-reckoning error after
256 steps: 5.99 cells == random guess (~6.0). Free torus: 0.00.

Fix: on a blocked move emit a DIRECTIONAL bump token BUMP_a (replacing the
redundant repeated observation). action_to_lie is a CONTEXT-FREE per-token
map, so a GENERIC bump provably cannot help (best single vector ~= 0;
measured 5.77 vs 5.76). With 4 tokens it can learn BUMP_a -> -delta(a),
making commanded_sum - blocked_sum = executed_sum EXACTLY (measured 0.00).

Both arms trained in the same batch, n=3. Chance=0.25, greedy=0.73.

| Arm | len 1-5 | len 6-10 | len 11-15 | len 16+ | all |
|---|---|---|---|---|---|
| nobump (n=3) | 0.446±0.008 | 0.482±0.001 | 0.534±0.007 | 0.532±0.001 | 0.503±0.003 |
| bump (n=3) | 0.457±0.014 | 0.512±0.002 | 0.543±0.009 | 0.547±0.010 | 0.522±0.004 |

| delta (bump - nobump) | +0.011 | +0.030 | +0.010 | +0.015 | +0.019 |

Greedy baseline = 0.730. bump all = 0.522 (does NOT clear greedy).
