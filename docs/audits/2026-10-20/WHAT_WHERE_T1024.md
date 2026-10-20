# What/where at T = 1024 (post hoc, eval-only; 2026-10-20)

Re-run of the causal what/where test (`docs/audits/2026-10-03/causal_whatwhere.py`) in the regime where a map is needed:
the SIGN_MATCHED checkpoints (torus, trained and tested at T = 1024, 1 layer, rank 4, n = 8 per arm; floors on these
walks blank 0.506, n-gram 0.575, retrace 0.843). Script `causal_whatwhere_T1024.py`, output `causal_whatwhere_T1024.txt`
/ `.json`, log `.log`. Rebuilt forward = model logits exactly (max |dlogit| 0.0) on all 32 checkpoints; intact accuracy
matches the stored registered eval to <= 0.0007. Changes from the T = 128 script (memory): evaluation in chunks of 10
trajectories, calibration on 20 trajectories, graded phase-lesion and lam_add grids dropped.

| condition | signed (path, learns the map) | abs | softplus | index RoPE |
|---|---|---|---|---|
| intact | 0.998 | 0.821 | 0.780 | 0.731 |
| object identity removed from the score (type kept) | 0.966 (-0.032, p 0.012) | 0.544 (-0.277) | 0.476 (-0.305) | 0.479 (-0.252) |
| one shared kernel x content gain | **0.998 (-0.000)** | 0.584 (-0.237) | 0.675 (-0.106) | 0.439 (-0.293) |
| additive position + content | 0.895 (-0.104) | 0.350 | 0.512 | 0.492 |
| type removed too (one global mean) | 0.113 | 0.295 | 0.302 | 0.399 |
| content-only (no rotation) | 0.505 | 0.399 | 0.385 | 0.490 |
| share of score variance explained by position alone | 0.925 | 0.644 | 0.577 | 0.618 |
| content x position interaction share | 0.061 | 0.187 | 0.174 | 0.353 |

Reading (analysis):
- The model that learns the map factorises what and where at T = 1024: a single position kernel per head, scaled by a
  content gain, reproduces it exactly (0.998); removing object identity costs 0.032 (p 0.012, 8/8 seeds lower; at
  T = 128 the cost was 0.011-0.025); position alone explains 92.5% of the score's variance; token TYPE (move vs
  observation) is essential (0.113 without it).
- The models that fail do not: removing object identity costs 0.25-0.31 and a shared kernel x gain does not restore them.
- Within the non-negative arms, separation tracks success: across 16 runs, accuracy correlates with the share of it that
  survives removing object identity (Spearman 0.84, p 5e-5); the two non-negative runs that reach > 0.95 keep 0.97 / 0.99
  of their accuracy, the failing ones collapse to 0.03-0.07.
So in the map regime what/where separation is a property of the solution training finds, not of the architecture.
