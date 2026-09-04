# Gates: D-dimensional torus rank test

T=128, 200 trajectories per config, n_obs_types=16, seed=0

## D=2, grid 32 (1024 cells, vocab 21, 4 actions)

  G1 chance (majority class over 5968 labels): 0.5064  [17 classes seen]
  G2 action-stream ngram, orders 1-5: 1:0.513  2:0.513  3:0.510  4:0.503  5:0.491
  G3 revisit rate: 0.233
  G4 scored positions per trajectory: 29.8
  G5 rank r=2: min separation 0.0472, decay exponent -1.02 (predicted -1.00)
  G5 rank r=4: min separation 0.0672, decay exponent -1.02 (predicted -1.00)

  verdict: PASS  (best ngram 0.513 vs chance 0.506; revisit 0.233)

## D=3, grid 10 (1000 cells, vocab 23, 6 actions)

  G1 chance (majority class over 7256 labels): 0.5227  [17 classes seen]
  G2 action-stream ngram, orders 1-5: 1:0.531  2:0.527  3:0.521  4:0.511  5:0.497
  G3 revisit rate: 0.283
  G4 scored positions per trajectory: 36.3
  G5 rank r=2: min separation 0.0330, decay exponent -1.04 (predicted -1.50)
  G5 rank r=3: min separation 0.0653, decay exponent -1.04 (predicted -1.00)
  G5 rank r=5: min separation 0.1934, decay exponent -1.03 (predicted -1.00)

  verdict: PASS  (best ngram 0.531 vs chance 0.523; revisit 0.283)

## D=5, grid 4 (1024 cells, vocab 27, 10 actions)

  G1 chance (majority class over 15870 labels): 0.5260  [17 classes seen]
  G2 action-stream ngram, orders 1-5: 1:0.536  2:0.534  3:0.527  4:0.522  5:0.516
  G3 revisit rate: 0.620
  G4 scored positions per trajectory: 79.3
  G5 rank r=2: min separation 0.0198, decay exponent -1.79 (predicted -2.50)
  G5 rank r=5: min separation 0.2180, decay exponent -1.23 (predicted -1.00)
  G5 rank r=7: min separation 0.3881, decay exponent -1.25 (predicted -1.00)

  verdict: PASS  (best ngram 0.536 vs chance 0.526; revisit 0.620)

## Summary

| D | grid | cells | chance | best ngram | revisit rate | scored/traj |
|---|---|---|---|---|---|---|
| 2 | 32 | 1024 | 0.506 | 0.513 | 0.233 | 29.8 |
| 3 | 10 | 1000 | 0.523 | 0.531 | 0.283 | 36.3 |
| 5 | 4 | 1024 | 0.526 | 0.536 | 0.620 | 79.3 |
