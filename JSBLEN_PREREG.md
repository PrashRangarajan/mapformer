# JSBLEN_PREREG -- length extrapolation on Bach Chorales

Written after a 60-iteration smoke run, before any full run. This tests the axis PoPE's Figure 2
is about (length extrapolation, which they measure on PG-19 after OpenWebText pretraining -- out of
reach here) on a dataset they do use, and it is the axis on which path integration wins elsewhere in
this project (Dyck-2, the torus, MiniGrid).

## Design

Identical to `JSB_PREREG.md` -- same data, same recipe (d256, 8 heads, 6 layers, dropout 0.2, batch
4, lr 6e-4 cosine to 6e-5, 3,000 iterations) -- with ONE change: the training context is **512
tokens**, sampled as a random crop from each piece, instead of the full 2,048. Evaluation is on
whole test pieces, with NLL reported by POSITION BUCKET: 0-512 (in distribution), 512-1024 (1-2x
beyond) and 1024-2048 (2-4x beyond). Test pieces have median 912 and maximum 2,048 tokens, so the
last bucket is real but thin (28 of 77 pieces reach 1024, 7 reach 1536).

Same 2x2, 5 seeds: {index, path integration} x {RoPE, PoPE}.

## Registered verdicts

- **R1** In the 0-512 bucket, PoPE - RoPE should reproduce the in-distribution result already
  measured (-0.0322 NLL at full context). A different sign here means the shorter context changed
  the task, and nothing else in the run is interpretable.
- **R2 (the point of the run)** Does path integration's advantage GROW with position bucket?
  Registered prediction: **yes** -- MapWM - RoPE and MapPoPE - PoPE become more negative (better)
  from 0-512 to 1024-2048. This is the pattern in every other length-extrapolation result in this
  project. Tested as the paired per-bucket contrast with its MDE, and as the bucket-to-bucket
  change with its own MDE.
- **R3** PoPE's encoding advantage should hold at every bucket (it is a decoupling claim, not a
  length claim).
- **R4** If path integration's advantage does NOT grow, that is the third null for path integration
  on PoPE-paper data and is reported as such -- the registered conclusion becomes "path integration
  pays only where the task has explicit push/pop or move-by-k structure", with Dyck-2 the sole
  positive.

At n=5 the MDE is 1.25 sd. The in-distribution effect size for reference was 0.032 NLL with seed sd
0.010; extrapolation differences in this project are usually larger than in-distribution ones, so
this is better powered than the JSB run was, but a small effect will still read as unmeasured.
