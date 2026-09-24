# Per-head r=2 pilot -- results (2026-09-24)

Pre-registration `RANK_PERHEAD_PREREG.md`; runs `runs/rank_perhead_pilot`; full output
`RANK_PERHEAD_PILOT_ANALYSIS.txt` (`python3 -m mapformer.analyze_rank_perhead`).

## Registered reading: 1/2 SOLVED -> uninformative at n=2

| seed | paper's per-head r=2 | our shared r=2 (stored) | our r=4 (stored) |
|---|---|---|---|
| 0 | STALLED, loss 1.049, T=1024 acc 0.656 | STALLED, 0.587, 0.849 | SOLVED, 0.009, 1.000 |
| 1 | **SOLVED**, loss 0.007, acc 1.000 | DESCENDING, 0.051, 0.998 | SOLVED, 0.003, 1.000 |

**Reproduction check passed exactly:** our r=2 seed 0 retrained in this batch matches the
stored run's per-epoch loss on all 900 epochs (max difference 0.0), so the stored comparison
arms are valid.

## What the two seeds suggest (a pointer, not a result)

The per-head version followed OUR r=2 seed for seed (seed 0 stuck, seed 1 good), not r=4
(both solved). But this is confounded: by construction the per-head model shares every
non-bottleneck initial weight with our r=2 at the same seed, while r=4's different `W_in`
shape shifts the random draws for ALL its weights. At n=2, "per-head behaves like r=2" cannot
be separated from "seed 0's initial weights are bad for search".

## Probe caveat

`probe_action_geometry` reads the LATENT (`W_in x`). For the per-head model it reports
opposition 1.94 / 1.04 even on the solved seed. A head whose `W_out^h` is near zero can carry
an arbitrary latent without moving the angle, so latent opposition is not the right readout
here; it needs measuring in angle space (`W_out^h W_in^h x`) per head.

## If this is taken further

Run the rank arms at MATCHED initialisation: build every arm from our r=2's base at the same
seed and replace only the bottleneck -- our r=2, the paper's per-head r=2, and our shared
r=4 built the same way -- 8 seeds each, T=1024, 900 epochs. That removes the init confound
this pilot exposed (and the stored r=4 comparison carries), at ~16 new runs (~5-6 h wall).
