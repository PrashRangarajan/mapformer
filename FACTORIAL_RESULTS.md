# Can depth or a loop let index attention catch up with path integration on the torus? -- results (2026-10-13)

Pre-registration `FACTORIAL_PREREG.md` + Amendment 1 (independent check). Batch 2026-10-12 20:38 -> 10-13 04:45 at 4041da9 on
GPU 0; torus, trained and tested at T = 1024, 900 epochs, --fast-attn, seeds 500-503 (n = 4 per cell). Registered output
`FACTORIAL_ANALYSIS.txt`, `FACTORIAL_VERDICTS.json`; evals `FACTORIAL_{L1,L4,LOOP}.json`, strata `FACTORIAL_*_STRATA.json`.

| cell | position code | architecture | params | accuracy T = 1024 (held-out map) | SOLVED | own training map |
|---|---|---|---|---|---|---|
| Path_L1 | path-integrated (MapWM r4) | 1 layer | ~204k | **1.000** (1.000 x 4) | 4/4 | 1.000 |
| RoPE_L1 | index | 1 layer | ~204k | 0.719 (0.69-0.74) | 0/4 (STALLED) | 0.777 |
| RoPE_L4 | index | 4 real layers | 798,741 | 0.860 (0.81-0.88) | 0/4 (STALLED) | 0.897 |
| RoPE_Loop | index | one block looped x4 | 203,925 | 0.864 (0.86-0.87) | 0/4 (STALLED) | 0.862 |
Floors on this stream: blank 0.507, best n-gram 0.576.

## Registered (n = 4 vs 4: only complete separations are readable; every firing below is a complete separation, p 0.029)
- **C0: PATH WINS AT ONE LAYER** -- Path_L1 - RoPE_L1 +0.281.
- **C1 (depth): INDEX DOES NOT CATCH UP** -- Path_L1 - RoPE_L4 +0.141; SOLVED 4/4 vs 0/4 (Fisher p 0.029); all four 4-layer
  index runs STALLED (none still descending) within 900 epochs.
- **C2 (loop): INDEX DOES NOT CATCH UP** -- Path_L1 - RoPE_Loop +0.136; 4/4 vs 0/4; all STALLED.
- **C3: depth and the loop each HELP index** -- +0.140 (4 layers) and +0.145 (loop x4) over one layer.
The pre-stated prediction (Amendment 1: from `LOOPED_PILOT.md`'s reach limits) held.

## Secondaries (declared; no verdict)
- **Where depth helps and where it does not** (revisit strata, T = 1024): depth or a loop lifts index accuracy on revisits
  within 128 moves (1 layer 0.74-0.79 -> 4 layers 0.90-0.96, loop 0.94) but not on long-gap revisits (>= 128 moves: 0.37-0.53
  at every depth) or on revisits reachable only by wrapping around the torus (0.38-0.52). Path integration at one layer is
  1.000 on every stratum.
- **Memorisation check:** 4-layer index scores 0.897 on its own training map vs 0.860 held-out (some map-specific learning);
  the looped model 0.862 vs 0.864; path 1.000 on both.
- The loop matches 4 real layers at a quarter of the parameters (0.864 vs 0.860), as on Match-Query and short-walk tori.

## What it means
- **On the torus, at the training length, 4 layers or a loop x4 of index attention do not substitute for one layer of path
  integration.** Depth buys the short-range part of the map (recent revisits) and nothing at long range: index attention has
  no way to compute net displacement over hundreds of moves in 4 layers. This differs from Dyck, where 3-4 index layers
  reached the path model at matched depth (`DYCK_MDEPTH_RESULTS.md`): the torus needs more than 4 layers of attention, or
  cannot be done this way at this budget. The exchange rate here is "more than 4 layers", not measured beyond that.
- Position code and architecture are separate axes: the architecture axis helps index attention locally; only the position
  code (path integration) solves long-range and wrap-around revisits.

## Caveats
- n = 4 per cell (complete separations only); 900 epochs (all index runs STALLED, so budget is unlikely to change the
  1-layer and 4-layer readings, but longer training was not tested); --fast-attn series, not comparable bit-for-bit with
  stored runs; one task, d 128, 2 heads; 1 and 4 layers only (no exchange rate beyond "more than 4").
