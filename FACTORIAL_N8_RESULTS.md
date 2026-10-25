# Does the torus depth result replicate on fresh seeds? FACTORIAL at n = 8 -- results (registered, `FACTORIAL_N8_PREREG.md` + Amendment 1)

Batch 2026-10-24 23:32 -> 2026-10-25 07:34 (8 h 02 min, including evaluation) on kalman GPU 1 (4 jobs at a time), code
`ab872fb` (clean; md5 guard over 104 files), torch 2.10.0+cu128. 16 runs, fresh seeds 2800-2803 (n = 4 per cell); pilot
seeds 2890-2891 outside the batch. Recipe = FACTORIAL's (torus 64 x 64, trained and tested at T = 1024, 900 epochs, lr 1e-3
cosine, batch 16, 2 heads, d 128, --fast-attn, no --deterministic), asserted per run against FACTORIAL's stored args.
Registered output `FACTORIAL_N8_ANALYSIS.txt`, `FACTORIAL_N8_VERDICTS.json`; evals `FACTORIAL_N8_{L1,L4,LOOP}.json`, strata
`FACTORIAL_N8_*_STRATA.json`, own-map `FACTORIAL_N8_*_OWN_s*.json`; run dirs `runs/factorial_n8/{L1,L4,LOOP}`. The analysis
was re-run here read-only: text identical, verdicts JSON equal, no output file changed (md5).

## Reading (registered primary: the fresh seeds alone, n = 4 vs 4)

**REPLICATES.** On four fresh seeds and new evaluation walks every FACTORIAL reading recurs with the same label: one
path-integrated layer solves the T = 1024 torus (0.999, 4/4) and index RoPE at one layer, at four real layers or as one
block looped four times does not (0.729 / 0.830 / 0.859, 0/4 each). Every firing is a complete separation at the n = 4
minimum p (0.029).

| contrast | registered label | effect | test |
|---|---|---|---|
| C0 Path_L1 - RoPE_L1 | **PATH WINS AT ONE LAYER** | +0.270 | perm p 0.029 |
| C1 depth: Path_L1 vs RoPE_L4 | **INDEX DOES NOT CATCH UP** [within 900 epochs; RoPE_L4 still DESCENDING 1/4] | +0.169; SOLVED 4/4 vs 0/4 | perm p 0.029; Fisher 0.029 |
| C2 loop: Path_L1 vs RoPE_Loop | **INDEX DOES NOT CATCH UP** [within 900 epochs; RoPE_Loop still DESCENDING 0/4] | +0.140; 4/4 vs 0/4 | perm p 0.029; Fisher 0.029 |
| C3 RoPE_L4 - RoPE_L1 | **HELPS** | +0.101 | perm p 0.029 |
| C3 RoPE_Loop - RoPE_L1 | **HELPS** | +0.130 | perm p 0.029 |

Replication summary (Amendment 1 A3, a function of the labels): **REPLICATES** (C1 and C2 both DOES NOT CATCH UP, as
FACTORIAL); same label as FACTORIAL for C0, C3 RoPE_L4 and C3 RoPE_Loop.

| cell | position code | architecture | params | acc T = 1024, held-out map (per seed) | SOLVED | classes |
|---|---|---|---|---|---|---|
| Path_L1 | path-integrated (MapWM, shared rank 4) | 1 layer | ~204k | **0.999** (0.996, 1.000, 1.000, 1.000) | **4/4** | 4 S |
| RoPE_L1 | index | 1 layer | ~204k | 0.729 (0.730, 0.740, 0.713, 0.733) | 0/4 | 4 T |
| RoPE_L4 | index | 4 real layers | 798,741 | 0.830 (0.840, 0.862, 0.811, 0.807) | 0/4 | 1 D, 3 T |
| RoPE_Loop | index | one block looped x4 | 203,925 | 0.859 (0.845, 0.861, 0.866, 0.866) | 0/4 | 4 T |

Floors on these evaluation walks (measured, model-free): blank 0.511, retrace 0.846 (per seed 0.851 / 0.841 / 0.847 / 0.845);
wrap-only share 0.077; best n-gram 0.576 (quoted from RANK_MI). S = SOLVED, T = STALLED, D = DESCENDING
(`stats_core.classify_run`).

## Declared secondaries (no verdict)

**1. The deep index models sit at the retrace floor.** Accuracy minus the retrace floor on the same walks, paired by seed:

| cell | acc - retrace (fresh) | seeds above | FACTORIAL (its own walks) | pooled n = 8 (derived, descriptive) |
|---|---|---|---|---|
| Path_L1 | +0.153 | 4/4 | +0.156 (4/4) | +0.154 (8/8) |
| RoPE_Loop | +0.014 | 3/4 | +0.020 (4/4) | +0.017 (7/8) |
| RoPE_L4 | **-0.016** | **1/4** | +0.015 (3/4) | **-0.000** (4/8) |
| RoPE_L1 | -0.117 | 0/4 | -0.125 (0/4) | -0.121 (0/8) |

Four index layers score what a map-free rule that copies the observation 2j steps back while a run of moves reverses the
previous one scores: in the fresh batch slightly below it, pooled at it. The loop is 0.01-0.02 above it.

**2. Revisit strata (T = 1024; fresh means; retrace floor in brackets).** Depth buys short-gap returns and nothing else:

| cell | short gap (< 128 moves) | long gap (>= 128) | wrap-only |
|---|---|---|---|
| Path_L1 | 1.000 (0.911) | 0.997 (0.514) | 0.987 (0.514) |
| RoPE_Loop | 0.931 (0.911) | 0.494 (0.514) | 0.493 (0.514) |
| RoPE_L4 | 0.913 (0.911) | **0.413** (0.514) | **0.407** (0.514) |
| RoPE_L1 | 0.779 (0.911) | 0.474 (0.514) | 0.476 (0.514) |

Retrace equals the blank level beyond short gaps (it can copy only within a reversal), so 0.514 is the floor there. Four
index layers are 0.002 above retrace on short gaps (FACTORIAL: 0.943, +0.031) and 0.10 below the blank level on long-gap
and wrap-only revisits: they predict a specific wrong object rather than abstaining. The loop is 0.02 above retrace on short
gaps and at the blank level beyond.

**3. Pooled n = 8 with FACTORIAL's seeds 500-503 (cross-batch, descriptive only: `train.py` and `train_variant.py` changed
additively since FACTORIAL's `4041da9`).**

| cell | pooled acc | SOLVED | fresh | FACTORIAL | batch shift (perm p) |
|---|---|---|---|---|---|
| Path_L1 | 0.999 | 8/8 | 0.999 | 1.000 | -0.001 (0.26) |
| RoPE_L1 | 0.724 | 0/8 | 0.729 | 0.719 | +0.010 (0.54) |
| RoPE_L4 | 0.845 | 0/8 | 0.830 | 0.860 | -0.029 (0.17) |
| RoPE_Loop | 0.862 | 0/8 | 0.859 | 0.864 | -0.005 (0.43) |

Pooled contrasts (would-read, not verdicts): C0 +0.275, C1 +0.155 (0/8 vs 8/8, Fisher 0.0002; 1/8 descending), C2 +0.138
(0/8 vs 8/8), C3 +0.121 and +0.138, each perm p 0.0002. No batch shift is detectable (every |shift| <= 0.029, p >= 0.17).

**4. Memorisation (own training map vs held-out).** RoPE_L4 0.919 vs 0.830 (+0.089; FACTORIAL +0.037), RoPE_Loop 0.869 vs
0.859, RoPE_L1 0.790 vs 0.729, Path_L1 0.999 vs 0.999. Four index layers learn more of their own map than of the task.

**5. Out of distribution (rule 10).** T = 512 / 2048: Path_L1 1.000 / 0.968; RoPE_L4 0.881 / 0.573; RoPE_Loop 0.900 / 0.632;
RoPE_L1 0.759 / 0.545.

**6. Rule 2.** r(final-5% training loss, acc T = 1024) -0.909 over the 16 fresh runs (FACTORIAL -0.94): accuracy is largely
loss here; the registered contrasts also fire on SOLVED.

## Checks

- **experiment_audit** (read-only; `--runs-dir runs/factorial_n8/{L1,L4,LOOP}` and the root): exit 2 on every dir, as
  Amendment 1 A4 expects (no (json, pt) pairs; check [1] only). Windowed classes equal the registered ones: L1 RoPE 4
  STALLED, Vanilla_r4 4 SOLVED; L4 RoPE 3 STALLED + 1 DESCENDING; LOOP 4 STALLED. Its "all arms converged" line is the flat-
  window rule rule 3 warns about; the registered classes are the analysis's.
- **Numbers:** every value above was re-read from `FACTORIAL_N8_ANALYSIS.txt` / `FACTORIAL_N8_VERDICTS.json`; per-cell means
  re-derived from the per-seed lists. Pooled retrace margins are arithmetic means of the two batches' printed means.
- **Power:** at 4 vs 4 a label fires only on a complete separation (p 0.029); every registered firing is one.

## Predictions vs outcomes (prereg, stated before the batch)

| prediction | outcome |
|---|---|
| C0 PATH WINS; Path_L1 1.000 (4/4) vs RoPE_L1 0.69-0.75 | yes: 0.999 (4/4) vs 0.729 |
| C1, C2 DOES NOT CATCH UP on both routes; RoPE_L4, RoPE_Loop 0.81-0.89, 0/4, STALLED (none DESCENDING); REPLICATES | yes, except one RoPE_L4 run DESCENDING (1/4); 0.830 / 0.859 |
| C3 HELPS for both (+0.10 to +0.17) | yes: +0.101 (near the stated risk) and +0.130 |
| index short-gap 0.90-0.96 | yes for 4 layers (0.913) and the loop (0.931); 1 layer 0.779 |
| long-gap and wrap-only 0.37-0.53, at or below blank | yes: 0.41-0.49 |
| RoPE_L4, RoPE_Loop overall within +/-0.05 of retrace | yes: -0.016, +0.014 |
| batch shift within +/-0.03 per cell | yes (RoPE_L4 -0.029, at the edge) |
| r(loss, acc) strongly negative | yes: -0.909 |

## Reading for the paper

The torus result now stands on eight seeds from two batches, four of them fresh and analysed alone: one path-integrated
layer learns the T = 1024 torus map (8/8 SOLVED), and index attention with four real layers or one block looped four times
does not (0/8 each), though both help over one layer. The fresh batch sharpens what depth buys: four index layers land on
the retrace floor (-0.016, pooled -0.000) and the loop just above it (+0.014), with short-gap returns at or barely above
the retrace rule and long-gap and wrap-only revisits below the blank level. Depth buys index attention what a map-free
retrace rule gives, and no map; the exchange rate on the torus stays "more than four layers".

## Scope

n = 4 fresh per cell (complete separations only); pooled n = 8 is cross-batch and descriptive. One task, one length
(T = 1024), one budget (900 epochs; 1 of 4 fresh RoPE_L4 runs still descending), 2 heads, d 128, 1 and 4 layers only; no
exchange rate beyond "more than four". --fast-attn without --deterministic: neither batch is bitwise reproducible.
