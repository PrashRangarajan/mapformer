# Is per-head rank 2's torus failure about wrap-around? -- pre-registration (2026-10-06, before any batch run)

## Question
Per-head rank 2 (= D) fails on the 2D torus at T=1024 and rank 3 solves (`RANK_ND_RESULTS.md`, 32-torus: 1/8 vs
8/8; paper torus, grid 64: 0-2/8 vs 6/8). The failures concentrate on WRAP-ONLY revisits -- the cell was seen before
only at a different unwrapped position, so it can be recognised only with a phase code exactly periodic in the torus
size (`RANK_ND_RESULTS.md` S2: rank 2 0.530 wrap-only vs 0.907 other revisits). PoPE's score repairs short-gap
revisits but not wrap-only ones (`SCORE_RANK_RESULTS.md`). **Hypothesis:** rank D is hard because of the exactly
periodic code the torus demands. **Prediction:** on a torus large enough that wrap-only revisits are absent, per-head
rank 2 solves about as well as rank 3. If rank 2 still fails there, the rank limit is general, not about closure.

## Task and CPU gate (rule 11; `docs/audits/2026-10-06/rank_nowrap_gate.py`, which CALLS `environment_nd.GridWorldND`)
RANK_ND's 2D walk (directed runs of 1-10 steps, K=16, p_empty 0.5, revisit-scored), T=1024. Held-out map seed 10000,
walk seed 10000; n-gram floors fit on a training map (seed 200). 100 walks per grid (`rank_nowrap_gate_out.txt`), 1000
for 128-256 (`rank_nowrap_gate_n1000_out.txt`), exact counts `rank_nowrap_wrapcount_out.txt`:

| grid | cells | revisit frac | targets / seq | wrap-only share | blank | retrace | best action n-gram | best obs n-gram |
|---|---|---|---|---|---|---|---|---|
| 32 | 1024 | 0.457 | 468 | **0.361** | 0.523 | 0.750 | 0.523 | 0.523 |
| 64 | 4096 | 0.317 | 325 | 0.080 | 0.510 | 0.844 | 0.511 | 0.510 |
| 96 | 9216 | 0.296 | 303 | 0.0185 | 0.498 | 0.865 | 0.496 | 0.498 |
| 128 | 16384 | 0.289 | 296 | 0.0018 | 0.500 | 0.871 | 0.500 | 0.500 |
| 192 | 36864 | 0.288 | 295 | 6.8e-05 (20 of 295k) | 0.501 | 0.871 | 0.500 | 0.501 |
| **256** | 65536 | 0.289 | 296 | **0 of 296k (both maps)** | 0.500 | **0.872** | 0.500 | 0.500 |

Grid 32 reproduces `nd_floor_wrap.py` (wrap 0.361, retrace 0.750) and grid 64 SCORE_RANK's retrace (0.843). **Chosen
large grid: 256** -- zero wrap-only revisits in 2000 walks while scored targets stay at 296 per sequence (63% of the
32-torus's 468; ~4700 per batch of 16). Above ~128 the task is the infinite plane: every statistic but the wrap share
is flat from 128 to 384. Chance-level n-grams (action stream, previous observations) PASS. The retrace floor is far
above chance (0.872), so accuracy is compared across grids only floor-relative (below).

## Design choices
- **Cells** (one batch, rule 12; every arm built from our r=2 base at the seed, as all per-head arms):
  A32 per-head rank 2 on the 32-torus; B32 rank 3 on the 32-torus (the within-batch control: must reproduce "rank 2
  fails, rank 3 solves"); AL rank 2 on the 256-torus; BL rank 3 on the 256-torus (the large-grid control).
- **omega's initialisation is fixed at grid 32 in every cell** (`train_rank_nowrap.py`: arms `Vanilla_r2ph_om32`,
  `Vanilla_r3ph_om32`). The stock rule sets omega_min = 2 pi / grid, so a stock 256 arm would start with 31 of 32
  frequencies per head slower (omega_min 0.0245 vs 0.196) -- the confound RANK_WRAP carried. Fixing it makes the initial
  weights at a seed bit-identical across grids (verified, max diff 0.0); A32 vs AL then differ ONLY in the data (map,
  wrap-around, revisit rate). On the 32-torus the arms are bit-identical to `Vanilla_r2ph` / `Vanilla_r3ph`. Cost of the
  choice: at 256 the slowest initial wavelength (32 steps) is shorter than the walk's extent (median max displacement
  82), so the code must slow its frequencies or combine channels; BL is the check that this is learnable. What the
  stock rule would do at 256 is NOT tested.
- **Training map fixed per seed** (`--seed` seeds the map): 1024 cells at 32 (RANK_ND's failing arms partly memorised
  it, +0.17), 65536 at 256 (205k parameters; memorisation implausible). Own-map vs held-out is a secondary.
- **Recipe** = `run_rank_nd.sh`'s 2D cells exactly: T=1024, 900 epochs x 98 batches, B16, lr 1e-3, warmup + cosine,
  1 layer, 2 heads, d 128, no landmarks, `--data-workers 3`, explicit attention path, `--save-full-state`.
- **Seeds 60-71 (n=12 per cell, 48 runs)**, fresh for these arms (RANK_ND 0-7, SCORE_RANK 10-21); seed outer, cell
  inner. Eval: `eval_nd` via `eval_rank_nowrap.py`, held-out map env seed 10000, walk seed 10000 + s, 100 walks.

## Readouts and branches (registered; `analyze_rank_nowrap.py::decide`)
SOLVED = final-5% training loss < 0.05 (`stats_core.classify_run`). Accuracy = held-out revisit accuracy at T=1024.
"X over Y" FIRES if two-sided Fisher on SOLVED has p < .05 with X's rate higher, OR the permutation test on accuracy
(`stats_core.perm2_p`) has p < .05 with mean(X) - mean(Y) >= the materiality floor. Within a grid accuracy is raw
(floor 0.02, as SCORE_RANK). Across grids it is **floor-relative**, (acc - retrace) / (1 - retrace) with the retrace
floor of that seed's own eval stream (floor 0.10, = 0.025 raw at 32, 0.013 at 256).
- K: B32 over A32 (control). G: AL over A32 (does removing wrap-around help rank 2). G': A32 over AL.
  R: BL over AL (rank gap on the large torus). R': AL over BL. G3': B32 over BL (rank 3 worse on the large torus).

Branches, evaluated in order (n = 12; "near" = SOLVED >= 9, "low" = SOLVED <= 3):
1. **CEILING** -- every run of every cell >= 0.999 accuracy.
2. **CONTROL FAILED (VOID)** -- K does not fire: the 32-torus deficit did not reproduce; nothing to explain.
3. **LARGE-GRID CONTROL FAILED** -- BL SOLVED < 6/12, or G3' fires: rank 3 does not work on the 256-torus at this
   init and budget; reported as it falls.
4. **LARGE GRID HURTS RANK 2** -- G' fires.
5. **PERIODIC CODE IS THE LIMIT** -- G fires, R does not, and AL SOLVED >= 9/12. (R not firing is "unmeasured",
   never "no gap".)
6. **PARTIAL** -- G fires but R fires or AL SOLVED < 9/12: rank 2 improves when wrap-around vanishes but stays short.
7. **RANK LIMIT IS GENERAL** -- G does not fire, R fires, AL SOLVED <= 3/12 (the 32-torus failure rate).
8. **INTERMEDIATE** -- G does not fire, R fires, AL SOLVED > 3/12.
9. **UNMEASURED** -- neither G nor R fires.
Qualifiers: REVERSAL if R' fires. **Hard-target qualifier (registered):** SOLVED is more lenient at 256 (87% of targets
are retrace-predictable vs 75% at 32), so a PERIODIC verdict carries "on the hard targets rank 2 still trails rank 3"
if, on non-retrace targets of the 256-torus (stratum `retrace_miss`), AL - BL <= -0.02 with permutation p < .05.
Void also: any run missing; md5 guard trips (checked at launch, before eval and before analysis).

## Power (`docs/audits/2026-10-06/rank_nowrap_power.py`, `_out.txt`; 400 simulations, runs decide() itself)
Pools of stored per-seed (SOLVED, floor-relative accuracy): per-head rank 2 3/16 SOLVED (RANK_ND D2 + RANK_MI), rank 3
14/16 (RANK_ND D2 + RANK3). P(correct branch):

| n per cell | AL like rank 3 -> PERIODIC | AL like rank 2 -> GENERAL (+ INTERMEDIATE) | AL at 0.5 -> INTERMEDIATE / PARTIAL / UNMEASURED |
|---|---|---|---|
| 8 | 0.85 | 0.75 (0.86) | 0.68 |
| 10 | 0.85 | 0.67 (0.93) | 0.89 |
| **12** | **0.94** | **0.77 (0.95)** | 0.85 |

The two poles never swap (P(PERIODIC | rank-2-like) and P(GENERAL | rank-3-like) are 0.00 at every n); the GENERAL
shortfall goes to INTERMEDIATE when AL draws 4+/12 SOLVED at p 0.19. n = 12 chosen; 0.8 for GENERAL alone is not
reached without widening "low", which would mislabel a half-solving AL.

## Secondaries (no verdict)
Interaction (BL - AL) - (B32 - A32) on SOLVED fraction, floor-relative and raw accuracy, bootstrap 95% CI; accuracy by
stratum on the eval stream (wrap-only / plain gap < 128 / plain gap >= 128 / retrace-predictable / not), asserted to
reproduce eval_nd's total; own training map vs held-out (40 walks) and SOLVED-but-held-out < 0.95 flags; r(final loss,
acc) overall and per grid; T=2048 (rule 10); run classes; clock / collapse / clean head classification
(`docs/theory/2026-10-04/scripts/basins.py` logic, per-head models, D from the config; declared descriptive) and
"SOLVED iff CLEAN" per cell; dropout-scale re-score (`rescore_hook`, attention x 1/(1-p)), branch recomputed and a FLAG
if any accuracy firing changes (verdict unchanged).

## Pilot (seeds 100-101, outside the batch) -- disclosed
- **Bitwise reproduction:** the batch's run path (`train_rank_nowrap`, `Vanilla_r2ph_om32`, grid 32, seed 0, RANK_ND
  flags) equals stored `runs/rank_nd/D2/Vanilla_r2ph_s0` on all 15 checked epochs, max |diff| 0.0
  (`docs/audits/2026-10-06/rank_nowrap_repro_out.txt`; a first attempt patched the wrong object and was inert).
- **Timing** (`rank_nowrap_pilot_timing.txt`): 8 pilot jobs, 4 per GPU, both GPUs also running another user's jobs:
  ~10 s/epoch at 4 jobs/GPU; 15-21 s/epoch on the GPU that carried 1-2 extra jobs. Same cost at grid 32 and 256.
- **Outcomes read before this file was finalised** (40-epoch schedule, end-to-end smoke of eval, re-score and analysis,
  `runs/rank_nowrap_pilot/SMOKE_ANALYSIS.txt`): all 8 runs unsolved/STALLED; held-out accuracy A32 0.63, B32 0.64,
  AL 0.60, BL 0.75. The branch structure was drafted before these were read; LOW and the G3' clause were added after a first
  power simulation (before any pilot accuracy existed), the hard-target qualifier while the smoke evals were running,
  before their output was read; no branch or threshold was changed after reading it. A 40-epoch schedule says nothing about 900-epoch SOLVED rates.
- Branch smoke test: 12/12 synthetic cases reach their branch (`rank_nowrap_branch_smoke.py`, `_out.txt`).

## Cost and ETA
48 runs x 900 epochs x ~10 s/epoch (4/GPU, measured under the current external load) = 120 GPU-slot-hours; at 8 slots
~15-18 h (10-12 s/epoch). Slots are shared with any concurrent mapformer batch (lib_driver counts every `mapformer.train_` job): with
another batch holding half the slots, ~30-35 h. Eval + re-score + analysis ~1 h.

## Scope and caveats (fixed now)
Grid size changes, besides the wrap share: the map (1024 vs 65536 cells; memorisation possible only at 32), the
revisit rate and targets per sequence (468 vs 296: less supervision per batch at 256, which works AGAINST rank 2 there),
and the target mix (retrace floor 0.750 vs 0.872, hence floor-relative accuracy and the hard-target qualifier). The
design separates wrap-around from initialisation, not from these. One length, one recipe, 2 heads, 1 layer, 900
epochs, omega initialised at grid 32; per-head rank only. PERIODIC would say rank D is hard on THIS torus because of
closure; it would not say rank 2 is sufficient in general (one plane walk, one budget).

Launch: `cd /home/prashr/mapformer && setsid nohup bash run_rank_nowrap.sh > /dev/null 2>&1 &`
