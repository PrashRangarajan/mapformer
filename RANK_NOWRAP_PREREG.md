# Is per-head rank 2's torus failure about wrap-around? -- pre-registration (2026-10-06, before any batch run)

> **Amendments 1-3 (end of file) supersede this body where they differ; each later one supersedes the earlier.** Final
> design: 5 cells (adds a redrawn-map rank-2 control M32 on the 32-torus), n = 10 (seeds 60-69, 50 runs; n = 12 is the
> recommendation put to the user, Amendment 3); the registered quantity is p = held-out accuracy on the PLAIN HARD
> targets (missed by the retrace-or-blank floor and not wrap-only), HIT = p >= 0.98; every accuracy arm is a permutation
> test on p (materiality 0.10); branch set of Amendment 2.

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
Qualifiers: REVERSAL if R' fires. **Hard-target qualifier (registered):** SOLVED is more lenient at 256 (the retrace-or-blank
floor -- the retraced observation inside a run reversing the previous one, blank otherwise -- predicts 87% of targets
vs 75% at 32; wording corrected in Amendment 1), so a PERIODIC verdict carries "on the hard targets rank 2 still trails rank 3"
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
stratum on the eval stream (wrap-only / plain gap < 128 / plain gap >= 128 / retrace_ok = predicted by the
retrace-or-blank floor / retrace_miss = non-blank targets outside a retrace run), asserted to
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
revisit rate and targets per sequence (468 vs 296; hard targets ~117 vs ~39 per sequence, but the 32-torus surplus is
ALL wrap-only: PLAIN hard targets are ~36 vs ~39 per sequence -- corrected in Amendment 3),
and the target mix (retrace floor 0.750 vs 0.872, hence floor-relative accuracy and the hard-target qualifier). The
design separates wrap-around from initialisation, not from these. One length, one recipe, 2 heads, 1 layer, 900
epochs, omega initialised at grid 32; per-head rank only. PERIODIC would say rank D is hard on THIS torus because of
closure; it would not say rank 2 is sufficient in general (one plane walk, one budget).

Launch: `cd /home/prashr/mapformer && setsid nohup bash run_rank_nowrap.sh > /dev/null 2>&1 &`

---

## Amendment 1 (2026-10-06, after an independent code/design audit, BEFORE launch; no batch result exists)
All changes are CPU-side; no GPU job was run for this amendment (the user's instruction while another user's jobs hold
the GPUs). Every item below is in the md5-guarded code as launched.

**D1 -- SOLVED is not comparable across grids.** The hard targets (missed by the retrace-or-blank floor) are 0.25 of
revisit targets at 32 but 0.128 at 256, so the 0.05 training-loss cut is ~2x more lenient at 256. Chosen fix: base
every registered count on floor-relative held-out accuracy, not on loss. Per run rel = (acc - f) / (1 - f), f the
retrace-or-blank floor of that seed's own eval stream, i.e. the share of the hard targets recovered; a run **HITS** if
rel >= 0.90. Justification: rel is the same quantity on both grids by construction (scaling the loss cut by the hard
share would still mix in the loss on easy targets and on the training map), it is held-out (immune to memorising the
training map), and on all 32 stored per-head runs it agrees with loss-SOLVED 32/32 (rank 2 3/16, rank 3 14/16; values in
`rank_nowrap_power_out.txt`). HIT replaces SOLVED in: the within-grid Fisher arms, the PERIODIC threshold (AL HIT >= 8/10),
the GENERAL threshold (AL HIT <= 2/10) and the large-grid control (BL HIT >= 5/10). **The cross-grid contrasts (G, G',
G3') fire only through the permutation test on rel** (d >= 0.10), never through a count. Loss-SOLVED and the
pre-amendment readout are printed as secondaries.

**D2 -- memorisation is a second explanation.** A32 can memorise its 1024-cell training map, AL cannot (65536 cells).
New cell **M32**: per-head rank 2 on the 32-torus with the observation map REDRAWN every trajectory
(`environment_nd_redraw.GridWorldNDRedraw`, variant `Vanilla_r2ph_om32_redraw`, same om32 init). It overrides only
`generate_trajectory` (redraw, then the parent's walk); the map is drawn by GridWorldND's own three lines from a
RandomState seeded by a CRC32 of the global RNG state, so it consumes nothing from the walk's RNG: at a seed M32's action
stream is identical to A32's. Checks and CPU gate (`rank_nowrap_redraw_gate.py`, `_out.txt`, all PASS): map code equals
GridWorldND's for fixed seeds; identical actions, cells, revisit masks and final RNG state vs the parent over 20 walks;
observations differ at 0.734 of steps (= 1 - (0.25 + 0.25/16), independent maps); 30/30 distinct maps; pickles and
redraws deterministically in a data worker. Gate T=1024: revisit 0.457, wrap-only 0.361 (as A32), blank 0.499, retrace-or-
blank 0.737, best action n-gram 0.496, observation n-gram 0.499 / 0.486 / 0.406 -- PASS (chance-level; nothing to
memorise). Data cost equal (0.096 vs 0.100 s per batch, `rank_nowrap_redraw_datatime.txt`). Trainer path smoke-tested on
CPU (the wrapper selects the redraw class by variant name). M32 is evaluated on the same fixed held-out map as A32.
New within-grid contrasts: **Mem** = M32 over A32; **Km** = B32 over M32.

**Branches (replace the body's list; evaluated in order; n = 10; near = HIT >= 8, low = HIT <= 2, half = 5):**
1. CEILING -- every run of every cell >= 0.999.
2. CONTROL FAILED (VOID) -- K (B32 over A32) does not fire.
3. LARGE-GRID CONTROL FAILED -- BL HIT < 5, or G3' fires.
4. LARGE GRID HURTS RANK 2 -- G' fires.
5. Rescue = G fires, R does not, AL HIT >= 8. Then:
   - **PERIODIC CODE IS THE LIMIT** -- Km fires and Mem does not (rank 2 still fails the 32-torus with nothing to
     memorise, and solves the torus without wrap-around);
   - **MAP MEMORISATION WAS THE LIMIT** -- Mem fires and Km does not (with a redrawn map rank 2 solves the 32-torus,
     wrap-around included: the 32-torus failure was the fixed map);
   - **LARGE TORUS RESCUES RANK 2, CAUSE UNRESOLVED** -- otherwise.
6. PARTIAL -- G fires, not a rescue.
7. If R fires (G does not): **MEMORISATION AT 32, LARGE TORUS FAILS** if Mem fires and Km does not; else **RANK LIMIT
   IS GENERAL** if AL HIT <= 2 and Km fires; else INTERMEDIATE.
8. UNMEASURED -- neither G nor R fires.
Qualifiers: REVERSAL (R' fires); the M32 status line on every branch.

**D3 -- supervision.** [CORRECTED in Amendment 3: the "one third" reading below is wrong -- the 32-torus's surplus of
hard targets is entirely wrap-only; plain hard targets per sequence are equal, ~36 vs ~39.] Hard-target supervision at
256 is about one third of that at 32 (~38 vs ~117 hard targets per sequence; 0.128 x 296 vs 0.25 x 468).

**D4 -- the hard-target qualifier is in the registered path.** The strata are computed before the verdict, the
qualifier is evaluated inside `decide()` (AL - BL on retrace_miss <= -0.02 with permutation p < .05) and printed on the
REGISTERED line as `[ON THE HARD TARGETS RANK 2 STILL TRAILS RANK 3]` when the branch is one of the three rescue
branches; on other branches it is printed as "(record)".

**N5 -- wording.** "retrace" everywhere means the retrace-or-blank predictor; retrace_ok = targets it predicts (inside
a retrace run, or blank); retrace_miss = non-blank targets outside a retrace run. Fixed in the body (two places).

**N6 -- paired G (declared secondary).** Per seed rel(AL_s) - rel(A32_s) (identical init and action streams per
seed), mean, sign count and exact sign-flip p.

**N8 -- duplicate-launch guard.** Before launching (and again after waiting for a slot) the driver skips a run whose
output dir already appears as `--output-dir <dir> ` in a running python3's argv (`ps -u $USER -o comm=,args=` + awk,
rule 23). Dry run with a fake trainer holding one run dir: 49 launches, that one skipped.

**N9 -- CPU fallback.** The strata-vs-eval_nd equality is asserted only on the GPU; on a CPU fallback a difference is
printed as a WARN (argmax ties), and the verdict is still produced. The registered accuracy always comes from eval_nd.

**n and power.** n = 10 per cell (seeds 60-69), 5 cells, 50 runs. Power (`rank_nowrap_power.py`, runs the amended
`decide()`, 400 simulations; stored-run pools as above), P(target branch):

| truth (AL, M32) | target | n = 8 | **n = 10** | n = 12 |
|---|---|---|---|---|
| rank-3-like, rank-2-like | PERIODIC CODE IS THE LIMIT | 0.76 | **0.84** | 0.93 |
| rank-3-like, rank-3-like | MAP MEMORISATION WAS THE LIMIT | 0.82 | **0.83** | 0.92 |
| rank-2-like, rank-2-like | RANK LIMIT IS GENERAL (+ INTERMEDIATE) | 0.71 (0.84) | **0.66 (0.91)** | 0.80 (0.95) |
| rank-2-like, rank-3-like | MEMORISATION AT 32, LARGE TORUS FAILS | 0.80 | **0.87** | 0.93 |
| AL solves half the time | INTERMEDIATE / PARTIAL / UNMEASURED | 0.74 | **0.87** | 0.89 |

PERIODIC and MAP MEMORISATION are never confused with each other (<= 0.01) and GENERAL never yields a rescue branch
(0.00) at n = 10. GENERAL alone is below 0.8 at n = 10; its shortfall goes to INTERMEDIATE (AL drawing 3+/10 hits).

**Branch smoke test** (`rank_nowrap_branch_smoke.py`, `_out.txt`): 16/16 synthetic cases reach their branch or
qualifier, every branch above included.

**Cost.** 50 runs x 900 epochs x 10-12 s/epoch (the earlier pilot, 4 jobs per GPU under the external load; the redraw
arm's data cost is equal) = 125-150 slot-hours: ~16-19 h with all 8 slots; ~30-38 h if another batch holds half.
Not re-measured on GPU for this amendment (no GPU use allowed now).

**Pilot disclosure.** No new outcome was read: the M32 pilot checkpoints (seeds 100-101) were trained on the CPU for
3 x 10 batches only, to run eval, re-score and analysis end to end on the CPU path; their accuracies are meaningless.
Earlier 40-epoch pilot outcomes are as disclosed above. The CPU smoke (`runs/rank_nowrap_pilot/amend1/`, `cpu_smoke.sh`, `SMOKE_ANALYSIS.txt`) ran train (M32) -> eval -> re-score -> analysis on the CPU with all five cells.
`train_rank_nowrap.py` gained a `main()` that swaps the environment for `*_redraw` variants only; the om32 path is the
same call (`train_variant.main()`), so the 15-epoch bitwise reproduction stands (not re-run: no GPU).

Launch (unchanged): `cd /home/prashr/mapformer && setsid nohup bash run_rank_nowrap.sh > /dev/null 2>&1 &`

---

## Amendment 2 (2026-10-06, after a second independent audit of Amendment 1, BEFORE launch; CPU only; no batch result exists)
The audit found the Amendment 1 code correct but one major design flaw and several smaller ones. Finding -> change:

**1 (MAJOR). Amendment 1's floor-relative accuracy is not grid-comparable.** rel = h - f/(1-f) (1 - acc_floor-predicted):
an error on a floor-predicted target costs f/(1-f) = 3.01 units of rel at 32 but 6.70 at 256 (blank targets are free
for the floor, not for a model), so identical per-stratum behaviour moves rel across grids and near-solved runs flip
HIT; the Amendment 1 power script had assumed rel invariant. **Change:** the registered quantity is now
**h = held-out accuracy on the hard targets** -- the stratum retrace_miss, the targets the retrace-or-blank floor gets
wrong (= the non-blank targets outside a retrace run; inside a retrace run the floor is always right: 0 exceptions on
every stream checked, asserted in the analysis). h is the same quantity on both grids. **HIT = h >= 0.90**; the
cross-grid contrasts G, G', G3' are permutation tests on h. Validation (`rank_nowrap_hard_validate.py`, `_out.txt`,
CPU, 60 held-out walks per run, 32 stored runs: RANK_ND D2 r2/r3 on the ND 32-torus, RANK_MI r2 / RANK3 r3 on the paper
64-torus): HIT agrees with training-loss SOLVED on **32/32** runs (lowest h among SOLVED 0.960, highest among not SOLVED
0.776; 0.90 sits in the gap). At 256 there are no stored runs; under per-stratum transport (each ND run's per-stratum
accuracies re-weighted by the 256 stratum mix: copy 0.743 / blank_out 0.127 / hard 0.130, vs 0.475 / 0.276 / 0.250 at
32) h and HIT are invariant by construction, whereas rel moved (e.g. rank-2 s5 -0.100 -> -0.421) and held-out NLL
roughly halved (the loss-cut leniency of Amendment 1's D1). Hard targets per held-out stream at 256: ~3800 (100 walks),
so h is precise to ~0.01.

**2. The within-grid materiality floor was not grid-neutral** (0.02 raw = 0.08 hard-target units at 32, 0.156 at 256).
**Change:** every accuracy arm, within and across grids, is a permutation test on h with materiality **0.10 in h units**;
within a grid a contrast also fires on Fisher over HIT counts. Raw accuracy is a secondary only (and decides CEILING).

**3. M32 branch asymmetry (absence of evidence).** **Change:** "M32 fails like rank 2" = Km fires AND Mem does not AND
(M32 HIT <= 2/10 OR Mem' fires); "M32 solves like rank 3" = Mem fires AND Km does not AND M32 HIT >= 8/10 (the threshold
AL needs). PERIODIC CODE IS THE LIMIT and RANK LIMIT IS GENERAL require the first; FIXED MAP WAS THE LIMIT and FIXED MAP
AT 32, LARGE TORUS FAILS require the second; anything else is CAUSE UNRESOLVED (rescue) or INTERMEDIATE (no rescue).

**4. M32 confounds.** **Change:** reverse test **Mem' = A32 over M32**; when it fires a qualifier says redrawing made the
32-torus HARDER for rank 2 (no fixed-map stepping stone; or the redraw adds map diversity / removes train-test overlap).
Branches renamed **FIXED MAP WAS THE LIMIT** and **FIXED MAP AT 32, LARGE TORUS FAILS**; their text says "the fixed map
(memorisation, train/test shift, or map diversity)", not "memorisation". **Paired Mem** (h(M32_s) - h(A32_s), shared init
and action streams per seed, sign-flip p) added as a declared secondary beside paired G.

**5.** The secondary "pre-amendment readout" is relabelled **"amended branches with loss-SOLVED in place of HIT"**.

**6. Dropout re-score.** The analysis now re-computes the strata itself under `rescore_hook.install('auto')`
(attention x 1/(1-p)), so the re-scored branch uses re-scored h, HIT AND hard-target qualifier; eval_nd's re-scored raw
accuracy (driver) is used only for CEILING. Firing differences are FLAGged; the registered verdict is unchanged.

**7. N7 of the first audit.** Its content was not relayed to this session (the first amendment request listed D1-D4,
N5, N6, N8, N9). It is therefore neither fixed nor dropped here: **OPEN -- to be supplied by the coordinator and
recorded in a further amendment before launch.**

**8. Provenance and devices.** `train_rank_nowrap.py` writes `map_redrawn`, `train_env_class` and `omega_init_grid` into
each checkpoint's config after training (atomic replace); the analysis asserts map_redrawn is True exactly for M32
(strict in the registered run; pilot checkpoints predating the key are accepted only in smoke mode). Eval and re-score
run on the GPU with the most free memory (driver `best_gpu`), the analysis picks its own (`pick_device`).

**Registered branches (replace Amendment 1's; in order; n = 10; near = HIT >= 8, low = HIT <= 2, half = 5):**
1 CEILING (every run of every cell raw >= 0.999); 2 CONTROL FAILED (VOID) (K does not fire); 3 LARGE-GRID CONTROL FAILED
(BL HIT < 5 or G3' fires); 4 LARGE GRID HURTS RANK 2 (G' fires); 5 rescue = G fires, R does not, AL HIT >= 8 ->
**PERIODIC CODE IS THE LIMIT** (M32 fails like rank 2) / **FIXED MAP WAS THE LIMIT** (M32 solves like rank 3) /
**LARGE TORUS RESCUES RANK 2, CAUSE UNRESOLVED**; 6 PARTIAL (G fires, not a rescue); 7 R fires: **FIXED MAP AT 32, LARGE
TORUS FAILS** (M32 solves like rank 3) / **RANK LIMIT IS GENERAL** (AL HIT <= 2 and M32 fails like rank 2) /
INTERMEDIATE; 8 UNMEASURED. Qualifiers: REVERSAL (R'), HARDER (Mem'), the M32 status line, and the hard-target
qualifier (AL - BL on h <= -0.02, p < .05), attached on rescue branches and "(record)" otherwise.

**Smoke tests.** Branch smoke (`rank_nowrap_branch_smoke.py`, `_out.txt`): 18/18 synthetic cases, every branch and
qualifier above. CPU end to end (`runs/rank_nowrap_pilot/amend2/`, `cpu_smoke.sh`, `SMOKE_ANALYSIS.txt`): M32 trained on
the CPU through the wrapper (config key written), eval, re-score, analysis incl. the in-analysis re-score.

**Power with per-stratum transport** (`rank_nowrap_power.py`, `_out.txt`, 400 simulations, runs the amended
`decide()`; pools = the 32 validated runs as per-stratum vectors). P(target branch):

| truth (AL, M32) | target | n = 8 | **n = 10** | n = 12 |
|---|---|---|---|---|
| rank-3-like, rank-2-like | PERIODIC CODE IS THE LIMIT | 0.66 | **0.63** (+0.25 CAUSE UNRESOLVED) | 0.77 |
| rank-3-like, rank-3-like | FIXED MAP WAS THE LIMIT | 0.79 | **0.75** | 0.86 |
| rank-2-like, rank-2-like | RANK LIMIT IS GENERAL | 0.61 | **0.48** (+0.44 INTERMEDIATE) | 0.65 |
| rank-2-like, rank-3-like | FIXED MAP AT 32, LARGE TORUS FAILS | 0.76 | **0.80** | 0.90 |

**GENERAL truth -> LARGE GRID HURTS RANK 2: 0.01 at n = 10** (0.02 at n = 8 and 12), against the audit's 0.29 for
Amendment 1's rel under transport. No truth produces its opposite pole (PERIODIC vs FIXED MAP <= 0.01; GENERAL -> any
rescue branch 0.00). The price of item 3 (positive evidence that M32 fails: HIT <= 2/10 has probability ~0.7 when M32 is
rank-2-like) is lower power for PERIODIC and GENERAL, whose shortfall goes to CAUSE UNRESOLVED / INTERMEDIATE, never to
a wrong verdict. **n = 10 is kept as instructed; n = 12 (60 runs, +20% cost) would lift PERIODIC to 0.77 and GENERAL
to 0.65** -- a decision for the user before launch.

**Cost** unchanged: 50 runs, ~125-150 slot-hours, ~16-19 h on 8 slots, ~30-38 h sharing half (GPU timing not
re-measured; no GPU use allowed).

Launch (unchanged): `cd /home/prashr/mapformer && setsid nohup bash run_rank_nowrap.sh > /dev/null 2>&1 &`

**N7 (first audit), resolved by the main session:** N7 was procedural -- "amend the analysis before launch, not after",
because `analyze_rank_nowrap.py` is md5-guarded and re-checked before eval, so a post-launch edit trips the guard and the
batch ends without a done marker. Amendments 1 and 2 were both made before launch; nothing to change in the code.

---

## Amendment 3 (2026-10-06, after a third independent audit, BEFORE launch; CPU only; no batch result exists)
The audit found the Amendment 2 code correct but a residual of the comparability problem. Finding -> change:

**1 (MAJOR). h is not composition-comparable.** On the 32-torus the hard set is 69% wrap-only (80.8 per sequence, lag
median 330, 45 turns) and 31% plain (36.0 per sequence, lag 80, 11 turns); on the 256-torus it is all plain (38.8 per
sequence, lag 80, 11 turns) -- the 256 hard set matches the 32 PLAIN hard targets, not the 32 hard set
(`rank_nowrap_hard_validate_out.txt`). Stored RANK_ND rank-2 runs score higher on the plain part than on h (e.g. s0
0.891 vs 0.622). So under GENERAL truth h rises at 256 merely because rank 2 stops being scored on what it fails (G and
PARTIAL inflated, G' / G3' biased toward not firing). **Change:** the registered quantity is **p = held-out accuracy on
the plain hard targets** (stratum hard_plain: missed by the retrace-or-blank floor AND not wrap-only). On the 256-torus
p = h; on the 32-torus p is the plain part. p is used for HIT, every within-grid test (K, R, R', Mem, Mem', Km) and the
cross-grid tests (G, G', G3'), and the hard-target qualifier. The question it asks: does training without wrap-around
make rank 2 better on the SAME kind of targets? h (all hard) and the wrap-only hard accuracy are declared secondaries,
with the within-cell wrap deficit (p - wrap-only) at 32 and B32/M32 - A32 on wrap-only targets, and the branch set
recomputed on h (labelled not kinematically matched).

**2. HIT threshold.** Validated on the plain hard targets of the 32 stored runs: lowest p among loss-SOLVED 0.990,
highest among not SOLVED 0.975 (a narrow gap, 0.015). **HIT = p >= 0.98**: agrees with loss-SOLVED 32/32; margin 0.010
to the lowest SOLVED run and 0.005 to the highest unsolved one (a paper-torus rank-3 run). Stated honestly: the criterion
is lenient by up to ~0.02 of plain hard targets, and runs within ~0.005 of the cut are classed by sampling noise (SE of
p ~0.0023 at 0.98 on ~3700 targets per stream). (h with its Amendment 2 cut 0.90 also agrees 32/32 but is not matched.)

**3. Power with WRAP-AWARE per-stratum transport** (`rank_nowrap_power.py`, `_out.txt`; ND 32-torus pools only, 8 rank-2
runs (1 HIT) and 8 rank-3 runs (8 HIT); strata copy / blank_out / hard_wrap / hard_plain, weights 0.475 / 0.276 / 0.173 /
0.077 at 32 and 0.743 / 0.127 / 0 / 0.130 at 256; 400 simulations through the registered `decide()`; GENERAL = rank 2's
per-stratum competence unchanged at 256). P(target branch) and the confusions:

| truth (AL, M32) | target | n = 8 | **n = 10** | **n = 12** |
|---|---|---|---|---|
| rank-3-like, rank-2-like | PERIODIC CODE IS THE LIMIT | 0.91 | **0.84** (0.15 CAUSE UNRESOLVED) | **0.91** (0.09) |
| rank-3-like, rank-3-like | FIXED MAP WAS THE LIMIT | 0.99 | **1.00** | **1.00** |
| rank-2-like, rank-2-like | RANK LIMIT IS GENERAL | 0.85 | **0.75** (INTERMEDIATE 0.21, PARTIAL 0.03, LARGE GRID HURTS 0.01) | **0.86** (0.10 / 0.03 / 0.01) |
| rank-2-like, rank-3-like | FIXED MAP AT 32, LARGE TORUS FAILS | 0.93 | **0.94** | **0.95** |
| AL half | INTERMEDIATE / PARTIAL / UNMEASURED | 0.79 | **0.90** | **0.88** |

Under GENERAL truth PARTIAL fires in 0.03 of worlds (the audit's up to one third came from h's composition). No truth
produces its opposite pole. n = 8 beats n = 10 on PERIODIC and GENERAL because "low" = floor(0.25 n) is 2 at both: the
M32 / AL <= low requirements are easier with 8 draws than with 10. CAVEAT: the ND32 rank-3 pool never fails (8/8), so
the controls (K, BL, Km) look perfect here; on the paper torus rank 3 failed 2/8, so these are upper bounds.
**Recommendation: n = 12** (60 runs, seeds 60-71, +20% cost) lifts PERIODIC 0.84 -> 0.91 and GENERAL 0.75 -> 0.86 and
gives margin for the optimistic rank-3 pool. The code is registered at n = 10 (N_SEEDS, driver seeds 60-69); choosing
n = 12 changes exactly `N_SEEDS = 12` (analysis) and `seq 60 71` / the checkpoint count 60 (driver), before launch.

**4. Wording.** "One third of the hard-target supervision" removed from the analysis (FIXED MAP AT 32 text) and corrected
in the body caveats and Amendment 1's D3 (marked CORRECTED): the 32-torus surplus of hard targets is all wrap-only;
plain hard targets per sequence are ~36 (32) vs ~39 (256). p and h count only non-blank labels (a lost model that
defaults to blank scores 0 there), identically on both grids.

**5. Re-score hooks asserted.** After `rescore_hook.install`, the analysis requires `STATS['hooked']` == (models loaded
after the install) x (layers per model) = 5 x n x 1 and no skipped layer classes, else it exits non-zero (no done marker).

**6. Device selection.** Driver `best_gpu`: nvidia-smi under `timeout 30`, empty output guarded, a GPU needs >= 3000 MiB
free, every try logged, up to 60 retries x 30 s, then the batch fails (no done marker). Analysis `pick_device`: the GPU
with most free memory if >= 3000 MiB (`mem_get_info` failures caught), polls up to 30 min, then the CPU (strata check
becomes a WARN), every decision printed.

**7. Crossed strata.** hard_wrap (hard x wrap-only) and hard_plain (hard x plain) are computed for every run and printed
with the other strata and in the stratum contrasts.

**8. Driver knobs.** `lib_driver.sh` pre-sets DRV_SPACING=45 and DRV_MINFREE=4500, so `DRV_SPACING="${DRV_SPACING:-15}"`
was a no-op (spacing 45). Now assigned unconditionally after the source: `DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING=15;
DRV_MINFREE=4500`, logged at start. MINFREE 4500 MiB is kept: a T=1024 B16 per-head job uses ~3.4 GB (lib_driver's
efficiency audit #4, rank config); not re-measured (no GPU use allowed). Verified by sourcing
(`rank_nowrap_driver_knobs_out.txt`: old 45, new 15).

**N7** (first audit) remains OPEN as recorded in Amendment 2 (content not relayed to this session).

**Tests re-run.** Branch smoke 18/18 (`rank_nowrap_branch_smoke_out.txt`); validation (`rank_nowrap_hard_validate_out.txt`);
power (`rank_nowrap_power_out.txt`); CPU end-to-end analysis on the Amendment 2 smoke checkpoints with the Amendment 3 code
(`runs/rank_nowrap_pilot/amend2/SMOKE_ANALYSIS_A3.txt`), including the hook-count assertion.

Cost: unchanged at n = 10 (50 runs, ~16-19 h on 8 slots); n = 12: 60 runs, ~19-23 h on 8 slots, ~38-45 h sharing half.
Launch (unchanged): `cd /home/prashr/mapformer && setsid nohup bash run_rank_nowrap.sh > /dev/null 2>&1 &`
