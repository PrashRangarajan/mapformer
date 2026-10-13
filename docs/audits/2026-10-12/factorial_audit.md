# FACTORIAL batch -- independent pre-launch code check (rule 29), 2026-10-12

Scope: `FACTORIAL_PREREG.md`, `analyze_factorial.py`, `run_factorial.sh` (+ `lib_driver.sh`), at commit 955ef61.
Read-only, no GPU, `runs/factorial` not read (does not exist). CPU tests wrote only under
`scratchpad/audit_fact/` (16 tiny `train_variant` runs with the driver's exact flags except
`--epochs 3 --n-batches 2 --batch-size 4 --n-steps 64 --device cpu`, the driver's eval and strata commands at
`--n-trials 2`, and a copy of `analyze_factorial.py` with only REPO / EPOCHS / the n_steps assert repointed).

**Verdict: no BUG that would corrupt a registered number. The pipeline runs end to end (train -> eval -> strata
-> analysis, rc 0; smoke ALL PASS). Two DESIGN items should go into an amendment before launch (D1 budget scoping,
D2 the CATCHES UP label); the rest is minor.**

## What was verified (no action needed)

1. **Each cell builds what it claims** (built through `VARIANT_MAP`, the trainer's own path):

   | cell | class | `len(layers)` | passes | params | plain `WMTransformerLayer` | SDPA calls / forward |
   |---|---|---|---|---|---|---|
   | RoPE_L1 (`RoPE`, `--n-layers 1`) | `MapFormerWM_RoPE` | 1 | 1 | 203,925 | 1 | 1 |
   | RoPE_L4 (`RoPE`, `--n-layers 4`) | `MapFormerWM_RoPE` | 4 | 4 | 798,741 | 4 | 4 |
   | RoPE_Loop (`RoPELooped`, `--n-layers 1`) | `MapFormerWM_RoPE_Looped` | 1 | 4 (`n_loops=4`) | 203,925 | 1 | 4 |
   | Path_L1 (`Vanilla_r4`, `--n-layers 1`) | `MapFormerWM_r4` (`model_rank._rank(4)`) | 1 | 1 | 204,757 | 1 | 1 |

   `RoPE` honours `--n-layers` (no Hourglass trap, rule 17). `RoPELooped` hard-codes 1 block and ignores
   `--n-layers` (built with 4: still 1 layer, 203,925 params); the driver passes 1, so the checkpoint's
   `n_layers=1` is truthful. `Vanilla_r4`: `w_in` (4, 128), `w_out` (64, 4) -- shared rank 4, the arm that solves
   8/8 at T=1024. `VARIANT_MAP` has a single entry for each key (no later duplicate overriding).
2. **`--fast-attn` takes effect in all four cells.** RoPE and RoPELooped use the plain `model.WMTransformerLayer`
   (imported from `model.py`), so the trainer's check finds `plain` non-empty and does NOT raise `SystemExit`; no
   "NOTE" line either (no subclass with its own forward). Counting `F.scaled_dot_product_attention` calls confirms
   SDPA runs in every layer / every loop pass. Eval-mode logits SDPA vs explicit: max |diff| 8.3e-07 (RoPE L4),
   7.2e-07 (RoPELooped, Vanilla_r4). The CPU runs with `--fast-attn` all trained and saved.
3. **Driver.** Seeds 500-503 (fresh; probe was seed 400). Flags = `run_rank_mi.sh` (T 1024, batch 16, 98 batches,
   lr 1e-3, cosine, 900 ep, 2 heads, d 128, `--n-landmarks 0`, `--data-workers 3`) plus `--fast-attn`; differences
   that do not touch results: OMP 3 vs 4, no `--save-full-state` (nothing can resume from optimizer state anyway:
   `--init-from` is a warm restart). Same recipe as SIGN_MATCHED's `RoPE` arm (the C0 expectation). Knobs are
   assigned after sourcing (rule 27). `DRV_GPUS=0` is exported and `drv_pick` iterates only over it. Data workers
   use the `spawn` context, so their argv carries neither `mapformer.train_` nor `--device`: they are not counted
   as trainers by `drv_ntrain`, the `running` check, or the wait loop. Duplicate guard: `flock -n`, skip on `.pt`,
   skip on a live process with the same `--output-dir`. 16 checkpoints required by count and by path. md5 guard
   covers every model module the four cells import plus the eval / analysis / stats code and the driver itself,
   re-checked before eval and before analysis. Done marker only after the analysis artifacts exist.
4. **Eval -> analysis names.** `eval_noise_refine` reads `<root>/p0/<V>_s<S>/<V>.pt` (tag `p0` for noise 0.0),
   writes `FACTORIAL_<root>.json` with keys `0.0|<V>|<T>` and rows `[seed, acc, nll]`; the analysis reads
   `J["0.0|<var>|1024"]`, `r[0]`, `r[1]`. Strata keys `<V>|<S>|1024`. Both evaluators rebuild with the checkpoint's
   `n_layers`, so the 4-layer RoPE loads strictly. RoPE at L1 and L4 share a variant name but live in different
   roots / JSONs and are told apart by the `n_layers` assert.
5. **Provenance asserts vs what is saved.** `train_variant` saves `config.n_layers`, `n_steps`, `epochs`,
   `fast_attn` and `losses` of length `--epochs`; all asserts pass on the CPU checkpoints (with n_steps / epochs
   repointed) and would fail on a wrong `--n-layers` or a missing `--fast-attn`.
6. **Decision functions vs prereg.** C0 / C1 / C2 / C3 conditions match the text. Directions are right:
   `perm2_p(a, b)` tests mean(b) - mean(a), and every call passes (index/base, path/deep) with d computed the same
   way. CATCHES UP and DOES NOT CATCH UP are mutually exclusive (CATCHES UP needs d <= 0.02 and SOLVED_i >= 3, so
   neither d >= 0.05 nor Fisher with Path ahead can hold). All branches reachable at n = 4: exact perm min p
   2/70 = 0.0286 (needs complete separation; one overlap gives >= 0.057), Fisher 4/4 vs 0/4 = 0.0286 (3/4 vs 0/4 or
   4/4 vs 1/4 = 0.143). SOLVED = `classify_run` final-5% mean < 0.05, as registered.

## DESIGN

**D1. The labels are not budget-scoped, and the expected outcome is already predicted (rules 3, 4).**
900 epochs is RANK_MI's budget for 1-layer models; it has never been run for 4-layer index RoPE at T=1024 (the
5-epoch probe: RoPE L4 loss 1.11 vs Vanilla_r4 L4 0.42). `LOOPED_PILOT.md` / `HORIZON_RESULTS.md` (T=128, 300 ep,
n=3) put RoPE L4 and RoPE loop x4 horizons at 17-32 steps (0.523 / 0.512 at gap 33-64), so DOES NOT CATCH UP is
the prior-predicted result; most T=1024 revisits (the `plain_lag>=128` and `wrap` strata) lie far past that
horizon. Fix (amendment): (a) cite that prediction in the prereg so the outcome is not read as new; (b) suffix
every C1/C2/C3 label with "at 900 epochs" and the per-cell class counts, and register: if >= 2/4 runs of an index
cell are DESCENDING, its DOES NOT CATCH UP reads "DOES NOT CATCH UP AT THIS BUDGET (k/4 still descending)" and the
named follow-up is a longer budget, not a capability claim.

**D2. CATCHES UP is a margin rule with no test, and it can fire while Path is significantly ahead.** In the
shipped smoke, `catch_up([0.99, 0.995, 0.99, 0.985], [0.998, 0.999, 0.997, 0.998], 4, 4)` returns
"INDEX CATCHES UP ... Path_L1 - RoPE_L4 +0.0080 (perm p 0.0286)". At n = 4 no equivalence test is possible
(the 95% permutation CI cannot be narrower than the data). Fix: rename to "WITHIN 0.02 OF PATH (margin rule, not an
equivalence test)" or keep the name and print `stats_core.perm2_ci(index, path)` beside it, registering that a
significant Path lead (perm p < .05) is reported in the same line.

**D3. SOLVED is a training-map criterion; deep index models can memorise the training map.** Training uses one
fixed 64x64 map per seed; evaluation redraws it (`--env-seed 10000`). An 800k-parameter 4-layer model could lower
its training loss by self-localising on the memorised map (CLAUDE.md: a 100-cell map is memorised, own 0.986 vs
unseen 0.273) -- SOLVED without in-context mapping. It cannot produce a false CATCHES UP (that also needs held-out
accuracy), but it can make SOLVED counts and the Fisher branch mean different things for index and path cells.
Fix (secondary, eval only, cheap): own-map accuracy per seed, e.g.
`for S in 500 501 502 503; do python3 -m mapformer.eval_noise_refine --runs-dir $R/L4 --variants RoPE --noises 0.0
--seeds $S --env-seed $S --lengths 1024 --n-trials 100 --out $REPO/FACTORIAL_L4_OWN_s$S.md; done` (and likewise per
root); report own - held-out per cell; flag MEMORISED when own-map is high and held-out is not.

**D4. What the design can and cannot claim.** Index at 4 layers (4x params) and loop x4 (matched params, matched
init: `RoPELooped` and `RoPE` L1 at the same seed are built from identical initial weights and see the same data
stream) against path at 1 layer answers "is depth 4 enough", not the exchange rate (Dyck's "~3 layers"): with no
L2 / L3 index cell a CATCHES UP bounds the rate only as <= 4. Say so in the scope line; a `RoPE --n-layers 2` cell
(~1.3x an L1 run) would bracket it if wanted. Single width, length, recipe, n_heads=2.

**D5. n = 4 and multiplicity.** Every perm verdict needs complete rank separation and every Fisher verdict 4/4 vs
0/4; any family-wise correction (Holm over C1, C2) would make the min p 0.057 and the verdicts unreachable. State
in the prereg that the four contrasts are separate questions with no family correction, and why. Adequate for the
predicted outcome (large, separated gap); the bimodal middle will read UNMEASURED by design (n = 6: min p 0.0022,
+50% cost, ~12 h).

## minor

- m1. `eval_rank_strata` is called without `--check-json "$REPO/FACTORIAL_${ROOT}.json"` (RANK_MI passes it); the
  cross-check that strata 'all' equals the primary accuracy is lost. Verified on CPU that adding it passes.
- m2. The analysis prints the RANK_MI floors as constants; the strata JSON carries the measured per-run constant
  floor on THIS stream (`floor` per stratum) -- print its mean (rule 1).
- m3. Registered secondaries with no committed code (rule 7): r(final loss, acc), accuracy at 512 / 2048 (only in
  `FACTORIAL_<root>.md`, not in the analysis), the dropout re-score (`rescore_hook.py --scale auto --
  mapformer.eval_noise_refine ...` for the L1 root). Size of the last: sign-batch RoPE +0.004 under re-score, so the
  eval-mode bias in C0 / C3 is far below the 0.05 thresholds.
- m4. Provenance does not assert lr, batch_size, schedule, n_batches, d_model, n_heads, n_landmarks (all saved in
  `config`), nor `n_loops == 4` for RoPELooped (a class attribute, not saved). Cheap to add.
- m5. The strata loop formats `v[st]['acc']` with `:.3f`; an empty stratum (acc None) raises TypeError after the
  registered lines but before `FACTORIAL_VERDICTS.json` is written, so the done marker would not be set. Unlikely at
  T=1024 x 100 trials (every stratum had n >= 59 at 2 trials), but guard it.
- m6. Evaluation runs the explicit attention path on SDPA-trained weights (logits agree to ~1e-6); say so in the
  prereg next to the fast-attn sentence. The fast-attn series is non-deterministic run to run (no
  `--deterministic`); C0's expectation (0.731 / 0.998) comes from explicit-path stored runs and is context only.
- m7. Probe numbers: `fact_probe2/mem.log` shows ~650-740 MiB per process, not ~2 GB; RoPELooped was not timed with
  `--fast-attn` (only RoPE L4 and Vanilla_r4 L4); timing was at 2 concurrent, the batch runs 4. The ~8 h estimate is
  consistent with GPU-bound throughput but unmeasured at 4 jobs.
- m8. Any fix to `analyze_factorial.py` or `run_factorial.sh` must land before launch: both are in the md5 guard, so
  a later edit makes the driver `drv_fail` before eval / analysis (intended).
