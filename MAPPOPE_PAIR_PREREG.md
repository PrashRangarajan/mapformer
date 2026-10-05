# MapPoPE's two changes, separated -- pre-registration (2026-10-04, before any run of the batch)

## Question
Our MapPoPE differs from MapWM in two ways at once: (1) the SCORE RULE -- PoPE's
a_ts = sum_c softplus(q)_c softplus(k)_c cos(theta_t,c - theta_s,c + delta_c) instead of rotating content Q/K by the
path angle; (2) the FREQUENCY COUNT -- one angle per element (64 per head, `model_pope._widen_to_d`) instead of one per
pair (32). Every MapPoPE vs MapWM number in the project changes both. The confound was flagged three times and never
tested (`MAPPOPE_R4_RESULTS.md`: the rank-4 upgrade gives MapPoPE +0.019 vs MapWM +0.085, suspected because 64 angles
make the rank-2 bottleneck bind less; `T2_RESULTS.md`, `JSB_LENGTH_RESULTS_RANK.md`: "needs a PoPE variant with
pair-wise frequencies, which is not built"). The only earlier runs at 32 PoPE angles predate commit 3fb40a4 (which also
changed the delta clamp) and were on the voided hier-goal task. On the paper torus at rank 2, MapPoPE beats MapWM
(paper2x2: 0.999 vs 0.971). Which change carries that?

## The new arm (`model_pope_pair.py`)
MapPoPE-Pair: MapWM's path machinery unchanged (32 angles per head, omega 2pi .. 2pi/64, same rank), each angle
shared by two adjacent elements, PoPE's layer unchanged (per-element softplus magnitudes, per-element delta_c clamped to
[-2pi, 0], zero init). Checks (committed): it equals MapPoPE-Flat whose 64 angles are tied in pairs, max logit diff
0.0 (`docs/audits/2026-10-04/mappope_pair_equiv.py` / `_out.txt`); all six arms causal (0 leak), omega range identical,
angle counts 32 / 64 / 32 as intended, ranks asserted. Parameters at the task's vocabulary (21): MapWM 204,373, MapPoPE-Pair
204,501, MapPoPE-Flat 204,693; r4 204,757 / 204,885 / 205,205. Trainer `train_pair.py` = `train_variant.main()` with the two arms added to VARIANT_MAP (the file
earlier batches md5-guard is not edited); eval `eval_pair.py` = `eval_noise_refine` the same way.

## Batch (`run_mappope_pair.sh`; one batch, all retrained, seeds 10-17 fresh)
Arms: MapWM r2 (`Vanilla`), MapPoPE-Pair r2, MapPoPE-Flat r2 (64 angles), MapWM r4 (`Vanilla_r4`), MapPoPE-Pair_r4,
MapPoPE_r4 (64). 48 runs (72 after Amendment 1). Recipe = paper2x2 exactly: 300 epochs x 98 batches, batch 128, T=128, 1 layer, 2 heads,
d 128, cosine, lr 1e-3, no landmarks. Eval = paper2x2's: `eval_noise_refine`, noise 0, 100 held-out-map trials (env
seed 10000) at T=128 (REGISTERED, matched length) and 512 / 1024 (extrapolation, no verdict). Floors (paper torus,
`docs/WHAT_WHERE_CHECKS.md`): best n-gram 0.598, always-blank 0.507.

## Primary readouts (rank 2, T=128)
Exact permutation on per-seed accuracy and Fisher on SOLVED (`classify_run`: final-5% loss < 0.05). A contrast FIRES
if (perm p < 0.05 AND |d| >= 0.01) -- direction from d; a Fisher-only firing is reported as "SOLVED rate
(convergence)" and does not count for a branch (the TW_NORMSTEP audit's rule).
- TOTAL = MapPoPE-Flat - MapWM (replication of paper2x2's +0.028 at fresh seeds).
- SCORE = MapPoPE-Pair - MapWM (score rule at MapWM's frequency count).
- COUNT = MapPoPE-Flat - MapPoPE-Pair (frequency count under PoPE's score).
Branches:
- **NO EFFECT TO DECOMPOSE** -- TOTAL does not fire positive.
- **SCORE RULE** -- SCORE fires positive, COUNT does not.
- **FREQUENCY COUNT** -- COUNT fires positive, SCORE does not.
- **BOTH** -- both fire positive.
- otherwise reported as it falls (e.g. TOTAL fires but neither part does: the split is unmeasured).
Headroom caveat: MapWM r2 sat at 0.971 and MapPoPE r2 at 0.999 in paper2x2, so the decomposable effect is ~0.03;
each part may be below the MDE, which the analysis prints.

## Secondaries (no verdict)
Rank-4 versions of SCORE and COUNT (ceiling expected, ~1.000); the rank-4 upgrade within each family at T=128 and at
T=1024 (OOD, rule 10: the MAPPOPE_R4 question -- if MapPoPE-Pair's upgrade at T=1024 looks like MapWM's (+0.085) and
not MapPoPE-Flat's (+0.019), the frequency count explains MapPoPE's small rank gain); SCORE and COUNT at T=512 / 1024;
r(final loss, acc) over 48 runs.

## Void
Any of 48 checkpoints missing; md5 guard trips (checked at launch and again before eval); the pilot's reproduction
check failing (below).

## What it can and cannot show
- It separates the two changes on ONE task (paper torus, T=128, 1 layer) where the total effect is small and near
  ceiling. It does not test Bach length extrapolation (the other suspect list), a relative-offset task, or depth.
- MapPoPE-Pair changes only the angle count relative to MapPoPE-Flat (exact equivalence above), but relative to MapWM
  it changes the whole score rule (non-negative magnitudes, no content phase, per-element delta): SCORE is that bundle.

## Amendment 1 (2026-10-04, after an independent code audit, BEFORE the batch was launched)
The audit (read-only, blind to the pilot) found no bug: the model is MapFormerWM.forward with only the layer type and
the angle duplication changed; the equivalence script fails on three positive controls (untied omega, misaligned
pairing, zeroed step map) and passes on the real model; train_variant / train.py never branch on the variant name (one
AdamW group, pope_delta included, no special init); eval rebuilds from the mutated VARIANT_MAP with strict loads; the
recipe and eval flags equal paper2x2's (data-workers default 0 then and now). Changes, all made before launch (the
analysis is md5-guarded, so nothing can change after):
1. **Power.** From paper2x2's seeds (`docs/audits/2026-10-04/mappope_pair_power.py` / `_out.txt`): P(an effect of
   paper2x2's size fires) 0.59 at n=8, 0.88 at n=12, **0.95 at n=16**; for a 50/50 split each half fires with 0.13 /
   0.14 / 0.21. The three rank-2 arms now run **16 seeds (10-25)**; the rank-4 arms stay at 8 (10-17; ceiling
   expected). 72 runs. A 50/50 split will most likely read SPLIT UNMEASURED.
2. **Branches made exhaustive.** "Does not fire" now means "does not fire positive"; SCORE RULE / FREQUENCY COUNT
   report when the other part fires NEGATIVE ("and the extra angles HURT" / "and PoPE's score alone HURTS"); new
   SPLIT UNMEASURED (TOTAL fires, neither part does); NO EFFECT DETECTED replaces "did not replicate" (states power).
3. **Ceiling.** A contrast where both arms are >= 0.999 on every seed reads CEILING (undetermined, rule 5), not
   "unmeasured"; if TOTAL is at ceiling the verdict is CEILING.
4. The rank-4 secondary's reference is the within-batch MapWM upgrade (paper2x2's own was +0.151 at T=1024), not
   RANK_SWEEP's +0.085.
5. The MDE uses the two-sample df (n_x + n_y - 2); docstring fixed (Fisher-only never counts); seeds asserted per key;
   T=128 revisit NLL added as a secondary (continuous, less ceiling-bound).
Noted, no change: SCORE is a bundle (non-negative magnitudes; with untied per-element deltas a pair gives a limited
content-dependent phase; 64 learned deltas; positive score mean); COUNT is exactly "untie the 32 angle pairs" (+64
omega, +64 W_out rows at r2). The dropout-scale issue cannot manufacture these contrasts on this task
(DROPOUT_RESCORE: paper2x2 path arms move <= 0.0007).

## Pilot (`runs/mappope_pair_pilot`, seeds 100 and 0; outside the batch's seeds)
Read before Amendment 1 was written. (a) Reproduction (pass criterion: bitwise-equal per-epoch losses against the
stored paper2x2 run on the same GPU type): `train_pair` with Vanilla s0 and MapPoPE-Flat s0 reproduces
`runs/paper2x2/p0` exactly, 300/300 epochs each -- the wrapper changes nothing on the existing path. (b) MapPoPE-Pair
s100: SOLVED (tail 0.0001), T=128 1.000, T=512 0.971, T=1024 0.925; MapPoPE-Pair_r4 s100: SOLVED, 1.000 / 0.989 / 0.949.
(c) ~1.6 s/epoch at 4 concurrent, ~17 min per run.

## Cost (measured in the pilot)
~1.6 s/epoch at 4 concurrent (~17 min per run). Batch: 72 runs at 8 concurrent, ~25-30 min per run -> ~3.5-4.5 h;
eval and analysis minutes.
