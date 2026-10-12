# CLOCK_SIDE independent code audit (rule 29), 2026-10-12

Scope: `CLOCK_SIDE_PREREG.md` (through Amendment 1), `run_clock_side.sh` + `lib_driver.sh`, `analyze_clock_side.py`,
`train_recency.py`, `environment_recency.py`, `model_sign.py`, `model_baseline_rope.py`, `train_variant.VARIANT_MAP`,
`stats_core.py`, `validate_recency.py` / `CLOCK_SIDE_GATES.md`. Compared with `run_recency.sh`, `RECENCY_PREREG.md`,
`RECENCY_RESULTS.md`, `SIGN_MATCHED_RESULTS.md`. Blind: no `runs/clock_side_calib/*` log or output read, and the
calibration accuracy table was not used. Nothing in the repo was modified except this file; no GPU job was launched.
CPU test scripts and outputs: `/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/audit_cs/`
(`t1_init.py` .. `t5_gate.py`, a tiny CPU run under `tiny/`, a dry-run copy of the driver under `drv/`).

## Verified correct (with evidence)

- **Arms are built as named.** `VARIANT_MAP` maps `Signed_r4 / CARoPE_r4 / Pos_r4` to `model_sign._sign_variant("signed"|"carope"|"pos", 4)`
  (train_variant.py:215-222) and `RoPE` to `MapFormerWM_RoPE` (train_variant.py:317). Built on CPU: classes
  `MapFormerWM_signed_r4 / _carope_r4 / _pos_r4`, 238,681 params each; RoPE 237,849. Step functions: model_sign.py:306-314.
- **Identical initial weights across the three path arms at a seed**: every state-dict tensor of CARoPE_r4, Pos_r4 (and Abs_r4)
  equals Signed_r4's at `torch.manual_seed(200)` (`t1_init.py`). RoPE has a different init (7 of 21 shared keys equal), as expected.
- **Shared data and eval streams.** Training draws from `np.random.RandomState(seed)` only (train_recency.py:184,191), independent of
  the model; eval from `RandomState(5000 + seed)` (train_recency.py:205-206) via `generate_batch`, so every arm at a seed sees the same
  training episodes and the same eval items (tiny CPU run: CARoPE and RoPE at s200 both score n = 213 at T 1024, 464 at 2048).
  Eval streams (5200-5207) do not overlap training streams (200-207) or the calibration seed (100).
- **min_gap = k_max at T and 2T**: `min_gap=None` -> `k_max` (environment_recency.py:111-112), same `kw` for `env_test`
  (train_recency.py:151-156); log header prints `k_max=128 min_gap=128`.
- **--fast-attn is valid and equivalent for all four arms** (all use `model.WMTransformerLayer`, no MapEM): CPU eval-mode logits
  SDPA vs manual max |diff| 8.3e-07 (Signed / CARoPE / Pos), 9.5e-07 (RoPE); dropout-free training gradient cosine 1.000001
  (`t2_sdpa.py`). Same flag in RECENCY and the calibration.
- **Driver** (dry run of a scratch copy with `DRV_DRYRUN=1`, slot cap lifted): 32 unique (variant, seed) launches, seeds 200-207,
  all `--device cuda:0`; flags identical to `docs/audits/2026-10-12/clock_side_calib.sh` and to `run_recency.sh` except
  `--k-max` (64 -> 128); `flock -n` lock, per-run "running" and "done" guards, md5 guard at start and again before the analysis,
  `drv_require` on all 64 artifacts, done marker only after `CLOCK_SIDE_ANALYSIS.txt` and `CLOCK_SIDE_VERDICTS.json` exist;
  missing runs fail closed ("done marker NOT set"). `DRV_*` knobs are assigned after sourcing (rule 27).
- **Analysis arithmetic**: `perm2_p(signed, mono)` / `perm2_ci(signed, mono)` test mean(b) - mean(a) = mean(mono) - mean(Signed),
  matching `d` (analyze_clock_side.py:55); D2 likewise RoPE - Signed. Branch order and thresholds match the prereg. `--smoke`
  ALL PASS; an end-to-end run on 32 fabricated run dirs (`t3_analysis.py`) reads the files, asserts and prints as intended.

## BUG

**B1. Registered secondaries are not implemented; they would be written after the results are read.** The prereg (line 45-46)
registers: dropout-scale re-score of the primary, r(final loss, acc), alpha per arm. `analyze_clock_side.py` computes only SOLVED +
Fisher and mean accuracy at 2T (lines 105-113). It also prints no floor (gate best trivial 0.082 / most-recent 0.0709), no
DESCENDING / STALLED counts, and no eval n. Fix: add them before launch (the re-score is a re-evaluation of the saved
checkpoints under `rescore_hook.install("auto")` with the trainer's own `evaluate`; the hook covers `model.WMTransformerLayer`,
which all four arms use, and hooks `o_proj`'s input so it works on the SDPA path), or mark them post hoc in Amendment 2.

**B2. Provenance asserts miss the training length and min_gap.** The `.pt` does not save `T`, `lr`, `schedule`, `n_batches`,
`batch_size`, `fast_attn` (train_recency.py:213-220); the analysis asserts `variant`, `seed`, `k_max`, 300 epochs and the eval keys
`{1024, 2048}` (lines 101-102). A run trained at another T with `--eval-T 1024 2048` passes. Fix: assert `b["min_gap"] == K_MAX`,
`n_layers == 1`, `d_model == 128`, `n_heads == 2`, and parse the run log header (`$R/${V}_s${S}.log`, line 2:
`... T=1024 k_max=128 min_gap=128 ...`, line 1: `[fast-attn] SDPA + TF32 enabled`) and assert T = 1024 and fast-attn.

## DESIGN major

**M1. The NO DIFFERENCE branch is practically unreachable, and power for CASE II is low.** Eval is hard-coded to 8 x 8 episodes
(train_recency.py:205): ~213 scored targets per run at T 1024 (vs 456 in RECENCY at k_max 64), binomial SE ~0.034 per seed at
acc 0.5. Simulation with the registered `d1()` (`t4_power.py`, 200 reps each; true acc 0.6):

| eval targets | seed sd | true d 0 | true d +0.05 | true d +0.10 |
|---|---|---|---|---|
| 205 | 0.00 | NO DIFF 0.09, UNMEAS 0.85 | CASE 0.90 | CASE 1.00 |
| 205 | 0.03 | NO DIFF 0.01, UNMEAS 0.92 | CASE 0.68 | CASE 1.00 |
| 205 | 0.08 | NO DIFF 0.00, UNMEAS 0.93 | CASE 0.19 | CASE 0.69 |
| 1640 | 0.00 | NO DIFF 1.00 | CASE 1.00 | CASE 1.00 |
| 1640 | 0.03 | NO DIFF 0.15, UNMEAS 0.80 | CASE 0.91 | CASE 1.00 |

So "signed matches the clock", one of the question's two poles, cannot be concluded from this design (a true null reads
UNMEASURED ~90% of the time), and below ceiling seed sd will likely exceed 0.03 (RECENCY's Pos / Abs were 0.041 / 0.068 near ceiling).
Fix (cheap, eval-only): register a re-evaluation of every saved checkpoint on >= 64 x 8 episodes (~1700 targets; seeds 5000+s as now)
as the primary readout; also register a paired sign-flip test as a secondary (M2's pairing is real, see m1); state the MDE; and
either widen the equivalence margin (e.g. +/-0.05) or accept that only CASE II / SIGNED BETTER / UNMEASURED are reachable.
n = 16 seeds would be the only fix for seed-sd-dominated power.

**M2. A CASE II ADVANTAGE would not be attributable to sign / monotonicity; the sign-only arm (Abs) was dropped.** `model_sign.py`
lines 257-275 states that Pos and CARoPE "carry an INIT CONFOUND that Abs does not ... That is precisely why Abs is the primary."
This batch has only those two. Measured at init (`t1_init.py`, s200, per-channel step by token class):

| arm | symbol step | filler step | dStep/dz at z=0 |
|---|---|---|---|
| Signed_r4 | 0.000 +/- 0.429 | -0.011 +/- 0.356 | 1 |
| CARoPE_r4 | 0.592 +/- 0.072 | 0.593 +/- 0.060 | -0.18 |
| Pos_r4 | 0.715 +/- 0.216 | 0.703 +/- 0.180 | 0.5 |
| Abs_r4 | 0.333 +/- 0.271 | 0.265 +/- 0.237 | +/-1 |

The monotone arms start as a uniform token clock (Case I with a rescaled omega); Signed starts as a random walk; under Adam the step
moves ~5.7x (CARoPE) / 2x (Pos) slower per update in step space (rule 15). The counter solution (step c on symbols, 0 on fillers,
queries, MASK) is at zero weights for Signed, but needs z of order +40 on fillers for CARoPE (1/(softplus(z)+1) ~ 0.024 c) and
z ~ -4 for Pos, against weight decay 0.05: the two "Case II" arms have different solution geometry, and averaging them can mask a
split. Also, the map-side matched-length result the "double dissociation" pairs with (`SIGN_MATCHED_RESULTS.md`) measured Abs
(primary) and Pos only; CARoPE has no matched-length map-side number, so the composite is not the same contrast on both sides.
Fix: add `Abs_r4` (8 runs, +25% cost; same init as the others at each seed, verified) and register Abs - Signed as the sign-only
contrast; reserve the "double dissociation" label for a positive Abs - Signed (or Pos - Signed, the one arm shared with the map
side); report CARoPE - Signed and Pos - Signed per arm with Holm. If Abs is not added, the prereg must say a CASE II ADVANTAGE is
"the published monotone parameterisations beat MapFormer's" and not "monotone beats signed".

**M3. CASE II ADVANTAGE can only be a search / speed effect; the budget is not controlled.** `RECENCY_PREREG.md` (Amendment
2026-09-07, "H1 is MALFORMED") established that Signed_r4 contains the monotone solutions, so "monotone beats signed" holds only up to
optimisation; CLOCK_SIDE re-asks that H1 without citing this. At a fixed 300 epochs below ceiling, a CASE II reading with Signed runs
still descending is a training-speed difference (rule 3/4). Fix: (a) state in the prereg that D1 is an optimisation /
inductive-bias test, not a capacity one; (b) print `classify_run` classes per arm and register a qualifier: if >= 3/8 runs of the
lower arm are DESCENDING, the verdict reads "at 300 epochs (speed not excluded)" and a 2x-budget rerun is pre-committed; (c) if CASE
II fires, the rule-14 check: construct the counter in Signed_r4 and warm-start it frozen and trainable.

**M4. The eval-mode dropout under-report bites exactly in this regime.** `DROPOUT_RESCORE.md`: one-layer models below ceiling are
the ones that move (text world up to +0.17 per seed); whether the shift differs by arm here is unknown. Fix: compute the re-scored
D1 in the registered analysis (B1) and register that a D1 label stands only if eval-mode and re-scored D1 land in the same branch,
else "SCORING-DEPENDENT". Keep eval mode as the named primary for comparability with RECENCY.

## DESIGN minor

- **m1. The registered unpaired test discards a real pairing.** Signed / CARoPE / Pos share initial weights, training stream and
  eval items at each seed (verified above). Add the paired sign-flip p (`stats_core.signflip_p`, min p 2/256) on per-seed
  differences as a secondary; keep the unpaired permutation as registered.
- **m2. The gate's flagged n-gram orders are noise plus a fallback leak.** `validate_recency.ngram_acc` picks the fallback symbol
  from the WHOLE answer stream including the scored half (`cnt = Counter(answers)`, validate_recency.py:49-51), and orders 3 / 5
  mostly fall back (unseen contexts). Answers are IID by construction at min_gap = k_max. Rerun (`t5_gate.py`): three fresh 800-episode
  streams give o3 0.071-0.082, o5 0.069-0.076; 8000 episodes (26,232 answers) give o1/o2/o3/o5 = 0.0607 / 0.0602 / 0.0612 / 0.0625.
  The gate passes in substance; the 0.082 "floor" overstates it (harmless, conservative). Fix: cite the 8000-episode rerun in
  Amendment 2, report the floor as most-recent 0.0709 (chance 0.0625), and fit the fallback on the first half only (separate change,
  not during a batch).
- **m3. TF32 under --fast-attn is shared but not neutral in magnitude.** The steps come from TF32 matmuls; the monotone arms carry a
  cumulative phase ~0.6 x 1023 at T 1024 vs a random walk for Signed, so the absolute rounding in theta is larger for them. It is
  per-token-type and deterministic, so train and eval see the same function; same setting as RECENCY and the calibration. Not
  checkable on CPU. Also: no `--deterministic` in `train_recency.py` and SDPA dropout RNG, so runs are not bit-reproducible. State both
  in the prereg.
- **m4. Per-offset readouts are meaningless at this eval size**: ~1.7 targets per k (213 over k = 1..128). If shown, bin k (e.g. 8
  bins of 16), or use the M1 re-evaluation.
- **m5. GPU budget is shared.** `drv_ntrain` counts every `mapformer.train_*` of this user on cuda:0; at audit time 4
  `train_ring_dial` pilot jobs occupied cuda:0, so at `DRV_MAXPG=4` the batch would wait for them (it did, in the first dry run). Not a
  correctness issue; the 1.5-2.5 h estimate assumes four free slots.
- **m6. NO DIFFERENCE also fires when p < .05 with 0 < |d| < 0.02** (a detected small difference inside the margin). Acceptable as
  equivalence, but label it "within +/-0.03 (a small difference detected)" when p < .05.

## Prereg should state but does not

- That RECENCY's registered composite was the mean of THREE arms (Abs, Pos, CARoPE: (-0.052 + 0.007 + 0.032) / 3 = -0.004,
  `RECENCY_RESULTS.md` sec. 2); line 38-39 calls the two-arm mean "RECENCY's registered composite", which it is not.
- The init / parameterisation confound of Pos and CARoPE (M2), and that D1 tests search, not capacity (M3, citing RECENCY's H1 amendment).
- The MDE / reachable branches at ~213 eval targets (M1).
- What happens if the batch lands at ceiling (the setting was chosen on one seed) or on the floor for Signed.
- The budget qualifier and run classes (M3), the dropout-scale rule (M4), TF32 / non-determinism (m3).

## Cosmetic

- `CLOCK_SIDE_GATES.md` verdict cell prints "FAIL o1=0.0653" although o1 passes (the failing orders are 3 and 5), and its boilerplate
  "the failures are the low-min_gap diagnostic rows and are EXPECTED" is wrong for a min_gap = k_max row (validate_recency.py:221).
- Prereg line 61 says "~205" targets; measured 213 at seed 200 (it varies by seed; print n per run).
- `CLOCK_SIDE_GATES.json` is untracked; commit it with Amendment 2.
