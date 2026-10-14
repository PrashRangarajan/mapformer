# Generality of the sign / learnability result: the torus existence test and published monotone methods -- pre-registration (2026-10-14, before any batch run)

## Questions
**A (TORUS_EXIST).** `WRAP_EXISTENCE_RESULTS.md` met rule 14 on the 1D ring: an exact non-negative rotary solution is held under
training (8/8) but never found from scratch (0/8 vs signed 8/8). Does the same hold on the 2D torus at T = 1024, the regime of
the headline sign result (`SIGN_MATCHED_RESULTS.md`: Abs - Signed -0.177, 0/8 vs 8/8)?
**B (MONO_BASELINES).** Our non-negative arms (|.|, softplus) are MapFormer's parameterisation with the sign removed. Do the
PUBLISHED monotone content-dependent methods -- CARoPE (per-head rate in (0, 1), channel i steps f^i) and CoPE (sigmoid-gated
counts, interpolated position logits) -- also fail where a map is needed? Both are representationally unable to cancel
(CARoPE: two move angles sum to at most 2 < 2 pi; CoPE: a count only grows), so B tests whether published methods in
Howard's Case II fail on map tasks in practice, not learnability.

## Models
- `SignedAO4_r4` / `AbsAO4_r4` (`model_wrap_exist.py`): Signed_r4 / Abs_r4 on the torus with every non-move token's step 0 by
  construction (the four moves free). Constructions by `build_torus_exist.py` (`wrap_construct`, Amendment 1 of
  WRAP_EXISTENCE): **AO_bal** (balanced: orthogonal move latents, omega' = max(|omega|, 1), steps in [0, 2 pi)) and its
  matched signed control **SO_bal** (same gauge, unwrapped signed angles). The min-norm construction is not used on the torus
  (pilot: |W_out| 186,000; the torus source's four move latents have condition number ~6000).
- `CARoPEF` and `CoPEF` (`model_mono_baselines.py`; faithful; verified: CARoPEF at init = index RoPE base 10^4 to 2e-6 rad;
  the CoPE module = a direct loop of the paper's formulas to 8e-7). CARoPEF is MapWM's layer with CARoPE's phase (204,183
  params); CoPEF is a pre-LN block with Listing 1's attention, npos_max 2048 (334,483 params on the ring vocabulary: the
  position table adds 131k). Deviations disclosed in the module docstring (CARoPE bias to start at RoPE).
- Reference arm: `Signed_r4` (MapWM, shared rank 4, observation steps free), retrained in this batch (rule 12).

## Arms, recipes, seeds
Torus (environment.GridWorld 64x64, T = 1024, batch 16, 98 batches, lr 1e-3, cosine 5% warm-up, 900 epochs, 1 layer, 2 heads,
d 128, --n-landmarks 0, --data-workers 3, --fast-attn except CoPEF, which has no rotary layer; seeds 700-707):
| part | arm | what |
|---|---|---|
| A | SO4 | SignedAO4_r4 from scratch |
| A | AO4 | AbsAO4_r4 from scratch |
| A | SO_bal / AO_bal | built from SO4's checkpoint; 300 epochs at **lr 1e-4** (cosine, --data-seed-offset 10^6): PRIMARY |
| A | SO_bal_hi / AO_bal_hi | the same at lr 1e-3: DESCRIPTIVE |
| B | Signed_r4, CARoPEF | from scratch (CoPEF on the ring only: see Process and cost) |
Ring (environment_cancel, 32 cells, p_plus 0.5, RING_DIAL recipe: T = 128, batch 128, lr 1e-3, cosine, 300 epochs; seeds
700-707): B: Signed_r4, CARoPEF, CoPEF (`train_ring_variant.py`).
Why lr 1e-4 is the primary continuation (pilot, seed 690): re-warming to lr 1e-3 lifts the SIGNED control itself from loss
~0 to ~0.1 on the torus (its source only went below 0.05 late, at low lr), so at 1e-3 the continuation cannot separate the
arms; 1e-4 is about the learning rate at which the source solution was reached.

Evaluation. Torus: `eval_noise_refine` (held-out map env seed 10000, T = 1024 primary, 2048 secondary, 100 trials, noise 0) and
`eval_rank_strata` (T = 1024); frozen constructions: `build_torus_exist.py` (held-out map, T = 1024, 50 trials, the same
trajectories for source and construction). Ring: `train_cancel.evaluate` (fresh stream, 400 trajectories) as RING_DIAL.
SOLVED = `stats_core.classify_run` (final 5% of the arm's own training losses < 0.05).

## Registered decisions (`analyze_generality.py`)
**A** (as WRAP_EXISTENCE + Amendment 1):
- **P0:** SO4 SOLVED count n_s; E1 / E2 are read on the solved sources; E2 UNREADABLE if n_s < 4.
- **E1 per construction: EXISTS** if frozen accuracy is within 0.005 of the source's on every solved source and the max
  move-angle error < 1e-2 rad (0.005, not 0.002: 50 torus trials vs 400 ring trials).
- **E2 (primary: AO_bal vs SO_bal at lr 1e-4), gated on E1.** A run is LOST if its max per-epoch training loss in the
  continuation is > 0.3, or > 10x its control's max at that seed AND > 0.05. HELD = AO_bal SOLVED on >= n_s - 1 AND mean
  accuracy (T = 1024) >= SO_bal - 0.01 AND LOST on <= 1 seed. NOT HELD = Fisher (SOLVED) p < .05 with SO_bal ahead, OR
  sign-flip p < .05 with d <= -0.05, OR Fisher (not LOST) p < .05 with more LOST in AO_bal. PARTIAL otherwise. The lr 1e-3
  pair gets the same labels, DESCRIPTIVE.
- **E3 (scratch): AO4 vs SO4**, gated on SO4 SOLVED >= 4: SIGNED LEARNS BETTER (Fisher p < .05 with SO4 ahead, or perm p
  < .05 with d <= -0.05) / NO SIGN COST (AO4 SOLVED >= SO4 - 1 and mean >= SO4 - 0.01) / UNMEASURED.
**B** (each monotone arm vs Signed_r4: CARoPEF on ring and torus, CoPEF on the ring):
- **MONOTONE FAILS** if Fisher (SOLVED) p < .05 with Signed_r4 ahead, OR perm p < .05 with d <= -0.05; **NO DIFFERENCE**
  if the monotone arm's SOLVED >= Signed_r4's - 1 and mean accuracy >= Signed_r4 - 0.01; UNMEASURED otherwise. Gated on
  Signed_r4 SOLVED >= 4 on that task. Three separate questions; no multiplicity correction.
Prediction, stated now: MONOTONE FAILS for both methods on both tasks (they cannot represent cancellation).

Secondaries (declared, no verdict): wrap share (`wrap_share`, wrap-invariant selection, pairs (0,1),(2,3) on the torus) for
SO4 / AO4 / SO_bal / AO_bal at construction and after training; revisit strata (plain lag < 128, >= 128, wrap-only) for every
torus arm; accuracy at 2048 (OOD); paired loss AUC AO4 - SO4; cross-batch and labelled: AO4 vs SIGN_MATCHED's Abs_r4 (with
observation steps, no fast-attn); CARoPEF vs FACTORIAL's RoPE 1-layer 0.719 on the torus (cross-batch, labelled).

## Process and cost
Pilots (seed 690, outside the batch; reported, never pooled): torus SO4 solved (loss 0.0002), AO4 stalled (0.46); the torus
constructions are exact (frozen 0.9996 = source); continuation at lr 1e-3: SO_bal ~0.05-0.1, AO_bal ~0.3-0.5; at lr 1e-4:
SO_bal ~0.001-0.003, AO_bal ~0.03-0.09 (first 50 epochs); ring CARoPEF / CoPEF (below). **CoPEF is not run on the torus:** its gated count materialises (B, H, L, L) tensors at
L = 2048 tokens; the probe measured 31 s/epoch and 8.2 GiB (~8 h per 900-epoch run, ~15 h for 8 seeds), so CoPE is tested on
the ring only (5.4 s/epoch).
Independent code check (rule 29) before launch; its findings become Amendment 1.
Cost (both GPUs, ~8 jobs at a time): torus 32 x 900 epochs + 32 x 300 epochs at ~2.4-3.5 s/epoch under load; ring 24 x 300
epochs (CoPEF ~27 min each): ~4-5 h.
