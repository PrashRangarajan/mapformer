# Testing the accounts behind the sign result -- pre-registration (2026-10-14, before any run)

The theory review (`docs/audits/2026-10-14/theory_review/THEORY_REVIEW.md`) and the CoPE diagnosis
(`docs/audits/2026-10-14/cope_diagnosis/COPE_DIAGNOSIS.md`) left four accounts stated but untested. Each test below is
registered with its prediction; T3 is a CPU toy (exploratory, predictions still stated now).

## T1 -- does weight decay cause the drift off the wrapped code?
Account (Prop. 2 of the review): rescaling a channel leaves the signed cancelling code a(-1) = -a(+1) unchanged and moves the
wrapped one a(+1) + a(-1) = 2 pi; AdamW's decoupled weight decay rescales every step. WRAP_EXISTENCE: the held non-negative
ring model drifted 87% -> 18% of channels cancelling; toy: weight decay 0 keeps 0.99 vs 0.43.
Design: the WRAP_EXISTENCE sources (signed observation-free ring models `runs/wrap_existence/p0/SO_s600-607/SO.pt`), the
balanced constructions, continued 300 epochs exactly as WRAP_EXISTENCE (lr 1e-3 cosine re-warmed, data offset 10^6,
`train_wrap_exist.py`) with weight decay 0.05 (as before) or 0. Arms AO_bal_wd, AO_bal_wd0, SO_bal_wd, SO_bal_wd0; seeds
600-607 (the sources' seeds; every arm retrained in this batch).
Readout: `wrap_share` at the end (wrap-invariant selection; share of channels with |wrap(a(+1) + a(-1))| < 0.1 rad), paired by
seed (same source), exact sign-flip test.
- **D1 (primary): AO_bal_wd0 - AO_bal_wd.** WEIGHT DECAY DRIVES THE DRIFT: mean difference >= +0.30 and sign-flip p < .05.
  NO EFFECT DETECTED: |mean difference| < 0.10. UNMEASURED otherwise. Prediction: WEIGHT DECAY DRIVES THE DRIFT.
- **D2 (control): SO_bal_wd0 - SO_bal_wd**, same labels. Prediction: NO EFFECT DETECTED (the signed code is rescaling-invariant).
Secondaries: accuracy at T and 4T, SOLVED, the LOST rule of WRAP_EXISTENCE, |W_out| and |omega| at the end.

## T2 -- is the signed escape from the |.| start collective (one latent direction)?
Account (review 3.8): for signed steps the cancelling set {z(-1) = -z(+1)} is a linear subspace of the move latents, so one
latent direction cancels every channel at once; per-channel gradient flow crosses zero in only ~12% of channels. MECHANISM:
signed from the |.| start escapes 7/8.
Design: `train_wrap_mech.py`, ring, observation-step-free, 300 epochs; arms SO_same, SO_absmag (as MECHANISM) and SO_same_fz,
SO_absmag_fz: the same starts with the two move latents FROZEN (W_in fixed, the move embeddings detached in the step path; they
still train through the content path), so only W_out and omega move the angles, channel by channel. Seeds 1000-1007.
- **F1: SO_absmag_fz vs SO_absmag** (Fisher on SOLVED, permutation on accuracy; labels BETTER / WORSE / NO EFFECT DETECTED
  / UNMEASURED as MECHANISM Amendment 1).
- **F2 (control): SO_same_fz vs SO_same.**
- Reading, fixed now: the collective-route account is SUPPORTED if F1 = WORSE and F2 != WORSE; NOT SUPPORTED if F1 = NO EFFECT
  DETECTED (the per-channel route suffices); AMBIGUOUS if both are WORSE (freezing hurts generally) or F1 UNMEASURED.
- Gate: UNREADABLE if SO_same solves < 4/8. Prediction: SUPPORTED.
Secondary: share of channels with opposite-sign move angles at the end (the through-zero escape), per arm.

## T3 (CPU toy, exploratory) -- why did the toy not escape the |.| start?
Candidate (review): the transformer is rewarded for retrieving an OBSERVATION, so keys at other cells with the same
observation also pay, smoothing the plateau along the escape route; the toy rewarded only same-cell keys.
Design: `toy_wrap.py --target obs` (an observation map per trajectory as environment_cancel: 16 types, p_empty 0.5; correct
keys = earlier moves whose cell shows the query cell's observation; queries are still the revisits), OMP_NUM_THREADS = 2 (the
toy is bit-deterministic only at a fixed thread count), 3000 steps, seeds 10-17; starts signed absmag, |.| absmag, signed
default, |.| default. Readout: same-cell retrieval accuracy (acc_cell, the map) >= 0.99 = MAP SOLVED, and the share of
channels with opposite-sign move angles at the end.
Prediction: signed absmag escapes on >= 5/8 seeds (vs 0/8 with the cell target), |.| absmag on <= 1/8. If not, the
observation target is not the difference.

## T4 -- does CoPE learn on a line that never wraps?
Account (CoPE diagnosis): CoPE's count gets opposite-move cancellation at init; the obstacle is the 32-cell WRAP, which
forces the move-gate difference onto a fine lattice (odd multiples of P/32, ~1e-3) with barriers between lattice points. On
a line that never wraps no lattice is needed: any small enough gate difference reads net displacement.
Design: `train_ring_variant.py` with `--size 256` (T = 128 moves: |displacement| <= 128 < 256, so the walk never wraps;
same walk statistics, same observation process), RING_DIAL recipe, 300 epochs; arms CoPEF (line), Signed_r4 (line), RoPE
(line), and CoPEF on the 32-cell ring (in-batch); seeds 1100-1107.
- **L1 (primary): CoPEF line vs CoPEF ring, SOLVED** (Fisher; final 5% training loss < 0.05 on both tasks): LINE EASIER if
  Fisher p < .05 with the line ahead; NO EFFECT DETECTED if SOLVED within 1; UNMEASURED otherwise. Prediction: LINE EASIER.
- **L2: CoPEF line vs Signed_r4 line** (labels as MECHANISM's M4: SIGNED LEARNS BETTER / NO SIGN COST / UNMEASURED, gated on
  Signed_r4 line SOLVED >= 4). Prediction: NO SIGN COST.
- Secondary: RoPE on the line (does the line need a map at all: index fails?); the line's blank-target share as a floor;
  the CoPE gate regime per run (diagnosis scripts, post hoc).

## Process and cost
Independent code check (rule 29) before launch; amendment before any result is read. GPU: T1 32, T2 32, T4 32 runs (~10-30
min each, 10-12 concurrent on both GPUs): ~3 h. T3: 32 CPU runs, ~1 h alongside.

## Amendment 1 (2026-10-14, after the independent check `docs/audits/2026-10-14/accounts_audit.md`; before launch)
The check found the defaults bit-identical to HEAD~1 (4 regression runs + the toy's default target) and ran all 96 arm/seed
combinations tiny on CPU end to end. Changes:
- **T1 readout (MAJOR):** the unweighted wrap share counts every channel equally, and AO_bal ends at accuracy 1.000 with only
  15-22% of channels cancelling: most of the drift is in channels the model does not use, which weight decay moves by
  construction. **D1 / D2 now use a usage-weighted wrap share** (`model_wrap_exist.usage_weights`, eval-only: per channel,
  mean |q_c| at move-token queries x mean |k_c| at observation-token keys, layer 1, 32 eval trajectories), at construction
  and at the end; same thresholds. The unweighted D1u / D2u are reported as secondaries.
- **T2 freeze (MAJOR):** detaching the move embeddings left them trainable through the content path and decayed, so the
  latents still moved (2.6% in 600 CPU steps). Now `freeze_move_latents()` stores a buffer copy of the move embeddings for the
  step path and W_in gets no grad: the latents are bit-identical after training (checked), while the embeddings still train
  through the content path.
- **T4 power (MAJOR):** with ring CoPE at ~1/8 SOLVED and an expected 4-5/8 on the line, 8 seeds give Fisher p 0.08-0.28.
  CoPEF line and CoPEF ring now run **16 seeds each (1100-1115)**; Signed_r4 and RoPE on the line stay at 8 (1100-1107).
  L1 adds a RING EASIER branch and NO EFFECT DETECTED = SOLVED within 2 of 16; L2 adds COPE BETTER and compares SOLVED
  fractions (16 vs 8).
- **Minor:** T3 adds the cell-target comparators (signed / |.| absmag) at the same seeds and code, so the toy contrast is
  within one run set; the driver's clean check runs before the md5 guard. Disclosed: the check's freeze probe used seeds
  1000-1001 with a different (tiny) recipe; it wrote nothing into the batch directories.
- Artifacts: T1 64 + T2 64 + T4 96 = 224.
