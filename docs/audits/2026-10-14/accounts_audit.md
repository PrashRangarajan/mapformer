# ACCOUNTS -- independent code check (rule 29), 2026-10-14, before launch

Scope: `ACCOUNTS_PREREG.md`, `analyze_accounts.py`, `run_accounts.sh`, `runs/accounts/t3/{run_t3.sh,summarize_t3.py}`, and the
HEAD (f2fd71a) changes to `model_wrap_exist.py`, `train_wrap_exist.py`, `train_wrap_mech.py`, `toy_wrap.py`. CPU only. No repo
file edited except this one. Scratch: `/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/`.

No BLOCKER. Three MAJOR (T1 readout, T2 freeze leak, T4 power), the rest MINOR.

## What was run

- **Regression vs HEAD~1** (`git archive HEAD~1` into scratch; OMP 1, `--epochs 2 --n-batches 2 --data-workers 1 --n-trials 20`):
  `train_wrap_mech` SO and SO_absmag (seed 1000), `train_wrap_exist` AO_bal and SO_bal (seed 600, `--init` SO_s600). State
  dicts, losses and eval.json are **bit-identical** in all 4. `toy_wrap --target cell` (default) vs HEAD~1, 50 steps: identical
  history and angles.
- **Full smoke layout** (`acc_smoke/{t1,t2,t4}`, all 96 runs, every seed of every arm, tiny budget): all ran, no error;
  `analyze_accounts --root acc_smoke --no-json` runs end to end (all labels exercised: NO EFFECT / UNREADABLE paths; blank
  floor 0.503). Readouts at this budget are meaningless.
- **T3**: `toy_wrap --target obs --mode signed --init absmag --seed 10 --steps 50`: runs; at init obs-target acc 0.283 (=
  P(same observation) for the adjacent cell ~ 0.25 + 0.25/16), acc_cell 0.000, as expected.
- **Freeze check** (T2): `SO_absmag_fz`, seeds 1000/1001, 30 epochs x 20 batches on CPU (see 2 and 12).

## Findings

**1. MAJOR -- T1: the readout counts channels the model does not use, so D1 can fire by passive decay.**
`wrap_share` weights every selected channel equally. WRAP_EXISTENCE's AO_bal ends at accuracy 1.000 on 8/8 seeds with only
15-22% of channels cancelling (median |wrap(a+ + a-)| 1.79-2.24 rad, `runs/wrap_existence/p0/AO_bal_s*/eval.json`): most of
the drifted channels cannot be carrying the map. In a channel the attention ignores, the loss gradient on its step is ~0 and
AdamW decay alone shrinks omega, W_out (and z) by exp(-wd sum lr) ~ 0.45 each over this recipe (sum lr ~ 16), which takes a
wrapped pair off 2 pi; with wd 0 such a channel does not move. So "wd 0 keeps ~87%" is close to guaranteed whether or not
weight decay acts on the code the model uses -- D1 cannot fail in the interesting direction. (D2 is by construction too:
rescaling keeps a signed pair cancelling; fine as a control.)
Fix (eval-only, no GPU): register beside D1 a usage-weighted wrap share -- weight each channel (head h, block i) by its mean
attention-weighted |q_hi| |k_hi| over revisit queries on the eval stream (or by the eval-NLL rise when that channel's
rotation is zeroed) -- and state that the account is about USED channels. Also print the unweighted share among channels with
usage above the median.

**2. MAJOR -- T2: the move latents are not frozen; z = W_in e_act still moves.**
`W_in` is truly fixed (requires_grad False -> grad None -> torch AdamW skips it, no decay; verified: W_in bit-unchanged after
600 steps), and the step path is detached. But `e_act` keeps training through the content path AND is decayed by AdamW, so
z = W_in e_act changes. Measured after 600 CPU steps (~2% of the registered sum lr): |z| -2.6% (both latents), cos(z+, z-)
0.000 -> +0.043 / +0.051, per-channel angle change from the z drift up to 0.11-0.14 rad (mean 0.015-0.017). Over the full
budget weight decay alone would shrink z ~0.45x and the direction drift is unbounded. The route is closed to the STEP loss's
gradient (the account's sense), but the prereg says "move latents FROZEN" and the arm does not do that.
Fix (3 lines): at freeze time store `self.register_buffer("e_act_frozen", e[self.act].detach().clone())` and use it in
`step_table` when `freeze_latent`; content path unchanged. Re-run the regression (non-fz arms must stay bit-identical).
Add a secondary: cos(z+, z-) and |z| at the end for every T2 arm.

**3. MAJOR -- T4 L1 is underpowered for the effect the diagnosis predicts.**
Ring CoPEF solved 1/8 in WRAP_COPE; COPE_DIAGNOSIS E1 predicts ~4-5/8 on a non-wrapping line. Fisher p at n = 8: 4/0 0.077,
5/0 0.026, 4/1 0.28, 5/1 0.12, 6/1 0.041. The registered prediction "LINE EASIER" is reachable only at >= 6/8 vs 1/8 (or 5/8
vs 0/8); the diagnosis's own expectation lands UNMEASURED. Fix: n = 16 for the two CoPEF arms (+16 runs, ~1 GPU-h): 8/1 p
0.016, 9/2 0.023, 8/2 0.054; or register the prediction as ">= 6/8" so a 4-5/8 outcome is not read as support.

**4. MINOR -- T4 account: the line is not lattice-free for CoPE, it needs |kappa| < ~16/R.**
With sigma = 1 (period 2), p mod 2 = D (1 + kappa/16). On the line any |D| up to the walk's range R must not alias: odd D
aliases at D = 16/kappa. Line ranges (2000 walks, T 128): median 39, p90 59, max 107, so the line needs |kappa| <~ 0.3-0.4.
The ring OFF heads ended at median |kappa| 0.554 (`cope_diagnosis/m7_init_vs_final_out.txt`), aliasing at |D| ~ 29, inside the
range. So "OFF counts as solved on the line" (E1) is not automatic: the line still requires a move to small kappa, continuous
and barrier-free, but with sigma precision scaling as kappa/8 / N. Fix: say this in the amendment; declare a post hoc
per-head kappa vs 16/R readout (the lib_cope lattice classifier is ring-32 specific).

**5. MINOR -- T4 otherwise checks out.** `--size 256`, T 128: max |displacement| 128 < 256 and position revisits are exact
(no mod aliasing; start offset irrelevant). At 4T (512) no walk reached range 256 in 400. CoPEF ring and line have
bit-identical inits at a seed (vocab 19 both; CoPEF ignores grid_size; npos 2048 > 1023 tokens at 4T). Signed_r4 line differs
from a ring Signed_r4 only in omega (grid 256: 2 pi .. 2 pi/256) -- not a confound for L2 (both on the line). Same observation
process (p_empty 0.5, 16 types); revisits per trajectory 97 (ring) vs 86 (line); predict-blank floor 0.503 (line) vs ~0.506
(ring). The 0.05 SOLVED threshold is an absolute CE on revisit targets with optimum 0 on both tasks: comparable. Walk streams
are not paired across ring/line (the occupancy draw consumes 32 vs 256 numbers), irrelevant for an unpaired Fisher.

**6. MINOR -- labels deviating from the prereg text.**
- L2: `NO SIGN COST` is one-sided (kl >= ks - 1 and d >= -0.01), whereas the prereg says "labels as MECHANISM's M4", which is
  two-sided (|ka - kr| <= 1 and |d| <= 0.01) with a separate BETTER. A CoPE line clearly ahead of Signed_r4 would read NO SIGN
  COST here. Either state the one-sided rule in the amendment or use `mech_compare`.
- L1 has no label for the line significantly behind the ring (falls to UNMEASURED). Add LINE HARDER.
- T2 reading: SUPPORTED when F1 = WORSE and F2 = UNMEASURED (prereg-consistent: "F2 != WORSE"). An UNMEASURED F2 can hide a
  general freezing cost; consider requiring F2 = NO EFFECT DETECTED for SUPPORTED, or print F2's label next to the reading.

**7. MINOR -- T1 declared secondaries not implemented.** The prereg lists "the LOST rule of WRAP_EXISTENCE, |W_out| and
|omega| at the end"; `analyze_accounts.t1` prints max loss only. Add LOST (`analyze_wrap_existence.py` L25: max loss > 0.3, or
> 10x the paired control's and > 0.05) and the norms.

**8. MINOR -- T1 otherwise checks out.** `--weight-decay 0` -> one AdamW param group with wd 0: no decay on any parameter
(omega, W_out, W_in, embeddings, attention). Sources = WRAP_EXISTENCE's `p0/SO_s600-607/SO.pt`; `wrap_construct` balanced /
signed_balanced unchanged since WRAP_EXISTENCE apart from the ring-equivalent generalisation in 47a766b (verified bit-identical
to HEAD~1 here); data offset 10^6, lr 1e-3 cosine, 300 epochs as before. `eval.json` "wrap"[1] = end share and
"frozen_wrap"[1] = construction share, both from `wrap_share` (selection |wrap(a+)| > 0.05, cancel |wrap(a+ + a-)| < 0.1:
wrap-invariant). D1/D2 thresholds match the prereg; sign-flip min p 0.0078 at n 8, so p < .05 is reachable.

**9. MINOR -- T3 contrast is out of batch and not stream-paired.** `obs_of(pos, rng)` draws from the training rng, so from
batch 2 on the obs-target walks differ from the cell-target walks at the same seed (eval walks are identical: the eval map is
drawn after the eval walks). The "0/8 with the cell target" comparator is `runs/toy_wrap/swap/signed_absmag_s10-17` (3000
steps, same args), whose thread count is not recorded (rule 12). Fix: draw maps from a separate `RandomState(seed + 10**5)` and
add the four cell-target starts to `run_t3.sh` (32 more CPU runs). Implementation otherwise correct: per-trajectory map as
environment_cancel (occupied w.p. 1 - p_empty, type 0-15, blank = 16, blank == blank counts), correct keys = strictly earlier
moves whose cell shows the query cell's observation, queries = position revisits, acc_cell on the same eval walks.

**10. MINOR -- T3 operations.** `run_t3.sh` backgrounds jobs with `&` but its header does not say to launch it under
`setsid nohup` (rule 20); `.done` is touched after `wait` without checking the 32 JSONs; no md5 record of `toy_wrap.py`.

**11. MINOR -- driver.**
- The clean-tree check runs AFTER `drv_md5_guard`; a first launch from a dirty tree records the dirty md5 in
  `runs/accounts/code_md5.txt`, then refuses, and every later launch after committing ABORTs on md5 mismatch. Swap the order.
- `drv_wait_dir "$R/"` also matches the T3 toy processes (`--out .../runs/accounts/t3/...`): the GPU driver will wait for T3
  if it runs alongside. Harmless; note it or wait on `$R/t1/ $R/t2/ $R/t4/`.
- `DRV_MAXPG=6`: 12 concurrent x (OMP 2 + 3 data workers) ~ 60 threads on 32 cores; WRAP_EXISTENCE used 5. Wall time only.
- Checked OK: 192 artifacts (16 runs x 2 files per T1/T2/T4 block of 32 runs); `go()` passes module + args, then
  `--device cuda:G --output-dir OUT`; `running()` matches the exact output dir; lock; `[ -f "$SRC" ]` guard; skip-if-done.
- GUARD covers every module on the code path (train_cancel's `evaluate`, model_sign, model_mono_baselines,
  model_baseline_rope, model, lie_groups, environment_*, data_parallel, train, stats_core, analyze_*). The 75 other modules
  `train_variant` imports (VARIANT_MAP) are unguarded but off-path; `toy_wrap.py` is unguarded (finding 10).

**12. Disclosure (rule 29 pilots).** The freeze check in 2 used batch seeds 1000-1001 (my error; it should have used outside
seeds), at a non-registered recipe (CPU, 30 epochs x 20 batches, lr schedule compressed accordingly). Observed, not a result:
SO_absmag_fz s1000 final loss 0.044, acc 0.998; s1001 loss 0.533, acc 0.877. The registered GPU runs are unaffected; note it
in the amendment.

## Verified without finding
Defaults bit-identical (1); W_in gets no grad and no decay; freeze applied after `set_move_angles`; SO_same_fz / SO_absmag_fz
share every initial weight with SO_same / SO_absmag at a seed (requires_grad_ consumes no RNG); `mech_load` picks SignedAO for
`SO_*_fz`; omega sign flips cannot create opposition (they flip both moves); with frozen orthogonal latents W_out[c, 0] and
W_out[c, 1] set a(+1)_c and a(-1)_c independently, so only the per-channel route remains (modulo 2).
