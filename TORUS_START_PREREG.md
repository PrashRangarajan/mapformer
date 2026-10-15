# Does a start past pi rescue the non-negative step on the TORUS? -- pre-registration (2026-10-14, before any run)

## Question
MECHANISM (`MECHANISM_RESULTS.md`, registered, ring): a non-negative rotary step started with move-angle sums past pi learns
the ring map 8/8 (vs 0/8 default, 2/8 large sub-pi), with no sign cost from that start. The headline sign result is on the 2D
torus (SIGN_MATCHED; GENERALITY: observation-free AbsAO4 0/8 vs SignedAO4 8/8). Does the mechanism carry to the torus?
(Plan `docs/plans/2026-10-14_next_steps.md`, item 8; the one GPU batch for the workshop version.)

## Arms (`model_torus_start.py`, `train_torus_start.py`; torus 64x64, T = 1024, batch 16, 98 batches, lr 1e-3, cosine 5%
warm-up, 900 epochs, 1 layer, 2 heads, d 128, SDPA attention / TF32 off; observation-step-free 4-move models; seeds
1200-1207, n = 8)
| arm | step | start of the four move angles |
|---|---|---|
| SO4 | signed | default |
| AO4 | non-negative | default (= |SO4's|) |
| AO4_high | non-negative | iid U(1.7, 4.5) per channel and move (every opposite pair sums past pi), balanced gauge |
| SO4_high | signed | the same angles as AO4_high |
| AO4_mid | non-negative | iid U(0.9, 1.5) (pairs sum below pi), balanced gauge |
Checked on CPU: starts exact (<= 5e-7 rad); only W_in, W_out and omega differ from the default arm at a seed; the global RNG
state after construction is identical; AO4_high and SO4_high share their angles. No gauge-control arm (MECHANISM's M3 found
none on the ring); disclosed.
Evaluation (in the trainer): held-out map (env seed 10000), T = 1024 (primary) and 2048, 100 trials, as eval_noise_refine;
strata (eval_rank_strata.evaluate, T = 1024). SOLVED = classify_run (final 5% of training loss < 0.05). Floors (RANK_MI
stream): blank 0.507, best n-gram 0.576.

## Registered decisions (`analyze_torus_start.py`; labels as MECHANISM Amendment 1: BETTER / WORSE (Fisher p < .05 in that
direction, or perm p < .05 with |d| >= 0.05) / NO EFFECT DETECTED (SOLVED within 1, |d| <= 0.01) / UNMEASURED; 95% CI)
- **TS1 (primary): AO4_high vs AO4** -- BETTER = RESCUES ON THE TORUS. Gate: UNREADABLE if SO4 SOLVED < 4.
- **TS2: AO4_high vs AO4_mid** -- BETTER = the sum past pi, not size, matters on the torus (partial mobility control).
- **TS3: AO4_high vs SO4_high** -- matched-start sign contrast (NO EFFECT DETECTED = no sign cost from the large start).
- **TS4: AO4 vs SO4** -- replication of GENERALITY's E3 in this batch.
Predictions: TS1 RESCUES, TS2 BETTER, TS3 NO EFFECT DETECTED, TS4 WORSE (sign cost). On the torus a start past pi is a
weaker lever than on the ring (each channel must also resolve 64 cells per axis and two opposite pairs share each channel),
so a partial rescue is plausible; any TS1 label is reported as it falls.
Secondaries: wrap share (pairs (0,1), (2,3)) at the end; strata; accuracy at 2048 (OOD); loss AUC.

## Process and cost
No pilot beyond the CPU construction checks (the classes and recipe are GENERALITY's). Independent code check (rule 29)
before launch. The driver waits for the running ACCOUNTS batch to finish. 40 torus runs, ~30-45 min each at ~8 concurrent
on both GPUs: ~4 h overnight.

## Amendment 1 (2026-10-14, after the independent check `docs/audits/2026-10-14/torus_start_audit.md`; before launch)
The check confirmed: recipe, environment and evaluation identical to GENERALITY's torus runs; SO4 / AO4 bit-equal to
GENERALITY's classes at init on all 8 seeds; starts exact (<= 9.5e-7 rad); only W_in, W_out, omega differ between arms; RNG
state identical; latents c I_4 (no degeneracy); CPU smoke clean. Changes:
- **Gates on every contrast:** TS1, TS2, TS4 UNREADABLE if SO4 SOLVED < 4; TS3 UNREADABLE if SO4_high SOLVED < 4 (the smoke
  read NO EFFECT DETECTED with all arms at the floor).
- **Driver:** the clean-tree check and the first md5 pin run BEFORE the wait for ACCOUNTS (an edit during the wait can no
  longer abort the night or be pinned unreviewed); a second md5 check runs just before launch.
- **Checkpoint config** now carries d_model / n_heads / n_layers / grid_size / env keys for the shared reload tools.
- **Disclosed:** cost is ~6-8 h (not 4); RESCUES can fire on accuracy alone (d >= 0.05) with few runs SOLVED -- the SOLVED
  counts are reported beside it; with no gauge-control arm, TS1 is read together with TS2.
