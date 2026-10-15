# TORUS_START: independent code check (rule 29), 2026-10-14, before launch

Scope: `TORUS_START_PREREG.md`, `model_torus_start.py`, `train_torus_start.py`, `analyze_torus_start.py`, `run_torus_start.sh`
at HEAD 46be0be. Context read: MECHANISM prereg / results / audit, GENERALITY prereg / results and `run_generality.sh`,
`train_variant.py`, `model_wrap_exist.py`, `model_sign.py`, `model.py`, `train.py`, `environment.py`, `data_parallel.py`,
`eval_noise_refine.py`, `eval_rank_strata.py`, `stats_core.py`, `lib_driver.sh`, `train_wrap_mech.py`. CPU only. Check
scripts are in the session scratchpad: `ts/c1.py` checks starts, RNG and weight identity. Smoke: 5 arms x seeds 1200-1201,
`--epochs 2 --n-batches 2 --batch-size 2 --n-steps 64 --data-workers 1 --n-trials 2 --device cpu`, then
`analyze_torus_start --seeds 1200-1201 --no-json`. All 10 runs and the analysis completed. No repo file was edited except
this one.

**Verdict.** The code does what the prereg says, and the recipe is GENERALITY's. There is no BLOCKER. Two MAJOR issues should
be fixed in an amendment before launch. Both fixes are cheap: one is a gate, the other moves two lines in the driver.

## Verified correct
- **Recipe = GENERALITY torus (`run_generality.sh` + `train_variant`).** Every item below matches:
  - Environment: `GridWorld(64, n_obs 16, p_empty 0.5, n_landmarks 0, seed=s)`. The other kwargs (action_mode, obs_mode,
    boundary, score_moves_only, action_record, n_headings, heading_noise) are the defaults in both entry points.
  - Training: T 1024, batch 16, 98 batches, lr 1e-3, wd 0.05 (`train` default), cosine, 900 epochs, 3 data workers.
  - Data stream: base seed = `torch.initial_seed()` in both.
  - Attention: `USE_SDPA = True` with TF32 matmul and cudnn off, set before seeding as in train_variant. No `--deterministic`
    in either.
  - Evaluation: `eval_noise_refine.evaluate` on map seed 10000 at T 1024 / 2048, 100 trials, seed 1234 + s;
    `eval_rank_strata.evaluate` at T 1024 with the same seed. In the smoke, strata 'all' equals the eval accuracy
    (0.3397 = 0.3397).
  - The differences are listed in finding 6.
- **SO4 / AO4 are GENERALITY's classes.** `ARMS["SO4"]` and `ARMS["AO4"]` are the same class objects as
  `VARIANT_MAP["SignedAO4_r4"]` and `VARIANT_MAP["AbsAO4_r4"]`. Under the trainer's seeding the state dicts are bit-equal, on
  all 8 seeds. AO4's angles equal |SO4's| exactly.
- **Starts** (seeds 1200-1207):
  - Exact: max wrap-aware error 9.5e-7 rad.
  - Only `w_in`, `w_out` and `omega` differ from SO4's state dict at a seed.
  - The torch and numpy global RNG states after construction are identical across all five arms, so dropout masks and the
    data stream are the same in every arm.
  - AO4_high and SO4_high have identical angles (difference 0.0).
  - Pair sums: High lies in [3.48, 8.95], so every pair is past pi. Mid lies in [1.82, 2.99], so every pair is below pi.
  - All starting steps are positive (High: min raw step 0.34-0.37), so `|.|` does not bind at init and AO4_high and SO4_high
    start with the same forward pass and the same gradient.
- **`torch.initial_seed()`** equals `--seed` at construction on every seed. Nothing reseeds between `manual_seed` and the
  model (GridWorld uses its own `RandomState`). The generator is CPU-side and independent of the data seed. A model rebuilt
  without seeding draws different angles, but `load_state_dict` overwrites them.
- **Balanced gauge with r = 4 moves.**
  - The latents are c·I4, with cond 1.0 in all three gauge arms against 4-12 at default init. There is no degeneracy. The
    "condition number in the thousands" belonged to trained torus sources and does not apply here.
  - Unlike the ring (2 moves, effectively rank 2), all four latent dims are used. No parameter has an identically zero
    gradient.
  - The `W_in` gradient stays in the span of the move embeddings, as it would anyway.
- **Analysis.**
  - `perm2_p(r, a)` tests mean(x) - mean(ref), the same sign as `d`. Fisher is two-sided and its direction is checked
    separately.
  - The thresholds match MECHANISM Amendment 1. Every label is reachable: BETTER/WORSE/NO EFFECT/UNMEASURED, the renamed
    ones, and UNREADABLE for TS1/TS2.
  - The eval.json fields used (`eval.1024.acc`, `eval.2048.acc`, `wrap[1]`, `strata.*.acc`) and `.pt["losses"]` all exist.
  - An empty stratum gives `acc None`. It is filtered out; if every seed is empty the mean is nan (a warning, not a crash).
  - SOLVED = `classify_run` on the 900 per-epoch losses.
- **Driver.**
  - Knobs are assigned after sourcing. The lock, the running/done skips and the `--output-dir` match are sound.
  - `drv_wait_dir "$R/"` matches every launch. 80 = 40 runs x 2 artifacts.
  - The md5 guard runs again before the analysis, and the marker is set only after the outputs exist.
  - On an ACCOUNTS failure (FAILED/ABORT/REFUSED in `accounts.log`) the driver waits for every `runs/accounts/` trainer to
    exit. MAXPG 4 is the same as GENERALITY, which ran this model at T 1024 on both GPUs.

## Findings

**1. MAJOR: TS3 and TS4 read "NO EFFECT DETECTED" vacuously at the floor.**
- The smoke had all arms at 0.34 and 0/2. It printed `TS3 ... NO EFFECT DETECTED` and `TS4 ... NO EFFECT DETECTED`, while
  TS1 and TS2 were correctly UNREADABLE.
- TS4 is GENERALITY's E3, which was gated on SO4 >= 4 (as was MECHANISM's M4); here it is not gated.
- TS3 is the realistic risk. The prereg itself expects a weaker lever on the torus. If both large-start arms fail, TS3 reads
  "no sign cost from the large start", when the truth is that neither class learns from that start.
- Fix:
  - TS4 becomes UNREADABLE if SO4 SOLVED < 4.
  - TS3 becomes UNREADABLE ("neither learns from the large start") if SO4_high SOLVED < 4.
  - Also state in the prereg that TS2 is gated: the code gates it, the prereg text names the gate only for TS1.

**2. MAJOR (operational): the clean-tree check and the first md5 guard run only after the ACCOUNTS wait, which lasts
hours.**
- (a) `git status --untracked-files=no` covers every tracked file, including `.claude-memory/*` and every results document.
  Any uncommitted edit made while ACCOUNTS finishes (state notes, a results write-up) makes the driver log REFUSED and exit,
  and the night is lost.
- (b) `code_md5.txt` is first written after the wait. Code edited and committed during the wait is therefore silently
  pinned as the batch's code, not the code this check reviewed. `model_wrap_exist.py` is also ACCOUNTS's live module, so an
  edit to it is plausible.
- Fix:
  - Run the clean check and `drv_md5_guard "$R" "${GUARD[@]}"` before the wait loop, so the code is pinned at launch.
  - After the wait, re-run only `drv_md5_guard`. Drop the whole-tree check there, or limit it to `${GUARD[@]}`.
  - Optionally, compare against the amended commit: `git diff --quiet <commit> -- "${GUARD[@]}"`.

**3. MINOR: "RESCUES ON THE TORUS" can fire with 0/8 SOLVED.**
- BETTER is also reached by perm p < .05 with d >= 0.05. AO4 sits at 0.88 (GENERALITY), which leaves room for that.
- Fisher needs at least 5/8 against 0/8: 4/8 gives p 0.077 and 5/8 gives p 0.026.
- Fix: label RESCUES only if BETTER and AO4_high SOLVED >= SO4 SOLVED - 1. Otherwise label it "HELPS, PARTIAL".

**4. MINOR: there is no gauge control, and on the torus the gauge also flattens the frequency ladder.**
- TS1 changes the start and the gauge together. The gauge makes the latents orthogonal and sets omega' = max(|omega|, 1). That
  sets 36 of 64 channels (omega 0.098-0.98) to exactly 1; on the ring it was about 30 of 64.
- It also changes the mobility of the angles under Adam (MECHANISM audit finding 3). TS2 is gauge-matched; TS1 is not.
  MECHANISM's ring M3 null on the gauge likely transfers, but it is not measured here.
- Fix (free): pre-state the reading. TS1 BETTER means "a large start in the balanced gauge". "Sums past pi" can be claimed
  only if TS2 is also BETTER. Register SO4_high vs SO4 as a descriptive check that the gauge plus the large start does not
  break the signed model on the torus.
- Fix (optional): an AO4_absmag arm (+8 runs) closes the gap.

**5. MINOR: the cost is underestimated.**
- The GENERALITY torus logs at 4 jobs per GPU show 4.6-6.9 s/epoch for SO4/AO4. That is 70-100 min per 900-epoch run, not
  30-45.
- 40 runs in 8 slots make 5 waves, which is about 6-8 h, not about 4 h. This is still overnight, but correct the prereg's
  figure.

**6. MINOR: differences from GENERALITY's torus runs** (no numerical effect expected; list them):
- (a) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not set, and `DRV_MINFREE` is 4000 against GENERALITY's 6000.
  These affect memory only.
- (b) Evaluation runs in-process with SDPA on. GENERALITY's evaluator reloaded the model on the explicit path. The two agree
  to about 1e-6 in the logits.
- (c) There is no dropout-scale re-score secondary; GENERALITY A4 and MECHANISM both had one.
- (d) The checkpoint config lacks `fast_attn`, `d_model`, `n_heads`, `n_layers` and `grid_size`, and its `variant` is the
  arm name, which is not in `VARIANT_MAP`. `eval_noise_refine`, `eval_rank_strata` and the `ckpt_guard` tools therefore
  cannot reload these checkpoints; any post hoc analysis needs `model_torus_start.ARMS`.
  Fix: add those keys plus `"class": <VARIANT_MAP name>`.

**7. MINOR: prereg text.**
- The start error is up to 9.5e-7 rad (seeds 1200-1207), not "<= 5e-7".
- State the measured pair-sum ranges (High [3.48, 8.95], Mid [1.82, 2.99]).

**8. MINOR: the md5 guard covers the compute path but not the full import tree.**
- `train_torus_start` pulls in 89 `mapformer.*` modules, through `train_variant`'s `VARIANT_MAP` via the evaluators.
- A grep finds no import-time patch of `model`, `torch.backends` or the RNG in any of them, so they cannot change the
  numbers.
- A syntax-broken edit to one of the roughly 75 unguarded modules mid-batch would crash later launches. `drv_require` would
  catch that, so it is loud, not silent.
- Fix (optional): md5 the module list from `sys.modules` at launch, or restate rule 22 for this batch.

**9. MINOR: wait-loop edge cases.**
- If the ACCOUNTS driver is killed without logging FAILED/ABORT/REFUSED, the loop waits forever. Fix: also break when no
  `run_accounts.sh` bash process and no `runs/accounts/` trainer remain.
- `drv_pick` checks only free memory, not other users' jobs (memory note on GPU sharing). Check by hand at launch.

**10. MINOR: the wrap-share secondary is unweighted.**
- ACCOUNTS Amendment 1's usage-weighted share is the better readout, because the ring's large-start models solved with
  23-34% cancelling channels.
- `usage_weights` defaults to `n_actions=2`, so the torus needs `n_actions=4`. This can be done post hoc from the
  checkpoints once finding 6d is fixed.

## Design (question 5)
- n = 8 is enough for a ring-sized effect (8/8 vs 0/8, Fisher p 0.0002).
- A partial rescue reaches significance by Fisher only from 5/8 upward. Accuracy-based BETTER covers the rest, which is the
  reason for finding 3.
- Without a gauge arm, TS1 alone cannot separate "start" from "gauge/mobility". Read it jointly with TS2 (finding 4).
- TS4 is a near-certain replication (GENERALITY: 0/8 vs 8/8).
