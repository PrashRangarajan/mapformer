# RANKCLOCK (G-1): does extra rank help because it gives the clock an exit? -- pre-registration

Registered 2026-10-21 before any batch run; amended the same day by Amendment 1 (the independent results-blind code check,
rule 29), folded into the sections below and listed at the end. Design: `docs/paper/expansion/SYNTHESIS_DRAFT2_PLAN.md` row
G-1 (consolidating `RANK_THEORY.md` Sec. 6 G1/G1+, `SEARCH_THEORY.md` 6.1/6.3, open item X2). Run on this host by the user's
split (2026-10-21: G-1 here, draft 2 on the other machine; GPU 1 only). Code: `model_rankclock.py`, `train_variant.py`
(+ `--snapshot-every`, `--weight-decay`), `train.py` (+ `epoch_callback`), `analyze_rankclock.py`, `run_rankclock.sh`,
`construct_rankclock.py` + `run_rankclock_existence.sh`; checks `test_rankclock.py` (10/10), `test_epoch_callback.py`
(bitwise, CPU and GPU, 0 and 3 data workers; the reviewer reproduced it through the real command line),
`analyze_rankclock.py --selftest` (21 checks). **This file is md5-guarded by the driver: it is final at launch.**

## 1. Question

On the T = 1024 torus our shared rank 2 solves 0/8 and matched-init shared rank 4 solves 8/8 (`RANK_MI_RESULTS.md`; four of
the eight r2 runs were still DESCENDING at 900 epochs). Two accounts of why extra rank helps:

- **Clock exit (RANK_THEORY, "parking").** Every move adds a step common to all moves -- a clock -- to the latent. At rank
  r > D (D = 2) a channel's readout can turn away from the clock without touching the map; at r = D the clock lies in the map
  plane and a readout that stops seeing it must stop resolving the drift direction. Extra rank helps because it gives the
  clock a map-free exit.
- **Over-parameterisation (Burer-Monteiro style).** Spare rank smooths the landscape of the map factorisation.

**There are two clock exits, and rank 4 has both (Amendment 1 finding 1).** Per inverse pair the clock is u_x = z_+x + z_-x
(likewise u_y). A walk's drift per channel is (t/2)(u_x + u_y)/4 -- a COMMON part growing with the step count -- plus
(n_x - n_y)(u_x - u_y)/4, a DIFFERENTIAL part growing with net axis imbalance; at initialisation the second is as large as the
first (|u_x - u_y| / |u_x + u_y| median 1.22 for r2, 1.01 for r4, reviewer's measurement). SEARCH_THEORY S6(b): two
independent pair clocks need r >= 4. So:

- **slotxy (primary)**: shared r2 + one learned per-channel slot per move AXIS (clock_x on the x-moves, clock_y on the
  y-moves; +128 parameters, zero init). It can cancel both parts while the map stays rank 2 -- the arm with rank 4's exits and
  without its spare map rank.
- **slot (secondary, the plan's original arm)**: shared r2 + one per-token slot (+64). It can cancel only the common part (and
  the mean observation step: on this stream it is equivalent to a learned index-RoPE rate). With clock_x = clock_y = 2c the
  axis slot equals it at every OBSERVATION position and differs by c at ACTION positions, so neither strictly contains the other.

The slots add parameters but not map rank, so the over-parameterisation account predicts r2-like behaviour for both; that
reading is "by elimination", and the slots are affine extra readout columns, so "the over-parameterisation account gets
nothing it needs" is stated as that and no more.

G1+ (RANK_TESTS' ring logic on the torus): **r2DF** forbids the clock by construction (each inverse pair's moves exact
negatives, observation steps 0); **r2AO** zeroes only the observation steps (moves free).

X2: r4 at weight decay 0. **What it can and cannot answer (Amendment 1 finding 2).** At wd 0 nothing fixes the latent's scale
or gauge, so a small clock RATIO can come from the map growing (DILUTED), and Adam removes a nearly-parked clock with a small
residual readout on its own, so REMOVAL at wd 0 also fits "park first, then remove". X2's end-state readout therefore answers
the plan's question -- does the final shrinkage need decay? -- and does not by itself test parking. The parking prediction is
tested by the ORDER in the snapshots (X2b): the read clock falls before the latent clock.

## 2. Arms (`test_rankclock.py`, measured)

| arm | variant | step map | params | init vs Vanilla at the seed |
|---|---|---|---|---|
| r2 | Vanilla | shared rank 2, all tokens step | 204,373 | -- |
| **slotxy** | Vanilla_r2clockxy | r2 + clock_x on moves 0/1, clock_y on moves 2/3 | 204,501 (+128) | shared tensors bitwise equal; slots 0; forward bitwise equal |
| slot | Vanilla_r2clock | r2 + clock on every token | 204,437 (+64) | shared tensors bitwise equal; slot 0; forward bitwise equal |
| DF | Vanilla_r2DF | pair (+x,-x): +-W_out W_in (e(+x)-e(-x))/sqrt 2; others 0 | 204,373 | state dict bitwise equal |
| AO | Vanilla_r2AO | four moves W_out W_in e(move) free; others 0 | 204,373 | state dict bitwise equal |
| r4 | Vanilla_r4mi | shared rank 4 built from r2's base | 204,757 | differs only in W_in, W_out |
| r4wd0 | Vanilla_r4mi, weight decay 0 | as r4 | 204,757 | as r4 |

Torus moves 0 = (-1,0), 1 = (+1,0), 2 = (0,-1), 3 = (0,+1) (checked against the env); pairs (1,0), (3,2). clock_x changes only
moves 0/1, clock_y only 2/3, observation steps never (measured). DF: non-move steps exactly 0, pairs exact negatives,
move-sum clock exactly 0. AO: non-move steps exactly 0, init move clock 0.743. After construction the global torch RNG state of
slotxy, slot, DF and AO equals Vanilla's (reviewer; re-measured here incl. slotxy, which postdates the review); r4mi's
differs, as in RANK_MI. **Shared, not per-head, rank** throughout (CLAUDE.md invariant: the
paper's bottleneck is per head; SEARCH_THEORY 6.1 phrased its test per head; the plan consolidated it to shared, on which
RANK_MI's 0/8 vs 8/8 is measured).

**Rule-15 hazard (Amendment 1 finding 3), measured.** omega reaches 2 pi, so one Adam step at lr 1e-3 moves a slot's
top-channel phase by 12.87 rad after 2048 tokens -- about 2.5x the latent route's sensitivity (reviewer). A slot failure could
therefore be an optimiser artefact; the trainable existence check below is what decides whether it can be read.

## 3. Existence before mechanism (rule 14): frozen AND trainable

Source: a trained shared r4 (`Vanilla_r4mi`, pilot seed 1990, outside the batch, this host, same recipe). Candidate step
tables: r2proj (best rank-2 approximation), slot (rank-2 of S - 1 c^T plus c, c the FREQUENCY-WEIGHTED mean step over the
stream), xy (moves minus their axis slot at rank 2, plus the slots), AO (moves at rank 2, observation steps 0), DF (moves
antisymmetrised, observation steps 0). Each class is scored FROZEN by the best construction it can represent (r2: r2proj, AO,
DF; slot: + slot; slotxy: + xy; AO: AO, DF; DF: DF), with `eval_noise_refine.evaluate` on fresh maps at T = 1024
(`construct_rankclock.py`; realisations exact to ~1e-7 rad). TRAINABLE: each class's best construction warm-started for 60
cosine epochs at peak lr 1e-3, batch recipe otherwise (`run_rankclock_existence.sh`); HELD = still SOLVED and frozen accuracy
within 0.02 of the construction's.

Run 2026-10-21 18:27-18:45 (`run_rankclock_existence.sh`, `runs/rankclock_pilot/construct_rankclock.json`). Source r4 (seed
1990): **frozen acc 1.0000, final-5% loss 0.0021**; its step table's singular values 12.61, 10.51, 0.028, 0.009 -- the trained
rank-4 map is effectively rank 2.

| class | frozen, best construction (acc) | other constructions | trainable: 60 cosine epochs, peak lr 1e-3 | |
|---|---|---|---|---|
| r2 | r2proj 1.0000 | DF 1.0000, AO 0.6045 | 0.9998 after, peak loss 0.289, tail 0.0073 SOLVED | **HELD** |
| slot | r2proj 1.0000 | slot 1.0000, DF 1.0000, AO 0.6045 | 1.0000 after, peak loss 0.210, tail 0.0025 SOLVED | **HELD** |
| slotxy | r2proj 1.0000 | xy 1.0000, DF 1.0000, AO 0.6044 | 1.0000 after, peak loss 0.197, tail 0.0052 SOLVED | **HELD** |
| AO | DF 1.0000 | AO 0.6045 | 1.0000 after, peak loss 0.135, tail 0.0002 SOLVED | **HELD** |
| DF | DF 1.0000 | -- | 1.0000 after, peak loss 0.017, tail 0.0001 SOLVED | **HELD** |

Every class CONTAINS a solution and HOLDS it under batch-strength updates, so a failure of any arm in the batch is a search
failure. The rule-15 hazard is real but not disqualifying: the slot classes are knocked off their solution (peak loss ~0.2) and
re-find it. Constructed weight scales are 2-3x the source's (max |w_in| 1.24-1.74 vs 0.56; DF max |w_out| 1.57 vs 0.53), well
inside the 8x case that broke an earlier existence test. Descriptive, one source: keeping the source's move steps but zeroing its
observation steps (AO) scores 0.60, antisymmetrising the moves restores 1.00 -- the trained r4 cancels part of its move clock
with its observation steps (finding 6's concern, measured).

## 4. Batch

Torus 64 x 64, T = 1024, the RANK_MI recipe verbatim: 900 epochs, lr 1e-3, 98 batches of 16, cosine, AdamW wd 0.05 (0 for
r4wd0), 1 layer, 2 heads, d 128, no landmarks, 3 data workers. Fresh seeds **1900-1907** (unused anywhere else; reviewer
re-checked); pilots on **1990-1992**, never pooled. 7 arms x 8 = **56 runs**. Geometry snapshots (step-map state only:
action_to_lie, path_integrator, token_emb) at init and every 10 epochs. Evaluation `eval_noise_refine` on fresh maps (env seed
10000), T = 512 / 1024 / 2048, 100 trials. Host: RTX 4090, GPU 1 only (pinned; CUDA_DEVICE_ORDER=PCI_BUS_ID), two jobs at a
time (2 concurrent 4.2 s/epoch, 3 concurrent 6.1 s/epoch); ~31.5 min of throughput per run, **~29 h**. torch 2.10.0+cu128
(driver CUDA 12.4; minor-version compatibility verified). The driver refuses modified tracked files or any uncommitted G-1
file, md5-guards every imported mapformer module plus `ckpt_guard.py`, driver, lib and this file before training and before
evaluation, and requires 56 checkpoints and 56 final snapshots before evaluating.

## 5. Registered analysis (`analyze_rankclock.py`)

SOLVED = `stats_core.classify_run` (final-5% training loss < 0.05); non-finite loss = DIVERGED, never SOLVED. Each run's recipe
(incl. data_workers, grid, env, noise, init_from, fast_attn, snapshot cadence), weight decay and len(losses) = 900 are asserted;
a missing evaluation row raises (Amendment 1 finding 7).

- **G0** r4 SOLVED >= 6/8, else UNREADABLE. **G1** r2 SOLVED <= 2/8, else RANK-2 DIFFICULTY NOT REPLICATED (P, P2, E2 not read).
- **P (primary)** slotxy: k >= 6 and Fisher(slotxy, r2) < .05 -> **CLOCK EXITS HELP**; k <= 2 -> **CLOCK EXITS DO NOT HELP**;
  otherwise PARTIAL. Accuracy at T = 1024 and its permutation p printed beside.
- **P2 (secondary)** slot, same thresholds -> A COMMON-MODE EXIT HELPS / A COMMON-MODE EXIT DOES NOT HELP / PARTIAL.
- **E2 (G1+)**, first match: k_AO >= 6 and k_DF <= 2 -> AO SOLVES, DF DOES NOT; k_AO >= 6 and Fisher(AO, r2) < .05 ->
  OBSERVATION STEPS ARE THE OBSTACLE; k_AO <= 2 and k_DF >= 6 and Fisher(DF, AO) < .05 -> THE CLOCK IS THE OBSTACLE;
  k_DF <= 2 and k_AO <= 2 -> NEITHER (a second obstacle at rank 2); otherwise PARTIAL (finding 4 closed the ring rule's gaps).
- **X2** over SOLVED r4wd0 runs (>= 4, else X2 UNREADABLE). Per inverse pair, latents z = W_in e: u = z_p + z_q,
  delta = (z_p - z_q)/2; angles WRAPPED (rule 8): read clock = |wrap(omega W_out u)|, map read = |wrap(omega W_out delta)|,
  readgain = read / map; averaged over the two pairs; relative to the run's own snapshot e0000: shrink = |u_end| / |u_init|,
  rgrel = readgain_end / readgain_init. Medians, first match: shrink <= 0.2 -> **REMOVAL**; shrink >= 0.5 and rgrel <= 0.1 ->
  **PARKING**; u_ratio_end = |u| / (|z_p| + |z_q|) <= 0.15 -> **DILUTED**; otherwise UNRESOLVED. (Self-normalised, because the
  per-pair readgain starts at median 2.85 and the RANK_THEORY 0.25 bound was on a different quantity -- finding 5.) The r4
  (wd 0.05) values are printed beside.
- **X2b (secondary, P4 ordering)**, r4wd0 and r4 separately, solved runs: t_read = first snapshot epoch with read clock <= 0.5 x
  init, t_lat = first epoch with |u| <= 0.5 x init; runs with both defined and t_read != t_lat enter a two-sided sign test:
  READ CLOCK FALLS FIRST / LATENT CLOCK FALLS FIRST (p < .05), else NO ORDER.
- **Also printed:** run classes per seed; geometry medians per arm incl. each slot's angle size; r(log10 final loss, acc) per
  length (rule 2); per-seed accuracy.

**Power (Fisher, n = 8).** 6/8 vs r2 0/8 p 0.007; vs 1/8 p 0.041; vs 2/8 p 0.13 (-> PARTIAL). P can read HELPS only if r2
replicates at <= 1/8 or the slot reaches >= 7/8; a slot at 3-5/8 reads PARTIAL by design.

**Budget (rules 3, 4).** "Solves within 900 epochs" is the registered claim; a slot win could be a speed-up of a solution r2
would also reach (RANK_MI's r2 had 4/8 DESCENDING). Run classes are printed beside every reading; no extension is registered.

## 6. Predictions

- Clock exit: slotxy >= 6/8 (CLOCK EXITS HELP); slot may or may not (only half the exit); E2 THE CLOCK IS THE OBSTACLE (DF
  solves, AO does not); X2b READ CLOCK FALLS FIRST. X2's end state is not a parking test (Sec. 1); REMOVAL at wd 0 means the
  gradient removes the clock without decay, PARKING that without decay a large clock is kept unread.
- Over-parameterisation: slotxy and slot <= 2/8 (like r2); nothing predicted for E2, X2, X2b.
- A slot failure is readable only if the trainable existence check HOLDS for that class (Sec. 3).
- The pilots (Sec. 7) are seeds outside the batch and change none of these.

## 7. Pilots (seeds 1990-1992, outside the batch; never pooled)

| pilot | arm | final-5% loss | class | acc @ T = 1024 | snapshots |
|---|---|---|---|---|---|
| 1990 | r4 (Vanilla_r4mi) | 0.0021 | SOLVED | 1.0000 | 91 |
| 1991 | slot (Vanilla_r2clock) | 0.2587 | DESCENDING | 0.9344 | 91 |
| 1992 | DF (Vanilla_r2DF) | 0.0001 | SOLVED | 1.0000 | 91 |

One seed each: anecdote, not evidence. Run with the pre-amendment code (no slotxy arm; identical training path -- the amendments
touched the analysis, the construction and the driver, and added an arm). Timing: 2 concurrent 4.2 s/epoch, 3 concurrent
6.1 s/epoch (r2/r4/slot); DF runs ~7.9 s/epoch when 3 share the GPU (its step table is built per forward), so DF arms take
~25% longer.

## Amendment 1 -- independent results-blind code check (rule 29), 2026-10-21

Reviewer read this file, the plan / RANK_THEORY / SEARCH_THEORY sections, all G-1 code and its precedents (RANK_MI, RANK_TESTS,
rank_geom), blind to `runs/` and logs; ran `test_rankclock`, the selftest, `test_epoch_callback`, the real `train_variant`
command with snapshots on/off and wd explicit/omitted (bitwise equal; wd 0 differs, as it must), and an RNG-state check. Found
the arms, the hook, the default paths, the recipe, the layout and the slot handling sound. Findings and dispositions:

| # | sev | finding | disposition |
|---|---|---|---|
| 1 | MAJOR | the per-token slot opens only the common-mode exit; the differential exit (u_x - u_y) is as large at init; rank 4 has both | ACCEPTED, user decision 2026-10-21: per-axis slot added as the PRIMARY; the plan's slot kept as P2 with "common-mode" labels; "by elimination" wording |
| 2 | MAJOR | X2's end-state readout cannot carry "parking": dilution at wd 0, and Adam removes a nearly-parked clock | FIXED: shrink and rgrel relative to e0000, DILUTED label, P4 ordering X2b from the snapshots, Sec. 1 / 6 reworded |
| 3 | MAJOR | existence was frozen only; the slot's phase sensitivity (12.9 rad per step) makes a failure possibly an optimiser artefact | FIXED: 60-epoch trainable warm start per class (Sec. 3); a slot failure is read only if its class HOLDS |
| 4 | MINOR | E2 gaps (AO solves but reads NEITHER; AO >= 6 with DF <= 2 reads observation steps); verdict fallback could hide a missing field | FIXED: AO SOLVES, DF DOES NOT added, NEITHER needs k_AO <= 2; selftest cases incl. r2 2/8 + slot 6/8 -> PARTIAL; no fallback |
| 5 | MINOR | readgain bound calibrated on a different quantity (per-pair init median 2.85); angles unwrapped | FIXED: X2 self-normalised to e0000; angles wrapped |
| 6 | MINOR | construction: plain-average slot c; false "can only beat r2" claim; observation steps zeroed without checking | FIXED: frequency-weighted c; claim removed; each class scored by the best of every construction it can represent |
| 7 | MINOR | recipe assert incomplete; a missing eval row silently read as accuracy 0 | FIXED: full recipe asserted (checked against train_variant's real parsed args); missing row raises (selftest) |
| 8 | MINOR | ckpt_guard.py not md5-guarded; guarded prereg would block a post-launch edit; inherited DRV_GPUS | FIXED: ckpt_guard added; this file final at launch; DRV_GPUS=1 unconditional + CUDA_DEVICE_ORDER=PCI_BUS_ID |
| 9 | NOTE | DF host sync per forward; DF init step sqrt 2 x Vanilla's half-difference (as the ring precedent) | sync removed (host copy of the pairs); scaling recorded |
| 10 | NOTE | with r2 at 2/8 HELPS needs the slot at 7/8 | kept in Sec. 5 beside the budget caveat |
