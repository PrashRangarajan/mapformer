# MEMORY_MAZE: path integration of given self-motion on a first-person 3D benchmark (pre-registration)

Status: BUILT, NOT LAUNCHED. Written 2026-10-27 before any batch run. Pilots on outside seeds are recorded in the
Pilots section and labelled as such. An independent code check (rule 29) must run before launch. Its findings go into an
amendment before any result is read.

Gates: `MEMORY_MAZE_GATE.md` (v1, STOP), `MEMORY_MAZE_GATE_V2.md` (v2, STOP), `MEMORY_MAZE_GATE_V3.md` (v3, PASS at
16 blind / 16 sighted, scored on fresh train-part0 episodes; post hoc diagnostic on the official eval split: class E
0.514, at the threshold). Review: `docs/audits/2026-10-26/CRITICAL_REVIEW.md` Part A.

## Question

On Memory Maze 15x15 (Pasukonis et al. 2022), the question is asked at the returns that need a map. The agent comes
back to a place after a stretch with no observations, and the earlier visit was more than 128 steps ago along a loop.
There, does one attention layer whose rotary phase integrates a learned step map of given self-motion retrieve the view
better than index-RoPE attention given the same inputs as tokens?

**Scope.** Heading is GIVEN: the agent's direction vector, a compass, enters the step input. So the claim is
"integrates given self-motion", not "navigates from pixels". P_odo and GPS-PE are given position outright, the class
of GTA / PRoPE. P_E and P_raw learn their step map. This is the IMAGE_TORUS/TEXTWORLD question on an external
first-person benchmark with egocentric actions, momentum and collisions.

## Data (what is on the cluster; no more downloads)

- **15x15 `train-part0`:** 2900 episodes; index = position in the sorted member list. The frame encoder is trained on
  0-299. Models train on 0-1599. **1600-2899 are the held-out secondary scoring set**, untouched by every gate and
  every fit. Pilots train on 0-1499 and validate on 1500-1599.
- **Official 15x15 eval:** 1000 episodes; the **primary scoring set**.
- **The split difference, stated in advance.** The official eval split was generated on 2022-09-20 and train-part0 on
  2022-09-23, in separate runs.
  - Forward-type collisions are 1.02% of forward-type steps in eval vs 0.57% in train; the action mix differs slightly.
  - The gate's linear class-E dead reckoning puts 0.514 +/- 0.011 of eval long-gap returns within 0.5 cell, against
    0.605 on fresh train-part0 episodes.
  - So the primary split is the harder one for the step input, and models are trained on the other generation run.
    The held split replicates the primary on the training run's distribution.
- **No other archive is used.** The other nine train archives (~90 GB) are not downloaded.

## Task

- **Sequence:** [f_0, s_1, f_1, ..., s_1000, f_1000], T = 2001 tokens per 1000-step episode. s_t is the step token of
  action a_t (which produced frame t); the output at s_t predicts f_t.
- **Blinding:** cell B 16 blind / S 16 sighted (period 32, half the frames seen), frame 0 always sighted.
  - Training: random phase per sample, drawn from (seed, epoch, batch).
  - Scoring: the gate's schedule exactly (`mm_data.sighted_mask` = `mm_gate_v2.sighted_mask`, phase seeded by the
    archive index).
  - A blind frame token carries a learned blank vector; the model sees only the step tokens there.
- **Scored steps (blind re-entries), the gate's definition** (`mm_data.scored_steps`, which `test_mm` checks row for
  row against `mm_gate_v2`): t is the first sighted step after a full 16-step blind run, its pose bin (cell + 8
  headings) differs from the last sighted frame's, and that pose bin was sighted before the run.
  - **Primary stratum: non-local** = gap >= 128 since that sighted visit, and not a backtrack.
  - Reported too: "all" and "long gap".
- **Readouts:**
  - **R1 (primary):** the episode's frame (any of the 1001) whose key is most similar to the query at s_t has the
    pose bin of t. Earlier same-bin frames are therefore correct answers.
  - **R2 (secondary):** f_t beats every frame in a different pose bin.
- **Loss:** InfoNCE. The query at s_t is scored against the keys of all 1001 frames of the episode (cosine / 0.1);
  the positive is f_t; frames u != t in the same pose bin as t are removed from the negatives (`train_mm.info_nce`;
  `test_mm` checks the mask exactly).
  - As first written: the mean over every sighted step t >= 1.
  - **Amendment 1 (pilot-informed):** the mean over sighted steps PLUS the mean over post-blind steps (the first
    sighted step after each blind run, ~31 per episode, scored or not), `--loss-steps both`, the trainer's default.
  - Reason: under the first loss every arm stayed near the floor (Pilots, wave 1).

## Frame encoder (frozen)

- **Architecture:** convolutional autoencoder, 64x64x3 -> 256 -> 64x64x3, four stride-2 convs, MSE loss.
- **Training:** every frame of train-part0 0-299 only, 8 epochs, Adam 1e-3, batch 256, seed 20261027. It is an
  autoencoder rather than time-contrastive, because a time-contrastive objective learns pose adjacency (review A-M1).
- **Storage:** frozen; embeddings of every train-part0 and eval frame precomputed once into a few large `.npy` arrays
  on cluster scratch (`mm_encoder.py prep`; GPFS dislikes many small files).
- **Pose leak** (`encoder_quality.json`, on eval frames):
  - cross-maze ridge probes from the embedding to heading bin and to cell, against majority / uniform chance;
  - single-frame identification: is the nearest earlier frame (s <= t - 16) in the same pose bin? This is the gate's
    "ident"; 0.36 in pixel space.
  - Reported in Pilots.

## Arms

Shared by every arm: d 128, 2 heads, the same input embedding and retrieval head, dropout 0.

Frames enter CONTENT ONLY: in the path arms the step map reads only the step feature of step tokens, and frame tokens
have step 0 (the IMAGE_TORUS design rule; `test_mm` checks it). The step-token content is the E feature for every arm,
plus d_t for the arms given position.

| arm | position mechanism | step / position input | role |
|---|---|---|---|
| **P_E** | path: theta = omega * cumsum(W_out W_in phi_t), rank 4 shared latent, omega geometric (Delta_max 64), 1 layer | phi_t = E feature, 168-d: one-hot (a_{t-1}, a_t) (42) x [dir_{t-1}, dir_t] (4). A LEARNED linear step map on it is exactly the gate's class E; the fitted table is never used as the step | primary path arm |
| **P_odo** | path, 1 layer | phi_t = true displacement d_t (2-d); content also gets d_t | given-odometry ceiling |
| **P_raw** | path, 1 layer | phi_t = one-hot a_t (raw egocentric action, no heading) | secondary: the rotation negative (`KNOB_SWEEP_n8.md`) on an external task |
| **R1** | index RoPE (canonical base 10000), 1 layer | same tokens as P_E | primary comparator |
| **R4** | index RoPE, 4 layers | same tokens as P_E | depth control |
| **GRU** | one-layer GRU, hidden 184 (core 197,008 params vs one transformer layer's 198,272) | same tokens as P_E | recurrent control |
| **GPS** | index RoPE + additive learned projection of sin/cos(2 pi p / lambda), lambda = 1..64 cells (16 per axis), p = pos_t - pos_0 | content also gets d_t | position given as an absolute code: separates "a position code" from "path integration" |

- **Parameter counts:** 304-313k at 1 layer (R4 899k); the GRU arm is 303k.
- **GRU caveat** (`RNN_TORUS_RESULTS.md`): those recurrent networks were trained on ONE map and memorised it. Memory
  Maze draws a new maze every episode, so this GRU is the many-map test that batch did not run. A GRU win here does
  not contradict RNN_TORUS, and a GRU loss does not replicate it.

**Recipe:**
- AdamW lr 3e-4, weight decay 0.05 on all parameters (the RANKWD condition for rank 4).
- Linear warmup over 2% of steps, then cosine to 0; grad-norm clip 1.0.
- Batch 16 episodes (100 steps per epoch on 1600 episodes); fp32.
- Fast attention (SDPA): exact up to ~1e-6. Dropout is 0, so there is no RNG difference (rule 19).
- Epochs: **E = 600** (Amendment 1, pilot-informed: every wave-3 validation curve plateaued between epochs 375 and
  575; the trainer's default).
- Dropout 0 deviates from the house default 0.1, to avoid the eval-mode 1/(1-p) under-report (CLAUDE.md NormStep row).

**Seeds:** 3600-3607, fresh; every arm at every seed (rule 12, one batch). Pilots use 3690-3699.

## Rule 14 (existence)

`model_mm.install_e_map` installs the least-squares E map, fitted on train-part0 0-299 (`e_map.npy`), as P_E's step
map. Latent 0/1 = (dx, dy); W_out sends it to every (head, block) along a random unit direction, so each block's angle
is omega_k u_k . (dead-reckoned position). `test_mm` checks this to 1e-3.

Two pilot runs at one pilot seed check that the class holds a solution:
- installed **frozen** (W_in, W_out, omega fixed; the rest trains);
- installed **trainable**;
- each against P_E from scratch, at the same budget.

The gate's ceiling for this class is integrator retrieval of 0.66-0.70 under R1.

## Registered contrasts (`analyze_mm.py`; `--selftest` reaches every label)

Metric per run: R1 on non-local blind re-entries of the official eval split. n = 8 per arm. Exact two-sample
permutation p on per-seed values (`stats_core.perm2_p`). Effect bar 0.05.

**PRIMARY: P_E vs R1**, d = mean(P_E) - mean(R1):
- INCOMPLETE (no label) if P_E, R1 or P_odo has fewer than 8 runs.
- UNREADABLE if mean(P_odo) < eval floor + 0.10: the task is not learned even with given odometry.
- PATH WINS if d >= 0.05 and p < .05.
- INDEX WINS if d <= -0.05 and p < .05.
- UNMEASURED otherwise; the exact-t MDE is printed.

**SECONDARIES** (Holm over the four):
- S1: P_E vs R4.
- S2: P_E vs GRU.
- S3: P_odo vs P_E (the cost of learning the step).
- S4: P_odo vs GPS.

Each reads '<A> ABOVE <B>' / '<B> ABOVE <A>' / 'UNMEASURED'.

**DESCRIPTIVES (no verdict):**
- every arm minus the eval floor;
- the primary rule replicated on the held split (labelled HELD);
- R2; the all-returns stratum;
- context destruction: input frames swapped with another episode's, and step features shuffled in time; a
  context-using model must fall toward the floor;
- P_raw vs P_E;
- loss-tail slope (DESCENDING count);
- the offline probe.

**Floors on the scoring splits** are the gate's best map-free floors under R1 (pixel / PCA spaces, `mm_gate_v2`
F2-F3):
- eval 0.0472 (`docs/audits/2026-10-27/memory_maze_gate_v3/diag_eval_split.json`);
- held: `mm_floors.py` on 1600-2899, filled in Pilots.

Floors are in the gate's feature spaces, not the model's key space. They bound map-free retrieval from the frames
themselves.

## Offline probe (standard secondary, for comparability)

Protocol read first-hand: Pasukonis et al. 2022 (2210.13383, Sec. 3.3, Table 4, App. D, Table E.3) and Deng et al.
2023 (2307.02064, App. D, Table 6).
- **Targets:**
  - Walls: `maze_layout` (225 tiles), BCE, metric = accuracy over tiles (%).
  - Objects: **`targets_vec`** (agent-centric, 6 objects x 2), MSE loss, metric = squared distance averaged over
    objects. The per-coordinate MSE is also reported; Amendment 1 says which definition reproduces the published
    constant baseline 64.8.
- **Probe input:** the final hidden state (after the last LayerNorm) at the frame token f_t (REGISTERED; the analogue
  of Deng et al.'s TSSM-XL h_t; 128-d against their 2048), concatenated with `agent_pos` and `agent_dir`.
- **Probe network:** an MLP with 4 hidden layers of 1024 units, LayerNorm and ELU. AdamW 3e-4, 10,000 steps of batch
  1024, trained on frozen states of the training episodes (all t), no gradient into the model.
- **Scoring:** eval steps 500-1000.
- **Streams:** registered on the blinded stream the model was trained on; unblinded as a secondary (a distribution
  shift for these models).
- **References at 15x15** (Walls % / Objects MSE): constant 78.3 / 64.8; VAE 78.7 / 56.2; RSSM-TBTT 80.5 / 33.8
  (Deng 81.7 / 32.6); TSSM-XL 79.8 / 35.1; S4WM 80.2 / 34.8; S5WM 81.8 / 25.3; supervised oracle 88.5 / 9.2.
- **Comparability limits:** Walls has ~3.5 points of headroom. P_odo and GPS are given position, so they are compared
  only within our arms. Our models see half the frames, and the published models see all of them.

## Predictions (before pilots)

- P_odo is high, near the gate's odometry integrator (0.93); UNREADABLE is not expected.
- P_E is between R1 and P_odo, near the class-E integrator (0.66 on eval).
- R1 is low: index attention cannot integrate ~300-step gaps (FACTORIAL); R4 is a little above R1.
- GPS is close to P_odo.
- GRU: no prediction (many-map).
- **Primary prediction: PATH WINS.**
- The probe: Objects better than constant for every arm with memory; Walls within a few points of constant for all.

## Power

- n = 8 per arm; exact permutation minimum p = 1.6e-4.
- MDE = exact-t multiplier (3.26 at n = 8) x sd x sqrt(2/8). At an sd of 0.02-0.05 that is 0.03-0.08 against the 0.05
  bar.
- The scored set is ~5 non-local re-entries per eval episode, about 5k steps, so the binomial SE per run is <= 0.007.
- Pilot sds are in Pilots.

## Cost

- Measured on a workstation 4090 (synthetic batch of 16): 11 ms per step for the 1-layer arms, 30 ms for R4 and GRU;
  peak memory 0.7-1.4 GiB.
- At 100 steps per epoch: ~1 s per epoch (R4 / GRU ~3 s).
- Scoring plus probe: minutes per run.
- The whole batch (7 arms x 8 seeds = 56 runs) is a few A40-h (see Pilots for the measured figure), well inside one
  8-9 h array window. Every run checkpoints each epoch and resumes.
- P_raw is included because cost allows.
- Encoder plus embeddings: one GPU job (Pilots).

## Launch procedure (cluster, preemptible partition; specifics in the local note, never here)

1. `hyak/mm_sync.sh` (workstation) pushes the code as a minimal package (empty `__init__.py`; md5 checked both ends).
2. Prep (DONE): encoder, arrays, E map, encoder quality.
   - It ran on the workstation (`run_mm_pilot.sh` stage 1, `mm_encoder.py prep`; 4 min on a 4090) because the cluster
     GPU queue had not started it after 2 h.
   - The arrays were copied to cluster scratch and verified there file by file against `MANIFEST.json`: 16 of 16 md5
     match. So the batch's frozen encoder and embeddings are byte-identical to the pilots'.
   - `hyak/mm_prep.slurm` is the cluster alternative, NOT used.
   - `hyak/mm_floors.slurm` (CPU, DONE): held-split floors.
3. Pilots (DONE, workstation): `run_mm_pilot.sh` (wave 1), manual wave 2, `run_mm_pilot3.sh` (wave 3).
4. Amendment 1 (written; see Pilots). The independent check's findings become Amendment 2, before launch.
5. The batch: the task table `hyak/mm_batch_tasks.txt`, "<ARM> <SEED>" lines (seed outer, arm inner). Since
   Amendment 2 it has 64 lines (the 8 arms incl. P_E_frozen x seeds 3600-3607), md5
   b448be6619378ad0eff8dce20fd79862; before that, 56 lines, md5 adf8812985b59b8c6c2dfa0044f91ef1. Submitted as an
   array with
   `--constraint "a40|a100|l40"`, `--requeue`, `--time 08:00:00`, and a second chained array (`afterany`) as a safety
   net. Completion = `DONE` in every run dir (written after `results.json`).
6. Fetch results.json files to the workstation, then
   `python3 -m mapformer.analyze_mm --root <runs> --floors-eval <diag json> --floors-held <floors json>`.

Cluster batches run torch 2.13 against the workstation's 2.10, and are never pooled with workstation runs.

## Pilots (outside seeds 3690-3694, workstation 4090s, torch 2.10; never pooled with the batch, which runs on the cluster under torch 2.13)

Artifacts: `docs/audits/2026-10-27/memory_maze_batch/` (`pilots/PILOT_WAVES12.*`, `pilots/PILOT_WAVE3.*`,
`encoder_quality.json`, `*_info.json`, `MANIFEST.json`, `floors_held.*`).

**Setup.** Pilots train on train-part0 0-1499 and validate every 25 epochs on 1500-1599: training-split episodes
never trained on and never a scoring split (`--pilot`). Validation is R1 on non-local blind re-entries, ~540 per
point, so SE ~0.02. No pilot touched the eval split or the held split, except the model-free constant probe baseline
on eval, used as a metric calibration.

The pilots span three code states: the loss option was added after wave 1, and a peak-memory logging fix landed in
wave 3. Each pilot's results.json carries its md5s. The `peak_gib` of some wave-1/3 runs reads 0.0 (that bug); the
synthetic benchmark measured 0.7-1.4 GiB.

**Encoder.**
- The autoencoder reaches reconstruction MSE 0.00068 on eval frames (0.00066 on train-part0 frames). The 300
  training episodes take about 5 s per epoch on a 4090.
- Pose leak on eval frames:

  | probe | value | chance |
  |---|---|---|
  | heading bin | 0.248 | majority 0.174 |
  | cell | 0.017 | uniform 0.004 |
  | single-frame identification | 0.121 | (pixel space in the gate: 0.36) |

- So one embedding identifies its pose bin LESS well than raw pixels do; review A-M1 is not a concern for this
  encoder.

**Wave 1: the loss as first written** (sighted steps only; 800 epochs; seed 3690):

| arm | validation |
|---|---|
| P_E | 0.039 |
| R1 | 0.035 |
| P_odo | 0.311 |
| GRU | 0.009 at epoch 200 (stopped) |

- Training loss fell to 0.8-0.9. The loss is dominated by locally predictable sighted steps: ~5 scored returns per
  ~500 sighted steps. Neither the path arm nor the index arm learned the returns; they sat near the map-free floor
  (0.047).

**Wave 2: loss "both"** (seed 3692, 400 epochs):

| arm | validation |
|---|---|
| P_E | 0.393 (a jump at ~150 epochs, from 0.09 to 0.30) |
| P_odo | 0.561 |
| R1 | 0.113 |

**Wave 3: loss "both", every arm** (seed 3693; P_E and R1 also at 3694; 600 epochs). Final validation, then the epoch
after which every later point is within 0.01 of the final one:

| arm | validation | plateau epoch |
|---|---|---|
| P_odo | 0.537 | 375 |
| R4 | 0.444 | 575 |
| GPS | 0.365 | 525 |
| P_E | 0.311 / 0.320 (s3694) | 550 |
| R1 | 0.170 / 0.135 (s3694) | 475 |
| P_raw | 0.157 | 550 |
| GRU | 0.150 | 525 |

- P_E jumped at ~250 epochs (s3693) and ~175 (s3694).
- **Rule 14:** the installed E map, frozen, reaches 0.283; installed and trainable, 0.376 (scratch P_E 0.311). The
  class holds a solution near scratch's level, and training from it ends a little higher.
- **P_odo far below the gate's odometry integrator (0.93).** At this size and budget the one-layer retrieval, not the
  position input, limits every arm. This is a property of the model class, not of the step input.

**Pilot reading (outside seeds, validation split; NOT results):**
- P_E is above R1 by ~0.15-0.28 (three P_E and three R1 pilots).
- R4 is ABOVE P_E (0.44 vs 0.31, one seed): four index layers retrieve better than one path layer here. S1 may well
  read 'R4 ABOVE P_E'. The registration does not change because of it, and the prediction "R4 a little above R1"
  written before the pilots is wrong on this pilot.
- Pilot sds: P_E 0.04, R1 0.03 (n = 3 each).
- Timing on a 4090 shared by 2-3 jobs: 2.5-3.8 s per epoch at 1 layer, 6.5 s for R4 and GRU. The probe adds ~10 min
  (578 s); scoring takes seconds.

**Probe calibration (model-free constant baseline, training-set means, eval steps 500-1000).** Walls 78.19%,
Objects 65.10 as the mean squared distance per object (per-coordinate MSE 32.55), against the published 78.3 / 64.8.
**The Objects metric is the squared distance averaged over objects** (Amendment 1). Pilot P_E on the validation
slice: Walls 78.37 (constant 78.41, no headroom, as published); Objects 36.48 blinded / 36.60 unblinded (constant
65.63).

## Amendment 1 (2026-10-27, after the pilots, before launch; pilot-informed changes, disclosed)

1. **Loss = "both"** (sighted mean + post-blind mean) for every arm, replacing sighted-only.
   - It is applied to every arm alike, and was chosen because the registered loss left the primary readout near the
     floor for every arm with a learned or index position.
   - It also opened the P_E - R1 gap in the pilots (from ~0 to +0.15..0.28). **The primary contrast is therefore
     partly pilot-informed**, and every report of it must say so.
2. **Budget E = 600 epochs** (wave-3 plateaus 375-575).
3. **Objects metric = squared distance averaged over objects**, the definition that reproduces the published constant
   baseline. The per-coordinate MSE is still recorded.
4. **Prep ran on the workstation.** Arrays verified byte-identical on the cluster (16/16 md5).
5. **The held-split floor** (`mm_floors.py` = gate v3 machinery on train-part0 1600-2899, fits on 300-599) is
   **0.0472** (F3 m4 k0, pix16). There, class E puts 0.641 within 0.5 cell, with 5.31 non-local returns per episode.
   The eval floor is 0.0472 (gate v3 diagnostic).
6. Predictions as written before the pilots stand; the R4 one is contradicted by the pilot (see above).
7. **Cost (measured):**
   - On an A40 alone, about 1.2 s per epoch at 1 layer and about 3 s for R4 and GRU (the 4090 synthetic benchmark;
     the A40 is similar or a little slower). Plus ~10 min of probe.
   - Per run: ~25 min at 1 layer, ~40-60 min for R4 and GRU.
   - 56 runs: ~30-40 A40-h. As one array of 56 tasks with 8-h limits, wall time is set by queue access.
   - P_raw is included (cost allows).

## Amendment 2 (2026-10-27, after the independent code check, before launch; no batch result read)

Source: `docs/audits/2026-10-27/memory_maze_audit.md` (no BLOCKER; MAJOR M1-M3; MINOR m1-m8). Nothing here changes the
primary arm pair, metric, label rule or threshold. One descriptive arm is added (item 2), and the code changed in ways
listed in item 6, so **the batch runs a code state that no pilot ran** (item 6).

1. **What the primary measures (M1).** In the pilots the READOUT, not the step input, limits every arm:
   - P_odo reached 0.54 against the odometry integrator's 0.93.
   - The installed exact class-E map scored 0.28 frozen and 0.38 trainable, against scratch P_E's 0.31 and the
     class-E integrator's 0.66. So a frozen exact E map is read out at less than half of what its phase supports.
   - P_E - R1 is therefore read as "how much ONE learned retrieval layer gains from a path phase over an index phase
     with the same inputs", not as the accuracy of integration.
   - The scratch P_E solution need not be class E (frozen E is below scratch).
   - Rule 14 is met only weakly: "the class holds a solution at scratch's level". The install's weight scale was not
     varied.
2. **New DESCRIPTIVE arm P_E_frozen** (seeds 3600-3607; no verdict).
   - It is P_E with the least-squares E map (train-part0 0-299) installed FROZEN: W_in, W_out and omega fixed.
   - Its trainable parameter count equals R1's exactly (304,128 vs 304,128; `test_mm` checks it).
   - So P_E_frozen - R1 is the cleanest "fixed path phase vs index phase at identical trainable parameters" reading.
     `analyze_mm` prints it as a DESCRIPTIVE line.
   - Task table: 64 lines, md5 b448be6619378ad0eff8dce20fd79862.
3. **Scope of a PATH WINS (M2), printed beside the primary by `analyze_mm`.**
   - It is a statement about one layer at E = 600, budget-scoped.
   - R4 has 2.95x P_E's parameters (898,944 vs 305,120), so S1 tests depth AND parameters together.
   - If S1 reads 'R4 ABOVE P_E', the house exchange rate "one path layer ~ three attention layers" (Dyck, H3) does not
     hold on this task. Every summary states it beside the primary, not only among failures.
   - **Registered predictions already contradicted by the pilots** (registration unchanged):

     | prediction | pilot |
     |---|---|
     | R4 "a little above R1" | above P_E, 0.44 vs 0.31 |
     | P_E "near the class-E integrator 0.66" | 0.31-0.39 |
     | P_odo "near 0.93" | 0.54-0.56 |

4. **The pilot-informed loss in every citation.**
   - Every printed primary and HELD label carries "(loss pilot-informed, Amendment 1)".
   - The fresh seeds and the untouched eval and held splits make the batch a CONFIRMATION of the pilot finding under
     the amended loss, not a test of the original registration, whose loss failed for every arm (wave 1).
5. **Convergence (M3).** Under cosine-to-zero the final-10% training-loss slope is flat by construction. DESCENDING is
   now read on the slope over **60-85% of training**: the linear fit's change across the window < -3% of the window's
   mean loss.
   - Validation on the 17 pilot curves: the training loss falls 1.9-5.6% across the window in EVERY pilot, while the
     validation retrieval moves -0.006 to +0.033 there, and -0.011 to +0.015 after 85% (about 1 SE).
   - At -3% the rule flags R4 (-5.6%), GRU (-4.0%), P_E s3694 (-3.4%), P_odo s3692 (-3.2%) and GPS (-3.1%). Their
     validation change after 85% was +0.015, +0.015, -0.002, -0.006, +0.013.
   - So it is a weak readout of whether retrieval is still improving. **Every secondary is declared budget-scoped at
     E = 600**, and the DESCENDING counts are descriptive.
6. **Provenance and precision (m1, m2, m3).**
   - Code md5s are now recorded at the START of each run (`config.md5_start`) as well as the end (`md5`); a change
     during a run is recorded (`md5_changed`), and `analyze_mm` refuses such runs.
   - **The pilots span FOUR code states:**

     | state | md5 at end of run |
     |---|---|
     | wave 1 (no loss option; its configs lack `loss_steps`) | `train_mm.py` d8e8c3 (hashed at the end, after the option was added) |
     | wave 2 and two wave-3 runs | d8e8c3 |
     | the other wave-3 runs | 1909c1 |
     | the batch code checked by the audit | f47340 |

     They differ only in the loss option, the epoch and loss defaults and a logging fix. That cannot be verified from
     git, because the files first appear in commit 8ebaba5.
   - **The batch now runs a fifth state.** `train_mm.py` 5c6ba9, `model_mm.py` d9da04, `mm_data.py` 396fd0,
     `model.py` 0a379a. Changes since the audit:

     | change | from audit item |
     |---|---|
     | TF32 OFF for matmuls and cuDNN (the GRU arm is now fp32; the pilots' GRU ran TF32); cuDNN RNN/conv precision set to 'ieee' explicitly | m2 |
     | step features stored fp32 (were fp16; the heading entries were rounded ~5e-4) | m3 |
     | the P_E_frozen arm | M1 |
     | GRU probe on its 184-d hidden state; Table E.3 epsilons; 500 scored probe steps | m6 |
     | md5 at start | m1 |
     | the cuDNN RNN precision is recorded from torch's C++ state (building `nn.GRU` imports `torch.backends.cudnn.rnn`, which shadows the Python attribute but not the setting; verified 'ieee' in a GRU run) | m2 |
     | docstrings | m3 |

   - None changes the forward pass of a pilot arm except through precision.
   - On GPU the fp32 SDPA backward is not bitwise deterministic, so a resumed run is a statistical, not bitwise,
     continuation (CPU resume is exact, `test_mm`).
7. **Analysis guards (m4, m5).** `analyze_mm.load` REFUSES:
   - any seed outside 3600-3607;
   - any run whose config differs from the registered one (epochs 600, loss both, install none / frozen for
     P_E_frozen, train 0-1600, held 1600-2900, eval 1000, batch 16, lr 3e-4, wd 0.05, probe 10k steps, not pilot, no
     validation);
   - more than one md5 set, a run whose code changed, or mixed torch versions. The torch version is recorded.
   - A significant |d| < 0.05 prints "[below the 0.05 bar]" beside UNMEASURED.
   - The selftest now also covers: d >= bar with p >= .05, INCOMPLETE via P_odo, the HELD replicate, `best_floor` on
     the real floor JSONs, `load()` acceptance and five refusals, and an exact (failable) Holm check. Holm is now
     inlined in `analyze_mm` (identical to `analyze_lrsweep.holm`; checked).
8. **Probe (m6).** It is a reading beside the published rows, not a replication. Deviations from Pasukonis et al.:
   - 1600 training episodes (1.6M states, 1600 layouts) against their 30M-step dataset;
   - 10k post hoc steps on frozen states against a probe trained alongside 1M model steps.

   Matched since this amendment: Table E.3 epsilons (Adam 1e-5, LayerNorm 1e-3), the GRU probed at its hidden state,
   and the second half as 500 steps (501-1000).
9. **Cost (m7).** On A40s, ~1.5-2.5 s per epoch at 1 layer and 4-7 s for R4 and GRU, plus scoring and 10-20 min of
   probe.
   - 56 runs ~35-55 A40-h; the 8 P_E_frozen runs add ~4-6.
   - **Total ~40-60 A40-h**, every run far inside the 8-h limit.
10. **Dropped floor (m8).** The function-identical twin noise floor named in gate v3's design is not run (rule 1). The
    8-seed sd covers seed noise; this is disclosed, not fixed.
11. **Fixed traps.**
    - `mm_encoder.py prep` now refuses to embed anything else as "eval" if `eval.zip` is missing (the arrays were
      built with eval present; `eval_info.json` names the 2022-09-20 run).
    - `mm_encoder.py`'s md5 in the arrays' MANIFEST predates this guard.
12. **Checks after this amendment.**
    - `test_mm` ALL PASS on the workstation (torch 2.10, CPU) and on the cluster (torch 2.13, CPU job).
    - `analyze_mm --selftest` ALL PASS on both.
    - A 2-epoch GPU smoke on the real arrays with the amended code, workstation 4090 (torch 2.10), PASSES for P_E,
      GRU and P_E_frozen:
      - configs as intended: install frozen for P_E_frozen; trainable 304,128 = R1's; TF32 off with the RNN 'ieee';
        no code change during the run;
      - 95 eval and 81 held non-local returns scored on 16 episodes each; probe ran; peak 0.72-0.82 GiB.
    - The same smoke on the cluster (A40/A100/L40, torch 2.13) is queued; its result is checked before launch.
