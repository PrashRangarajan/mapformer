# Memory Maze CPU gate v2, BLIND STRETCHES: registration (written 2026-10-27, before any v2 number)

Follows `MEMORY_MAZE_GATE.md` (v1). v1's results (`docs/audits/2026-10-26/memory_maze_gate/`): S1 PASSES, so a 15x15
maze is large enough: open cells median 145, 99.3% of mazes >= 128; loop-closing re-entries with gap >= 128 are 73% of
scored steps; gap median 229, quartiles 119 / 402. S2 FIRED: a linear continuation from the two previous frames scores
0.89 under the pose-bin readout R1, so with every frame shown the view at a re-entry is predictable without a map. S3
FIRED: the 8-bin action x heading table dead-reckons only 13% of long-gap re-entries to within 0.5 cell. Descriptive
results: the previous-action x action map with continuous heading (E) reached 55%, and continuous heading alone (C) 29%.

v2 tests the user's two remedies, on the data already downloaded (15x15 eval, 1000 episodes; fits on the first 300
episodes of 15x15 `train-part0`; never eval episodes): (1) **blind stretches**, where observations are withheld for a
stretch of the walk so the view on re-entering a place cannot be continued from the last frames; (2) a **richer step
input** for the inverted design. Code: `mm_gate_v2.py`, which imports v1's helpers. Cluster job: `hyak/mm_gate_v2.slurm`.
Results go to `docs/audits/2026-10-27/memory_maze_gate_v2/`. CPU only. Definitions not restated here (cell, heading bin,
pose bin, backtrack, feature spaces `pix16` / `pcaw64`, readouts R1 / R2) are v1's.

## (a) Blind-stretch protocol

- **Schedule (primary).** Each episode alternates sighted runs of S = 32 steps with blind runs of B = 32 steps. The
  phase is a random offset in [0, S + B), drawn per episode from a fixed seed (episode index). Frame 0 is always
  sighted. During a blind step the model receives the step input (action / heading / displacement, depending on the
  arm) and a blank frame token. **Descriptive grid:** B in {16, 32, 64} x S in {16, 32}. Every measure is reported for
  each cell; the stop rule reads only the primary cell (B 32, S 32).
- **Why B = 32.** The agent moves about 0.2 cell per step (v1: forward-type steps 0.24, 92% of actions), so 32 blind
  steps carry it about 6 cells. That is far beyond the one-step ego-motion that the linear continuation exploited in
  v1. 32 is also well below the gap quartile 119, so most earlier visits to the re-entered place come before the
  stretch rather than inside it. Doubling to 64 halves the sighted share of the walk at S = 32, and with it the chance
  that the earlier visit was seen; that cell is in the grid. **Why S = 32.** Half of all frames stay visible, so a map
  built from sighted frames covers most places; the S = 16 cells show the cost of seeing less.
- **Scored step (blind re-entry).** The first sighted step t after a blind run [b0, t - 1] whose pose bin was SIGHTED
  at some s < b0, and whose pose bin differs from that of the last sighted frame b0 - 1 (otherwise copying the last
  frame answers it). Gap = t - (most recent sighted s < b0 in that bin). **Primary stratum: non-local** = gap >= 128
  and not a backtrack, as in v1. Strata "all" and "long gap" are reported too.
- **Target and readouts.** The frame f_t, predicted from inputs that end before f_t: sighted frames before b0, blank
  tokens and step inputs through t. Scored under R1 (registered): the nearest of the episode's 1001 frames to the
  prediction has the pose bin of t, so earlier same-bin frames count as positives. Also under R2: f_t must be nearer
  than every frame in a different pose bin, so same-bin frames are removed from the negatives. As in v1, the R1 pose
  oracle is 1.0 by construction.

## (b) Baselines, in two classes

**Map-free floors.** These do not integrate the step input into a position and keep no pose memory. They read only
sighted frames and the action sequence. Each predicts f_t:
- F1 last-sighted copy: f_{b0-1}.
- F2 blind continuation: ridge regression from [f_{b0-1}, f_{b0-2}, counts of each action over the blind run and t,
  1] to f_t, in `pix16`, fitted on train episodes with the same B. This is a lower bound on a learned continuation; a
  CNN could do better.
- F3 view recognition on sighted frames: the earlier sighted step s that best matches the last sighted frame (m = 1)
  or the last four (m = 4), with s <= b0 - 1 - k (k = 0, 16). It predicts f_{s + (t - b0 + 1)}, the frame seen the
  same number of steps after the matched one, which is a map-free "same route again". s is restricted to steps whose
  predicted frame was itself sighted and comes before b0. Four variants.
- The floor is the best of F2 and the four F3 variants (F1 is a check), in the better feature space.

**Integrators (not floors).** These integrate a step input into a position and retrieve by it, so they already do
what the path model is supposed to do. They are the ceiling for each step-input class, a rule-14 existence check, and
never count as floors:
- I_V for each step class V: P_hat = cumsum of V's predicted displacement, over the whole episode. Among sighted u < b0
  in the same heading bin as t (heading is given in the inverted design), take the u with the nearest P_hat and
  predict f_u. Scored under R1 / R2.
- I_odo: the same with the true displacement (given odometry, survey arm P). Pose oracle: the most recent sighted
  frame in the same pose bin (R1 = 1.0; R2 reported).

## (c) Step-input classes (inverted design), on blind re-entries

All classes are fitted on train and evaluated on eval:
- A8: table (action x 8 heading bins), v1's S3 class.
- A32: table (action x 32 heading bins), a finer heading token.
- E8: table (previous action x action x 8 heading bins).
- C: per-action linear map of the continuous heading vectors [dir_{t-1}, dir_t].
- E: C per (previous action, action), with continuous heading. This is "E with continuous heading".

Reported for each class:
- **full-gap dead-reckoning error**: |(P_hat_t - P_hat_s) - (p_t - p_s)| from the earlier visit s to t, the error a
  cumsum phase carries. The share under 0.5 cell is the existence measure.
- blind-only error (b0 - 1 to t), descriptive.
- the integrator retrieval accuracy I_V under R1.

## (d) STOP RULE (fixed before any v2 number; primary cell B 32, S 32; 15x15)

S1 of v1 passed and is taken as given.

- **S2v2, a map is still not needed.** STOP if the best map-free floor (F2 or F3) under R1 on non-local blind
  re-entries is >= 0.85, i.e. within Y = 0.15 of the R1 ceiling 1.0. The reason is v1's: 0.150 is the only measured
  twin noise floor on a visual navigation task in this project.
- **S3v2, no step input dead-reckons.** STOP the inverted primary if NO class among A8, A32, E8, C and E puts >= 50% of
  non-local blind re-entries within 0.5 cell, using the full-gap error. Reason: past half a cell the phase match picks a
  neighbouring cell's view about as often as the right one. The full gap is the reckoning that matters, because a
  cumsum phase cannot be re-anchored by a view. If S3v2 passes, the primary step input is the PASSING class with the
  FEWEST parameters (order A8 < A32 < C < E8 < E). Every class that passes is reported.
- **S4v2, too few targets.** STOP if non-local blind re-entries average < 3 per eval episode. That would be 3k scored
  steps on the 1k-episode eval split and ~87k training targets on the 29k train episodes. With 3000 scored steps the
  binomial SE of an arm's accuracy is <= 0.009, small beside effects of 0.15 or more. Below 3 per episode, most
  episodes would carry 0-2 targets, the per-episode readout would be mostly empty, and training would rely on a
  handful of supervised re-entries per walk.
- **Pass** = none of S2v2, S3v2, S4v2 fires. Then the registration of the inverted design may be written (not run).

## Not measured

A trained CNN continuation over the blind gap; any trained model; frame-shuffle destruction; the twin noise floor.

## Design if it passes (design note added 2026-10-27 while the v2 job ran, before any v2 number was read; the stop rule above is unchanged)

User decision: report BOTH (1) the blind-stretch re-entry task as the registered primary and (2) the published Memory
Maze offline probe as a standard secondary, for comparability.

**(1) Primary: blind-stretch re-entry.**
- Protocol, scored steps and readouts as in (a) above, at the primary cell.
- Sequence [step_1, f_1, step_2, f_2, ...]; T = 2000 tokens per 1000-step episode; blank frame token on blind steps.
- Frozen frame encoder with embeddings precomputed once. It is trained on train-split frames only. To keep it from
  learning pose adjacency (review A-M1) it is an autoencoder, not time-contrastive. Its single-frame pose leak is
  re-measured on eval frames before registration.
- Loss: InfoNCE on the next-frame embedding at sighted steps, with negatives from the same episode. Same-pose-bin frames
  are not negatives.
- Arms, n = 8, one batch (rule 12), rank 4 per head, AdamW wd 0.05:
  - P_sel: path model, 1 layer, step map on the S3v2-selected step input (the passing class with the fewest
    parameters);
  - P_odo: given-odometry ceiling, a step map on the true displacement;
  - P_raw: raw egocentric actions with no heading; replicates the rotation negative; secondary;
  - R1 and R4: index RoPE, 1 and 4 layers, same inputs as tokens;
  - GRU: same inputs; integrates by construction (review A-M2);
  - GPS-PE: index attention given cumsum(true displacement) as an absolute sinusoidal position code; separates "the
    position code" from "path integration".
- Primary contrast: P_sel - R1 on non-local blind re-entries under R1. Secondaries: P_sel - R4, P_sel - GRU,
  P_odo - P_sel (the cost of learning the step), P_sel - GPS-PE.
- Floors beside every number: the v2 map-free floors, plus the trained-model floors (frame-shuffle destruction and a
  function-identical twin).

**(2) Secondary: Memory Maze offline probe.** Protocol to copy, read first-hand in Pasukonis et al. 2022 (2210.13383,
Sec. 3.3, Table 4, App. D, Table E.3) and Deng et al. 2023 (2307.02064, App. D, Table 6):
- **Targets.** Walls: `maze_layout` (15x15 binary), BCE loss, metric = accuracy averaged over all tiles. Objects:
  **`targets_vec`** (agent-centric object positions, (6, 2) at 15x15), NOT `targets_pos`; MSE loss, metric = MSE
  averaged over objects. Both fields are in the npz.
- **Probe input.** The model's representation at step t, concatenated with the agent's position and orientation
  (`agent_pos`, `agent_dir`). Pasukonis et al. add these because the wall target is in global grid coordinates.
- **Probe network.** MLP with 4 hidden layers of 1024 units, LayerNorm and ELU after each (Pasukonis; Deng used SiLU).
  It is trained on the frozen representation, with no gradient into the model.
- **Splits and steps.** The probe is trained on train-split episodes and evaluated on the 1k eval episodes. The score
  is averaged over the second half (steps 500-1000) of each eval trajectory.
- **References for 15x15** (Walls acc % / Objects MSE): constant baseline 78.3 / 64.8; VAE no-memory 78.7 / 56.2;
  supervised oracle 88.5 / 9.2; RSSM-TBTT 80.5 / 33.8 (Pasukonis) and 81.7 / 32.6 (Deng); TSSM-XL 79.8 / 35.1;
  S4WM 80.2 / 34.8; S5WM 81.8 / 25.3. The 9x9 rows are in the same tables, but our primary is 15x15.
  - Headroom on 15x15 Walls is 3.5 points above constant for the best model and 10 points to the supervised oracle. A
    Walls difference between our arms will likely sit inside the noise.
  - Objects (64.8 constant to 25.3 best) has room.
- **Comparability of the "memory state".** Attention has no single recurrent state vector. Deng et al. probe their
  transformer (TSSM-XL) with the output of the final block at step t, concatenated with the stochastic latent. Our
  closest equivalent, to be REGISTERED, is the final-layer output at the frame token of step t: d = 128 for a one-layer
  model, against their 2048-d states. A probe query token would change the model and is not used. For the GRU arm the
  representation is its hidden state.
  - The probe reads the same input stream the model was trained on (blinded). This is a deviation from the benchmark,
    whose models see every frame; the deviation is stated. As a secondary, the probe is also read on the unblinded
    stream, which is a distribution shift for blind-trained models.
  - Our P_odo and GPS-PE arms receive odometry or position as input, and the probe protocol itself hands the probe
    the agent's pose. Their Walls scores are therefore not "memory from pixels". Compare them only within our arms;
    only the published rows are external context.
- **Cost of (2).** One probe per trained run, on subsampled train states (e.g. 64 steps x 29k episodes): about 0.2-0.4
  A40-h per run.

**Cost (cluster preemptible partition, A40; workstation GPUs not used).**
- Encoder training and embedding precompute: ~2 A40-h. Note that only `train-part0` and eval are downloaded so far;
  the remaining 9 train archives (~90 GB) are needed for full training.
- 7 arms x 8 seeds = 56 runs at ~1-2 A40-h each (T = 2000, precomputed embeddings): 56-112 A40-h, plus probes
  ~11-22 A40-h.
- This is 9-17 array windows of 8 h, at several tasks in parallel, with per-epoch resumable checkpoints.
- Trimmed variant: P_sel, P_odo, R1, GRU at n = 8 (32 runs, ~40-75 A40-h with probes). Its power is set by the
  per-episode count of non-local blind re-entries measured here.

## Result (2026-10-27; stop rule applied as written): STOP -- S3v2 and S4v2 fire at the primary cell; S2v2 passes

Output: `docs/audits/2026-10-27/memory_maze_gate_v2/gate_v2_15x15.{json,log}` (1000 eval episodes, fits on 300 train
episodes, 535 s on CPU). Primary cell B 32, S 32 (sighted share 0.50, 15.1 blind runs per episode):

| check | measured | threshold | verdict |
|---|---|---|---|
| S2v2 best map-free floor, R1, non-local blind re-entries | **0.022** (F3 m1 k0, pix16; CI 0.016-0.028); F2 0.004; F1 0.000 | >= 0.85 stops | passes: blinding removes the continuation floor (v1: 0.89) |
| S3v2 full-gap dead-reckoning share < 0.5 cell (non-local) | A8 0.121, A32 0.204, C 0.240, E8 0.148, **E 0.490** | some class >= 0.50 | **FIRES** (E misses by 0.010; binomial SE ~0.010 on 2441 steps, so a knife edge) |
| S4v2 non-local blind re-entries per eval episode | **2.44** (6.7% of episodes have none) | >= 3 | **FIRES** |

Integrators (not floors), R1 on non-local blind re-entries at the primary cell: given odometry 0.929, E 0.637, C 0.522,
A32 0.511, E8 0.474, A8 0.433. The pose-oracle R2 is 0.103.

Descriptive grid (decides nothing):

| cell (B, S) | non-local per episode | best floor R1 | E share < 0.5 cell | I_E R1 | I_odo R1 |
|---|---|---|---|---|---|
| 16, 16 | 5.13 | 0.047 | 0.530 | 0.671 | 0.927 |
| 16, 32 | 4.21 | 0.085 | 0.537 | 0.645 | 0.922 |
| 32, 16 | 2.67 | 0.030 | 0.539 | 0.678 | 0.936 |
| 32, 32 | 2.44 | 0.022 | 0.490 | 0.637 | 0.929 |
| 64, 16 | 1.01 | 0.025 | 0.521 | 0.706 | 0.920 |
| 64, 32 | 1.23 | 0.021 | 0.516 | 0.651 | 0.915 |

Error accumulated over the blind run alone, for E: < 0.5 cell on 0.941 of non-local re-entries (median 0.095 cell). E's
failures therefore come from the full gap (median 301 steps), not from the blind run.

Reading:
- Blinding does what it was meant to do: no map-free floor reaches 0.09 in any cell.
- The registered primary still stops, for two reasons.
  - Too few targets. Only ~19% of blind runs end on a sighted pose bin that was seen before.
  - No linear step class reliably carries a cumsum phase across ~300-step gaps. Momentum-aware E comes closest.
- The B 16 cells would pass all three checks (E 0.530 / 0.537; 5.1 / 4.2 per episode; floors <= 0.085). Choosing one
  now would be post hoc and needs a fresh registration.
