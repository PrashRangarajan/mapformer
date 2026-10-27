# Memory Maze CPU gate v3: the 16-blind-step cell on fresh episodes (registration, written 2026-10-27 before any v3 number)

Follows `MEMORY_MAZE_GATE.md` (v1) and `MEMORY_MAZE_GATE_V2.md` (v2). v2 STOPPED at its primary cell (B 32 blind,
S 32 sighted): S3v2 fired (best class E 0.490 within 0.5 cell, one SE below 0.50) and S4v2 fired (2.44 non-local blind
re-entries per episode). Its descriptive grid showed both 16-blind-step cells passing all three checks. v3 registers one
of them and re-tests it on episodes no earlier gate has touched. Code: `mm_gate_v3.py`, which wraps
`mm_gate_v2.py` / `mm_gate.py` unchanged. Cluster job: `hyak/mm_gate_v3.slurm`. Results:
`docs/audits/2026-10-27/memory_maze_gate_v3/`. CPU only.

## Post hoc selection, disclosed

The cell was chosen AFTER seeing v2's grid on the 15x15 eval split, so v2's numbers for it are selected and are not
evidence. v3 is the test. It scores only fresh episodes and fits on a disjoint fresh set.

## The registered cell: B = 16 blind steps, S = 16 sighted steps (period 32, half the frames sighted)

Of the two 16-blind cells, (16, 16) is chosen over (16, 32), on v2's grid:
- (16, 16) had more targets: 5.13 vs 4.21 non-local re-entries per episode, so S4's margin is larger.
- (16, 16) had a lower map-free floor: 0.047 vs 0.085 under R1.
- The E share was about the same: 0.530 vs 0.537.
- (16, 16) keeps v2's sighted share of 0.50.
- Risks this choice carries:
  - 16 blind steps carry the agent only about 3 cells, so local continuation is less excluded than at 32. The S2 check
    measures that directly.
  - A short sighted run (16) gives the map fewer views. v2's oracle and integrator numbers at (16, 16) did not suffer
    from it.

Blinding schedule as in v2: per-episode random phase offset (seeded by the episode's index in the archive), frame 0
sighted, scored step = the first sighted step after a full-length blind run whose pose bin was sighted before the run
and differs from the last sighted frame's. Primary stratum: non-local (gap >= 128, not a backtrack). Readouts R1
(registered) and R2, floors F1-F3, integrators, and the step classes A8, A32, C, E8 and E are all as defined in v2.

## Fresh episodes (no overlap with any earlier fit or score)

The 15x15 `train-part0` archive holds 2900 episodes; indices are positions in its sorted member list.
- **Already used:** v1 and v2 fitted on indices 0-299 of `train-part0` (`--n-train 300`, the first 300 sorted
  members). Both scored only the eval archive. The 9x9 eval archive was scored by v1, descriptively.
- **v3 fit set:** `train-part0` indices **300-599** (300 episodes). This is the PCA, the blind-continuation ridge for
  B 16, and all step tables and maps.
- **v3 score set:** `train-part0` indices **600-1599** (1000 episodes), disjoint from the fit set and from everything
  earlier.
- Indices 1600-2899 stay untouched for any later check.

## Checks and thresholds (same as v2; S1 recomputed)

- **S1v3, extent and locality** (v1's rule, on the 1000 score episodes, all frames sighted as in v1). STOP if fewer
  than half the mazes have >= 128 open cells, OR loop-closing re-entries with gap >= 128 are < 10% of scored steps.
- **S2v3, a map is not needed.** STOP if the best map-free floor (F2 or the four F3 variants, better feature space)
  under R1 on non-local blind re-entries is >= 0.85.
- **S3v3, a step input that dead-reckons exists.** STOP the inverted primary unless some class among A8, A32, C, E8
  and E puts >= 0.50 of non-local blind re-entries within 0.5 cell over the full gap. The primary step input is the
  passing class with the fewest parameters (order A8 < A32 < C < E8 < E).
- **S4v3, enough targets.** STOP if non-local blind re-entries average < 3 per score episode.

**Within one SE of a threshold counts as FIRES (conservative; fixed now).** A check passes only if it clears its
threshold by more than one standard error, on the passing side:
- S2v3 passes iff floor + SE < 0.85.
- S3v3 passes for a class iff share - SE >= 0.50.
- S4v3 passes iff mean - SE >= 3.
- S1v3 (a) passes iff share_ge128 - SE >= 0.50; (b) passes iff share - SE >= 0.10.

**SE** = the larger of the binomial SE, sqrt(p(1 - p)/n) over the pooled scored steps (or mazes for S1a), and the
episode-bootstrap SE (1000 resamples of episodes; it covers the clustering of steps within an episode). For S4 it is the
SD of the per-episode count / sqrt(1000). Each check reports value, SE and margin. A pass needs S1v3, S2v3, S3v3 and S4v3
all to pass.

## If it passes: design carried to the pre-registration (not run here)

As written in `MEMORY_MAZE_GATE_V2.md`, "Design if it passes", with the cell (16, 16) and the S3v3-selected step class.
Its main points:
- **Primary:** blind re-entry retrieval under R1 on non-local returns.
- **Secondary:** the published offline probe. The Objects target is **`targets_vec`** (agent-centric), not
  `targets_pos`; Walls is `maze_layout`. The probe is a 4x1024 MLP with LayerNorm and ELU on the frozen
  representation plus the agent's pose, scored over steps 500-1000 of the eval episodes, with the published 15x15
  references.
- **Arms:** path with the passing class, odometry ceiling, index RoPE 1 and 4 layers, GRU/LSTM, GPS-PE control.
- **Recurrent arm caveat:** `RNN_TORUS_RESULTS.md` notes the recurrent networks there were trained on ONE map and
  memorised it. Memory Maze draws a new maze per episode, so this arm is the many-map test that batch did not run. Its
  result is not comparable to the RNN_TORUS cells.

## Result (2026-10-27; rule applied as written): PASS -- all four checks clear their thresholds by more than one SE

Output: `docs/audits/2026-10-27/memory_maze_gate_v3/gate_v3_15x15.{json,log}`. Fit on train-part0 300-599 (300
episodes), scored on 600-1599 (1000 episodes); 17 min on CPU. The registered run used `mm_gate_v3.py` as first
written. The `--score-zip` option was added AFTERWARDS for the diagnostic below; the default path is unchanged.

| check | value | SE | threshold | margin past one SE | verdict |
|---|---|---|---|---|---|
| S1v3 (a) mazes with >= 128 open cells | 0.985 (median 146) | 0.004 | 0.50 | +0.481 | passes |
| S1v3 (b) loop-closing re-entries, gap >= 128, all frames sighted | 0.759 | 0.003 | 0.10 | +0.656 | passes |
| S2v3 best map-free floor, R1, non-local blind re-entries | 0.048 (F3 m4 k0, pix16; F2 0.005; F1 0) | 0.003 | < 0.85 | +0.799 | passes |
| S3v3 full-gap share within 0.5 cell, non-local | **E 0.605**; C 0.318, A32 0.244, E8 0.165, A8 0.123 | 0.011 (E) | 0.50 | +0.094 (E only) | passes: **E** is the primary step input |
| S4v3 non-local blind re-entries per episode | 5.36 (0.5% of episodes have none) | 0.07 | 3 | +2.29 | passes |

Integrators (not floors), R1 on non-local blind re-entries:

| odometry | E | C | A32 | E8 | A8 |
|---|---|---|---|---|---|
| 0.924 | 0.696 | 0.600 | 0.562 | 0.498 | 0.457 |

Within the blind run alone, E keeps 0.978 of returns within 0.5 cell (median 0.05 cell). The gap from the earlier visit
has median 307 steps.

**Post hoc diagnostic (not registered; decides nothing).** Same v3 fits, scored on the official 15x15 eval split
(`diag_eval_split.{json,log}`):
- E 0.514 +/- 0.011. It clears 0.50 by only 0.002 past one SE.
- Map-free floor 0.047; 5.13 returns per episode; I_E 0.659; I_odo 0.927.

So the train-split vs eval-split difference in E (0.605 vs 0.514, about 6 SE) is the SPLIT, not the fit set. v2's
eval-split value at this cell, with fits on train-part0 0-299, was 0.530.

The two splits are separate generation runs:
- eval members are dated 2022-09-20, train-part0 2022-09-23;
- forward-type collisions are 1.02% of forward-type steps in eval vs 0.57% in train;
- the action mixes differ slightly.

Consequence for the design: on the official eval split, the linear E class sits at the existence threshold. A model's
learned step map may do better or worse than the linear class. The pre-registration must choose where the primary is
scored (see the design below) and must say this.

## Concrete design for the pre-registration (gate passed; nothing here is run)

**Task (primary).** Blind re-entry retrieval at cell (B 16, S 16), scored under R1 on non-local returns: gap >= 128
since the last sighted visit to the pose bin, not a backtrack. Secondary readouts: R2, and the all-returns stratum.
- Input: sequence [step_1, f_1, step_2, f_2, ...], T = 2000 tokens per 1000-step episode. Blind steps carry a learned
  blank frame token.
- Frames: a frozen convolutional autoencoder, trained on train-split frames only. It is not time-contrastive, to avoid
  learning pose adjacency (review A-M1). Its 256-d embeddings are precomputed once in fp16. Its single-frame pose leak
  is measured on eval frames before registration; the v1 PCA probe found heading 0.206 vs majority 0.209 and cell
  0.015 vs uniform 0.004.
- Loss: InfoNCE on the next-frame embedding at every sighted step. Negatives come from the same episode; same-pose-bin
  frames are excluded from the negatives.

**Where it is scored: an open decision, to be fixed in the registration.**
- Option 1, the official eval split. The benchmark's split and the probe's. There the linear E class is at the
  existence threshold (0.514, post hoc).
- Option 2, held-out train-split episodes. For example parts not used for training, or train-part0 1600-2899. This is
  where the gate passed cleanly (0.605).
- Recommendation: the official eval split as the primary, with the held-out train-split episodes as a declared
  secondary. The registration says in advance that the eval split was generated separately, with about twice the
  collisions.

**Arms** (n = 8, one batch, rule 12; one layer unless stated; rank 4 per head with AdamW weight decay 0.05, the RANKWD
condition; cosine schedule):

| arm | step / position input | role |
|---|---|---|
| P_E | step map on class E: one-hot (a_{t-1}, a_t) (36) times [dir_{t-1}, dir_t] (4) = 144-d, through the low-rank W_out W_in; frames enter content only (ActOnly) | primary path arm. Heading is given (a compass): privileged, and said so |
| P_odo | step map on the true displacement (2-d) | odometry ceiling, in the class of GTA / PRoPE (given poses) |
| R1 | index RoPE, 1 layer; same 144-d step feature and frame embedding as tokens | primary comparator |
| R4 | index RoPE, 4 layers | depth control |
| GRU | 1-2 layer GRU at matched parameters, same inputs | recurrent control |
| GPS-PE | index attention plus a sinusoidal absolute code of cumsum(true displacement) | separates "a position code" from "path integration"; compare with P_odo |
| (P_raw) | raw actions only, no heading | optional secondary: replicates the rotation negative |

- **Contrasts.** Primary: P_E - R1, with the house statistics (permutation / Fisher on per-seed accuracy; SOLVED
  classes). Secondaries: P_E - R4, P_E - GRU, P_odo - P_E (the cost of learning the step), P_odo - GPS-PE.
- **GRU caveat.** `RNN_TORUS_RESULTS.md`'s recurrent networks were trained on ONE map and memorised it. Memory Maze
  draws a new maze every episode, so this GRU is the many-map test that batch did not run. A GRU win here does not
  contradict RNN_TORUS, and a GRU loss is not a replication of it.
- **Floors beside every number:** the gate's map-free floors (0.048 on fresh train-split episodes, 0.047 on eval),
  trained-model frame-shuffle and step-shuffle destruction, and a function-identical twin noise floor.
- **Rule 14 before launch:** install the fitted linear E map as P_E's step map, frozen and trainable, at the trained
  weight scale. Check that the class holds a solution: the gate's I_E ceiling is 0.66-0.70 under R1.
- **Rule 29:** pilots on seeds outside the batch; an independent code-verification agent before any result is read.

**Standard secondary: the Memory Maze offline probe** (protocol in `MEMORY_MAZE_GATE_V2.md`, read first-hand):
- Targets: Walls = `maze_layout` (accuracy over tiles, BCE). Objects = **`targets_vec`** (agent-centric, MSE averaged
  over objects).
- Probe: an MLP with 4 hidden layers of 1024 units, LayerNorm and ELU, trained on frozen representations of
  train-split states. Its input is the final-layer output at step t's frame token (REGISTERED choice; the analogue of
  Deng et al.'s TSSM-XL h_t), concatenated with `agent_pos` and `agent_dir`.
- Scored on eval-split steps 500-1000.
- 15x15 references: constant 78.3 / 64.8; RSSM-TBTT 80.5 / 33.8 (Deng 81.7 / 32.6); TSSM-XL 79.8 / 35.1; S4WM
  80.2 / 34.8; S5WM 81.8 / 25.3; supervised oracle 88.5 / 9.2.
- Walls has about 3.5 points of headroom over constant. The P_odo and GPS-PE arms are given position, so they are
  compared only within our arms.

**Data.**
- On the cluster: 15x15 train-part0 (2900 episodes) and eval (1000).
- Full training wants the other nine train archives (~90 GB), dependent on the Drive quota. Training on train-part0
  alone (2900 episodes x ~5.4 non-local targets plus all sighted-step predictions) is the fallback; say which.
- Gate episodes (train-part0 300-1599) may be used for training, since the gate is not the test. Under option 2 the
  test episodes must be excluded from training.

**Cost** (cluster preemptible A40 partition; workstation GPUs not used; numbers are estimates until a pilot measures
them):

| item | A40-h |
|---|---|
| encoder training plus embedding precompute | ~2 (embeddings ~15 GB fp16 for 30M frames, ~1.5 GB for train-part0 alone) |
| one run (T = 2000, d 128, 1 layer, precomputed embeddings) | ~1-2 |
| 6 arms x 8 = 48 runs | 48-96 |
| probes (~0.3 per run) | ~15 |
| total | ~65-115 |

- As array jobs of one run per task, with per-epoch resumable checkpoints and 8-9 h windows: 1-2 windows of wall time
  if ~48 GPUs are free, plus queue waits of 30-55 min per window.
- Trimmed variant: P_E, P_odo, R1 and GRU (32 runs, ~45-80 A40-h).
- Pilots on outside seeds first: ~4 runs, ~8 A40-h, to measure the real per-run cost before the batch is costed for
  the user.
