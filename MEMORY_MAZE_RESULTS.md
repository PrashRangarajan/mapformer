# Path integration of given self-motion on a first-person 3D benchmark: Memory Maze 15x15 -- results (MEMORY_MAZE, registered, `MEMORY_MAZE_PREREG.md` + Amendments 1-2)

Batch 2026-10-28 03:25 -> 07:06 (cluster time) on the cluster's GPUs (48 runs on A40, 16 on A100), code `d5c0ec4`, torch
2.13.0+cu130, Python 3.12, fp32 throughout (TF32 off for matmul and cuDNN; cuDNN RNN precision 'ieee', recorded per run).
64 runs = 8 arms x seeds 3600-3607, each with `DONE` and `results.json`; no resume, no error in any log. One code md5 set
at the start and at the end of every run (`train_mm.py` 5c6ba9, `model_mm.py` d9da04, `mm_data.py` 396fd0, `model.py`
0a379a, `lie_groups.py` 1deedf, `mm_gate.py` 36720d, `mm_gate_v2.py` f1cfb8; equal to the committed files),
`md5_changed` empty everywhere. Every config is the registered one (600 epochs, loss "both", batch 16, lr 3e-4, wd 0.05,
train 0-1599, held 1600-2899, eval 1000, probe 10k steps, not pilot, install "none", "frozen" for P_E_frozen).
Independent results-blind code check before launch (`docs/audits/2026-10-27/memory_maze_audit.md`: no BLOCKER, MAJOR
M1-M3, MINOR m1-m8, folded into Amendment 2). Pilots on seeds 3690-3694 (workstation, torch 2.10; never pooled). Measured
cost ~35 GPU-h (training + scoring + probe; estimate 40-60 A40-h). Registered output `MEMORY_MAZE_ANALYSIS.txt`,
`MEMORY_MAZE_VERDICTS.json`; run dirs outside the repo (results.json and logs only, fetched from the cluster).

**Checks done for this file (2026-10-28).** `analyze_mm` re-run read-only on the fetched runs: text identical, verdicts JSON
identical. Every number below re-derived independently from the 64 results.json with an own exact permutation over all
12,870 splits and an own window-slope rule (`docs/audits/2026-10-28/memory_maze/verify_numbers.py` ->
`verify_numbers_out.txt`): all match to the printed precision. P_E_frozen's trainable count is 304,128 = R1's
(305,120 total); R4 898,944 (2.95x P_E). Scored sets: 5,132 non-local blind re-entries on the eval split (5.13 per
episode; = the gate's diagnostic), 6,900 on the held split (= `floors_held`).

**Task, in one paragraph** (details in the pre-registration). Memory Maze 15x15 (Pasukonis et al. 2022), 1000-step
first-person episodes, a new maze every episode. Sequence [f_0, s_1, f_1, ..., s_1000, f_1000]: f_t a frozen
autoencoder embedding of the 64x64 frame, s_t the step token of action a_t. Frames are blinded in a 16 blind / 16 sighted
schedule; a blind frame token is a learned blank. Scored: the first sighted step after a full blind run whose pose bin
(cell x 8 headings) differs from the last sighted frame's and was sighted before; primary stratum non-local = that
earlier visit >= 128 steps back and not a backtrack (loop-closing re-entries). Readout R1: the episode frame whose key is
nearest the query at s_t has the pose bin of t. Loss InfoNCE over the episode's 1001 frames on sighted plus post-blind
steps (Amendment 1). One attention layer, 2 heads, d 128, 600 epochs, AdamW, warmup + cosine, dropout 0. **Heading is
GIVEN** (a compass in the step input). Arms: **P_E** path phase theta = omega cumsum(W_out W_in phi_t), rank 4, phi_t =
one-hot (a_{t-1}, a_t) x [dir_{t-1}, dir_t] (168-d), a learned step map; **P_odo** path phase from the true displacement
(given odometry; ceiling); **P_raw** path phase from the raw egocentric action, no heading; **R1 / R4** index RoPE at 1 / 4
layers, same tokens as P_E; **GRU** one layer, hidden 184, same tokens; **GPS** index RoPE plus an additive sinusoidal code
of the given position pos_t - pos_0; **P_E_frozen** (descriptive) P_E with the least-squares class-E map installed and
frozen. Frames enter content only; in the path arms only step tokens step the phase.

## Reading (registered)

**PRIMARY: PATH WINS (loss pilot-informed, Amendment 1).** One path-integrated attention layer whose phase integrates a
learned step map of the given actions and heading retrieves the view at blind long-gap loop-closing re-entries at
**0.331** against **0.118** for one index-RoPE layer given the same inputs as tokens: d **+0.213**, exact permutation p
0.00016 (the minimum at n = 8), MDE 0.093; complete separation (P_E's lowest seed 0.273, R1's highest 0.132). Registered
label, verbatim: "PATH WINS: P_E 0.3309 vs R1 0.1175, d +0.2133, perm p 0.00016, MDE 0.0928 (loss pilot-informed,
Amendment 1)". The HELD split (train-part0 1600-2899, the training run's distribution) replicates it: 0.382 vs 0.133,
d +0.249, p 0.00016 (descriptive). Floor on both splits 0.0472.

Scope note printed beside the primary by `analyze_mm` (Amendment 2), verbatim: "scope: one path layer vs one index layer
at E = 600, budget-scoped; the one-layer retrieval limits every arm (Amendment 2); S1 tests depth AND parameters (R4
2.95x); if S1 reads 'R4 ABOVE P_E', 'one path layer ~ three attention layers' does not hold here".

The loss disclosure (Amendment 1): under the loss as first registered (sighted steps only) every arm with a learned or
index position stayed near the floor in the pilots; the amended loss ("both") was chosen on pilot seeds and opened the
P_E - R1 gap there (+0.15 to +0.28). This batch, on fresh seeds and untouched eval and held splits, is a confirmation of
the pilot finding under the amended loss, not a test of the original registration.

**S1 P_E vs R4 (four index layers): UNMEASURED.** 0.331 vs **0.389**, d -0.059, Holm p 0.071 (raw 0.071), MDE 0.097.
R4 is numerically ABOVE P_E, as in the pilot (0.44 vs 0.31, one seed), and **8/8 R4 runs are still DESCENDING**
(budget-scoped at E = 600). R4 has 2.95x P_E's parameters, so S1 tests depth and parameters together. **The house
exchange rate "one path layer ~ three attention layers" (Dyck, H3) is not supported on this task**; four index layers
are not separated from one path layer and sit above it on average.

**S2 P_E vs GRU: P_E ABOVE GRU.** 0.331 vs 0.156, d +0.175, Holm p 0.00062. The GRU (one layer, hidden 184, same
tokens, matched core parameters) is trained on a new maze every episode: the many-map test RNN_TORUS did not run.

**S3 P_odo vs P_E: P_odo ABOVE P_E.** 0.539 vs 0.331, d +0.208, Holm p 0.00062: learning the step from actions and
heading costs 0.21 against given odometry.

**S4 P_odo vs GPS: P_odo ABOVE GPS.** 0.539 vs 0.297, d +0.241, Holm p 0.00062: given the same position information,
a relative rotary phase retrieves better than an additive absolute code of it.

All four secondaries are budget-scoped at E = 600 (Amendment 2.5).

## Per arm (eval = primary split; held = train-part0 1600-2899; R1 and R2 readouts; non-local blind re-entries)

| arm | position from | params | eval R1 (sd) | held R1 (sd) | eval R2 | held R2 | all returns R1 | DESCENDING (60-85%) | final-5% loss |
|---|---|---|---|---|---|---|---|---|---|
| P_odo (ceiling) | given displacement, path phase | 304,712 | **0.539** (0.023) | 0.546 (0.024) | 0.461 | 0.471 | 0.536 | 5/8 | 5.22 |
| R4 | index RoPE, 4 layers | 898,944 | 0.389 (0.027) | 0.417 (0.023) | 0.346 | 0.376 | 0.387 | **8/8** | 3.68 |
| **P_E** | learned step map, path phase | 305,120 | **0.331** (0.080) | 0.382 (0.096) | 0.205 | 0.242 | 0.340 | 1/8 | 5.66 |
| GPS | index RoPE + absolute code of given position | 312,704 | 0.297 (0.005) | 0.327 (0.010) | 0.281 | 0.309 | 0.299 | 1/8 | 4.75 |
| P_E_frozen (descriptive) | installed class-E map, frozen | 305,120 (304,128 trainable) | 0.242 (0.019) | 0.274 (0.021) | 0.142 | 0.167 | 0.247 | 0/8 | 5.97 |
| GRU | recurrent, same tokens | 302,864 | 0.156 (0.011) | 0.175 (0.013) | 0.131 | 0.147 | 0.155 | **8/8** | 5.42 |
| P_raw | raw egocentric action, no heading | 304,472 | 0.129 (0.034) | 0.142 (0.035) | 0.081 | 0.093 | 0.127 | 0/8 | 5.94 |
| **R1** | index RoPE, 1 layer | 304,128 | **0.118** (0.009) | 0.133 (0.005) | 0.072 | 0.088 | 0.116 | 0/8 | 6.13 |
| floor (map-free, gate F3) | -- | -- | 0.0472 | 0.0472 | -- | -- | -- | -- | -- |

Per seed, eval R1 (3600-3607): P_odo 0.543 0.532 0.532 0.491 0.568 0.543 0.546 0.553; R4 0.407 0.383 0.383 0.406 0.340
0.428 0.370 0.399; **P_E 0.276 0.278 0.273 0.464 0.435 0.281 0.274 0.367**; GPS 0.303 0.305 0.292 0.298 0.301 0.293 0.290
0.299; P_E_frozen 0.234 0.246 0.268 0.234 0.239 0.271 0.222 0.220; GRU 0.163 0.158 0.146 0.164 0.139 0.172 0.150 0.154;
P_raw 0.152 0.124 0.138 0.141 0.048 0.159 0.139 0.130; R1 0.130 0.116 0.112 0.119 0.106 0.132 0.109 0.115.

**Floors and ceilings.** Floor: the gate's best map-free retrieval under R1 (F3, pixel space), 0.0472 on eval
(`docs/audits/2026-10-27/memory_maze_gate_v3/diag_eval_split.json`) and on held (`floors_held.json`); every arm is above
it. Ceilings: the gate's odometry integrator retrieves 0.924 (fresh train-part0) / 0.927 (eval diagnostic); the class-E
integrator 0.659 on eval. With the model's one-layer retrieval, given odometry (P_odo) reaches **0.539**: the readout,
not the position input, limits every arm (Amendment 2, M1), and P_E's 0.331 is 61% of that ceiling.

**Convergence (registered rule, Amendment 2.5).** DESCENDING = the linear fit of the training loss across 60-85% of
training falls by more than 3% of its mean. R4 8/8 (-5.4 to -5.9%), GRU 8/8 (-3.8 to -4.2%), P_odo 5/8, P_E 1/8, GPS 1/8
(-3.2%, at the boundary), P_E_frozen, P_raw, R1 0/8. The rule is a weak readout of retrieval still improving (validated on
the pilots); every secondary is budget-scoped. r(final-5% training loss, eval R1) over the 64 runs -0.56 (analysis); the
loss mixes locally predictable sighted steps with the scored returns, so R4's much lower loss (3.68) is not matched by its
retrieval (rule 2).

## Declared descriptives

- **P_E_frozen - R1 = +0.124** (p 0.00016): with the class-E map installed and frozen, the trainable parameters are
  exactly R1's (304,128), so this is the cleanest "fixed path phase vs index phase" reading; a fixed dead-reckoning phase
  alone lifts one-layer retrieval from 0.118 to 0.242.
- **P_raw - P_E = -0.202** (p 0.00016): a path phase from raw egocentric actions without heading is at R1's level (0.129;
  one seed at the floor, 0.048). Turn-and-forward actions do not integrate to position without heading, as on the torus
  (`KNOB_SWEEP_n8.md`, `ALLOCENTRIC_RECODING.md`); here heading is given, and that is what the path arm integrates.
- **Context destruction** (scored like the primary; input frames from another episode with own keys, or step features
  shuffled in time):

  | arm | intact | frames swapped | steps shuffled |
  |---|---|---|---|
  | P_odo | 0.539 | 0.072 | 0.007 |
  | R4 | 0.389 | 0.057 | 0.014 |
  | P_E | 0.331 | 0.028 | 0.008 |
  | GPS | 0.297 | 0.177 | 0.012 |
  | P_E_frozen | 0.242 | 0.032 | 0.007 |
  | GRU | 0.156 | 0.040 | 0.009 |
  | P_raw | 0.129 | 0.034 | 0.020 |
  | R1 | 0.118 | 0.033 | 0.019 |

  Every arm falls to or below the floor (0.047) when the steps are shuffled, and every arm except GPS to <= 0.072 when the
  frames are swapped. GPS keeps 0.177 with foreign frames: given an absolute position code it predicts part of the view
  without the episode's own frames (keys are content-only for every arm; the mechanism was not analysed).
- **Offline probe** (Pasukonis et al. 2022 protocol: frozen state at the frame token concatenated with `agent_pos` and
  `agent_dir`, 4x1024 MLP, 10k steps, scored on eval steps 501-1000; Walls accuracy %, Objects squared distance averaged
  over objects):

  | arm | Walls, blinded | Objects, blinded | Walls, unblinded | Objects, unblinded |
  |---|---|---|---|---|
  | P_odo | 77.52 | 36.91 | 76.80 | 37.07 |
  | R4 | 75.16 | 37.02 | 77.60 | 37.16 |
  | P_E | 77.97 | 36.87 | 77.45 | 37.01 |
  | GPS | 75.99 | 36.94 | 76.51 | 37.19 |
  | P_E_frozen | 77.94 | 36.90 | 77.30 | 37.12 |
  | GRU | 78.20 | 36.86 | 78.30 | 36.78 |
  | P_raw | 77.78 | 36.89 | 77.50 | 37.06 |
  | R1 | 77.11 | 36.92 | 77.16 | 37.19 |
  | constant (training means) | 78.19 | 65.09 | | |
  | **pose only, no memory (analysis, below)** | -- | **36.75** | | |
  | published 15x15: constant / VAE (no memory) / RSSM-TBTT (Pasukonis; Deng) / TSSM-XL / S4WM / S5WM / supervised oracle | 78.3 / 78.7 / 80.5; 81.7 / 79.8 / 80.2 / 81.8 / 88.5 | 64.8 / 56.2 / 33.8; 32.6 / 35.1 / 34.8 / 25.3 / 9.2 | | |

  **The probe does not discriminate between our models.** Walls is at or below the constant for every arm (75.2-78.2 vs
  78.19). Objects is 36.8-37.2 for EVERY arm, R1 and the memory-poor P_raw included. The reason (analysis,
  `docs/audits/2026-10-28/memory_maze/pose_only_probe.py` -> `pose_only_probe_out.txt`, CPU, no model): `targets_vec` is
  an exact rotation of (object world position - agent position) by the heading (reconstructed world positions are
  constant within an episode to 4e-7), and the probe is GIVEN the pose. A predictor with no memory that rotates the
  objects' training-mean world positions into the agent's frame scores **36.75**, below every arm. Every arm's Objects
  number is therefore what the pose alone buys; none adds object information through its state. The published rows are
  not comparable: their probe is trained alongside the model on 30M steps (ours 10k post hoc steps on 1600 episodes), and
  their memory-free VAE row (56.2) is far above this pose-only level, so the protocols do not reach the same baseline.
  Registered as a standard secondary for comparability; read here as uninformative.

## Further analysis (unregistered, from the same results.json)

- **P_E is bimodal.** Five seeds at 0.273-0.281 and three (3603, 3604, 3607) at 0.367-0.464. The high seeds start
  slower (loss 6.9-7.2 at epoch 100 vs 6.5) and end lower (5.43-5.55 vs 5.74-5.78); within the arm r(final loss, R1)
  -0.99. The pilots showed a jump of the validation curve at epoch 150-250. The primary does not rest on the high mode:
  the low mode alone (0.273-0.281) is above every R1 seed (max 0.132). S1 does: every low-mode P_E seed is below every R4
  seed (min 0.340), and two high-mode seeds are above R4's max (0.428).
- P_E - P_E_frozen +0.089 (p 0.00016): the learned step map retrieves better than the installed exact class-E map,
  frozen (the pilot showed the same: 0.311 scratch vs 0.283 frozen). The scratch solution need not be class E.
- R4 - R1 +0.272 (p 0.00016): three more index layers buy most of what the path phase buys, and more on average.
- GPS - P_E -0.034 (p 0.27): an index layer given an absolute code of the TRUE position is not separated from a path
  layer that learns its step; with the same given position as a relative phase (P_odo) retrieval is 0.24 higher (S4).
- Under R2 (the target frame beats every frame of another pose bin) the order differs: P_odo 0.461, R4 0.346, GPS
  0.281, P_E 0.205, P_E_frozen 0.142, GRU 0.131, P_raw 0.081, R1 0.072. P_E - R1 stays positive (+0.133); P_E is below
  GPS on R2.

## Predictions vs outcomes

Pre-registered before the pilots; the first three rows were already contradicted by the pilots (Amendment 2.3,
registration unchanged).

| prediction | pilot (validation split) | batch (eval split) | |
|---|---|---|---|
| R4 "a little above R1" | 0.44 vs R1 0.14-0.17, above P_E 0.31 | 0.389 vs 0.118; above P_E on average, S1 UNMEASURED | **contradicted** |
| P_E near the class-E integrator 0.66 | 0.31-0.39 | 0.331 | **contradicted** |
| P_odo near the odometry integrator 0.93; UNREADABLE not expected | 0.54-0.56 | 0.539; readable (floor + 0.10 = 0.147) | 0.93 **contradicted**; readable held |
| P_E between R1 and P_odo | yes | 0.118 < 0.331 < 0.539 | held |
| R1 low | 0.11-0.17 | 0.118 | held |
| GPS close to P_odo | 0.365 vs 0.537 | 0.297 vs 0.539 (S4 P_odo ABOVE GPS, +0.241) | **contradicted** |
| GRU: no prediction | 0.150 | 0.156 (S2 P_E ABOVE GRU) | -- |
| PRIMARY PATH WINS | P_E above R1 by 0.15-0.28 | PATH WINS, +0.213 | held (loss pilot-informed) |
| Probe: Objects better than constant for every arm with memory | 36.5 (P_E) vs 65.6 | 36.8-37.2 vs 65.09 for every arm, R1 included; a pose-only predictor scores 36.75 | numerically held, **uninformative** |
| Probe: Walls within a few points of constant | 78.37 vs 78.41 | 75.2-78.2 vs 78.19, none above | held |

## Gate history

v1 STOP (S2: a linear two-frame continuation predicts the re-entry view, 0.89 under R1; S3: the 8-bin action x heading
table dead-reckons 13% of long-gap re-entries); v2 STOP at 32 blind / 32 sighted (class E 0.490 within 0.5 cell, one SE
below 0.50; 2.44 returns per episode). v3 PASS at 16 blind / 16 sighted on fresh episodes (cell chosen post hoc from v2's
grid, disclosed): floor 0.048, class E 0.605, 5.36 returns per episode; on the eval split class E 0.514 (at the threshold).

## Reading for the paper

On a first-person 3D benchmark with a new maze every episode, egocentric actions, momentum and collisions, one attention
layer whose rotary phase integrates a learned step map of the given actions and heading retrieves the view at blind
loop-closing re-entries 2.8x as often as one index-RoPE layer given the same inputs (0.33 vs 0.12; floor 0.05), and above
a parameter-matched GRU trained across mazes (0.16). A frozen dead-reckoning phase alone, at R1's exact trainable
parameters, lifts retrieval to 0.24. Without heading (raw egocentric actions) the path phase is at the index level.
Four index layers (2.95x the parameters) are not separated from one path layer (0.39, still descending), so the
torus/Dyck exchange rate does not carry over here. With given odometry the same one-layer readout reaches only 0.54
against an integrator ceiling of 0.93: the retrieval layer, not the position input, limits every arm.

## Scope

- **Heading is given** (a compass enters the step input): the claim is "integrates given self-motion", not "navigates
  from pixels". P_odo and GPS are given position outright.
- One layer for every arm except R4; one budget (600 epochs, a pilot-informed choice); one recipe; n = 8. The loss is
  pilot-informed (Amendment 1); the primary is a confirmation of the pilot finding on fresh seeds and untouched splits.
- **The one-layer retrieval limits every arm** (P_odo 0.54 vs 0.93). The primary measures how much ONE learned retrieval
  layer gains from a path phase over an index phase with the same inputs, not the accuracy of integration.
- **R4 is unmeasured against P_E and still descending (8/8)**; with more budget it may separate above P_E. The GRU is also
  8/8 descending. "One path layer ~ three attention layers" is not supported here.
- Train-part0 only (1,600 training episodes of the 29k in the full dataset); a frozen autoencoder frame encoder; 16/16
  blinding; the eval split comes from a separate generation run with more collisions (the harder split for the step
  input). Cluster runs (torch 2.13) are never pooled with the workstation pilots (torch 2.10).
- The offline probe does not discriminate (pose alone explains its Objects reading); no claim from it.
- Rule 14 met weakly (Amendment 2.1): the class holds a solution at about scratch's level; the install's weight scale was
  not varied. The function-identical twin noise floor was not run (Amendment 2.10); the 8-seed sd covers seed noise.
