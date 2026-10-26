# Image-input tasks for path-integrated position: survey and plan (2026-10-25, updated 2026-10-26)

Question: which tasks with IMAGE observations test what our models are about -- a phase
`theta = omega * cumsum(W_out W_in x_t)` that must integrate a known action stream so that revisits are predicted
from a map, not from local undo/retrace or from recognising the current view? Survey only: no GPU job, no code
change. Sources marked *(web survey)* come from a literature-search agent and were not read first-hand; arXiv ids
not checked against the paper are marked *(id unverified)*.

Terms. *Path model* = MapWM family (path-integrated phase). *Index RoPE* = rotary phase by token index. *ActOnly* =
the step map reads action tokens only, so an observation never moves the phase. *Retrace* = the best map-free rule
(copy the observation across a run of moves that undoes the previous run). *Revisit* = a step whose location was
visited before in the episode; *long-gap* / *wrap-only* = revisits not reachable by retrace.

---

## 1. What the project has done with images

**Until 2026-10-25 the project had NO result with image inputs.** Every "visual" environment was used as a
geometry generator with synthetic or symbolic tokens:

| env | what the model actually saw | files | how it ended |
|---|---|---|---|
| MiniWorld (OneRoom, continuous 3D) | a fixed random token per discretised floor cell (`obs_map`); `render_obs` is stubbed out (`miniworld_env.py` lines 5-8, 70-80). Only the continuous turn/forward geometry was real | `MINIWORLD_*.md`, `ALIASING_CONTROLLED.md`, `VISITS_TEST.md`, `prebuild_buffers.py`, `.claude-memory/project_miniworld_flip_negative.md` | Converged: grid 8 -0.010, grid 16 +0.015, grid 32 **+0.305** at matched aliasing -> the citable **map-extent threshold** (between 128 and 512 occupied cells). Aliasing account falsified (sign inverted at fixed grid), visits-per-cell no effect. Earlier "path integration is a liability" (-0.086/-0.174) was a LinearLR non-convergence artefact |
| MiniGrid (DoorKey 8/16, symbolic) | ONE token for the cell in front of the agent (`obj_only`/`obj_color`/`full`), not the 7x7 image (`minigrid_env.py` docstring) | `MINIGRID_*.md` | Small effects (+0.02 allocentric vs -0.02 raw, `MINIGRID_ALLOCENTRIC_2X2X2.md`); 16x16 map is below the extent threshold; index PoPE arms as good as path arms |
| Habitat (habitat-sim 0.3.3, py3.9 conda env) | nothing trained | `HABITAT_BUILD.md`, `PERCEPTION_EXPERIMENT_PLAN.md`, `report/inventory/D_environments.md` D13 | Not ported: navmesh slides on 69-91% of forward moves (continuous displacement magnitude); published numbers are RL policies; ENTL-style 256 VQ tokens/frame gives ~130k tokens/episode; Gibson/MP3D/HM3D need licence agreements |

**Why the image path was abandoned -- at the tokenizer, not because images "did not help".** Every model here then
consumed discrete tokens, so the plan was to quantise frames into a codebook. A reconnaissance (2026-08-23/24) on
MiniWorld RGB found "two reasonable settings giving 1 code and 400 codes over 185 positions -- degenerate in opposite
directions" (`HABITAT_BUILD.md` lines 102-105; `PERCEPTION_EXPERIMENT_PLAN.md` 67-69: "the scenes are so plain that
codes don't cleanly map to location"). **The method (VQ-VAE or k-means, codebook size, features) is not recorded
anywhere in the repo or git history**; only the outcome survives. Read: a coarse quantiser collapsed MiniWorld's
plain walls/floor to one code; a fine one gave ~2 codes per position (view/heading-specific), so neither gave a
code that is stable across revisits and keyed to location. The planned feasibility gate P0 (code<->cell mutual
information) was never run. The obvious fix -- a **continuous** learned encoder whose output is the observation
embedding, no codebook -- was planned (Phase A) but never built.

**Since then (2026-10-25): IMAGE_TORUS** (`IMAGE_TORUS_RESULTS.md`, registered, `IMAGE_TORUS_PREREG.md` +
Amendments 1-3). The FACTORIAL torus (64x64, 16 classes, T=1024, 1 layer, 900 ep) with each observation token
replaced by a 28x28 image of its class (MNIST 0-9 + six Fashion-MNIST classes), a fresh instance at every visit,
test-split images at eval, read by an 80k-param CNN trained end to end.
- **P1 PATH WINS IN PIXELS**: image ActOnly path model **0.9946** (8/8 SOLVED, every seed within 0.01 of the
  perception ceiling 0.9927) vs image index RoPE **0.7010** (0/8, below retrace 0.8432); +0.294, perm p 0.00016.
  Long-gap and wrap-only returns 0.994 vs retrace 0.51.
- **P2 / G: the encoder rule.** If images also drive the step at token scale (gain 30.37), 5/8 runs collapse to the
  memoryless level (encoder outputs a constant; the phase becomes a clock ticking at every observation slot) and
  the 3 that learn stay below ceiling; at gain 1 (image step 0.001-0.006 of a move) 8/8 reach ceiling.
  **Rule: actions drive the step; images only enter content.**
- Scope: synthetic class images on a torus; perception is trivial (classifier 0.980); the task is still our
  location-keyed token task with a pixel front end. It settles "does the mechanism survive a learned encoder", not
  "does it help on a visual benchmark".

---

## 2. Candidates, ranked against our axis

Criteria: (A) actions known to the model, ideally allocentric or recodable; (B) revisits that need the action stream
(not retrace, not view recognition), and how common; (C) map size / aliasing controllable; (D) held-out-map protocol;
(E) cost on 1-2 4090s or Hyak A40s; (F) baselines; (G) licence; (H) plumbing effort.

| rank | candidate | obs / actions | (B) map-needing revisits | (C) control | (D) held-out maps | (E) cost | (F) baselines | (G) licence | (H) effort | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 (done) | **Image torus** (class images on torus cells) | 28x28 / 4 allocentric moves | all wrap/long-gap revisits; retrace measured | full | yes (map + test-split images) | 11.4 h for 48 runs on 2 4090s | ours | MNIST/F-MNIST open | done | **done, positive** |
| 1 | **DeepMind Memory Maze** (Pasukonis, Lillicrap, Hafner 2022, arXiv 2210.13383) | 64x64x3 first-person / 6 discrete egocentric (noop, fwd, left, right, fwd-left, fwd-right) | yes: scripted waypoint policy "forms spatial loops"; frequency not published | maze size 9/11/13/15; aliasing not separately controllable (random textures/colours, unverified) | every episode a fresh maze; 29k/1k train/eval split | official offline data 2x30M steps (~100 GB each); sim ~36 steps/s/process (9x9) | Dreamer/IMPALA RL; offline probes: RSSM/TBPTT (README); Deng et al. 2307.02064 RNN/Transformer/S4 world models, every model near the constant floor at 15x15 *(web survey)*; R2I 2403.04253; Po et al. 2505.20171 long-context SSM video world model with a retrieval task *(web survey)* | MIT, pip `memory-maze` | medium: dataset reader, frozen encoder, pose-difference recode, retrieval loss | **best external benchmark** |
| 2 | **Glimpse torus** (crop of a large natural image at the torus location) | k x k crop / 4 allocentric moves | yes, but neighbouring crops overlap -> local continuation is a strong map-free predictor | image size, crop size, stride, texture statistics | new image per episode | ~ image torus | none found for revisit prediction of crops *(web survey: open slot)*; RAM (Mnih 2014, 1406.6247), Jayaraman & Grauman 2018 (1709.00507) are glimpse policies, not map tests | ImageNet/COCO terms; or procedural textures | low (reuse IMAGE_TORUS) | **controlled bridge #2**, cheap |
| 3 | Memory Gym (Pleines et al. 2023, 2309.17207) | 84x84 pixel / discrete | Mystery Path: invisible path, retrace of own mistakes; partly a map | endless variants | procedural | cheap (2D) | PPO+GRU vs TransformerXL *(web survey)* | MIT *(unverified)* | medium | memory, not much map |
| 4 | DeepMind Lab (Beattie 2016, 1612.03801) | RGB first-person / egocentric + velocities available | yes (random mazes); Banino 2018 (grid cells, velocity input), Uria 2020 spatial memory pipeline | maze generator | procedural | moderate; old build, Bazel | Dedieu et al. 2401.05946 (k-means frame tokens, CSCG/transformer; in repo) | GPL-2.0; maintenance minimal | high (build) | superseded by Memory Maze |
| 5 | MiniWorld pixels | 60x80 RGB / turn, fwd | yes, but rooms are visually plain (the tokenizer failure) | room geometry, add textures | procedural | sim ~2k steps/s CPU, EGL contexts don't fork | few | Apache-2.0 (Farama) | low (adapter exists) | only with per-cell textures |
| 6 | MiniGrid / BabyAI pixel obs (2306.13831, 1810.08272) | 7x7 egocentric partial view rendered / turn, fwd | maps small (<= 16x16, below our extent threshold) | grid size | procedural | trivial | many RL | Apache-2.0 | low | below threshold; skip |
| 7 | ViZDoom (Kempka 2016, 1605.02097) | RGB first-person / egocentric | custom mazes needed | via WAD editing | custom | fast | RL | MIT (engine GPL) | high | no map-test protocol |
| 8 | Crafter (2109.06780) | 64x64 local top-down / 17 | open world, revisits incidental | limited | procedural | fast | DreamerV3 etc. | MIT | medium | survival, not mapping |
| 9 | Procgen Maze/Heist (1912.01588) | 64x64, mostly full view | full view -> no map needed | level gen | yes | fast | PPO | MIT | -- | skip (no partial obs) |
| 10 | Habitat/Gibson/HM3D, AI2-THOR/ProcTHOR (2206.06994) | photoreal RGB(-D) / turn, fwd | yes | scene choice only | scene splits | heavy; navmesh sliding | DD-PPO, OVRL; ENTL 2304.02639 | HM3D/MP3D need agreements; THOR Apache-2.0 | high | credibility, not control; later |
| 11 | Minecraft (Gornet & Thomson 2024, Nat. Mach. Intell., visual predictive coding cognitive map *(id unverified)*; WorldMem, VMem, Context-as-Memory 2025 *(ids unverified)*) | RGB / camera + move | yes; loop-closure evals in video world models | scene choice | partial | heavy (video models) | diffusion world models | Minecraft EULA / MineRL | high | prior art to cite, not a testbed |
| 12 | CSCG with images (George et al. 2021 Nat. Comms; Raju et al. 2212.01508; Dedieu 2401.05946) | VQ / k-means frame tokens | yes | by construction | yes | cheap | CSCG | -- | -- | same codebook route that failed us |
| 13 | Blind agents (Wijmans et al. 2023, 2301.13261) | no vision (egomotion only) | -- | -- | -- | -- | LSTM | -- | -- | evidence maps form from egomotion; not an image task |
| 14 | VPR sequences (Nordland, Oxford RobotCar) | photos + odometry | route repeats, few loops | none | route splits | data-heavy | VPR methods | mixed | high | poor fit |
| 15 | Action-conditioned video WMs (Genie 2402.15391, DIAMOND 2405.12399, NWM 2412.03572, GameNGen 2408.14837) | frames / actions | loop consistency rarely measured | none | -- | very heavy | diffusion/ST-transformers | mixed | very high | out of budget |

**Prior-art risk.** Rotary encodings driven by geometry on images exist but take **given** poses: GTA (Miyato et
al. 2310.10375), PRoPE / "Cameras as relative positional encoding" (Li et al. 2507.10496), ViewRope, ReRoPE, LieRE
(in `papers/txt/liere.txt`), GridPE (2406.07049, coordinates given). The web survey found **no work that drives the
rotary phase by integrating actions on image-observation navigation**, and no glimpse/crop revisit-prediction task
with a held-out-map protocol. Both are open slots; recheck before claiming novelty (`docs/lit/`).

---

## 3. Recommended path

### 3a. Controlled bridge: done (IMAGE_TORUS). Optional follow-on: glimpse torus

IMAGE_TORUS answered the cheapest decisive question. The one thing it could not test is **view similarity between
neighbours** (its images are i.i.d. per class). A glimpse torus does: observation = k x k crop of a fresh natural
image tiled on the torus at the agent's cell, 4 allocentric moves, T=1024. Arms: image ActOnly path, image RoPE 1L
(and 2L), same encoder. Floors: retrace; **local continuation** (predict the next crop from the current crop's border,
a trained 2-frame model); exact-crop ceiling. Readout: retrieval of the true crop among the episode's crops. Cost ~
IMAGE_TORUS (~12 h on the two 4090s for 48 runs). Run only if Memory Maze's gate shows view similarity is the open
risk.

### 3b. External benchmark: Memory Maze on Hyak (ckpt, A40)

**Measured feasibility (job 42334056).** memory-maze 1.0.3 + dm_control 1.0.50 + mujoco 3.17 + gym 0.26.2 in
`/gscratch/cse/prashr/mapformer_mm/mm-env`; EGL rendering works on A40 (OSMesa unavailable, so generation needs GPU
nodes); ~36 env steps/s per process at 9x9, ~21 at 15x15; 8 parallel processes ~33 each (~260 steps/s/node). ExtraObs
(`global_observables=True`) gives `agent_pos`, `agent_dir`, `maze_layout`, `targets_pos/vec`, `target_color`.
6 discrete actions. 9x9 episodes 1000 steps (README: 11x11 2000, 13x13 3000, 15x15 4000).

**Data.** Use the **official offline dataset** first: 30k trajectories x 1000 steps per maze size (9x9, 15x15),
29k train / 1k eval, ~100 GB each, scripted waypoint policy under action noise chosen to form loops. Its npz files
**do include** `image` (64x64x3), `action` (one-hot 6), `agent_pos`, `agent_dir` (unit vector), `maze_layout`,
`targets_pos`, `targets_vec`, `target_pos/vec`, `target_color` (README, checked 2026-10-26), so no simulation is
needed for the core study. Generate with ExtraObs (array jobs) only for what the dataset lacks: **15x15 episodes at
their native 4000 steps** (the dataset caps 15x15 at 1000 steps, which limits long-gap revisits) or a controlled
policy. Cost at ~150 steps/s/node at 15x15: 1000 episodes x 4000 steps ~ 7.4 node-h, i.e. one 8-h ckpt array task per
1000 episodes; write per-episode npz and a `.done` marker per shard so preempted tasks resume.

**Actions: the egocentric-turn problem and the recode.** Raw Memory Maze actions are egocentric (turns change what
"forward" means); on the torus rotation actions cut the path effect to +0.050 vs **+0.488** allocentric (8/8,
`KNOB_SWEEP_n8.md`); recoding the record to world-frame displacement restores it (`ALLOCENTRIC_RECODING.md`,
+0.049 -> +0.485 at n=3). Habitat taught that the record must come from the **pose difference**, not the command
(collisions; `HABITAT_BUILD.md`). So the step input is `d_t = agent_pos[t] - agent_pos[t-1]` (2-d, continuous, world
frame), optionally plus `agent_dir` change; fed through `W_in` directly (continuous) or quantised (24 direction x 3
magnitude bins, MiniWorld style). This is privileged odometry, as in Banino 2018 / Uria 2020; say so. The raw-action
path model is a secondary arm (expected to fail; replicates the rotate cell). Index arms see the same inputs as tokens.

**Encoder (the IMAGE_TORUS rule).** Continuous embedding, no codebook. Stage 1: a small CNN (64x64 -> 256-d)
pretrained on train-split frames (time-contrastive or autoencoder), **frozen**, embeddings precomputed once (fp16,
~15 GB per 30M frames; ~1 GPU-h). Transformer runs then cost like token runs. End-to-end fine-tuning only as a
secondary. **The step map reads `d_t` only (ActOnly)**; frames enter content only. Gain-1 MapWM may be a secondary
arm; token-scale image steps are excluded (P2/G collapse).

**What is predicted at a revisit.** Sequence `[d_1, f_1, d_2, f_2, ...]`, T = 2000 tokens per 1000-step episode.
- *Primary*: next-frame **retrieval**: predict `f_t`'s embedding from context up to `d_t`; score = the true frame
  ranked first among all frames of the same episode (hard negatives from the same maze); InfoNCE as the loss. Scored
  on **pose revisits**: `(cell, 8 heading bins)` seen before, stratified short-gap / long-gap (> retrace reach,
  > 128 steps) / first-visit (not scored).
- *Secondary, for external comparability*: the published **offline probes** -- `maze_layout` and `targets_pos`
  decoded from the frozen hidden state (MLP probe on train split, eval split readout), beside Deng et al.'s table.
- `target_color` at a location is too sparse (3-6 objects) to be the primary target.

**Floors (rules 1, 11; gate before any GPU, CPU on the dataset).** (i) revisit statistics: pose-revisit rate, gap
distribution, share reachable by retrace, and **occupied cells vs the 128-512 extent threshold** (9x9 has <= 49 open
cells, 15x15 <= 169: 9x9 is in the regime where index attention matched path integration in MiniWorld) -> 15x15 is
the primary size unless the gate says otherwise; (ii) copy-current-frame; (iii) retrace; (iv) **view-recognition
floor**: find the most similar earlier frame to the current one and copy its successor (the map-free "visual loop
closure"); (v) action/odometry-only n-gram over pose bins; (vi) pose-oracle ceiling: most recent frame at the same pose
bin; (vii) after training: context destruction (shuffle frames; shuffle `d_t`) and a function-identical twin for the
noise floor; (viii) encoder probe: how well a single frame identifies its pose bin (if near 1, view recognition can
solve revisits and the task does not test the map).

**Arms (n = 8, one batch, rule 12).** P = path model ActOnly on `d_t`, 1 layer; R1 = index RoPE 1 layer; R4 = index
RoPE 4 layers (FACTORIAL: depth helps index on short revisits only); E = path model on raw egocentric actions
(secondary). Rank 4 per head with weight decay 0.05 (the rank and RANKWD results). Primary contrast P - R1 on
long-gap retrieval at 15x15; secondaries P - R4, P - E, probe scores vs Deng et al. Pilots on outside seeds;
independent verification agent before reading (rule 29).

**Cost.** Dataset download ~200 GB to gscratch (check quota); encoder pretrain + embedding ~2-3 A40-h. One run at
T = 2000, d 128, 1 layer on precomputed embeddings ~ 2-4x a torus T=1024 run, roughly 1-2 A40-h; 32 runs (4 arms x 8)
~ 32-64 A40-h, i.e. 4-8 ckpt array tasks of 8 h at 2-4 jobs per GPU. Every run must checkpoint each epoch and resume
(ckpt preemption; 8-9 h windows). If the native-length 15x15 data is generated: + ~10 node-h per 1000 episodes.

---

## 4. Risks

1. **View recognition solves revisits without a map.** First-person frames identify place + heading; a model can
   match the current frame to an earlier one and copy what followed. Measure floor (iv) and the encoder probe (viii)
   before training; score the long-gap stratum, and the step whose frame is predicted BEFORE it is seen.
2. **Map below the extent threshold.** 9x9 mazes are small; attention may integrate short revisits (MiniWorld grid
   8: -0.010, both solve). Use 15x15 and long episodes; read revisit lags first.
3. **The encoder becomes the bottleneck.** Index arms and path arms then tie at a perception ceiling. IMAGE_TORUS
   handled this with a measured ceiling (ceil_vote); Memory Maze needs the pose-oracle ceiling (vi).
4. **The leak / clock.** Images driving the step at token scale collapse training (IMAGE_TORUS P2: 5/8 memoryless).
   ActOnly on odometry; never let frames set the step in the primary arm.
5. **Odometry is privileged.** `d_t` from `agent_pos` is GPS-difference input; the claim is "integrates given
   self-motion", not "infers self-motion from pixels". The raw-action arm E bounds what is lost.
6. **Continuous displacement magnitude** (as Habitat): a cumsum of continuous `W_in d_t` is exact for linear input;
   quantisation adds error. Check the pose-reconstruction R^2 of the cumsum before the batch (rule 14: construct the
   solution first).
7. **Policy statistics.** The scripted waypoint policy may revisit mostly by retrace-like paths; the gate measures
   this. Our own generator allows a controlled policy at extra cost.
8. **Loss choice changes the floors.** Retrieval accuracy depends on the negative set (same-episode frames); report
   chance as 1/(frames in episode) and every floor under the same readout.
9. **Comparability.** Published Memory Maze numbers are RL returns or probe scores from world models trained with
   other objectives; only the probe secondary is directly comparable, and only as a reading, not a replication.
