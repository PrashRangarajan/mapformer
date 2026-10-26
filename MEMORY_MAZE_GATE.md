# Memory Maze CPU gate: registration (written 2026-10-26, before any gate number was computed)

Purpose: decide, on the official offline dataset and with no GPU, whether Memory Maze can test the project's axis at
all, before any design is registered. Answers the BLOCKERs of `docs/audits/2026-10-26/CRITICAL_REVIEW.md` Part A
(A-B1 the given-odometry step is the answer; A-B2 the retrieval target as first written penalises the map; A-B3 15x15
may sit below the map-extent threshold) and A-M4 (dataset facts were README-level). Code: `mm_gate.py`; cluster
scripts: `hyak/mm_download.sh`, `hyak/mm_gate.slurm`. Results: `docs/audits/2026-10-26/memory_maze_gate/`.

## What was looked at before this file was written (disclosure)

Format only, from the first two episodes of the 15x15 eval archive (a 12 MB partial read): the npz keys and shapes
(`image` (1001, 64, 64, 3) uint8; `action` (1001, 6) one-hot, row 0 all zero; `agent_pos`, `agent_dir` (1001, 2);
`maze_layout` (1001, 15, 15); `targets_pos` (1001, 6, 2); `target_color`, `target_pos/vec`, `targets_vec`, `reward`,
`terminal`, `reset`), that positions are in cell units (cell centres at k + 0.5, range 0..15) and that a moving step is
about 0.25 cell. No revisit, floor, open-cell or action statistic was computed before this file.

## Data and subset

Official offline dataset (Pasukonis et al. 2022, 2210.13383; README of github.com/jurgisp/memory-maze): per maze size
30k episodes x 1000 steps, 29k train (10 zip parts) / 1k eval (one zip), scripted policy that walks to random points
under action noise. Gate subset: **the 1000 eval episodes of each size** (15x15 decides; 9x9 is descriptive). Fits
(PCA, linear continuation, action tables, probes) use the first 300 episodes of the 15x15 `train-part0` archive, never
eval episodes. Alignment: `image[t]` is the frame after `action[t]`, t = 1..1000; `image[0]` precedes the first action.

## Definitions (fixed here)

- **Cell** `floor(agent_pos)` (unit cells). **Heading bin**: `atan2(agent_dir)` in 8 bins of 45 deg centred on the
  axes. **Pose bin** = (cell, heading bin).
- **Layout orientation**: of the 8 transposes/flips of `maze_layout`, the one under which the largest share of visited
  cells carries a single value; that value is "open". The share is reported (must be ~1.0, else the gate flags it).
- **Open cells** per maze = number of open cells under that convention. Also reported: distinct cells and pose bins
  visited in 1000 steps, and mean prior visits to the cell at scored steps (the visits-per-cell predictor that
  `VISITS_TEST.md` could not separate from extent).
- **Scored step (re-entry)**: t with pose bin(t) != pose bin(t-1) and pose bin(t) occupied at some s < t-1. Steps
  that stay in the previous bin ("dwell") are not scored (copy-the-current-frame solves them); first entries are not
  scored. **Gap** = t - (most recent earlier step in the same pose bin).
- **Backtrack vs loop closure**: the cell sequence from that earlier step to t, run-length compressed; a palindrome
  is a backtrack (out and back along the same cells), anything else a loop closure.
- **Long gap**: gap >= 128 steps (the survey's "beyond retrace reach"; the torus FACTORIAL strata used 128).
  **Non-local** = long-gap loop closures.

## Readouts on a predicted frame (the review's fix A-B2)

Each predictor outputs a predicted frame (a vector in a feature space) for f_t, using only what precedes f_t.
- **R1 (registered): pose-bin match.** The nearest frame of the same episode (all 1001 frames, f_t included) to the
  prediction; correct iff its pose bin equals pose bin(t). Earlier same-bin frames are therefore positives, not
  negatives. The pose-oracle ceiling is 1.0 by construction under R1.
- **R2 (secondary): exact frame, same-bin frames removed from the negatives.** Correct iff f_t is nearer to the
  prediction than every frame in a different pose bin. Measures sub-bin variation; the pose-oracle ceiling is < 1.

Feature spaces: `pix16` (64x64 RGB block-averaged to 16x16x3, /255, L2) and `pcaw64` (whitened PCA-64 of `pix16`,
fitted on train frames). Every floor is reported in both; the stop rule uses the better (higher) of the two.

## Measurements

(a) **Extent**: open cells per maze (median, quartiles, share of mazes with >= 128 and >= 512), distinct cells
visited, prior visits at scored steps; versus the project's threshold (-0.010 / +0.015 / +0.305 at 32 / 128 / 512
occupied cells, `ALIASING_CONTROLLED.md`, `VISITS_TEST.md`).

(b) **Revisits under the dataset's policy**: scored steps per episode, gap quantiles, shares with gap >= 16 / 64 /
128 / 256, backtrack vs loop-closure share, and the **non-local share** (gap >= 128 and loop closure) of scored steps,
with an episode-bootstrap 95% CI. Gap shares double as the window-W oracle: a method whose memory spans W steps and has
perfect local odometry solves at most the scored steps with gap <= W.

(c) **Floors and ceiling on scored steps (R1 and R2), overall and on the non-local stratum**:
- copy-current: predict f_{t-1} (0 under R1 by the scored-step definition; reported as a check);
- linear two-frame continuation: per-action ridge regression (f_{t-1}, f_{t-2}) -> f_t in `pix16`, fitted on train
  episodes (a LOWER bound on the continuation floor: a trained CNN would be higher);
- view recognition (map-free visual loop closure): the earlier step s whose frame (m = 1) or last four frames
  (m = 4) best match those before t, among s with the same next action as a_t and s <= t - 2 - k (k = 0 and k = 16);
  predict f_{s+1}. Four variants (m x k); the floor is their maximum;
- single-frame pose identification (encoder-leak probe (e), uses f_t itself, so an upper bound on view-based
  localisation): the nearest earlier frame to the TRUE f_t among s <= t - 16; R1 on it;
- pose oracle: the most recent earlier frame in the same pose bin (1.0 under R1; R2 reported).

(d) **Actions, for the inverted design (A-B1: egocentric action + heading as the step input)**: action shares
(noop, forward, left, right, forward-left, forward-right); heading change and displacement per action; collisions
(forward-type command, displacement < 0.05 cell, nominal ~0.25); alignment of displacement with heading.
**Existence check in P_joint's class (rule 14)**: a 48-entry table d(action, heading bin) (mean displacement, fitted
on train; the best constant-per-token step map, i.e. what a linear step map on a joint token can represent) integrated
between the earlier visit and t; on non-local scored steps, the share whose relative-displacement error is < 0.5 cell
(true heading bins given) and the same with heading also dead-reckoned from a per-action turn table. Also the per-step
R^2 of the table.

(e) **Encoder pose leak (optional)**: the single-frame identification in (c), plus cross-maze ridge probes from
`pcaw64` to heading bin and to cell (train-part0 frames -> eval frames), with chance levels.

## STOP RULE (fixed before any number)

Applied to the 15x15 eval subset. Any STOP means: do not register the 15x15 x 1000-step primary as designed.

- **S1 extent / locality.** STOP if (a) fewer than half of the mazes have >= 128 open cells, OR (b) the non-local
  share (gap >= 128 and loop closure) is < 10% of scored steps (pooled; X = 10%). Reasons: (a) at 128 occupied cells
  the project measured +0.015, a null, and at 32 cells -0.010; a maze below 128 is in the regime where index attention
  matched path integration. (b) Below 10%, a pooled readout is dominated by steps solvable without a map, and the
  stratified non-local readout has under a tenth of the scored steps per episode -- too few to resolve effects of the
  0.1-0.3 size seen on the torus at n = 8. This is the review's OR rule (stricter than requiring both).
- **S2 a map is not needed.** STOP if the best map-free floor (max over the linear continuation and the four view
  recognition variants, better feature space) under R1 on the non-local stratum is within Y = 0.15 of the pose-oracle
  ceiling (i.e. >= 0.85, since the ceiling is 1.0 under R1). Reason: 0.150 is the only measured function-identical-twin
  noise floor on a visual navigation task in this project (MiniWorld); headroom below it cannot hold a reportable effect.
- **S3 the inverted primary has no solution in its class (design branch, not a benchmark stop).** If the 48-entry
  (action x heading bin) table, with true heading bins, puts < 50% of non-local scored steps within 0.5 cell of the true
  relative displacement, P_joint as a table step map cannot find long-gap revisits even when it learns the table
  exactly (collisions and action noise break dead reckoning). Then P_joint is not registered as designed; the options
  are an odometry step inferred from frames (A-B1 (ii)) or a step input that includes collision information.

If S1 or S2 fires, the next options are those of review Part A item 3: native-length (4000-step) or larger custom mazes
generated with the simulator, or the glimpse torus instead. 9x9 numbers are reported beside 15x15 but decide nothing.

## Not measured here

An action/odometry n-gram over pose bins (survey floor (v)); a trained CNN continuation; frame-shuffle context
destruction and a function-identical twin (need trained models); the published layout probes.

## Amendment 1 (2026-10-26, after a 2-episode smoke test, before the gate run)

The smoke test (the two format-check episodes used as both train and eval, so every fitted floor was in-sample and is
not a result) showed that the 8-bin table of S3 leaves a per-step residual that accumulates over long gaps, and that
turning commands move the agent (momentum: the body keeps velocity). Two **secondary** dead-reckoning variants are added
to (d) to say why S3 passes or fails; S3 itself is unchanged (8-bin table, true heading bins):
- C: per-action least-squares map from the continuous heading vectors [dir_{t-1}, dir_t] to the displacement
  (a step map that multiplies an action token by a continuous heading input);
- E: the same per (previous action, action) pair (a step map that also sees the previous token: momentum).
Reported as per-step R^2 on eval and the same long-gap relative-error shares as S3. No threshold is attached to them.
No other change: stop rule, definitions, subset and readouts as registered above.

## Status (2026-10-26 02:45): gate NOT run -- the dataset could not be downloaded yet

- Source: the README's public Google Drive folder (one folder per size; `eval.zip` 1k episodes, 3.47 GB for 15x15;
  `train-part0..9.zip` 29k episodes; ~100 GB per size; 22 archives in all). No mirror exists (Hugging Face searched;
  the authors' issue tracker: Dropbox was abandoned for bandwidth limits, Drive is the only source, and the dataset
  generation code is not public -- so "generate more with the same policy" is not possible).
- Every full-file request since 01:34 returns Drive's "Quota exceeded" page (HTTP 200, HTML); byte-range requests
  are still served, but `hyak/mm_download.sh` deliberately does not use ranges to get around the quota (it starts or
  resumes a file only when a plain request serves it). It probes hourly inside each 8-h CPU job and keeps archives as
  zips (CRC-tested, member count, field check, then an `.ok` marker).
- Cluster side: target is the cluster's large scratch (about 4.1 TB free at 01:34; both sizes, ~200 GB, fit with
  margin). That scratch is purged when files sit idle (about 30 days): copy anything needed long-term elsewhere.
- Queued: two consecutive 8-h download windows, each followed by the CPU gate (`hyak/mm_gate.slurm`, which skips
  sizes already done and exits cleanly when an archive is missing). No GPU anywhere.
- Code check: `mm_gate.py` ran end to end on the two format-check episodes, locally and on the cluster's CPU nodes
  (python 3.12 / numpy 2.5 there), with matching outputs; those numbers are in-sample and not results.
