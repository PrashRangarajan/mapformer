# Does path integration survive pixel observations? The image torus -- pre-registration (DRAFT, 2026-10-25, before any run)

Status: DRAFT for the independent check (rule 29). Not committed, not launched. Nothing in it has been run on a GPU.
Literature survey for this bridge (`docs/IMAGE_TASKS_SURVEY_2026-10-25.md`, another agent) did not exist when this was
written; check it for prior art before any novelty claim.

Declared blocks: A

## Question
Every navigation result in this repo gives the model its observations as TOKENS: a symbol per object class. The headline
torus result (`FACTORIAL_RESULTS.md`, `SIGN_MATCHED_RESULTS.md`): at T = 1024 a one-layer path-integrated MapWM solves the
map (1.000, 8/8) where one-layer index RoPE does not (0.72). **Does that survive when each observation is an IMAGE of its
class, a fresh instance at every visit, recognised by a small CNN trained end to end -- and does the step map, which reads
every token's embedding, now leak per-instance image variation into the "where"?** Two registered contrasts at one layer:
path vs index (P1) and the what-to-where leak (P2: MapWM's step reads the observation embedding; ActOnly's does not).

## Task (the token task, observation tokens swapped for images)
- **Environment: unchanged.** `environment.GridWorld(64, 16 classes, p_empty 0.5, seed = run seed)`, exactly as
  `train_variant` builds it for FACTORIAL: one training map per run (seeded by the run seed), fresh walks every batch,
  walks of 1-10 repeated moves, interleaved `[a1, o1, a2, o2, ...]`, loss on revisited observation positions only,
  evaluation on a held-out map (env seed 10000). NOTE (disagreement with the brief): the brief said "map redrawn per
  sequence as now"; on this torus the training map is fixed per run and only evaluation redraws it (per-sequence maps
  exist only in `environment_newobj.py`). The torus protocol is kept so that every floor and the token twin are FACTORIAL's.
- **The token stream is GridWorld's, byte for byte.** Targets are the class (or blank) tokens; the revisit mask, the
  floors, the strata and the loss are the token task's at the same seed by construction (`gate_image_torus.py` checks the
  trainer's environment against a fresh GridWorld per seed; `test_image_torus.py` too).
- **What the model sees.** `model_image.ImageObsEmbedding` replaces `token_emb`: action tokens keep their learned rows;
  at every observation slot (the odd positions) a CNN encodes an image -- class token k -> a random instance of class k,
  drawn fresh at every forward (so every visit is a new instance; recognition must be invariant); blank -> the all-zero
  image (constant, the image analogue of the token task's single blank token; a noise image was rejected: it would add a
  second invariance problem with nothing to learn from it, and change blank-step behaviour vs the token twin).
- **Images: 16 classes = MNIST digits 0-9 + six Fashion-MNIST classes** (trouser, dress, sandal, sneaker, bag, ankle boot;
  `image_bank.py`). 16 = GridWorld's default class count, so the label stream equals the token task's; 10 classes (MNIST
  alone) would change the class stream, the n-gram floors and chance, and break the token twin. The four top-wear Fashion
  classes are left out (the confusable ones; this asks about the map, not fine-grained perception). Train split: 96,000
  images (5,421-6,742 per class); test split: 16,000 (892-1,135 per class); 0 byte-identical images across splits.
  Canonical files (torchvision's md5s), downloaded 2026-10-25 to `data/image_torus/raw/` (42 MB, gitignored); cache
  `data/image_torus/bank16.npz` (85 MB, gitignored), bank md5 `72cdafaf92a9a42b1b6266ff72423e09` (recorded in every image
  checkpoint and the gate; the analysis voids on a mismatch). torchvision is not installed and is not needed (numpy IDX
  parser). CIFAR-10 (163 MB) is reachable for the harder follow-up; not downloaded.
- **Splits.** Training draws ONLY train-split images (the embedding raises if asked for test images in training mode;
  the trainer asserts zero test draws and records the count). Every registered readout draws ONLY test-split images
  (counted per readout; the analysis voids otherwise): perception transfer on top of map transfer.
- **Encoder.** `ImageEncoder`: conv 1->16 (5x5, stride 2) -> ReLU -> conv 16->32 (3x3, s2) -> ReLU -> conv 32->32 (3x3, s2)
  -> ReLU -> linear 512->128 = d_model; 79,968 parameters (the one-layer transformer is ~204k); no normalisation layer
  (batch statistics would couple sequence positions). Initialised from its own seed (run seed x 7919 + 17) inside
  `fork_rng`, so at a seed every arm has the same encoder and the same base weights as its token twin, and the global RNG
  (dropout stream) is the token twin's (tested). Banks are non-persistent uint8 buffers on the GPU (no loader, no
  checkpoint bloat); the CNN runs on every odd slot (static shapes, no host-device sync in training).

## Recipe (FACTORIAL's)
`python3 -u -m mapformer.train_image --arm ARM --obs {image,token} --n-layers NL --seed S --epochs 900 --lr 1e-3
--n-batches 98 --batch-size 16 --n-steps 1024 --schedule cosine --data-workers 3 --fast-attn --device cuda:G --output-dir
OUT` (AdamW wd 0.05, warmup + cosine, 1 layer unless stated, 2 heads, d 128; TF32 off for matmuls and convs; no
`--deterministic`, as FACTORIAL). `train()` and `GridWorld` are called unchanged. The token twin of MapWM is bitwise
`train_variant --variant Vanilla_r4` at the same seed at initialisation (tested).

## Cells and seeds (one batch, every cell retrained; rule 12)
| cell | arm | obs | layers | block | role |
|---|---|---|---|---|---|
| I_MapWM | MapWM (train_variant `Vanilla_r4`, shared rank 4) | image | 1 | A | path, the model of record |
| I_ActOnly | `model_codes.MapWM_ActOnly` (steps from action tokens only) | image | 1 | A | leak remedy |
| I_RoPE | train_variant `RoPE` (index rotary) | image | 1 | A | index at one layer |
| T_MapWM | `Vanilla_r4` | token | 1 | A | token twin: cost of pixels, token leak |
| I_NormStep | `model_codes.MapWM_NormStep` (step reads LayerNorm(emb)) | image | 1 | B | non-oracle LEAK remedy |
| I_RoPE_L4 | `RoPE`, 4 real layers | image | 4 | B | depth reference |
| T_RoPE | `RoPE` | token | 1 | B | token twin of index |
Seeds **3100-3107** (8 per cell; grep of runs/ and every top-level .md/.sh/.py, 2026-10-25: unused on this machine; koopman
not checked). Pilot seeds 3180-3181 (outside). Block A = 32 runs (declared above); block B = 24 more, optional -- choosing
AB must be an amendment BEFORE launch (the driver refuses if `IMG_BLOCKS` differs from the `Declared blocks:` line).

**Why ActOnly, not NormStep, is the core remedy.** With images the action/observation split is the input MODALITY, so an
action-only step is what a multimodal architecture would do; it is not the oracle label it was on tokens. NormStep removes
only the step's sensitivity to the embedding NORM; per-instance variation in DIRECTION passes LayerNorm, so with images
NormStep is predicted to leak like MapWM. It is a good block-B question, not the decisive one. (LEAK_RESULTS.md also showed
the zeroing readout means something else for NormStep: its LayerNorm bias gives every observation a common step.)

## Floors, gate and perception ceiling (measured now: `gate_image_torus.py` -> `IMAGE_TORUS_GATE.md/json`; no model)
Evaluation walks of seeds 3100-3107 (held-out map, np seed 1234 + s, 100 walks, T = 1024; ~32,500 revisit targets/seed):
| floor (mean over the 8 seeds) | value |
|---|---|
| always-blank | 0.508 (0.504-0.515) |
| retrace (copy across a reversing run; the best trivial predictor) | **0.843** (0.838-0.848) |
| token n-gram, best of orders 0-5 (fitted on the run's training map) | 0.517 |
| action-stream-only n-gram, best of orders 1-5 | 0.508 (= blank: actions alone predict nothing) |
| retrace by stratum: short-gap (83% of targets) / long-gap (9%) / wrap-only (8%) | 0.911 / 0.509 / 0.508 |
| standalone classifier (same encoder + linear head, 10 epochs): test / train accuracy | 0.980 / 0.986 (classes 0.958-0.994) |
| **perception ceiling ceil_vote** (perfect map; Bayes vote of that classifier over all earlier visits) | **0.993** (0.992-0.993) |
| ceil_last (perfect map; recognises the most recent earlier visit only) | 0.990 |
The ceiling bounds what a perfect map can score given recognition this good; the trained encoder may be better or worse
(its own linear probe is secondary S3). Stream checks passed (environment = GridWorld per seed; tokens < 21). Context
destruction is planned on the trained models (S5).

## Registered primary (`analyze_image_torus.py`; accuracy on revisit targets, T = 1024, held-out map, test images)
Exact two-sided permutation test, 8 vs 8 (`stats_core.perm2_p`, 12,870 relabellings, min p 0.00016); **Holm over P1 and
P2** at family alpha 0.05; 95% test-inversion CI (`perm2_ci`).
- **P1: I_MapWM - I_RoPE** (does path integration survive pixels at one layer?).
  PATH WINS IN PIXELS: d >= 0.05 and p_Holm < 0.05. INDEX WINS IN PIXELS: d <= -0.05 and p_Holm < 0.05. WITHIN 0.05:
  CI inside (-0.05, +0.05). UNMEASURED otherwise.
- **P2: I_ActOnly - I_MapWM** (does the observation-driven step cost accuracy with pixels?).
  THE LEAK COSTS IN PIXELS: d >= 0.01 and p_Holm < 0.05. ACTION-ONLY HURTS: d <= -0.01 and p_Holm < 0.05. WITHIN 0.01:
  CI inside (-0.01, +0.01). UNMEASURED otherwise.
- A WITHIN label is suffixed "[NOT LEARNED ...]" when both arms' means are at or below the retrace floor. Every label
  carries, per arm: SOLVED / STALLED / DESCENDING counts (`stats_core.classify_run`: final-5% loss < 0.05), seeds above the
  retrace floor, seeds within 0.01 of ceil_vote (budget scope: "within 900 epochs").
- Thresholds: 0.05 for P1 as FACTORIAL's C0; 0.01 for P2 as LEAK's "within 0.01" and above LEAK's token effect (+0.0107,
  MDE 0.0035). SOLVED may be unreachable for image arms if recognition errors keep the loss above 0.05; that is why the
  primaries are accuracy contrasts and SOLVED is a tag.
- **Void:** a declared run missing; ActOnly's leak readout |zeroobj - test| > 0.002 on any seed (wiring); a training run
  that drew a test-split image (or recorded no train draws), or a registered readout that drew a train-split image; bank
  md5 differs across runs or from the gate's. The md5 guard tripping stops the driver before the analysis.

## Declared secondaries (no verdict)
- **S1 cost of pixels:** I_MapWM - T_MapWM and I_ActOnly - T_MapWM (perm p); each image cell minus ceil_vote, paired by seed.
- **S2 leak:** L = acc(object-observation steps zeroed at eval; actions and blank keep theirs) - acc(intact), per path cell;
  L(I_MapWM) - L(T_MapWM) ("leaks more with images"); step statistics on 10 walks: rms object step / rms action step and the
  WITHIN-CLASS (per-instance) rms of object steps / rms action step (0 by construction for tokens). Ratios within a model
  are invariant to the omega-vs-step scale gauge (rule 8).
- **S3 perception:** linear probe (16-way softmax regression, LBFGS) on each trained encoder's features, fitted on all
  train images, scored on all test images; transfer gap acc(train-split images) - acc(test-split images) on the held-out
  map at T = 1024 (sign-flip p).
- **S4 strata:** short-gap / long-gap / wrap-only accuracy per cell beside the retrace floor on that stratum.
- **S5 context destruction:** observation tokens permuted among the input's observation slots (targets unchanged): a
  map-reading model must fall to about the blank floor (0.51).
- **S6:** run classes and final loss per cell; r(final-5% loss, accuracy) over all runs (rule 2); accuracy at T = 512 and
  2048 (2048 out of distribution, rule 10); the run's own training map (memorisation).
- **P1':** I_ActOnly - I_RoPE. **S7 (block B only):** I_RoPE_L4 - I_RoPE; I_MapWM - I_RoPE_L4; I_NormStep - I_MapWM and
  L(I_NormStep); T_MapWM - T_RoPE; I_RoPE - T_RoPE.
Eval mode is used throughout, as every registered torus batch (the attention-dropout rescale of
`docs/audits/2026-10-04/DROPOUT_RESCORE.md` is not applied).

## Predictions (stated before any run)
- P1 **PATH WINS IN PIXELS**: I_MapWM 0.97-0.99, I_RoPE 0.66-0.75 (token RoPE_L1 0.69-0.74 in FACTORIAL; recognition
  errors can only lower it), d ~ +0.25.
- P2 **THE LEAK COSTS IN PIXELS**, d +0.01 to +0.04: ActOnly within 0.005 of ceil_vote on most seeds; MapWM slower to
  converge (more DESCENDING at 900 epochs) because its object steps carry per-instance noise until the encoder learns to
  keep its output out of W_in's row space.
- S1: T_MapWM 1.000 (8/8 SOLVED); I_ActOnly - T_MapWM about -(1 - ceil_vote) ~ -0.007. S2: L(I_MapWM) > L(T_MapWM);
  instance/action ratio > 0 for I_MapWM, exactly 0 for I_ActOnly. S3: probe >= 0.98; transfer gap 0 to +0.005. S5: all
  cells 0.50-0.56.
- Stated risk: I_MapWM may stall near the retrace floor if early per-instance step noise wrecks the phase code before the
  encoder learns to null it. Then P1 reads UNMEASURED or WITHIN [NOT LEARNED] and P2 fires large; that is a result
  (MapWM's step design does not survive pixels; the action-only step does), not a failure of the test.

## Power (rule 5)
P1: per-cell sd in FACTORIAL ~0.02 (RoPE_L1) and 0 (MapWM): MDE (exact t, 8 vs 8, 80%) ~0.02-0.03 << 0.05. P2: MDE ~1.6 x
the pooled per-cell sd; the 0.01 threshold is detectable if the sd is <= ~0.006 (LEAK's MapWM sd was 0.003). A bimodal
I_MapWM (some seeds stalled) would make P2 large but noisy; the permutation test is used for that reason.

## Cost (rule 13: show it; the pilot replaces these with measurements)
Basis: FACTORIAL on this GPU type (RTX 4090, 4 jobs/GPU): RoPE_L1 49 min, Vanilla_r4 58-61, deep 173-186 per run (wall,
4-way packed). Encoder FLOPs: ~0.52 MMAC/image x 16,368 images/step = 8.5 GMAC forward, vs ~15-23 GMAC for the one-layer
transformer at L = 2047 -> an image run is estimated at **1.5-2.1x** a token run (convs at 16-32 channels run below matmul
efficiency; the pilot measures it; the brief's budget is 1.5-2x). Indicative only, CPU (4 threads, B 16, T 1024,
fwd + bwd per step, 2026-10-25): token MapWM 2.3 s, image MapWM 3.1 s (1.36x; the encoder alone 0.94 s). Memory: ~+0.7 GB activations + 88 MB banks per image job
(~1.7 GB; 4/GPU fits).
- **Block A (32 runs): 24 image runs x 90-126 min + 8 token runs x 60 min = 44-58 lane-hours -> ~5.5-7.3 h on both GPUs
  (4 jobs each), ~11-15 h on one.** Evaluation ~1-2 min per run (split over the GPUs, ~0.5 h). CPU analysis < 1 min.
- Block B (+24 runs): I_NormStep 8 x 90-126, T_RoPE 8 x 49, I_RoPE_L4 8 x 210-246 min = 47-56 lane-hours -> +5.8-7 h on two.
- Disk: ~2.5 MB per checkpoint (< 150 MB for A+B); data 127 MB (raw + cache), already on disk.

## Pilot (`run_image_torus_pilot.sh`; seeds 3180-3181, outside the batch; after both registered batches finish)
`cd /home/prashr/mapformer && DRV_GPUS=1 PILOT_LEARN=1 setsid nohup bash run_image_torus_pilot.sh > /dev/null 2>&1 &`
(~20 min + ~15 min with PILOT_LEARN; PILOT_B=1 adds the 4-layer timing). Stage 1: solo 10 epochs each of T_MapWM,
I_MapWM, I_MapWM with `--cudnn-benchmark`, I_RoPE: s/epoch and peak memory. Stage 2: block A's four arms at once (the
batch's packing), 15 epochs. Stage 3: evaluator wall time per run. Stage 4 (PILOT_LEARN=1): I_MapWM, I_RoPE, T_MapWM for
150 epochs; training loss at 50 / 100 / 150 only (no evaluation read). Pre-declared uses:
- If contended I_MapWM s/epoch > 2.0 x T_MapWM's: adopt `--cudnn-benchmark` if it brings it under 2.0x; otherwise keep the
  design and recompute the cost for the user. The encoder is NOT changed after the pilot without an amendment.
- If at epoch 150 I_MapWM's training loss is not below I_RoPE's while T_MapWM's is: stop, do not launch (the image path is
  not learning the map at all; redesign before spending the batch).
- Cost numbers above are replaced by the measured ones in an amendment.

## Process
- Independent code check (rule 29), blind to results, before launch; findings into an amendment here.
- Commit everything guarded first: the driver refuses on modified tracked files or an uncommitted guarded file.
- Driver `run_image_torus.sh` (sources `lib_driver.sh`): flock single instance; refuses unless `.ssm_sign_a2_done` and
  `.factorial_n8_done` exist (override `DRV_SKIP_MARKERS=1`, logged); `DRV_GPUS`, `DRV_MAXPG`, `IMG_BLOCKS` required; md5
  guard over the import closure of train_image / eval_image_torus / analyze_image_torus / gate_image_torus plus
  `lib_driver.sh`, the driver, THIS FILE, `IMAGE_TORUS_GATE.json/.md`, `test_image_torus.py`, and the data files (bank cache
  + raw image files; md5 only, gitignored), re-checked before every launch, before evaluation and before the analysis;
  seed-major launch order; evaluation split over the GPUs; artifacts before `.image_torus_done`.
- Launch (after commit, the check's amendment, the pilot's amendment):
  `cd /home/prashr/mapformer && IMG_BLOCKS=A DRV_GPUS="0 1" DRV_MAXPG=4 setsid nohup bash run_image_torus.sh > /dev/null 2>&1 &`
- Before reading: `python3 -m mapformer.experiment_audit --runs-dir runs/image_torus/I_MapWM` (exit 2 is expected on this
  layout), `md5sum -c runs/image_torus/code_md5.txt` from the repo root.

## Checks done before the independent check (CPU, 2026-10-25)
- `python3 -m mapformer.test_image_torus`: 11/11 (bank and splits; train-split-only in training, test-split at eval,
  auto split by mode, explicit test split refused in training; fresh instance per visit and paired reseed across arms;
  stream and revisit mask = GridWorld's; the step map's input per arm with negative controls; token twin bitwise
  Vanilla_r4, global RNG untouched by the encoder init, shared base weights across MapWM / ActOnly / NormStep; checkpoint
  round trip without the banks; interleaving check; eval readouts' split counts and the zeroing hook (a no-op for ActOnly
  only); context-destruction shuffle; eval stream = eval_noise_refine's).
- `python3 -m mapformer.analyze_image_torus --selftest`: every P1 / P2 label, the NOT LEARNED suffix, block B, the
  DESCENDING tag, all five void conditions and Holm.
- CPU smoke: every cell of A and B trained 5 epochs (tiny shapes), `--data-workers 2` path once, evaluator on all seven
  checkpoints (every readout and the probe); the real analysis code run on the real eval-JSON structure.
- Driver dry run (`DRV_DRYRUN=1 IMG_BLOCKS=A`): 32 launches, eval and analysis logged; IMG_BLOCKS=AB refused against
  "Declared blocks: A"; md5 guard over 110 files.

## Files
`image_bank.py` (bank, md5s, synthetic test banks), `model_image.py` (encoder, embedding, arms), `train_image.py`,
`eval_image_torus.py`, `gate_image_torus.py` (+ `IMAGE_TORUS_GATE.md/json`), `analyze_image_torus.py` (`--selftest`),
`test_image_torus.py` (CPU), `run_image_torus.sh`, `run_image_torus_pilot.sh`. No existing file is edited; no deferred
patch is needed. (After results: rows for CLAUDE.md / RESULTS_INDEX.md, written then.)

## What this can and cannot show
Can: whether one-layer path integration keeps its advantage over index RoPE when observations are pixels with
instance-level variation and test-split images at evaluation; whether MapWM's observation-reading step costs accuracy
there (and how much more than with tokens). Cannot: anything about natural images, egocentric views or real visual
aliasing (one image per cell visit, allocentric, 28x28, classes that a 80k-parameter CNN recognises at 98%); anything at
other lengths, depths (block A) or recipes; mechanism (ActOnly bundles "no object step" with "no blank step", as in LEAK).
CIFAR-10 is the harder follow-up and would change the ceiling, not the design.

---

## Amendment 1 (2026-10-25, before any run; independent code check `docs/audits/2026-10-25/image_torus_audit.md`)
Results-blind (no run exists; nothing has run on a GPU). The check found 3 MAJOR, 7 MINOR, 10 NOTE, no blocker. Decisions
(coordinator) and what was changed, all in new files only:

1. **md5 guard (M1).** `mapformer.model_codes` (the ActOnly / NormStep classes, imported lazily inside
   `model_image._classes`) was missing from the guard's import closure. The driver now imports it on the guard line and
   aborts unless `model_codes.py`, `model_image.py`, `train_variant.py` and `model_rank.py` are all in the guard list.
2. **Encoder output scale (M2), option (i).** Before: at init an image observation embedding had norm ~0.4 vs ~11.3 for a
   token row, and image MapWM's object steps were 0.057x its action steps (token twin ~1x). Now every image arm multiplies
   the encoder output by a FIXED gain **G = 30.3729**, a registered constant buffer (`ImageObsEmbedding.enc_gain`, saved in
   the checkpoint, not a parameter, so the optimiser sees the same parameterisation in every arm -- rule 15: it is a
   fixed scale of the encoder's output, not a learnable one; AdamW still updates the same encoder weights). Rule (seed-
   independent, `model_image.calibrate_gain`, recomputed in the tests): G = sqrt(128) / (mean over the encoder inits of run
   seeds 0-15 of the mean output norm on the first 64 train images of each class); mean init norm 0.3725 (sd 0.035 over
   the 16 inits). The batch seeds 3100-3107 were not used to set it. Measured at init after the change (gate section 4,
   `IMAGE_TORUS_GATE.md`, 10 evaluation walks per seed):
   | init, seeds 3100-3107 | embedding norm action / object / blank | rms object step / action step | within-class (per-instance) / action |
   |---|---|---|---|
   | token twin (T_MapWM) | 10.8-11.8 / 11.0-11.4 / 9.8-12.6 | **1.055** (0.59-1.31) | 0 (by construction) |
   | image MapWM, gain 30.37 | 10.8-11.8 / 10.5-13.0 / 9.5-11.7 | **0.907** (0.42-1.71) | **0.199** (0.12-0.29) |
   | image MapWM before the gain (audit, s3100) | 11.3 / 0.42 / 0.39 | 0.057 | ~0.05 |
   The object/action ratio now matches the twin's (means 0.91 vs 1.06, overlapping ranges), and image MapWM starts with a
   per-instance step component ~0.2 of an action step. So the prereg's mechanism wording stands: P2 asks whether an
   initial, per-instance observation step is removed (MapWM) or never present (ActOnly). S1 and S2's token-vs-image
   contrasts now compare models with the same observation scale at init. The gain does not touch the gate's standalone
   classifier (scale-free under its linear head); floors, ceilings and classifier regenerated byte-identical.
3. **P2 power (M3), decided as revised below BEFORE ANY DATA.** The Power section's sentence "the permutation test is used
   for that reason" was wrong: with a few slow / stalled MapWM seeds the exact permutation p is bounded by where the
   outliers fall (2 C(16-k, 8-k) / C(16, 8): 0.20 at k = 3, 0.077 at k = 4). A first version of this amendment made a
   Fisher count a second registered route for P2 with Holm over three tests. The simulation below showed that the third
   Holm test halves P2's power at k = 3-4 (0.26 -> 0.14, 0.63 -> 0.28) and that the Fisher route cannot fire below k = 5
   (8 vs 4 within the ceiling has minimum Fisher p 0.077), so it adds nothing where it was meant to help. **Final rule
   (coordinator, before any run): P2 is a single route -- accuracy, exact permutation test -- with Holm over two tests
   (P1, P2), exactly as registered before this amendment.** The count of seeds within 0.01 of their seed's ceil_vote,
   I_ActOnly vs I_MapWM, two-sided Fisher exact (`stats_core.fisher_solved`), is a **declared secondary printed beside P2:
   no verdict, outside Holm**; it cannot change P2's label (selftest case: count 8/8 vs 1/8, Fisher p 0.0014, with d 0.005
   below the 0.01 threshold, P2 reads WITHIN 0.01).
   Power simulation (ActOnly N(0.990, 0.002) x 8; MapWM: k seeds N(0.90, 0.02), the rest N(0.990, 0.002); ceil_vote 0.993;
   P1 at its minimum p; 200 reps; script `power_sim.py` in the session scratchpad):
   | MapWM seeds short (k of 8) | mean d | **final rule: accuracy, Holm over 2** | rejected: accuracy route, Holm over 3 | rejected: Fisher route, Holm over 3 | rejected: either route |
   |---|---|---|---|---|---|
   | 0 | 0.000 | **0.00** | 0.00 | 0.00 | 0.00 |
   | 1 | +0.011 | **0.04** | 0.04 | 0.00 | 0.04 |
   | 2 | +0.023 | **0.09** | 0.07 | 0.00 | 0.07 |
   | 3 | +0.034 | **0.26** | 0.14 | 0.00 | 0.14 |
   | 4 | +0.045 | **0.63** | 0.28 | 0.00 | 0.28 |
   | 5 | +0.057 | **1.00** | 0.98 | 0.98 | 0.98 |
   | 6-8 | +0.067 to +0.091 | **1.00** | 1.00 | 1.00 | 1.00 |
   | uniform shift 0.010 (sd 0.002) | +0.010 | **0.57** | 0.57 | 0.57 | 0.57 |
   | uniform shift 0.015 (sd 0.005) | +0.015 | **1.00** | 1.00 | 1.00 | 1.00 |
   | uniform shift 0.020 (sd 0.010) | +0.020 | **0.99** | 0.99 | 0.99 | 0.99 |
   **Corrected Power text:** P2 detects a consistent shortfall of >= ~0.015 (any spread up to sd 0.01) or a majority of
   short MapWM seeds (k >= 5: power ~1.0; k = 4: 0.63); it does NOT reliably detect 1-3 short seeds out of 8 (power
   <= 0.26), which read UNMEASURED. A uniform shift of exactly 0.010 is a coin flip (0.57) because of the d >= 0.01 floor.
4. **T_ActOnly added to block A (m3).** Block A = I_MapWM, I_ActOnly, I_RoPE, T_MapWM, **T_ActOnly** (40 runs). The pixel
   attribution of the leak is the declared secondary **S2b: (I_ActOnly - I_MapWM) - (T_ActOnly - T_MapWM)**, per seed (the
   four cells at a seed share map, walks and base init), sign-flip test. P2's words "IN PIXELS" name the condition; S2b
   says whether the effect is specific to pixels. S1 gains T_ActOnly vs I_ActOnly. Void: |zeroobj - test| > 0.002 on any
   seed now applies to both ActOnly cells.
5. **Evaluation numerics (m1).** The evaluator turns TF32 off for matmuls and cuDNN, as in training (the pilot's stage 3
   inherits it).
6. **Floor qualifier on every label (m2).** A directional label (PATH / INDEX WINS, LEAK COSTS, ACTION-ONLY HURTS) whose
   winning arm's mean is at or below the mean retrace floor gets "[winning arm at or below the retrace floor]"; WITHIN keeps
   its [NOT LEARNED] suffix. The retrace floor assumes perfect recognition, so it is optimistic for pixel arms.
7. **Analysis hygiene (m4).** Per-seed floors and ceilings are looked up BY SEED, never by position; the analysis voids if
   the gate's seeds differ from the batch seeds. Selftest adds: a case where Holm (not the raw p) changes both labels
   (raw p 0.032 / 0.039 with d above both thresholds -> Holm 0.063 / 0.063, both UNMEASURED), PATH WINS with the winner
   below retrace, the "no train draws recorded" void, the gate-seed void, and a case showing the within-ceiling Fisher
   count is a secondary line only (significant count, P2 label unchanged).
8. **Rule 11 (m5).** `train_image.make_env(seed)` is the one environment construction; the trainer, the gate's stream
   check and the tests call it. The gate's evaluation walks are `eval_image_torus.walks` itself, and the test compares
   them against `eval_noise_refine.evaluate` CALLED with a recording stub model (with a different-seed negative control).
   The trainer also checks interleaving once on a batch drawn with the numpy RNG saved and restored (walk stream unchanged;
   audit n4). The full train x test duplicate scan is now a test (0 duplicates; n7).
9. **Blocks line (m6).** The driver reads the LAST "Declared blocks:" line (below); no line still refuses.
10. **Run classes (m7).** SOLVED / STALLED / DESCENDING are TAGS only, in every cell, never a branch input. In image arms
    the training loss has a recognition floor (train-split errors), so STALLED is not "stuck" and SOLVED may be unreachable;
    read them with "within 0.01 of ceil_vote" and the DESCENDING count.

**Notes recorded (no change).** (n8) The markers `.ssm_sign_a2_done` and `.factorial_n8_done` make the driver REFUSE, not
wait: launch only after both exist; if either batch ends VOID (no marker), launch with `DRV_SKIP_MARKERS=1` (logged). The
driver counts only this user's trainers and free memory (>= 4000 MiB); check other users' GPU jobs by hand at launch
(`nvidia-smi`, `feedback_gpu_sharing_and_cost.md`). (n6) **P1 is a total effect** (map + learned perception: I_RoPE's
encoder may get a weaker gradient because it cannot find the earlier visit); S3's linear probe separates perception from the
map. (n1) n-gram floors are fitted on one training map per seed (retrace dominates). (n2) ceil_vote comes from a standalone
classifier and can be exceeded by a trained encoder. (n3) Evaluation pairing across the two GPUs relies on both being RTX
4090 (same Philox output); the analysis is unpaired except S2b, whose cells are evaluated on GPUs fixed by the round-robin
(identical images need identical devices; a cross-device mismatch would add noise, not bias). (n5) Eval mode without the
attention-dropout rescale, as declared. (n9) Pilot stage 4 is a compressed 150-epoch cosine: a smoke check, not a learning
curve; its memory baseline is read once. (n10) P2's 0.01 threshold is well above evaluation sampling noise (binomial SE
~0.0006 at 0.99 on ~32,500 targets); on this torus T_MapWM's token leak cost is expected to be 0.

**Cost (replaces the block-A line of Cost).** Block A, 40 runs: 24 image runs x 90-126 min + 16 token runs x 60 min =
52-66 lane-hours -> **~6.5-8.3 h on both GPUs (4 jobs each), ~13-16.6 h on one**, plus evaluation ~0.5-0.7 h (split over the
GPUs). Block B unchanged (+5.8-7 h on two GPUs). Still to be replaced by the pilot's measurements.

**Checks re-run after this amendment (CPU, 2026-10-25):** `test_image_torus` 12/12 (adds the gain test and the
eval_noise_refine call test); `analyze_image_torus --selftest` PASSED (18 checks, after the M3 revision); gate regenerated (floors, ceilings,
classifier byte-identical; init-scale section added); CPU smoke of all eight cells (A and B) and the evaluator; driver dry
run in a scratch clone (below).

Declared blocks: A
