# Does path integration survive pixel observations? The image torus -- pre-registration (DRAFT, 2026-10-25, before any run)

Status: DRAFT for the independent check (rule 29). Not committed, not launched. Nothing in it has been run on a GPU.
[SUPERSEDED STATUS, Amendment 3: registration and Amendment 1 committed at 4955021; pilots 1 (seeds 3180-3181) and 2
(3182-3183) ran on GPUs; Amendments 2-3 redesign the registered contrasts; the batch (seeds 3100-3107) has not run.]
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
not checked). Pilot seeds 3180-3181 (outside) [superseded: pilot seeds 3180-3183, Pilots section]. Block A = 32 runs
(declared above) [superseded: 48 runs, Amendment 2 item 4]; block B = 24 more, optional -- choosing
AB must be an amendment BEFORE launch (the driver refuses if `IMG_BLOCKS` differs from the `Declared blocks:` line).

[Framed for the original P2; superseded in part by Amendment 2: ActOnly is now P1's path arm.]
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
[SUPERSEDED by Amendment 2's Predictions; kept as registered. The I_MapWM prediction (0.97-0.99) was contradicted by
the pilots.]
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
   per-instance step component ~0.2 of an action step. [The next two sentences are SUPERSEDED by Amendment 2 item 6.]
   So the prereg's mechanism wording stands: P2 asks whether an
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

---

## Pilots (outside seeds; never pooled with the batch; rule 29)

### Pilot 1 (`run_image_torus_pilot.sh`, seeds 3180-3181, kalman GPU 0, 2026-10-25 02:41-03:00; `IMAGE_TORUS_PILOT.txt`)
- Timing (stage 1 solo / stage 2 four jobs on one GPU): T_MapWM 0.9 / 4.2 s/epoch; I_MapWM 2.1 / 5.8; I_ActOnly - / 5.8;
  I_RoPE 2.0 / 6.9; `--cudnn-benchmark` 1.7 solo. Peak +1.0 GB (token) / +1.5 GB (image) per job, +5.6 GB for four.
  Evaluator 10 s per image run, 2 s per token run. Pre-declared timing rule: contended image / token = 5.8 / 4.2 = 1.38x
  < 2.0x, so `--cudnn-benchmark` is NOT adopted.
- **Pre-declared stop rule FIRED** (stage 4, s3181, gain 30.37, 150 epochs, training loss at 50 / 100 / 150): I_MapWM 2.049 /
  2.047 / 2.046 -- the memoryless level (ln 2 + 0.5 ln 16 = 2.079): it never learned -- vs I_RoPE 1.288 / 1.228 / 1.200 and
  T_MapWM 0.912 / 0.781 / 0.518. In stage 2 (15 epochs) I_ActOnly had reached 0.654 (T_MapWM 1.048). Per the rule the batch
  was not launched. Stage-4 evaluations were not read.

### Pilot 2, diagnostic (`run_image_torus_pilot2.sh`, seeds 3182-3183, kalman GPUs 0-1 (2 jobs each; no other user's jobs),
2026-10-25 12:02-12:15; `IMAGE_TORUS_PILOT2.txt`; code md5 in `runs/image_torus_pilot2/code_md5.txt`)
150 epochs, pilot-1 stage-4 recipe; training loss at 50 / 100 / 150 and, as a pilot, the full held-out evaluation:
| run | arm | encoder gain | seed | lr | loss 50 / 100 / 150 | held-out acc T=1024 | zeroobj | shuffle | probe | step object/action, instance/action |
|---|---|---|---|---|---|---|---|---|---|---|
| P2a | ActOnly | 30.37 | 3182 | 1e-3 | 0.174 / 0.081 / 0.015 | **0.9944** | 0.9944 | 0.258 | 0.984 | 0, 0 |
| P2b | MapWM | 1.0 | 3182 | 1e-3 | 0.190 / 0.063 / 0.015 | **0.9927** | 0.9376 | 0.236 | 0.983 | 0.003, 0.001 |
| P2c | MapWM | 30.37 | 3183 | 1e-3 | 2.071 / 2.069 / 2.073 | **0.5027** (= blank) | 0.5027 | 0.503 | **0.129** | 4.27, 0.000 |
| P2d | MapWM | 30.37 | 3182 | 3e-4 | 2.063 / 2.081 / 2.080 | **0.5086** (= blank) | 0.5086 | 0.509 | 0.870 | **572, 122** |
Readings (pilot, n = 1 each, descriptive):
- The stuck run replicates: image MapWM at the init-matched gain is at the memoryless level on all three seeds tried (3181,
  3183; 3182 at lr 3e-4). A lower learning rate does not rescue it, so it is not simply a step-size issue at 1e-3.
- With gain 1.0 the same MapWM learns, as fast as ActOnly (0.9927 vs 0.9944, both at or above ceil_vote 0.993 for that
  seed's walks), and it USES its small observation steps: zeroing them costs 0.055 (LEAK's "systematic small step" pattern).
- The working reading "per-instance image variation enters the step map at token scale and scrambles the phase" fits P2d
  (object steps 572x, per-instance part 122x the action step; the encoder still carries class information, probe 0.87).
  P2c ended differently: the encoder COLLAPSED (probe 0.13, chance 0.0625; per-instance step part 0) to a constant output,
  which removes the noise and the information together -- also memoryless. Both are failures of training when images
  drive the step map strongly; which end state a run reaches is not predictable from these four runs.
- I_ActOnly at gain 30.37 learns (P2a 0.9944; probe 0.984). Context destruction (shuffle) falls below the blank floor in the
  learned runs (0.24-0.26): they predict confidently from the destroyed context.

---

## Amendment 2 -- redesign after the stop rule (DRAFT, 2026-10-25; not committed, not launched; independent re-check pending)

**Why.** Pilot 1's pre-declared stop rule fired (image MapWM at the init-matched gain never left the memoryless level), and
the diagnostic pilot 2 showed it on three seeds and at a lower learning rate, while ActOnly (same gain) and MapWM at gain 1.0
learn to ~0.99. The registered P1 (image MapWM vs image RoPE) would compare a model that cannot start against index RoPE: it
would answer "does MapWM's step design survive pixels" (pilots: no, at gain 30.37), not "does path integration survive
pixels". The design is changed before any batch seed (3100-3107) has been run.

**Changes (decided by the coordinator with the user's approval of a redesign; drafted here):**
1. **P1 (registered): I_ActOnly - I_RoPE**, image path (action-only step, encoder gain 30.37) vs image index RoPE, one layer:
   "does path integration survive pixel observations?" Labels, thresholds (0.05), floor suffixes and Holm unchanged. The
   old P1 (I_MapWM - I_RoPE) becomes the declared secondary P1'.
2. **P2 (registered): I_ActOnly - I_MapWM** (image MapWM at gain 30.37), the what-to-where leak, unchanged in form (0.01
   threshold, exact permutation test, Holm over P1 and P2). **Disclosed as PILOT-INFORMED, NOT BLIND:** its direction and
   rough size are known from pilots 1-2 (three seeds, MapWM at the memoryless level, ~0.50 vs ~0.99). Its role is
   confirmation on 8 fresh seeds and the size of the effect, not discovery.
3. **Registered secondary G (own label, outside Holm): I_MapWM_g1 - I_MapWM**, image MapWM with encoder gain 1.0 vs 30.37 --
   does the cost scale with how strongly images enter the step map? LEAK SCALES WITH GAIN: d >= 0.05 and perm p < 0.05;
   GAIN DOES NOT HELP: the 95% CI of d lies below +0.05; AS IT FALLS otherwise. Beside it, I_ActOnly - I_MapWM_g1 (the
   residual cost at gain 1; d and perm p, no label). Added because pilot 2 showed gain 1 learns (P2b). Also pilot-informed
   (one seed).
4. **Cells.** Block A = I_ActOnly, I_MapWM (gain 30.37), I_RoPE, **I_MapWM_g1** (new; `--enc-gain 1.0`), T_MapWM, T_ActOnly
   = 6 cells x 8 seeds = **48 runs**, seeds 3100-3107 (unchanged, still unused). Block B unchanged and still optional.
   `train_image.py --enc-gain FLOAT` (default ENC_GAIN = 30.3729) is the runtime flag; a non-default gain is part of the
   cell name (`_g1`) and of the checkpoint config, and the persistent `enc_gain` buffer is restored by a strict load (test).
5. **Secondaries kept:** S1 (cost of pixels), S2 (leak readouts, step statistics) now also for I_MapWM_g1, S2b (pixel
   attribution (I_ActOnly - I_MapWM) - (T_ActOnly - T_MapWM)), S3-S6, the within-ceiling Fisher count beside P2, P1'
   (I_MapWM - I_RoPE). Voids unchanged (ActOnly wiring applies to I_ActOnly and T_ActOnly).
6. **The original mechanism wording** ("per-instance step noise until the encoder learns to keep its output out of W_in's
   row space") is withdrawn as a prediction for gain 30.37: in the pilots the encoder did not learn to null it within 150
   epochs; it either kept the noise (P2d) or collapsed (P2c).

**Predictions (stated before the batch; informed by the pilots, so P2 and G are not blind):**
- P1 PATH WINS IN PIXELS: I_ActOnly 0.98-0.995 (pilot 0.9944 at 150 epochs), I_RoPE 0.66-0.75, d ~ +0.25.
- P2 THE LEAK COSTS IN PIXELS: I_MapWM at the blank level (0.50-0.52) on most or all seeds, d ~ +0.45-0.49 (a seed that
  escapes would shrink d but not the label at these values).
- G LEAK SCALES WITH GAIN: I_MapWM_g1 0.98-0.995, d ~ +0.45; residual I_ActOnly - I_MapWM_g1 within +/-0.01.
- T_MapWM and T_ActOnly ~1.000; S2b ~ +0.47. Run classes are tags (Amendment 1, item 10).

[The Power paragraph below is SUPERSEDED by Amendment 3 item 4.]
**Power.** P1 and P2 effects predicted at 5-50x the MDE; the Amendment-1 power table (minority-short regime) is no longer
the relevant case for P2, whose predicted shortfall is on all seeds.

**Cost (pilot-measured).** Four jobs per GPU: image runs 5.8 s/epoch (900 epochs ~87 min), token runs 4.2 s/epoch (~63
min). Block A: 32 image x 87 + 16 token x 63 = 3,792 lane-min = 63 lane-hours -> **~7.9 h on both GPUs (4 jobs each)**,
~15.8 h on one; evaluation ~10 s per image run (~5 min in all). Block B adds ~6 h on two GPUs. GPUs are shared with another
user (danmuir): check `nvidia-smi` and their processes by hand before launch.

**Checks after this amendment (CPU, 2026-10-25):** `test_image_torus` 13/13 (adds the `--enc-gain` flag test);
`analyze_image_torus --selftest` PASSED (21 checks: P1 now ActOnly vs RoPE in every case, the Holm-changes-labels case
rebuilt, the three G labels); driver dry run in a scratch clone: 48 launches with `--enc-gain 1.0` on I_MapWM_g1 only, eval
and analysis logged, md5 guard over 111 files.

**Launch (after commit, the independent re-check and its amendment):**
`cd /home/prashr/mapformer && IMG_BLOCKS=A DRV_GPUS="0 1" DRV_MAXPG=4 setsid nohup bash run_image_torus.sh > /dev/null 2>&1 &`

Declared blocks: A

---

## Amendment 3 (2026-10-25, before any batch run; independent re-check `docs/audits/2026-10-25/image_torus_amendment2_audit.md`)
Results-blind for the batch (no run of seeds 3100-3107 exists; re-confirmed below). The re-check found 1 MAJOR, 6 MINOR,
9 NOTES, no blocker. Trainer, model and driver unchanged in behaviour: the default gain was verified bitwise against
4955021 on CPU, `--enc-gain` end to end, dry run 48 launches, md5 guard 111 files. Changes, analysis and text only:

1. **P2 wording and end states (M1).** P2's labels become **THE OBSERVATION STEP COSTS IN PIXELS** / ACTION-ONLY HURTS /
   WITHIN 0.01 / UNMEASURED. The rule, the threshold, the test and Holm are unchanged. The pilots showed two failures at
   gain 30.37: a per-instance step that scrambles the phase (a leak, P2d) and a collapsed encoder (probe 0.13, P2c). So the
   label does not assert "leak". Beside P2, a declared descriptive line (no verdict) gives the end state of each image
   MapWM seed, with thresholds fixed now, before any batch data:
   - **LEARNED**: accuracy at T = 1024 above that seed's retrace floor.
   - **COLLAPSED**: otherwise, if the encoder's linear-probe test accuracy is below 0.5.
   - **NOISY**: otherwise, if the within-class (per-instance) step / action step is at least 1 (probe at least 0.5).
   - **OTHER**: anything else.
   - The states are tested in that order, and counts per state are printed. The same line is printed for I_MapWM_g1
     beside G.
   - S2 (leak readout, step ratios) and S3 (probe, transfer gap) are printed per cell as median and range, not mean: a
     single seed at a step ratio of 572 would dominate a mean.
   - "Leak" is to be used in reports only for NOISY seeds.
2. **Disclosure in the output (m1, m2).** The analysis prints the following strings on the label lines and writes them,
   with `pilot_informed`, to `IMAGE_TORUS_VERDICTS.json`:
   - P2: "[PILOT-INFORMED, NOT BLIND: direction known from pilots 1-2 (3 seeds at 150 epochs)]".
   - G: "[PILOT-INFORMED, NOT BLIND: one pilot seed]".
   - P1, which is **partly pilot-informed**: "[partly pilot-informed: image ActOnly 0.9944 on one pilot seed, image RoPE's
     pilot training loss known; image RoPE accuracy unseen]". Image RoPE's pilot loss was 1.20 at epoch 150.
   - Image RoPE's accuracy has never been measured. The pilot-1 checkpoint `runs/image_torus_pilot/learn/I_RoPE/I_RoPE.pt`
     will not be evaluated before the batch is read.
3. **The original P1 (m3).** P1' (I_MapWM - I_RoPE) is printed with the ORIGINAL P1 label rule applied to its raw p,
   marked "original P1 rule, outside Holm, superseded by Amendment 2", and stored as `secondary.P1_original` in the verdicts.
   - The pre-pilot P1 prediction (I_MapWM 0.97-0.99) was **contradicted** by the pilots (memoryless on three seeds).
   - The registration's stated risk anticipated this outcome ("that is a result").
   - The pilot's pre-declared stop rule required the redesign. Both readings are reported.
4. **P2 power (m4), replacing Amendment 2's Power paragraph.** Simulation from the re-check: ActOnly N(0.990, 0.002) x 8;
   k image MapWM seeds escape to N(0.990, 0.002), the rest stay at N(0.505, 0.003); Holm over two; 100 repetitions.

   | MapWM seeds that escape | median d | P(P2 fires) |
   |---|---|---|
   | 0-3 | +0.485 to +0.303 | 1.00 |
   | 4 | +0.243 | 0.71 |
   | 5 / 6 / 7 | +0.18 / +0.12 / +0.06 | 0.19 / 0.13 / 0.07 |

   - **P2 fires if at most 3 of 8 image MapWM seeds escape within 900 epochs.** With 4 it probably fires (0.71).
   - With 5 or more it reads UNMEASURED although d stays at least ~0.18: the exact permutation p is bounded by where the
     escaped seeds fall.
   - The pilots ran a 150-epoch cosine, so their runs were still near peak learning rate when they stopped. The 900-epoch
     schedule keeps the rate near peak far longer (93% of peak at epoch 150), so seeds escaping is the live risk; the
     end-state line will show it.
5. **G labels (m5).**
   - **GAIN 1 RESCUES MAPWM**: d >= 0.05 and perm p < 0.05.
   - **NO RESCUE >= 0.05**: 95% CI of d below +0.05.
   - **AS IT FALLS**: otherwise.
   - The gain multiplies the whole image embedding: the step map's input AND the residual stream / attention values. G is
     therefore a total effect of the encoder-output scale at two values. It is neither a dose-response ("scaling") claim
     nor specific to the step map.
   - P2 is the step-specific contrast: ActOnly and MapWM differ only in the step at gain 30.37, and ActOnly learns there.
     An interaction with the residual-stream scale is not excluded.
6. **Text (m6).** Superseded markers added, without rewriting registered text, to:
   - the Status line;
   - the pilot-seed and block-A lines;
   - the "core remedy" framing;
   - the main Predictions section;
   - the last sentences of Amendment 1 item 2;
   - Amendment 2's Power paragraph.
7. **Notes.**
   - **Cost** with image RoPE at its measured 6.9 s/epoch (8 x 104 min instead of 87): 3,928 lane-min = **~65.5 lane-hours,
     ~8.2 h on two GPUs** (4 jobs each), ~16.4 h on one; plus launch spacing.
   - **Gain void added** (n4): I_* cells other than I_MapWM_g1 must record gain 30.3729 (+/-1e-4), I_MapWM_g1 must record
     1.0, else VOID.
   - **End-to-end gain test added** (n5): `train_image.main` with `--enc-gain 1.0` produces `I_MapWM_g1.pt` with config and
     buffer 1.0, the evaluator restores it, and the analysis gain constant equals ENC_GAIN.
   - **Block B (n7).** If block B is ever declared, its depth reference reads I_RoPE_L4 against I_ActOnly, not the
     non-learning I_MapWM.
   - Stale driver comment fixed (n8).
   - **P1's wording (n9).** I_ActOnly's step never sees an image, so P1 asks: given a position code driven by actions
     alone, can content attention match pixel instances across visits well enough to beat index RoPE? It is still a total
     effect (map plus learned perception). The token reference for index RoPE at one layer is FACTORIAL's 0.72; it is
     stored and descriptive only (T_RoPE is in block B).
   - P2's cells are evaluated on different GPUs (both RTX 4090; P2 is unpaired).
8. **Checks re-run (CPU, 2026-10-25):**
   - `test_image_torus` 14/14.
   - `analyze_image_torus --selftest` PASSED, 23 checks. New checks: the renamed P2 and G labels, the gain void, the end
     states (2 LEARNED / 3 COLLAPSED / 2 NOISY / 1 OTHER on synthetic data), the three disclosure strings in the output
     and the JSON, P1' under the original rule, medians.
   - Dry run in a scratch clone: 48 launches.
   - Seeds 3100-3107: still no run directory and no use outside the image-torus files.

**Launch** (after commit; GPUs shared with danmuir -- check `nvidia-smi` and their processes by hand first):
`cd /home/prashr/mapformer && IMG_BLOCKS=A DRV_GPUS="0 1" DRV_MAXPG=4 setsid nohup bash run_image_torus.sh > /dev/null 2>&1 &`

Declared blocks: A
