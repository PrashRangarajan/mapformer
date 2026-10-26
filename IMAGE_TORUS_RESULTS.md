# Does path integration survive pixel observations? The image torus -- results (registered, `IMAGE_TORUS_PREREG.md` + Amendments 1-3)

Batch 2026-10-25 12:31 -> 23:55 (11 h 24 min, including evaluation; the amendment estimated ~8.2 h) on kalman, both RTX
4090s (4 jobs each), code `6c4fdf4` (clean; md5 guard over 111 files, data files included), torch 2.10.0+cu128. Block A
only: 6 cells x 8 seeds = 48 runs, seeds 3100-3107; pilots on seeds 3180-3183 (outside, never pooled). Task: the
FACTORIAL torus (64 x 64, 16 classes, p_empty 0.5, trained and tested at T = 1024, 900 epochs, lr 1e-3 cosine, batch 16,
1 layer, 2 heads, d 128, --fast-attn, no --deterministic) with each observation token replaced by a 28 x 28 image of its
class (MNIST digits 0-9 + six Fashion-MNIST classes), a fresh instance at every visit, train-split images in training,
test-split images at every registered readout, read by a small CNN (79,968 params) trained end to end. Registered output
`IMAGE_TORUS_ANALYSIS.txt`, `IMAGE_TORUS_VERDICTS.json`; floors `IMAGE_TORUS_GATE.md/json`; run dirs `runs/image_torus/`
(gitignored checkpoints, `eval_image.json` per run). The analysis was re-run here read-only (verdicts to a scratch path):
text identical, verdicts JSON equal, no repo file changed.

Words used below. *Path* = a path-integrated phase (MapWM family). *ActOnly* = the step is computed from action tokens
only, so an image never moves the phase. *MapWM* = every token, images included, sets a step. *Gain* = a fixed scale on
the encoder output: 30.37 matches an image embedding's norm to a token embedding's at initialisation (Amendment 1), 1.0 is
the raw encoder output (~30x smaller). *Retrace* = the best map-free rule (copy the observation across a run of moves that
undoes the previous run). *ceil_vote* = the perception ceiling: a perfect map with a standalone classifier voting over
every earlier visit.

## Reading (registered; Holm over P1 and P2)

**Path integration survives pixel observations.** One path-integrated layer whose step is driven by the actions learns the
1024-move torus from images to 0.9946 (8/8 seeds SOLVED, every seed within 0.01 of the perception ceiling and above it),
against 0.7010 for one-layer index RoPE (0/8, every seed below the retrace floor). It keeps long-gap (0.994) and wrap-only
(0.994) returns where retrace scores 0.51.

**An image-driven step at token scale breaks training.** When images also set the step at the scale of a token (gain
30.37), 5 of 8 runs end at the memoryless level (their encoder collapses to a constant), and the 3 that learn stay
below the ceiling (0.920-0.962). The same model at gain 1, where an image moves the phase by 0.001-0.006 of a move,
learns on 8/8 seeds to the ceiling (0.9938).

| contrast | registered label (verbatim, with its disclosure) | effect | test |
|---|---|---|---|
| **P1** I_ActOnly - I_RoPE | **PATH WINS IN PIXELS** [partly pilot-informed: image ActOnly 0.9944 on one pilot seed, image RoPE's pilot training loss known; image RoPE accuracy unseen] | **+0.294** (0.9946 vs 0.7010), 95% CI [+0.282, +0.306] | perm p 0.00016, Holm 0.0003 |
| **P2** I_ActOnly - I_MapWM | **THE OBSERVATION STEP COSTS IN PIXELS** [PILOT-INFORMED, NOT BLIND: direction known from pilots 1-2 (3 seeds at 150 epochs)] | **+0.327** (0.9946 vs 0.6677), 95% CI [+0.167, +0.487] | perm p 0.00016, Holm 0.0003 |
| **G** (registered secondary, outside Holm) I_MapWM_g1 - I_MapWM | **GAIN 1 RESCUES MAPWM** [PILOT-INFORMED, NOT BLIND: one pilot seed] | **+0.326** (0.9938 vs 0.6677), 95% CI [+0.167, +0.487] | perm p 0.00016 |
| P1' (original P1 rule on the raw p, outside Holm, superseded by Amendment 2) I_MapWM - I_RoPE | **UNMEASURED** | -0.033 (0.6677 vs 0.7010), 95% CI [-0.198, +0.127] | perm p 0.728 |

Beside P2 (declared, no verdict): seeds within 0.01 of their ceil_vote, I_ActOnly 8/8 vs I_MapWM 0/8 (Fisher p 0.00016).
Beside G (no label): residual I_ActOnly - I_MapWM_g1 +0.0007 (perm p 0.091, CI [+0.0002, +0.0012]), far inside the 0.01
that P2 would call a cost. The pre-pilot prediction for the original P1 (image MapWM 0.97-0.99) was **contradicted**: by
the pilots (memoryless on three seeds) and by the batch (0.668). The registration's stated risk named this outcome.

## The six cells (T = 1024, held-out map, test-split images; n = 8 each)

| cell | observations | step reads | gain | acc (mean +/- sd) | per seed (3100 ... 3107) | classes | within 0.01 of ceil_vote | probe (median) |
|---|---|---|---|---|---|---|---|---|
| I_ActOnly | images | actions only | 30.37 | **0.9946 +/- 0.0009** | 0.9954 0.9937 0.9937 0.9933 0.9952 0.9957 0.9945 0.9951 | 8 S | 8/8 | 0.985 |
| I_MapWM_g1 | images | every token | 1.0 | **0.9938 +/- 0.0007** | 0.9930 0.9937 0.9927 0.9944 0.9943 0.9937 0.9945 0.9945 | 8 S | 8/8 | 0.984 |
| I_MapWM | images | every token | 30.37 | 0.6677 +/- 0.2207 | 0.9197 0.5052 0.5076 0.5037 0.5141 0.9616 0.9200 0.5094 | 1 S, 5 T, 2 D | 0/8 | 0.071 (range 0.071-0.978) |
| I_RoPE | images | (index) | 30.37 | 0.7010 +/- 0.0161 | 0.6778 0.6957 0.6931 0.7220 0.7025 0.6875 0.7244 0.7047 | 8 T | 0/8 | 0.965 |
| T_ActOnly | tokens | actions only | -- | 0.9998 +/- 0.0005 | 1.0000 1.0000 1.0000 0.9986 1.0000 1.0000 1.0000 1.0000 | 8 S | 8/8 | -- |
| T_MapWM | tokens | every token | -- | 0.9595 +/- 0.1119 | 1.0000 0.9988 0.9999 0.9949 1.0000 0.6827 1.0000 1.0000 | 7 S, 1 T | 7/8 | -- |

Floors on these walks (model-free, mean over the 8 seeds): always-blank 0.5079, **retrace 0.8432** (per seed 0.838-0.848),
token n-gram 0.5166, action-only n-gram 0.5079 (= blank). **Perception ceiling ceil_vote 0.9927** (per seed 0.992-0.993;
standalone classifier test accuracy 0.9801). Retrace by stratum: short-gap 0.911, long-gap 0.509, wrap-only 0.508. Classes:
S = SOLVED, T = STALLED, D = DESCENDING (`stats_core.classify_run`, final-5% loss < 0.05; tags only, Amendment 1 item 10).
Probe = linear probe of the trained encoder on test-split images (16 classes; chance 0.0625; 0.0709 is the share of the
most frequent test class, i.e. a constant feature).

### P2's end states, per seed (declared descriptive; thresholds fixed in Amendment 3 before the batch)

| seed | end state | acc T = 1024 | retrace | probe | object / action step | per-instance / action step | zeroing object steps | final loss (class) |
|---|---|---|---|---|---|---|---|---|
| 3100 | LEARNED | 0.920 | 0.842 | 0.978 | 0.030 | 0.001 | 0.821 | 0.131 (D) |
| 3101 | COLLAPSED | 0.505 (= blank) | 0.838 | 0.071 | 26.5 | 0 | 0.505 | 2.080 (T) |
| 3102 | COLLAPSED | 0.508 (= blank) | 0.844 | 0.071 | 22.9 | 0 | 0.508 | 2.102 (T) |
| 3103 | COLLAPSED | 0.504 (= blank) | 0.841 | 0.071 | 17.6 | 0 | 0.504 | 2.058 (T) |
| 3104 | COLLAPSED | 0.514 (= blank) | 0.846 | 0.071 | 38.0 | 0 | 0.514 | 2.073 (T) |
| 3105 | LEARNED | 0.962 | 0.843 | 0.978 | 0.003 | 0.0001 | 0.968 | 0.050 (S) |
| 3106 | LEARNED | 0.920 | 0.848 | 0.972 | 9.4 | 0.014 | 0.520 | 0.318 (D) |
| 3107 | COLLAPSED | 0.509 (= blank) | 0.844 | 0.071 | 48.4 | 0 | 0.509 | 2.073 (T) |

Counts: LEARNED 3, COLLAPSED 5, NOISY 0, OTHER 0. Gain-1 MapWM: LEARNED 8/8. Step ratios are rms over 10 walks, relative
to the action step (gauge-invariant within a model). Final loss = final-5% training-loss mean; the memoryless level is
ln 2 + 0.5 ln 16 = 2.079.

- **No seed of the batch is NOISY**, so by Amendment 3 the word "leak" does not apply to this batch: the per-instance step
  that scrambled the phase in pilot run P2d (572x / 122x the action step) did not recur in 900-epoch training.
- **Collapsed seeds** score exactly their seed's always-blank floor at every length and under every readout, with the loss
  at the memoryless level at the end (2.06-2.10). Four never left it (s3101: 2.091 at epoch 5, 2.080 at 900); s3103
  dipped to 1.91 at epoch 25, during warm-up, and was back at 2.03 by epoch 35. Their encoder outputs one constant (probe 0.0709,
  object step = blank step, per-instance step 0). The action steps shrank to 0.04-0.15 while that constant observation
  step stayed at 1.0-7.5, 17-48x an action step (post hoc reading of the registered step statistics: the phase is then a
  clock that ticks at every observation slot, which carries no map).
- **Learned seeds** escape early (s3100, s3105: loss ~0.54-0.63 by epoch 50) or late (s3106: 2.06 at epoch 100, 0.46 at
  400). Two null their observation step (0.03x, 0.003x); s3106 keeps a large constant one (9.4x) and depends on it
  (zeroing it: 0.920 -> 0.520). None reaches the ceiling within 900 epochs; two are still descending.
- Post hoc, descriptive: the initial object / action step ratio (gate, Amendment 1) does not separate the outcomes
  (learned seeds 0.42-1.71, collapsed 0.44-1.39).

## Declared secondaries (no verdict)

**S1 cost of pixels.** For the path model with action-driven steps, pixels cost **0.005**: I_ActOnly - T_ActOnly -0.0052
(perm p 0.00016, CI [-0.0057, -0.0047]), less than the standalone classifier's error (1 - 0.980) and of the size of the
ceiling's shortfall (1 - 0.9927 = 0.007). Each image cell minus its seed's ceil_vote, paired: I_ActOnly **+0.0019** (min
+0.0006: every seed above the ceiling; the trained encoder probes at 0.985 against the standalone classifier's 0.980),
I_MapWM_g1 +0.0011 (min +0.0002), I_RoPE -0.292, I_MapWM -0.325. I_MapWM - T_MapWM -0.292 (perm p 0.007); I_ActOnly -
T_MapWM +0.035 (perm p 0.9994; T_MapWM's one stalled seed). Index RoPE on images (0.701) is ~0.02 below the token one-layer
RoPE of FACTORIAL (0.719; FACTORIAL_N8 0.729; cross-batch, descriptive: T_RoPE is block B, not run).

**S2 leak readouts** (L = accuracy with object steps zeroed at evaluation - intact; median [range]; step ratios to the
action step):

| cell | L | object / action step | per-instance / action step |
|---|---|---|---|
| I_MapWM | +0.000 [-0.400, +0.007] | 20.3 [0.003, 48.4] | 0.000 [0.000, 0.014] |
| I_MapWM_g1 | -0.047 [-0.090, -0.001] | 0.0027 [0.0012, 0.0058] | 0.0007 [0.0006, 0.0008] |
| T_MapWM | -0.194 [-0.252, -0.103] | 0.022 [0.013, 0.150] | 0 (by construction) |
| I_ActOnly, T_ActOnly | 0 (wiring check passed) | 0 | 0 |

L(I_MapWM) - L(T_MapWM) **+0.132** (perm p 0.028; the registered name of this direction is 'leaks more with images'). It
is not a leak here: the image MapWM median is 0 because collapsed models are memoryless (zeroing anything changes
nothing), while token MapWM uses its small systematic object steps (zeroing them costs 0.19). Gain-1 MapWM also uses its
small object steps (-0.047), as in pilot P2b and LEAK's systematic small step.

**S2b pixel attribution** (Amendment 1 item 4): (I_ActOnly - I_MapWM) - (T_ActOnly - T_MapWM) **+0.287**, paired by seed,
sign-flip **p 0.039**; the token difference T_ActOnly - T_MapWM is +0.040 (carried by T_MapWM s3105, 0.683). The cost of
an observation-driven step is specific to pixels at this gain, not a property of MapWM's step on this torus.

**S3 perception** (linear probe of the trained encoder; transfer gap = accuracy with train-split images - with test-split
images, on the held-out map; median [range]): I_ActOnly probe 0.985 [0.984, 0.986], gap +0.0052 (sign-flip p 0.0078);
I_MapWM_g1 0.984, gap +0.0056 (p 0.0078); I_RoPE 0.965, gap +0.0098 (p 0.0078); I_MapWM 0.071 [0.071, 0.978], gap 0.000
(p 0.25). The path models learn a better encoder than index RoPE does (0.985 vs 0.965), as Amendment 1 note n6 anticipated
(P1 is a total effect: map plus learned perception); the probe gap (0.02) is far smaller than P1 (0.29).

**S4 strata** (T = 1024; cell accuracy / retrace floor on the same revisits):

| cell | short gap | long gap | wrap-only |
|---|---|---|---|
| I_ActOnly | 0.995 / 0.911 | **0.994** / 0.509 | **0.994** / 0.508 |
| I_MapWM_g1 | 0.994 / 0.911 | 0.993 / 0.509 | 0.991 / 0.508 |
| I_MapWM | 0.688 / 0.911 | 0.589 / 0.509 | 0.544 / 0.508 |
| I_RoPE | 0.745 / 0.911 | 0.482 / 0.509 | 0.483 / 0.508 |
| T_MapWM | 0.961 / 0.911 | 0.958 / 0.509 | 0.947 / 0.508 |
| T_ActOnly | 1.000 / 0.911 | 1.000 / 0.509 | 0.997 / 0.508 |

The image path model holds every stratum; index RoPE on images keeps part of short-gap returns (below retrace) and is at
or below the blank level beyond them, the pattern of every failing model in the token batches.

**S5 context destruction** (observations permuted among the observation slots, targets unchanged; blank floor 0.508):
I_ActOnly 0.269, I_MapWM_g1 0.278, T_MapWM 0.292, T_ActOnly 0.290, I_RoPE 0.359, I_MapWM 0.425 (its collapsed seeds stay
at blank). Every model that learned falls below the blank floor: it predicts confidently from the destroyed context (seen
in pilot 2; the original prereg's prediction 0.50-0.56 was wrong).

**S6.** Out of distribution (T = 2048, rule 10): I_ActOnly 0.989, T_ActOnly 0.992, I_MapWM_g1 0.970, T_MapWM 0.943,
I_MapWM 0.638, I_RoPE 0.493. T = 512: I_ActOnly 0.994, I_RoPE 0.726. Own training map: I_ActOnly 0.995 (= held out),
I_RoPE 0.751 vs 0.701 held out (memorises part of its map). Train-split images at evaluation: I_ActOnly 1.000.
r(final-5% training loss, accuracy at T = 1024) over 48 runs **-0.988** (rule 2: accuracy is largely loss here; the
registered contrasts also separate on SOLVED and on within-ceiling counts).

## Predictions vs outcomes (Amendment 2, stated after the pilots and before the batch)

| prediction | outcome |
|---|---|
| P1 PATH WINS IN PIXELS: I_ActOnly 0.98-0.995, I_RoPE 0.66-0.75, d ~ +0.25 | **yes**: 0.9946, 0.7010, d +0.294 |
| P2 fires: I_MapWM at the blank level (0.50-0.52) on most or all seeds, d ~ +0.45-0.49 | **label yes, size smaller**: 5/8 at blank, 3/8 escaped (0.92-0.96); d +0.327. Amendment 3's power table: P2 fires if at most 3 seeds escape (3 did) |
| G: I_MapWM_g1 0.98-0.995, d ~ +0.45 (labelled LEAK SCALES WITH GAIN in Amendment 2, renamed GAIN 1 RESCUES MAPWM in Amendment 3) | **yes**: 0.9938, d +0.326 (smaller for the same reason) |
| residual I_ActOnly - I_MapWM_g1 within +/-0.01 | **yes**: +0.0007 |
| T_MapWM and T_ActOnly ~1.000 | T_ActOnly yes (0.9998, 8/8); **T_MapWM 7/8** (s3105 stalled at 0.683; FACTORIAL's token MapWM was 8/8) |
| S2b ~ +0.47 | +0.287 (p 0.039): the escaped seeds and T_MapWM's stalled seed shrink it |
| Amendment 3 risk: seeds escaping under the longer schedule | 3 of 8 escaped (pilots: 0 of 3 at 150 epochs) |
| (original prereg, superseded) image MapWM 0.97-0.99 | **contradicted** (pilots, then 0.668) |
| (original prereg secondaries) probe >= 0.98; transfer gap 0 to +0.005; S5 0.50-0.56 | probe yes for path arms (0.984-0.985), not for RoPE (0.965); gap +0.005 (path) / +0.010 (RoPE); S5 **wrong** (0.27-0.36) |

## Pilot history (stop rule, then redesign)

1. **Pilot 1** (seeds 3180-3181, 2026-10-25 02:41-03:00; `IMAGE_TORUS_PILOT.txt`). Timing: contended image / token
   s/epoch 5.8 / 4.2 = 1.38x, under the declared 2.0x, so `--cudnn-benchmark` was not adopted. **The pre-declared stop rule
   fired**: at 150 epochs image MapWM (gain 30.37) sat at the memoryless level (loss 2.049 / 2.047 / 2.046 at 50 / 100 / 150)
   while image RoPE (1.288 / 1.228 / 1.200) and token MapWM (0.912 / 0.781 / 0.518) descended. The batch was not launched.
2. **Pilot 2, diagnostic** (seeds 3182-3183, 12:02-12:15; `IMAGE_TORUS_PILOT2.txt`), 150 epochs: ActOnly at gain 30.37
   0.9944; MapWM at gain 1.0 0.9927; MapWM at gain 30.37 memoryless on a second seed (0.5027, encoder collapsed, probe
   0.129) and at lr 3e-4 (0.5086, per-instance step 122x an action step: the one NOISY run).
3. **Amendment 2** (with the user's approval of a redesign, before any batch seed ran): P1 became image ActOnly vs image
   RoPE; P2 became image ActOnly vs image MapWM, disclosed PILOT-INFORMED, NOT BLIND; G (gain 1 vs 30.37) added; block A
   grew to 48 runs; the original P1 kept as P1'. **Amendment 3** (independent re-check: 1 MAJOR, 6 MINOR, no blocker): P2
   renamed (it no longer asserts "leak"), end states with fixed thresholds, disclosure strings in the output, G labels,
   gain void, P2 power table.

## Checks

- **experiment_audit** (read-only, from /home/prashr; `--runs-dir runs/image_torus/<cell>` for all six cells and the root):
  exit 2 on every dir, as the prereg expects for this layout (no (json, pt) pairs at its default key; check [1] only).
  Windowed classes equal the registered tags in every cell: I_ActOnly 8 S; I_MapWM 1 S, 5 T, 2 D (its "NOT ALL FLAT"
  line); I_MapWM_g1 8 S; I_RoPE 8 T; T_MapWM 7 S, 1 T; T_ActOnly 8 S. Its "all arms converged" line on the other cells is
  the flat-window rule rule 3 warns about; the registered classes are the analysis's.
- **md5sum -c runs/image_torus/code_md5.txt** from the repo root: 111/111 OK (code, prereg, gate, driver, bank cache and raw
  image files).
- **Numbers:** every value above was re-read from `IMAGE_TORUS_ANALYSIS.txt` / `IMAGE_TORUS_VERDICTS.json`; per-seed values
  and step statistics from each run's `eval_image.json` and the eval logs; per-seed classes recomputed with
  `stats_core.classify_run` (they reproduce the analysis counts). No void condition tripped (test-split draws in training
  0, gains as recorded, bank md5 equal, ActOnly wiring |zeroobj - test| = 0).
- **Literature:** the survey the prereg names (`docs/IMAGE_TASKS_SURVEY_2026-10-25.md`) does not exist; no novelty claim is
  made here for pixel observations as such.

## Reading for the paper

Path integration is not a property of token inputs. With every observation a fresh 28 x 28 image of its class, recognised
by a CNN trained end to end and tested on held-out images and a held-out map, one path-integrated layer whose step is
driven by the actions learns the 1024-move torus to the perception ceiling (0.995, 8/8 seeds at or above it), holds
long-gap and wrap-only returns (0.994 each against a retrace floor of 0.51), and costs 0.005 against the same model on
tokens; one index layer stays at 0.701, below the retrace floor on every seed. The design rule is the what-to-where rule
of the leak line, made sharper by pixels: **actions drive the step; images do not.** An image-driven step at the scale of a
token embedding stops training on 5 of 8 seeds (a collapsed encoder and a clock-like phase) and leaves the other 3 below
the ceiling; scaling the image embedding down 30x (gain 1) restores learning on 8/8, and the pixel-specific part of the
cost is +0.29 (S2b). P2 and G were pilot-informed and are confirmations on fresh seeds, not discoveries.

## Scope

MNIST / Fashion-MNIST class images on a discrete 64 x 64 torus: one top-down image per cell visit, allocentric moves, 16
classes an 80k-parameter CNN recognises at 98%; not first-person views, not natural images, no visual aliasing beyond class
identity. One layer, 2 heads, d 128, one length (T = 1024), one recipe (900 epochs, lr 1e-3 cosine, AdamW wd 0.05), n = 8
per cell, block A only (no NormStep, deeper index or token RoPE cell). Images are fresh per visit; training uses the train
split only, evaluation the test split only. G is a total effect of the encoder-output scale (step input and residual
stream together), at two values; it is not a dose-response. P1 is a total effect (map plus learned perception).
--fast-attn without --deterministic: not bitwise reproducible. Memory Maze (first-person 3D) is the next test and is not
run.
