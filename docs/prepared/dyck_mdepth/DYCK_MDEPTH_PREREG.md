# Dyck position effect at MATCHED nesting depth -- pre-registration (DRAFT, 2026-09-25)

Written before any run of this batch. Driver `run_dyck_mdepth.sh`; gates
`validate_dyck_mdepth.py`; readout `analyze_dyck_mdepth.py`; metric `dyck_mdepth_common.py`;
trainer change `train_dyck.patch` (opt-in flags, defaults unchanged).

## Why

`DYCK_LADDER_RESULTS.md` is carried in CLAUDE.md as "the one converged, matched-length,
floor-reported positive result in the language line": at L32 D12 the position main effect is
+0.290 / +0.209 / +0.159 / +0.168 at 1-4 layers (8/8), with the index arms plateauing at
0.76-0.78. The 2026-09-24 audit found that training is L32 **D4** (`train_dyck.py` hard-codes
`world.batch(bs, 32, 4, rng)`), so L32 D12 is matched in LENGTH but not in DEPTH. The gates of
this batch quantify it: **53.7% of the closers scored at L32 D12 sit at stack depth > 4, and
under D4 training that fraction is exactly 0** (median closing distance 9 vs 1). At the actual
training cell (L32 D4) the effect shrinks to +0.019 at 4 layers as the index arms reach 0.979.

Every "helps out of distribution" claim in this project that got a matched control has died
(code OOD, rank +0.085, the navigation +0.461). This is that control for the last one.

## Design -- one batch, 210 runs, only the TRAINING DISTRIBUTION changes

Arms, widths, rank, recipe and evaluation sequences are the ladder's (n_heads 2, d 128, r=2 for
the path arms, omega base 32, index RoPE/PoPE at base 10000, AdamW 1e-4, wd 0.01, 5% warmup +
cosine to 10%, batch 128, 560k sequences = 4375 steps). The training sampler is
`DyckWorld.batch(128, 32, D, rng)` with D set by the new opt-in flags.

| block | runs | trained on | budget | arms x depths x seeds | role |
|---|---|---|---|---|---|
| **T12x3** | 48 | L32, every sequence reaches depth 12 | **3x** (13,125 steps) | RoPE, PoPE, RoPE_b32, PoPE_b32, MapWM, MapPoPE x 4L x 8 | **PRIMARY** |
| T12 | 128 | L32 D12 | 1x | RoPE, PoPE, MapWM, MapPoPE x 1-4L x 8 | depth shape at matched depth |
| Tmix | 32 | L32, D uniform on {4..12} per batch (separate RNG) | 1x | 4 arms x 4L x 8 | D12 in support, not dominant |
| repro | 2 | ladder defaults (L32 D4) | 1x | RoPE-4L, MapWM-4L_r2, seed 0 | must reproduce `runs/dyck_ladder` |

`_b32` = index arm at rope_base 32, the path arms' own ladder base; it closes the base
confound (`EXPERIMENTS.md` #4) at the primary cell, where it has never been measured.

**Why 3x at the primary.** The ladder's index arms were still descending under the decayed LR
(audit, `EXPERIMENTS.md` #3). A positive claim must survive the budget most favourable to the
baseline; the paper budget is kept for the 1-4L shape (S1), and S4 says whether it was
budget-limited.

## Gates (run by the driver before any training; exit non-zero on failure) -- PASS, 2026-09-25

`DYCK_MDEPTH_GATES.md`: G1 every training sequence valid with max depth exactly D (all three
distributions); G2 the L32 D12 cell's statistics equal the T12 training distribution's (closers
at depth > 4: 0.537 vs 0.538; > 8: 0.254 vs 0.255; closing distance 9 / 23 / 31 both) and are
absent from D4 training; G3 the sampler's exact distribution scores A2f = 1.000 on every cell;
G4 floors: best A2f floor at L32 D12 **0.594** (n-grams of order 1-6 fitted on T12; stack-free
heuristic 0.543), 0.588 for the ladder's D4 fit. Chance 0.500.

Trainer gates (run 2026-09-25, outputs in the scratchpad, NOT reused by the batch):
- the patched `train_dyck.py` at default flags reproduces `runs/dyck_ladder` RoPE-1L s0 and
  PoPE-4L s0 with **0 of 199,845 / 795,205 state-dict elements differing** and an identical eval grid; the
  logged loss curve differs by <= 6.7e-8, and so does the UNPATCHED current code, so the
  loss-curve delta predates this change (weights and eval are the reproduction criterion);
- parameter counts are the ladder's (only the data changes); names carry the training
  distribution (`_tL32D12`, `_tL32Dset4-12`) and the JSON records `train_L`, `train_D`,
  `train_D_set`, per-D batch counts, `train_ce_floor` and `n_sequences`, all asserted by the
  readout;
- driver dry run (`DRV_DRYRUN=1`): gate ran and passed inside the driver, 210 launches with
  the intended flags, `drv_require` failed as designed with nothing trained; `bash -n` clean.

**Disclosed before registration:** four n=1 smoke runs with NON-default flags at seed 0 were trained to check the
flags and timing, and their TRAINING losses were seen (no A2/A2f was computed on any of them):
RoPE-1L at D12 final 0.771 against the D12 floor 0.520; MapWM-4L at D12 0.522; RoPE-4L at D12,
3x budget, 0.528; RoPE-4L base 32 on the mixture 0.830 against 0.681, still descending. A
4-layer index model that trains to within 0.008 nats of the floor at D12 points toward
CLOSES. The batch retrains seed 0 from scratch; nothing from the smoke runs enters it.

## The primary metric changes from A2 to A2f -- why, and why it is safe

A2 (Hewitt's distance-averaged closing accuracy, the ladder's primary) scores every prefix where
a closer is grammatical. Under a fixed-(L, D) sampler, 32% of those prefixes at L32 D12 are
FORCED OPENS (closing would make depth 12 unreachable). The CE-optimal predictor puts zero
mass on both closers there, so its A2 is 0.940, not 1 -- and a model TRAINED at D12 learns
exactly that, leaving A2 at those prefixes a ratio of two vanishing probabilities. **A2f**
scores only prefixes where the sampler can emit a closer (read from `DyckWorld`'s own
per-prefix entropy); the CE-optimal predictor scores 1.000 on every cell. On the ladder's
D4-trained 4-layer checkpoints A2f and A2 differ by <= 0.003 (RoPE 0.752 / 0.755, PoPE 0.778 /
0.781, MapWM 0.922 / 0.923, MapPoPE 0.948 / 0.949; position effect +0.170 vs +0.168), so the
depth-OOD reference carries over unchanged. A2 is reported beside A2f everywhere.

## PRIMARY readout and branches (fixed now)

**P = position main effect, A2f, cell L32 D12, 4 layers, T12x3**, per seed
`((MapWM - RoPE*) + (MapPoPE - PoPE*)) / 2`, where RoPE* / PoPE* is, per encoding, whichever
index base (10000 or 32) has the higher MEAN A2f (arm-level, conservative for a positive
claim). Both single-base versions are reported. n = 8 paired; DETECTABLE = |mean| > exact-t
MDE (t_.975,7 + t_.80,7 = 3.26 x sd / sqrt 8); the house 2.8 is printed beside it.

- **SURVIVES** -- P detectable, P >= 0.10, >= 7/8 seeds positive: path integration beats index
  coding at matched depth; the Dyck row becomes the language line's first matched-DISTRIBUTION
  capability result, with the base confound closed at 4 layers.
- **CLOSES** -- P within its MDE or P < 0.05: the ladder's +0.168 at L32 D12 was depth
  extrapolation. The language line then has no matched-distribution positive result beyond the
  training-cell effect that shrinks to a ceiling (+0.019 at 4L), and "matched vs mismatched
  length" in CLAUDE.md becomes "matched vs mismatched DISTRIBUTION".
- **SHRINKS** -- otherwise: quote the matched-depth P, never +0.168.

Anchors (rule 15: boundaries set against named quantities, not a band that swallows a branch):
0.05 is the ladder's largest training-cell effect at >= 3 layers (+0.048 at 3L), the level
the project already reads as "shrinking to a ceiling"; 0.10 is ~60% of the depth-OOD +0.168
and ~3.5x the 4-layer MDE on the ladder data (0.029 exact-t). Both sit well outside the noise
floor (MDEs 0.011-0.042 on these cells).

## Secondaries (reported whatever the primary says)

- **S1** depth shape at matched depth (T12, 1x): P at 1-4L; index arms 3L -> 4L (plateau: within
  MDE). If S4 fires, S1 is labelled budget-limited and no plateau sentence may be written.
- **S2** Tmix at 4L: P at L32 D4, L32 D12, L128 D12. Descriptive (D12 is 1/9 of training).
- **S3** base: RoPE_b32 - RoPE and PoPE_b32 - PoPE at 4L, 3x, L32 D12.
- **S4** budget: 3x - 1x at 4L per arm; the index mean gain DETECTABLE with >= 7/8 positive
  means the 1x T12 ladder is budget-limited.
- References (cross-batch, descriptive only): the ladder's D4-trained checkpoints re-scored with
  A2f on the same 512 sequences per cell; L128 D12 (length extrapolation) for every block.
- Convergence, per arm: final loss (mean of the last 10% of the curve), gap to the training
  distribution's sampler entropy (the exact CE floor), median final slope per 1k steps.
- Floors from the gates beside every cell.

## Void conditions

The in-batch reproduction differs in any weight or in the eval grid (code drift: stop and
investigate before reading anything); any of the 210 runs missing; the gates fail; arms split
across batches; the md5 guard trips (code changed mid-batch).

## Registered prediction

None beyond the disclosure above. Four of this project's generalisations from a within-task
result to a cross-task rule failed; the smoke losses are n = 1 and are not the readout.

## Scope, stated in advance

One task (Dyck-2), one length (L32), one width (n_heads 2, d 128), r = 2 for the path arms,
the paper's optimiser, 8 seeds. A2f at L32 D12 is the registered cell; L32 D4 under T12 is a
shallower-than-training reference, not a test.

## Cost (measured solo, 2026-09-25; the ladder ran 1.5-1.8x slower at 2 jobs/GPU)

1L 24 s, 4L 48 s, 4L 3x 133 s, 4L mixture 45 s per run alone. 3.5 run-hours alone, ~5.6 at
2 jobs/GPU; with 4 slots ~1.5 h wall including gates (~1 min) and readout (~5 min, 338
checkpoints scored). About 3 GPU-hours of occupancy on the two 4090s.
