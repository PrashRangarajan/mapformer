# Rank at matched length -- pre-registration (2026-09-23, written before any T=1024 checkpoint exists)

## Why

`RANK_SWEEP.md` recommends r=4 over MapFormer's r=2 on the strength of **+0.085 at
T=1024 after training at T=128** (+0.038 at T=512). At the training length both arms
are at ceiling in accuracy (r=2 0.993, r=4 1.000). So the whole recommendation is an
out-of-distribution effect -- the pattern that turned out to be length robustness,
not capability, on code (`CODE_RESULTS.md`, C1 closed as a reversal) and that has
never had a matched-length control on the torus (rank, InEKF, forget gate and
PoPE-wrapping were all trained at T=128 and tested at T=512/1024).

## What the pre-launch audit found on the OLD checkpoints (eval_rank_strata.py)

Scored revisits at T=1024 split three ways (shares of scored events, 8 seeds):

| kind of revisit | share | r=2 | r=4 | share of the +0.085 | floor |
|---|---|---|---|---|---|
| plain, gap < 128 steps | 83.5% | 0.903 +/- 0.078 | 0.999 +/- 0.001 | +0.080 (94%) | 0.515 |
| plain, gap >= 128 steps | 9.3% | 0.563 | 0.625 | +0.006 | 0.496 |
| reachable only by wrapping the torus | 7.2% | 0.416 | 0.413 | ~0 | 0.506 |

94% of the old effect is revisits whose gap DID occur in T=128 training but which
happen later in the sequence than training reached. Matched-length training removes
exactly that difference, and r=2 had room to fail there (0.903), so a tie is
informative, not a ceiling artefact. The wrap stratum is below its floor for both
arms and never occurs at T=128; at T=1024 it is 7.7% of training targets.

Also found, and registered here so it is read rather than rediscovered:
- **The old losses did NOT overlap.** Final training loss r=2 0.0011-0.0874, r=4
  0.0000-0.0006; r=4 NLL lower on 8/8 seeds at every length (T=128 0.013 vs 0.000,
  T=1024 1.155 vs 0.387). "Both at ceiling in distribution" held in accuracy only.
- **r=2's action code is bimodal across seeds** (seeds 1/2/3/6 cancel cleanly,
  opposition 0.09-0.14, but put both axes on one line, |cos| ~0.99; seeds 0/4/5/7
  have opposition 0.51-1.22). Within r=2 neither metric predicts T=1024 accuracy
  (r = -0.35 / +0.30), so "the skew is the mechanism" holds only at arm means.

## Design

`Vanilla` (r=2, 204,373 params) vs `Vanilla_r4` (204,757), 8 seeds, one batch,
`runs/rank_matched`. The RANK_SWEEP recipe with ONE change, trajectory length:

    --epochs 300 --lr 1e-3 --n-batches 98 --n-layers 1 --n-heads 2 --d-model 128
    --n-landmarks 0 --schedule cosine --data-workers 3
    --n-steps 1024 --batch-size 16        (was --n-steps 128 --batch-size 128)

Audited equal across arms and against the old recipe: input tokens per step 32,752 vs
32,640 (+0.34%), 29,400 optimizer steps and 1,470 warmup steps both, same data
stream per seed across arms (batch i seeded by index). What changes with length:
8x fewer independent walks (470K vs 3.76M), and +35% scored targets per batch. The
model does not see `--n-steps` (no positional table, omega from grid size only).
Checkpoints now record `n_steps` / `batch_size` in their config.

Eval: held-out map (env seed 10000), trajectories seeded 1234+s so arms and the old
checkpoints are paired, 100 trials. `eval_noise_refine` at T = 512 / 1024 / 2048 and
`eval_rank_strata` at T = 1024 / 2048, on BOTH the new and the old checkpoints.

## Readouts

MDE = 2.8 * sd(paired difference) / sqrt(8). Detectable = |mean| > MDE.

**Primary**: r4 - r2 overall revisit accuracy at **T=1024, the training length**.

**Co-primaries** (T=1024, per stratum): plain gap<128, plain gap>=128, wrap. Floor
reported beside each.

**Secondary**: NLL at T=1024 overall and per stratum; T=512; **T=2048** (2x beyond
training -- an extrapolation readout again, read only as such).

**Required checks, reported whatever they show**:
- Final training loss per seed, both arms, and whether the ranges OVERLAP. If they do
  not, the accuracy contrast is also a training-quality contrast: report r(log final
  loss, T=1024 acc) pooled and within arm, and the loss-matched residual.
- Convergence: a run is FLAT if its mean loss over epochs 271-300 is within 5% of its
  mean over 241-270. Report flat counts per arm. If fewer than 6/8 are flat in either
  arm the verdict is "budget-limited" and no branch fires.
- Always-blank / constant floor per stratum.

## Branches (boundaries set by the MDE, not by a band)

- **R1 -- robustness (the likelier).** Primary within MDE, or both arms >= 0.99 on
  the plain gap<128 stratum. The old +0.085 was the cost of r=2 generalising to
  unseen lengths. "Use r=4" narrows to "use r=4 if you will run past the training
  length"; it still costs only 384 parameters.
- **R2 -- capability.** Primary > MDE, positive on >= 7/8 seeds. r=2 has a deficit
  at long sequences even when trained on them; the RANK_SWEEP recommendation stands
  as a capability claim, at matched length.
- **R3 -- reversal.** Primary < -MDE. r=2 is better when trained at length.

The wrap and gap>=128 strata are read separately and do not decide R1/R2: they are
where BOTH arms have headroom once trained on them, so they can show a difference the
overall number dilutes.

**Exploratory prediction (geometry)**: trained on 1024-step walks, where opposite
moves must cancel over far longer stretches, r=2's action code gets cleaner --
opposition falls toward r=4's. Read per seed (`probe_action_geometry --runs-dir
runs/rank_matched/p0`), never as a mean alone.

## Amendment 1 (2026-09-23, after the 300-epoch batch landed budget-limited)

The 300-epoch batch failed the convergence check (flat r=2 4/8, r=4 0/8;
`RANK_MATCHED_RESULTS.md`), so no branch was read. The only change is the budget.

**Pilot**: seeds 0 and 1, both arms, **900 epochs** (3x), everything else identical,
`runs/rank_matched_e900`, `EPOCHS=900 TAG=_e900 SEEDS="0 1" PILOT=1 bash run_rank_matched.sh`.
The pilot is read on TRAINING LOSS CURVES ONLY -- no evaluation, so it cannot steer the
readout. Seeds 0 and 1 are the first two in the default order, not chosen by result.

**Decision rule**: if all 4 pilot runs are flat (the registered 5% criterion, epochs
871-900 vs 841-870), run the remaining seeds 2-7 at 900 epochs into the same directory
(the pilot runs are kept as seeds 0-1; same seed and code give the same run), then
evaluate with every readout and branch above unchanged. If any pilot run is not flat,
report and choose a longer budget before spending the 8-seed batch. A run that is flat
at a loss above 0.1 is reported as a possible plateau (rule 10), not silently accepted.
