# CLOCK_SPEED independent code audit (rule 29), 2026-10-12

Scope: `CLOCK_SPEED_PREREG.md`, `analyze_clock_speed.py`, `run_clock_speed.sh` + `lib_driver.sh` (diffed against
`run_clock_side.sh`, audited in `clock_side_audit.md`), the loss bookkeeping in `train_recency.py`, `stats_core.signflip_p` /
`classify_run`. `runs/clock_speed` does not exist and was not created in the repo. `runs/clock_side/p0` (finished, disclosed
in the prereg) was read to test the analysis and to size the branches. Nothing in the repo was modified except this file; no
GPU job. Scratch: `/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/audit_cspeed/`
(`t1_decomp.py`, `t2_power.py`, dry-run copy of the driver under `drv/`).

## Verified correct

- **sed derivation is clean.** `diff run_clock_side.sh run_clock_speed.sh` touches only: header comment, log name, lock
  (`.run_clock_speed.lock`), run dir (`runs/clock_speed/p0`, in the md5 guard, the wait regex and REQ), seeds 210-217 (launch
  loop and REQ), arms Signed_r4 / Abs_r4 / CARoPE_r4 (both loops), the GUARD list (drops `analyze_clock_side.py`,
  `eval_clock_side.py`, `rescore_hook.py`, which this batch does not run; adds `analyze_clock_speed.py`, `run_clock_speed.sh`),
  the eval step removed, artifact count 80 -> 48 (= 24 runs x {.json, .pt}), done marker `.clock_speed_done` gated on
  `CLOCK_SPEED_ANALYSIS.txt` and `CLOCK_SPEED_VERDICTS.json` (the analysis's default `--out`). No stale `clock_side` name remains.
  Training flags are byte-identical to CLOCK_SIDE's.
- **Dry run** (scratch copy, `DRV_DRYRUN=1`, spacing set to 0): 24 unique (variant, seed) launches, seeds 210-217, all
  `--device cuda:0`; md5 list written (11 files); missing artifacts fail closed ("done marker NOT set"). `DRV_*` knobs assigned
  after sourcing (rule 27). GPU restriction holds unless the caller's environment already sets `DRV_GPUS` to something else
  (launch line sets `DRV_GPUS=0` explicitly). At audit time no `mapformer.train_*` job was running.
- **Seeds are fresh**: no `*_recency.pt` under `runs/` for seeds 210-217; CLOCK_SIDE used 200-207, calibration 100.
- **Pairing is real for this readout**: the training stream is `RandomState(seed)`, independent of the model; epoch-1 losses of
  the four path arms on CLOCK_SIDE agree to < 3e-4 at every seed and first differ by > 1e-3 at epoch 23-27 (shared init and data).
- **Analysis**: `d_i = AUC(other) - AUC(Signed)`, negative = other faster, matching the branch labels; `signflip_p` is exact,
  two-sided, min p 2/256 = 0.0078. All four branches are reachable (`--smoke` ALL PASS). AUC = `mean(losses)` over the 300 saved
  per-epoch means, as registered. `first()` returns the last epoch of the first 10-epoch window below the threshold (correct;
  censored 301). Provenance asserts: variant, seed, k_max, min_gap, 300 epochs, and T / k_max / min_gap from the log header (the
  regex hits the header, not the `[held-out] T=` lines). Run end to end on CLOCK_SIDE's 24 runs (`--runs .../clock_side/p0
  --seeds 200..207`): reproduces the prereg's numbers (S1 UNMEASURED d -0.268 p 0.086 6/8; S2 FASTER d -0.291 p 0.0078 8/8;
  unpaired p 0.168 / 0.092; r(AUC, acc) -0.908).

## BUG

None found that blocks launch.

## DESIGN major

**D1. NO DIFFERENCE is a point-estimate rule, not an equivalence test, and fires on material effects.** It requires only
|mean d| < 0.05 and p >= .05. On CLOCK_SIDE the per-seed sd of d is 0.216 (CARoPE), 0.331 (Abs), 0.492 (Pos): the paired 95% CI
half-width at n = 8 is 0.18-0.28 and the MDE (3.26 sd / sqrt n) 0.25-0.38, 2.5-4x the 0.10 materiality. Simulation (`t2_power.py`,
bootstrap of CLOCK_SIDE's d, 2000 reps):

| true effect | Abs (sd 0.33) | CARoPE (sd 0.22) |
|---|---|---|
| 0 (centred) | NO DIFF 0.34, UNMEAS 0.61 | NO DIFF 0.48, UNMEAS 0.47 |
| half of CLOCK_SIDE's (-0.13 / -0.15, above materiality) | NO DIFF 0.17, FASTER 0.09 | NO DIFF 0.08, FASTER 0.37 |
| CLOCK_SIDE's (-0.27 / -0.29) | FASTER 0.35, UNMEAS 0.64 | FASTER 1.00 |

So a true Abs effect of -0.13 reads NO DIFFERENCE one time in six. Rule 5: below the MDE the label is "unmeasured". Fix: NO
DIFFERENCE only if the paired 95% CI (sign-flip test inversion, or t) lies within [-0.10, +0.10]; at these sds that is nearly
unreachable at n = 8, and the prereg should say so (RING_DIAL's DC branch already uses the CI-within-margin form).

**D2. S1, the contrast the "double dissociation" rests on, is underpowered.** The ring side measured Abs_r4 vs Signed only
(RING_DIAL / RING_BUDGET), so only S1 is the same contrast on both tasks; S2 (CARoPE) carries the init / gradient-scale confound
documented in `clock_side_audit.md` M2 (uniform clock at init, ~5.7x slower step per update). Even at CLOCK_SIDE's effect size,
which is the post hoc selected (winner's-curse-inflated) estimate, S1 fires ~35% of the time at n = 8. Fix (choose one, before
launch): (a) n = 16 for Signed_r4 and Abs_r4 (16 more runs, ~+1.3 h on GPU 0; CARoPE can stay at 8), or (b) state in the prereg that
S1 is expected UNMEASURED with ~1/3 power, that the "double dissociation" label requires S1 (not S2) to fire, and that S2 alone
reads "CARoPE's parameterisation learns the clock faster", not "the non-negative step".

**D3. "Double dissociation" compares different readouts.** Ring side: solved count at a fixed budget; clock side: loss AUC. The
ring runs save 300-epoch loss curves (`runs/ring_dial/p0/*/Abs_r4.pt['losses']`), so the same AUC readout on the ring is
computable from existing data. Either declare it now as a post hoc companion (existing data, so not registered) or drop the
"double dissociation" wording from the registered label.

## DESIGN minor

- **m1. Is mean training loss a fair speed readout?** Mostly yes: same CE over the 16-symbol block (chance ln 16 = 2.77), same
  targets and data at a seed, identical dropout (p 0.1) in every arm; the plateau level contributes almost nothing (epochs 1-100
  give -0.004 / -0.007 of the -0.27 / -0.29 AUC differences on CLOCK_SIDE). Two caveats to state: (i) dropout inflates the training
  loss by a solution-dependent amount: final train loss minus eval NLL at T on CLOCK_SIDE is 0.171 Signed, 0.146 Abs, 0.229 CARoPE
  (per-seed range 0.03-0.40), so after escape it can shift d by a few hundredths (conservative for S2, slightly anti-conservative
  for S1); report the paired eval-NLL / accuracy contrast beside the AUC verdict. (ii) AUC mixes escape time with the level
  reached: on CLOCK_SIDE about half the difference comes from epochs 201-300. Name it "loss AUC over 300 epochs" and state that
  "faster" means that; the escape-epoch secondary should agree in direction before the word "speed" is used in prose.
- **m2. Choosing the paired test after seeing CLOCK_SIDE is acceptable** given fresh seeds and that the pairing was recommended a
  priori (`clock_side_audit.md` m1). The remaining costs are the winner's curse in D2 and that the hypothesis, AUC readout,
  thresholds and materiality were all set post hoc on seeds 200-207: never pool 200-207 into the verdict (rule 6), report fresh
  seeds alone; the analysis already does.
- **m3. Two primaries, no multiplicity rule.** State "no correction, two separately labelled contrasts" or Holm over S1/S2
  (min p 0.0078 survives Holm).
- **m4. Learning speed is recipe-scoped**: one lr (1e-3), cosine, batch 16; a speed difference at one lr can be a difference in
  optimal lr between parameterisations (rule 15). Scope the label to the recipe.
- **m5. Provenance asserts** still do not check `n_layers / d_model / n_heads` (in the .pt) or `[fast-attn] SDPA + TF32 enabled`
  (log line 1) -- `clock_side_audit.md` B2 asked for both. Cheap to add; must be added BEFORE launch, because the md5 guard
  aborts the analysis step if `analyze_clock_speed.py` changes after the batch starts.
- **m6. Wording**: "in-training eval accuracy" is the held-out eval after training, in eval mode (the one-layer dropout
  under-report applies to it, not to the AUC); epoch-1 loss is 2.925, above the 2.77 "chance" (init); `SOLVED x/8` is hard-coded
  in the print.

## Prereg should state

The CI-based NO DIFFERENCE rule or its unreachability (D1); expected power for S1 and that S1 carries the dissociation (D2); the
readout mismatch with the ring side (D3); the dropout and level caveats of training-loss AUC (m1); recipe scope (m4); that seeds
200-207 are excluded from the verdict (m2).
