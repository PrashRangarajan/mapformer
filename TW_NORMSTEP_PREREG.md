# NormStep on the text world -- pre-registration (2026-10-03, before any run of the batch)

## Question
NormStep (the step map reads LayerNorm(embedding) instead of the embedding) removed MapWM's what-to-where leak on
the new-object task (`LEAK_RESULTS.md`: +0.0107, 8/8 SOLVED vs 8/8 DESCENDING). There the stream strictly
alternates one action and one observation, so the step every token shares through the LayerNorm bias beta (W beta,
independent of the token) is a per-move gauge the actions absorb (`docs/NORMSTEP_NOTES.md` section 2). In the text
world a move is rendered as a variable number of words (fillers, adverbs, asides), so W beta is a per-WORD tick:
a word-count clock leaking "when" into "where", which the gauge argument no longer cancels. Questions: (A) does
NormStep cost (or buy) accuracy on navigation told in words, at the training length; (B) does it make the phase
code more clock-like than MapWM's, and is the LayerNorm bias the cause?

## Task, recipe, eval (unchanged from `TEXTWORLD_PREREG.md`)
`environment_textworld.TextWorld`, 64x64 torus, K=16, 58 words, T = 1024 words; 1 layer, d 128, 2 heads, r=4 shared,
batch 16, 900 epochs x 98 batches, lr 1e-3, warmup + cosine, `--data-workers 3`. Eval: 200 trials on the held-out
map (env seed 10000), np seed 10**6 (no overlap with any training stream; the text-world batch used seed 0), at
T=1024 (registered) and 2048 (no verdict). Floors on THIS eval stream (`docs/audits/2026-10-03/tw_normstep_floors.py`,
6109 targets): best constant 0.512, reversal-copy rule 0.602.

## Arms (`train_tw_normstep.py`, `model_textstep.py`), one batch, all retrained, seeds 10-17 (fresh)
- MapWM: `Vanilla_r4` (the text-world path arm).
- NormStep: `model_codes.MapWM_NormStep` (LayerNorm with gamma and beta, the leak batch's arm unchanged).
- NormStepNB: the same with a bias-free LayerNorm (gamma only): no token-independent step exists.
- DirOnly: steps only from the 12 direction words (reference; told which words move; TEM-t's action-only update).
Every base weight equals Vanilla_r4's at the seed (the extra modules draw no random numbers; checked: identical
state dicts on the shared keys; DirOnly with an all-ones mask reproduces MapWM's logits exactly, max diff 0.0).
32 runs, seed outer, arm inner, MAXPG 3 (6 concurrent).

## Readouts (`tw_normstep_readouts.py`; `analyze_tw_normstep.py`)
- acc: held-out revisit object accuracy at T=1024. Run class from `stats_core.classify_run` (SOLVED = final-5% loss
  < 0.05).
- drift (the clock readout, `docs/audits/2026-09-27/tw_clock_probe.py` generalised to any step arm; it reproduces
  that probe's counts on the committed text-world runs, s0 31 and s1 2): on 40 held-out walks (np seed 0, T=1024),
  for every revisit of a cell, the wrapped phase difference omega * (cumsum step) between the revisit's object slot
  and the first visit's; a (head, block) channel DRIFTS if its mean |dtheta| exceeds 1.0 rad (random ~ pi/2).
  Count of 64 channels. A map channel returns to the same phase at the same cell; a clock channel does not. In
  the text-world batch MapWM's per-step clock seeds had 31-38, its map seeds 2-4.
- bias_step: ||W beta|| / mean ||direction-word step|| (NormStep only).
- move ratio (non-direction / direction step norm); ablate: accuracy with every non-direction word's step zeroed.

## Primary A: accuracy, NormStep - MapWM
Exact permutation test on per-seed accuracy and Fisher on SOLVED. FIRES if (perm p < 0.05 AND |d| >= 0.02) or
Fisher p < 0.05. The 0.02 floor keeps a 1% gap between two solved arms from firing (an audit finding on earlier
batches). Direction: SOLVED counts if Fisher fires, else d.
- **NORMSTEP HURTS IN WORDS** -- fires, NormStep below.
- **NORMSTEP HELPS IN WORDS** -- fires, NormStep above.
- **NO DETECTABLE DIFFERENCE** -- otherwise; reported as unmeasured below max(MDE, 0.02).

## Primary B: clock, drift channels
Exact permutation on per-seed drift counts; a contrast FIRES if p < 0.05 AND |d| >= 4 channels.
- **WORD-COUNT CLOCK** -- NormStep > MapWM fires AND NormStep > NormStepNB fires (the bias is the cause).
- **MORE CLOCK, BIAS NOT SHOWN TO CAUSE IT** -- NormStep > MapWM fires, NormStep vs NormStepNB does not.
- **FEWER CLOCKS** -- NormStep < MapWM fires.
- **NO CLOCK DIFFERENCE** -- otherwise.
Composite (computed from A and B, no new test): **NORMSTEP CARRIES OVER TO WORDS** if A is not HURTS and B is
neither WORD-COUNT CLOCK nor MORE CLOCK; else **DOES NOT CARRY OVER CLEANLY**.

## Secondaries (no verdict)
NormStepNB - MapWM, NormStepNB - NormStep and DirOnly - each, on accuracy, SOLVED and drift; clock seeds (drift
> 16) NormStep vs MapWM, Fisher; bias_step per NormStep seed (near 0 = the model learned to put beta in W's null
space); move ratio; non-direction ablation; r(final loss, acc) over 32 runs (rule 2); r(drift, acc) within arm;
T=2048.

## What it can and cannot show
- Drift counts any channel whose phase differs between visits to one cell: a per-move clock (MapWM's) and a
  per-word clock both count. bias_step and NormStepNB separate the bias route; a per-word clock built from the
  normalised directions (not beta) is not separated from a per-move one.
- MapWM's clock seeds solved as well as its map seeds (text world, Fisher p 1.00), so B is a representation readout
  ("where" kept clean of time), A the capability readout. A clock with no accuracy cost is still a leak of "when"
  into "where" for any downstream use of the phase.
- The leak NormStep removed on the new-object task needs unseen objects; here objects are 16 learned words, so this
  batch tests NormStep's side effects on language-like input, not its leak benefit.

## Amendment 1 (2026-10-03, after an independent code audit, before the pilot or any batch result was read)
The audit (read-only, CPU, blind) found no bug: arms, shared init, `_StepOverride` forward (exact in eval and train
mode), DirOnly mask (ids 46-57), eval-stream disjointness (worker seeds `seed*1_000_003 + i`), floors (0.5120 /
0.6024), drift = tw_clock_probe on all 8 committed seeds, the ablation hooks and bias_step all check out. It found
four design problems; the following REPLACE the corresponding text above, which is kept for the record.
1. **The old primary B was near-powerless.** Drift counts are bimodal per seed, so the test is a test on the number
   of clock seeds: with MapWM at 4/8, NormStep needs 0/8 or 8/8 to fire (p 0.051 / 0.047). It is demoted to a
   declared secondary; a non-firing reads UNMEASURED.
2. **The old B could not tell a per-word clock from MapWM's per-move one.** New PRIMARY B, `drift_opt_rad`: the same
   revisit phase comparison with the phase integrated over OPTIONAL positions only (adverb, fillers, aside sentence
   -- present a variable number of times per move), mean |wrapped dtheta| over all 64 channels and all revisit
   pairs, in radians. Each move holds exactly one verb, direction word, two seeing words, object and '.', so a
   constant moved among those (the per-move gauge) does not touch this readout; a step on optional words cannot be
   absorbed. Baseline (committed MapWM text-world runs, s0-s7): 0.048-0.141 rad (count of channels > 1 rad: 0/64 on
   8/8). Sensitivity (`docs/audits/2026-10-03/tw_normstep_sensitivity_out.txt`: a token-independent tick injected
   into committed MapWM models): +0.03 to +0.25 rad at 0.05 of a direction step, +0.09 to +0.43 rad at 0.1.
   A contrast FIRES if exact permutation p < 0.05 AND |d| >= 0.05 rad.
   - **WORD-COUNT CLOCK (removing the bias removes it)** -- NormStep > MapWM fires AND NormStep > NormStepNB fires.
   - **WORD-COUNT CLOCK, BIAS NOT SHOWN TO CAUSE IT** -- NormStep > MapWM fires, NormStep vs NormStepNB does not.
   - **LESS PER-WORD DRIFT THAN MAPWM** -- NormStep < MapWM fires.
   - **NO WORD-COUNT CLOCK DETECTED** -- otherwise; stated with the sensitivity above.
   Composite: **CARRIES OVER TO WORDS** if A is not HURTS and B is not a WORD-COUNT CLOCK branch.
3. **A Fisher-only firing of A is a convergence statement** (SOLVED is a loss threshold; in the leak batch NormStep
   converged faster with r(loss, acc) -0.94). Accuracy decides HURTS / HELPS (perm p < 0.05 and |d| >= 0.02); a
   Fisher-only firing reads **SOLVED RATE LOWER / HIGHER FOR NORMSTEP (convergence; accuracy unmeasured)**.
4. **bias_step and NormStepNB measure the beta parameter, not "the" shared tick.** LayerNorm centres over features,
   not tokens, so a step shared by all tokens can also come from a shared component of gamma * norm(e) (MapWM's
   clock seeds built one with no bias), and a token can cancel W beta through its own embedding. "No
   token-independent step exists" in NormStepNB was wrong: it lacks beta's by construction only. NormStepNB stays a
   one-knob ablation of beta; bias_step is descriptive.
Also: move ratio and the non-direction ablation are not per-move-gauge invariant (descriptive only); the void
condition is now asserted in code (DirOnly move ratio exactly 0 and its ablation a no-op within 0.002);
`stats_core.py` added to the md5 guard, which is re-checked before the analysis; paired sign-flip tests by seed
added as secondaries (the registered tests stay unpaired); the MDE line uses df = n - 1 (conservative ~8%).

## Void
Any of the 32 runs missing; md5 guard trips; DirOnly with any non-direction word step non-zero.

## Pilot (`runs/tw_normstep_pilot`, seed 100, outside the batch; read before this file was committed)
Seed 100, 900 epochs, all four arms, 4 concurrent (`runs/tw_normstep_pilot/analysis.txt`). Read AFTER Amendment 1.
| arm | acc T=1024 | class | drift (of 64) | word drift rad | own map | eval / train mode (40 walks) | bias_step |
|---|---|---|---|---|---|---|---|
| MapWM | 0.834 | SOLVED | 27 | 0.075 | 0.844 | 0.841 / 0.991 | -- |
| NormStep | 0.865 | SOLVED | 29 | 0.073 | 0.878 | 0.865 / 0.995 | 0.112 |
| NormStepNB | 0.943 | DESCENDING | 35 | 0.061 | 0.950 | 0.946 / 0.983 | 0 |
| DirOnly | 0.970 | SOLVED | 0 | 0.000 | 0.978 | 0.960 / 0.963 | -- |
All three learned-step arms took the per-move clock solution on this seed; no per-word drift above MapWM's baseline;
NormStep's beta step is 0.11 of a direction step yet does not show as per-word drift (cancelled or in low-omega
channels). Void check OK. Pilot additions, declared BEFORE the batch (Amendment 2):
- **Eval mode under-reports clock-type solutions.** SOLVED (training loss < 0.05) yet 0.83-0.87 held out: the same
  walks score 0.99 in train mode. Attention-probability dropout alone restores it (0.84 -> 0.99); residual and FFN
  dropout do not; the committed text-world clock seeds s0 / s7 show it too (0.88 / 0.90 -> 0.97 / 0.99), map seed s1
  and DirOnly do not (`docs/audits/2026-10-03/dropout_mode_check.py` / `_out.txt`). The registered accuracy stays
  eval mode (comparable with TEXTWORLD_RESULTS); declared secondary: train-mode accuracy (all dropout on, mean of 3
  dropout seeds) and eval-mode accuracy on the same 40 held-out walks, per arm, and NormStep - MapWM in train mode.
  If A's eval-mode verdict and the train-mode contrast disagree, the report says so.
- Own-map accuracy (training map, 100 walks): secondary, separates memorisation from a missing map (pilot: no
  memorisation, own map within 0.01-0.02 of held out).

## Cost (measured)
Pilot: 1.5 s/epoch at 4 concurrent jobs (~23 min per run); readouts 20 s for 4 runs on CPU. Batch at 6 concurrent: ~2 s/epoch expected, 32 runs
~2.5-3 h; readouts on CPU ~10 min.
