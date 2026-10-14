# WRAP_COPE independent code check (rule 29), 2026-10-14

Scope: HEAD 79eb3a5. Files: `WRAP_COPE_PREREG.md`, `model_mono_baselines.py` (CoPEF_Sin/Hi/Lo), `train_ring_variant.py`,
`train_variant.py`, `analyze_wrap_cope.py`, `run_wrap_cope.sh`. CPU only. Scratch scripts are in the session scratchpad
(`chk_init*.py`, `washout.py`, `pilot_table.py`, `cope_smoke/`).

Blindness: the pilot (seed 990, outside the batch) had already finished when I listed `runs/wrap_cope_pilot/`, and I saw its
log tails. I used the pilot only for findings 3 and 4, which are about whether the readouts are reachable and well placed. No
batch run exists.

## Verified correct
- **Arms differ only in the table.** At seed 900 the three arms differ only in `layers.0.attn.cope.pos_emb`. The torch RNG
  state after construction is identical (next `rand` 0.18995 in all three), so data and dropout streams match across arms.
- **CoPE is Listing 1.** It uses content logits `q.k/sqrt(d_h) + mask.log()`, sigmoid gates, the flip-cumsum-flip count
  over k = j..i (the key and the query included), integer-knot interpolation, and adds the result to the logits.
  `papers/txt/cope.txt` ~1065-1135 matches.
- **Premise arithmetic holds at init.** Measured at seed 900 on 32 trajectories:
  - gates average 0.503 (sd 0.076); content logits have sd 0.309.
  - The count over an out-and-back is 2.018 +/- 0.168. That span is 4 tokens, o_s..a_t, because the query is the action
    token that predicts o_t.
  - Per-channel angles at that count are Hi 1.21-1.92 pi and Lo 0.20-0.91 pi. So the nearest multiple of 2 pi is 2 pi for
    every Hi channel and 0 for every Lo channel, as stated.
- **No aliasing from interpolation.** Both bands are below the knot Nyquist frequency (pi).
  - At the integer knots, Hi channel m equals (-1)^p x [cos(delta p), -sin(delta p)] with delta = pi - omega in
    [0.05, 0.4] pi. Hi is therefore the audit's period-2 carrier (generality_audit finding 1) times a slow envelope.
  - The interpolated function is a zig-zag between knots. That is harmless: the period-2 construction lives on the knots.
- **Analysis directions are right.** `perm2_p(r, a)` and `perm2_ci(r, a)` test mean(x) - mean(ref). `fisher_solved` is
  two-sided, and its direction is taken from ka vs kr. The eval keys "128"/"512" match `train_ring_variant`. The DFT
  frequency axis is angular per count unit (rfftfreq x 2 pi).
- **Re-score works.** CoPEAttn is in `rescore_hook.KNOWN`, and the hook fired in the smoke (0.5059 -> 0.4959).
- **Smoke ran clean.** 3 arms x seeds 900-901, 2 epochs x 2 batches, CPU, 20 trials: all exit 0. The analysis with
  `--root/--seeds 900-901 --no-json` runs end to end. At 4 steps every arm reads 0.5059 and NO EFFECT DETECTED, which is the
  vacuous case in finding 5.
- **The guard covers the execution path.** train_ring_variant -> train_variant, model_mono_baselines, model, lie_groups,
  train, data_parallel, environment{,_cancel,_nd}, train_cancel, stats_core, rescore_hook. The ~80 other modules that
  `train_variant` imports are unguarded, as in every prior driver.
- **Driver bookkeeping is right.** 24 x 2 = 48 artifacts; the clean check uses `--untracked-files=no`; `flock` on
  `.run_wrap_cope.lock`; `DRV_*` is set after sourcing; `running()` and `drv_wait_dir` match the absolute `--output-dir`.

## Findings

**1. BLOCKER -- the driver runs the wrong analysis and overwrites a registered verdict file.**
- `run_wrap_cope.sh` line 35 runs `python3 -u -m mapformer.analyze_wrap_mech`, not `analyze_wrap_cope`.
- `analyze_wrap_mech` defaults to `runs/wrap_mech/p0` and writes the tracked `MECHANISM_VERDICTS.json` (line 97). It would
  also put the MECHANISM analysis into `WRAP_COPE_ANALYSIS.txt`.
- `drv_done` then fails because `WRAP_COPE_VERDICTS.json` is never written.
- `analyze_wrap_mech.py` is not in GUARD, while `analyze_wrap_cope.py` is.
- Fix: change it to `mapformer.analyze_wrap_cope --root "$R" --seeds 900-907`, then recommit before launch. The md5 guard
  has not been written yet, since `runs/wrap_cope/` does not exist.

**2. MAJOR (how W1 is read) -- Hi and Lo match in amplitude, not in table slope or table logit scale.**
- The mean knot-to-knot change |e[p+1] - e[p]| is 0.118 for Hi and 0.053 for Lo, 2.2x apart. This is the
  dlogit/dcount = q.(e[ceil] - e[floor]) term that carries the position gradient into the gates.
- The table logit sd at init runs the other way: Hi 0.213, Lo 0.306 (content 0.309).
- So W1 BETTER cannot separate "past pi" from "a steeper table gives the gates more gradient". This is the analogue of
  MECHANISM's mobility control (AO_mid, Amendment 1, M2b).
- Fix, either one:
  - add an arm, CoPEF_LoS: Lo with amplitude scaled 2.2x so mean |delta e| matches Hi (8 runs, ~15 min); or
  - amend the W1 reading to "a near-wrap (period-2-carrier) table start, confounded with table slope".
- Also report the logit-scale mismatch beside "amplitude 0.1".

**3. MAJOR (secondary) -- the table-power DFT has no band for the solution it is looking for.**
- The audit's representing table is period 2 (omega = pi, the Nyquist bin). That bin is excluded: the Hi band stops at
  0.95 pi.
- On the pilot tables (seed 990, outside the batch), 0.72 (Hi) and 0.74 (Lo) of the counts-0..255 power is in
  (0.95, 1] pi. Hi-band shares are 0.06 / 0.04 and Lo-band shares 0.04 / 0.02.
- As written, the secondary would read "training left both bands" in both arms, which is wrong.
- Two further problems with the window:
  - Every entry 0..2047 moves under AdamW weight decay. Unvisited entries keep a decayed copy of the init: corr(final,
    init) is 0.37 over counts 64-2047, against 0.04-0.06 over 0-64.
  - The 0..255 window includes counts that are rarely reached (the max count at init is ~170-194).
- Fix:
  - add a Nyquist band (0.95, 1] pi and a DC band [0, 0.1) pi, so the four shares cover the spectrum;
  - compute the shares on the visited range, e.g. up to the max count on the eval stream, recorded per run;
  - optionally also on pe - pe_init.

**4. MAJOR (registered labels may be unreachable on W1).**
- SOLVED requires a final-5% loss < 0.05. Both pilot arms end at loss 0.157 / 0.195, unsolved, with accuracy near 0.97.
- If the batch looks like that, W1's only route to BETTER is perm p < .05 with d >= +0.05. That cannot happen when the
  reference is above 0.95.
- W1 would then read NO EFFECT DETECTED or UNMEASURED by construction. GENERALITY's zero-init CoPEF, at 0.887, leaves W2 and
  W3 reachable.
- Fix, before launch:
  - register a continuous readout that is not ceiling-compressed: perm on log final-5% loss (classify_run `tail`), or on
    the T = 128 NLL already in eval.json; or
  - state in an amendment that a W1 null at ceiling is uninformative.
- Extending the budget is the alternative (rule 4).

**5. MINOR -- vacuous outcomes.**
- NO EFFECT DETECTED fires when both arms sit at the floor (the smoke shows it: 0.5059 x 3, below the 0.554 floor).
- Fix: add a gate before labelling. If both arms' mean accuracy is <= floor + 0.02 and both solve 0/n, label the contrast
  "VACUOUS (both at floor)". Print the floor (0.554) beside each contrast.

**6. MINOR -- what "below pi" means for Lo depends on the gate scale.**
- Gates are trainable and shared with content attention. If the out-and-back count grows from ~2 toward 4 (gates -> 1), Lo
  channels with omega > 0.25 pi cross pi.
- So "Lo stays sub-pi" holds only at init. On the pilot, Lo's table also ended period-2 dominated (finding 3).
- Fix: add a secondary, the mean out-and-back count at the end of training (the gates on the eval stream). This places
  omega x count for each arm, so a W1 null can be read as "Lo also reached the wrap" or "Hi did not".

**7. MINOR -- the 4T secondary is confounded by the untrained table.**
- At T = 512 the counts exceed every count seen in training. Those entries hold a weight-decayed sinusoid in Hi/Lo and
  near-zero in CoPEF.
- Fix: label the 4T row "partly reads the untrained init", or drop it from the arm comparisons.

**8. MINOR -- confirm that the re-score hooked.**
- Fix: print `rescore_hook.STATS` (hooked/fired/skipped) after the re-score loop, as the other rescore users do.

**Note (not a finding) -- the init is not washed out quickly.**
- Constant lr 1e-3, batch 64, seed 990, CPU: corr(table, init) over the visited counts is 0.97 after 100 steps and 0.96
  after 200, for both Hi and Lo. The start does shape early training, so the design premise is not void.
- By the end of the pilot, corr over counts 0-64 is 0.04-0.06, so the start is fully overwritten there.

## Verdict
Do not launch until 1 is fixed. Amend 2-4 before any batch result is read: 2 sets how W1 may be read, 3 is a bug in a
declared secondary, and 4 decides whether W1 can come out BETTER at all. 5-8 can go in the same amendment.
