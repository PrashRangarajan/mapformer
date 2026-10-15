# Fix-14 re-analyses: predictions and decision rules (written 2026-10-15 before any script was run)

Post hoc, CPU, eval-only, on committed checkpoints. Nothing here is registered. Scripts and outputs sit beside this file.
The mock review (`docs/paper/MOCK_REVIEW_draft3.md`, Part 3.1 fix 14 and questions 3, 4, 7) asked for three checks.

## R1. SOLVED threshold sensitivity (`solved_thresholds.py`)

Criteria, fixed now: final-5% mean training loss < 0.02 / < 0.05 (registered) / < 0.1; and accuracy >= 0.99 at the
training length in the batch's stored eval (eval mode, as registered; no dropout re-score).
Check first: at 0.05 every arm's count must equal the registered count (else the loader is wrong and nothing is read).
Prediction: the direction of every registered sign contrast (signed solves more) holds at all four criteria; at 0.1 some
ring |.| runs (accuracy 0.95-0.97) may cross, narrowing the ring contrasts; the torus contrasts do not move.

## R2. Channel-by-channel capture, AO_high (`capture_by_channel.py`)

For every channel of the 8 AO_high runs (and SO_high), run the one-channel flow of `potential.py` (plain gradient, |.|
or signed, 3000 steps, the THEORY_REVIEW settings) from that channel's own initial angles, and compare its predicted
end (cancelling or not) with the trained model's end, among channels with |wrap(a(+1))| > 0.05 at the end.
Not an independent prediction: the binned version (`mech_capture_out.txt`) was already seen. From it, expected: the flow
predicts capture for channels within ~0.5 rad of 2 pi and none beyond; observed capture is 0.19-0.25 beyond 0.5 rad, so
agreement is expected to be weak (Cohen's kappa < 0.2), i.e. the 28% vs 23% match is aggregate, not per channel.

## R3. Collapsed channels and the wall (`collapse_wall.py`)

Collapsed (move, channel): |a| <= 0.05 rad (raw angle; the |.| arms' raw angles are >= 0), plus the existing
definition |wrap(a(+1))| <= 0.05 for continuity with `fig_mechanism_out.txt` (44% for AO_absmag).
(a) Collapse fraction in every |.| / softplus arm with stored checkpoints (ring: RING_DIAL, RING_BUDGET, WRAP_EXISTENCE AO,
MECHANISM AO / AO_absmag / AO_mid / AO_high; torus: SIGN_MATCHED Abs / Pos, GENERALITY AO4, TORUS_START AO4 arms) and
their signed controls.
(b) For the ring observation-step-free arms (where per-move angles are the whole step table): did collapsed channels
start near zero (initial |a| from `eval.json` `init_angles` where stored)? Does attention read them (usage weight,
`model_wrap_exist.usage_weights`)? Which way does the training-loss gradient point at the end, dL/dDelta per
(move, channel) (Delta = the step after |.|; > 0 means the loss wants the step smaller, i.e. into the wall)?
Probe check first: autograd dL/dDelta must match a central finite difference on single entries (relative error < 1e-3).
Decision rules (fixed now), for the collapsed (move, channel) entries of an arm, pooled over seeds:
- WALL (P4 consistent): >= 2/3 have dL/dDelta > 0 with |dL/dDelta| >= 0.1 x the median over non-collapsed entries, and
  their median usage weight is >= 0.5 x the non-collapsed median.
- UNUSED (not P4): median usage weight < 0.2 x the non-collapsed median (attention does not read them; the zero step is
  then weight decay on an irrelevant channel).
- MIXED otherwise.
Prediction (P4 account): AO_absmag reads WALL. Competing prediction: UNUSED. No preference stated; the draft says the
question is open.

---

## Outcome (appended after the runs, 2026-10-15; outputs beside this file)

- Probe checks passed first: R1's tail rule on constructed curves and agreement with `classify_run`, and all 47 arms
  reproduce their registered counts at 0.05; R2's potential.flow predicate on constructed starts (and the batched
  re-implementation, which agreed to 2e-16 over 20 steps but drifted near the |.| fold over 400, so the script calls
  potential.flow itself); R3's autograd dL/dDelta against central finite differences (10 entries, rel. err <= 1.1e-4).
- R1 (`solved_thresholds_out.md`): prediction held except one ring cell. RING_DIAL p_plus 0.5 at loss < 0.1 reads 4/8 vs
  8/8 (Fisher p 0.077, no longer firing); at 0.02 and acc >= 0.99 it is 2/8 vs 8/8 (p 0.007). Every other sign contrast,
  every Table 2 contrast and ACCOUNTS T2 F1 fire under all four criteria; p_plus 0.9 is unmeasured under all four.
  Not predicted: CoPE on the line passes at loss < 0.1 in 8/16 runs (ring 2/16, p 0.054; acc >= 0.99 7/16 vs 2/16,
  p 0.11): the registered L1 (1/16 vs 0/16) is criterion-sensitive.
- R2 (`capture_by_channel_out.txt`): as predicted, weak. AO_high: flow 0.329 vs observed 0.230, kappa +0.086 (per run
  -0.26 to +0.51); channels within 0.5 rad of 2 pi: flow 0.95, model 0.28; 59% of the model's cancelling channels are not
  captured by the flow. SO_high kappa +0.198.
- R3 (`collapse_wall_out.md`): MIXED for AO, AO_absmag and WRAP_EXISTENCE AO (gradient into the wall for 51-57% of the
  collapsed steps; usage ratio 0.39-1.10). Neither WALL nor UNUSED. AO_mid reads WALL on 10 entries only (not
  interpretable); AO_high has 1 collapsed entry. Collapse is partly inherited from the start (77-81% of |.| steps that
  start within 0.05 rad end collapsed; signed 29-35%).
