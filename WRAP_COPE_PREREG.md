# Does a near-wrap start let a published monotone method (CoPE) learn the map? -- pre-registration (2026-10-14, before the batch)

## Question
MECHANISM (`MECHANISM_RESULTS.md`, registered): a non-negative rotary step cannot pass through zero, so its only cancelling
solution is the wrap; started with move-angle sums past pi it learns the ring map 8/8 (vs 0/8 from a small start, 2/8 from a
large sub-pi start). Design implication: start a monotone content-dependent position method near the wrap. Test it on a
PUBLISHED method that can represent ring cancellation: CoPE (GENERALITY: 0/8 on the ring at 0.887, though a period-2 position
table represents the map). CoPE has no rotation; its "wrap" is a periodic position table read at the gated count.

## Arms (`model_mono_baselines.py`; ring 32, p_plus 0.5, RING_DIAL recipe: T = 128, batch 128, 98 batches, lr 1e-3, cosine,
300 epochs, 1 layer, 2 heads, d 128; `train_ring_variant.py`; seeds 900-907, n = 8)
| arm | position table at init |
|---|---|
| CoPEF | zeros (the paper's init; as GENERALITY) |
| CoPEF_Hi | sinusoidal, pos_emb[2m, p] = 0.1 cos(omega_m p), [2m+1, p] = 0.1 sin(omega_m p), omega_m evenly in [0.6, 0.95] pi |
| CoPEF_Lo | the same with omega_m evenly in [0.1, 0.45] pi |
Every other weight is identical across arms at a seed (checked: only `cope.pos_emb` differs). At init the gates are ~0.49, so
a (move, observation) pair advances the count by ~1 and an out-and-back by ~2: in each table channel an angle ~2 omega, past
pi for Hi (1.2-1.9 pi), below pi for Lo (0.2-0.9 pi) -- the analogue of MECHANISM's high and mid starts. Amplitude 0.1 puts
the table logits on the scale of the content logits (q.k / sqrt(d) std ~0.35 at init).
Evaluation: `train_cancel.evaluate`, fresh stream, 400 trajectories, T = 128 (eval mode); SOLVED = classify_run (final 5% of
training loss < 0.05). Floor 0.554.

## Registered decisions (`analyze_wrap_cope.py`; Fisher on SOLVED, unpaired permutation on accuracy, 95% permutation CI)
Labels per contrast: BETTER (Fisher p < .05 with the first arm ahead, or perm p < .05 with d >= +0.05) / WORSE (the mirror) /
NO EFFECT DETECTED (SOLVED within 1 and |d| <= 0.01; not equivalence) / UNMEASURED.
- **W1 (primary): CoPEF_Hi vs CoPEF_Lo** -- BETTER reads "a near-wrap table start rescues CoPE" (sum past pi, not table size).
- **W2: CoPEF_Hi vs CoPEF** (vs the paper's zero init).
- **W3: CoPEF_Lo vs CoPEF** (does a sub-pi sinusoidal table change anything).
Prediction: W1 BETTER, W2 BETTER. Less certain than for the rotary step: CoPE's gates depend on the query-key pair, so the
per-move count increment is not a per-token constant and the analogy may not hold.
Secondaries (declared): accuracy at 4T; loss AUC; the dropout re-score (CoPEAttn is in rescore_hook's KNOWN list); the
learned table's power in the Hi band vs the Lo band at the end (does training keep or move the frequency content).

## Process and cost
Pilot (seed 990, outside the batch): CoPEF_Hi and CoPEF_Lo (reported, never pooled). Independent code check before launch.
24 runs at ~9-15 min, both GPUs: ~30-45 min.

## Amendment 1 (2026-10-14, after the independent check `docs/audits/2026-10-14/wrap_cope_audit.md`; before launch)
The check confirmed: arms differ only in `pos_emb`; CoPE matches Listing 1; the init arithmetic holds (gates ~0.49, an
out-and-back advances the count ~2); smoke clean. Pilot (seed 990, outside the batch, never pooled): Hi 0.970, Lo 0.975, both
unsolved (final loss 0.16 / 0.19), both above GENERALITY's default CoPEF (0.887). Changes:
- **BLOCKER fixed:** the driver called `analyze_wrap_mech` (a sed slip); it now runs `analyze_wrap_cope --root "$R" --seeds
  900-907`.
- **Slope confound, new arm CoPEF_LoS:** CoPE interpolates linearly between integer knots, so the table's knot-to-knot step
  sets the gradient reaching the gates; Hi's is ~2.2x Lo's. CoPEF_LoS = the Lo band with amplitude scaled (0.224) so its mean
  knot step matches Hi's (1.11 vs 1.05). **W1b = CoPEF_Hi vs CoPEF_LoS**, same labels. Reading: Hi beats Lo AND LoS -> the band
  (past pi) matters; Hi beats Lo but not LoS -> table steepness, not the wrap.
- **Below-ceiling readout:** both pilot arms end unsolved near 0.97, so SOLVED may not separate; registered **W1n / W1bn**: held-
  out NLL at T = 128 (from eval.json), Hi vs Lo and Hi vs LoS, unpaired permutation: LOWER NLL / HIGHER NLL / NO DIFFERENCE
  DETECTED (p < .05).
- **Floor gate:** every W label is UNREADABLE if both arms' mean accuracy is below floor + 0.05 (0.604).
- **Secondaries:** the DFT now also reports power near pi (> 0.95 pi: the period-2 construction; the pilot tables carry
  0.72-0.74 of their power there, seen incidentally by the check); rescore_hook STATS printed. Disclosed: Lo's "below pi" is
  an init statement (gates move); accuracy at 4T reads table entries beyond the trained counts (p > ~255), which keep their
  init (sinusoid for Hi / Lo / LoS, zeros for CoPEF) -- not comparable across arms.
- Arms 4 x 8 = 32 runs, 64 artifacts; ~40-60 min on both GPUs.
