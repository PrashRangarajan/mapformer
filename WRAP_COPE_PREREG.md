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
