# Does PoPE's score rule rescue per-head rank 2 on the long-walk torus? -- pre-registration (2026-10-05, before any run of the batch)

## Question
On the paper torus at T=128, PoPE's score rule made rank 2 work: MapPoPE with MapWM's 32 angles (`MapPoPE-Pair`)
solved 16/16 seeds against MapWM rank 2's 10/16, at the same rank and angle count (`MAPPOPE_PAIR_RESULTS.md`). On the
long-walk torus (trained and tested at T=1024, 900 epochs) MapWM at per-head rank 2 solves 0-2/8 and at rank 4 8/8;
a rank-2 solution exists and is held under training, so the failure is search, not capacity (`RANK_MI_RESULTS.md`,
`RANK_SEP_RESULTS.md`, `RANK_PROJ_RESULTS.md`); 2x the budget does not rescue it (`LOOP_RANK_E1800_P1_RESULTS.md`).
**Does PoPE's score rule rescue rank 2 at T=1024?**

Why it matters (theory T1, `docs/theory/2026-10-04/00_PLAN.md`): at per-head rank r = D a head is drift-free only if
the step common to every move is exactly zero; failed rank-2 runs end as CLOCK (drift) or COLLAPSE (lost axis). That
lemma is about the angle map and does not involve the score. In MapWM's score `q^T R(dtheta) k` each pair's content
(the orientation of q and k within the pair) adds a token-dependent phase, so content can move the attention peak; in
PoPE's `sum_c softplus(q_c) softplus(k_c) cos(dtheta_c + delta_c)` the phase is position-only plus a learned constant
per element, so content cannot move the peak (up to the limited content phase that untied per-element deltas give a
pair -- `MAPPOPE_PAIR_PREREG.md` Amendment 1). If PoPE's score rescues rank 2, the rank-2 search failure depends on
the score letting content move the peak; if it does not (while PoPE at rank 4 trains), the failure lives in the angle
map's search alone, as T1 reads it.

## Arms (one batch, all retrained, rule 12) -- a 2 x 2 at matched initialisation
| arm | variant | score | per-head rank | angles / head | params | seeds |
|---|---|---|---|---|---|---|
| A2 | `Vanilla` (`model.MapFormerWM`) | MapWM | 2 (shared latent) | 32 | 204,373 | 10-21 (n=12) |
| P2 | `MapPoPE-Pair` (`model_pope_pair`, unchanged) | PoPE | 2 | 32 | 204,501 | 10-21 (n=12) |
| A4 | `Vanilla_r4mi` (`model_rank_perhead`, unchanged) | MapWM | 4 (shared latent) | 32 | 204,757 | 10-17 (n=8) |
| P4 | `MapPoPE-Pair_r4mi` (`model_pope_pair_mi`, NEW) | PoPE | 4 | 32 | 204,885 | 10-17 (n=8) |

Within a rank only the score rule differs; within a score rule only the bottleneck differs. **Matched initialisation**
(`docs/audits/2026-10-05/score_rank_init_check.py` / `_out.txt`, PASS on seeds 10-17 and 100): at a given seed all four
arms share the embeddings, omega and readout; A2 and P2 share the rank-2 bottleneck; A4 and P4 share the rank-4
bottleneck; A2 and A4 share the MapWM layer; P2 and P4 share the PoPE layer (incl. `pope_delta`). P4 is built as base
r2, then the r4 bottleneck drawn exactly as `Vanilla_r4mi` draws it, then the RNG rewound to the post-base state and
the PoPE layer drawn exactly as `MapPoPE-Pair` draws it. Further checks there: 32 angles per head in every arm, per-head
rank 2/2/4/4, every arm causal, P4 with P2's weights embedded in its rank-4 bottleneck gives P2's logits (max diff
<= 9.5e-07, float32 rounding from the longer reduction); the check FAILS on two positive controls (no RNG rewind; the older
`MapPoPE-Pair_r4`).

**Why `Vanilla_r4mi` and not `Vanilla_r4`.** Both solve 8/8 at this recipe (rank_mi C; rank_matched_e900 `Vanilla_r4`),
but `Vanilla_r4mi` is the positive control already validated twice here (RANK_MI 8/8; its seed-0 rerun bitwise in
RANK_SEP), and it shares every non-bottleneck initial weight with `Vanilla` at the same seed, which is what makes the
full 2 x 2 matched. `Vanilla_r4` draws its wider bottleneck inside the base constructor and shares no initial weight
with `Vanilla`; so does the pair batch's `MapPoPE-Pair_r4`, which is therefore not used.

Entry points: `train_score_rank.py` (= `train_variant.main()` with `MapPoPE-Pair` and `MapPoPE-Pair_r4mi` added to
VARIANT_MAP; `train_variant.py`, which earlier batches md5-guard, is not edited) and `eval_score_rank.py` (runs
`eval_noise_refine` or `eval_rank_strata` unchanged via runpy with the same registration).

## Recipe (= `run_rank_mi.sh` / `run_rank_sep.sh` exactly)
Torus 64x64, 16 observation types, p_empty 0.5, no landmarks; 900 epochs x 98 batches, batch 16, T=1024 (2047 input
tokens), 1 layer, 2 heads, d 128, AdamW lr 1e-3 wd 0.05, cosine (5% warmup, decay to 10%), `--data-workers 3`,
`--save-full-state`, explicit attention path (no `--fast-attn`; the PoPE layer has its own attention). The data stream
is seeded by `--seed` only (`data_parallel`, base seed = torch.initial_seed()), so the four arms at a seed see
identical walks on the identical training map.

Seeds: 10-21 / 10-17 are fresh at this recipe (every rank batch used 0-7). They are the same training-map seeds as the
T=128 pair batch (10-25), a different task length; no checkpoint is shared.

## Evaluation (driver `run_score_rank.sh`)
`eval_noise_refine` (via `eval_score_rank`): noise 0, held-out map (env seed 10000), 100 walks per run (np seed
1234 + s), lengths 512 / 1024 / 2048, eval mode; **T=1024 is registered (matched length)**. `eval_rank_strata` at 1024
(asserts its 'all' column equals the registered JSON). `rescore_hook --scale auto` re-score at 1024 (secondary).

## Floors on this eval stream (`docs/audits/2026-10-05/score_rank_floors.py` / `_out.txt`, no model, seeds 10-25)
| T | always-blank | best token n-gram (orders 0-5, fit on the training maps) | retrace | wrap-only share of revisits |
|---|---|---|---|---|
| 512 | 0.509 | 0.588 | 0.884 | 0.022 |
| **1024** | **0.507** | **0.576** | **0.843** | 0.077 |
| 2048 | 0.508 | 0.566 | 0.790 | 0.183 |

Retrace = while a run of actions reverses the previous run, step j copies the observation 2j steps back (walks are
runs of 1-10 repeated actions). **The rank batches quoted only the 0.507 constant floor; the trivial retrace predictor
reaches 0.843 at T=1024.** MapWM r2's stored mean (0.894) is 0.05 above it, and RANK_MI's seed 6 (0.773) was below
it. Accuracy for unsolved runs therefore sits near a map-free predictor; SOLVED is the readout that says whether the
map was found. Every accuracy is reported beside these floors.

## Registered primary: P2 vs A2 at T=1024
- **SOLVED** (`stats_core.classify_run`: final-5% training loss < 0.05), two-sided Fisher (`fisher_solved`); fires if
  p < .05, direction from the rates. Unlike the pair batch, a Fisher firing COUNTS here: the question is whether training
  finds the solution, and SOLVED is untouched by the eval-mode dropout-scale issue (theory T2).
- **Accuracy** at T=1024, permutation test (`stats_core.perm2_p`, exact up to 250k relabellings, else 200k MC); fires
  if p < .05 AND |d| >= 0.02 (materiality floor; the analysis prints the MDE beside it).

**Branches** (`analyze_score_rank.decide`, evaluated in this order; every branch smoke-tested on synthetic data):
0. **VOID** -- positive control fails: A4 solves <= 5/8. (MapWM r4 has solved 24/24 at this recipe; at a true rate 0.9,
   P(<= 5/8) = 0.038.) Also VOID: a missing checkpoint, an md5 trip (at launch and again before eval), the pilot's
   reproduction check failing.
1. **CEILING** -- both rank-2 arms >= 0.999 on every seed: undetermined.
2. **NO DEFICIT TO RESCUE** -- A4 vs A2 on SOLVED does not fire (MapWM's rank-2 failure did not replicate on fresh
   seeds; with A4 at 8/8 it fires for A2 <= 6/12). Contrasts printed, no rescue verdict.
3. **RESCUE** -- SOLVED and accuracy both fire positive, and P2 is not detectably below P4 on SOLVED (Fisher p >= .05).
   With A2 at 0/12 and P4 at 8/8 this needs P2 >= 7/12.
4. **PARTIAL RESCUE** -- both fire positive but P4 solves detectably more than P2 (rank still matters under PoPE's
   score); or exactly one of SOLVED / accuracy fires positive and the other does not fire negative (which one is printed).
5. **CONFLICT** -- SOLVED fires positive, accuracy fires negative.
6. **POPE SCORE HURTS AT RANK 2** -- accuracy fires negative, SOLVED does not fire positive.
7. **NO RESCUE** -- neither fires; accuracy "unmeasured below MDE".
Qualifier on any branch past 2: if P4 solves detectably fewer than A4 (Fisher p < .05), "PoPE's score fails at rank 4
too: the rank-2 contrast is uninformative about rescue".
Multiplicity: RESCUE needs both readouts; a one-readout PARTIAL runs at a family-wise rate near 0.10 and is labelled.

## Power (`docs/audits/2026-10-05/score_rank_power.py` / `_out.txt`; computed, and n fixed, before the pilot outcome was seen)
Fisher (two-sided .05) power for P2 > A2 with A2's true solve rate 0 (observed 0/8): at n=8, 0.36 / 0.89 / 1.00 for a
P2 solve rate of 0.5 / 0.75 / 1.0; **n=12, 0.81 / 1.00 / 1.00**; n=16, 0.96 / 1.00 / 1.00. At A2 rate 0.125: n=12 0.36 /
0.86 / 1.00. Minimum P2 count that fires against 0/n: 5/8, 5/12, 5/16. Joint (both readouts fire; Monte Carlo from
RANK_MI's stored accuracies): n=12 0.41 / 0.82 / 1.00. **Rank-2 arms n=12 (seeds 10-21); rank-4 arms n=8** (a positive
control and a sanity arm). A full rescue is detected at any n; n=12 buys a half rescue on SOLVED (0.36 -> 0.81) for
8 more runs (~2.6 h, below). A rescue of a quarter of the seeds is unmeasured at every n considered.

## Declared secondaries (no verdict)
- S1 rank effect within MapWM (A4 - A2): fresh-seed replication of RANK_MI. S2 within PoPE's score (P4 - P2). S3 the
  score rule at rank 4 (P4 - A4). SOLVED (Fisher) and accuracy (perm) each.
- S4 **basins (T1)**: `docs/theory/2026-10-04/scripts/basins.py`'s classification, re-implemented in
  `analyze_score_rank.head_stats` -- per head kappa (w-weighted norm of the per-move common step over the mean axis norm)
  and indep (s_min/s_max of the w-weighted D x nb axis matrix); a run is CLEAN if some head has kappa <= 0.01 and indep
  >= 0.2, COLLAPSE if some head has kappa <= 0.01 but none such has indep >= 0.2, else CLOCK. **Channel weights,
  declared now:** MapWM as basins.py (w_j = mean over action-token queries of |q_pair j| x mean over observation-token
  keys of |k_pair j|); PoPE-Pair, whose angle j drives elements 2j and 2j+1, w_j = |sum over those two elements e of
  mean_action softplus(q_e) x mean_obs softplus(k_e) x exp(i delta_e)| (the amplitude of channel j's cosine in the
  score at the mean magnitudes, delta clamped as in the forward). Unweighted also printed. Readouts: per arm, the count
  of runs where "SOLVED iff CLEAN" holds; CLOCK runs P2 vs A2 (Fisher). T1 as an end-state lemma does not involve the
  score, so it predicts "SOLVED iff CLEAN" in all four arms, PoPE included; a PoPE run that solves with no clean head
  would contradict it. Post hoc thresholds (set on the 80 rank runs); end states, not causes.
- S5 revisit strata at T=1024 (`eval_rank_strata`): wrap-only, plain lag >= 128, plain lag < 128, each with its
  best-constant floor; P2 - A2 per stratum.
- S6 T=512 (shorter than training) and T=2048 (past it; rule 10, robustness not capability).
- S7 r(final-5% loss, acc@1024) over all 40 runs and within arm. S8 dropout-scale re-score (attention x 1/(1-p);
  a one-layer correction, valid here). S9 revisit NLL at 1024. S10 run classes (SOLVED / STALLED / DESCENDING / RISING).
- S11 speed: the epoch at which each run's 10-epoch running loss first falls below 0.05. PoPE learns faster at T=128
  too; a rescue within 900 epochs could be speed. (MapWM r2 does not solve at 1800 epochs either, so speed alone does not
  rescue MapWM, but this batch cannot separate "PoPE finds the basin" from "PoPE gets there sooner".)

## Pilot (`docs/audits/2026-10-05/score_rank_pilot.sh`; readout `score_rank_pilot_check.py` / `_out.txt`; seeds 0, 100, 101)
(a) **Reproduction** (pass = bitwise-equal per-epoch losses): `Vanilla` s0 trained through `train_score_rank` for the
full 900 epochs against the stored `runs/rank_mi/p0/Vanilla_s0` -- **PASS: 900/900 epochs bitwise equal (max diff 0.0)**.
The c4 and c8 reruns of the same pilot arm/seed also give identical final losses: concurrency does not change results.
(b) **Timing** (per-epoch wall time printed by train.py): 8 concurrent (4/GPU, 2 RTX 4090): MapWM 8.5-9.1 s/epoch,
PoPE-Pair 11.9-12.4; 4 concurrent (2/GPU): MapWM 4.3, PoPE-Pair 6.0-6.2. Per-GPU throughput is the same (0.39 vs 0.40
epochs/s for one MapWM + one PoPE job), so the driver runs 2/GPU, as the rank batches did.
(c) **Outcome, READ** (a 40-epoch cosine schedule, not the batch's; seeds outside the batch): PoPE-Pair r2 SOLVED on
both seeds (final-5% loss 0.021, 0.014; T=1024 accuracy 1.000 on a 3-walk smoke eval of s100), MapWM r2 STALLED
(1.20, 1.34; 0.647); MapWM r4 not solved at 40 epochs (0.758); PoPE-Pair r4 SOLVED on 1 of 2. The branches,
`decide()`, n=12 and the power table were written before this outcome was seen (the early training-loss prints had been
seen: epoch 15, PoPE-Pair 0.41 vs MapWM 1.32); nothing was changed after it. It
makes RESCUE the likely outcome, and the batch's job is to say whether it holds over 12 fresh seeds at the registered
recipe.

## Cost (measured)
Total work 40 runs x 900 epochs = 36,000 run-epochs, half MapWM half PoPE. At 2/GPU the measured mixed throughput is
~0.40 run-epochs/s per GPU -> ~0.79/s on two GPUs -> **~12.7 h of training** (+ ~0.5 h queue tail), then evaluation
(40 runs x 512/1024/2048 x 100 walks, strata, re-score) ~1 h: **ETA ~14 h** from launch. Per run: MapWM ~65 min,
PoPE-Pair ~92 min at 2/GPU. n=8 for the rank-2 arms would be 32 runs, ~11.5 h; n=16, 48 runs, ~16.5 h.

## What it can and cannot show
- Can: whether PoPE's score rule, at MapWM's rank, angle count, initialisation and data, lets a rank-2 path model find
  the T=1024 torus map within 900 epochs; with A4 / P4, whether rank still matters under PoPE's score.
- Cannot: which part of the score bundle does it (non-negative magnitudes, position-only phase, per-element deltas,
  positive score mean); speed vs basin (S11 describes, does not separate); any task but the torus, n_heads=2, 1 layer,
  one recipe; T1 as a cause (basins are end states).

## Void conditions
Positive control (branch 0); any of 40 checkpoints missing; md5 guard trips at launch or before eval (training code,
models, wrappers, evaluators, rescore hook, analysis, stats_core); the pilot reproduction failing.


## Amendment 1 (2026-10-05, after an independent code audit, BEFORE the batch was launched)
The audit (read-only, blind to the PoPE pilot outcomes) re-verified: the init check (byte-identical, PASS on seeds
10-17 and 100, positive controls fail), the recipe and eval flags against RANK_MI, the pilot reproduction (900/900
per-epoch losses and final weights bitwise equal to runs/rank_mi/p0/Vanilla_s0), the wrappers (no variant-name
branching; strict loads), the driver (40 runs, md5 at launch and before eval), every registered branch reachable on
synthetic data, and the thresholds (premise fires at A2 <= 6/12; SOLVED fires at P2 >= 5/12 vs 0/12; Fisher power at a
true P2 rate of 0.5 is 0.806). Changes:
1. **BUG fixed (analysis).** SOLVED firing NEGATIVE with accuracy POSITIVE was labelled PARTIAL RESCUE; it is now
   CONFLICT. SOLVED NEGATIVE with accuracy not firing was labelled NO RESCUE; it now reads POPE SCORE HURTS AT RANK 2
   ("fires on SOLVED"). PARTIAL requires the other readout not to fire negative. Both cases re-tested.
2. **Wording.** "Within a rank only the score rule differs" was overstated. At a seed, token_emb, omega, out_norm /
   out_proj and the layer norms are shared by all four arms; action_to_lie within a rank; q/k/v/o and FFN within a score
   rule only (A2 = A4, P2 = P4). The primary P2 - A2 is matched in data, map, embeddings, omega, readout and bottleneck,
   NOT in the attention/FFN initial draws (independent draws from the same distribution). Docstrings fixed.
3. **RESCUE is reported with P2's count and a 95% Clopper-Pearson interval**, never as "full": with P4 at 8/8, P2 7/12
   reads RESCUE (P2 vs P4 Fisher p 0.055) and 6/12 PARTIAL (p 0.042), and at a true rate of 0.5, P(P2 >= 7/12) = 0.39.
4. **Dropout-scale re-score**: if the accuracy readout's firing state differs between eval mode and the re-score, the
   verdict line is FLAGGED; the verdict itself is unchanged.
5. **SOLVED is measured on the training map**: the verdict line also prints the held-out accuracy of P2's SOLVED runs
   against the retrace floor (0.843).
6. The md5 guard is re-checked before the analysis step too.
Noted, no change: the speed readout (S11) is descriptive and partly selected (it averages only runs that reach the
threshold; a dip below 0.05 counts); "MapWM r4 solved 24/24 at this recipe" mixes shared and per-head r4 classes (the
VOID threshold is still sound: P(A4 <= 5/8) = 0.038 at a true rate of 0.9); the registered accuracy test is unpaired
although arms share data per seed (conservative).
