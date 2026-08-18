# Results index (regenerated 2026-08-18)

**48 void files live in `archive/void/`.** Nothing there is citable; each carries
a banner naming the evidence that invalidated it. The DIAGNOSTICS that
established those verdicts are kept HERE — they are current results.

---

## THE HEADLINE

**Path integration is what makes in-context cognitive maps work, and it is not
the positional-encoding scheme.**

A 2x2 on **the paper's own task**, at the paper's own config, matched recipe,
seeds and parameters (within 0.4%), against a **measured** always-predict-blank
floor of **0.506** (`INDEX_BASELINE_PAPER_TASK.md`):

| encoding | index position | path-integrated |
|---|---|---|
| **RoPE** | 0.514 +/- 0.004 | 0.989 +/- 0.011 (`Vanilla`) |
| **PoPE** | 0.509 +/- 0.004 | **1.000 +/- 0.001** (`MapPoPE-Flat`) |

Both index cells sit at the floor; both path-integrated cells solve it. The
encoding moves the result ~0.005-0.011; path integration moves it ~0.48. It holds
under the paper's own OOD-d / OOD-s protocol too (`PAPER_OOD_WITH_POPE.md`).

Four independent legs support it:

1. **The paper's task** — the 2x2 above, plus OOD.
2. **Match-Query** (shortcut-gated, invented here) — 0.730 +/- 0.247 (n=5, 64^2)
   and 0.823 +/- 0.043 (n=3, 128^2) vs index 0.154 / 0.192, chance 0.0625, no
   seed overlap; survives context destruction (0.918 -> 0.074).
   The decisive within-task control: MapPoPE-Hier (PoPE + path int.) 0.847 vs
   PoPE-Flat (PoPE + index) 0.117 — same attention, opposite outcome.
3. **Family tree** (non-commutative relational structure) — path integration
   +0.115 over index; passes context destruction, with the map-destroyed
   condition landing on the 0.163 hub floor to three decimals.
4. **The residual is explained, not waved away** — index models exceed the floor
   only at recurrence interval 1-2 (+0.05 to +0.07), i.e. out-and-back retraces
   readable from the ACTION TOKENS AS CONTENT with no position code
   (`REVISIT_DISTANCE.md`). The same +0.07 reappears at OOD-d.

What changed on 2026-08-18: this claim used to rest almost entirely on
Match-Query, a task invented here. It now rests on the paper's own benchmark.

---

## Other results worth citing

**MapPoPE-Flat is the strongest configuration measured on this benchmark.**
1.000 / 0.995 / 0.996 (IID / OOD-d / OOD-s l=512) against the paper's reported
MapWM 0.99/0.99/0.96 and MapEM-os 1.0/0.99/0.97. PoPE is not a better idea than
RoPE in itself — it is inert without path integration (0.509) — but combined with
it, it beats everything else here. `PAPER_OOD_WITH_POPE.md`

**Level 1.5's correction is MEASUREMENT-DRIVEN, and now bounded.**
+24.8pp over Vanilla on lm200 OOD T=512 (`LM200_CORRECTED_MULTISEED.md`, n=3);
1.000 vs 0.993 on the clean paper task at 50 epochs with 16x lower training loss;
but **no advantage on Match-Query** (0.876 vs 0.888), whose blind query phase
gives the filter nothing to correct with. Not capacity — a control with MORE
parameters scores below both. `LEVEL15_MEETS_GATED_*.md`

**The parallel-scan claim holds.** 2.6-3.3x scaling vs 14.5x (MapEM-NC) and 120x
(TEMFaithful) over a 16x length increase. `TIMING_BENCHMARK.md`

**Two of the paper's stated-but-unmeasured conjectures are refuted.**
Separate q0/k0 (+0.358 against it on Match-Query, four tasks total) and the value
of non-commutativity (+0.005-0.014 for 34x the cost).

**CSCG's stitching negative control reproduces in MapFormer's attention.**
Paired difference +0.131 +/- 0.024 (floor exactly 0) vs index -0.005 +/- 0.016;
transitive retrieval 2.98x vs a 3.32x within-room yardstick. `STITCH_ATTENTION.md`

---

## Standing rules (each bought by a failure)

1. **n-gram on the ACTION STREAM ALONE, orders 1-5**, before any demonstration
   task. A copy-previous baseline tests order 1 only and certified a task whose
   order-3 shortcut was 0.971.
2. **Context-destruction ablation on trained models.** hier-goal: 0.912 -> 0.913
   (void). Match-Query: 0.918 -> 0.074 (passes). As of 2026-08-18, family tree
   and compositional have also passed; three of the four citable results have
   now been through this gate, where before only one had.
3. **Never compare a fresh variant to a stored baseline.** Retrain every arm in
   the same batch.
4. **Report the measured chance rate**, and check which column it belongs to.
   The index models read 0.80 on OOD-s and 0.27 on OOD-d while doing the SAME
   thing — the blank floor moved with p_empty.
5. **Verify the training budget before reading a weak number as a negative.**
   Level15 on the paper task: 0.938 at 16 epochs, **1.000 at 50**. Map-Query
   needed 8x its initial budget.
6. **Three seeds is not a point estimate.** 0.888 +/- 0.140 (n=3) became
   0.730 +/- 0.247 (n=5) on the same config.
7. **A gate must CALL the task code, not reimplement it.** A validator that
   duplicated the walk silently tested a different task from the trainer.
8. **A retraction must be applied to the GENERATORS, not just recorded.**
   `eval_paper_task.py` kept printing "the paper reports MapFormer-WM 0.955,
   MapFormer-EM 0.999" into every report it produced for months after those
   figures were retracted in CLAUDE.md — they appear in no table of the paper.
   Fixed 2026-08-18. Grep the code for retracted numbers, not just the docs.

Two method notes that are not rules but cost real time:

- **`shuffle` and `resample` are not interchangeable.** Permuting slots also
  destroys the walk's autocorrelation and puts the input off-manifold;
  substituting a stream from an independent episode does not. On the paper task
  resample was MORE destructive (0.178 vs 0.231); on family tree and
  compositional it was LESS. Report both.
- **An ablation landing BELOW the floor** means the model fails confidently
  rather than hedging (NLL 3.7-5.6 against ln(21)=3.04 for uniform). Worth
  checking with an on-manifold resample before blaming the manipulation.

---

## VERIFIED — safe to cite (24)

- `ABLATE_COMPOSITIONAL.md`   (new 2026-08-18)
- `ABLATE_FAMILY_TREE.md`     (new 2026-08-18)
- `EM_COMP_SAMEBATCH.md`
- `FAMILY_TREE_GATES.md`
- `FAMILY_TREE_RESULTS.md`
- `INDEX_BASELINE_PAPER_TASK.md`  (new 2026-08-18)
- `LEVEL15_MEETS_GATED_matchq.md` (new 2026-08-18)
- `LEVEL15_MEETS_GATED_paper50.md` (new 2026-08-18)
- `LM200_CORRECTED_MULTISEED.md`  (RECLASSIFIED — see note below)
- `MATCH_QUERY_EM.md`
- `MATCH_QUERY_GATES.md`
- `MATCH_QUERY_LONGQ.md`
- `MATCH_QUERY_RESULTS.md`
- `MATCH_QUERY_SCALE.md`
- `NOISE_CLEAN_REVALIDATION.md`
- `PAPER_OOD_PROTOCOL.md`
- `PAPER_OOD_WITH_POPE.md`     (new 2026-08-18)
- `PAPER_TASK_ABLATION.md`     (new 2026-08-18)
- `PAPER_TASK_ACCURACY.md`
- `PAPER_VALIDATION.md`
- `REVISIT_DISTANCE.md`        (new 2026-08-18)
- `STITCH_ATTENTION.md`        (new 2026-08-18)
- `TIMING_BENCHMARK.md`
- `VOCAB_SWEEP_MULTISEED.md`

**Reclassification note.** `LM200_CORRECTED_MULTISEED.md` was previously filed
under DIAGNOSTICS, which made it read as a post-mortem rather than a current
result. It is the corrected, fresh, multi-seed lm200 leaderboard under current
code and it is citable: Level15 0.990 +/- 0.005 vs Vanilla 0.742 +/- 0.075.
Its one outstanding gap is that lm200 has never had a context-destruction
ablation (rule 2).

## DIAGNOSTICS — what invalidated things (10)

- `AP_KERNEL_DIAGNOSTIC.md`
- `CORRECTED_LM200_LEADERBOARD.md`
- `HIERGOAL_ABLATION.md`
- `HIERGOAL_CLOSEDLOOP.md`
- `LAP_GATES.md`
- `LAP_TRANSFER.md`
- `LAP_TRANSFER_NOREWARD.md`
- `LM200_CORRECTED_MULTISEED.md`
- `MAP_QUERY_GATES.md`
- `PLANNER_TASK_AUDIT.md`

## PARTIAL — lm200 rows void, clean/noise VALID (14)

- `CAPACITY_PERREGIME.md`
- `GSF_FULL_RESULTS.md`
- `LEVEL15BETA_RESULTS.md`
- `NOBYPASS_RESULTS.md`
- `NODROP_PARETO_RESULTS.md`
- `RESULTS_PAPER.md`
- `TEM_NOISE_FFN_RESULTS.md`
- `TEM_RESULTS.md`
- `TEM_T_MULTISEED.md`
- `TEM_T_RESULTS.md`
- `V3_RESULTS.md`
- `V4_CONTROL_RESULTS.md`
- `V4_MULTISEED.md`
- `V4_RESULTS.md`

## CONTAINS RETRACTED CLAIMS (3)

- `REPORT_ADDENDUM.md`
- `REPORT_v2.md`
- `RESULTS_SUMMARY_2026-05-10.md`

## SUPERSEDED / corrected (4)

- `CAPACITY_CONTROL.md`
- `COMPOSITIONAL_RESULTS.md`
- `MAP_QUERY_RESULTS.md`
- `VOCAB_SWEEP_RESULTS.md`

## Known-open, ranked

1. **Level15 / correction line vs the gated tasks, beyond n=3.** The Match-Query
   comparison is underpowered (sd 0.14-0.26; every arm has a 1.000 seed).
   It establishes "no advantage", not "a deficit".
2. **lm200 context-destruction ablation** — the +24.8pp result has never been
   through rule 2. Blocked: no lm200 checkpoints on the koopman machine.
3. **`DOG_RESULTS_FIXED.md` has never been produced.** The existing
   `DOG_RESULTS.md` used all-zero DoG targets (unnormalised Gaussians cancel at
   d=0), so the Sorscher hex test is VACUOUS, not negative.
4. **`STOCHASTIC_TRANSITION_RESULTS.md` never landed** (queued 2026-05-01).
5. **RoPE on Match-Query** — its index controls are PlainFlat and PoPE-Flat, not
   the architecture-matched RoPE that made the paper-task result tight.
6. **Map-Query** — room query is learnable at 7.6x chance but on ONE variant,
   ONE seed; the multi-seed table is the 25-epoch undertrained one, and
   `train_map_query.py` still defaults to 25 epochs.
7. **Schema task** — still NOT READY; needs the redesign stitch received.
8. **Vanilla (WM) on the family tree** — it has EM and MapEM-NC but no plain WM.

## other current (79)

- `AGGREGATE_EXTRAS.md`
- `AGGREGATE_MULTISEED.md`
- `AGGREGATE_TASK_RESULTS.md`
- `BOUNDED_MEMORY.md`
- `BOUNDED_MEMORY_RESULTS.md`
- `BUMP_TOKEN_RESULTS.md`
- `CASCADE_MULTISEED_RESULTS.md`
- `CASCADE_REPRO_TEST.md`
- `CASCADE_ZEROSHOT_S0.md`
- `CLAUDE.md`
- `CLOCK_SCAN.md`
- `CLONE_ANALYSIS_LEVEL15PC.md`
- `CLONE_TRANSFER_NOBYPASS.md`
- `CNAV_HEX_Level15.md`
- `CNAV_HEX_Level15EM.md`
- `CNAV_HEX_Vanilla.md`
- `CNAV_HEX_VanillaEM.md`
- `CNAV_RESULTS.md`
- `COMPOSITIONAL_EXPERIMENT.md`
- `COMPOSITIONAL_MULTISEED.md`
- `DAGGER_DK6_RESULTS.md`
- `DAGGER_EMPTY_RESULTS.md`
- `DAGGER_RESULTS.md`
- `DETAILED_RESULTS.md`
- `DOG_RESULTS.md`
- `DOORKEY_BC_RESULTS.md`
- `EM_FIX_COMP.md`
- `EM_HOPFIELD_CROSSSCALE.md`
- `EM_P0_COMP.md`
- `EM_P0_PAPER.md`
- `EXTRAHEAD_CONTROL.md`
- `FAMILY_TREE_D7_GATES.md`
- `GENERALIZATION_REPORT.md`
- `HIERGOAL_LONGT.md`
- `HIER_ATTN_LONGT.md`
- `HIPPOCAMPAL_ANALYSIS.md`
- `HIPPOCAMPAL_GRID.md`
- `HIPPOCAMPAL_GRIDL15PC.md`
- `HIPPOCAMPAL_GRID_FREE.md`
- `HIPPOCAMPAL_HIDDEN.md`
- `HIPPOCAMPAL_HIDDEN_GRIDFREE.md`
- `HIPPOCAMPAL_LEVEL15PC.md`
- `HOPFIELD_NOMAINAP_RESULTS.md`
- `HOURGLASS_README.md`
- `LEVEL15EM_CROSSSCALE.md`
- `LONG_SEQ_clean.md`
- `MATCH_GATES_128_16.md`
- `MATCH_GATES_64_16.md`
- `MATCH_GATES_64_4.md`
- `MINIGRID_DK16_RESULTS.md`
- `MINIGRID_DOORKEY_CACHED.md`
- `MINIGRID_DOORKEY_LONGT.md`
- `MINIGRID_DOORKEY_RESULTS.md`
- `MINIGRID_DOORKEY_ROPE_DIAG.md`
- `MINIGRID_MEMORY_RESULTS.md`
- `MULTICLASS_MULTISEED_RESULTS.md`
- `MULTICLASS_RESULTS.md`
- `MULTISEED_FOLLOWUP.md`
- `MULTISEED_FOLLOWUP_RESULTS.md`
- `NUMBERLINE_RESULTS.md`
- `OMEGA_RESCALE_clean.md`
- `OOD_GRID_RESULTS.md`
- `PERSCALE_OMEGA_RESULTS.md`
- `PER_VISIT_clean.md`
- `PUBLICATION_VENUES.md`
- `README.md`
- `RECURSIVE_RESULTS.md`
- `REPORT.md`
- `ROUTE_ATTN_RESULTS.md`
- `R_T_DISTRIBUTION_3WAY.md`
- `SESSION_2026-05-01.md`
- `SESSION_HIERARCHICAL_CASCADE.md`
- `SPACETIME_HIER_RESULTS.md`
- `TEM_BACKGROUND_BASELINES.md`
- `TEM_CROSSSCALE_DIAGNOSTIC.md`
- `TOPOLOGY_RESULTS.md`
- `VECTOR_NAV_V2_RESULTS.md`
- `ZERO_SHOT_TRANSFER_clean.md`
- `ZERO_SHOT_TRANSFER_clean_brokeninit.md`
