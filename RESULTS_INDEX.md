# Results index (regenerated 2026-08-09, after archiving)

**47 void files moved to `archive/void/`.** Nothing there is citable;
each carries a banner naming the evidence that invalidated it, and the
diagnostics that established those verdicts are kept HERE as current results.

## The four results worth citing

1. **Path integration is necessary for in-context cognitive maps.**
   Match-Query: 0.730 ± 0.247 (n=5, 64²) and 0.823 ± 0.043 (n=3, 128²) vs
   index-position 0.154 / 0.192, chance 0.0625, no seed overlap. Survives
   gates, a context-destruction ablation (0.918 → 0.074), a 4× larger map and
   8× blind-query length. Boundary: breaks per-seed at n_obs=4.
   Corroborated independently on the family tree (+0.115).
2. **The parallel-scan claim holds.** 2.6–3.3× scaling vs 14.5× (MapEM-NC)
   and 120× (TEMFaithful) over a 16× length increase. → `TIMING_BENCHMARK.md`
3. **Paper replication** under the paper's own OOD protocol, WM and EM.
   → `PAPER_OOD_PROTOCOL.md`
4. **Two of the paper's stated-but-unmeasured conjectures refuted** —
   separate q0/k0 (+0.358 against it on Match-Query, four tasks total) and
   the value of non-commutativity (+0.005–0.014 for 34× the cost).

## Standing rules (each bought by a failure)

1. n-gram on the **action stream alone, orders 1–5**, before any
   demonstration task. A copy-previous baseline tests order 1 only and
   certified a task whose order-3 shortcut was 0.971.
2. **Context-destruction ablation** on trained models. hier-goal: 0.912 →
   0.913. Match-Query: 0.918 → 0.074.
3. **Never compare a fresh variant to a stored baseline.**
4. **Report the measured chance rate** — it is often not what it looks like
   (0.500 = the always-predict-blank floor; the family tree's floor is the
   0.146 hub baseline, not 0.125 chance).
5. **Verify the training budget** before reading a chance-level table as a
   negative. Map-Query needed 8× its initial budget.
6. **Three seeds is not a point estimate.** 0.888 ± 0.140 (n=3) became
   0.730 ± 0.247 (n=5) on the same config.
7. **A gate must call the task code, not reimplement it.** A validator that
   duplicated the walk silently tested a different task from the trainer.

## VERIFIED — safe to cite (14)

- `EM_COMP_SAMEBATCH.md`
- `FAMILY_TREE_GATES.md`
- `FAMILY_TREE_RESULTS.md`
- `MATCH_QUERY_EM.md`
- `MATCH_QUERY_GATES.md`
- `MATCH_QUERY_LONGQ.md`
- `MATCH_QUERY_RESULTS.md`
- `MATCH_QUERY_SCALE.md`
- `NOISE_CLEAN_REVALIDATION.md`
- `PAPER_OOD_PROTOCOL.md`
- `PAPER_TASK_ACCURACY.md`
- `PAPER_VALIDATION.md`
- `TIMING_BENCHMARK.md`
- `VOCAB_SWEEP_MULTISEED.md`

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

