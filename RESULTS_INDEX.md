# Results index — current state (2026-05-21)

Master index of all live result files after the dead-end cleanup. Files
that were incorrect, superseded, or dead-end were moved to
`archive_stale/` (see bottom). Start here, then read the consolidated
reports.

## Consolidated reports (read these first)

| File | What it is |
|---|---|
| `REPORT.md` | Full consolidated report — all multi-seed results, controls, mechanisms, probes (8 sections). |
| `REPORT_ADDENDUM.md` | Deltas since REPORT.md — active-inference null, per-scale ω, SR-aux null, cwd fix. |
| `GENERALIZATION_REPORT.md` + `.pdf` | Generalization-only deep dive — 12 experiments, motivation→setup→results→interpretation. |
| `RESULTS_PAPER.md` | Canonical multi-seed paper table. |
| `RESULTS_SUMMARY_2026-05-10.md` | Cognitive-map necessity table (6 demands). |
| `DETAILED_RESULTS.md` | Long-form experiment history. |

## Headline: TEM-setting novel-environment generalization

| File | Status |
|---|---|
| `TEM_NOVEL_ENV_RESULTS.md` | **PRIMARY.** n=3, 4 axes: multi-env / cross-topology / cross-scale / cross-class. |
| `MULTIENV_CLEAN_2x2.md` | n=3, multi-env clean×lm200 2×2 disambiguation. |
| `MULTICLASS_MULTISEED_RESULTS.md` | n=3, cross-class (torus + DoorKey). |
| `MULTISEED_FOLLOWUP_RESULTS.md` | n=3, cross-topology / cross-scale / multi-env tightened. |
| `TEM_BACKGROUND_BASELINES.md` | TEMFaithful single-env + multi-env clean (n=3). |
| `TOPOLOGY_RESULTS.md`, `MULTISIZE_RESULTS.md`, `MULTIENV_RESULTS.md` | Older single-seed precursors — kept for history. |

## Controls (essential for defending the headline)

| File | Status |
|---|---|
| `VANILLANODROP_CONTROL.md` | n=3. Proves InEKF does real work, not just dropout removal. |
| `EXTRAHEAD_CONTROL.md` | n=3. Cross-scale Hopfield win = extra-head capacity, not Hopfield structure. |
| `CAPACITY_CONTROL.md` | **⚠ SUPERSEDED** — lm200-only; "CAPACITY" verdict is an lm200 artifact. Read `CAPACITY_PERREGIME.md`. |
| `CAPACITY_PERREGIME.md` | n=3. Per-regime + length sweep T∈{512,1024,2048}. **ARCHITECTURE** on clean / noise / length / arithmetic / calibration; lm200 is a content-channel effect. |
| `SINGLE_SIZE_CONTROL.md` | n=3. Confirms coupled-ω is the cross-scale small-grid bottleneck. |

## Cross-scale architecture investigation

| File | Status |
|---|---|
| `TEM_CROSSSCALE_DIAGNOSTIC.md` | Analytical — why TEM dominates small grids. |
| `PERSCALE_OMEGA_RESULTS.md` | n=3. Per-scale ω: +10pp at size 32 (partial fix). |
| `EM_HOPFIELD_CROSSSCALE.md` | n=3. EM backbone + Hopfield head cross-scale. |
| `HOPFIELD_NOMAINAP_RESULTS.md` | n=3. Position-modulated main attention IS load-bearing. |
| `LEVEL15EM_CROSSSCALE.md` | n=3. EM backbone worse than WM at every scale. |

## NoDrop / GSF (the dropout + multi-modal-Bayes findings)

| File | Status |
|---|---|
| `GSF_NODROP_RESULTS.md` | n=3. NoDrop & GSF are accuracy-substitutes, NLL-complements. |
| `NODROP_PARETO_RESULTS.md` | n=3. NoDrop Pareto-shift verification. |
| `DROPOUT_ABLATION_RESULTS.md` | Dropout-removal ablation. |
| `LEVEL15BETA_RESULTS.md` | β was a red herring; dropout was load-bearing. |

## Other regime tests

| File | Status |
|---|---|
| `LONGT_EVAL_RESULTS.md` | Length extrapolation T→2048. |
| `VOCAB_SWEEP_RESULTS.md` | Vocab scaling; paper's EM-scaling claim is l=16-specific. |
| `SPARSE_LANDMARKS_RESULTS.md` | Landmark-density sweep; TEM dominates sparse. |
| `STOCHASTIC_TRANSITION_RESULTS.md` | Action-noise ≡ stochastic-transition MDP. |
| `MINIGRID_DOORKEY_RESULTS.md`, `_LONGT.md`, `_ROPE_DIAG.md`, `_CACHED.md`, `MINIGRID_DK16_RESULTS.md` | MiniGrid-DoorKey. |
| `MINIGRID_MEMORY_RESULTS.md` | MemoryS13 — cleanest "wins on a real env" result. |

## TEM-specific

| File | Status |
|---|---|
| `TEM_T_MULTISEED.md` | n=3 TEM-T (transformer-formulation of TEM). |
| `TEM_T_RESULTS.md` | Single-seed TEM-T. **Stale TEMFaithful rows removed.** |
| `TEM_RESULTS.md` | **Stale TEMFaithful rows removed** (pre-bug-fix). TEM-GRU rows kept. |

## Behavioural / probes (mostly honest negatives)

| File | Status |
|---|---|
| `GOAL_DIRECTED_RESULTS.md` | Goal-directed match-acc. |
| `GOAL_CLOSEDLOOP_RESULTS.md` | Closed-loop 1-2% — honest negative (BC distribution shift). |
| `ACTIVE_INFERENCE_RESULTS.md` | Active-inference planning — null. |
| `SR_PRETRAIN_RESULTS.md`, `SR_PROBE_RESULTS.md` | Successor-rep aux pretraining — null. |
| `PROBE_GOAL_RESULTS.md`, `PROBE_GOAL_DISTANCE.md`, `STATE_PROBES.md` | Goal / distance / state probes. |
| `DAGGER_RESULTS.md`, `DAGGER_DK6_RESULTS.md`, `DAGGER_EMPTY_RESULTS.md` | DAgger on DoorKey. |
| `DOORKEY_BC_RESULTS.md` | DoorKey behavioural cloning. |
| `ZERO_SHOT_TRANSFER_clean.md`, `ZERO_SHOT_TRANSFER_lm200.md` | Zero-shot transfer (safe-init). |
| `LONG_SEQ_*.md`, `PER_VISIT_*.md`, `OMEGA_RESCALE_*.md` | Older eval splits. |
| `GOAL_TASKS_RESULTS.md`, `MODEOMEGA_FOLLOWUP_RESULTS.md`, `GSF_MODES_DIAGNOSTIC.md`, `GSF_FULL_RESULTS.md` | Misc. |
| `VECTOR_NAV_V2_RESULTS.md` | Vector-nav probe v2. |

## In-flight (will auto-commit when done)

- `run_tem_noise_and_ffn.sh` → `TEM_NOISE_FFN_RESULTS.md` — TEM on noise regime + TEMFaithful_FFN direct machinery test.

## Known open items

1. ~~`CAPACITY_CONTROL.md` inconclusive~~ — RESOLVED by `CAPACITY_PERREGIME.md` (architectural on clean / noise / length / arithmetic / calibration; lm200 is a content-channel effect).
2. GPU 1 occupancy is intermittent; per-regime sweep ran fine there sequentially. Pair-onto-GPU-1 OOMs remain a footgun if two processes share it.
3. `archive_stale/` holds 35 dead-end / incorrect / superseded files (DoG/hex dead-ends, PC v3/v4 saga, broken-init, single-seed superseded versions). Recoverable; not deleted.

*Generated 2026-05-21 during the dead-end cleanup.*
