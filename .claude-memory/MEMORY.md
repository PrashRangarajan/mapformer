Rules and conventions live in CLAUDE.md (rules 1-30); these files hold the why and the detail. Retired files: `archive/` (README there).

## State and findings
- [Project state](project_state.md) -- **read first.** Live state only: running (nothing, 2026-10-28), last results, open decisions with costs.
- `docs/WHERE_THINGS_STAND.md` -- **read second.** One-page orientation (rewritten 2026-10-29): thesis (one path layer
  learns the long-walk map; sign and per-head rank decide whether training finds it), what survives with numbers and
  status, what is open ranked, what is stale, the documents (drafts 2 and 6). Week of 2026-09-27..10-03 in `docs/SESSION_2026-09-27_to_10-03.md`; 2026-10-04..09 in `docs/SESSION_2026-10-04_to_10-09.md`.
- [Robustness is not capability](project_robustness_vs_capability.md) -- matched length AND depth decide; OOD-only effects are robustness until controlled; sign is the one that survived its control.
- [Rank, and Selective RoPE](project_rank_and_selective_rope.md) -- per-head rank decides it: rank 2 per head 0-2/8, rank 3 6/8, rank 4 8/8 (with decay 0.05); the rank-2 obstacle is the clock (RANKCLOCK); 2x budget and three recipes do not rescue rank 2; not hard in 1D; faithful Selective RoPE 0/8 (update block 2026-10-29).
- [PoPE/MapFormer asymmetry](project_mappope_asymmetry.md) -- PoPE's encoding helps the path row (Bach); "path integration hurts PoPE" is unmeasured; score rule carries MapPoPE's gain, peak pinning is the mechanism (corrected 2026-10-03, 2026-10-29).
- [EM vs WM mechanism](feedback_em_vs_wm_mechanism.md) -- WM is not additive; EM's recency deficit is search.
- [Paper direction](project_paper_direction.md) -- Howard's three integrator cases = the rotary families (a framing); in the rotary slot sign is learnability, not representation; venue gaps of 10-13 since run; framing now drafts 2 and 6.
- [Clock vs map](project_clock_vs_map.md) -- the 2026-09 account: cancellation makes a map, drift a clock; corrected 2026-10-29 (non-negative rotary steps can hold a map; the rank-2 obstacle is a clock); the PoPE-decoupling corollary is withdrawn.
- [Sign of the phase increment](project_sign_axis.md) -- a learnability cost of |.| and softplus increments (not representation): matched length, best rates, ring and torus, attention and an SSM; history to 2026-10-13 with a 2026-10-29 summary of the later batches.
- [Loop and correction](project_loop_and_correction.md) -- loop main effect survives (it matches 3 real layers, not beats); Level 1.5 is not inference; PC and Kalman are duals.
- [Hierarchy](project_hierarchy_negative.md) -- helps only if a summary is a sufficient statistic; compositional claim unpowered; its lm200-config numbers are not citable.
- [Map-size threshold](project_miniworld_flip_negative.md) -- aliasing falsified; threshold between 128 and 512 occupied cells.
- [What/where, language, leak](project_what_where_and_language.md) -- text world (reg), context step (pilots), what/where (post hoc), leak remedies (reg), NormStep, state changes (location + recency, not semantics); what is prior art (docs/lit/).

## Reference
- [Documents, shared report, corpus](reference_review_documents.md) -- drafts 2 and 6 (current) and the stale September documents; report link (v22); 44 + 28 papers.
- [Prior art](reference_positional_landscape.md) -- GRAPE and Mamba-3 publish the taxonomy; its rank paragraph corrected 2026-10-29. Read before theory.
- [Language and PoPE](reference_language_and_pope.md) -- enwik8 caveats; RoPE is Delta=1; PoPE changes magnitude.
- [Looped transformers](reference_looped_transformer_lit.md) -- Mixture-of-Recursions owns "recursion substitutes for depth".
- [Memory is this directory](reference_shared_memory.md) -- the memory path is a symlink to `.claude-memory/`; pull before, push after.

## Method (detail behind CLAUDE.md rules)
- [Convergence, floor, power, recipe](feedback_convergence_first.md) -- rules 1-5, 12; lm200 and the 0.415 ceiling.
- [Validate the task; audit the design](feedback_validate_task_first.md) -- rules 11-13; premise, runtime knobs, split hypotheses.
- [GPU sharing and cost](feedback_gpu_sharing_and_cost.md) -- shared server; no launches while another user's jobs run; show cost, offer trimmed variants.
- [Compute: kalman and Hyak](feedback_compute_kalman_only.md) -- 2026-10-26: GPU work on kalman and Hyak only, not koopman.
- [Follow-on drift](feedback_follow_on_drift.md) -- 2026-10-11 review: chains of post hoc follow-on batches, promoted secondaries; tie each GPU job to the plan and the main claim.
- [Batch workflow](feedback_batch_workflow.md) -- rule 29: verification agent per batch, amend before reading, pilots on outside seeds, show cost arithmetic.
- [Existence before mechanism](feedback_existence_before_mechanism.md) -- rules 14-15; warm-start frozen and trainable; gauges.
- [Probes, agents and summaries lie confidently](feedback_probe_verification.md) -- rule 9; verify before relaying.
- [Borrowed benchmarks](feedback_borrowed_benchmarks.md) -- Flip-Flop, MQAR: nulls our data predicted.
- [Scheduler, destructive-command and path traps](feedback_scheduler_and_measurement_traps.md) -- rules 20-27; orphaned data workers.
- [Seed ordering](feedback_seed_ordering.md) -- seed outer, variant inner.

## Context
- [User style](user_style.md) -- terse, no emojis, honest, no Co-Authored-By; reports lead with positives.
- [Action-noise framing](feedback_action_noise_framing.md) -- say stochastic-transition MDP; vocabulary only, no performance claim.
