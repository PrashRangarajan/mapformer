Rules and conventions live in CLAUDE.md (rules 1-29); these files hold the why and the detail.

## State and findings
- [Project state](project_state.md) -- **read first.** Live state only: running (nothing, 2026-10-03), last results, open decisions with costs.
- `docs/WHERE_THINGS_STAND.md` -- **read second.** One-page orientation: thesis (most effects are robustness, not
  capability), what survives with numbers and status (registered / post hoc / pilot), what is open ranked with costs,
  what is stale. Week of 2026-09-27..10-03 in `docs/SESSION_2026-09-27_to_10-03.md`; 2026-10-04..09 in `docs/SESSION_2026-10-04_to_10-09.md`.
- [Robustness is not capability](project_robustness_vs_capability.md) -- matched length AND depth decide; OOD-only effects are robustness until controlled; sign is the one that survived its control.
- [Rank, and Selective RoPE](project_rank_and_selective_rope.md) -- per-head rank decides it: rank 2 per head 0-2/8, rank 3 6/8, rank 4 8/8; 2x budget does not rescue rank 2; rank = D hard in 3D too, D+1 fine on large tori; sharing and scale unmeasured.
- [PoPE/MapFormer asymmetry](project_mappope_asymmetry.md) -- PoPE's encoding helps the path row (Bach); "path integration hurts PoPE" is unmeasured (corrected 2026-10-03).
- [EM vs WM mechanism](feedback_em_vs_wm_mechanism.md) -- WM is not additive; EM's recency deficit is search.
- [Clock vs map](project_clock_vs_map.md) -- signed = map, monotone = clock; the PoPE-decoupling corollary is withdrawn.
- [Sign of the phase increment](project_sign_axis.md) -- monotone cannot represent a -1 action; the cost survives matched length (2026-09-28); a replication.
- [Loop and correction](project_loop_and_correction.md) -- loop main effect survives; Level 1.5 is not inference; PC and Kalman are duals.
- [Hierarchy](project_hierarchy_negative.md) -- helps only if a summary is a sufficient statistic; compositional claim unpowered.
- [Map-size threshold](project_miniworld_flip_negative.md) -- aliasing falsified; threshold between 128 and 512 occupied cells.
- [What/where, language, leak](project_what_where_and_language.md) -- text world (reg), context step (pilots), what/where (post hoc), leak remedies (reg) and NormStep; what is prior art (docs/lit/).

## Reference
- [Documents, shared report, corpus](reference_review_documents.md) -- review / results paper / record; report link (v10); 40 papers.
- [Prior art](reference_positional_landscape.md) -- GRAPE and Mamba-3 publish the taxonomy. Read before theory.
- [Language and PoPE](reference_language_and_pope.md) -- enwik8 caveats; RoPE is Delta=1; PoPE changes magnitude.
- [Looped transformers](reference_looped_transformer_lit.md) -- Mixture-of-Recursions owns "recursion substitutes for depth".
- [Memory is this directory](reference_shared_memory.md) -- the memory path is a symlink to `.claude-memory/`; pull before, push after.

## Method (detail behind CLAUDE.md rules)
- [Convergence, floor, power, recipe](feedback_convergence_first.md) -- rules 1-5, 12; lm200 and the 0.415 ceiling.
- [Validate the task; audit the design](feedback_validate_task_first.md) -- rules 11-13; premise, runtime knobs, split hypotheses.
- [GPU sharing and cost](feedback_gpu_sharing_and_cost.md) -- shared server; no launches while another user's jobs run; show cost, offer trimmed variants.
- [Batch workflow](feedback_batch_workflow.md) -- rule 29: verification agent per batch, amend before reading, pilots on outside seeds, show cost arithmetic.
- [Existence before mechanism](feedback_existence_before_mechanism.md) -- rules 14-15; warm-start frozen and trainable; gauges.
- [Probes, agents and summaries lie confidently](feedback_probe_verification.md) -- rule 9; verify before relaying.
- [Borrowed benchmarks](feedback_borrowed_benchmarks.md) -- Flip-Flop, MQAR: nulls our data predicted.
- [Scheduler, destructive-command and path traps](feedback_scheduler_and_measurement_traps.md) -- rules 20-27; orphaned data workers.
- [Seed ordering](feedback_seed_ordering.md) -- seed outer, variant inner.
- [Backfill standard baselines](feedback_baselines_backfill.md) -- within-family tables need a RoPE column.

## Context
- [User style](user_style.md) -- terse, no emojis, honest, no Co-Authored-By; reports lead with positives.
- [Action-noise framing](feedback_action_noise_framing.md) -- say stochastic-transition MDP; vocabulary only, no performance claim.
