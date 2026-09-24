Standing rules and conventions live in CLAUDE.md; these files hold the why and the detail.

## State and findings
- [Project state](project_state.md) -- **read first.** Live state only: running, next, last results.
- [Robustness is not capability](project_robustness_vs_capability.md) -- only matched-length results survive; train at target length or add the decay envelope.
- [PoPE/MapFormer asymmetry](project_mappope_asymmetry.md) -- PoPE's encoding helps the path row; path integration hurts PoPE on clock tasks.
- [Rank, and Selective RoPE](project_rank_and_selective_rope.md) -- r=4 +0.085 is OOD-only; matched-length batch ran (r=4 8/8 vs r=2 0/8 solved), verdict unreadable. MapWM only.
- [EM vs WM mechanism](feedback_em_vs_wm_mechanism.md) -- WM is not additive; EM's recency deficit is search.
- [Clock vs map](project_clock_vs_map.md) -- signed = map, monotone = clock; the PoPE-decoupling corollary is withdrawn.
- [Sign of the phase increment](project_sign_axis.md) -- a monotone clock cannot represent a -1 action (replication).
- [Loop and correction](project_loop_and_correction.md) -- loop raises the floor; Level 1.5 is stabilisation; PC and Kalman are duals.
- [Hierarchy](project_hierarchy_negative.md) -- helps only if a summary is a sufficient statistic; compositional claim unpowered.
- [Map-size threshold](project_miniworld_flip_negative.md) -- aliasing falsified; threshold between 128 and 512 occupied cells.

## Reference
- [Documents, report and corpus](reference_review_documents.md) -- review / results paper / record / shared report link; 40 papers in `papers/`.
- [Prior art](reference_positional_landscape.md) -- GRAPE and Mamba-3 publish the taxonomy. Read before theory.
- [Language and PoPE](reference_language_and_pope.md) -- enwik8 caveats; RoPE is Delta=1; PoPE changes magnitude.
- [Looped transformers](reference_looped_transformer_lit.md) -- Mixture-of-Recursions owns "recursion substitutes for depth".
- [Memory is this directory](reference_shared_memory.md) -- `~/.claude/projects/.../memory` is a symlink to `.claude-memory/`; pull before, push after.

## Method (detail behind CLAUDE.md rules)
- [Convergence, floor, power, recipe](feedback_convergence_first.md) -- rules 1-5; a task stuck at 0.415 was undertrained.
- [Validate the task; audit the design](feedback_validate_task_first.md) -- rules 11-13; pre-launch audits.
- [Existence before mechanism](feedback_existence_before_mechanism.md) -- rules 14-15; gauges.
- [Probes, agents and summaries lie confidently](feedback_probe_verification.md) -- rule 9; verify before relaying.
- [Borrowed benchmarks](feedback_borrowed_benchmarks.md) -- Flip-Flop, MQAR: nulls our data predicted.
- [Scheduler, destructive-command and path traps](feedback_scheduler_and_measurement_traps.md) -- rules 20-27.
- [Seed ordering](feedback_seed_ordering.md) -- seed outer, variant inner.
- [Backfill standard baselines](feedback_baselines_backfill.md) -- within-family tables need a RoPE column.

## Context
- [User style](user_style.md) -- terse, no emojis, honest, no Co-Authored-By; reports lead with positives.
- [Action-noise framing](feedback_action_noise_framing.md) -- stochastic-transition-MDP vocabulary only; its "+10pp is a lower bound" claim is contradicted by MQ_NOISE_2X2.
