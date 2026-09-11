## Project state and findings

- [Project state snapshot](project_state.md) — **read first.** Citable, retracted, in flight, open. Goal: factorisation-and-transfer.
- [EM vs WM mechanism](feedback_em_vs_wm_mechanism.md) — WM is NOT additive; EM's recency deficit is SEARCH, spread over 64 per-token wrapped rewinds.
- [Clock vs map: what cancellation chooses](project_clock_vs_map.md) — signed = map, monotone = clock; recency does NOT need a clock (rewind).
- [The sign of the phase increment](project_sign_axis.md) — a monotone clock cannot represent a −1 action. Prior art: Sarrof/Grazzi/SRoPE.
- [Rank, and Selective RoPE](project_rank_and_selective_rope.md) — use r=4 on MapWM (+0.085, 384 params); r=2 is SKEWED. Not for MapPoPE.
- [Hierarchy helps only if a summary is a sufficient statistic](project_hierarchy_negative.md) — negative on retrieval; compositional claim unpowered.
- [Looping beats the Kalman correction; refining theta is dead](project_loop_and_correction.md) — loop raises the FLOOR; Level15 is not inference.
- [Position effect: aliasing FALSIFIED; map-size THRESHOLD](project_miniworld_flip_negative.md) — threshold between 128 and 512 occupied cells.

## Reference

- [The three documents and the paper corpus](reference_review_documents.md) — review / results paper / record, split 2026-09-08.
- [Positional-encoding landscape and prior art](reference_positional_landscape.md) — GRAPE and Mamba-3 publish the taxonomy. Read before theory.
- [Paper corpus is stored locally](reference_paper_corpus.md) — 40 papers at `papers/`, read first-hand. Grep it, don't re-search.
- [Language numbers, theta without actions, PoPE](reference_language_and_pope.md) — enwik8 caveats; RoPE is Δ=1; PoPE changes magnitude.
- [Looped-transformer literature](reference_looped_transformer_lit.md) — Mixture-of-Recursions owns "recursion substitutes for depth".
- [Memory is shared via git](reference_shared_memory.md) — this dir mirrors to `.claude-memory/`; pull before reading, push after.

## Method — measurement

- [Existence before mechanism](feedback_existence_before_mechanism.md) — construct, then warm-start frozen AND trainable at training scale; gauges.
- [Verify convergence, noise floor and power FIRST](feedback_convergence_first.md) — MDE before any null; measure the floor; is acc just loss?
- [Validate the task before spending GPU](feedback_validate_task_first.md) — action-stream n-gram, context-destruction, chance. Gate first.
- [Check the recipe before believing a ceiling](feedback_recipe_before_architecture.md) — a task stuck at 0.415 was undertrained.
- [A borrowed benchmark usually doesn't test your axis](feedback_borrowed_benchmarks.md) — Flip-Flop, MQAR: nulls our data predicted.
- [Check the premise, prefer runtime knobs, split hypotheses](feedback_premise_before_test.md) — eval-only sweeps before training sweeps.
- [Probes lie confidently](feedback_probe_verification.md) — analysis bugs that printed clean wrong verdicts. Verify what a probe measures.
- [Verify before relaying](feedback_verify_before_relaying.md) — agents, summarisers and own probes return confident wrong answers.
- [The lm200 era is retracted](feedback_lm200_stuck_baselines.md) — a leaderboard ranked convergence, not architecture.

## Method — operations

- [Verify state before and after destructive commands](feedback_verify_before_destructive.md) — failed `git add`; rm of a COMPLETED batch.
- [Relative paths in `python3 -m` resolve to the PARENT dir](feedback_cwd_and_module_paths.md) — fails silently; use an absolute REPO.
- [Scheduler and measurement traps](feedback_scheduler_and_measurement_traps.md) — fill-first pickers; `pgrep -f` self-match; no script edits.
- [Run one seed of everything first](feedback_seed_ordering.md) — seed outer, variant inner: a full low-confidence table lands fastest.

## Context

- [User authoring style](user_style.md) — terseness, no emojis, honest reporting.
- [Action-noise framing](feedback_action_noise_framing.md) — stochastic-transition-MDP vocabulary; lead with non-circular wins.
- [PC-Kalman duality](feedback_pc_kalman_duality.md) — forward and inverse models are duals; descent finds the degenerate optimum.
- [Backfill standard-transformer baselines](feedback_baselines_backfill.md) — within-family tables need a RoPE column.
