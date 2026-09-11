## Project state and findings

- [Project state snapshot](project_state.md) — **read first.** What is citable, what is retracted, what is open and ranked. The research goal is factorisation-and-transfer, not encoding for its own sake.
- [Clock vs map: what cancellation chooses](project_clock_vs_map.md) — signed = net displacement (a map), monotone = elapsed path (a clock); mutually exclusive, each correct for one job. Tested by crossover. α is a re-description of opposition, not a third cause.
- [The sign of the phase increment](project_sign_axis.md) — a monotone clock cannot represent a −1 action. Prior art is Sarrof/Grazzi/Selective RoPE; navigation and the isolation are ours.
- [Rank, and Selective RoPE](project_rank_and_selective_rope.md) — use r=4 on the MapWM family (+0.085 for 384 params). r=2's code is SKEWED, not too small. Does NOT transfer to MapPoPE.
- [Hierarchy helps only if a summary is a sufficient statistic](project_hierarchy_negative.md) — negative on retrieval and aggregation; the compositional claim survives a recipe re-measurement in size but is unpowered at n=8.
- [Looping beats the Kalman correction; refining theta is dead](project_loop_and_correction.md) — the loop raises the FLOOR, and much of its win is convergence. Level15 = bounded state + token-type gate, not inference.
- [Position effect: aliasing FALSIFIED; map-size THRESHOLD](project_miniworld_flip_negative.md) — less aliasing gives a LARGER effect; the real axis is a threshold between 128 and 512 occupied cells.

## Reference

- [The three documents and the paper corpus](reference_review_documents.md) — 21pp arena review / 17pp results paper / 38pp record, split 2026-09-08. Rules the documents are held to.
- [Positional-encoding landscape and prior art](reference_positional_landscape.md) — the log-polar frame, and that GRAPE and Mamba-3 already publish the taxonomy and the content-dependent rotation. Read before claiming theory.
- [Paper corpus is stored locally](reference_paper_corpus.md) — 40 papers at `papers/`, all read first-hand. Grep it, don't re-search the web.
- [Language numbers, theta without actions, what PoPE actually does](reference_language_and_pope.md) — enwik8 with power caveats; RoPE is the Δ=1 special case; PoPE changes magnitude, not angle.
- [Looped-transformer literature](reference_looped_transformer_lit.md) — Mixture-of-Recursions already owns "recursion substitutes for depth".
- [Memory is shared via git](reference_shared_memory.md) — this dir mirrors to `.claude-memory/`; pull before reading, push after writing.

## Method — measurement

- [Verify convergence, noise floor and power FIRST](feedback_convergence_first.md) — four retractions in a week, one root cause. MDE before calling anything a null; measure the floor; check whether accuracy is just the loss.
- [Validate the task before spending GPU](feedback_validate_task_first.md) — n-gram on the action stream, context-destruction, measured chance rate. Gate BEFORE training.
- [Check the recipe before believing a ceiling](feedback_recipe_before_architecture.md) — a task stuck at 0.415 for months was undertrained; the recipe was worth more than any architecture effect on it.
- [A borrowed benchmark usually doesn't test your axis](feedback_borrowed_benchmarks.md) — Flip-Flop and MQAR both returned nulls our own data predicted. Ask what it discriminates, and for whom, before running it.
- [Check the premise, prefer runtime knobs, split hypotheses](feedback_premise_before_test.md) — 16 runs replicating a known negative; a 90-second eval-only sweep replaced a 12-run training sweep.
- [Probes lie confidently](feedback_probe_verification.md) — five analysis bugs that each printed a clean wrong verdict. Verify what a probe measures, not just that it ran.
- [Verify before relaying](feedback_verify_before_relaying.md) — agents, web summarisers and my own probes all returned confident wrong answers. Run the check yourself.
- [The lm200 era is retracted](feedback_lm200_stuck_baselines.md) — a whole leaderboard ranked convergence, not architecture. Four derived findings died with it.

## Method — operations

- [Verify state before and after destructive commands](feedback_verify_before_destructive.md) — a failed `git add` stages nothing; a `rm -rf` destroyed a COMPLETED batch. Both times the terminal had already said otherwise.
- [Relative paths in `python3 -m` resolve to the PARENT dir](feedback_cwd_and_module_paths.md) — fails silently, four debugging rounds. Use an absolute REPO constant inside modules.
- [Scheduler and measurement traps](feedback_scheduler_and_measurement_traps.md) — fill-first GPU pickers idle a device; `pgrep -f` matches your own shell; never edit a running script.
- [Run one seed of everything first](feedback_seed_ordering.md) — seed outer, variant inner: a full low-confidence table lands fastest.

## Context

- [User authoring style](user_style.md) — terseness, no emojis, honest reporting.
- [Action-noise framing](feedback_action_noise_framing.md) — use stochastic-transition-MDP vocabulary; lead with non-circular wins.
- [PC-Kalman duality](feedback_pc_kalman_duality.md) — forward and inverse models are duals, not complements; gradient descent finds the degenerate joint optimum.
- [EM vs WM mechanism](feedback_em_vs_wm_mechanism.md) — CORRECTED: MapWM is not additive (per-pair rotated-content kernel); EM's recency deficit is learnability, not expressivity.
- [Backfill standard-transformer baselines](feedback_baselines_backfill.md) — within-family tables need a RoPE column before submission.
