---
name: Backfill standard-transformer baselines for paper submission
description: This session's headline results compare within the MapFormer family; for the cognitive-map narrative we need every meaningful table to also include RoPE / LSTM / MambaLike. Track which tables still need backfilling.
type: project
originSessionId: continued-2026-05-10
---

**Discovered 2026-05-10.** Many recent results tables compare only within the
MapFormer family (Vanilla / Level15 / Level15EM / Level15NoDrop / Level15GSF
/ TEMFaithful). They answer "which MapFormer variant is best." They do *not*
answer the paper's actual headline question: "does the cognitive-map inductive
bias matter at all, vs a standard transformer?"

**Tables that need a standard-transformer baseline (RoPE at minimum) before
paper submission:**

- `LEVEL15BETA_RESULTS.md` — has Vanilla, Level15, Level15Beta, TEMFaithful. Add RoPE.
- `VOCAB_SWEEP_RESULTS.md` — has Vanilla{,EM}, Level15{,EM}. Add RoPE at each vocab.
- `NODROP_PARETO_RESULTS.md` — has Vanilla, Level15, Level15NoDrop. Add RoPE.
- `DROPOUT_ABLATION_RESULTS.md` — has Vanilla, Level15, NoDrop, Beta, TEMFaithful. Add RoPE.
- `GSF_RESULTS.md` / `GSF_NODROP_RESULTS.md` — Level15 family + TEMFaithful only. Add RoPE.
- `GOAL_DIRECTED_RESULTS.md` / `GOAL_TASKS_RESULTS.md` — needs RoPE goal-directed trained.
- `DOORKEY_BC_RESULTS.md` — needs RoPE BC.
- `PROBE_GOAL_RESULTS.md` — needs RoPE frozen-probe.
- `DAGGER_RESULTS.md` — needs RoPE DAgger.

**Minimum-viable backfill set:**
- RoPE on all (lm200 checkpoints exist; just eval)
- LSTM, MambaLike on the headline lm200 table (existing checkpoints)
- VanillaEM, Level15EM where the EM vs WM comparison is interesting (DoorKey BC already has it)

**Already done in 2026-05-10 cognitive-tier:**
- `LONGT_EVAL_RESULTS.md` (queued): minimal set INCLUDING RoPE
- `SPARSE_LANDMARKS_RESULTS.md` (queued): minimal set INCLUDING RoPE
- `MULTIENV_RESULTS.md` (queued): minimal set INCLUDING RoPE

**Why this matters:** the cognitive-map claim is "standard transformers fail
on tasks requiring path integration; MapFormer's inductive bias is necessary;
Level15+ are improvements on top." Without RoPE in the table, reviewers will
correctly ask "but is the cognitive map even needed?" Backfill before submission.
