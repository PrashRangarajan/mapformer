---
name: feedback-prelaunch-audit
description: Audit the experimental design and code before any GPU launch; the user asks for it and it has repeatedly caught bugs that otherwise cost a retraction.
metadata:
  type: feedback
---

**Rule: before launching a batch, have an independent audit check the design and code** -- does the changed
flag change only what it claims (e.g. `--n-steps` also changes task composition), are tokens/steps/params
matched, are scored-position counts and floors known, does the eval reproduce a stored number exactly, do
the arms' training losses overlap.

**Why:** the Dyck width confound (1L arms d=64 vs 2L d=128), the frequency-ladder confound (index arms
base 10000 vs path arms from grid_size) and the occupancy-blind GPU picker were all design bugs caught
only after GPU time, each costing a retraction or a re-run. The rank matched-length audit (2026-09-23)
found, before any GPU: the old r=2/r=4 training losses never overlapped, 94% of the headline effect sat
in one stratum (short-gap revisits late in the sequence), and wrap revisits were below floor for both
arms. The user asks for this step repeatedly and for results to be verified before they are relayed.

**How to apply:** write the pre-registration, then run the audit (an agent is fine), record its verdict in
the prereg, then launch. An eval-only stratification of existing checkpoints is usually free and is the
fastest way to learn what the new batch can and cannot show. Related: [[feedback_validate_task_first]],
[[feedback_convergence_first]], [[project-robustness-vs-capability]].
