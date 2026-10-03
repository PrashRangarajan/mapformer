---
name: feedback-batch-workflow
description: How the user wants every GPU batch run: independent code-verification agent, amendments before results, honest cost arithmetic, pilots on seeds outside the batch, push when asked.
metadata:
  type: feedback
---

For every new batch the user asks "have an agent verify the code is right" -- launch a read-only verification agent
(blind to the batch's results) as soon as a batch is launched, and write its findings into a pre-registration
AMENDMENT before any result is read. These audits repeatedly found real problems: branch sets missing the likeliest
outcome, accuracy contrasts firing on 1% gaps between solved cells, floors far above chance (retrace predictors),
tests that cannot fail (iid "new objects"), probe errors (key-step phase).

**Why:** two of my own design predictions were wrong in a row (context step), and three audits caught issues that
would have mis-stated registered verdicts. **How to apply:**
- Pilot on seeds OUTSIDE the registered batch (e.g. seed 100); never claim "pilot not reused" without checking --
  textworld and cancel pilots were byte-identical to batch seeds 0-1 (fresh-seed numbers had to be added).
- Before launching, show the measured cost arithmetic (s/epoch at the planned concurrency x epochs x runs /
  concurrent slots); "1-1.5 days" surprised the user. Measure, do not guess; ~8-10 s/epoch at 8 concurrent jobs.
- Rerun an agent's analysis script yourself before relaying its numbers (byte-identical output), and commit the
  script + output (rule 7). Swap tests and probes run inline once had no committed script.
- Ask before long or cost-heavy runs only if the value is doubtful; the user stopped a 31 h batch once the window
  hypothesis turned out near-guaranteed by construction. State what each experiment can and cannot show.
- Push when the user says "push"; commits are single-author (no Co-Authored-By).
