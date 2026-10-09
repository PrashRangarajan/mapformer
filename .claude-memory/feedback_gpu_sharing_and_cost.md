---
name: feedback-gpu-sharing-and-cost
description: The user is cost-sensitive about GPU time and does not want our jobs on the shared server's GPUs while another user's jobs are running; check owners and ask before launching.
metadata:
  type: feedback
---

The server (kalman, 2x RTX 4090) is shared with another user (vsathish). On 2026-10-08 the user said "if another job
is running, don't run stuff yet, but fix all the code", and later "a day long is a lot" about a ~1.5-day queue.

**Why:** the user does not want to slow the other user down, and judges experiments by value per GPU-hour.
**How to apply:**
- Before any launch, check `nvidia-smi --query-compute-apps=pid` owners; if another user's jobs are substantial, build and
  validate CPU-only and ask before launching.
- Show measured cost (s/epoch at the planned concurrency) and offer trimmed variants (fewer seeds or cells) with their
  power cost; recommend one. Postpone low-urgency mechanism tests rather than queueing a day of GPU.
- Queue follow-ups behind running batches with a comm-matched waiter (rule 23), never auto-launch a batch whose pilot
  has not been reviewed and amended.
Related: [[feedback-batch-workflow]], [[project-state]].
