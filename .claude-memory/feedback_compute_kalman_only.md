---
name: feedback-compute-kalman-only
description: "As of 2026-10-26, run GPU work on kalman and Hyak (klone) only -- not koopman."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 224d7c23-b570-4513-8742-cf8a41acc4a3
  modified: 2026-10-26T08:29:26.438Z
---

On 2026-10-26 the user first said "only stick to kalman". Minutes later they corrected it to "kalman and hyak, not
koopman".

**Why:** not stated. Koopman ran G-3 and RANKRECIPE earlier that week. Hyak was set up for Memory Maze that week
and passed feasibility: EGL rendering works and the venv is installed. Cluster lessons are in `docs/HYAK_NOTES.md`.

**How to apply:**
- Plan new GPU batches for kalman's two GPUs and for Hyak only. Don't propose or launch anything on koopman.
- Small batches stay on kalman: no porting, and the same torch 2.10 as every earlier batch.
- Hyak is for work that needs its storage or parallelism, such as Memory Maze and large sweeps. A Hyak batch is
  self-contained (torch 2.13) and is never pooled with kalman runs.

See [[feedback-gpu-sharing-and-cost]] for kalman's sharing rules.
