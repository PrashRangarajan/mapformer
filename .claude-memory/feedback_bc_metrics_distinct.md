---
name: BC eval metrics are distinct — don't conflate them
description: match-acc / closed-loop success / probe-acc each measure something different. Reporting only one and calling the result "goal-directed performance" is misleading. Document which metric is which in the paper, and report match-acc + closed-loop side by side when both apply.
type: feedback
originSessionId: continued-2026-05-10
---

**Discovered 2026-05-10.** Our DoorKey-8x8 BC results gave match-acc 0.81-0.94 but closed-loop success 0.19-0.25 — same models, same env, *radically* different "accuracy."

The three goal-directed metrics in our pipeline measure conceptually distinct things:

| Metric | What it measures | What it MASKS |
|---|---|---|
| **match-acc** | Open-loop: feed the expert trajectory, ask model what action it would pick. | Distribution shift. The model never has to recover from its own mistakes. |
| **closed-loop success** | Model picks action, env executes, model picks next. Reach goal or not. | Per-step quality. A 25% success rate could come from 87% per-step acc compounding badly, OR from 50% per-step acc with no recovery. |
| **linear-probe acc** | Frozen backbone, single linear head, predict expert action from hidden state at each pos. | Trainability effects — measures REPRESENTATION CONTENT, not what the model would do. |

**Pitfalls observed:**

- *DoorKey closed-loop success ~0.20 across all variants:* easy to misread as "the cognitive map doesn't help." Actually it's the BC ceiling — distribution shift dominates everything. Use match-acc OR DAgger here, not raw closed-loop.
- *Frozen probe vs goal-directed BC giving different answers:* the BC result has Level15-WM and Level15-EM tied (both 0.95 on goal-directed). The frozen probe also has them tied (both 0.63 held-out). But on DoorKey BC, Level15-EM 0.94 > Level15-WM 0.88. The contradiction is only apparent — the EM advantage shows up on egocentric obs (where A_X is noisy), not on full-obs torus. Each result is true in its regime.
- *match-acc + closed-loop on the same row*: the right way to present BC results. match-acc tells you about per-step decision quality; closed-loop tells you about recovery. Both matter.

**For the paper, frame as:**
- "Cognitive maps differ in goal-directed CONTENT" → use frozen probe.
- "Models can be TAUGHT to use them for actions" → use match-acc (open-loop BC).
- "Models can ACT goal-directedly in closed loop" → use DAgger success rate (or RL).

Never collapse these into one number called "goal-directed performance."

**Concrete recommendation:** in any table or figure, label which metric you're showing. If a paper says "Method X is 24% better at goal-directed navigation" without specifying match-acc vs closed-loop vs frozen-probe vs DAgger-trained-success-rate, ask for clarification.
