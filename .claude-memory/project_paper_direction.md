---
name: project-paper-direction
description: The paper framing chosen 2026-10-11..13 -- Howard's integrator cases as the rotary-position families; learnability (not representation) separates signed from non-negative rotary steps; depth cannot buy the map on the torus. Status of each claim and what a venue needs.
metadata:
  type: project
---
**Framing (user's choice 2026-10-11: "map vs clock").** Howard et al. 2014 (J Neurosci 34:4692): one integrator with a
constant rate (time), a non-negative speed (path length) or a signed velocity (position). Rotary families: index RoPE =
Case I, monotone content-dependent phase (CoPE / CARoPE / GRAPE-AP) = Case II, signed (MapFormer, Selective RoPE) = Case
III. Targeted search found this mapping not stated anywhere (`docs/lit/LIT_HOWARD_TRANSFORMER.md`); the dichotomy itself
and "signed steps are needed" are prior art (`docs/lit/LIT_CLOCK_MAP.md`).
**The twist (2026-10-12, RING_DIAL audit M1, verified):** in the ROTARY slot Cases II and III coincide representationally
(angles wrap mod 2 pi, so a non-negative step can reproduce any signed code); they differ in LEARNABILITY. Evidence: ring
(RING_DIAL 2/8, 1/8 vs 8/8; RING_BUDGET 3x budget no help; never finds the wrap solution), torus wrap probe (cancelling
channels Signed 0.98, Abs 0.11), ring loss-AUC companion (signed faster 8/8 at every p). The clock side (non-negative
faster on content counting) is UNMEASURED twice (CLOCK_SPEED p 0.053, REP p 0.19): asymmetric, no double dissociation.
**Depth cannot buy the map on the torus** (FACTORIAL, n = 4): RoPE 1L 0.72, 4L 0.86, looped 0.86 vs path 1L 1.00.
**Natural text:** TinyStories / code phases are pure clocks (alpha ~1.0), no gain over RoPE at our scale (CoPE gains at scale).
**Venue (assessed 2026-10-13):** workshop paper: enough now; TMLR: add torus baselines (LSTM, Mamba-like, CoPE, Selective
RoPE), factorial n = 8, wrap existence test, one scale check (~2 GPU days); main conference: not with this material.
Claim-by-claim status: `docs/paper/CLAIM_MAP_clock_map.md`. Related: [[project-sign-axis]], [[project-clock-vs-map]],
[[feedback-follow-on-drift]].
