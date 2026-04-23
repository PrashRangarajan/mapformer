# 6. Future Work

This paper focuses on a single correction mechanism (heteroscedastic InEKF)
in a single Lie group (SO(2)) on a single task (MapFormer's 2D torus
navigation). Several natural extensions sit at the intersection of classical
robotics and modern deep learning, and appear underexplored in the current
literature.

## 6.1 Higher-Dimensional Lie Groups

MapFormer's SO(2) state is a natural starting point but limits applications
to planar navigation. Extensions:

- **SE(2) (planar pose = position + heading).** Direct generalization with
  no conceptual overhead; adds 2 translation components to the state. A
  natural fit for 2D robot navigation tasks (e.g., BabyAI, MiniGrid, ViZDoom).
- **SO(3) and SE(3) (3D orientation and full 6-DoF pose).** Requires the
  3D version of Barrau-Bonnabel's invariant EKF (well-understood in
  robotics). Would enable cognitive-map learning for drones, 3D
  navigation, and first-person VR/AR.
- **Sim(3) (pose with scale).** For monocular SLAM-style tasks where scale
  is uncertain. Used in ORB-SLAM, DROID-SLAM.

Our parallel-scan machinery extends directly: Möbius matrices for SO(2)
generalize to Adjoint-weighted recurrences for higher-dimensional groups
(following standard InEKF derivations; see Solà et al. 2018 for the
bookkeeping).

## 6.2 IMU-Preintegration-Style Context Compression

Forster et al.'s IMU preintegration compresses 40 raw IMU samples into one
preintegrated SE(3) measurement for SLAM. The analogous idea for
long-context transformers: compress a chunk of action/observation tokens
into a single "preintegrated" position update carrying a covariance.

Practically: replace the `cumsum(ω·Δ)` over all tokens with a
multi-resolution representation that commits past chunks into a single
summary and keeps only recent tokens at full resolution. Combined with
Level 1.5's bounded-error property, this could extend MapFormer's
effective context dramatically without per-token attention.

We are unaware of any existing paper taking this direction.

## 6.3 Equivariant Attention on Lie Groups

RoPE is SO(2)-equivariant by construction. Level 1.5 generalizes via
input-dependent rotations. But neither is SE(n)-equivariant — they don't
correctly handle the coupling of translation and rotation. A natural
extension: SE(n)-equivariant attention where Q/K rotate AND translate
under the group action.

Recent work on equivariant transformers (Finzi et al. 2021; Fuchs et al.
2020; Equiformer) provides the machinery. Combining it with input-dependent
actions (MapFormer's core innovation) appears unexplored.

## 6.4 Differentiable Pose-Graph SLAM Inside Neural Networks

Theseus-AI (Pineda et al. 2022) provides a differentiable SE(3) bundle
adjustment module for PyTorch. An interesting direction: MapFormer's
internal θ̂ state could be treated as pose estimates and refined via a
Theseus-style optimization step at each layer or at evaluation time. This
converts the "cognitive map" into a literal pose-graph SLAM backend with
learned perception — potentially closing the gap between learned
transformer reasoning and classical geometric consistency.

## 6.5 Connection to Biological Neural Data

We hypothesize Level 1.5's learned R_t distribution should match the
uncertainty encoding observed in hippocampal pyramidal cells and grid
cells during exploration. Specifically:

- R_t should be small (high K_t) at distinctive landmarks — matching
  "boundary cell" or "object cell" activations
- R_t should be large (low K_t) in featureless space — matching the
  broader place-cell widths observed in open environments

Recently-published neural-recording datasets (Sun et al. 2024, Nature;
Nieh et al. 2021, Nature) provide suitable data. A direct comparison
would elevate this work from ML methodology to computational
neuroscience contribution.

## 6.6 Multi-Task / Instruction-Conditioned Navigation

Current MapFormer evaluates "predict the next observation at revisits."
Real cognitive-map tasks involve goal-directed behavior: "navigate to the
red door." This requires:

- Action prediction rather than observation prediction
- Instruction conditioning (language or goal-state input)
- Imitation learning or RL objectives

BabyAI (Chevalier-Boisvert et al. 2019) provides a well-designed suite
of such tasks. Extending Level 1.5 to BabyAI would validate the
architecture on tasks with richer observation structure and genuine
generalization demands.

## 6.7 Scaling Study

The MapFormer paper uses 1 layer, 2 heads, d=128 — very small. We replicate
this scale but suspect the Level 1.5 advantages scale favorably:

- More layers → more opportunities for θ̂ correction (see our multi-layer
  experiments, Table X)
- More heads → per-head uncertainty could capture multimodal hypotheses
- Larger d → more expressive R_t head

A systematic scaling study at 2/4/8 layers, 4/8 heads, d=256/512/1024 would
establish whether the architectural advantage persists at scale or shrinks.
This is on our explicit roadmap.

## 6.8 Real-World Robotics Deployment

The Kalman regime of Level 1.5 — bounded error, calibrated uncertainty,
sharp landmark handling — is precisely what real robotic navigation needs.
We see a clear path from the current work to practical impact:

1. SE(3) generalization of Level 1.5 (Section 6.1)
2. BabyAI or MiniGrid validation (Section 6.6)
3. Sim-to-real on small mobile robots (e.g., Clearpath Jackal)
4. Integration with existing SLAM stacks via the Theseus-AI differentiable-
   optimization bridge

This would close the loop on the "robotics Lie theory → deep learning"
transfer we advocate, with empirical validation on physical platforms.

## 6.9 Comparison to Other Learned-Filter Frameworks

KalmanNet (Revach et al. 2022) learns the Kalman gain while keeping the
rest classical. Level 1.5 learns both the gain (via R_t) and the
measurement model (via the z-head) but retains the Lie-group structure.
Understanding empirically where each framework wins (and for which tasks)
is a natural comparison study.

Other frameworks for comparison:
- **Particle filters** with learned proposals (Karkus et al. 2018)
- **Variational Bayesian filters** (Krishnan et al. 2017)
- **Deep Kalman Filters** (Krishnan et al. 2015)

## 6.10 Unification with SSMs

Level 1.5's scan machinery is algorithmically identical to Mamba's
selective-scan. This suggests deep connections between:

- State-space models (Mamba, S4) — generic recurrence
- Classical filtering (Kalman, EKF, UKF) — optimal estimation under Gaussian
  assumptions
- Equivariant architectures — Lie-group symmetry

A unified theoretical framework covering all three would likely yield new
architectures combining the strengths of each. We note that "Mamba as a
Linear Kalman Filter" (Wang et al. 2025) has made preliminary connections;
extending this to the Lie-group setting with explicit invariance
guarantees is an open direction.
