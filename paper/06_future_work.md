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

We tested two predictions empirically (`hippocampal_analysis.py`,
`hippocampal_hidden_eval.py`) and both came back **falsified**:

- **Hexagonal grid cells.** Sargolini grid scores on hidden-state rate
  maps top out at 0.05–0.15 across all variants — well below the
  0.3 threshold considered grid-like.
- **R_t at landmark tokens.** Predicted ordering was
  landmark < aliased < blank (smaller R = more informative). Actual
  ordering for Level 1.5-WM is aliased < landmark < blank; for
  Level 1.5-EM it is blank < landmark < aliased. The basic
  blank-vs-non-blank distinction holds, but the predicted fine-grained
  ordering does not.

The third test (Stensola √2-spaced ω modules) holds approximately,
but mostly inherited from the geometric initialization rather than
discovered by training.

The diagnostic explanation for the absence of hexagonal cells is
architectural — see §6.11. Open directions:

- Compare R_t distributions to neural recordings from Sun et al. (2024,
  *Nature*) or Nieh et al. (2021, *Nature*) directly. Even if Level 1.5
  is not Bayesian-informativeness-optimal, the learned R_t may still
  match neural firing patterns under the right transformation.
- Train Level 1.5 with an explicit Bayesian-informativeness regulariser
  on R_t to test whether the predicted ordering can be recovered.
- Investigate place-cell-like single-peak patterns at the *attention*
  level rather than at hidden-state level — attention rows project
  position onto a sparse subset of cells, which may exhibit place-cell
  selectivity even when hidden states do not show grid structure.

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

## 6.11 MapFormer-Grid: An Architecture That Can Produce Hexagonal Cells

Our hippocampal-correspondence analysis (§6.5) found that current
MapFormer variants do not produce hexagonal grid-cell-like
representations. The reason is structural: MapFormer assigns one
path-integrator block per ω frequency, but the geometric optimum for
2D spatial tiling — three sinusoidal waves at the *same* frequency
oriented at 60° — requires multiple blocks per scale. With one wave
per ω, hexagonal interference patterns are mathematically inaccessible
regardless of how the model is trained.

**Proposed architectural change.** Replace the single block per ω
with a *module* of `n_orientations` blocks at the same ω with
orientations {0°, 60°, 120°} in 2D action space. Specifically:

- `action_to_lie` outputs a 2D vector `(Δ_x, Δ_y)` per module instead
  of a scalar per block
- For each block in module m at orientation θ_o:
  `Δ_{m,o} = cos(θ_o) · Δ_x + sin(θ_o) · Δ_y`
- The cumulative angle is then
  `θ_{m,o,t} = ω_m · cumsum(Δ_{m,o})`
- At the hidden-state level, `cos(θ_{m,0}) + cos(θ_{m,60}) + cos(θ_{m,120})`
  produces hexagonal interference at scale ω_m

Trade-offs:

- More parameters per module, fewer distinct ω values (e.g., 10 modules
  × 3 orientations = 30 blocks at 10 frequencies, vs current 32 blocks
  at 32 frequencies)
- Path integration remains parallelisable via cumsum; O(log T) preserved
- Level 1.5 InEKF correction extends naturally to the new state
- Action embedding requires a 2D learnable projection (small change)

**Predicted outcomes.**

1. *Hexagonal patterns emerge* with grid scores > 0.3 at multiple scales,
   matching the Sargolini et al. (2006) grid-cell signature
2. *Multi-scale modular organisation* matching the Stensola et al. (2012)
   √2-ratio spacing observed in entorhinal cortex
3. *Cognitive-map task performance preserved or improved*, since
   hexagonal codes are theoretically optimal for 2D position
   representation under bandwidth constraints (Mathis et al. 2012)

If validated, this would close the gap between MapFormer's mathematical
structure and biological grid-cell organisation, providing the first
transformer-based architecture that *necessarily* produces grid-cell
representations from first principles. If falsified — i.e., hexagonal
patterns don't emerge despite the structural enabling — that would be
a significant negative result indicating the gradient landscape itself
disfavours hexagonal solutions even when architecturally accessible.

**Estimated cost:** ~1–2 weeks (new model class, multi-seed training
on three configs, hippocampal-correspondence eval, paper writeup).
