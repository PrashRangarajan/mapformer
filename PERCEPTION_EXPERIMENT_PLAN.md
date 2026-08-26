# Future experiment: perception-in-the-loop cognitive maps (MiniWorld pixels -> Habitat)

Status: PLANNED (not started). Written 2026-08-25. Prereq: the current fresh-map
allocentric factorial (geometry/action axis, synthetic location tokens) should
land first — this plan only makes sense if that establishes the geometry result.

## Why this experiment exists

Every MiniWorld experiment so far DISCARDS the rendered image and uses a
synthetic location-determined token (fixed random token per discretized cell),
exactly like the torus / TEM / CSCG / MapFormer paper. That is the correct,
controlled choice for isolating PATH INTEGRATION (does allocentric recoding
rescue the path-integrated arm through continuous-rotation geometry?). But it
means we test spatial reasoning, NOT perception. This plan closes that gap:
can the model build the cognitive map from ACTUAL PIXELS, not a handed-in
location token?

This is the real bridge to Habitat, whose entire point is photorealistic
perception. A positive geometry result does not transfer to Habitat until we
show the map can be built from a learned visual front-end.

## The scoping boundary being crossed (and its cost)

- Current regime: perception offloaded -> ~600K-3M param transformer suffices
  (research 2026-08-25: MapFormer 600K, TEM, CSCG, Trajectory-Transformer all
  work only because inputs are pre-tokenized symbols).
- Perception regime: the transformer must also parse images -> the proven band
  is 8M-100M params WITH A SEPARATE VISUAL ENCODER (IRIS 8M+VQ-VAE, ENTL
  96M+VQ-GAN@256 tokens/frame, SMT+ResNet18). Do NOT ask a small transformer to
  learn dynamics AND perception from scratch; that is the near-chance trap.

So this is a deliberate step up in cost and a different model class. Gate it on
the geometry result being positive first.

## Core design

Keep the two experimental axes we already have, ADD a perception axis:
- action encoding: {raw turn/forward, allocentric displacement}  (the flip axis)
- position source: {path-integrated, index}                       (the position axis)
- NEW observation source:
    (a) SYNTHETIC location token          -- current control (perception off)
    (b) LEARNED code from the real image  -- perception on

Observation pipeline for (b):
1. Render MiniWorld POV (we already render it every step and currently discard
   it — here we KEEP it). Start at a modest resolution (e.g. 60x80, MiniWorld
   default) and downscale in the encoder.
2. Visual encoder producing a compact per-frame token/embedding:
   - Phase A (cheapest): a small CNN encoder trained END-TO-END with the CE
     objective (no separate tokenizer). ~ResNet-9 / 3-conv stack -> d_model
     embedding, prepended in place of the obs-token embedding. Tests whether a
     jointly-trained encoder can supply a location signal at all.
   - Phase B (if A is degenerate): a discrete VQ-VAE tokenizer trained first
     (reconstruction), FROZEN, then its codes fed as obs tokens (IRIS/ENTL
     recipe). Decouples perception from dynamics; the transformer stays the
     size that already works.
3. The obs the model must PREDICT: predict the NEXT frame's code (Phase B, a
   classification over the codebook) or a contrastive/embedding target
   (Phase A). Revisit-masked, same as now.

Model size: start d=256, n_layers=4-8, 4-8 heads (~5-25M, IRIS lower band) as
research recommended; the encoder adds its own params. Do NOT reuse the 600K
config once pixels are in the loop.

## Why MiniWorld pixels may be a poor testbed (check FIRST, cheaply)

Recon (CLAUDE.md HABITAT_BUILD) found MiniWorld RGB tokenizes degenerately:
"1 code, or ~400 codes over 185 positions" — the scenes are so plain that codes
don't cleanly map to location. So before any training:

  GATE P0 (perception feasibility, CPU-cheap): render N frames along a known
  trajectory, train/fit a VQ-VAE or even k-means on frames, and measure MUTUAL
  INFORMATION between the code and the (x,z) cell. If code<->cell MI is near
  zero, MiniWorld pixels CANNOT support a cognitive map regardless of model
  size, and the perception experiment must move straight to a visually richer
  env (Habitat / MiniWorld with added textures/objects). This mirrors the
  validate-task-first rule and avoids burning GPU on an unlearnable setup.

If P0 fails on stock MiniWorld: enrich the env (per-cell distinct wall textures
/ objects) so appearance is location-informative, OR jump to Habitat.

## Staged plan

Phase 0  Perception feasibility gate (P0 above). CPU. Decide MiniWorld-pixels
         vs enriched-MiniWorld vs straight-to-Habitat.
Phase 1  MiniWorld pixels, encoder Phase A (end-to-end CNN), single seed:
         obs-source (a synthetic control) vs (b pixels), path-int vs index,
         RAW actions only. Question: can a jointly-trained encoder build ANY
         in-context map from pixels (beat chance, generalise to a new map)?
Phase 2  If Phase 1 pixels generalise: add the allocentric axis + 3 seeds ->
         does the flip survive with a learned front-end? (the real result)
Phase 3  If Phase A is degenerate: switch to Phase B (frozen VQ-VAE codes),
         repeat Phases 1-2.
Phase 4  Habitat port (separate, large). Only if MiniWorld-pixels validates the
         pipeline. Note the known Habitat blockers (CLAUDE.md HABITAT_BUILD):
         navmesh sliding on 69-91% of forward moves (continuous displacement
         magnitude, not just direction), RL-policy baselines (DD-PPO) not
         supervised next-token, and ~130k tokens/episode at 256 img-tokens/frame
         vs our 4096 cap. Habitat needs the allocentric CONTINUOUS-magnitude
         recode (record full displacement vector, not just direction bin), which
         is the natural generalisation of the discrete allocentric recode.

## Validity gates (same standing rules, adapted)

- n-gram on the ACTION STREAM alone must stay at chance (both encodings), BEFORE
  training. Unchanged.
- Context-destruction ablation: shuffle in-context obs codes AND action stream
  -> accuracy must collapse to chance. This is MORE important with a learned
  encoder (the encoder could memorise frame->answer shortcuts).
- Solvable-arm requirement: at least one arm must clearly solve held-out before
  a null is interpretable (F1). With perception, also report an oracle upper
  bound (synthetic-token arm = the perception-off ceiling).
- P0 code<->cell MI reported beside every result (the perception analogue of
  "report the measured floor").
- Retrain every arm in one batch; >=3 seeds; NLL-led; measured chance floor.
- NEW encoder-specific risk: guard against the encoder learning a
  frame->answer map that bypasses position tracking. The action-shuffle ablation
  catches this (scramble path -> if accuracy holds, it's a perception shortcut,
  not a cognitive map).

## What would make this worth doing / not

DO IT if: the geometry factorial shows the flip (allocentric rescues path-int),
so perception is the last missing piece before a Habitat claim.
SKIP / DEFER if: the geometry factorial shows NO flip even in the fresh-map
in-context regime -> then continuous-3D path integration is not the story, and
adding perception won't change that; write up the negative instead.

## Rough cost

Phase 0: hours, CPU. Phase 1-2 (MiniWorld pixels, ~5-25M models): comparable to
the current factorial but heavier per step (encoder + kept rendering) — plan
1-2 GPU-days for a 3-seed factorial. Phase 4 (Habitat): weeks; a separate
project with its own conda env (habitat-sim py3.9, already built per
HABITAT_BUILD.md) and likely a DD-PPO or offline-trajectory data pipeline.
