# Habitat port — build log and verified behaviour

Built step by step, each unit tested against the published specification before
the next was added. Nothing below is inferred; every number is measured.

## Step 1-2: installation

`habitat-sim` is **not on PyPI** — conda-only via the `aihabitat` channel, and
**py3.9 only**. The main environment is py3.12 + torch 2.6.0+cu124 and must not
be disturbed, so habitat lives in an isolated env:

    conda create -n habitat python=3.9
    conda install -n habitat -c aihabitat -c conda-forge \
      habitat-sim=0.3.3=py3.9_headless_linux_...

This determines the architecture: **habitat generates and tokenises trajectories
into an on-disk buffer; training runs in the main env against that buffer.** Same
split `MiniGridWorld_Cached` already uses, so it is the established pattern here
rather than a new one.

The `headless` build is required — the server has no `DISPLAY`; rendering goes
through EGL. Verified working on the RTX 4090 (OpenGL 4.6, NVIDIA 550.144.03).

## Step 3: DISCREPANCY between the library default and the task spec

| | `habitat_sim` default | PointNav task spec |
|---|---|---|
| `move_forward` | 0.25 m | 0.25 m |
| `turn_left` / `turn_right` | **10.0°** | **30.0°** |

The library ships 10° turns (36 headings); the PointNav benchmark uses 30°
(12 headings). **The adapter must set 30.0 explicitly** or it silently simulates
a different task from the one the literature reports. Our own H=12
continuous-displacement experiment was modelled on the task spec, so it matches.

## Step 4: action semantics — 4 unit tests, all PASS

    TEST 1  turn_left rotates EXACTLY 30 deg, |dpos| = 0.000000
    TEST 2  move_forward translates EXACTLY 0.25 m
    TEST 3  displacement DEPENDS ON ACCUMULATED HEADING:
              0 deg -> ( 0.00, 0, -0.25)     180 deg -> (0.00, 0, +0.25)
             90 deg -> (-0.25, 0,  0.00)     270 deg -> (0.25, 0,  0.00)
    TEST 4  12 x 30 deg closes the circle exactly (0.00 -> 360.00)

**Test 3 confirms the premise in the real simulator.** Habitat's action space is
precisely the regime `KNOB_SWEEP.md` measured as costing −0.388 of the −0.438
position effect: a cumsum of fixed per-token deltas cannot represent it.

## Step 5-6: scenes, and 4 more tests

`habitat_test_scenes` (106 MB, free, no licence agreement) — the three scenes the
official tutorials use. Gibson / MP3D / HM3D need signed agreements and are for
later.

    TEST 5  navmesh loads; navigable area 9.2 / 52.9 / 226.7 m^2
    TEST 6  walls genuinely block motion (12.7-28.8% of forward attempts)
    TEST 7  see below -- the finding that changes the design
    TEST 8  renders vary with position (RGB spread 24.7/36.1/37.1 across views)

## Step 7: THE FINDING — displacement is continuous, not 12-valued

Habitat's agent is **navmesh-constrained**: a forward move that would clip
geometry is *slid along the surface* rather than executed or blocked. Measured
over 361 forward attempts per scene:

| scene | area | exact 0.25 m | partial slide | fully blocked |
|---|---|---|---|---|
| van-gogh-room | 9 m² | 19.7% | 51.5% | 28.8% |
| apartment_1 | 53 m² | 30.7% | 48.5% | 20.8% |
| skokloster-castle | 227 m² | **8.6%** | **78.7%** | 12.7% |

("exact" = displacement predicted from pose alone to <1e-4 m, which confirms the
yaw formula is right; the rest are collisions, not arithmetic errors.)

**Only 9-31% of forward moves produce the commanded displacement.** In the
largest scene it is under 9%.

### What this does to the allocentric recoding design

`ALLOCENTRIC_RECODING.md` records the absolute displacement instead of
turn/forward, which restored MapFormer completely at 4 exact directions and
partially at 12 (`CONTINUOUS_ALLOC.md`). Two consequences here:

1. **The record must come from the POSE DIFFERENCE, not the commanded action.**
   Commanded-action recoding would be wrong on 69-91% of forward steps. The
   simulator exposes `agent.get_state().position` and `.rotation`, so the true
   displacement is always available — this is not a limitation, just a
   requirement.
2. **Direction-only quantisation is insufficient.** Our continuous experiment
   quantised direction into 12 bins with the magnitude fixed at one step. In
   Habitat the *magnitude* is continuous too, because slides are partial. A
   faithful recoding must encode direction AND magnitude — a 2-D displacement
   grid, not a 1-D direction code.

**So `CONTINUOUS_ALLOC.md` UNDERSTATES the difficulty of the Habitat case**, and
its partial recovery (+0.110 -> +0.263) should be read as an upper bound rather
than an estimate. That is worth knowing before committing to the port, and it is
the kind of thing only building it surfaces.

## Still open before a Habitat result is possible

- **Observation tokenisation.** Renders are RGB; every model here consumes
  discrete tokens. The MiniWorld reconnaissance showed two reasonable settings
  giving 1 code and 400 codes over 185 positions — degenerate in opposite
  directions. Unresolved, and worse in a photorealistic scene.
- **Scene scale.** The free test scenes are 9-227 m². Published PointNav numbers
  are on Gibson/MP3D/HM3D, which need licence agreements.
- **Which task.** Realistic PointNav (no GPS+Compass) is the only setting with
  headroom: with the sensor, agents reach 99.8% and there is nothing to measure
  (Partsey et al., CVPR 2022, arXiv:2206.00997).
