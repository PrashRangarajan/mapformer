---
name: project-miniworld-flip-negative
description: The allocentric-recoding flip does NOT extend to continuous-3D MiniWorld; falsifies the MiniWorld->Habitat premise.
metadata: 
  node_type: memory
  type: project
  originSessionId: 11c678ec-9c7c-4954-8b14-36979f03e955
  modified: 2026-08-26T21:29:57.307Z
---

The MiniGrid allocentric flip (position effect −0.02 raw → +0.02 recoded) does
**NOT** reproduce on MiniWorld continuous-3D, at either regime (2026-08-26):

- **Fixed-map** (path integration on a known map): no flip, effect ~+0.02 both
  encodings — attention supplies coarse position, path-int not load-bearing.
- **Fresh-map** (in-context, the regime where path-int SHOULD matter, n=3):
  **decisively negative** — position effect raw −0.086 / allo −0.174 (T=512),
  raw −0.051 / allo −0.184 (T=1024), all seeds. Index BEATS path-int, and
  allocentric WIDENS the index lead (RoPE-allo 0.501 = best arm). Fully validated:
  n-gram gate PASS both encodings, context-destruction PASS all 24, solvable arm
  present, flagged arms converged not undertrained.

**Mechanism = ATTENTION SUBSTITUTABILITY (corrected 2026-08-26 by the oracle
experiment — an earlier "reconstruction fidelity" story was REFUTED).** What sets
the sign is whether ATTENTION can integrate position from the action tokens within
its ~2–32-step horizon (grid size / revisit distance):
- Torus 64×64 (long revisits): attention CAN'T span them (index arms at chance
  floor) → the hardwired cumsum is the ONLY integrator → path-int +0.46.
- MiniWorld 8×8 (short revisits): attention CAN integrate → and given good tokens
  BEATS the rigid cumsum → path-int loses.
Decisive test: an ORACLE exact-cell recode (R²→1, clamp rate 0) did NOT flip
path-int positive as the fidelity hypothesis predicted — instead the INDEX arm's
attention near-SOLVED the task (RoPE 0.977, PoPE 0.938) while path-int lagged
(0.32–0.45); position effect went MORE negative (allo −0.174 → oracle −0.571).
So token fidelity modulates magnitude, NOT sign; the forensics R²-correlation was
confounded. Consistent with the standing finding that attention path-integrates and
the SO(2) code is an inductive bias, not privileged info [[project_hierarchy_negative]].
**FINAL (2026-08-28), superseding an intermediate "crossover CONFIRMED" note that
was itself withdrawn.** Once BOTH arms are trained to a flat loss (400 epochs, 5%
warmup + cosine — the default LinearLR-from-step-one prevented convergence):

| environment | cells sharing an obs token | converged effect (path-int − index) |
|---|---|---|
| MiniWorld grid 8 | 2 | **−0.010** (n=3, 6/6 flat) — no effect, both solve it |
| MiniWorld grid 32 | 32 | **+0.173** (n=3, 6/6 flat) — above the 0.150 noise floor |
| Torus 64×64 | 128 | +0.461 (n=8, index arms at the chance floor) |

**THERE IS NO CROSSOVER.** The −0.529 at grid 8 that anchored it was Vanilla failing
to train (0.448 at 100ep/linear → 0.990 at 400ep/cosine). Index never actually beat
path integration anywhere; there is a regime where the choice does not matter.

**THE ATTENTION-HORIZON MECHANISM IS FALSIFIED** by our own gate G6: revisit lags
SHORTEN with grid size (47/43/38/33) and the fraction inside the ~32-step horizon
RISES (0.43→0.50). Collected before training, never cross-checked until an
adversarial review found it.

**THE ALIASING CLAIM IS FALSIFIED, SIGN INVERTED (2026-08-30).** It was
correlational -- aliasing co-varied with map size across those environments. Holding
grid FIXED at 32 and varying n_obs alone (which relabels the obs_map and changes
nothing else: label mass 50.4/traj and revisit lag median 33 are byte-identical
across conditions) gives the OPPOSITE ordering:

| n_obs | cells/token | effect | converged |
|---|---|---|---|
| 16 | 32 | +0.178 (n=5) | 10/10 flat, 400 ep |
| 64 | 8 | +0.310 (n=4) | 6/8 flat, 400 ep |
| 256 | 2 | **+0.305** (n=3) | **6/6 flat, 800 ep** |

LESS aliasing gives a LARGER effect. Endpoints differ +0.127, t=2.52, all converged.
Pre-registered outcome B fired. See ALIASING_CONTROLLED.md.

**Budget mattered, and convergence-conditioning LIED about which way.** At 400 ep
the n_obs=256 index arm was flat 1/5 and read +0.374; the both-flat-only sensitivity
said non-convergence was SUPPRESSING the effect. Doubling to 800 ep converged it 3/3
and the effect FELL to +0.305. Trust budget extensions over conditioning arguments.

**WHAT REPLACES IT: a THRESHOLD in map size, at matched aliasing (2.0 cells/token).**
grid 8 (32 occupied) -0.010, grid 16 (128) +0.015, grid 32 (512) **+0.305**. Flat,
flat, jump between 128 and 512 occupied cells. Not graded.

**"Distinct cells visited" was my replacement hypothesis and it is ALSO DEAD.**
grid 32 @ T=128 (48 distinct, 1.95 prior, 512 occupied) gives +0.275 where grid 8 @
T=512 (46 distinct, 8.64 prior, 32 occupied) gives -0.010 -- matched on distinct
cells, opposite results.

**STRUCTURAL LIMIT, check before designing anything in MiniWorld:** prior-visit
counts per grid size DO NOT OVERLAP (grid 8 spans 5.67-18.35 over T=128..2048; grid
32 spans 1.95-4.13). Small maps FORCE frequent revisits, so map extent and visit
statistics are near-inseparable here at any episode length. Measured counts are
8.64/4.61/3.05 for grid 8/16/32 at T=512 -- NOT the T/n_occupied arithmetic
(16/4/1), because the walk is directed. Only grid 32 @ T=2048 separates them; not run.

**SURVIVING CLAIM (narrowed):** the position effect is driven by some environment
statistic that is threshold-like in map size and is NOT observation aliasing and NOT
distinct-cells-visited. Which of visits-per-cell vs map extent remains unresolved and
may be unresolvable in this environment. See [[feedback_convergence_first]].
