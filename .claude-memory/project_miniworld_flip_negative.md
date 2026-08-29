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

**SURVIVING CLAIM:** the position code matters in proportion to how ALIASED the
observations are. Content that nearly identifies location makes integrated position
worthless; ambiguous content makes it necessary. Monotone, no reversal.

**SCOPE:** aliasing co-varies with grid size here rather than being manipulated
independently. The clean test — fix grid 32, vary n_obs 16/8/4 — is NOT yet run.

**Also still open:** the original "allocentric flip doesn't extend to 3D" negative was
measured at 100 epochs with the bad schedule, i.e. the same confound that inverted the
grid-8 sign. It should be re-run at 400ep/cosine before being cited.
See [[feedback_convergence_first]].
