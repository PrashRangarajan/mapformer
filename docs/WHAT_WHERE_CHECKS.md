# What/where: three eval-only checks on existing checkpoints (2026-10-03)

Post hoc, not pre-registered, no training. Scripts and outputs in `docs/audits/2026-10-03/`; each was re-run
independently and reproduced byte-for-byte. Literature context: `docs/lit/LIT_WHAT_WHERE.md`, `LIT_NEW_OBJECTS.md`.

## 1. Do trained path models NEED the content x position interaction? (`causal_whatwhere.*`)
48 paper-torus checkpoints (`runs/paper2x2/p0`, 8 seeds x 6 arms, 1 layer, T=128); the rebuilt forward reproduces
every model's logits exactly. Held-out-map accuracy (floor: best n-gram 0.598, always-blank 0.507):

| score used | RoPE | PoPE | MapWM r2 | MapWM r4 | MapPoPE r2 | MapPoPE r4 |
|---|---|---|---|---|---|---|
| intact | 0.805 | 0.679 | 0.971 | 1.000 | 0.999 | 1.000 |
| position-only (object identity removed, action/obs type kept) | 0.463 | 0.476 | 0.837 | 0.989 | 0.974 | 0.986 |
| shared kernel x content gain (content cannot move the peak) | 0.430 | 0.435 | 0.791 | 1.000 | 0.988 | 1.000 |
| additive position + content | 0.471 | 0.378 | 0.640 | 0.957 | 0.956 | 0.990 |
| one global mean (type removed too) | 0.421 | 0.439 | 0.332 | 0.194 | 0.704 | 0.554 |

Converged path models do not need the interaction (cost 0.011-0.025, below the MDE: unmeasured); a gain on one shared
kernel restores 1.000. Index models depend on it entirely (below floor without it). Token TYPE (action vs observation)
is needed. The learned content dependence is multiplicative: forcing it additive is worse than removing it.

## 2. Is separation forced by redrawn maps? (`probe_whatwhere_nd.*`)
Per-head-rank models on the N-D torus, one FIXED training map per seed (`runs/rank_wrap`): grid 10 (memorised: own map
0.986, unseen 0.273) vs grid 32 (generalised: 0.999). Probe verified exact (logits diff 0.0).
- Grid 32, never-redrawn map: as separated as the redrawn paper torus (inter/pos 0.034, peak0 1.000, leak 0.024,
  87% of observation attention on same-cell keys). Redrawing maps is not needed for separation.
- Grid 10: no relational "where" at all -- observation steps as large as actions (leak 1.13, untrained 1.14), no
  cancellation, displacement explains 0.5% of real scores, attention ~3 steps back: an n-gram-like local lookup.
- Within one data condition (3D grid 10, 4/8 solved), separation tracks success across seeds (Spearman inter/pos vs
  held-out accuracy -0.90, same-cell attention +0.98): which solution training finds decides it, not the data alone.
- Not refuted: Whittington-style "separation needs content-position independence in the data" -- a fixed 1024-cell map
  is still nearly factorised (0.21 bits vs 0.87 for 100 cells), confounded with map size.

## 3. Where does "what" leak into "where"? (`e0_*`)
New-object pilot checkpoints (n=1 per arm). In distribution, all remaining object error of path arms is leak:
zeroing object-token steps lifts unseen-object accuracy 0.990-0.993 -> 0.9996-0.9999. The leak is the code's
projection onto the 4-d row space of the object step map, scaled by the embedding norm (the step reads the embedding
before LayerNorm; the content path is protected by LayerNorm): x2 / x4 code norm costs 0.04 / 0.11-0.14 with the
content floor unchanged; codes orthogonal to that row space leak exactly 0; rotations, sparse codes and norm-matched
random maps leak like the originals. The position-only-score arm leaks like the others (the leak is in the step).
Zeroing BLANK steps also costs accuracy: the blank step cancels a common-mode drift of the action code.

## Correction found on the way
The 2026-10-01 what/where probe added the key's own step to the key phase (it cancels); fixed and re-run -- converged
arms unchanged, MapWM r2 and untrained PoPE inits changed (`docs/WHAT_WHERE_ANALYSIS.md` section 6 banner).
