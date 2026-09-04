# paper_rank -- status

Replaces `paper/` (retracted; see `paper/RETRACTED.md`).

**Thesis.** Positional mechanisms in rotary attention are one machine -- a linear
accumulator read out into the rotation phase -- parameterised by state rank.
Rank binds twice, at a geometric threshold and an optimisation threshold, and it
is the only design axis in this family with a measurable effect.

The authoritative statement of the theory is `../mapformer_math.tex`. This
directory is the paper-shaped presentation. When they disagree, the math note is
right.

## Evidence status per section

| section | claim | evidence | status |
|---|---|---|---|
| 2 Frame | the polar decomposition; two factors | derivation | written |
| 2 Frame | the generator family, nesting | derivation; factorisation verified to 9e-07 | written |
| 3 Rank | packing bound, exponent `-max(D/r,1)` | fitted -1.02/-1.51/-2.17 vs -1.00/-1.50/-2.00 | written |
| 4 2D | r=2 is the worst rank; +0.085 for +384 params | `RANK_SWEEP.md`, n=8, t=3.57, 8/8 | done |
| 4 2D | mechanism is a skewed basis, not capacity | `ACTION_GEOMETRY.md`, n=8 | done |
| 4 2D | paper Fig. 4 reproduction | `PAPER_FIG4_REPRO.md`, n=8; 3 of 4 claims | done |
| 4 2D | C4 may be an EM-only claim | `run_em_fig4.sh` | **IN FLIGHT** |
| 5 Generator | SRoPE's knobs buy nothing MapFormer's rank does not | `SELECTIVE_ROPE.md`, n=16/8 | done |
| 5 Generator | the gate is not a token suppressor | `GATE_PROBE.md`, n=8 | done |
| 6 D x r | the geometric threshold scales with D | gates pass (`ND_GATES.md`); training **NOT LAUNCHED** | **PENDING** |
| 7 Discussion | placement is the untested axis | -- | acknowledged limitation |

## Rules this draft must keep

Every headline number carries its seed count, its measured chance rate and its
MDE (`2.8*sd/sqrt(n)`). Nothing from `archive/void/` may be cited. No comparison
against a stored baseline checkpoint -- every arm in a table was trained in one
batch. Convergence is verified before any table is read.
