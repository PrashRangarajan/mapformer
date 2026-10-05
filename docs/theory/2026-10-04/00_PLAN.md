# Where to go from here: theory and plan from the five-agent review (2026-10-04)

Inputs: `01_synthesis.md`, `02_literature.md`, `03_formal.md`, `04_language.md`, `05_redteam.md` (this folder; scripts in
`scripts/` and in the appendices). Everything below is POST HOC unless marked; nothing here is registered. Claims marked
VERIFIED were re-run by the main session (outputs under `docs/audits/2026-10-04/`); the rest are the agents' and
should be re-run before citing.

## 1. What we now think is going on

**T1. Per-move drift decides whether a one-layer model finds the map (rank theory).** Two agents derived the same lemma
independently. At per-head rank r = D, any step shared by every move (a clock component c) is, in every channel, a
false drift of the decoded position, U^-1 c cells per move. So a rank-D head is clean only if c = 0 exactly; otherwise
it either drifts (CLOCK) or loses a spatial dimension (COLLAPSE). At r >= D+1 a full drift-free map can coexist with a
clock placed off the axes. On the 80 matched-length rank runs, "SOLVED iff some head is CLEAN" holds on 79/80
(VERIFIED: 40 CLEAN/solved, 1 CLEAN/unsolved, 39 CLOCK or COLLAPSE/unsolved). Thresholds were set after seeing the
data, and "solved implies a clean head" is close to circular; the informative part is that failures split exactly into
the lemma's two types. Status: theorem about end states; NOT shown to be the cause. Against causality: a CPU toy did
not reproduce the rank split, forcing c = 0 at rank 2 made the toy worse (1/8), clean D+1 heads end with c removed
rather than parked, and multi-layer rank-2 runs solve with unclean heads (depth routes around it). The red team's
alternative reading: a Burer-Monteiro / over-parameterisation effect (exact-rank factorisations get stuck, a spare
rank lets them escape). Both readings predict the escape test below differently.

**T2. Eval mode under-reports runs below ceiling because of the dropout SCALE (VERIFIED).** Inverted attention dropout
keeps the mean, but with peaked attention the typical training-time output is 1/(1-p) times the eval output; models
below ceiling learned to rely on it. x1.111 at eval recovers every gap run (0.83 -> 0.99), noise alone does not.
Affects any accuracy-only contrast in committed batches; solved-count (Fisher) contrasts are untouched.

**T3. In language the phase is a two-level address: place, plus role in the clause.** An oracle with steps only on
direction words caps at 0.972 because aside nouns share the cell's phase (100% of its errors on 2 seeds). Learned
steps give asides a small net phase offset (0.06-0.33 rad, a 1.4-11 logit gap). TW_NORMSTEP's registered "per-word
clock" is that offset (VERIFIED: aside sentences 0.126 of 0.133 rad). "Only actions move" (TEM-t, CSCG, Vector-HaSH)
is the wrong target in language.

**T4. The leak is a gap-accumulated phase error** (formal agent, not yet re-run): 0 below 128 moves, 0.003-0.018 at
128-511, 0.047-0.122 at >= 512, all 8 MapWM seeds. Predicts NormStep's in-distribution gain vanishes at short T.

**T5. Wrap-only revisits are the non-contractible loops** (theorem): the only revisits that force each channel's
frequency onto the lattice, and the longest-gap ones. The small-grid failures look like a memorisation race against a
fixed training map, not a wrap limit (prediction: redrawn maps make rank D+1 solve grid 10).

**T6. One rate parameter covers time, distance and position** (literature agent, "possibly new"): Howard et al.'s
alpha(t) -- constant for time, speed for path length, signed velocity for position -- is the same object as MapFormer's
content-dependent step and the selective-SSM step. Our sign result is then a matched-length learning test of Case II
vs Case III. A stability argument (a signed rate is unstable in a decay slot, bounded in a phase slot) predicts LEC's
drifting time code vs MEC's phase-based grid code.

**T7. When path integration beats an index on text** (language agent): the target is decided by RETURN to a
coordinate, updates are relative with no landmark naming it (UNNAMED), the text is in event order (NARRATION order),
and the events commute (ABELIAN). The scripted text world meets all four; natural text rarely does (Jericho names its
rooms; code and music are clock-like). Narrative time ("two days later", flashbacks) is the best natural target.
Token-additive steps cannot represent "k days later" beyond parity (proved there); a context gate fixes it.

## 2. Ranked next steps (information per GPU-hour; costs from comparable batches)

| # | step | cost | what it decides |
|---|---|---|---|
| 1 | **Re-score every committed registered batch** in train mode and at x1/(1-p) (paper2x2, rank_*, loop_rank*, sign, dyck_mdepth, cancel, textworld, leak, tw_normstep); report any verdict that flips | CPU, a few hours | Closes the dropout exposure (rank 3, RANK_WRAP 3D, the leak's +0.0107, H3 at p 0.9, the Dyck ladder). Cannot be null |
| 2 | **Leak vs gap re-check** (T4) and the aside probe (04 appendix): re-run the agents' scripts, commit | CPU, minutes | Turns two agent claims into checked ones |
| 3 | **1D ring, per-head rank 1 vs 2**, plus rank 1 with drift forbidden (H3 task, 24 runs) | ~1.2 h | Cheapest test that can break T1: rank 1 solving >= 6/8 kills it |
| 4 | **Rank escape test**: lift stalled rank-2 checkpoints to rank 3/4 (small new direction), train 150-300 ep; control at rank 2 with the same perturbation | ~2 GPU-h, 16 runs | Escape = spurious-minimum (Burer-Monteiro) reading; no escape = basin chosen early |
| 5 | **Forbid drift at rank 2 on the 2D torus** (odd action steps, zero observation steps, or a penalty on c), snapshots every 10 epochs | ~6 h, 16 runs + repro | The causal test of T1 at full scale (the toy said no) |
| 6 | **Rank 2 vs 3 on a large 2D grid with near-zero wrap share** (gate wrap share on CPU first) | ~3-4 h, 16 runs | Is the rank result about toroidal periodicity or about maps in general |
| 7 | **Redrawn-map small grids** (grid 10, rank D+1, 2D and 3D) | ~3 h, 16 runs | T5: memorisation race vs wrap limit |
| 8 | **Landmarks vs path integration in words** (place names at rates 0 / 0.5 / 1, names stripped at eval; path 1L vs RoPE 2L) | ~14 GPU-h | Do names overshadow path integration (and explain the Jericho failure) |
| 9 | **Narrative time with flashbacks** (single words vs "two days later" phrases; context-free step vs context gate vs RoPE 2L) | ~9 GPU-h | A provable limit of context-free steps and its fix, on a non-spatial dimension |
| 10 | **Baseline table on the torus at matched length** (RoPE 2L/4L, LSTM, TEM-t, unfactorised angle map, MapWM r4; causal score readout registered) | ~25-30 GPU-h | Required by any reviewer for either paper |
| 11 | **Rank at scale** (per-head r2 vs r4 at 4 layers x 4 heads) | ~25 GPU-h | Is rank a one-layer phenomenon |

Free: pre-register "one-layer SOLVED iff CLEAN" (T1) as a secondary on whatever rank batch runs next.

## 3. Paper stories (red team)
- **Cognitive-map audience**: a transformer learns a separable "where" only through a path-integrated phase; whether
  training finds it depends on per-head rank relative to the space's dimension (T1), failing on the loops that demand
  an exactly periodic code (T5); in language the right what/where boundary is learned, not "only actions move" (T3).
  Missing: TEM-t / LSTM / CSCG baselines (#10), the near-zero-wrap rank test (#6), a neural-data prediction (T6 is the
  best candidate), the dropout re-score (#1).
- **Positional-encoding / ML audience**: sign and per-head rank matter in distribution; most PE gains are robustness
  past the training distribution. More defensible today (Fisher-based rows survive the dropout issue). Missing: scale
  (#11), Mamba-3 / unfactorised and LSTM baselines, standard state-tracking benchmarks, the escape test (#4).

## 4. What not to claim
Hexagonal or grid-cell emergence (live negative); "rank causes failure through drift" (T1 is about end states; toy
negative); the agents' literature citations marked "from memory" or "abstract only" in 02/03 until read first-hand;
anything post hoc here as registered.
