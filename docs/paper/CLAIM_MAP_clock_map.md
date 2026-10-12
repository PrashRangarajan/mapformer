# Claim map: "a learned path-integrated phase is a map or a clock" (2026-10-12)

Working thesis for a paper: **a learned additive (path-integrated) phase is a MAP when its increments cancel and a CLOCK
when they do not; it beats the token index exactly when the task needs a map, and on clock-like data the model learns a
clock that does no better than counting tokens.** Each claim below is mapped to its evidence, its status and its gap.
Status words as in `docs/WHERE_THINGS_STAND.md`: REG (pre-registered batch), citable (CLAUDE.md row), POST HOC, PILOT,
WITHDRAWN. Prior art: `docs/lit/LIT_CLOCK_MAP.md` (2026-10-12; summary in the section below).

## Claims

| # | claim | evidence (numbers) | status | file | gap |
|---|---|---|---|---|---|
| C1 | Signed increments make the interval sum a net displacement (same place at different times collides: a map); monotone ones make it a path length (a clock) | by construction | definition | `.claude-memory/project_clock_vs_map.md` | prior art (C1 is likely known; see lit) |
| C2 | Forcing a clock destroys a map task, at the training length | torus trained and tested at T = 1024: Abs - Signed -0.177 (p 0.0002), 0/8 vs 8/8 SOLVED; opposite moves cancel only when signed (0.06 vs 1.92-1.97). (The often-quoted crossover -0.280 is OUT OF DISTRIBUTION: trained at T = 128, read at 1024) | REG | `SIGN_MATCHED_RESULTS.md` (`SIGN_ABLATION.md` for the OOD figure) | replication of Grazzi / Sarrof / Selective RoPE in a new regime: say so |
| C3 | A clock costs nothing where a clock is enough, and a signed phase can act as one | k-back recency, trained at T = 1024: at the training length monotone arms 0.961-1.000 vs signed 1.000 (near ceiling; no registered contrast at this length); the registered -0.004 is read at T = 2048 (OUT OF DISTRIBUTION). The unconstrained signed model runs a clock there (C4). The index (RoPE) fails the task (0.234): it counts tokens, not content (C5) | REG, but the clock-side contrast is OOD or at ceiling | `RECENCY_RESULTS.md` | a matched-length clock-side contrast below ceiling; "recency NEEDS a clock" is withdrawn (a rewind solves it) |
| C4 | The free phase chooses map or clock per task | growth exponent alpha of the unconstrained arm 0.591 (torus) vs 0.967 (recency), +0.376, se 0.009; alpha is a re-description of opposition (r +0.9995) | REG (H2) | `RECENCY_RESULTS.md`, `LOCALISATION.md` | alpha is a statistic, not a lever |
| C5 | What a clock counts matters | recency at the training length (T = 1024), eval-only intervention on the trained signed model: a constant step on content tokens vs on all tokens, magnitude-matched, 0.783 vs 0.189 (+0.594, 8/8); the index arms (which count tokens) sit at 0.234 | citable; the matched pair was designed after a failed control (labelled post hoc in the file) | `RECENCY_GATE_ABLATION.md` | one task; an intervention, not trained arms |
| C6 | Where a map is needed, path integration beats the index at matched length | torus +0.243 (8/8, converged recipe); Match-Query 0.730 vs 0.154 (n = 5); Dyck at matched depth +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers; navigation in words 0.969 vs 0.505; on Dyck the accumulated distance tracks stack depth (r 0.755) not token distance (0.278) | REG / citable (Match-Query not pre-reg., n = 5; the depth probe POST HOC) | `PAPER2X2_RESULTS.md`, `MATCH_QUERY_SCALE.md`, `DYCK_MDEPTH_RESULTS.md`, `TEXTWORLD_RESULTS.md`, `CROSS_RESULTS.md` | baselines (LSTM / Mamba-like / deeper RoPE) on the torus |
| C7 | A within-task dial: as steps stop cancelling, the index stops needing depth | biased 1D walk: index needs 3 layers at p_plus 0.5 / 0.75 / 0.9 (0.9 knife-edge, gap 0.0115 vs 0.01), 1 layer at p_plus 1.0 (a clock); path 1 layer solves every cell | REG secondary; the registered primary fell in NO branch | `CANCEL_RESULTS.md` | the only within-task dial; a clean registered readout would strengthen it |
| C8 | On clock-like natural data the learned phase is a clock and buys nothing over the index | TinyStories: alpha 1.00, R^2(phase ~ token index) 0.999 (3/3); MapWM = RoPE (1.5053 vs 1.5050). Code (9 layers, 2048): alpha 0.99-1.02, R^2 0.994-0.998, 75-92% of each byte's step shared; path integration costs +0.0056 bpc at matched length (p 0.024, n = 3). Bach: alpha ~1.00. With the clock removed, TinyStories' content part shows no cancelling structure (real order = shuffled); code's shows a weak one (real order slower on 5/6) | POST HOC / PILOT; code n = 3 | `docs/audits/2026-10-12/alpha_language_out.txt`, `TINYSTORIES_PILOT_RESULTS.md`, `docs/audits/2026-10-05/code_step_table_out.txt`, `CODE_FULLVAL_RESULTS.md` | **confound: clock tasks use 6-9-layer models, map tasks 1 layer**; n = 3 |
| C9 | Map and clock can coexist in separate channels given rank | text world: 4/8 seeds run a per-step clock in 31-38 of 64 channels alongside the map, solving as well | REG secondary | `TEXTWORLD_RESULTS.md`; theory `THEORY_KERNEL.md` Thm 1 (scoped) | -- |
| C10 | When a map fails, it fails by a clock leaking into it | rank-D heads that fail end with a drifting (CLOCK) head (39/40 on SCORE_RANK; thresholds set post hoc); 1D ring: the two slow rank-1 seeds are the drifting ones; drift forbidden by construction solves ~2x faster | descriptive; REG secondaries | `SCORE_RANK_RESULTS.md`, `RANK_TESTS_RESULTS.md` | **rank = D is NOT hard in 1D (REG)**: no causal or general claim |

## Correction 2026-10-12 (RING_DIAL audit M1; applies to C1, C2, C8)
For a ROTARY phase the clock / map split is NOT representational: a non-negative step wraps mod 2 pi, so the second
token's step 2 pi / omega - Delta(+1) reproduces the signed code exactly. Signed and non-negative rotary arms can represent
the same codes; C2's matched-length cost is learnability (search within the budget), and C1's "monotone = clock" holds for
Howard's real, non-wrapping rates (decays, gates), not for rotations. This is itself a candidate contribution: in the
imaginary (rotary) slot, Howard's Cases II and III coincide representationally and differ only in how easily training
finds a map.

## Update 2026-10-12: the clock side (`CLOCK_SIDE_RESULTS.md`, registered)
On k-back recency (k_max 128, T = 1024, 300 epochs) every Case II contrast reads UNMEASURED: Abs / CARoPE / Pos - Signed
+0.08 / +0.17 / +0.08 (perm p 0.16-0.53), no arm converged (0/40 SOLVED; r(loss, acc) -0.99), monotone arms directionally
ahead; the index fails (RoPE - Signed -0.45, p 0.001). C3's gap stays open: a converged, below-ceiling clock-side contrast
needs a longer budget (or a registered speed readout).

## Update 2026-10-12: the ring dial (`RING_DIAL_RESULTS.md`, registered)
Within one task family and one model, a non-negative rotary step (which CAN represent the map by wrapping) learns it worse
where steps cancel: SOLVED 2/8 and 1/8 vs 8/8 at p_plus 0.5 / 0.75 (Fisher p 0.007 / 0.0014); UNMEASURED at 0.9. Abs does
not find the wrap solution (4-7% of channels cancel mod 2 pi vs 63-82% for Signed). No graded dial. This supports C2 as a
LEARNABILITY claim within a task, and the rotary-slot statement (Cases II and III coincide representationally).

## Do not claim (withdrawn or failed)
- Clock vs map predicts when PoPE's encoding helps: both within-task tests failed (`T2_RESULTS.md`; CLAUDE.md withdrawn).
- Recency needs a clock (a signed rewind solves it).
- alpha causes out-of-distribution degradation (route not established; low-frequency account refuted).
- The forget gate is a clock (untested); the InEKF wrap bounds the accumulator (false).
- Rank = D is hard because of the clock (fails in 1D).
- A non-negative rotary phase cannot represent a map (it can, by wrapping; the sign result is learnability).

## Gaps, cheapest first
1. **Prior art** (`docs/lit/LIT_CLOCK_MAP.md`, agent running): which of C1-C5 are published.
2. **The cross-task confound in C8.** Map tasks are 1-layer, clock tasks 6-9 layers. A same-size comparison: MapWM vs RoPE
   at the TinyStories size on a map task (or 1-layer on a clock task), or a registered within-task dial (C7 with a clean
   readout). GPU, ~1-3 h.
3. **Baselines for C6** (the trimmed baseline table, ~8-10 h).
4. Statistical status of C8 (n = 3, pilot): either run as a registered small batch or present as descriptive.

## Prior art (`docs/lit/LIT_CLOCK_MAP.md`, 2026-10-12; two claims spot-checked in `papers/txt/`)
- **C1 is published**: Howard et al. 2014 (J Neurosci 34:4692) -- one integrator with a constant rate (time), a speed rate
  (path length: different routes to one place differ) and a signed velocity rate ("like a bead on a wire": the same code
  for a position). The clock / map table is Howard's three cases applied to a rotary phase. Cite as the frame.
- **C2 (sign) is published as theory and in architectures**: Sarrof 2024, Grazzi 2025, Khavari 2025 (proofs); Selective
  RoPE and Mamba-3 (signed content-dependent rotation); MapFormer itself (CoPE "cannot represent ... its inverse ...
  even in 1D", `papers/txt/mapformer.txt:586`; its Fig. 9). Ours: the sign ablated on its own at matched length on
  navigation -- a replication in a new regime.
- **C3 "a signed phase can emulate a clock" is trivial** (the signed range contains the monotone one); keep only as a remark.
- **C6 depth substitution is known qualitatively** (Liu 2023 prefix sums, Yao 2021, PaTH, MapFormer). Ours: the matched-depth
  exchange rate (1 path layer ~ 3 attention layers) and navigation told in words.
- **C8 must be scoped**: CoPE (monotone, content-counting) beats RoPE in distribution on language and code perplexity
  (`papers/txt/cope.txt` Table 5); Grazzi's signed DeltaNet helps code and math. So "on clock-like data a learned phase
  buys nothing" is at OUR scale only, and our TinyStories "sentence clock" is close to what CoPE was built to learn.
- **Drop** "beats the index EXACTLY when the task needs a map" and "no better than counting tokens" as general claims.
- **"A monotone phase cannot be a map"** holds per channel in a rotary score, not for a population (two monotone channels
  read as a ratio give net displacement: Howard's own construction).
- **Plausibly new (empirical)**: the matched-length sign ablation on navigation; the free phase's per-task growth exponent
  (0.59 vs 0.97) and its use on language checkpoints; the step-table decompositions on code / Bach / TinyStories; the
  matched-depth exchange rate; navigation in words. **Plausibly new as a statement, may be called obvious**: Howard's
  rate alpha(t) = the selective-SSM step = MapFormer's content-dependent step, with sign separating his speed and
  velocity cases.
