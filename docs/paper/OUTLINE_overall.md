# Overall paper outline: path-integrated rotary position (2026-10-16)

Scope: positive, citable results only. Negatives, withdrawn claims and the live-negatives list appear only as one-line
scope notes where a reviewer would demand them. Every number below was re-read from the file named beside it on
2026-10-16; a CORRECTED / AUDIT block at the top of a results file supersedes its body and has been applied.
Status words: **REG** = pre-registered batch, verdict as registered; **REG-sec** = registered declared secondary (no
verdict); **REG-nb** = registered, result fell in no branch; **IC** = independent results-blind code check amended in
before launch; **NR** = not pre-registered; **POST HOC** = eval-only on stored checkpoints; **PILOT** = n <= 3, not a
result. **ML** = trained and tested at the same length (and depth, where the task varies it).

Companion: the sign line already exists as `docs/paper/DRAFT5_workshop.md` (+ `APPENDIX_draft5.md`,
`APPENDIX_theory_v2.md`). This outline is for the overall paper; Sec. 7 argues for keeping draft 5 as a short companion.

---

## 1. Title options and abstract draft

1. **Path-integrated position learns maps that depth cannot buy, and fails by search, not capacity** (recommended)
2. Where a learned rotary phase becomes a map: depth substitution, language, and two search failures
3. Maps from movements: what path-integrated rotary position buys a transformer, and why training sometimes misses it

**Abstract (draft, ~200 words).** Rotary position can be driven by content: each token adds a learned angle to a
running phase, so attention compares net displacements instead of token distances (MapFormer). We study when this
path-integrated phase helps and why it sometimes fails, training and testing every model at the same length and depth
under pre-registered decision rules. On a 2D torus walk a one-layer path-integrated model solves the map (0.998 at
1024 moves, 8/8 seeds) where one-layer index RoPE stays below a map-free retrace rule (0.731 vs 0.843). One path layer
does the work of about three index layers on Dyck-2 at matched nesting depth and on a biased 1D walk, and of more than
four on the torus, where extra depth or a looped block buys only short-gap revisits. The advantage survives when the
walk is told in English (0.98 vs 0.51 on fresh seeds) and largely disappears on ordinary text, where the learned phase
is a token clock. When it fails on its home task it fails by search: with per-head rank 2 the torus is solved on 0-2
of 8 seeds against 8/8 at rank 4, and with a barrier (|.|) step parameterisation on 0-2 of 8 against 8/8 signed; in
both cases an exact solution exists and survives training when installed (held, or knocked out and re-found), but is
not found from scratch within budget. All models are small (1-4 layers) and all map tasks synthetic.

## 2. Thesis and contributions

**Thesis (adjusted to the evidence).** A path-integrated rotary phase is a strong inductive bias for maps built from
movements: at matched length and depth it substitutes for about three layers of index attention on Dyck-2 and a biased
1D walk and for more than four on the 2D torus, and it carries over to navigation told in words; on ordinary text at
our scale it reduces to a token clock. Its failures on its home task are failures of search, along two axes -- the
per-head rank of the content-to-angle map and the parameterisation of the step's sign: in both an exact solution
exists and survives training once installed (held for rank and for sign on the ring; knocked out and re-found for sign
on the torus), but training from scratch does not reach it within budget.

Changes to the working thesis: (a) "held under training" is true for rank (RANK_PROJ) and sign-on-the-ring
(WRAP_EXISTENCE); on the torus the sign solution is perturbed and re-found (GENERALITY E2 NOT HELD by the LOST rule);
(b) "necessary for in-context maps" rests on NR batches (n = 5, n = 3 destruction) and is demoted to supporting
evidence until registered (gap G7); (c) "rank = D is hard" holds in 2D and 3D but not in 1D (RANK_TESTS), so the rank
axis is stated per task, not as a law; (d) the sign failure is a cost of the barrier *parameterisation* (|.|,
softplus), not of non-negative *values* (a non-negative rotary step represents the signed code by wrapping), and the
through-zero route is what signed steps use, not a demonstrated cause of the |.| failure (GAUGE); (e) "token clock" is
PILOT / POST HOC and scale-limited (CoPE gains on language at scale).

**Contributions.**
1. *Matched-length map advantage, anchored to a map-free floor.* Torus T = 1024: path 1 layer 0.998 (8/8) vs index RoPE
   0.731 (0/8), +0.267, p 0.0002; index sits below the retrace rule 0.843 on every run (REG, n = 8, ML;
   `SIGN_MATCHED_RESULTS.md`, floors `docs/audits/2026-10-16/sign_floors_out.txt`). At T = 128, +0.243 -- but a map-free retrace rule scores 0.934 there (index 0.805 below it, path r2 0.971 just above; `docs/audits/2026-10-16/paper2x2_T128_floors_out.txt`, post hoc), so T = 128 is not map evidence (REG, n = 8, ML;
   `PAPER2X2_RESULTS.md`).
2. *A depth exchange rate.* One path layer ~ 3 index layers on Dyck-2 at matched depth (+0.353 / +0.130 / +0.045 /
   +0.024 at 1-4 layers, 8/8 each; REG, n = 8) and on a biased 1D walk (index needs 3 layers wherever steps can cancel;
   REG-sec, fresh seeds agree); more than 4 on the torus (index 4 layers 0.860, looped x4 0.864 vs path 1 layer 1.000;
   REG + IC, n = 4, complete separations p 0.029), where depth lifts only revisits within 128 moves.
3. *Maps from words.* Torus walk told in English: path 1 layer 0.979 vs index 0.505 on the six fresh seeds (6/6 vs 0/6
   SOLVED, p 0.0022); every seed learns which words move it and cancels opposite directions (REG + IC;
   `TEXTWORLD_RESULTS.md`).
4. *Rank failure is search.* Per-head rank decides torus learnability (rank 2: 0-2/8; rank 3: 6/8; rank 4: 8/8; per-head
   rank contrast Fisher p 0.0070; REG, n = 8, ML); 2x the budget leaves rank 2 at 0/8 vs 7/8 (REG); a rank-2 solution
   exists (0.9955 frozen) and is held under training (7/8 vs control 7/8; REG). MapFormer's r = 2 read literally (per head)
   solves 2/8 at T = 1024.
5. *Sign failure is search, in the parameterisation.* A |.| step costs -0.177 on the torus (0/8 vs 8/8, p 0.0002; REG,
   ML; 0-2/8 vs 8/8 in four torus batches) and 0-4/8 vs 6-8/8 in five ring batches with a signed control (REG + IC);
   an exact |.| solution exists (provably, by wrapping;
   softplus too) and is held (ring 8/8, 0/8 lost) or re-found (torus 8/8 at end), while from scratch the class solves
   0/8 vs 8/8 on both tasks (REG + IC); signed steps reach the map through zero, collectively through their latents
   (frozen latents 1/8 vs 7/8; REG + IC). The cost holds without weight decay, at lr / 3 and at 2 layers d 256; it is
   unmeasured at lr x 3.
6. *Published monotone content phases do not learn these maps.* Faithful CARoPE 0.667 on the torus, below index 0.722
   (p 0.040; it provably cannot cancel); faithful CoPE 0/8 on the ring though it can represent the map (REG + IC;
   `GENERALITY_RESULTS.md`; credit MapFormer v4 Table 2 for CoPE failing navigation).

---

## 3. Section-by-section outline (main paper, 8-9 pages)

Bullet format: claim -- numbers -- n -- status -- ML? -- source.

### 3.1 Introduction (~0.7 page)
- Rotary position as an integrator; content-dependent steps (MapFormer, Selective RoPE, Mamba-3, CARoPE, CoPE). The
  question: when does a learned step make a map, what does it substitute for, and why does training sometimes miss it.
- Method stance in one paragraph: every claim at matched length and depth (the project's OOD-only effects closed under
  matched controls; sign is the one that survived -- one sentence, no list), pre-registered decisions, n = 8, measured
  floors including a map-free retrace rule, SOLVED vs STALLED classes.
- Contributions (Sec. 2 list, compressed).

### 3.2 Background and framework (~0.8 page)
- **Score identity.** Channel c of the score depends on position only through exp(-i omega_c (Phi_{c,t} - Phi_{c,t'})),
  Phi the per-channel cumulative step (Mamba-3 Prop. 3; THEORY_CHECK A0 fix: per-channel Phi; inhomogeneous term
  first-order).
- **Howard integrator framing (one paragraph; earns its place only as vocabulary).** Constant rate = index RoPE (time),
  non-negative rate = CARoPE / CoPE / GRAPE-AP (path length), signed rate = MapFormer / Selective RoPE (position) (Howard
  et al. 2014). The point we need: in the rotary slot the last two coincide representationally (P1 below), so any
  difference between them is learnability. Mapping not found stated elsewhere (`docs/lit/LIT_HOWARD_TRANSFORMER.md`);
  present as framing, not a contribution.
- **P1 (representability by wrapping; proved).** If at most r tokens step, a |.| rank-r model reproduces any signed
  angle table mod 2 pi; annihilating observation embeddings leaves only the moves stepping (2 ring, 4 torus; r = 4).
  **P1(f) softplus (proved, THEORY_CHECK item 5):** omega' = 2 pi / ln 2 makes softplus(0) = ln 2 a full turn; built
  from SO_s800, logits match to 2.8e-11. Existence *under training* for softplus untested.
- **Lemma 1 (relative-sign gauge), with THEORY_CHECK fixes.** Negating a channel's steps (via omega_c; in the signed
  class also row c of W_out) plus reflecting the second coordinate of its q/k leaves every score unchanged. (i) |.|
  fixes only relative signs within a channel (assume omega_c != 0). (ii) **fixed wording:** the barrier excludes no map
  *function* for which a code with sign-coherent rows exists, and that code needs no wrapping (one stepping token; or
  +1/+2). (iii) for an inverse pair, through-zero cancellation is excluded; the wrapped code (a, 2 pi - a) remains.
  Learnability corollary NOT confirmed (GAUGE): state as representation only.
- **Deviation from MapFormer** (must appear): our latent is shared across heads; the paper's W_in is per head. Say so at
  every rank comparison.

### 3.3 Setup and protocol (~0.8 page)
- Model: MapWM, 1 layer, 2 heads, d 128, AdamW, step Delta = W_out W_in x (rank r shared; per-head variants named).
  Reproduction: 0.989 (WM) / 0.987 (EM) on the paper task, matching v1 and marginally below v4 Table 2 (name v4).
- Tasks: torus 64 x 64 (T = 128 and 1024 moves interleaved with observations; held-out map, so accuracy is transfer);
  1D ring 32 cells (T = 128; p_plus dial); Dyck-2 L32 D12 (A2f closing accuracy, chance 0.5, floor 0.594); Match-Query
  (explore then blind query; chance 0.0625); text world (torus walk in 58-word English, T = 1024 words).
- Floors (table in App. B): blank, best n-gram (orders 0-5), retrace (copy the observation 2j steps back while a run
  reverses the previous one). Torus T = 1024: 0.506 / 0.575 / 0.843 (POST HOC, own streams); ring p 0.5 / 0.75 / 0.9
  retrace 0.860 / 0.758 / 0.612; Dyck 0.594; text world constant 0.505, reversal-copy 0.597. **Torus T = 128 retrace
  floor not yet computed (gap G3).**
- SOLVED = final-5% training loss < 0.05; STALLED / DESCENDING by windowed slope. Sensitivity of SOLVED to the threshold
  in App. C (ring p 0.5 sign contrast fails at 0.1; ROBUST torus fails at 0.1).
- Protocol: pre-registration with branches against the measured noise floor; independent code check before launch
  (from 2026-09-28 on; earlier batches named in App. A as without one); pilots on outside seeds (two exceptions,
  TEXTWORLD and CANCEL, reported on fresh seeds); permutation / Fisher tests; "unmeasured" below threshold; "does not
  learn" = within budget. Attention-dropout eval scale (x1/(1-p)) re-score: no registered verdict changes
  (`docs/audits/2026-10-04/DROPOUT_RESCORE.md`).

### 3.4 Results I: where path integration helps, at matched length (~2.3 pages)

**4.1 The torus map, and what index attention learns instead**
- Path 1L vs index 1L at T = 1024 -- 0.998 (8/8) vs 0.731 (0/8), +0.267, perm p 0.0002 -- n = 8 -- REG (no IC; repro
  audit byte-identical) -- ML -- `SIGN_MATCHED_RESULTS.md`. Same contrast replicated: Signed_r4 1.000 vs RoPE 0.722
  (GENERALITY, REG + IC, n = 8) and FACTORIAL 1.000 vs 0.719 (n = 4).
- Index is below the map-free retrace floor (0.843) on every run; it does not even learn the reversal rule -- POST HOC
  floor on registered checkpoints -- `docs/audits/2026-10-16/sign_floors_out.txt`, `generality_floors_out.txt`.
- Short walks: position main effect +0.243 (MDE 0.038, 8/8; index 0.805, path r2 0.971, r4 1.000) -- n = 8 -- REG -- ML. Retrace floor 0.934 on these walks (post hoc): mostly solvable without a map; appendix only
  (T = 128) -- `PAPER2X2_RESULTS.md`. Loss-matched it is -0.004: at matched length the accuracy gap is the fit gap
  (expected; say so). Do NOT cite +0.461 (withdrawn 16-epoch number).
- Supporting (one sentence each, status flagged): in-context maps (Match-Query) 0.730 +/- 0.247 vs 0.154, chance 0.0625,
  n = 5, NR; context destruction 0.918 -> 0.074 (explore observations shuffled) and -> 0.076 (query actions shuffled),
  n = 3, NR (`MATCH_QUERY_SCALE.md`, `MATCH_QUERY_RESULTS.md`). Main text only if G7 lands; otherwise App. L.
- Scope line: the advantage needs a map large enough that lookup fails (threshold between 128 and 512 occupied cells;
  NR, n = 3-5, pooled post hoc; `VISITS_TEST.md`, `ALIASING_CONTROLLED.md`); a 100-cell map is memorised (own 0.986,
  unseen 0.273; REG; `RANK_WRAP_RESULTS.md`).

**4.2 The exchange rate with depth**
- Dyck-2 at matched depth (trained and tested L32 D12, 1x budget): position main +0.353 / +0.130 / +0.045 / +0.024 at
  1-4 layers (8/8 each; index 0.648 / 0.889 / 0.964 / 0.982, path 0.996-1.000; floor 0.594) -- n = 8 -- REG-sec (S1) of
  DYCK_MDEPTH (no IC) -- ML incl. depth -- `DYCK_MDEPTH_RESULTS.md`. At 3x budget and 4 layers every arm is at ceiling
  (+0.002): state "about three layers, budget-scoped" and that index arms close by 4 layers. Mixture training (D 4..12)
  keeps +0.110 at D12 (8/8).
- Biased 1D walk (ring, T = 128): path 1L solves 32/32 cells; fewest index layers within 0.01 of path = 3 at p_plus
  0.5 / 0.75 / 0.9 and 1 at p_plus 1.0 (index 1L 0.717 / 0.676 / 0.825 / 1.000; 3L 0.998-0.999) -- n = 8 -- REG-sec;
  primary REG-nb; no IC; seeds 0-1 were the pilot, fresh seeds s2-s7 agree; p 0.9 knife-edge (2-layer gap 0.0115 vs
  0.01) -- ML -- `CANCEL_RESULTS.md`. Per NUMBERS_AUDIT, quote index 1L from fresh seeds and cite GENERALITY's in-batch
  ring RoPE 0.730 where a single number is needed.
- Torus at T = 1024: path 1L 1.000 (4/4) vs index 1L 0.719, 4 real layers 0.860, looped x4 0.864 (each 0/4, all
  STALLED; complete separations, p 0.029) -- n = 4 -- REG + IC -- ML -- `FACTORIAL_RESULTS.md`. Strata (REG-sec): depth
  or loop lifts index on revisits within 128 moves (0.74-0.79 -> 0.90-0.96) but not long-gap (0.37-0.53) or wrap-only
  (0.38-0.52); path 1.000 on every stratum. Read: depth buys index the short-range (retrace-like) part, not the map;
  4-layer index 0.860 is just above retrace 0.843. The loop matches 4 real layers at 1/4 the parameters.
- Framing against prior art: depth substitution is known qualitatively (Liu et al. 2023; PaTH; Selective RoPE one-layer
  parity). Ours: the matched-depth exchange rate on three tasks and its strata.

**4.3 Maps from words**
- Verdict A (registered): path 1L 0.969 +/- 0.054 (7/8) vs index 1L 0.505 (0/8), +0.464, p 0.0002; index 2L 0.772 (0/8)
  -- n = 8 -- REG + IC -- ML (T = 1024 words) -- `TEXTWORLD_RESULTS.md`. Lead with fresh seeds only: +0.474, 6/6 vs 0/6,
  p 0.0022. Dropout-scale re-score: path 0.993, +0.488 (AUDIT block). Index 1L sits at the constant floor and below
  reversal-copy (0.597).
- What it learns (REG-sec, 8/8 seeds): with the common component removed, north+south and west+east cancel to
  0.002-0.012; zeroing direction-word steps drops every seed to the floor; synonyms collapse (cos 1.000). Half the seeds
  add a real per-step clock in 31-38 of 64 channels alongside the map (CORRECTED block) and solve as well.
- Extension, App. G (scoped): with "took / dropped the X" sentences, path 0.994 vs index 0.516 (REG + IC, n = 8; a
  sanity check); first return after a drop 0.983 vs a location-free floor 0.649 shows **location + recency among nouns
  at the place**, not state semantics (verb-agnostic rule scores 1.000 there; CORRECTED 2026-10-11)
  -- `TW_STATECHANGE_RESULTS.md`.
- Scope line: scripted grammar, context-free steps; the registered step-table criterion (B) did not fire as written.

**4.4 Where a map is not needed: the learned phase is a token clock (short; labelled)**
- TinyStories (6 layers, d 256): MapWM 1.5053 vs RoPE 1.5050 nats/token; phase growth exponent 1.00, R^2(phase ~ token
  index) 0.999 -- n = 3 -- PILOT / POST HOC -- ML (512) -- `TINYSTORIES_PILOT_RESULTS.md` (CORRECTED block: `mr` cancels
  `.`; "spatial 0%" withdrawn), `docs/audits/2026-10-12/alpha_language_out.txt`.
- Code (9 layers, byte level, trained and tested at 2048): exponent 0.99-1.02, R^2 0.994-0.998; path integration costs
  +0.0056 bpc (3/3, t p 0.024, full val) -- n = 3 -- REG rescore -- ML -- `CODE_FULLVAL_RESULTS.md`.
- Required caveats in the paragraph: n = 3; map tasks use 1-layer models, text 6-9 layers (confound); CoPE gains on
  language at scale, so "buys nothing" is at our scale only. Could move entirely to App. J for an 8-page limit.

### 3.5 Results II: failures on the home task are search, not capacity (~2.3 pages)

**5.1 Per-head rank of the content-to-angle map** (torus T = 1024, 900 epochs, every arm from the same r = 2 initial
weights)
- Split by per-head rank: rank 2 arms 0/8, 2/8, 2/8 (acc 0.885-0.948); rank 4 arms 8/8, 8/8 (0.998-0.999); per-head rank
  at matched sharing (D - C_bd) Fisher and perm p 0.0070; sharing (D - C) and W_out scale (C_bd - B) UNMEASURED -- n = 8
  -- REG (no IC) -- ML -- `RANK_SEP_RESULTS.md`.
- Rank 3: 6/8, 0.987 vs rank 2 0.885 (+0.102, perm p 0.027, Holm 0.054; at the registered edge on every count); 4 vs 3
  UNMEASURED -- n = 8 -- REG -- ML -- `RANK3_RESULTS.md`. Unsolved rank-3 seeds sit in rank 2's non-cancelling basin
  (opposition 1.58 / 1.83 vs solved 0.017-0.076; POST HOC).
- Budget: at 1800 epochs from scratch rank 2 0/8 (0.908) vs rank 4 7/8 (Fisher p 0.0014); 5/8 rank-2 runs still
  descending, so "never" is not shown -- n = 8 -- REG -- ML -- `LOOP_RANK_E1800_P1_RESULTS.md`.
- Existence and holding: the rank-2 projection of each solved rank-4 model scores 0.9955 frozen (>= 0.95 on 8/8); started
  there and trained it solves 7/8 = the rank-4 control's 7/8, code stays cancelling through the restart kick -- n = 8
  -- REG (S1 STABLE) -- ML -- `RANK_PROJ_RESULTS.md`. Untested: whether rank 2 can leave its stalled configurations
  (escape test X1 PARTIAL, unmeasured at 600 epochs; `RANK_TESTS_RESULTS.md`; App.).
- Search aids: rank 2 + loop x4 (identical params) 2/8, 0.973; 4 real layers 5/8, 0.990; plain 0/8, 0.894; both aids
  fire on accuracy (perm p 0.0034 / 0.0012), only depth on SOLVED; three loss regimes, the aids never reach rank 4's --
  n = 8 -- REG, verdict UNMEASURED (threshold inside the aids' spread) -- ML -- `LOOP_RANK_RESULTS.md`. App. or one
  sentence.
- Dimension: rank = D is hard in 2D and 3D (2D grid 32 rank 2 1/8 vs rank 3 8/8, Fisher p 0.0014; 3D grid 10 rank 3
  1/8), rank D+1 suffices on a large 3D torus (grid 18 rank 4 8/8, 1.000); failures sit on wrap-only revisits (0.53-0.80
  vs 0.91-0.97) -- n = 8 -- REG-nb + REG (IC) -- ML -- `RANK_ND_RESULTS.md`, `RANK_WRAP_RESULTS.md` (3D half clean; 2D
  high-wrap cell memorised). **Not general**: in 1D rank 1 = D solves 7/8 vs 8/8 (REG + IC; other task and recipe) --
  `RANK_TESTS_RESULTS.md`.
- Practical line: the paper's r = 2 read literally at 2 heads is our per-head r = 2 (2/8); with a full W_out it is our
  r = 4 (8/8). Recommendation: per-head rank >= D + 1. Scope: n_heads 2, one length, one recipe.

**5.2 The sign of the step, i.e. the barrier parameterisation** (condensed from draft 5; details in the companion)
- Torus, ML: |.| -0.177 (0.821 vs 0.998; 0/8 vs 8/8; perm p 0.0002), softplus -0.218 (1/8; init confound) -- n = 8 --
  REG (no IC; repro byte-identical) -- `SIGN_MATCHED_RESULTS.md` (2026-10-16 banner: learnability, not replication).
  |.| mean is below retrace 0.843 (3/8 runs below; POST HOC).
- Ring dial: |.| 2/8 and 1/8 vs 8/8 at p_plus 0.5 / 0.75 (Fisher p 0.007 / 0.0014), unmeasured at 0.9; 3x budget 3/8 vs
  3/8 -- n = 8 -- REG + IC -- `RING_DIAL_RESULTS.md`, `RING_BUDGET_RESULTS.md`. Ring evidence is the SOLVED criterion
  (accuracy gap -0.03; |.| 0.95-0.97, above retrace 0.860).
- Code readout (POST HOC on REG checkpoints): signed models cancel opposite moves mod 2 pi in 64-98% of channels, barrier
  models 3-11% including the runs that solve (Fig. 1 of draft 5; redraw usage-weighted, gap G4).
- Existence: exact |.| construction (balanced; angle error <= 1.4e-6 rad, weights within ~5x signed) held on the ring
  (8/8 SOLVED, 0/8 LOST; usage-weighted cancelling share 0.99 -> 0.98) and on the torus knocked out (peak loss 0.11-0.21)
  then re-found (8/8 SOLVED, 1.000); from scratch the observation-step-free |.| class solves 0/8 vs 8/8 on both (ring
  p 0.0002; torus -0.120, p 0.0002) -- n = 8 -- REG + IC -- `WRAP_EXISTENCE_RESULTS.md` (banner), `GENERALITY_RESULTS.md`,
  `ACCOUNTS_RESULTS.md` T1. Do not count ACCOUNTS T1 runs as replication of E2 (bit-identical; SYNTHESIS item 8).
- Route (ring): from the |.| start signed steps still solve 7/8 (|.| 0/8; Fisher p 0.0014), ending 98-99% opposite-signed
  through zero; freezing the two move latents blocks it (1/8 vs 7/8, Fisher p 0.010) -- n = 8 -- REG + IC --
  `MECHANISM_RESULTS.md`, `ACCOUNTS_RESULTS.md` T2. A start with move-angle sums > pi rescues |.| on the ring (8/8 vs 0/8)
  but not on the torus (0/8; `TORUS_START_RESULTS.md`): one sentence, ring-only.
- Recipe (REG + IC, fresh seeds): cost holds at wd 0 (0/8 vs 8/8), lr / 3 (0/8 vs 6/8), 2 layers d 256 torus (2/8 vs 8/8,
  -0.032, 6/8 |.| still descending); UNMEASURED at lr x 3 (4/8 vs 8/8, -0.013) -- `ROBUST_RESULTS.md`. The lr direction
  is the open objection (gap G6).
- Scope lines: with same-direction moves (+1/+2) the comparison is unmeasured (|.| 2/8 vs signed 5/8; GAUGE), so "the
  cost needs an inverse pair" is not shown; torus rank slack differs between classes (uncontrolled).
- Published monotone methods (REG + IC, n = 8): faithful CARoPE torus 0.667 (0/8) vs index 0.722 (p 0.040), ring 0.741
  = index; it cannot represent cancellation (P6, proved). Faithful CoPE ring 0.887, 0/8 vs signed 8/8 (p 0.0002) though
  it can (constructed), and beats index +0.157 -- `GENERALITY_RESULTS.md`. CoPE fails on a non-wrapping line too (1/16;
  ACCOUNTS T4), with the SOLVED fragility of REPRO note 3 stated.

**5.3 What failure looks like (unifying, POST HOC on registered checkpoints; one figure, one paragraph)**
- Every failure mode leaves a non-cancelling code and fails the same revisits: index at any depth, rank 2, |.| from
  scratch all lose long-gap and wrap-only revisits while keeping short-gap ones (FACTORIAL strata; RANK_ND wrap-only
  0.53-0.80; GENERALITY AO4 0.928 / 0.709 / 0.585 for lag < 128 / >= 128 / wrap-only). Rank stalls: opposition
  0.87-1.78 (RANK_MATCHED / RANK3 geometry); |.|: cancelling share 3-11%. Present as description, not mechanism.
- Parallel to state, not to claim: EM's recency deficit is the same pattern (installed rewind frozen 1.000; EM - WM
  -0.375; `SEARCH_RESULTS.md`, `EM_WM_STATE.md`) -- App. only.

### 3.6 Related work (~0.5 page; pointers)
- TEM / TEM-t (Whittington 2020, 2022); MapFormer (Rambaud et al., v4); CSCG; Howard et al. 2014.
- Content-dependent rotary: Selective RoPE, Mamba-3, CARoPE, CoPE, GRAPE (taxonomy published), PaTH, FoX, Zhang et al.
  2503.17407 (survey of content-dependent phase).
- Sign and state tracking: Sarrof 2024 (Thm 2), Grazzi 2025 (Thms 1-3, App. C.1), Khavari 2025; **Shakerinava et al.
  ICLR 2026 (must cite: optimisation, not expressivity, for input-dependent rotations)**; Helfrich 2018 scoRNN
  (rotation orientation fixed at init); Wang and Wang 2026 / Tran 2026 (RoPE symmetries, scopes Lemma 1).
- Sign-constrained parameterisations: Woodworth 2020, Amid and Warmuth 2020 (Prop. 2), LRU (Orvieto 2023), Dale's law
  (Cornford 2021; Li 2023).
- Depth and shortcuts: Liu 2023, Hahn and Rofin 2024; looped transformers and Mixture-of-Recursions (loop result is
  theirs in kind).
- Check `docs/lit/LIT_*.md` and `docs/paper/review_draft5/PRIOR_ART.md` before each novelty sentence.

### 3.7 Limitations (one short list)
- Small models (1-4 layers, 2 heads, d 128-256), synthetic map tasks, one recipe per task (lr not swept: G6, G9).
- "Does not learn" is budget-scoped (300-1800 epochs); several failing arms still descending.
- Ring sign evidence is a loss threshold (fails at 0.1 for p 0.5); torus at scale: accuracy gap a quarter, 6/8
  descending.
- n = 4 for the torus depth factorial; Match-Query NR (n = 5); CANCEL without IC and with pilot seeds; text clock n = 3.
- No recurrent / SSM baselines; no exp(.) (LRU-type) parameterisation; CARoPE / CoPE at one layer.
- Shared-latent deviation from MapFormer; per-head rank shown at n_heads 2 only; rank = D not hard in 1D.
- Fixed per-token steps cannot integrate egocentric (turn) actions; allocentric recoding restores it (+0.050 -> +0.488,
  8/8), but at the 16-epoch recipe without a matched-length control (`KNOB_SWEEP_n8.md`): scope only.
- Mechanism of the |.| failure from small starts unresolved (route is what signed uses, not shown to be why |.| fails).

### 3.8 Appendix plan
- A. Models, recipes, batches (seeds, IC yes/no, commit, fast-attn flags); reproduction of MapFormer v4 Table 2.
- B. Floors and gates per task (blank, n-gram, retrace; Match-Query gates; Dyck floor).
- C. Per-seed tables for every main-text contrast; SOLVED threshold sensitivity; dropout re-score.
- D. Constructions (rank-2 projection; |.| balanced; softplus P1(f); naive-construction failure; LOST rule).
- E. Theory: score identity, P1, Lemma 1 (fixed), P3 cut circle, P4 sticky wall (omega restored, "along s"), P6 CARoPE.
- F. Dyck-2 at matched depth (full ladder, mixture, base 32 vs 10000 settled).
- G. Text world (step tables, clock seeds, state-change extension with its corrected scope).
- H. CARoPE / CoPE faithful implementations and the CoPE diagnosis (POST HOC).
- I. Rank: RANK_SEP arms, rank 3 geometry, search aids, escape test, 1D ring, 3D / wrap.
- J. Text and code clock (PILOT / POST HOC).
- K. Looped block (FACTORIAL loop = 4 layers at 1/4 params; Match-Query +0.346 unpaired, NR, batches do not reproduce:
  single-batch interaction withdrawn; loop-count sampling NR, single batch -- or leave out).
- L. What/where (POST HOC: converged path models keep 0.974-0.989 with object identity removed from the score; shared
  kernel x content gain 0.988-1.000; index/PoPE below floor; separation on a never-redrawn 32x32 map; partly prior art,
  MapFormer Fig. 9) and the leak (REG + IC: ActOnly / NormStep +0.0107, p 0.0002, 16/16 SOLVED vs 8/8 descending, partly
  speed; fresh seeds replicate SOLVED 8/8 vs 0/8 but not accuracy, +0.0045, p 0.46; `LEAK_RESULTS.md`,
  `GAIN_PHASE_RESULTS.md`).
- M. Match-Query (NR) and the map-extent threshold (NR) if not in the main text.

---

## 4. Figure and table plan

| # | shows | data | exists? | cost |
|---|---|---|---|---|
| F1 | Exchange-rate plot: accuracy vs index layers (1-4) with the path 1-layer line, three panels: Dyck L32 D12, ring p_plus 0.5/0.75/0.9/1.0, torus T = 1024 (1L, 4L, loop); floors drawn (incl. retrace) | `DYCK_MDEPTH` analysis, `CANCEL_ANALYSIS.txt` / fresh-seed file, `FACTORIAL_*.json` | no | CPU, ~2 h |
| F2 | Revisit strata for every failure mode (short-gap / long-gap / wrap-only): index 1L/4L/loop, rank 2 vs 4, \|.\| vs signed | `FACTORIAL_*_STRATA.json` (`fig_factorial_strata.png` covers index), `GENERALITY_TORUS_*_STRATA.json`, `RANK_ND` / `SIGN_MATCHED_STRATA.json` | partly (`fig_factorial_strata`; fix floor 0.507 -> 0.513 drawn / 0.509 own stream) | CPU, ~2 h |
| F3 | Rank: SOLVED count by per-head rank (2D torus arms A-D, rank 3; 2D/3D/1D panels) | results tables (RANK_SEP, RANK3, RANK_ND, RANK_WRAP, RANK_TESTS JSONs) | no | CPU, ~1 h |
| F4 | Training-loss curves, solved vs stalled: rank 2 vs rank 4; signed vs \|.\| (torus and ring) -- the "stalled, not slow" evidence the mock review asked for | per-epoch losses stored only in checkpoints (do not clear run dirs) | no | CPU, ~2 h |
| F5 | Existence schematic + bars: from scratch vs installed (frozen / trained) for rank 2, \|.\| ring, \|.\| torus | RANK_PROJ, WRAP_EXISTENCE, GENERALITY analyses | no | CPU, ~1 h |
| F6 | Text-world step table: direction-word steps in the map plane, opposites cancel after the common component; clock vs map seeds | `TEXTWORLD_PROBE.json`, `TEXTWORLD_SECONDARY.json` | no | CPU, ~1 h |
| F7 (app.) | Cancelling share per run (signed vs barrier), **usage-weighted** | `fig_wrap_share.py` + ACCOUNTS T1 weighting | exists unweighted (`fig_wrap_share.png`) | CPU, ~1 h redraw |
| F8 (app.) | Move-angle planes, ring mechanism | `fig_mechanism.png` | exists; fix shading (only mixed quadrants closed to \|.\|) | CPU, <1 h |
| F9 (app.) | Ring strata | `fig_ring_strata.png` | exists | -- |
| T1 | Main results table: task, contrast, numbers, n, SOLVED, status, ML | results files | no | text |
| T2 | Rank arms (RANK_SEP five-arm table + rank 3) | `RANK_SEP_RESULTS.md`, `RANK3_RESULTS.md` | in files | text |
| T3 | Sign: torus, ring, existence, recipe cells | draft 5 tables, `ROBUST_RESULTS.md` | in draft 5 | text |
| T4 (app.) | Floors per task | `APPENDIX_draft5.md` B + new T = 128 retrace | partly | CPU for T = 128 |

No figure needs a GPU. Every one is CPU from stored JSONs / checkpoints.

---

## 5. Strength ladder

Ranked by evidential strength. Columns: registered? IC? n, fresh seeds, ML, reproduction audit.

**Tier 1 -- REG, IC, n = 8 fresh seeds, ML, repro-audited byte-identical.** Main text.
- GENERALITY (torus E3 0/8 vs 8/8; E2 re-found; CARoPE, CoPE). WRAP_EXISTENCE (E2 HELD, E3). MECHANISM (route, start).
  ACCOUNTS T2 (collective escape). RING_DIAL, RING_BUDGET. ROBUST (4/5 cells). GAUGE G2 (replication) and TORUS_START
  (as scope).

**Tier 2 -- REG, IC, n = 8, ML, with a disclosed defect or no repro audit.** Main text with the flag.
- TEXTWORLD (pilot seeds 0-1 in batch -> lead with fresh n = 6; verdict B no branch). RANK_ND / RANK_WRAP (no branch /
  3D half clean). RANK_TESTS (1D rank, as scope). TW_STATECHANGE A (sanity; C rescoped). LEAK (small, partly speed;
  accuracy did not replicate on fresh seeds).

**Tier 3 -- REG, n = 8, ML, no IC (pre-dated rule 29).** Main text; add a post hoc independent reproduction audit (G5).
- SIGN_MATCHED (repro-audited). PAPER2X2. DYCK_MDEPTH (ladder is a registered secondary). RANK_SEP. RANK_PROJ. RANK3
  (Holm 0.054 edge). LOOP_RANK_E1800 (5/8 still descending). LOOP_RANK (verdict UNMEASURED; secondary detail).

**Tier 4 -- REG but structurally weak.** Main text only with explicit flags.
- FACTORIAL (IC, but n = 4: only complete separations readable, p floor 0.029). CANCEL (exchange rate is a secondary,
  primary no branch, no IC, pilot seeds, p 0.9 knife-edge; fresh seeds agree).

**Tier 5 -- NR, n = 3-8.** Appendix, or main text as one labelled supporting sentence.
- Match-Query necessity (n = 5) and context destruction (n = 3). Looped block on Match-Query +0.346 unpaired (two batches
  that do not reproduce; interaction withdrawn). Map-extent threshold (n = 3-5; pooled post hoc). Allocentric recoding
  (16-epoch recipe, no matched control): scope line only.

**Tier 6 -- POST HOC / PILOT / n <= 3.** Appendix or descriptive sentences.
- Retrace floors and strata readouts (POST HOC but simple and audit-able: fine as floors). Cancelling-share and
  opposition readouts. What/where checks. TinyStories (n = 3 PILOT), code growth exponent (POST HOC), code +0.0056 (n = 3,
  t p 0.024). Step tables (Dyck, Bach, Indirect, code).

**Left out.** Loop-count sampling (NR, single batch, n not stated in `LOOP_SAMPLED.md`); MapPoPE score rule (+0.0243,
ceiling task, fresh seeds +0.009: too small for this story); GAIN_GRAIN (no headroom); clock/map crossover numbers
(OOD); recency content-clock intervention (+0.594, post hoc design, one task); EM search (keep a pointer in 5.3 at most);
Bach / Indirect / PoPE replications (reproduction appendix of a different paper); everything in CLAUDE.md's withdrawn
and live-negatives lists.

---

## 6. Gaps that block submission, cheapest first

| # | gap | kind | cost | blocks |
|---|---|---|---|---|
| G1 | Reword the thesis and claims to the evidence: "held (rank, ring sign) / re-found (torus sign)"; "about three layers, budget-scoped"; words on fresh seeds; clock "at our scale, PILOT"; sign = parameterisation not value; route = what signed uses | text | 1-2 h | everything |
| G2 | THEORY_CHECK fixes (Lemma 1(ii) "no map function"; P4 omega factor and "along s"; P3 \|sigma\| = pi and "past pi", not "unreachable"; collective route needs full-column-rank W_out; P1(a) wording; A0 first-order); add softplus P1(f); NUMBERS_AUDIT items 1-12; citations (Shakerinava ICLR 2026, Helfrich 2018, Wang and Wang / Tran 2026, theorem numbers, MapFormer v4 Table 2 and Figs. 3d / 9b credit) | text | 0.5 day | theory section, related work |
| G3 | Floors still missing: retrace on the PAPER2X2 T = 128 stream; retrace on Match-Query explore/query (if main text); FACTORIAL own-stream floor (0.509) | CPU | < 1 h | 4.1 headline at T = 128 |
| G4 | Figures F1-F6 new, F7 usage-weighted redraw, F8 shading fix, F2 floor fix | CPU | ~1.5 days scripting | presentation |
| G5 | Independent reproduction audit (like `REPRO_AUDIT.md`) of the non-sign main-text analyses: PAPER2X2, DYCK_MDEPTH, CANCEL, TEXTWORLD, FACTORIAL, RANK_SEP, RANK_PROJ, RANK3, E1800, LOOP_RANK, RANK_ND, RANK_WRAP, RANK_TESTS; plus a blind code check of `train_cancel.py` / `analyze_cancel.py` (labelled post hoc) | CPU (agent) | 2-4 h | Tier 3-4 credibility |
| G6 | **Learning-rate objection (SYNTHESIS A1, mock W1).** \|.\| SOLVED rises with lr (0/8 at lr/3, 1/8 at 1e-3, 4/8 at 3e-3); signed is compared at its best lr. Pre-registered per-arm lr sweep on the observation-step-free ring (1e-3 / 3e-3 / 1e-2, n = 8 fresh seeds, best rate per arm): 48 runs. (A 16-run 1e-2 extension on ROBUST seeds is valid only if registered as an extension.) Retitle by the result | GPU | ~3-4 h wall (48 runs); ~1 h (16-run extension) | sign claim and title, both papers |
| G7 | Register Match-Query (path vs index, converged recipe, n = 8 fresh, context destruction as a registered secondary) -- or demote "necessary for in-context maps" to App. M | GPU | not measured; time one run first (the 200-epoch NR runs are the reference) | thesis clause on in-context maps |
| G8 | FACTORIAL to n = 8 (+4 seeds per cell), so ">4 layers" rests on more than complete separations at n = 4 | GPU | ~8 h (project_state estimate) | contribution 2, torus half |
| G9 | Same lr objection for rank and for the index arms: r = 2 vs r = 4 (and index 1L / 4L) at lr x3 and / 3, n = 8 | GPU | ~8-16 h wall (estimate from FACTORIAL / ROBUST throughput; re-measure s/epoch) | contribution 4 at an 8-page venue |
| G10 | Recurrent / SSM baselines on the torus (LSTM, a Mamba-style diagonal SSM with signed eigenvalues, Selective RoPE) -- reviewers will note an RNN integrates trivially | GPU | ~8-10 h (paper-direction estimate) | 8-page / TMLR version only |
| G11 | Venue and length. Main paper is 8-9 pages: CCN 2027 proceedings (abstract 1 Feb, paper 4 Feb 2027; 8 pages; archival DOI in 2026, 2027 terms unverified) or TMLR (rolling). CPAL proceedings (25 Nov 2026, 9 pages, archival) is too soon given G6-G9. The 4-page tracks (ICLR 2027 workshops, list 29 Nov 2026, papers ~1 Feb 2027) cannot hold the overall paper: they fit the sign companion cut to draft 5 Secs. 4.1-4.3 + Lemma 1 (`VENUES_2026-10-15.md`) | decision | -- | which document goes where |

Minimum for a submittable main paper: G1-G5 (text + CPU, ~3 days) and G6 (~4 h GPU). G7-G8 move two thesis clauses
from "flagged" to "registered n = 8" (~8 h + Match-Query cost). G9-G10 are what an 8-page reviewer or TMLR will ask
for (~1-1.5 GPU days).

---

## 7. One paper or two

**Recommendation: two -- a main paper (this outline) and the existing sign paper as a short companion.**

- *For two.* The sign line is self-contained and dense: ten registered batches, its own theory (P1, Lemma 1, P3-P6),
  and an open objection (lr) that decides its title. It already exists as a 4-page-shaped draft (draft 5) aimed at a
  workshop whose genre (controlled falsification) suits it. Folded in whole, it would take ~3 of the main paper's 8-9
  pages and push depth, words and rank into the appendix -- the opposite of the overall thesis. Condensed to ~1 page
  (Sec. 5.2 above), it serves the main paper as the second instance of "solution exists, search fails".
- *For one.* Double reporting: the main paper repeats the sign numbers. A non-archival workshop (ICLR 2027 workshops,
  CPAL Spotlight) allows it; an archival short venue would not. The rank result alone is also thinner than sign as a
  standalone paper, so "rank + sign" as one search paper is not the right split.
- *Split.* Main paper: "Path-integrated position learns maps that depth cannot buy, and fails by search, not capacity"
  (Secs. 4-5 above; CCN 2027 8-page or TMLR). Companion: draft 5 ("Rotary phase steps parameterised to be non-negative
  rarely learn a map ...", retitled after G6), 4 pages, ICLR 2027 workshop (non-archival), cited by the main paper.
  Check each venue's dual-submission rule before submitting both (UNVERIFIED for the 2027 editions).
