# Red team and prioritisation (2026-10-04)

Agent 5 of 5. Read-only review; no runs, no edits elsewhere. Every number is quoted from the file named beside it.
Costs are wall-clock on the two 4090s at 2 jobs/GPU, from comparable batches (checkpoint mtimes, first-to-last
finish, so lower bounds): 1-layer T=1024 900-ep runs ~0.2-0.4 GPU-h each (text world 24 runs 2.4 h wall; leak 24
runs 5.7 h; rank_wrap 32 runs 7.0 h); 4-pass / 4-layer runs ~1.4 GPU-h each (loop_rank 32 runs 13.5 h); H3's 128
small ring runs 16.8 h.

## 0. Bottom line

1. The project has two papers in it, both empirical and both small-model. Neither is ready: each lacks one
   baseline table and one scale/scope check a reviewer will require.
2. The single largest unaddressed threat is **the eval/train-mode gap** (TW_NORMSTEP_RESULTS.md, Amendment 2;
   `docs/audits/2026-10-03/dropout_mode_check_out.txt`): up to +0.16 on runs below ceiling (text world s0 0.8835
   eval -> 0.9672 train; NormStepNB s12 0.828 -> 0.993). It has been checked only on text-world checkpoints. Note:
   `project_state.md` open decision 1 reads "DONE 2026-10-04. New, no GPU: eval-mode vs train-mode gap ..."; I
   found no output of that audit in `docs/audits/2026-10-04/` (only `dironly_aside_errors*`,
   `tw_normstep_wordsteps*`), so I read "DONE" as referring to the NormStep item and the audit as NOT run. If that is
   wrong, ignore section 2's "unknown" column.
3. The rank result is the most original thing here and also the most exposed: a reviewer from optimisation will
   call it Burer-Monteiro (exact-rank factorisations have spurious minima; one spare rank removes them), and a
   reviewer from ML will say it is a 1-layer, 2-head phenomenon that 4 layers largely erase (LOOP_RANK: r=2 at 4
   real layers 0.990, 8/8 below loss 0.08). Both attacks are answerable cheaply; neither has been answered.
4. Ranked next steps (section 4): (1) eval/train-mode audit, CPU; (2) lift stalled rank-2 checkpoints to rank 3
   (escape test), ~2 GPU-h; (3) rank D vs D+1 at near-zero wrap in 2D, ~3 GPU-h; (4) baseline table at matched
   length (LSTM, TEM-t, unfactorised/Mamba-3-style phase, RoPE 2-4L), ~8-10 GPU-h; (5) rank at 4 layers x 4 heads,
   ~12 GPU-h.

## 1. The reviews

Format per claim: claim / evidence / attacks (strongest first) / minimal addition to survive.

### C1. Path integration is what lets a transformer learn an in-context map (torus, Match-Query, at matched length)

- **Evidence.** PAPER2X2_RESULTS.md (REG, n=8): T=128 position main effect +0.243 (MDE 0.038, 8/8); RoPE 0.805,
  MapWM r2 0.971, r4 1.000. Matched length T=1024 (SIGN_MATCHED_RESULTS.md): Signed - RoPE +0.267, 8/8 vs 0/8.
  Match-Query (MATCH_QUERY_SCALE.md, not pre-reg., n=5): 0.730 +/- 0.247 vs 0.154; context destruction
  0.918 -> 0.074.
- **Attacks.**
  1. *It is the MapFormer paper's own Table 2.* As a contribution it is a replication with a better recipe. Fine
     as a foundation, not as a result.
  2. *The index arm is a 1-layer, 2-head RoPE that never converged.* PAPER2X2: index final losses 0.68-0.96 vs path
     0.0000-0.38, non-overlapping; the loss-matched position effect is -0.004 (unmeasured). Your own Dyck and H3
     results say depth substitutes for path integration; the torus has no matched-length RoPE depth ladder (the
     MapFormer paper itself ran baselines to 4 layers). A reviewer will say +0.243 is "1 layer of attention cannot
     path-integrate", which is known (Liu et al. 2023 shortcuts to automata; Merrill and Sabharwal on depth).
  3. *Missing non-transformer baselines.* An LSTM given actions path-integrates a torus; TEM / TEM-t are built to;
     Mamba-3 is the published equivalent (data-dependent RoPE, Prop. 3). RESULTS_INDEX.md: "TEM / Mamba / LSTM on A
     -- they exist only in the lm200 column" (withdrawn era). The code exists (`model_tem_t.py`,
     `model_baselines_extra.py`: LSTMBaseline, MambaLikeBaseline, CoPE).
  4. Match-Query is n=5, not pre-registered, bimodal (0.398-1.000), and the T=1024 base cell is n=2 (correction
     block). Cannot be a headline as it stands.
  5. Dropout: RoPE 0.805 and the weak path seeds are below ceiling; direction of any correction unknown.
- **Minimal addition.** One matched-length (T=1024) baseline table, n=8, one batch: RoPE 1/2/4 layers, LSTM, TEM-t,
  an unfactorised Mamba-3-style phase arm, MapWM r=4. Re-register Match-Query at n=8 (cheap: 1-layer). Both modes
  of eval reported.

### C2. The per-head rank of the content-to-angle map decides whether training finds the map (search, not capacity)

- **Evidence.** RANK_SEP_RESULTS.md (REG): per-head rank 2 solves 0/8, 2/8, 2/8; rank 4 8/8, 8/8; D - C_bd Fisher
  and perm p 0.0070. RANK_PROJ: a rank-2 solution exists (0.9955) and is held. RANK3_RESULTS.md: rank 3 6/8, 0.987
  (vs r2 +0.102, perm p 0.027, Holm 0.054; accuracy only). LOOP_RANK_E1800_P1: 2x budget 0/8 vs 7/8 (Fisher 0.0014),
  5/8 still descending. RANK_ND / RANK_WRAP: rank = D 1/8 in 2D and 3D; D+1 8/8 in 2D grid 32 and 3D grid 18;
  failures sit on wrap-only revisits (0.53-0.80 vs 0.91-0.97).
- **Attacks.**
  1. *Burer-Monteiro / overparameterised factorisation, not a positional-encoding fact.* The function class is
     identical for every rank >= D (each rotation block's phase is a plane wave of the D-dim displacement; RANK_PROJ
     shows the rank-2 solution exists). So the effect is optimisation of a factorised linear map `W_out W_in`. That
     exact-rank factorisations have spurious minima/saddles and that rank r*+1 removes them is textbook (Burer and
     Monteiro 2003; Boumal, Voroninski, Bandeira 2016; Ge, Lee, Ma 2016; Arora, Cohen, Hazan 2018 on implicit
     acceleration). The stalled basin you measured (opposition 1.58-1.83, |cos(N,E)| 0.65-0.99: actions collapsed
     onto one direction; RANK3_GEOMETRY.md) is what a rank-collapsed spurious solution looks like. This is the
     framing a reviewer will impose; better to adopt it and test it.
  2. *Scale.* n_heads=2, 1 layer, d=128, one recipe, one length. LOOP_RANK: rank 2 at 4 real layers is 0.990, 5/8
     at the 0.05 cutoff and 8/8 at 0.08. The paper's 12-layer OpenWebText model already uses rank 4
     (`papers/txt/mapformer.txt`, appendix table "rank size 4"). "Use rank > D" may be a 1-layer-toy rule.
  3. *Tension with the MapFormer paper.* Their Table 2 has MapWM-r2 at 1.00 in 2D. Yours fails at T=1024 training,
     and at T=128 r=2 is 0.971 (PAPER2X2). The resolution (failures are on wrap-only revisits, which are rare at
     T=128 on grid 64) is plausible but not tested: the 2D half of RANK_WRAP was voided by memorisation
     (100 cells: own 0.986, unseen 0.273). So "rank = D is hard" may be "exact torus periodicity is hard at rank
     D", which is torus-specific and of less interest to a cognitive-map audience (bounded arenas have no
     wrap-only revisits).
  4. *Accuracy-only firings are exposed to the dropout gap.* Rank 3 fires on accuracy only (Holm 0.054), RANK_WRAP
     3D on accuracy (perm 0.0085; Fisher 0.077), LOOP_RANK aids on accuracy. The rank-2 arms sit below ceiling
     (0.885-0.908), exactly where the gap concentrates. SOLVED counts use training loss (train mode) and are not
     exposed.
  5. *Prior art adjacent.* MapFormer ablates r (r=1 fails in 2D; v4 Table 6 r=2 collapses in 5D); LieRE sweeps
     generator density (peak at 8x8). Your contribution is the per-head split, D vs D+1, and search-not-capacity.
- **Minimal addition.** (a) Escape test: lift stalled rank-2 checkpoints to rank 3/4 with a small random new
  direction (function-identical at init up to the perturbation) and train briefly; escape in most seeds = the
  spare dimension is an escape route (BM mechanism); no escape = the effect acts early in training. (b) Rank 2 vs
  3 at near-zero wrap share in 2D (grid >= 128, T=1024). (c) Rank 2 vs 4 per head at 4 layers x 4 heads. (d) An
  unfactorised (full `n_b x d`) angle map arm: if it solves 8/8 the rule becomes "do not bottleneck below D+1",
  and it doubles as the Mamba-3-style baseline. (e) Train-mode accuracy beside every rank contrast.

### C3. Depth substitution: one path-integrating layer ~ three attention layers (Dyck, biased 1D walk)

- **Evidence.** DYCK_MDEPTH_RESULTS.md (REG): matched depth L32 D12, position main +0.353 / +0.130 / +0.045 /
  +0.024 at 1-4 layers (8/8 each); at 4L 3x budget all at ceiling (+0.002). CANCEL_RESULTS.md (REG, no branch):
  exchange rate k = 3 at p_plus 0.5 / 0.75 / 0.9, 1 at 1.0; path 32/32 SOLVED.
- **Attacks.**
  1. *Path is at ceiling in every cell,* so "1 layer" is an upper bound and the exchange rate is read against a
     ceiling (CANCEL caveat). "k=3" is a threshold artefact of a 0.01 rule; at p_plus 0.9 the 2-layer gap is
     0.0115 (knife-edge), index 1L and 23/24 index 2L STALLED.
  2. *Theory predicts it and you do not engage it.* Biased +/-1 walks on a ring are the word problem for Z_32; Liu
     et al. (2023) give O(1)-depth shortcuts for solvable groups and O(log T) in general; Merrill and Sabharwal
     bound what constant-depth attention can do. The natural question -- does k grow with T? -- is unmeasured.
     Without it "~3 layers" is a constant at one T (128) and one budget.
  3. *Dropout gap.* Index arms below ceiling (H3 2L 0.954 / 0.969 / 0.988; Dyck 1-3L) could move by more than the
     0.01 threshold. k could drop to 2 at p_plus 0.9 from the eval-mode correction alone.
  4. Parameter efficiency is a weak claim at small scale (Mixture-of-Recursions, looped transformers own the
     framing; `reference_looped_transformer_lit.md`).
- **Minimal addition.** Train-mode rescoring (CPU). Then the exchange rate as a function of T (H3 at T = 64 / 128 /
  256 / 512, matched), reported as "fewest index layers within epsilon" with epsilon swept, not one threshold.

### C4. Sign of the increment is a capability, at matched length

- **Evidence.** SIGN_MATCHED_RESULTS.md (REG): Abs - Signed -0.177 (perm p 0.0002), 0/8 vs 8/8 (Fisher 0.0002);
  opposition 0.06 vs 1.92-1.97.
- **Attacks.** Replication (Grazzi et al. negative eigenvalues; Sarrof; Selective RoPE). Near-tautological on a
  task with inverse actions: a monotone phase cannot cancel N with S, and GRAPE-AP's softplus is not designed for
  navigation. Monotone arms STALLED (budget-scoped). Pos carries an init confound.
- **Minimal addition.** None needed to cite as a replication. It is a supporting row, not a headline. Its Fisher
  test is on training loss, so it is robust to the dropout gap.

### C5. Navigation told in words: the model learns which words move it, from prediction alone

- **Evidence.** TEXTWORLD_RESULTS.md (REG A): path 1L 0.969 vs RoPE 1L 0.505 (floor 0.505) and 2L 0.772; fresh
  seeds +0.474, 6/6 vs 0/6. Opposites cancel on 8/8 after removing a common component; 4/8 seeds also run a real
  per-step clock. TW_NORMSTEP_RESULTS.md (post hoc): a direction-only oracle (DirOnly) is capped at 0.970-0.974
  because 100% of its errors (78/78, 89/89, 2 seeds) are aside nouns; learned-step models make none.
- **Attacks.**
  1. *Scripted grammar, 58 words, context-free steps.* A direction word never appears outside a movement clause, so
     the step map is a lookup table and the task is the torus with filler tokens. "Learns which words move it" is
     then the expected outcome of gradient descent on a per-token linear step.
  2. *Baselines.* RoPE 2L STALLED 8/8; no 4L RoPE, no LSTM, no pretrained LM. A reviewer will ask whether a small
     pretrained LM fine-tuned on this solves it.
  3. *Context-dependent steps* (the interesting language case) are pilots only, and the mechanisms are published
     (LIT_CONTEXT_STEPS.md).
  4. Dropout: s0 0.8835 eval vs 0.9672 train. This strengthens verdict A (RoPE 1L at floor), but the "clock vs map"
     4/4 split and per-seed solved labels interact with the gap (NormStep notes: "the dependence on dropout goes
     with not having converged to the clean solution").
- **The DirOnly finding is the interesting part** and survives better than the headline: "only actions should
  move" (TEM-t's action-only update) is the wrong target in language, because language places objects where the
  agent is not. That is a crisp, falsifiable claim against a design principle the cognitive-map field uses.
  Currently post hoc, 2 of 8 seeds.
- **Minimal addition.** Register the aside analysis on all 8 DirOnly seeds plus an aside-free grammar control
  (DirOnly should reach ceiling there). Add RoPE 4L. Keep "scripted grammar" in the title-level scope.

### C6. What/where: converged path models do not need the content x position interaction; the content-to-step leak costs exactly what removing it gains

- **Evidence.** docs/WHAT_WHERE_CHECKS.md (POST HOC): position-only score keeps 0.974-0.989 (cost 0.011-0.025,
  below MDE); RoPE/PoPE fall below floor. LEAK_RESULTS.md (REG): ActOnly / NormStep - MapWM +0.0107 (perm p
  0.0002), = MapWM's own leak.
- **Attacks.**
  1. *Prior art.* "Separation is learned" is MapFormer Fig. 9; ActOnly is TEM-t's update; transfer to unseen codes
     is Chen et al. 2019 (LIT_WHAT_WHERE.md section 4, LIT_NEW_OBJECTS.md).
  2. *The leak gain is tiny and confounded with convergence.* MapWM 8/8 DESCENDING, r(loss, acc) -0.944. +0.0107
     is the size of the eval/train gap seen on below-ceiling runs elsewhere (text world MapWM +0.023 mean). MapWM
     at 0.989 is below ceiling. This registered result is the one most likely to move under the dropout audit.
  3. The x4 test was unfailable (Amendment 1). The causal-score result is post hoc, 1 layer, one task, and
     "below MDE" is "unmeasured", not "zero".
  4. Whittington 2023's "separation needs factorised data" is confounded with map size, not refuted.
- **Minimal addition.** Train-mode rescoring of `runs/leak/p0` (CPU, minutes). If the gap survives: MapWM to
  convergence (2-3x budget, 8 runs). Register the causal-score test on a fresh batch (it is cheap: eval-only on
  any registered torus batch, so register it on the baseline-table batch of C1 before that batch is read).

### C7. Methodological: most positional-encoding gains measured past the training distribution vanish at matched distribution

- **Evidence.** CLAUDE.md citable/withdrawn tables: code -3.694 -> -0.0033 at matched length; Dyck +0.168 -> +0.002
  at matched depth; sign the exception (-0.177 survives). WHERE_THINGS_STAND.md thesis.
- **Attacks.** Known in spirit (Zhou et al. "length generalization is not robust", in `papers/txt/`; Kazemnejad et
  al. 2023 on PE and length generalisation; the general train/test-mismatch critique). As a list of your own
  retractions it reads as a lab notebook, not a result. A reviewer wants a systematic table across PE schemes x
  tasks, each trained matched and mismatched, under one recipe.
- **Minimal addition.** It already has 4-5 rows (code, Dyck, sign, torus 8x, H3 T=512). Present it as one table
  with the same columns, then it is a useful short paper or a section; no new runs needed for a workshop version.

### C8. Boundary: a relational map appears only above a map-size threshold; small maps are memorised

- **Evidence.** ALIASING_CONTROLLED.md / VISITS_TEST.md (not pre-reg.): -0.010 / +0.015 / +0.305 at 32 / 128 / 512
  occupied cells; RANK_WRAP 2D grid 10 memorised (own 0.986, unseen 0.273); WHAT_WHERE_CHECKS: grid 10 has no
  relational "where" at all, grid 32 never-redrawn map is as separated as the redrawn torus.
- **Attacks.** Not pre-registered; index arms behind +0.305 still descending; cell count co-varies with revisit
  rate. The close prior art is the task-diversity threshold for in-context learning (Raventos et al. 2023; Chan et
  al. 2022 data distributional properties), which a reviewer will cite.
- **Why it matters.** For a cognitive-map audience this is a clean, interpretable boundary: memorise vs map, with a
  probe that reads which one the network chose. It is cheap to register.
- **Minimal addition.** One registered batch: map cells {64, 128, 256, 512, 1024} x {MapWM r4, RoPE} x 8, fixed
  training map, unseen-map eval, plus the separation probe as secondary.

## 2. The dropout eval issue: which committed results are exposed

Facts: dropout 0.1 is the default in every model family (`model_continuous.py` etc.); eval scripts call
`.eval()` (e.g. `eval_rank_strata.py:132`); SOLVED labels use final-5% training loss, i.e. train mode, so solved
counts and Fisher tests are not exposed. The gap is concentrated on runs below ceiling, so it tends to raise the
weaker arm and shrink a contrast where the weaker arm is below ceiling and the stronger at ceiling.

| result | primary readout | arm below ceiling | exposure |
|---|---|---|---|
| RANK_SEP per-head rank | Fisher on SOLVED (+ perm on acc) | rank-2 arms 0.885-0.948 | low (Fisher carries it) |
| H1 part 1 (2x budget) | Fisher on SOLVED | r=2 0.908 | low |
| SIGN_MATCHED | perm on acc and Fisher | Abs/Pos 0.78-0.82 | low (Fisher 0.0002) |
| RANK_ND 2D control | Fisher and perm | A2 0.769 | low |
| **RANK3** (3 vs 2) | accuracy only (perm 0.027, Holm 0.054) | B 0.885 | **high** |
| **RANK_WRAP 3D** | accuracy (perm 0.0085; Fisher 0.077) | 3H 0.847 | **high** |
| **LOOP_RANK** aids | accuracy (perm 0.0034 / 0.0012) | A 0.894, L 0.973 | **high** |
| **LEAK** remedies | accuracy +0.0107 | MapWM 0.989, 8/8 descending | **high** (effect is gap-sized) |
| **H3 exchange rate** | 0.01 accuracy threshold | index 2L 0.954-0.988 | **high** (k at p 0.9 is knife-edge) |
| **Dyck matched-depth ladder** | A2f per layer count | index 1-3L | medium (exchange rate may shift) |
| PAPER2X2 +0.243 | accuracy | RoPE 0.805 | medium (index arm may rise) |
| TEXTWORLD A | accuracy | path s0 (0.88 -> 0.97 measured) | none for A; affects B secondaries |
| Match-Query | accuracy | bimodal path seeds | could only widen the gap if path rises |
| WHAT_WHERE causal scores | accuracy of surrogates | MapWM r2 | medium |

Also unresolved: which mode is the right one. Eval mode is the standard and should stay primary; a train-mode
column (mean of 3+ dropout seeds) is a sensitivity analysis. A model that scores 0.16 higher with dropout on is
not using its deterministic forward the way it was trained, which is itself worth one sentence in any paper. For
future headline batches, a dropout-0 arm (or dropout 0 throughout) removes the question; that is a recipe change
and needs its own determinism and convergence check (rules 15, 19).

## 3. Paper stories

### 3a. Cognitive-map audience (NeurIPS neuro-AI, CCN, PLoS CB / Neural Computation)

**Thesis.** A transformer learns a relational "where", separable from "what", only through a path-integrated
phase; whether gradient descent finds that map depends on the dimensionality of the action-to-phase code relative
to the space's degrees of freedom, and it fails exactly on the revisits that demand an exactly periodic code.

Results (3-4):
1. Path vs index at matched length: torus, in-context map (Match-Query, needs n=8), and in words (text world),
   with "the model learns which words move it, that opposites cancel and synonyms are one move" (C1, C5).
2. Rank = D is hard, D+1 suffices (2D and 3D), failures on wrap-only revisits, search not capacity (C2).
3. What/where: separation is causal in the score (position-only score keeps accuracy), absent on memorised small
   maps, present on a never-redrawn large map (C6, C8).
4. "Only actions should move" is wrong in language: an action-only oracle is capped by asides (C5, DirOnly).

Missing, in order: (i) TEM-t and CSCG (or at least TEM-t and an LSTM) on the same tasks at matched length --
this audience will not accept a transformer-only comparison when TEM-t is MapFormer's ancestor; (ii) the rank
result in a regime with few or no wrap-only revisits, because real arenas are bounded -- if rank D fails only on
wrap-only revisits, the rank result is about toroidal closure, which you should then connect to grid-cell module
periodicity explicitly, not leave implicit (do NOT claim hexagonal structure: hex emergence is a live negative);
(iii) any representational prediction for neural data (e.g. what the learned omega / W_out spectrum predicts
about module ratios or phase coding); without it the neuroscience link is analogy; (iv) the dropout audit; (v)
DirOnly on 8/8 seeds with an aside-free control.

### 3b. Positional-encoding / ML audience (ICLR / NeurIPS main, or TMLR)

**Thesis.** For content-dependent rotary phase (MapFormer, Selective RoPE, Mamba-3), two design choices matter in
distribution -- the sign of the increment and the per-head rank of the content-to-angle map -- while most
claimed benefits of positional schemes are robustness past the training distribution and vanish at matched
length and depth.

Results:
1. The matched-distribution table (C7): code, Dyck, torus 8x, H3 T=512, each matched vs mismatched; sign the
   survivor.
2. Sign at matched length (C4, replication in a new regime).
3. Per-head rank: rank = D hard, D+1 enough, not rescued by 2x budget, partly by depth (C2), with the
   Burer-Monteiro framing adopted and tested.
4. Depth substitution 1 path layer ~ 3 attention layers on Dyck and Z_32 walks (C3), ideally as k(T).

Missing, in order: (i) scale -- the rank and depth results at 4+ layers and 4+ heads, and one non-toy sequence
task where the rank axis is measured at matched length (the paper's own OWT model used rank 4; HGRN says
data-dependent phase does not help language, so a null there is likely and must be framed in advance); (ii)
Mamba-3 / an unfactorised phase projection and Gated DeltaNet / LSTM baselines on the state-tracking tasks;
(iii) standard state-tracking benchmarks (group word problems, parity, S5) so readers can compare -- the torus is
the Z_N x Z_N word problem plus recall, say so; (iv) the dropout audit; (v) BM/overparameterisation citations
and the escape test, or a reviewer will write them for you.

The ML story is more defensible today (most rows are registered and robust to dropout via Fisher tests); the
cognitive-map story is more interesting but needs (i) and (ii) first.

## 4. Next experiments ranked by information per GPU-hour

| # | experiment | cost | result that changes the story | risk of uninformative null |
|---|---|---|---|---|
| 1 | **Eval vs train mode on every committed registered batch** (paper2x2, rank_sep, rank_mi, rank3, rank_nd, rank_wrap, loop_rank, loop_rank_e1800, sign_matched, dyck_mdepth, cancel, textworld, leak, tw_normstep: ~730 checkpoints, small). Use `m.train()` under `no_grad`, 3+ dropout seeds, same eval streams as the registered evals; report per contrast whether any verdict flips. | **CPU only**, ~a few CPU-hours (32 cores idle); or minutes on an idle GPU, eval-only | Any of RANK3, RANK_WRAP 3D, LOOP_RANK, LEAK, H3 k, Dyck ladder flipping. LEAK and H3 at p 0.9 are the likeliest to move | none: "no flips" is itself citable and closes the issue |
| 2 | **Escape test for rank** (BM mechanism): take the 8 stalled `runs/rank_mi` / `loop_rank_e1800` r=2 checkpoints, lift to per-head rank 3 and 4 with a small random new direction, train 150-300 ep; control: same perturbation kept at rank 2 | ~16 short runs, ~2 GPU-h | Escape on most seeds: rank acts as an escape route from a spurious rank-collapsed minimum (mechanism, and matches the measured N/E collapse). No escape: the basin is chosen early; rank matters at init only | low; both outcomes are mechanism |
| 3 | **Rank D vs D+1 at near-zero wrap share, 2D** (grid >= 128, T=1024, 900 ep; per-head r=2 vs r=3; the clean 2D half RANK_WRAP lacked; check wrap share and memorisation on CPU first) | 16 runs, ~3-4 GPU-h | Rank 2 solves at zero wrap: "rank" is about exact toroidal periodicity, not maps in general -- rewrite C2 accordingly. Rank 2 still fails: rank is general | low; gate wrap share with the environment code before launch (rule 11) |
| 4 | **Matched-length baseline table on the torus** (T=1024, 900 ep, n=8): RoPE 2L and 4L, LSTM, TEM-t, unfactorised/Mamba-3-style full angle projection (1L), MapWM r4 reference. Register the causal-score readout (C6) as a secondary on the same checkpoints | ~48 runs; 4L arms ~1.4 GPU-h each -> ~25-30 GPU-h; drop RoPE 4L to halve it | LSTM or TEM-t at 8/8: "path integration is necessary" becomes "necessary within attention", reframe C1. RoPE 4L at ceiling: torus becomes a depth-substitution result like Dyck. Unfactorised 8/8: rank rule = "do not bottleneck" | low; it is a table every reviewer requires |
| 5 | **Rank at scale**: per-head r=2 vs r=4 at 4 layers x 4 heads (d=256), T=1024 | 16 runs, ~1.4-2 GPU-h each -> ~25 GPU-h, ~12 h wall | Gap gone (both solve): rank is a 1-layer phenomenon; say so and the ML story loses result 3. Gap holds in loss regime: rank survives scale | moderate (likeliest outcome is a shrunken gap; pre-register a loss-regime readout, LOOP_RANK showed regimes separate cleanly) |
| 6 | **Leak at convergence**: MapWM on the new-object task at 2-3x budget (only if #1 leaves +0.0107 standing) | 8 runs at 2-3x, ~4-6 GPU-h | Gap closes at convergence: the leak is speed only; drop it as a capability claim | moderate |
| 7 | **Exchange rate vs T** (H3 at T = 64/128/256/512, matched; index 1-4L; epsilon swept) | ~128 small runs, ~15-20 h wall (H3 took 16.8 h) | k grows with T: connects to log-depth shortcut theory (Liu et al.); k constant: an O(1) shortcut found, also citable | low-moderate; index arms stall at 300 ep, so budget must be extended first (rule 4) |
| 8 | **Map-size threshold, registered** (C8): cells {64..1024} x {MapWM r4, RoPE} x 8, fixed training map, unseen-map eval, separation probe secondary. Subsumes LIT_WHAT_WHERE P2 (separation vs data) | ~80 runs 1L, ~10-12 h wall | Threshold location and whether separation appears with it; ties to task-diversity ICL literature | low |
| 9 | **DirOnly aside analysis on all 8 seeds + an aside-free grammar control** | CPU for the 8 seeds (minutes); control 8-16 runs ~2 GPU-h | DirOnly at ceiling without asides confirms "language needs what-side steps" | low |
| 10 | H1 part 2 (loop / 4L rank 2 at 1800 ep) | ~6 h wall | Only refines "partly recovers"; superseded by #2 and #5 | high |
| 11 | Re-register Match-Query at n=8 (1L, cheap) | 16 runs, ~2-3 h | Firms up an n=5 headline; fold into #4's batch | low |
| 12 | Window-limit cue-distance curve / Mamba-3 gate vs HSR | ~10-13 h each | Mechanisms published; near-guaranteed by construction | high |
| 13 | Anisotropic SWAP leak test | ~10 h | Tests a 0.01-sized effect that may not survive #1 | high |
| 14 | Talk the Walk / real language | unscoped | Would be the non-toy anchor for 3a; scope only after #4 | -- |

Other CPU-only items worth doing alongside #1: present C7 as one table from existing files (no runs); omega and
W_out spectra of solved vs stalled rank checkpoints (are solved codes commensurate with the grid period, as the
wrap-only story predicts?); Urrutia / Song-Zhong probes at depth on `runs/dyck_mdepth` and text world
(LIT_WHAT_WHERE P3) for comparability with published separation numbers.

Sequencing: #1 first (it may delete the LEAK row and change RANK3's wording before anything else is built on
them). Then #2 and #3 together (~6 GPU-h, one evening) -- they decide which framing C2 takes. Then #4, which is
needed by both paper stories. #5 only after #2/#3, because its pre-registration depends on whether rank is a
periodicity effect. Everything in rows 10-13 is low value for either paper.

## 5. What not to claim, whatever happens

- "Path integration is necessary for in-context maps" without the LSTM / TEM-t row.
- "Rank D+1 is a general rule" before #3 and #5.
- The leak as a capability cost before #1 and #6.
- "1 layer ~ 3 layers" as a constant; it is at one T, one budget, against a ceiling.
- Any hexagonal / grid-cell emergence claim (live negative).
- Novelty for the content-dependent rotation, the taxonomy, sign, "separation is learned", or action-only updates.
