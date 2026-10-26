---
name: project-state
description: LIVE STATE ONLY -- what is running, what the user must decide, the last results. History is docs/LOG.md and docs/SESSION_*.md; citable and withdrawn claims are in CLAUDE.md; orientation is docs/WHERE_THINGS_STAND.md.
metadata:
  type: project
---

Updated 2026-10-26. Goes stale fast: check `git log`, `.done` markers, the logs named below.
New here? Read `docs/WHERE_THINGS_STAND.md` next; the narrative of 2026-10-04..09 is `docs/SESSION_2026-10-04_to_10-09.md`.

## Running (2026-10-26)
- Nothing running; both GPUs on kalman idle. IMAGE_TORUS (block A, 48 runs, seeds 3100-3107, code `6c4fdf4`, kalman GPUs
  0-1, 2026-10-25 12:31 -> 23:55; marker `.image_torus_done`) is done and folded in (`IMAGE_TORUS_RESULTS.md`).
- Next: **Memory Maze on Hyak** (first-person 3D; scoping, nothing registered). memory-maze 1.0.3 installed at
  `/gscratch/cse/prashr/mapformer_mm/mm-env` with gym 0.26.2; EGL rendering works on A40; ~36 env steps/s per process at
  9x9, ~21 at 15x15; ExtraObs gives agent_pos / agent_dir / maze_layout / targets. Plan (user's): the official 9x9 dataset first,
  then scrubbed or generated data via the arrays, then a pre-registration. Cluster lessons: `docs/HYAK_NOTES.md`
  (cleaned; connection details are in an uncommitted local file outside the repo -- ask the user): read before submitting jobs.
- Optional from the BIG_PICTURE triage: the LSTM/GRU torus baseline (~8 h), if TMLR is the target. IMAGE_TORUS block B
  (NormStep on images, 4-layer index, token RoPE; ~6 h on two GPUs) is registered as optional and was not declared.

## Last results (2026-10-26): IMAGE_TORUS (`IMAGE_TORUS_RESULTS.md`): PATH WINS IN PIXELS
- Torus T = 1024, each observation a fresh MNIST / Fashion-MNIST class image (test split at eval), CNN end to end, 1 layer.
  P1 (partly pilot-informed): action-driven path step (ActOnly) 0.9946 (8/8, at/above the perception ceiling 0.9927;
  long-gap / wrap-only 0.994) vs index RoPE 0.7010 (below retrace 0.843), +0.294, Holm p 0.0003. Pixels cost the path model 0.005.
- P2 (PILOT-INFORMED, NOT BLIND): MapWM with the image in its step at token scale (gain 30.37) 0.668: 5/8 COLLAPSED
  (memoryless, constant encoder), 3/8 learned (0.92-0.96), 0 NOISY. G (pilot-informed): gain 1 0.9938, 8/8. S2b +0.287
  (p 0.039). Design rule: actions drive the step, images do not.
- Folded into CLAUDE.md (row + leak-row pointer), claim map, RESULTS_INDEX, draft 2 (abstract, contribution 5, Sec. 8
  "Maps from pixels", design guidance, Limitations; App. A3 row, K4 new) and its PDFs.

## Last results (2026-10-25): G-3 block A2 (`G3_SSM_A2_RESULTS.md`): COST FOR |.| AND SOFTPLUS, NO SIGN COST FOR EXP
- SSM, ring 32, lr 1e-3, 300 ep: |.| 0/8 (0.756) and softplus 0/8 (0.938) vs signed 7/8 (0.994): SIGNED LEARNS BETTER
  (Fisher 0.0014; softplus Holm 0.0042). exp 6/8 (0.985): NO SIGN COST (Fisher 1.0, perm 0.72), no RUNAWAY. Amendment 3.2:
  "no sign cost for exp at lr 1e-3", read with the past-pi-start caveat; exp's 3e-3 cell stays instability. The sign claim
  is now "|.| and softplus increments", not every non-negative increment (CLAUDE.md sign row, claim map, drafts 2 and 6).
- Mechanism (analysis, `docs/audits/2026-10-25/g3_a2_exp_mech.py`): exp cancels by wrapping (0.21 of 0.22 cancelling
  channels), latents cos +0.60, steps positive and bounded (max 4-8); 90 of 95 cancelling channels started past pi. In
  attention exp solved 1/8 at both rates (SIGNROUTE), so the start alone does not obviously explain it (not tested).
- Signed at 1e-3: 4/8 runs solve with partial codes (T = 512 0.43-0.78).

## Last results (2026-10-25): FACTORIAL_N8 (`FACTORIAL_N8_RESULTS.md`): REPLICATES
- Fresh seeds 2800-2803 alone: path 1L 0.999 (4/4) vs index 1L 0.729, 4L 0.830, loop 0.859 (0/4 each); C0 PATH WINS,
  C1 / C2 DOES NOT CATCH UP, C3 both HELP, all p 0.029. Pooled n = 8 (descriptive): 0.999 8/8 vs 0.724 / 0.845 / 0.862.
- Secondary: 4 index layers at the retrace floor (-0.016, 1/4 above; pooled -0.000), loop +0.014; depth buys short-gap
  returns only (0.913 vs retrace 0.911). Folded into CLAUDE.md, claim map, RESULTS_INDEX, draft 2 (Sec. 6, Fig. 1/3
  captions, Limitations, App. A3, C2, F).

## Last results (2026-10-24): G-4 TORUS_LRSWEEP (`TORUS_LRSWEEP_RESULTS.md`): COST AT BEST RATES
- Torus T = 1024, observation-step-free rank 4, 900 ep, lr 1e-3 / 3e-3 / 1e-2 (wd 0.05 coupled): |.| best 3e-3 0/8 (0.890)
  vs signed best 1e-3 8/8 (1.000), decision p 0.0002 SOLVED / 0.0095 accuracy (selection-aware); 5/8 |.| descending
  (budget-scoped); stage 2 not triggered. |.| 0/8 at every rate; signed 0/8 at 3e-3 and 1e-2 (all descending; predicted
  7-8/8 at 3e-3: missed). Closes LRSWEEP's "ring only". Pilot (koopman s2492-2493): DO NOT EXTEND. Folded into CLAUDE.md,
  claim map, RESULTS_INDEX, draft 2 (abstract, contribution 3, Sec. 7.2 / 7.3, Limitations, App. G9 / H12) and draft 6.

## Last results (2026-10-24): SIGNSEP (`SIGNSEP_RESULTS.md`): UNMEASURED
- Route vs wrap barrier for the |.| cost, 2x2 from one common |.| start, ring, 80 runs (seeds 2300-2307). lr 3e-3: W (|.| +
  collective route to the wrap, barrier kept) 0/8 vs signed 8/8 (A-LIKE; half-reading BARRIER BLOCKS, control P from past
  pi 7/8 on the wrap sheet tau_sum 2.00); F (signed, frozen latents: no barrier, no route) 4/8 UNMEASURED. lr 1e-3
  UNREADABLE (signed anchor 5/8; 7/8 twice before from this start). Folded into draft 2 (Sec. 7.3 paragraph, Limitations,
  App. H11) and one sentence in draft 6 Sec. 4.3. Ops: supervisor restarted 00:38 pinned to GPU 1 (another user on GPU 0);
  stale lock `.run_signsep.lock.stale_20261024`.

## Last results (2026-10-23): G-3 SSM (`G3_SSM_RESULTS.md`): SIGN COST IN THE SSM
- One-layer diagonal complex SSM (= unnormalised rotary linear attention), ring 32, lr 3e-3, 300 ep, minimal_h2 (104 runs):
  |.| 0/8 (0.821, below retrace 0.864) vs signed 8/8 (p 0.0002): the sign cost is not specific to softmax attention.
  softplus 1/8 (5 descending, budget-scoped); exp 0/8 = 8/8 RUNAWAY (instability at 3e-3; A2 at 1e-3 not run).
- Rule 14: exact constructions incl. exp (first build of (g)); |.| perturbed and re-found at 3e-3, held only by secondary
  E2' at 1e-3 (registered E2 PERTURBED via the 10x clause): the registered SEARCH COST joint did not form. Folded into
  draft 2 (Sec. 7.3, abstract, Limitations, App. H10). Open, if wanted: A2 (1e-3 from scratch, 40 runs) would settle exp.

## Running
- Nothing running (LRSWEEP finished 2026-10-17 05:03).
- Draft 6 (`docs/paper/DRAFT6_workshop.md`) carries every review fix; title stands (LRSWEEP: cost at best rates). Overall two-paper outline:
  `docs/paper/OUTLINE_overall.md`. Reproduction audits: sign line and non-sign line both clean (Match-Query not reproducible).
- Draft 5 (`docs/paper/DRAFT5_workshop.md`, `APPENDIX_draft5.md`) has ROBUST filled (4.1, Limitations, abstract, appendix C13).
- Draft 4 (`docs/paper/DRAFT4_workshop.md`) complete incl. Sec. 4.5 (GAUGE) and the end-to-end read (`docs/paper/READ_draft4.md`:
  shareable with co-authors; not submittable: main text 4,302 words vs ~3,500, appendices A-D, F-I still plans; title and
  contribution 3 lean on an account GAUGE did not confirm). Venues: `docs/paper/VENUES_2026-10-15.md` (ICLR 2027 workshop
  list out 29 Nov 2026, papers ~1 Feb 2027; CCN 2027 4 Feb; CPAL Spotlight 18 Jan; Cosyne abstracts 18 Oct 2026).
- Mock review of draft 3: `docs/paper/MOCK_REVIEW_draft3.md` (6/10).

## Running / handoff (2026-10-20)
- Nothing running (SROPE_TORUS finished 2026-10-21 07:54).
- Runbook for another machine: `docs/RUNBOOK_NEXT_EXPERIMENTS.md` (setup, protocol, queue G-1..G-11, CPU work).

## Second machine (koopman-side clone, /home/prashr/mapformer), 2026-10-20/21
- Set up per `docs/RUNBOOK_NEXT_EXPERIMENTS.md`: torch 2.10.0+cu128 in the user site (driver 550.144.03 = CUDA 12.4; works via
  minor-version compat, kernels verified); miniconda's 2.6.0+cu124 kept as fallback. `test_guards` 6/13 here: the 7 failures
  are missing `.pt` under runs/{dof,warm,unfreeze} (rsync needed), not code. One job per GPU: packing measured slower.
- SIGNROUTE ran here (GPU 1, done 2026-10-21 07:43). Its torus part (frozen move latents) is NOT built; held until the read.
- G-1 RANKCLOCK ran here (GPU 1, 2026-10-21 18:48 -> 10-23 02:16). Nothing running on koopman after it.

## Last results (2026-10-24): RANKRECIPE (`RANKRECIPE_RESULTS.md`): RANK D+1 NEEDED SURVIVES A THREE-RECIPE SEARCH
- Rank 2 0/8 at each of lr 1e-3/wd 0.05, lr 3e-3, wd 0.2 (0/24) vs rank 4 8/8; best (lr 3e-3) vs rank 4 p 0.0002. Ran on
  koopman GPU 1 (03:59-21:10). The rank claim is robust to these recipes; not to all recipes (no lower-decay arm).
- Folded into draft 2 (Sec. 7.2, Limitations, App. G9), claim map and RESULTS_INDEX on 2026-10-24.

## Last results (2026-10-23): RANKWD (`RANKWD_RESULTS.md`): DECAY NEEDED
- Rank 4 at AdamW wd 0.05 8/8 vs wd 0 1/8 (p 0.0014), replicating G-1's post-hoc split. Decay is on ALL parameters, so not
  attributable to the step map. CORRECTED 2026-10-24: omega's norm is gauge-dependent; the gauge-invariant per-move angle
  ends at 0.45x with decay and grows ~9x without. RANKDECAY (omega 2x2) shelved on branch rankdecay-wip (gauge flaw).
  Ran on koopman GPU 1 (03:00-11:35). The 1800-epoch clock-slot test and G-4 were dropped by the user for now.

## Last results (2026-10-23): RANKCLOCK / G-1 (`RANKCLOCK_RESULTS.md`): PARTIAL primary, CLOCK IS THE OBSTACLE
- Drift-forbidden rank 2 8/8 (= rank 4) vs actions-only 1/8 (p 0.0014): rank 2 fails because of the clock. Clock slots at
  rank 2 (per-axis, per-token) 3/8 each vs rank 2 1/8: the 'clock exit' mechanism is NOT supported (PARTIAL, budget-scoped).
  Post hoc: rank 4 at wd 0 solves 1/8 (vs 8/8) -> weight decay looks necessary; X2 unreadable. Ran on koopman, GPU 1.

## Last results (2026-10-21): SROPE_TORUS (`SROPE_TORUS_RESULTS.md`): FAILS THE MAP BUT USES ITS PHASE
- Faithful Selective RoPE 0.961 0/8 (all descending) vs MapWM 8/8; +0.236 over index RoPE; state-tracking config 0/8 with
  all-positive steps. Observations step ~0.9x moves (leak at full size).

## Last results (2026-10-21): SIGNROUTE (`SIGNROUTE_RESULTS.md`): SUPPORTED, scope narrowed
- Hybrid (non-negative readout matrix, signed increment) 16/16 = signed vs |.| 0/8, 2/8 at lr 1e-3 / 3e-3; ends antipodal like
  signed. Cost is the non-negative INCREMENT, not the readout. Does NOT separate S3 from S1/P3 (Amendment 1); 'decisive'
  withdrawn for G-2, so G-1 carries more of the clock-theory weight. abs_tied (no solution in class) 0/16 as required.

## Last results (2026-10-20): DELTA_FREE (`DELTA_FREE_RESULTS.md`): PEAK-PINNING SUPPORTED
- Freeing PoPE's offset: index +0.122 (R3+ +0.180), path at ceiling; MapPoPE mechanism now causal at T = 128.
- Also 2026-10-20: seven expansion reports + draft-2 plan (`docs/paper/expansion/`), what/where at T = 1024
  (`docs/audits/2026-10-20/WHAT_WHERE_T1024.md`). Next: decisive clock-route GPU tests (~22 h), draft 2.

## Last results (2026-10-17): LRSWEEP (`LRSWEEP_RESULTS.md`): COST AT BEST RATES
- Each arm at its own best of 4 lr: |.| 2/8 (3e-3, bracketed) vs signed 8/8 (1e-3), p 0.007; the lr objection is answered
  on the ring. Stage 2 (decay-matched) not triggered. Next: reframe the overall outline (MapFormer already owns "path
  integration helps"; ours: benchmark floors / matched-length re-measurement, depth exchange rate, failure by search, words).

## Last results (2026-10-16): ROBUST (`ROBUST_RESULTS.md`): PARTLY UNMEASURED
- Sign cost holds with wd 0, lr /3, and on the torus at 2 layers d 256 (2/8 vs 8/8, but acc gap only -0.032, 6/8 |.| still
  descending); at lr x3 it shrinks to UNMEASURED (4/8 vs 8/8, -0.013). Own-stream torus retrace floor 0.844; all runs above.

## Last results (2026-10-15): GAUGE (`GAUGE_RESULTS.md`): AS FALLS
- Sign cost replicates on +1/-1 (|.| 1/8 vs 8/8); on +1/+2 (same-direction moves) UNMEASURED: |.| 2/8 vs signed 5/8, harder
  for both; |.| no better on +1/+2 than +1/-1. Lemma 1's learnability corollary not confirmed; draft 4.5 filled, contribution 3
  and limitations reworded.

## Last results (2026-10-15): TORUS_START (`TORUS_START_RESULTS.md`)
- Past-pi start does NOT rescue on the torus (0/8; TS1 NO EFFECT); the large start drags signed to 2/8; sign cost replicates
  (1/8 vs 8/8). The start lever is ring-only; "initialise near the wrap" withdrawn as a general rule.

## Last results (2026-10-15): ACCOUNTS (`ACCOUNTS_RESULTS.md`)
- T2 SUPPORTED (collective latent escape); T1: no drift in used channels (old 87->18% withdrawn), weight decay irrelevant;
  T4: CoPE fails the line too (wrap not the obstacle); T3 toy: observation target not the difference.

## Last results (2026-10-14): "do 1 and 2" done
- MECHANISM (`MECHANISM_RESULTS.md`, reg): |.| removes the path through zero; signed escapes the |.| start through zero (7/8);
  |.| from a start with move-angle sums > pi learns 8/8 by wrapping (M2 RESCUES, M2b sum not size, M6 no sign cost from there);
  M5 sign cost at the small start. Toy's "start not constraint" did not transfer (stated).
- GENERALITY (`GENERALITY_RESULTS.md`, reg): torus E3 0/8 vs 8/8; E2 NOT HELD = perturbed and re-found (8/8 SOLVED at end);
  CARoPE fails, worse than index on torus; CoPE (ring) fails though representable, beats index +0.157.
- WRAP_COPE (`WRAP_COPE_RESULTS.md`, reg, 2026-10-14 17:54): near-wrap CoPE table start UNMEASURED (all contrasts; bimodal);
  every arm's table ends period-2 -- in CoPE the wrap is in a free table, so the start lever is for rotary steps only.
- THEORY REVIEW (`docs/audits/2026-10-14/theory_review/THEORY_REVIEW.md`): per-claim A-D verdicts. Proved: wrapping when <= r
  tokens step (Prop 1), cut circle / sticky wall / pi-mirror (Props 3-5), rescaling sensitivity (Prop 2), CARoPE can't cancel
  (Prop 6). Mechanism dynamics NOT derivable per channel -> stated as an account (collective; signed has one latent route
  z(-) -> -z(+)). Depth result is learnability (Liu 2023 constant-depth gridworld; abstract-level, verify). GRAPE-AP is real-s.
- COPE DIAGNOSIS (`docs/audits/2026-10-14/cope_diagnosis/COPE_DIAGNOSIS.md`, post hoc): one-layer gates are a table; solution =
  gate lattice (period P = 2/3, +1 move odd multiple of P/32, ~1e-3); regimes lattice 9 / off-lattice 11 / obs-dependent 20;
  eval-only gate snap lifts wrap-only 0.63 -> 0.90; barrier 11/11. Search, different obstacle from rotary.
- Draft 2 (`docs/paper/DRAFT_workshop.md`) has both folded in.
- Proposed tests (not run; each needs a prereg): AO_bal ring continuation with weight decay 0 (drift account, ~15 min GPU);
  signed with frozen move latents from the |.| start (collective-route account, ~1 GPU-h); toy with observation-retrieval
  target (toy vs transformer difference, CPU); CoPE on a non-wrapping line / sep-keys (~1.6 GPU-h each). Next: Fig. 3 (mechanism), an end-to-end read (rule 28);
  optional: the start manipulation on the torus (does a past-pi start rescue AbsAO4?).

## Last session's work (2026-10-13): existence test done, outline drafted
- WRAP_EXISTENCE (`WRAP_EXISTENCE_RESULTS.md`, registered): E1 EXISTS (all constructions); E2 PRIMARY HELD (balanced
  non-negative construction 8/8 SOLVED, 0/8 LOST vs signed control 8/8); E3 SIGNED LEARNS BETTER (observation-free,
  from scratch 0/8 vs 8/8, p 0.0002). Secondary: held model drifts off the cancelling code (87% -> 18%); wrapped code more
  fragile to 1% noise. Ill-conditioned constructions lose and re-find (descriptive). Rule 14 met for the learnability claim.
- Outline `docs/paper/OUTLINE_workshop.md` with Figures 1-2 (`docs/paper/figs/`). Next if the user wants: a first draft
  (CPU only); optional factorial at n = 8 (~8 h GPU) for a stronger workshop version.

## Ready, not launched (each audited + amended; user chooses)
- (STOPPED 2026-10-11 per review) TW_AMBIG (`TW_AMBIG_PREREG.md`, c6d6a3e): same direction word as action or observation; 64 runs at 1800 ep ~15 h (or
  ~11 h without DirOnlyRole and RoPE2) + ~3 h pilot. Recommended after state-change.
- TW_LANDMARK (`TW_LANDMARK_PREREG.md`, e9cd479): named places vs path integration, 2-layer arms, in-distribution probe
  O-ID; 48 runs ~7 h + pilot.
- RANK_NOWRAP (`RANK_NOWRAP_PREREG.md`, Amendments 1-4, N_SEEDS 12): POSTPONED by the user (cost ~19-23 h); 8 seeds if
  time-limited.
The user is cost-sensitive (a day of GPU is "a lot") and does not want our jobs on the GPUs while another user's jobs
are there; check `nvidia-smi` owners before launching and ask if another user is active.

## Last results (newest first; one line each, numbers in the file)
- 2026-10-13 WRAP_EXISTENCE: exact non-negative ring solution HELD under training 8/8; from scratch 0/8 vs signed 8/8; held model drifts off the cancelling code (secondary). `WRAP_EXISTENCE_RESULTS.md`
- RANK_TESTS (`RANK_TESTS_RESULTS.md`, REG): 1D ring rank 1 = D solves 7/8 (vs R2 8/8): "rank = D is hard" is not general;
  escape test PARTIAL / unmeasured at 600 ep (fresh rank-4 step map on the trained nets also only 2/8). Secondary: lifted
  unsolved runs keep a CLOCK head, fresh step maps are CLEAN.
- TinyStories word-level pilot (`TINYSTORIES_PILOT_RESULTS.md`, PILOT, n=3): MapWM = RoPE (1.5053 vs 1.5050 nats/token);
  MapWM's step is a discourse clock (clock share 0.79-0.87; <eos>, sentence ends, quotes, clause links, speech verbs on
  top; motion verbs 1%, spatial 0%; no opposite pair cancels); uniform clock costs +0.07-0.09 nats. Data in
  data/tinystories (gitignored), word vocab 8192; trainer train_tinystories.py; runs peak ~8.5 GB (2 per 4090).
- TW_STATECHANGE (`TW_STATECHANGE_RESULTS.md`, REG): A PATH NEEDED FOR LOCATION (+0.481); C STATE BOUND TO PLACE
  (T2drop 0.983 / 0.979 vs F2 0.649, every seed); B state verbs OFF the map plane (0.028 moves), not distinguished from
  asides; DirOnly WORSE -0.024 (re-score flag); NormStep 5/8 vs MapWM 7/8 SOLVED, unmeasured. SUMMARY: PREDICTION HOLDS.
- TW_STATECHANGE pilot (Amendment 2, n=1, seed 150): repro bitwise pass; NormStep binds state to place (T2drop 0.998
  vs last-dropped floor 0.649); MapWM descending (T2drop 0.673); DirOnly 0.928; RoPE stalled 0.515; shift_sc < 0.10.
- GAIN_PHASE (`GAIN_PHASE_RESULTS.md`, REG): SEPARATE DEFECTS (leak in theta; step fix removes it, score fix does not);
  THE GAIN-PHASE MAP WORKS (0.9994, 8/8, min 0.998; non-inferior to NormStep); D1r replicates LEAK on SOLVED only
  (accuracy +0.0045, p 0.46); gain score no difference; no speed-up (x1.03).
- Word-class breakdown of text-world models (`docs/audits/2026-10-08/tw_wordclass_out.txt`, POST HOC): only direction-word
  identity carries position (-0.49 when mean-substituted); other classes act as blocks (per-move offsets); adverbs and
  fillers are inert; NormStep keeps position more on direction words.
- GAIN_GRAIN (`GAIN_GRAIN_RESULTS.md`, REG): (a) SCALAR GAIN SUFFICES (per-module too) vs MapPoPE-Pair, NO HEADROOM
  qualifier (MapWM 18/20); (b) softplus(q.k) MapEM WORSE than signed MapEM (12/20 vs 19/20). Scalar gain fastest.
- SCORE_RANK (`SCORE_RANK_RESULTS.md`, REG): NO RESCUE -- PoPE's score leaves rank 2 at 3/12 SOLVED at T=1024 (MapWM
  3/12; r4 8/8 both); it repairs short-gap revisits (+0.115, p 0.036), not wrap-only ones. T1 'SOLVED iff clean' 39/40.
- Post hoc step tables: Indirect Indexing (letters = counter, digits step by value), Bach (pure clock), code (clock on
  whitespace/brackets), Dyck (push/pop; MapPoPE a depth code, MapWM type axes) -- `docs/audits/2026-10-05/*step_table*`.
- Neuro review + design (`docs/theory/2026-10-05/neuro_positional.md`, `neuro_design.md`, ANALYSIS): MapPoPE-Pair tops a
  brain-constraint scorecard; remap probe (post hoc): MapEM pure gain, MapPoPE gain + width (stronger = narrower), MapWM
  gain + skew; rank-2 rescue mechanism (post hoc): PoPE tolerates defective codes, MapWM leans on a content phase offset.
- MapPoPE separated (`MAPPOPE_PAIR_RESULTS.md`, REG): the score rule carries the gain (+0.024, 16/16 vs 10/16 SOLVED at
  r2); the doubled angle count adds +0.0002, CI [-0.0004, +0.0008]. Open: does PoPE's score rescue rank 2 at T=1024?
- NormStep on words (`TW_NORMSTEP_RESULTS.md`, REG): accuracy no detectable difference (+0.006, MDE 0.096); predicted
  word-count clock fired but tiny (+0.057 rad optional-word drift, p 0.027, 0/64 channels), not from the LN bias.
  Post hoc: DirOnly oracle capped at 0.972 by aside objects (100% of its errors); eval mode under-reports runs below
  ceiling (attention dropout; up to +0.16 in train mode) -- UNCHECKED on other batches (next: eval-only CPU audit).
- NormStep analysis (`docs/NORMSTEP_NOTES.md`): scale robustness is by construction; zeroing its obs steps removes a
  per-move gauge, not leak; identity leak ~5x smaller than MapWM's (2 seeds). ANALYSIS.
- Leak remedies (`LEAK_RESULTS.md`, REG): ActOnly and NormStep +0.0107 unseen-object acc in distribution (p 0.0002)
  = MapWM's leak; 16/16 SOLVED vs MapWM 8/8 descending; x4 test unfailable (Amendment 1).
- What/where checks (`docs/WHAT_WHERE_CHECKS.md`, POST HOC): path models do not need the content x position
  interaction; separation without map redraw; leak = pre-LayerNorm step.
- Literature reviews (`docs/lit/LIT_CONTEXT_STEPS.md`, `LIT_WHAT_WHERE.md`, `LIT_NEW_OBJECTS.md`): what is prior art.
- New-object pilot (`runs/newobj_pilot`, n=1): path arms 0.976-0.982 on unseen codes; transfer is prior art.
- Rank x wrap (`RANK_WRAP_RESULTS.md`, REG): 3D grid 18 rank 4 8/8; the 3D shortfall was the small grid; 2D 100-cell
  cell memorised its map.
- 3D rank (`RANK_ND_RESULTS.md`, REG, no branch): rank = D hard in 2D and 3D; failures on wrap-only revisits.
- What/where analysis (`docs/WHAT_WHERE_ANALYSIS.md`, section 6 corrected 2026-10-03). ANALYSIS.
- Context step (`CTXSTEP_HSR_PILOT.md`, PILOT): HSR learns a step 4/4; registered batch STOPPED (`CTXSTEP_PREREG.md`).
- Earlier this week (REG): H1 part 1 (`LOOP_RANK_E1800_P1_RESULTS.md`), text world (`TEXTWORLD_RESULTS.md`), H3
  (`CANCEL_RESULTS.md`), sign matched (`SIGN_MATCHED_RESULTS.md`), rank 3 (`RANK3_RESULTS.md`), code full-val
  (`CODE_FULLVAL_RESULTS.md`).

## Theory review (2026-10-04)
Five-agent review: `docs/theory/2026-10-04/00_PLAN.md` (theory T1-T7 and a ranked plan; reports 01-05). Corrections
applied: TW_NORMSTEP verdict B is the aside offset, not a word clock; the eval-mode gap is the 1/(1-p) attention
scale (CORRECTED block, verified). Plan step 1 (re-score) done: no verdict changes. Next: 1D ring rank 1 vs 2;
rank escape test.

## Open decisions (user picks; costs are wall-clock estimates on both GPUs from comparable batches -- re-measure s/epoch before launching)
1. DONE 2026-10-04: dropout-scale re-score of every committed batch (`docs/audits/2026-10-04/DROPOUT_RESCORE.md`): no
   registered verdict changes; only text-world path arms move. Next from the plan: 1D ring rank 1 vs 2; rank escape test.
2. Separation vs data at matched map size (`LIT_WHAT_WHERE.md` P2) -- 48 runs, ~3.5 h.
3. Window limit as a cue-distance curve (`LIT_CONTEXT_STEPS.md` P3) -- ~10-12 h; Mamba-3-style gate vs HSR (P1) ~13 h;
   HSR decomposition (P2) ~9 h.
4. Leak test that can fail: anisotropic codes + SWAP shift (`LIT_NEW_OBJECTS.md` E2; CPU gate first) -- ~10 h.
5. H1 part 2 (loop and 4-layer rank-2 arms at 1800 ep) -- ~6 h.
6. No GPU: trainer consolidation (train_cancel, train_textworld, train_ctxstep{,2,3}, train_newobj; verify
   loss-exact); documents -- the .tex papers lack rank 3, H1 part 1, H3, text world, context step, 3D rank/wrap,
   what/where, leak, NormStep on words; shared report v17 (`report/language_summary.html`, republish WITH url=) is current.

## Loose ends
- `runs/rank_mi/p0/` is untracked on purpose; `Vanilla_r2ph_s0/s1` there are symlinks into `runs/rank_perhead_pilot/p0/`.
- Checkpoints are the only copy of per-epoch loss curves; do not clear run dirs.
- `CANCEL` eval stream for seed-0 runs overlaps training batch 0; `train_newobj` / ctxstep trainers use eval seed 10**6.
- Killing a trainer orphans its `--data-workers`; after stopping runs check `ps` for parent-1 python3 processes.
- Jericho environment and feasibility outputs live outside the repo: `/home/prashr/jericho_env`, `/home/prashr/jericho_data`.
