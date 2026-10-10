# Results index (regenerated 2026-09-24; rank, documents and statistics updated 2026-09-25; rank separation, Dyck matched depth, loop-rank and catalogue updated 2026-09-27; sign at matched length, rank 3, H1 part 1, H3, text world, code full-val, context-step pilots and catalogue updated 2026-09-30; 3D rank, rank x wrap, leak remedies, what/where checks, NormStep notes, new-object pilot, literature reviews and catalogue updated 2026-10-03)

A catalogue with statuses, not a narrative. **`CLAUDE.md` is the authority** for conventions, the
standing rules (numbered 1-29 there; this file keeps no rule list of its own), the withdrawn list and
the invariants. Live state is `.claude-memory/project_state.md`; history is `docs/LOG.md`. A CORRECTED
or AUDIT block at the top of a results file supersedes its body.

**Status** is from the 2026-09-24 experiment audit (a 58-claim ledger), checked against CLAUDE.md;
every number below was re-read from the file named beside it.
SOLID = holds as stated at its scope; NEEDS CONTROL = real measurement, but the claim outruns it
(usually: read only past the training length or depth, or at a superseded recipe); UNDERPOWERED =
inside a t-based MDE or n <= 4. "OOD" = read past the training length or depth, i.e. robustness until a
matched control exists (CLAUDE.md rule 10). The house "DETECTABLE" (|mean| > 2.8 sd/sqrt(n), i.e. |t| > 2.8)
is a 10.7% false-positive test at n=3 (2.7% at n=8), so an n < 8 DETECTABLE is read with its t-test p
(CLAUDE.md rule 5; e.g. code C1 MapPoPE - PoPE: +0.0034 on the full val file, t-test p 0.19).

Documents: `positional_review.pdf` (review), `axes_measured.pdf` (results paper), `mapformer_math.pdf`
(record), `report/report.pdf` and `report/report_short.pdf`; brought into line with this index on
2026-09-27 (rank separation, Dyck matched depth, torus loop-rank) and corrected 2026-09-30 where they
contradicted the later results (sign at matched length, code full-val). They do NOT carry rank 3, H1
part 1, H3, the text world, the context-step pilots, 3D rank / rank x wrap, the what/where checks or
the leak remedies. `report/language_summary.html` (the shared report source, published as v10 on
2026-09-30) carries rank 3, H1 part 1, H3 and the text world, the context-step pilots in one sentence,
and none of the 2026-10-01..03 results (3D rank / wrap, what/where, leak).

## Citable results

| claim | key number | n | floor / chance | status | file |
|---|---|---|---|---|---|
| Path integration helps on the paper's torus task, at training length | position **+0.243** (MDE 0.038, 8/8); index RoPE 0.805, path 0.971; +0.359 at 8x (OOD). The often-quoted +0.461 is the 16-epoch recipe (index arm on the floor) | 8 | blank floor 0.506 | SOLID at this budget; the index arm was still slowly descending, so its ceiling is unmeasured | `PAPER2X2_RESULTS.md` |
| ...and is necessary for in-context maps (Match-Query) | 0.730 +/- 0.247 vs index 0.154; context destruction 0.918 -> 0.074 | 5 | chance 0.0625 | SOLID; its index control is PlainFlat, never the architecture-matched RoPE arm | `MATCH_QUERY_SCALE.md`, `MATCH_QUERY_RESULTS.md` |
| It is the PER-HEAD rank of the content-to-angle map that decides whether training finds the torus solution (trained and tested at T=1024, 900 ep, every arm built from our r=2's initial weights) | SOLVED within 900 ep, five arms: per-head rank 2 -- our shared r=2 **0/8** (0.894), per-head r=2 **2/8** (0.885), block-diagonal r=4 **2/8** (0.948); per-head rank 4 -- shared r=4 **8/8** (0.998), per-head r=4 **8/8** (0.999). Per-head rank FIRES (D - C_bd, Fisher and perm p 0.0070, Holm 0.028); sharing (D - C, Fisher 1.00, perm 0.59) and W_out per-entry scale (C_bd - B, Fisher 1.00, perm 0.24) UNMEASURED. A rank-2 projection of each solved r=4 scores 0.9955 frozen and is held under training (7/8 vs the r=4 control's 7/8): a SEARCH deficit | 8 | constant floor 0.506 (wrap-only 0.507) | SOLID, budget-scoped (within 900 epochs); r=4's smaller `W_out` init (bound 0.5 vs 0.707) was tested (C_bd - B) and is UNMEASURED, not shown to matter; low initial angle scale is not necessary for failure (A 0.335 / B 0.328 fail like C_bd 0.208); n_heads=2, one task, one length, one recipe; MapWM family only | `RANK_SEP_RESULTS.md`, `RANK_SEP_PREREG.md`, `RANK_MI_RESULTS.md`, `RANK_PROJ_RESULTS.md`, `RANK_MATCHED_RESULTS.md` |
| Search aids partly recover rank 2 but do not reach rank 4 (torus, T=1024, 900 ep) | r=2 + loop x4 (bit-identical params and init to r=2) 2/8 solved, 0.973 (perm p 0.0034 vs r=2); r=2 at 4 real layers 5/8, 0.990 (Fisher 0.0256, perm 0.0012); plain r=2 0/8, 0.894; r=4 8/8, 0.998. 4 real layers beat the loop +0.017 (perm p 0.0009). Final losses form three regimes; no rank-2 run enters r=4's | 8 | constant floor 0.506 | registered H1 verdict UNMEASURED; the 0.05 solved cutoff falls inside the aids' spread (at 0.08 H1's condition would have been met), so uncertain rather than negative; depth (4x params) beats the matched-parameter loop, so NOT search at constant capacity | `LOOP_RANK_RESULTS.md`, `LOOP_RANK_PREREG.md` |
| Rank 3 per head sits with rank 4 (torus, trained and tested at T=1024, 900 ep, built from our r=2's base) | per-head r=3 **6/8** SOLVED, 0.987 at T=1024 (r=2 per head 2/8, 0.885; r=4 per head 8/8, 0.999). 3 - 2: +0.102, perm p 0.027, Holm (2) 0.054; solved 6/8 vs 2/8 Fisher 0.13. 4 - 3: +0.012, p 0.19, UNMEASURED. The two unsolved r=3 seeds sit in the non-cancelling basin (opposition 1.58 / 1.83 vs 0.017-0.076 solved). Reproduction of a stored D seed exact | 8 | constant floor 0.506 | registered RANK 3 SUFFICES at its boundary on every count (accuracy only, Holm 0.054, exactly 6/8); budget-scoped; 2-DOF torus only (a 3D torus is the untested prediction) | `RANK3_RESULTS.md`, `RANK3_PREREG.md`, `RANK3_GEOMETRY.md` |
| H1 part 1: twice the budget does not rescue rank 2 (torus, T=1024, from scratch, 1800 ep) | shared r=2 **0/8** SOLVED (0.894 -> 0.908 from 900 to 1800 ep), shared r=4 **7/8** (0.998 -> 0.994); C - A +0.086, perm p 0.0003, Fisher 0.0014 | 8 | constant floor 0.506 | SOLID as "2x the budget solves none"; NOT "never": 5/8 rank-2 runs still DESCENDING (3 STALLED); one r=4 run stalled at 1800 that solved at 900. Part 2 (loop and 4-layer arms, the registered H1 primary) deferred | `LOOP_RANK_E1800_P1_RESULTS.md`, `LOOP_RANK_E1800_PREREG.md` |
| Per-head rank = D is hard in 2D and 3D; rank D+1 suffices when wrap-around revisits are a minority (N-D torus, one fixed training map per seed, held-out map at eval, trained and tested at T=1024, 900 ep, every arm from the r=2 base) | `RANK_ND`: 2D (grid 32) rank 2 **1/8** (0.769) vs rank 3 **8/8** (0.999), Fisher 0.0014, perm 0.0014; 3D (grid 10) rank 3 **1/8** (0.800) vs rank 4 **4/8** (0.847), Fisher 0.28, perm 0.50 (UNMEASURED). Every failing arm fails on WRAP-ONLY revisits (0.53-0.80; other revisits 0.91-0.97) and partly memorises its training map (+0.09 to +0.17). `RANK_WRAP`: 3D grid 18 (wrap-only share 0.35) rank 4 **8/8**, 1.000 vs grid 10 (0.71) 4/8, 0.847: acc +0.153 (perm p 0.0085; Fisher 0.077). 2D grid 10 (100 cells) memorised its map: own 0.986, unseen 0.273 | 8 | chance ~0.52; retrace floor 0.750 (2D grid 32) / 0.653 (3D grid 10) | `RANK_ND`: no registered branch. `RANK_WRAP`: registered WRAP DRIVES IT, but its 2D half is a memorisation effect, so only the 3D pair is a clean measurement; grid size co-varies with wrap share, cell count, revisit rate and omega's init range; budget-scoped; determinism 16/16 exact | `RANK_ND_RESULTS.md`, `RANK_ND_PREREG.md`, `RANK_WRAP_RESULTS.md`, `RANK_WRAP_PREREG.md` |
| Dyck-2 depth ladder at the training cell (L32 D4), width fixed | position +0.293 / +0.081 / +0.048 / +0.019 at 1-4 layers (8/8); index RoPE 0.979 at 4L | 8 | chance 0.5 | SOLID (matched length and depth) | `DYCK_LADDER_RESULTS.md` |
| Dyck-2 at matched depth: path integration is worth ~3 layers of attention (trained AND tested at L32 D12, A2f) | position main +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers (8/8 each, 1x budget). Mixture training (D in 4..12, 4L) keeps +0.110 at L32 D12 (8/8) and +0.043 at D4; L128 D12 unmeasured | 8 | chance 0.500; best floor 0.594 | SOLID as depth-substitution (parameter efficiency), not as something depth cannot buy: at 4L and 3x budget every arm is at ceiling (index 0.997-0.998, path 1.000), effect +0.002 | `DYCK_MDEPTH_RESULTS.md`, `DYCK_MDEPTH_PREREG.md` |
| The same exchange rate on a biased 1D walk (H3, the cancellation knob; 32-cell ring, T=128, 300 ep, matched length) | path 1 layer 1.000 in every cell (32/32 SOLVED); fewest index layers within 0.01 of it: 3 at p_plus 0.5 / 0.75 / 0.9, 1 at 1.0. Registered primary (index 1L a1(p) = 0.717 / 0.676 / 0.825 / 1.000) is non-monotone: NO registered branch. Fresh seeds s2-s7 (s0-s1 were the pilot): a1 0.710 / 0.677 / 0.824 / 1.000, every pair differs (perm p 0.0022-0.0108) | 8 (6 fresh) | floors 0.554 / 0.544 / 0.520 / 0.498 | exchange rate SOLID at p_plus 0.5 / 0.75; KNIFE-EDGE at 0.9 (2-layer gap 0.0115, fresh 0.0133, vs the 0.01 threshold; 7/8 STALLED); budget-scoped (index 1L and 23/24 index 2L STALLED); path at ceiling, so "one layer" is an upper bound | `CANCEL_RESULTS.md`, `CANCEL_PREREG.md` |
| Index code cannot count contextually (reproduces CoPE) | +0.750 at T=1024 (8/8) | 8 | chance 0.0625 | SOLID (matched length) | `RECENCY_RESULTS.md` |
| Sign of the increment AT MATCHED LENGTH (a replication of Sarrof / Grazzi / Selective RoPE in navigation; torus, trained and tested at T=1024, 900 ep, r=4 shared) | Signed 8/8 SOLVED, 0.998; Abs 0/8, 0.821; Pos (softplus) 1/8, 0.781; RoPE 0/8, 0.731. Abs - Signed **-0.177** (MDE ~0.14, perm p 0.0002; Fisher 0.0002), Pos - Signed -0.218 (p 0.0003). Opposition 0.06 signed vs 1.92-1.97 monotone; no monotone Delta negative after training. Trained at T=128 the in-distribution cost was -0.054, unmeasured | 8 | RoPE 0.731 | SOLID: the first never-controlled OOD claim to get its matched-length control, and it survived. Budget-scoped (monotone arms STALLED / DESCENDING); accuracy is loss here (r -0.992); Pos carries the original's init confound | `SIGN_MATCHED_RESULTS.md`, `SIGN_MATCHED_PREREG.md`, `SIGN_MATCHED_PROBE.md`, `SIGN_ABLATION.md` |
| Navigation told in words: PATH WINS IN WORDS (the torus walk as English, 58 words, synonyms, fillers; held-out map; trained and tested at T=1024 words, 900 ep) | path 1 layer (r=4) **0.969 +/- 0.054** (7/8 SOLVED), RoPE 1L 0.505 (0/8), RoPE 2L 0.772 (0/8); path - RoPE 1L +0.464 (perm p 0.0002, Fisher 0.0014). Fresh seeds s2-s7 only (s0-s1 were the pilot): +0.474, 6/6 vs 0/6, p 0.0022. Step table: registered B did NOT fire (4/8 meet the criterion); declared secondaries: opposite directions cancel on 8/8 after a common component is removed; on 4/8 seeds that component is a real per-step clock (31-38 of 64 phase channels drift between visits vs 2-4) | 8 (6 fresh) | best constant 0.505; reversal-copy rule 0.597 | SOLID for verdict A at this budget (RoPE arms STALLED, creeping); scripted grammar, context-free steps (a direction word never appears outside a movement clause) | `TEXTWORLD_RESULTS.md`, `TEXTWORLD_PREREG.md`, `docs/audits/2026-09-27/fresh_seeds.txt`, `docs/audits/2026-09-27/tw_clock_probe.txt` |
| Removing the what-to-where leak (new-object task: 2D torus 32x32, fresh map and 16 fresh objects per sequence, objects = fixed random codes through a learned encoder, disjoint test pool; T=1024, 900 ep, 1 layer, r=4) | unseen-object accuracy in distribution: MapWM 0.9890 (its leak, steps zeroed - intact, +0.0107), ActOnly (action-only step, oracle token type) 0.9997, NormStep (step reads LN(emb)) 0.9998; remedy - MapWM **+0.0107** for each (perm p 0.0002, MDE ~0.0035, 8/8 vs 8/8). Remedies 16/16 SOLVED (final loss 0.016 / 0.020); MapWM 8/8 still DESCENDING (0.070); r(final loss, x1 acc) -0.944 | 8 | object chance 0.0625 among the 16 in-sequence objects | registered REMEDY for both arms, but the registered x4 test could not fail (both remedies are invariant to code norm by construction; Amendment 1): x2/x4 are construction checks, robustness is not a finding. The x1 gain is at least partly training speed (rule 2). Zeroing NormStep's observation steps (-0.197) measures a per-move gauge, not leak; its object-identity leak is ~5x smaller than MapWM's (2 seeds, `docs/NORMSTEP_NOTES.md`) | `LEAK_RESULTS.md`, `LEAK_PREREG.md`, `docs/NORMSTEP_NOTES.md` |
| NormStep on navigation told in words (text world, 64x64 torus as English, T=1024 words, 1 layer, r=4, 900 ep; MapWM, NormStep, NormStepNB (no LN bias), DirOnly (direction words only), seeds 10-17) | acc MapWM 0.973, NormStep 0.979, NormStepNB 0.941, DirOnly 0.972 +/- 0.001; NormStep - MapWM +0.006 (perm p 0.47, MDE 0.096); optional-word drift 0.077 vs 0.133 rad, +0.057 (p 0.027); NormStep - NormStepNB +0.032 (p 0.17) | 8 | constant 0.512, reversal-copy 0.602 | registered A NO DETECTABLE DIFFERENCE; B WORD-COUNT CLOCK, BIAS NOT SHOWN TO CAUSE IT (tiny: 0/64 channels > 1 rad); composite DOES NOT CARRY OVER CLEANLY. Post hoc: DirOnly's errors are all aside objects at the same cell; eval mode under-reports runs below ceiling (attention dropout) | `TW_NORMSTEP_RESULTS.md`, `TW_NORMSTEP_PREREG.md` |
| MapPoPE's score rule vs its frequency count (paper torus T=128, 1 layer; MapWM / MapPoPE at 32 angles (pairwise) / MapPoPE at 64 angles; rank 2 n=16 seeds 10-25, rank 4 n=8) | r2: MapWM 0.9752 (10/16 SOLVED), pairwise 0.9995 (16/16), 64-angle 0.9997 (16/16); SCORE +0.0243 (p 0.012), COUNT +0.0002 (95% CI [-0.0004, +0.0008]), TOTAL +0.0244 (p 0.009); r4 all 1.000 | 16 / 8 | best n-gram 0.598, always-blank 0.507 | registered SCORE RULE; angle count bounded near zero; it does not explain MapPoPE's small rank-4 gain (OOD secondary) | `MAPPOPE_PAIR_RESULTS.md`, `MAPPOPE_PAIR_PREREG.md` |
| Does PoPE's score rescue per-head rank 2 at T=1024? (torus, 900 ep, 1 layer; MapWM vs MapPoPE-Pair at r2 (n=12) and matched-init r4 (n=8), 32 angles) | SOLVED r2 3/12 vs 3/12, r4 8/8 vs 8/8; acc r2 0.848 vs 0.946 (+0.098, p 0.089, MDE 0.155); strata: short-gap +0.115 (p 0.036), wrap-only -0.015 | 12 / 8 | blank 0.507, n-gram 0.576, retrace 0.843 | registered NO RESCUE; PoPE fixes the local map not the periodic code; T1 'SOLVED iff clean head' 39/40 (declared secondary) | `SCORE_RANK_RESULTS.md`, `SCORE_RANK_PREREG.md` |
| Gain granularity between MapEM and MapPoPE (paper torus T=128, rank 2, 32 angles; MapWM, MapPoPE-Pair, GainScalar, GainMod4, MapEM, MapEM softplus(q.k); n=20, seeds 26-45) | acc / SOLVED: W 0.990 18/20, P 0.9995 20/20, S 0.9999 20/20, M 0.9997 20/20, E 0.997 19/20, N 0.982 12/20; S - P +0.0004, M - P +0.0002 (non-inferior at -0.01, p < 0.0001); N - E -0.015 (p 0.026), Fisher p 0.020 | 20 | best n-gram 0.598, always-blank 0.507 | registered (a) SCALAR GAIN SUFFICES with NO HEADROOM qualifier, (b) NON-NEGATIVE WORSE (softplus on q.k); scalar gain trains ~3x faster (descriptive) | `GAIN_GRAIN_RESULTS.md`, `GAIN_GRAIN_PREREG.md` |
| The gain-phase map: STEP (raw / NormStep) x SCORE (rotary / gain) on the new-object task (T=1024, rank 4, 32x32 torus; n=8, seeds 8-15) | unseen-object acc / SOLVED: MapWM 0.9895 0/8, NormStep 0.9940 8/8, GainRaw 0.9906 0/8, GainPhase 0.9994 8/8 (min 0.998); leak L_ms median +0.0069 / +0.0002 / +0.0100 / +0.0001; GainPhase - MapWM +0.0099 (p 0.0002); GainPhase - NormStep +0.0054 (p 0.135, non-inferior at -0.005) | 8 | retrace 0.518, last object 0.141 | registered D3 SEPARATE DEFECTS, D4 THE GAIN-PHASE MAP WORKS, D1 NORMSTEP HELPS UNDER BOTH SCORES (D1r on SOLVED only), D2 NO DIFFERENCE, D5 NO DIFFERENCE (no speed-up) | `GAIN_PHASE_RESULTS.md`, `GAIN_PHASE_PREREG.md` |
| State changes told in words: text world + take/drop sentences (T=1024, 1 layer; MapWM, NormStep, DirOnly, RoPE 1L; n=8, seeds 50-57) | all / T2drop / SOLVED: MapWM 0.994 / 0.983 / 7/8, NormStep 0.961 / 0.979 / 5/8, DirOnly 0.970 / 0.926 / 8/8, RoPE 0.516 / 0.011 / 0/8; state sentence shift 0.028 / 0.027 moves | 8 | constant 0.514; T2drop F1 0.211, F2 0.649 | registered A PATH NEEDED FOR LOCATION, C STATE BOUND TO PLACE (both), B OFF THE MAP PLANE (not distinguished from asides), D1 NO DIFFERENCE, D2 DirOnly WORSE (flag) | `TW_STATECHANGE_RESULTS.md`, `TW_STATECHANGE_PREREG.md` |
| Loop on path integration (Match-Query) | loop main effect **unpaired** +0.346 (t 3.75); loop arm pooled 0.803 +/- 0.200, 1/16 failures | 8 / 16 | chance 0.0625 | SOLID for the main effect. The paired interaction +0.315 and "r=4 + loop x4 0.986, 8/8 >= 0.941" are paired / one-batch statistics on a task whose same-seed retrains drift 0.185: CONTRADICTED (see Withdrawn) | `REFINE_RESULTS.md`, `LOOP_HEADROOM.md`, `MQ_RANK_2X2.md` |
| EM's recency deficit is search | EM - WM -0.375 (0/8, MDE 0.154); installed rewind frozen 1.000 (8/8); per-pair origins +0.215 = pathway +0.124 + freedom +0.091 (n=48) | 8 / 48 | chance 0.0625 | SOLID as a fixed-budget learnability result | `RECENCY_EM_RESULTS.md`, `WARM_RESULTS.md`, `SEARCH_RESULTS.md`, `PAIRSPLIT_RESULTS.md`; `EM_WM_STATE.md` Sec 3-4 gives the status of every EM/WM file |
| Phase freedom in q0/k0 | +0.146 vs a matched-optimiser control (22/24); fresh seeds +0.113 | 24 | -- | SOLID; mechanism unidentified | `MAGONLY_RESULTS.md`, `D5_RESULTS.md` |
| Paper replications | MapFormer v4 Dyck-2: ordering MapWM-1L - RoPE-2L +0.370 (8/8, F1; on Hewitt closing accuracy +0.064, 0.638 vs 0.574 at L128 D12, same direction -- that cell is 3x the training depth and 4x its length, i.e. depth-extrapolation, and the 4L depth effect closes at matched depth, `DYCK_MDEPTH_RESULTS.md`); levels do not replicate and sit at the F1 floor 0.884. PoPE's Indirect Indexing at 200k iters 7/8 (1/8 at their 100k). PoPE's Bach: PoPE - RoPE -0.032 NLL (5/5) | 8 / 8 / 5 | F1 no-stack 0.884 | SOLID as replications; the Dyck ordering is read OOD in depth (robustness) | `DYCK_RESULTS_bs128.md`, `INDIRECT_RESULTS_200k.md`, `JSB_RESULTS.md` |
| Our PoPE is faithful; non-negativity is not what extrapolates | 1.7e-06 max logit difference vs the authors' code; NoSigma penalty -0.0108 vs RoPE +3.5885; 80.7% of `pope_delta` frozen in its clamp | 3 | batch floor 0.0021 / 0.0028 bpc | SOLID as corrected (the NoSigma Table-5 cell is unmeasured) | `ABLATE_RESULTS.md` |
| PoPE's encoding helps the path row (Bach) | MapPoPE - MapWM -0.0165 NLL (5/5, MDE 0.0111) | 5 | -- | SOLID on Bach. Dyck 2L +0.050 is depth-OOD (trained D4, read at D12; at matched depth L32 D12 both arms are at 1.000); code on the full val file -0.0054 (t p 0.086) and MapPoPE - PoPE +0.0034 (t p 0.19) are UNMEASURED (n=3) | `JSB_RESULTS.md`, `.claude-memory/project_mappope_asymmetry.md` |
| Code: the OOD encoding "win" is extrapolation cost | at matched 2048, rescored on the full val file, the encoding effect is -0.0033 (1/3 seeds, t p 0.31, UNMEASURED) against -3.694 extrapolating from 512 | 3 | no-memory brackets 0.858 | SOLID as a retraction | `CODE_FULLVAL_RESULTS.md`, `CODE_RESULTS.md`, `CODE_GATES.md` |
| Code at matched length, full-val rescore (36 `.final.pt` checkpoints scored on the whole val file instead of `best_val_bpc`) | no sign flips. Keep a t-test: C1 position main +0.0056 (path integration costs at 2048; 3/3, p 0.024); C2 PoPE-Decay - RoPE-Decay +0.0054 (p 0.002); envelope on MapPoPE -0.0059 (p 0.008) and MapWM -0.0170 (p 0.036). UNMEASURED: encoding main, MapPoPE - PoPE (+0.0034, p 0.19), MapPoPE - MapWM, MapPoPE-Decay - PoPE-Decay, "RoPE-Decay best of eight", envelope on RoPE (cross-batch) and PoPE | 3 | -- | n=3: significant rows are t-test p < .05 only, below the n where any distribution-free test can reach .05; envelope rows are cross-batch. Full-val sits +0.048-0.051 above `best_val_bpc` at 2048 and 0.024-0.035 below it at 512 (unexplained); do not compare absolute bpc across the two readouts | `CODE_FULLVAL_RESULTS.md`, `CODE_FULLVAL_PREREG.md` |
| Recipe beats architecture on the compositional task | warmup + cosine +0.160 (7/8) | 8 | floor 0.072 | SOLID (hierarchy's +0.136 on the same task is UNDERPOWERED) | `COMP_HEADROOM.md`, `HIER_RECHECK.md` |
| Parallel scan | 2.6-3.3x over a 16x length increase; MapEM-NC 14.5x; TEMFaithful 120x | -- | -- | SOLID | `TIMING_BENCHMARK.md` |
| CSCG stitching control reproduces | paired +0.131 +/- 0.024 vs index -0.005 | 3 | exactly 0 | SOLID, n=3 | `STITCH_ATTENTION.md` |
| Metric and data findings | Dyck F1 has a 0.88 no-stack floor (n-gram 0.857 at the hardest cell; use Hewitt closing accuracy); Bach is overfitting-limited: transposition 0.107 NLL vs the 0.032 PoPE-RoPE gap | -- / 5 | -- | SOLID | `DYCK_LITERATURE_METRICS.md`, `AUG_RESULTS.md` |
| Robustness repairs (OOD, labelled as such) | 48-parameter decay envelope: Bach MapPoPE 4.616 -> 0.622, PoPE 1.597 -> 0.626 NLL at 2-4x (5/5); MapWM - RoPE -0.662 at 2-4x beyond a 512 context | 5 | -- | SOLID as robustness | `DECAY_RESULTS.md`, `JSB_LENGTH_RESULTS.md` |
| Hierarchy on text is efficiency only | 1.4537 vs 1.4506 bpc at parameter parity; 1.23x throughput, -14.1% peak memory | 1 | checkpoint sd 0.003-0.007 | a null at n=1 (consistent with, not proof of) | `ENWIK8_HIERARCHY.md` |

## Listed as citable in CLAUDE.md, but NEEDS CONTROL

(The sign row left this table 2026-09-30: its matched-length control was run and it survived; see the citable table.)

| claim | key number | what is missing | file |
|---|---|---|---|
| Clock/map crossover | monotone costs -0.280 on the torus, -0.004 on recency; magnitude-matched content increment +0.594 (8/8) | both halves are extrapolation readouts at different ratios | `RECENCY_RESULTS.md`, `RECENCY_GATE_ABLATION.md` |
| Map extent is a threshold | -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells, matched aliasing | the index arms behind +0.305 and +0.015 were still DESCENDING under a flat-slope "converged" label | `ALIASING_CONTROLLED.md`, `VISITS_TEST.md` |
| Rotation actions; allocentric recoding fixes it | +0.050 -> +0.488 (8/8); 12 headings +0.26..+0.38 (bimodal budget curve) | all at the 16-epoch recipe with the index arm on the 0.508 floor; rerun under the converged recipe | `KNOB_SWEEP_n8.md`, `ALLOCENTRIC_RECODING.md` (n=3), `H12_BUDGET_CURVE.md` |
| Decay envelope vs long-range retrieval | steepness +0.136 and metric +0.145 of +0.281 (8/8) | residual collinear with convergence (r = -0.995, non-overlapping losses) | `CROSS_RESULTS.md` |

## Open: underpowered, OOD-only or unfinished

- **Rank, the old numbers.** r=4 +0.085 at T=1024 when trained at T=128 is OOD only (94% from
  short-gap revisits late in the sequence; `RANK_SWEEP.md`); on Bach the order inverts (r=1 best,
  `JSB_LENGTH_RESULTS_RANK.md`). The matched-length result is now citable (table above). The
  registered matched-length verdicts of `RANK_MATCHED_RESULTS.md` (900 ep, and 900 + 900 with a warm
  restart) are UNREADABLE (four r=2 runs still descending); `RANK_SEP_RESULTS.md` (with `RANK_MI_RESULTS.md`)
  is the readable test, and it separates the cause: per-head rank.
  "r=2 loses because its basis is skewed" is withdrawn: within r=2 skew does not predict accuracy, and a
  rank-2 solution exists. Our bottleneck is shared across heads, the paper's is per head (2 heads: our
  r=2 has 2 latent dims, the paper's 4).
- **Level 1.5 / InEKF**: loss-matched +0.062 / +0.124 at T=512/1024, OOD only; no matched-length arm
  (`L15_ABLATION.md`).
- **Forget gate** +0.086 at r=2, T=1024, OOD only, mechanism unidentified; the forget-clock batch was
  deleted and, as registered, reads OOD lengths only (`FORGET_GATE.md`, `FORGET_CONTROL.md`,
  `FORGET_CLOCK_PREREG.md`).
- **PoPE wrapping**: the length half holds 3/3, OOD only (`POPE_WRAPPING.md`).
- **EM_P0 - WM on the paper task** +0.035 / +0.070 / +0.085 at l=512/1024/2048, OOD only
  (`EM_WM_THEORY.md`).
- **"The loop's torus gain is convergence, not representation"** is a loss-matched residual at
  matched length, the inference withdrawn for rank; the LoopedSampled count-curve flattening itself is
  solid (`L15_LOOP_2X2.md`, `LOOP_SAMPLED.md`).
- **Code C1 "reversal"** (MapPoPE - PoPE +0.0034 full-val, t p 0.19) and the UNMEASURED C2 / envelope
  contrasts: rescored on the full val file 2026-09-27 (`CODE_FULLVAL_RESULTS.md`, row above); no sign
  flipped, n=3 throughout. More seeds are the only way to read them.
- **Context-dependent step (pilots, NOT results)** (`CONTEXT_STEP_DESIGN.md`, `CTXSTEP_PILOT1.md`,
  `CTXSTEP_PILOT2.md`, `CTXSTEP_PILOT3.md`, `CTXSTEP_HS_RECIPE.md`): the text world with decoy uses of
  direction words. Pilot 1 leaked a trailing cue; pilot 2's predicted lead/trail dissociation did not
  happen (both window steps use either side). Pilot 3 (n=1 per cell): window-limited steps (context
  gate, Selective-RoPE generator) suppress decoys with the cue 1-3 tokens away (ratio 0.01-0.42, acc
  0.97-1.00) and fail with it 6-13 away (ratio 0.94-1.00, acc at the context-free level) -- ready to
  pre-register. Hidden-state step: every run that learned a step ignored far decoys (5/5, ratio
  0.01-0.10) but 5/10 far-cue runs never learned one. The fix HSR (step from emb + alpha LN(h1), alpha
  init 0; `CTXSTEP_HSR_PILOT.md`, n=2 per cell) learned a step 4/4 and ignores far decoys (0.966-0.998).
  The registered batch (`CTXSTEP_PREREG.md`) was STOPPED 2026-10-01 before any result, for cost (~31 h)
  and because its window hypothesis is near-guaranteed by construction. The SR arm is not one-knob (full
  rank, no omega). Seeds 0, 1, 2, 3, 5 are used. Prior art: `docs/lit/LIT_CONTEXT_STEPS.md`.
- **MiniGrid 2x2x2**: index arms best (0.955 / 0.953), MapWM last (0.823); convergence unreported
  (`MINIGRID_FULL_2X2X2.md`).
- **Indirect Indexing**: path integration faster (6/7, directional) and more padding-robust (0.430 vs
  0.149, uncontrolled) (`INDIRECT_RESULTS_200k.md`, `INDIRECT_OOD.md`).
- **Family tree** non-commutativity +0.013 (MDE 0.008 z-based; fails the t-based MDE at n=3)
  (`N3_AUDIT.md`, `FAMILY_TREE_RESULTS.md`).
- **COUNTER** (n=4): with an identical installed counter MapWM 1.000, EM 0.740, TEM 0.337
  (`COUNTER_RESULTS.md`).
- **Addition**: signed MapFormer is the only learned code to learn 30 digits (3/3), holds to ~33-38;
  the control gate FAILED (role-format oracle 0.697) (`SAMEBLOCK_RESULTS.md`).
- **Map-Query**: the room query is learnable at 7.6x chance on one variant and one seed; the
  multi-seed table is 25-epoch undertrained (`MAP_QUERY_GATES.md`, `MAP_QUERY_RESULTS.md`).
- **lm200, corrected**: Level15 0.990 vs Vanilla 0.742 stands as numbers; the interpretation is
  withdrawn (a filter-free capacity control ties it), and lm200 never had a context-destruction
  ablation (`LM200_CORRECTED_MULTISEED.md`, `EXTRAHEAD_CONTROL.md`).

## Post hoc analyses, pilots and literature (no registered verdict)

| what | status | key numbers | file |
|---|---|---|---|
| What/where checks: does a trained path model NEED the content x position interaction; is separation forced by map redraw; where does "what" leak into "where" | POST HOC, eval-only on stored checkpoints, each re-run byte-identical | paper torus (`runs/paper2x2/p0`, 8 seeds x 6 arms): object identity removed from the score, path models keep 0.837 (MapWM r2) / 0.974-0.989 (converged arms; cost 0.011-0.025, below the MDE); a shared kernel x content gain gives 0.988-1.000 for those; RoPE / PoPE fall below the 0.598 n-gram floor. Never-redrawn 32x32 map separates like the redrawn torus; a memorised 100-cell map has no relational "where". New-object pilot: zeroing object steps lifts unseen-object accuracy 0.990-0.993 -> 0.9996-0.9999; x2 / x4 code norm costs 0.04 / 0.11-0.14 (n=1) | `docs/WHAT_WHERE_CHECKS.md`, `docs/audits/2026-10-03/` |
| What/where analysis: the attention score of every positional scheme, from the code; separation probe | ANALYSIS + descriptive probe, post hoc; section 6 CORRECTED 2026-10-03 (key-step phase error; converged arms unchanged) | trained 1-layer path models put most score variance in one shared displacement kernel (interaction share MapWM r4 0.083, MapPoPE r4 0.128, MapEM r4 0.104); PoPE separates at init, not more at the end of training | `docs/WHAT_WHERE_ANALYSIS.md` |
| NormStep notes: definition, scale theorem, per-move gauge, identity leak | ANALYSIS; gauge and leak checked on weights of seeds 0 and 3 (`docs/audits/2026-10-03/normstep_gauge.py`) | robustness to code norm is a theorem (LN is scale-invariant); opposite actions plus the shared observation step cancel to 0.001 of an action step; object-identity leak spread NormStep 0.0007-0.0009 vs MapWM 0.0038-0.0041 of an action step; not provable that NormStep must leak less (W A = 0 is feasible for MapWM). Predicted risk on language: the LN bias step becomes a word-count clock | `docs/NORMSTEP_NOTES.md` |
| New-object transfer pilot (fixed random codes, disjoint test pool) | PILOT, n=1 per arm, seed 100 | unseen-code accuracy MapWM 0.982, MapPoPE 0.977, MapEM 0.976, PosOnly 0.976 vs RoPE 0.441, PoPE 0.394 (all revisits, T=1024). The transfer could not fail: unseen iid codes through a shared linear encoder are known to transfer (Chen et al. 2019) | `runs/newobj_pilot/*/eval.json`, `docs/lit/LIT_NEW_OBJECTS.md`, `docs/audits/2026-10-03/e0_table.md` |
| Context-dependent step (decoy direction words) | PILOTS, n=1-2 per cell; registered batch STOPPED before any result | see the context-step bullet under Open | `CTXSTEP_PILOT1-3.md`, `CTXSTEP_HS_RECIPE.md`, `CTXSTEP_HSR_PILOT.md`, `CTXSTEP_PREREG.md`, `CONTEXT_STEP_DESIGN.md` |
| Literature reviews (~85 sources, read depth marked per paper) | REVIEWS, not results | context-gated / window / hidden-state step generators and identity-at-init are published (Mamba-1/2/3, Selective RoPE, PaTH, RWKV-7, ReZero, Flamingo); "separation is learned" is in MapFormer's own Fig. 9; transfer to unseen iid codes is Chen 2019; TEM never tested unseen objects. Possibly ours: cue distance x generator source on a signed path phase; the causal score form in path models; a quantified content-to-step leak. Each file ends with costed proposals | `docs/lit/LIT_CONTEXT_STEPS.md`, `docs/lit/LIT_WHAT_WHERE.md`, `docs/lit/LIT_NEW_OBJECTS.md` |

## Live negatives and withdrawn claims

**Live negatives** (do not re-run): CLAUDE.md, "Live negatives". **Withdrawn** (do not cite):
CLAUDE.md, "Withdrawn -- do not cite", plus `archive/void/` (54 files, each bannered) and
`archive_stale/` (35 files).

Closed 2026-09-26 (formerly NEEDS CONTROL here; on CLAUDE.md's withdrawn list): the Dyck ladder's
position effect at L32 D12 (+0.168 at 4 layers, trained at D4) as a capability result, and "the index
arms plateau / depth closes 40% of the gap then stops". Trained at D12, every 4-layer arm reaches ceiling
at 3x budget (+0.002); the 1x budget limits the index arms (3x - 1x = +0.021, 8/8). What survives is the
matched-depth row above (`DYCK_MDEPTH_RESULTS.md`).

Contradicted by the 2026-09-24 audit and **not yet on CLAUDE.md's withdrawn list**:
- Loop x path integration "super-additive" (paired interaction +0.315), "`r=4 + loop x4` 0.986,
  8/8 >= 0.941" and "the loop raises the floor, 8/8 >= 0.77": paired or one-batch statistics on a task
  whose same-seed retrains drift 0.185 per seed (`REFINE_RESULTS.md`). CLAUDE.md's loop row now marks
  the interaction and "never fails" withdrawn and the 0.986 as one batch.
- "The encoding moves the torus result ~0.003 / PoPE is inert without path integration (0.509)":
  16-epoch recipe. Converged, the encoding main effect is detectable at every length (-0.049 / +0.114
  / +0.189, `PAPER2X2_RESULTS.md`).
- "Index models exceed the floor only at recurrence interval 1-2" (`REVISIT_DISTANCE.md`): under the
  converged recipe index RoPE leads index PoPE at 5-16-step revisits (`REVISIT_2X2.md`).
- The shared report's "on text, code and music, plain PoPE still wins" (its own code row and
  `ABLATE_RESULTS.md` say otherwise) and its unsourced "six cases" sentence.
- Indirect Indexing "0.965 against the paper's 0.948" as like for like: 0.965 is the mean among
  solvers; the paper's figure is over all runs.

## Two readout notes carried over from the previous index (not rules)

- **`shuffle` and `resample` are not interchangeable** in a context-destruction ablation: permuting
  slots destroys the walk's autocorrelation and puts the input off-manifold; substituting the stream
  from an independent episode does not. Report both (`ABLATE_COMPOSITIONAL.md`).
- **An ablation landing BELOW the floor** means the model fails confidently rather than hedging. Check
  with an on-manifold resample before blaming the manipulation.

## Catalogue of results files, by line

Top-level `*.md` (510 files) plus the docs-level notes; regenerated 2026-10-09 with `docs/tools/catalog_results_index.py` (zero unclassified). `*` = a CORRECTED / RETRACTED / WITHDRAWN / SUPERSEDED / VOID /
STALE marker in the first 12 lines (a correction block further down also supersedes the body).
Files starting with `_` are raw per-seed dumps; names ending `_PREREG` are pre-registrations and
`_GATES` task gates.

**Torus paper task, recipe and reproduction** (54)

`ALLOCENTRIC_RECODING`, `AUDIT_HEADLINE`, `BASELINE_TABLE`*, `CLOCK_SCAN`, `DETAILED_RESULTS`, `DRIFT_PROBE`, `FREQ_CONTROL`, `GENERALIZATION_REPORT`, `H12_BUDGET_CURVE`, `HORIZON_L1d128e16`, `HORIZON_L1d128e50`, `HORIZON_L2d128e16`, `HORIZON_L2d256e16`, `HORIZON_L4d128e16`, `HORIZON_L4d256e16`, `HORIZON_RESULTS`, `HORIZON_TASK_DISTANCES`, `INDEX_BASELINE_PAPER_TASK`, `INDEX_BASELINE_PAPER_TASK_n8`, `KNOB_SWEEP`, `KNOB_SWEEP_n8`, `LONG_SEQ_clean`, `N3_AUDIT`, `NOISE_CLEAN_REVALIDATION`, `OMEGA_RESCALE_clean`, `OOD_GRID_RESULTS`, `PAPER2X2_PREREG`, `PAPER2X2_RESULTS`, `PAPERTASK_PREREG`, `PAPERTASK_RESULTS`, `PAPER_OOD_EXTENDED`, `PAPER_OOD_EXTENDED_n8`, `PAPER_OOD_PROTOCOL`, `PAPER_OOD_RERUN`, `PAPER_OOD_WITH_POPE`, `PAPER_TASK_ABLATION`, `PAPER_TASK_ACCURACY`*, `PAPER_TASK_FLOORS`, `PAPER_VALIDATION`, `PERSCALE_OMEGA_RESULTS`, `PER_VISIT_clean`, `RECIPE_POWER`, `REVISIT_2X2`, `REVISIT_DISTANCE`, `ROPE_CANONICAL`, `ROPE_CONVERGE`, `TIMING_BENCHMARK`, `TOPOLOGY_RESULTS`, `ZERO_SHOT_TRANSFER_clean`, `ZERO_SHOT_TRANSFER_clean_brokeninit`, `_PAPER2X2_RAW`, `_RECIPE_C0`, `_RECIPE_C1`, `_RECIPE_C2`

**Rank, generator and accumulator** (73)

`ACCUMULATOR`, `ACTION_GEOMETRY`*, `CONV_KERNEL_PROBE`, `DXR_PRELIM`, `DXR_RANK_THRESHOLD`, `FAST_ATTN_RANK`, `GAIN_GRAIN_EVAL`, `GAIN_GRAIN_PREREG`, `GAIN_GRAIN_RESCORE`, `GAIN_GRAIN_RESULTS`, `GATE_PROBE`, `LEARNED_RANK`, `LOCALISATION`, `LOCALISATION_PREREG`, `LOCALISATION_RANK`, `MAPPOPE_PAIR_PREREG`, `MAPPOPE_PAIR_R2`, `MAPPOPE_PAIR_R4`, `MAPPOPE_PAIR_RESULTS`, `MAPPOPE_R4`, `MAPPOPE_R4_PREREG`, `MAPPOPE_R4_RESULTS`, `ND_GATES`, `PAPER_FIG4_EM`, `PAPER_FIG4_REPRO`, `RANK3`, `RANK3_GEOMETRY`, `RANK3_PREREG`, `RANK3_RESULTS`, `RANK_MATCHED`, `RANK_MATCHED_GEOMETRY`, `RANK_MATCHED_PREREG`, `RANK_MATCHED_RESULTS`, `RANK_MATCHED_e900`, `RANK_MATCHED_e900_GEOMETRY`, `RANK_MATCHED_e900c`, `RANK_MATCHED_e900c_GEOMETRY`, `RANK_MI`, `RANK_MI_GEOMETRY`, `RANK_MI_PREREG`, `RANK_MI_RESULTS`, `RANK_ND`, `RANK_ND_PREREG`, `RANK_ND_RESULTS`, `RANK_NOWRAP_PREREG`, `RANK_PERHEAD_PILOT`, `RANK_PERHEAD_PILOT_GEOMETRY`, `RANK_PERHEAD_PILOT_RESULTS`, `RANK_PERHEAD_PREREG`, `RANK_PROJ_FROZEN`, `RANK_PROJ_PREREG`, `RANK_PROJ_RESULTS`, `RANK_PROJ_TRAIN`, `RANK_PROJ_TRAIN_GEOMETRY`, `RANK_SEP`, `RANK_SEP_GEOMETRY`, `RANK_SEP_PREREG`, `RANK_SEP_RESULTS`, `RANK_SWEEP`*, `RANK_TRUNCATION`, `RANK_WRAP_PREREG`, `RANK_WRAP_RESULTS`, `SCORE_RANK_PREREG`, `SCORE_RANK_R2`, `SCORE_RANK_R4`, `SCORE_RANK_RESCORE_R2`, `SCORE_RANK_RESCORE_R4`, `SCORE_RANK_RESULTS`, `SELECTIVE_ROPE`, `THEORY_NARRATIVE`*, `THEORY_NUMBERS`, `THEORY_SEARCH_AND_LENGTH`, `_SELECTIVE_TORUS`

**Sign, clock/map and recency** (37)

`COUNTER_BATCH`, `COUNTER_RESULTS`, `FLIPFLOP_GATES`, `FLIPFLOP_RESULTS`, `FORGET_CLOCK_PREREG`, `FORGET_CONTROL`, `FORGET_GATE`, `GATED_PREREG`, `GATED_RESULTS`, `GATED_SEPARATION`, `GATED_TORUS`, `LAMBDA_TRACE`, `MONOTONE_PREREG`, `MONOTONE_RAW`, `MONOTONE_RESULTS`, `MQAR_PREREG`, `MQAR_RESULTS`, `RECENCY_GATES`, `RECENCY_GATES_K16SET`, `RECENCY_GATES_K4SET`, `RECENCY_GATES_K64`, `RECENCY_GATE_ABLATION`, `RECENCY_H2`, `RECENCY_PREREG`, `RECENCY_RESULTS`, `SIGN_ABLATION`, `SIGN_ABLATION_PREREG`, `SIGN_MATCHED`, `SIGN_MATCHED_PREREG`, `SIGN_MATCHED_PROBE`, `SIGN_MATCHED_PROBE_SIGNED`, `SIGN_MATCHED_RESULTS`, `SIGN_PROBE`, `TEM_RECENCY_DIAG`, `TEM_RECENCY_PILOT`, `_MONOTONE_TORUS`, `_SIGN_RAW`

**EM vs WM and the position kernel** (49)

`AP_KERNEL_DIAGNOSTIC`, `AUDIT_2026-09-10`, `D5_PREREG`, `D5_RESULTS`, `DOF_PREREG`, `DOF_RESULTS`, `EM_COMP_SAMEBATCH`, `EM_FIX_COMP`, `EM_HOPFIELD_CROSSSCALE`, `EM_P0_COMP`, `EM_P0_PAPER`, `EM_WM_STATE`*, `EM_WM_THEORY`, `HOPFIELD_NOMAINAP_RESULTS`, `MAGONLY_PREREG`, `MAGONLY_RESULTS`, `MATCH_QUERY_EM`, `MINIGRID_EM`, `MINIGRID_EM_FIX`, `MINIGRID_EM_FIX_PREREG`, `MINIGRID_EM_PREREG`, `N5_PREREG`, `N5_RESULTS`, `NOLEAK_PREREG`, `NOLEAK_RESULTS`, `PAIRCONST_PREREG`, `PAIRCONST_RESULTS`, `PAIRORIGIN_PREREG`, `PAIRORIGIN_RESULTS`, `PAIRSPLIT_PREREG`, `PAIRSPLIT_RESULTS`, `RECENCY_EM_RESULTS`, `REC_EM_PREREG`, `SEARCH_PREREG`, `SEARCH_RESULTS`, `SPREAD2_PREREG`, `SPREAD2_RESULTS`, `SPREAD_PREREG`, `SPREAD_RESULTS`, `TALE_OF_TWO_ALGORITHMS`, `THEORY_KERNEL`, `UNFREEZE_PREREG`, `UNFREEZE_RESULTS`*, `VOCAB_EM`, `VOCAB_EM_PREREG`, `WARM_PREREG`, `WARM_RESULTS`*, `_DOF_TORUS_RAW`, `_N5_TORUS_RAW`

**Match-Query, loop and algorithmic tasks** (45)

`ADDITION_CHO_REPRO`, `ADDITION_DESIGN`, `ADDITION_GATES`, `ADDITION_PILOT`, `ADDITION_PILOT2`, `ALGORITHMIC_GATES`, `ALGORITHMIC_RESULTS`, `FRONTIER_ALGORITHMIC`, `HIER_PARITY`*, `L15_LOOP_2X2`, `LOOPED_L1`, `LOOPED_L4`, `LOOPED_Loop4`, `LOOPED_PILOT`, `LOOP_DEPTH_STRATA`, `LOOP_HEADROOM`, `LOOP_HIER_COMPUTE`, `LOOP_HIER_PARITY`, `LOOP_RANK`, `LOOP_RANK_E1800_P1`, `LOOP_RANK_E1800_P1_RESULTS`, `LOOP_RANK_E1800_PREREG`, `LOOP_RANK_GEOMETRY`, `LOOP_RANK_PREREG`, `LOOP_RANK_RESULTS`, `LOOP_SAMPLED`, `MATCH_GATES_128_16`, `MATCH_GATES_64_16`, `MATCH_GATES_64_4`, `MATCH_QUERY_GATES`*, `MATCH_QUERY_GATES_P010`, `MATCH_QUERY_LONGQ`, `MATCH_QUERY_NOISE_ABLATION`, `MATCH_QUERY_RESULTS`, `MATCH_QUERY_SCALE`, `MQ_NOISE_2X2`, `MQ_NOISE_2X2_C2`, `MQ_RANK_2X2`, `RECURSIVE_RESULTS`, `REFINE_RESULTS`, `SAMEBLOCK_COMPILE_CHECK`, `SAMEBLOCK_PREREG`, `SAMEBLOCK_RAW`, `SAMEBLOCK_RESULTS`, `_L15_LOOP_RAW`

**Hierarchy, compositional and planner tasks** (35)

`ABLATE_COMPOSITIONAL`, `AGGREGATE_EXTRAS`, `AGGREGATE_MULTISEED`, `AGGREGATE_TASK_RESULTS`, `BOUNDED_MEMORY`, `BOUNDED_MEMORY_RESULTS`, `COMPOSITIONAL_EXPERIMENT`, `COMPOSITIONAL_MATCH_QUERY`, `COMPOSITIONAL_MATCH_QUERY_CURRIC`, `COMPOSITIONAL_MATCH_QUERY_GATES`, `COMPOSITIONAL_MATCH_QUERY_RESULTS`, `COMPOSITIONAL_MATCH_QUERY_STAB`, `COMPOSITIONAL_MULTISEED`, `COMPOSITIONAL_PLAIN_RESULTS`, `COMPOSITIONAL_RESULTS`*, `COMP_HEADROOM`, `COMP_HEADROOM_PREREG`, `CORRECTION_COMPOSITIONAL`, `CSCG_TASK_GATES`, `DISSOCIATION_SWEEP`, `HIERGOAL_ABLATION`, `HIERGOAL_CLOSEDLOOP`, `HIERGOAL_LONGT`, `HIER_ATTN_LONGT`, `HIER_RECHECK`, `LAP_GATES`, `LAP_TRANSFER`, `LAP_TRANSFER_NOREWARD`, `MAP_QUERY_GATES`, `MAP_QUERY_RESULTS`, `PLANNER_TASK_AUDIT`*, `ROOMS_GOAL_RESULTS`, `ROUTE_ATTN_RESULTS`, `SPACETIME_HIER_RESULTS`, `STITCH_ATTENTION`

**Family tree** (6)

`ABLATE_FAMILY_TREE`, `FAMILY_TREE_D7_GATES`, `FAMILY_TREE_D7_RESULTS`, `FAMILY_TREE_GATES`*, `FAMILY_TREE_RESULTS`, `FAMILY_TREE_WM_GAP`

**MiniGrid, MiniWorld, Habitat** (54)

`ALIASING_CONTROLLED`, `ALIASING_COVARIATE`, `ALIASING_GATES`, `CONTINUOUS_ALLOC`, `CROSSOVER_CONVERGED`, `DAGGER_DK6_RESULTS`, `DAGGER_EMPTY_RESULTS`, `DAGGER_RESULTS`, `DOORKEY_BC_RESULTS`, `HABITAT_BUILD`, `MINIGRID_2X2`, `MINIGRID_2X2X2`, `MINIGRID_2X2X2_n8`, `MINIGRID_ALLOCENTRIC_2X2X2`, `MINIGRID_ALLOCENTRIC_8CELL`, `MINIGRID_ALLO_8THCELL`, `MINIGRID_DK16_RESULTS`, `MINIGRID_DOORKEY_CACHED`, `MINIGRID_DOORKEY_LONGT`, `MINIGRID_DOORKEY_RESULTS`, `MINIGRID_DOORKEY_ROPE_DIAG`, `MINIGRID_FULL_2X2X2`, `MINIGRID_MEMORY_RESULTS`, `MINIGRID_REPRO_CONTROL`, `MINIWORLD_ENDPOINTS`, `MINIWORLD_FIXED_FINDINGS`, `MINIWORLD_FIXED_RESULTS`, `MINIWORLD_FIXED_RESULTS_T1024`, `MINIWORLD_FRESH_ABLATION`, `MINIWORLD_FRESH_FINDINGS`, `MINIWORLD_FRESH_GATES_ALLO`, `MINIWORLD_FRESH_GATES_RAW`, `MINIWORLD_FRESH_RESULTS`, `MINIWORLD_FRESH_RESULTS_T1024`, `MINIWORLD_GATES_ALLO`, `MINIWORLD_GATES_FIXED`, `MINIWORLD_GATE_CONTROL`, `MINIWORLD_GRID16_GATES`, `MINIWORLD_GRID24_GATES`, `MINIWORLD_GRID32_GATES`, `MINIWORLD_GRID_SWEEP`, `MINIWORLD_GRID_SWEEP_HIER`, `MINIWORLD_HIER_ABLATION`, `MINIWORLD_ORACLE_ABLATION`, `MINIWORLD_ORACLE_GATES`, `MINIWORLD_ORACLE_RESULTS_T1024`, `MINIWORLD_ORACLE_RESULTS_T512`, `MINIWORLD_PROBE3`, `MINIWORLD_RESULTS`, `MINIWORLD_SROPE_COMPONENTS`, `MINIWORLD_TODO`, `PERCEPTION_EXPERIMENT_PLAN`, `POSITION_EFFECT_CONVERGED`, `VISITS_TEST`

**Dyck-2 and the cancellation knob (H3): depth substitution** (21)

`CANCEL_PREREG`, `CANCEL_RESULTS`*, `DYCK_DECAY_PREREG`, `DYCK_DECAY_PROBE`, `DYCK_DECAY_RESULTS`, `DYCK_DEPTH_PREREG`, `DYCK_DEPTH_RESULTS`, `DYCK_FAR_PROBE`, `DYCK_GATES`, `DYCK_LADDER_PREREG`, `DYCK_LADDER_RESULTS`*, `DYCK_LITERATURE_METRICS`, `DYCK_MDEPTH_GATES`, `DYCK_MDEPTH_PREREG`, `DYCK_MDEPTH_RESULTS`, `DYCK_PREREG`, `DYCK_RESULTS_POPE`, `DYCK_RESULTS_bs128`, `DYCK_SHAREABLE`, `DYCK_STACK_PROBE`, `DYCK_T3_RESULTS`

**Navigation told in words and the context-dependent step** (9)

`CONTEXT_STEP_DESIGN`*, `CTXSTEP_HSR_PILOT`, `CTXSTEP_HS_RECIPE`, `CTXSTEP_PILOT1`, `CTXSTEP_PILOT2`, `CTXSTEP_PILOT3`, `CTXSTEP_PREREG`, `TEXTWORLD_PREREG`, `TEXTWORLD_RESULTS`*

**New objects, what/where and the what-to-where leak** (10)

`GAIN_PHASE_PREREG`, `GAIN_PHASE_RESULTS`, `LEAK_PREREG`, `LEAK_RESULTS`, `TW_AMBIG_PREREG`, `TW_LANDMARK_PREREG`, `TW_NORMSTEP_PREREG`, `TW_NORMSTEP_RESULTS`*, `TW_STATECHANGE_PREREG`, `TW_STATECHANGE_RESULTS`

**Indirect Indexing** (5)

`INDIRECT_ARITHMETIC`*, `INDIRECT_OOD`, `INDIRECT_PREREG`, `INDIRECT_RESULTS`*, `INDIRECT_RESULTS_200k`

**Bach chorales, decay envelope and the MapPoPE collapse** (27)

`AUG_PREREG`, `AUG_RESULTS`, `CROSS_PREREG`, `CROSS_RESULTS`, `DECAY_PREREG`, `DECAY_RESULTS`, `JSBLEN_PREREG`, `JSB_LENGTH_RESULTS`, `JSB_LENGTH_RESULTS_BASE`, `JSB_LENGTH_RESULTS_RANK`, `JSB_PREREG`, `JSB_RESULTS`, `MAESTRO_PLAN`, `MAPPOPE_VS_POPE`, `POPE_WRAPPING`, `RECENCY_T3_PREREG`, `RECENCY_T3_RESULTS`*, `T1_PREREG`, `T1_RESULTS`, `T2_PREREG`, `T2_RESULTS`, `T3GEN_PREREG`, `T3GEN_RESULTS`*, `T3_PREREG`, `T3_RESULTS`, `THEORY_MAPPOPE`, `TORUS_T3_RESULTS`

**PoPE ablation, code and enwik8** (17)

`ABLATE_PREREG`, `ABLATE_RESULTS`*, `BF16_RESULTS`, `CODE_DECAY_RESULTS`*, `CODE_FULLVAL_PREREG`, `CODE_FULLVAL_RESULTS`, `CODE_GATES`, `CODE_PREREG`, `CODE_RESULTS`*, `CODE_RESULTS_OOD`, `ENWIK8_2X2`, `ENWIK8_COMPOSITION_PREREG`, `ENWIK8_HIER`, `ENWIK8_HIERARCHY`, `ENWIK8_LONG`, `ENWIK8_SEEDS`, `LANGUAGE_LANDSCAPE`

**Level 1.5 / InEKF / PC / TEM / grid cells (April-August lines)** (56)

`BUMP_TOKEN_RESULTS`, `CAPACITY_CONTROL`*, `CAPACITY_PERREGIME`*, `CASCADE_MULTISEED_RESULTS`, `CASCADE_REPRO_TEST`, `CASCADE_ZEROSHOT_S0`, `CLONE_ANALYSIS_LEVEL15PC`, `CLONE_TRANSFER_NOBYPASS`, `CNAV_HEX_Level15`, `CNAV_HEX_Level15EM`, `CNAV_HEX_Vanilla`, `CNAV_HEX_VanillaEM`, `CNAV_RESULTS`, `CORRECTED_LM200_LEADERBOARD`, `DOG_RESULTS`, `EXTRAHEAD_CONTROL`, `GSF_FULL_RESULTS`*, `HIPPOCAMPAL_ANALYSIS`, `HIPPOCAMPAL_GRID`, `HIPPOCAMPAL_GRIDL15PC`, `HIPPOCAMPAL_GRID_FREE`, `HIPPOCAMPAL_HIDDEN`, `HIPPOCAMPAL_HIDDEN_GRIDFREE`, `HIPPOCAMPAL_LEVEL15PC`, `L15_ABLATION`, `LEVEL15BETA_RESULTS`*, `LEVEL15EM_CROSSSCALE`, `LEVEL15_MEETS_GATED_matchq`, `LEVEL15_MEETS_GATED_paper`, `LEVEL15_MEETS_GATED_paper50`, `LM200_ABLATION`, `LM200_CORRECTED_MULTISEED`*, `MULTICLASS_MULTISEED_RESULTS`, `MULTICLASS_RESULTS`, `MULTISEED_FOLLOWUP`, `MULTISEED_FOLLOWUP_RESULTS`, `NOBYPASS_RESULTS`*, `NODROP_PARETO_RESULTS`*, `NOISE_REFINE`, `NUMBERLINE_RESULTS`, `RESULTS_PAPER`*, `R_T_DISTRIBUTION_3WAY`, `SESSION_HIERARCHICAL_CASCADE`, `TEM_BACKGROUND_BASELINES`, `TEM_CROSSSCALE_DIAGNOSTIC`, `TEM_NOISE_FFN_RESULTS`*, `TEM_RESULTS`*, `TEM_T_MULTISEED`*, `TEM_T_RESULTS`*, `V3_RESULTS`*, `V4_CONTROL_RESULTS`*, `V4_MULTISEED`*, `V4_RESULTS`*, `VECTOR_NAV_V2_RESULTS`, `VOCAB_SWEEP_MULTISEED`, `VOCAB_SWEEP_RESULTS`*

**Project meta, reports and infrastructure** (12)

`CLAUDE`*, `GUARDS`, `HOURGLASS_README`, `KNOWN_BUGS`, `PUBLICATION_VENUES`, `README`*, `REPORT`, `REPORT_ADDENDUM`*, `REPORT_v2`*, `RESULTS_INDEX`*, `RESULTS_SUMMARY_2026-05-10`*, `SESSION_2026-05-01`

**Docs-level notes** (11)

`docs/LOG.md`, `docs/NORMSTEP_NOTES.md`, `docs/SESSION_2026-09-27_to_10-03.md`, `docs/SESSION_2026-10-04_to_10-09.md`, `docs/WHAT_WHERE_ANALYSIS.md`, `docs/WHAT_WHERE_CHECKS.md`, `docs/WHERE_THINGS_STAND.md`, `docs/kalman_mapformer_intuition.md`, `docs/lit/LIT_CONTEXT_STEPS.md`, `docs/lit/LIT_NEW_OBJECTS.md`, `docs/lit/LIT_WHAT_WHERE.md`
