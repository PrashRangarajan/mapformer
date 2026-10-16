# Number-and-status audit of draft 5 (2026-10-16)

Scope: `docs/paper/DRAFT5_workshop.md` (abstract, Secs. 1-6, figure list) and `docs/paper/APPENDIX_draft5.md` (A-D, F-K),
against the results files (CORRECTED / AUDIT blocks read first), `*_ANALYSIS.txt`, `*_VERDICTS.json`, eval JSONs, audits
under `docs/audits/`, the 2026-10-16 floors, `docs/paper/figs/*_out.txt`, and `CLAIM_MAP_clock_map.md` ("Do not claim"
lists and the per-batch "say" wordings). Read-only: nothing in the draft, appendix or results files was edited. Theory
statements (appendix E) were checked only where the draft quotes a number from them (P6, P7).

Verdicts: **OK** (matches the source, rounding included); **WRONG** (value or statement does not match the source);
**UNSOURCED** (no file supports it); **STATUS-MISLABELLED** (registered / post hoc / declared secondary / pilot status is
wrong); **ROUNDING** (same quantity rounded inconsistently); **POOLED** (pilot seeds inside a quoted number);
**CLAIM-MAP** (contradicts the claim map's wording).

Count: **252 table rows**, covering about **760 individual values** (per-seed tables are one row per table column or
caption, every cell compared). **12 errors** (listed after the table); everything else OK.

## 1. Main text

| location | claim as written | source | source value | verdict |
|---|---|---|---|---|
| Abstract l.11 | barrier steps solve 0-4 of 8 seeds per batch | SIGN_MATCHED, RING_DIAL, WRAP_EXISTENCE, MECHANISM, GENERALITY, TORUS_START, GAUGE, ROBUST | |.| / softplus 0,1; 2,1,4; 0; 0; 0; 1; 1 (+1/-1); ROBUST 1,0,4,0 | OK |
| Abstract l.11-12 | "against 8 of 8 for signed steps" | ROBUST_ANALYSIS lrd3; GAUGE G1 | signed 6/8 at lr /3 (quoted in the same abstract), 5/8 on +1/+2 | WRONG (scope; error 11) |
| Abstract l.12-13 | exact |.| solutions exist, kept (ring) / re-found (torus); from scratch 0-1 of 8 | WRAP_EXISTENCE E1/E2; GENERALITY E1/E2/E3; TORUS_START TS4 | HELD 8/8; NOT HELD by LOST, 8/8 re-SOLVED; AO 0/8, AO4 0/8, 1/8 | OK |
| Abstract l.15-16 | +1/+2: both classes struggle, comparison unmeasured | GAUGE G1 | UNMEASURED, 2/8 vs 5/8 | OK |
| Abstract l.16 | holds without wd, at lr /3, in 2-layer d 256 torus; shrinks at lr x3 | ROBUST_RESULTS | wd0 0/8 v 8/8, lrd3 0/8 v 6/8, torus 2/8 v 8/8 (fire); lrx3 UNMEASURED | OK on numbers; CLAIM-MAP (error 10): omits "and in accuracy at scale" |
| Abstract l.16 | budgets 300-900 epochs | A3, C3 | 300 ring, 900 torus and RING_BUDGET | OK |
| Sec. 1 l.21 | CARoPE rate in (0, 1) | appendix E P6; model_mono_baselines | f in (0, 1) | OK |
| Contrib. 2 | from scratch 0-1/8 | as abstract | 0/8, 0/8, 1/8 | OK |
| Contrib. 3 | registered learnability test inconclusive | GAUGE "AS FALLS", G1 UNMEASURED | | OK |
| Contrib. 4-5 | ring route; CoPE representable yet fails | MECHANISM M2/M5; GENERALITY B | | OK |
| Sec. 2 Model | 1 layer, 2 heads, d 128, rank-4 shared latent | A1; run drivers | | OK |
| Sec. 2 | softplus(0) = 0.69 | ln 2 = 0.693 | | OK |
| Sec. 2 Tasks | torus 64 x 64, T 1024, 900 ep; held-out map; ring 32, T 128, 300 ep, maps redrawn; line 256 cells | GENERALITY, RING_DIAL, ACCOUNTS T4; DRAFT5_changes factual correction | | OK |
| Sec. 2 | SOLVED = final 5% loss < 0.05 | stats_core.classify_run | | OK |
| Sec. 2 Protocol | n = 8 per arm unless stated | all batches | 8 (CoPE T4 16, FACTORIAL 4: stated) | OK |
| Sec. 2 Protocol | every main-text batch except the torus sign batch had an independent pre-launch check | preregs' Amendments | RING_DIAL, RING_BUDGET, WRAP_EXISTENCE, GENERALITY, MECHANISM, WRAP_COPE, ACCOUNTS, TORUS_START, GAUGE, ROBUST: yes; SIGN_MATCHED: no amendment | OK for these batches; but Sec. 4.1's "0.72-0.73" draws on CANCEL (no amendment, no independent check) -- error 4 |
| Sec. 2 Protocol | torus sign batch: authors' pre-launch gate | docs/audits/2026-09-27/gate_sign_matched.py | exists | OK |
| Sec. 2 Protocol | "an independent audit after the run" | SIGN_MATCHED_RESULTS ("reproduced by the 2026-09-30 audit"); commit f17e934 | an audit is recorded; no file records that it was independent | UNSOURCED (error 7) |
| Sec. 2 Protocol | pilots used seeds outside the batch | RING_DIAL 130, WRAP_EXISTENCE / GENERALITY 690, WRAP_COPE 990, GAUGE 1390, ROBUST 1490; others none | | OK for main-text batches (TEXTWORLD, CANCEL violate it; both outside the main-text contrasts except error 4) |
| Sec. 3 | 2 stepping moves on ring, 4 on torus, r = 4 | A2; model_wrap_exist | | OK |
| Sec. 3 Lemma 1 | J R(phi) J = R(-phi), checked bitwise | APPENDIX_theory_v2 T1 | | OK |
| Sec. 3 | Howard propagator = Mamba-3 Prop. 3 | APPENDIX_theory_v2 A0 | | OK |
| 4.1 l.108 | |.| -0.177, 0.821 vs 0.998, perm p 0.0002, 0/8 vs 8/8 | SIGN_MATCHED_ANALYSIS | -0.177, 0.8211 / 0.9983, 0.0002, 0/8 v 8/8 | OK |
| 4.1 l.109 | 3/8 |.| runs still descending | SIGN_MATCHED_ANALYSIS classes | DESC s0, s1, s4 | OK |
| 4.1 l.109 | softplus -0.218 (1/8; init confound) | SIGN_MATCHED_ANALYSIS | -0.218, 1/8 | OK |
| 4.1 l.110-111 | (post hoc) blank 0.506, n-gram 0.575, retrace 0.843 | docs/audits/2026-10-16/sign_floors_out.txt | 0.5059, 0.5749, 0.8429 | OK |
| 4.1 l.111-112 | |.| mean below retrace, 3/8 runs below; softplus 0.781, 4/8; index every run 0.72-0.74 | SIGN_MATCHED.json vs per-seed retrace | Abs 0.637/0.690/0.717 below; Pos 4 below; RoPE 0.720-0.744 all below | OK |
| 4.1 l.113 | ring |.| 2/8, 1/8 vs 8/8, Fisher 0.007, 0.0014 | RING_DIAL_ANALYSIS | 0.0070, 0.0014 | OK |
| 4.1 l.113-114 | p_plus 0.9 unmeasured, 4/8 vs 8/8, -0.046 | RING_DIAL_ANALYSIS | -0.0459, Fisher 0.0769 | OK |
| 4.1 l.114 | 3x training leaves |.| at 3/8 | RING_BUDGET_ANALYSIS | 3/8 vs 3/8, PARTIAL | OK |
| 4.1 l.114-115 | ring gap -0.03; |.| 0.95-0.97 | RING_DIAL_ANALYSIS | -0.0275/-0.0320/-0.0459; 0.954-0.973 | OK |
| 4.1 l.115 | n-gram 0.554, retrace 0.860 (post hoc) | CANCEL_PREREG; ring_floors_out.txt | 0.554; 0.8596 | OK |
| 4.1 l.116 | one-layer index 0.72-0.73 at p_plus 0.5 | CANCEL_RESULTS (CORRECTED block); GENERALITY C | 0.717 pools pilot seeds 0-1; fresh 0.710; GENERALITY 0.730 | POOLED (error 4) |
| 4.1 l.117-118 | post hoc: at loss 0.1 p_plus 0.5 no longer fires, 4/8 v 8/8, p 0.077 | solved_thresholds_out.md | 4/8 v 8/8 (0.0769) | OK |
| 4.1 l.118-119 | every other sign contrast holds at 0.02, 0.1, acc >= 0.99 | solved_thresholds_out.md; ROBUST threshold secondary | true for Table C10; ROBUST torus 5/8 v 8/8 (p 0.20) at 0.1 does not hold | WRONG (error 2) |
| 4.1 l.121-122 | signed cancel in 64-98% (arm means), |.| / softplus 3-11% | docs/paper/figs/fig_wrap_share_out.txt | 0.64-0.98; 0.03-0.11 | OK |
| 4.1 l.123 | ring AO 0/8 v 8/8 (two batches); torus 0/8 v 8/8 (-0.120), 1/8 v 8/8 (-0.104) | WRAP_EXISTENCE E3, MECHANISM M4, GENERALITY E3, TORUS_START TS4 | as written | OK |
| 4.1 l.125 | ROBUST registered, fresh seeds | ROBUST_PREREG, seeds 1400-1407 | | OK |
| 4.1 l.126 | wd 0: 0/8 v 8/8, -0.127 | ROBUST_ANALYSIS | -0.1273 | OK |
| 4.1 l.126 | lr /3: 0/8 v 6/8, Fisher p 0.007 | ROBUST_ANALYSIS | 0.0070 | OK |
| 4.1 l.127 | default 1/8 v 8/8 | ROBUST_ANALYSIS | | OK |
| 4.1 l.127-128 | lr x3 unmeasured, 4/8 v 8/8, p 0.077; 0.987 v 1.000 | ROBUST_ANALYSIS | 0.0769; 0.9868 | OK |
| 4.1 l.128 | 2 layers d 256, 8x params | ROBUST_RESULTS (1.59M v 0.20M) | 7.95x | OK |
| 4.1 l.128-129 | 2/8 v 8/8, Fisher 0.007; 0.967 v 0.999, p 0.0005 | ROBUST_ANALYSIS | 0.0070; 0.9671 / 0.9993; 0.0005 | OK |
| 4.1 l.129 | accuracy gap a quarter of one-layer gap (out of batch) | -0.032 / -0.120 (GENERALITY E3) | 0.27 | OK |
| 4.1 l.129-130 | 6/8 |.| still descending | ROBUST_ANALYSIS classes | 6 DESC | OK |
| 4.1 l.130 | SOLVED fails at 0.1 (5/8 v 8/8; post hoc) | ROBUST_PREREG l.81 declared secondary; ROBUST_ANALYSIS | 5/8 v 8/8, p 0.2000, declared secondary | STATUS-MISLABELLED (error 3) |
| 4.1 l.131-132 | solving |.| runs without the wrapped code: 0.04-0.13 vs 0.43-1.00 | ROBUST secondaries | | OK |
| 4.2 l.137 | angle error <= 1.4e-6 rad; frozen = source 8/8 | WRAP_EXISTENCE E1 (1.2e-6), GENERALITY E1 (1.4e-6) | | OK |
| 4.2 l.138 | ring weights within ~5x | WRAP_EXISTENCE_ANALYSIS W_out 0.31-0.38 -> 1.07-1.70 | up to 5.0x | OK |
| 4.2 l.138-139 | naive construction exact, ill-conditioned, lost within ~10 epochs | WRAP_EXISTENCE_RESULTS | | OK |
| 4.2 l.141 | ring: 8/8 stays solved, never leaves the solved region | WRAP_EXISTENCE E2 | 8/8 SOLVED, 0/8 LOST | OK |
| 4.2 l.142 | usage-weighted 99% -> 98% | ACCOUNTS T1 | 0.99 -> 0.98 | OK |
| 4.2 l.143 | torus peak loss 0.11-0.21 v <= 0.017; 6/8 leave | GENERALITY_ANALYSIS | 0.110-0.208; 0.004-0.017; LOST 6/8 | OK |
| 4.2 l.144 | re-finds, 8/8 SOLVED, 1.000, lr 1e-4 fixed before launch (pilot) | GENERALITY_PREREG l.37-39, E2 | | OK |
| 4.2 l.145 | lr 1e-3 (descriptive) re-solves 6/8 | GENERALITY_ANALYSIS AO_bal_hi | 6/8 | OK |
| 4.2 l.145 | from scratch 0/8 on both tasks | WRAP_EXISTENCE AO, GENERALITY AO4 | 0/8, 0/8 | OK |
| 4.3 Table 1 | default 8/8 / 0/8; same-direction 7/8 / 0/8; sub-pi -- / 2/8; past pi 8/8 / 8/8 | MECHANISM_ANALYSIS | SO 8, AO 0, SO_absmag 7, AO_absmag 0, AO_mid 2, SO_high 8, AO_high 8 | OK |
| Table 1 caption | n = 8; contrasts same at 0.02, 0.1, acc >= 0.99 (post hoc) | solved_thresholds_out.md | all four MECHANISM rows fire / null as registered | OK |
| 4.3 (i) | 0/8 v 7/8, Fisher 0.0014; -0.092 [-0.135, -0.045] | MECHANISM M5 | | OK |
| 4.3 (ii) | 8/8 v 0/8, +0.109 [+0.085, +0.135], p 0.0002 | MECHANISM M2 | +0.1085 | OK |
| 4.3 (iii) | sub-pi 2/8, Fisher 0.007 v 8/8 | MECHANISM M2b | 0.0070 | OK |
| 4.3 (iv) | 8/8 v 8/8 | MECHANISM M6 | | OK |
| 4.3 | four registered contrasts | MECHANISM_PREREG (M5, M2, M2b, M6) | | OK |
| 4.3 post hoc | 98-99% opposite-signed (from 0%); none wrap; past pi 88% / 100% by wrapping | docs/audits/2026-10-14/mech_escape_out.txt | 0.99/0.99/0.98; 0.00; 0.88/1.00 | OK |
| 4.3 collective | registered, fresh seeds; 1/8 v 7/8, Fisher 0.010; -0.075 [-0.130, -0.020] | ACCOUNTS T2 F1 (seeds 1000-1007) | 0.0101 | OK |
| 4.3 collective | 15% v 82% opposite-signed | ACCOUNTS_ANALYSIS | 0.15 / 0.82 | OK |
| 4.3 collective | default start: no detected effect, 7/8 v 8/8 | ACCOUNTS F2 | | OK |
| 4.3 loose ends | rescued |.| cancel 23% | MECHANISM secondary | 0.23 | OK |
| 4.3 loose ends | Cohen's kappa +0.09 | capture_by_channel_out.txt | +0.086 | OK |
| 4.3 loose ends | 24-42% of ring move steps collapse | collapse_wall_out.md (a) | 0.235-0.424 | OK |
| 4.3 loose ends | gradient into the wall for about half; MIXED by a pre-fixed rule | collapse_wall_out.md (b); PREDICTIONS.md | 0.51-0.57, MIXED | OK |
| 4.4 | past-pi start 0/8 v 1/8, +0.005 [-0.060, +0.070], no effect detected | TORUS_START TS1 | +0.0045 | OK |
| 4.4 | signed slowed to 2/8 v 8/8 (secondary) | TORUS_START secondaries | | OK |
| 4.4 | 5/8 per large-start arm descending | TORUS_START_ANALYSIS | AO4_high 5, SO4_high 5, AO4_mid 5 | OK |
| 4.4 | default sign cost 1/8 v 8/8, Fisher 0.0014 | TS4 | | OK |
| 4.5 Table 2 row 1 | 1.000, 8/8 v 0.931, 1/8; -0.069 [-0.095, -0.045], Fisher 0.0014; signed learns better | GAUGE G2 | 0.9309; -0.0691 | OK |
| 4.5 Table 2 row 2 | 0.947, 5/8 v 0.883, 2/8; -0.064 [-0.200, +0.075]; Fisher 0.31; unmeasured | GAUGE G1 | 0.9470, 0.8834, -0.0637, 0.3147 | OK |
| 4.5 | third obs-step-free ring batch | WRAP_EXISTENCE, MECHANISM, GAUGE | | OK |
| 4.5 | index RoPE 0.649, 0/8; floor ~0.50 | GAUGE | 0.6494 | OK |
| 4.5 | 0.92 vs 0.01 same-sign channels | GAUGE secondaries | | OK |
| 4.5 | |.| +1/+2 2/8 v +1/-1 1/8, Fisher 1.00 | GAUGE G3 | | OK |
| 4.5 | laps 0.893 / 0.884, post hoc reading of a declared secondary | GAUGE strata | | OK |
| 4.6 | P6: (0, 1] per token, shortest return (0, 4] | APPENDIX_theory_v2 P6 | (0, 4] obs-obs, (0, 3] move-obs | OK |
| 4.6 | CARoPE torus 0.667 < RoPE 0.722, p 0.040; ring 0.741 v 0.730 unmeasured | GENERALITY C | 0.6666, 0.7219, 0.0396; 0.7410, 0.7301 | OK |
| 4.6 | CoPE 0/8, 0.887 v 1.000, p 0.0002; +0.157 over index | GENERALITY B / C | 0.8871, +0.1570 | OK |
| 4.6 | 1 of 32 at the paper's init over three ring batches, pooled post hoc | GENERALITY 0/8, WRAP_COPE zeros 1/8, ACCOUNTS T4 ring 0/16 | 1/32, labelled | OK |
| 4.6 | line 1/16 v ring 0/16; signed 8/8; index 0.772 | ACCOUNTS T4 | | OK |
| 4.6 | half the line runs pass at a looser threshold (post hoc) | solved_thresholds_out.md | 8/16 at 0.1 | OK |
| Sec. 6 | 3x ring budget did not help | RING_BUDGET | | OK |
| Sec. 6 | recipe check wording | ROBUST | matches CLAIM_MAP "say" wording | OK |
| Sec. 6 | p_plus 0.5 fails at 0.1 (post hoc) | solved_thresholds_out.md | | OK |
| Sec. 6 | per-head rank decides torus learnability for signed steps | RANK_SEP_RESULTS (CLAUDE.md) | | OK |
| Fig. 1 caption | 0.1 rad, wrap-invariant; signed 0.79/0.78/0.64/0.98; |.| 0.06/0.03/0.06/0.11; softplus 0.03 | fig_wrap_share_out.txt | identical | OK |
| Fig. 2 (a) | 7/8; 0.06 -> 0.89, through zero | fig_mechanism_out.txt | 0.062 -> 0.895, opposite-sign 0.883, wrap 0.000 | OK |
| Fig. 2 (b) | 0/8; 0.04 of which 0.035 collapsed | fig_mechanism_out.txt | 0.042; 0.035 | OK |
| Fig. 2 (c) | 2/8; 0.05 by wrapping | fig_mechanism_out.txt | 0.047 by wrap | OK |
| Fig. 2 (d) | 8/8; 0.07 -> 0.23 by wrapping | fig_mechanism_out.txt | 0.072 -> 0.230 | OK |
| Fig. K1 | n = 4; means, range | fig_factorial_strata.py | | OK |
| Fig. K1 | dashed blank floor "averaged over strata and runs (overall 0.507)" | FACTORIAL_*_STRATA.json via fig_factorial_strata.py | line drawn at 0.513; own-stream overall 0.509; 0.507 is the RANK_MI stream (seeds 10-25) | WRONG (error 1) |
| Fig. K1 | n-gram 0.576 not drawn | score_rank_floors_out.txt | another stream | OK (label the stream; error 1) |
| Figures list | "Appendix tables C1-C12 (and C9b)" | APPENDIX_draft5 | C13 exists | WRONG (error 12, cosmetic) |

## 2. Appendix A (models, recipes, batches)

| location | claim as written | source | source value | verdict |
|---|---|---|---|---|
| A1 | 32 blocks per head, 64 channels; omega ladder 2 pi (1/Delta_max)^(i/31); Delta_max = grid | model.py l.105-110 | omega_max 2 pi, delta_max = grid_size | OK |
| A1 | ~204k params, one-layer torus MapWM | FACTORIAL_RESULTS | ~204k | OK |
| A1 layers row | 1 (K: 4 and loop; F: 1-4) | -- | ROBUST torus is 2 layers d 256 (C13) | WRONG by omission (error 6) |
| A2 | Abs/Pos after W_out; softplus(0) 0.693 | model_sign.py | | OK |
| A2 | AO_mid U(0.9, 1.5); AO_high/SO_high U(1.7, 4.5) | MECHANISM_RESULTS | | OK |
| A3 | lr 1e-3, AdamW wd 0.05, 5% warm-up + cosine to 10% | run_sign_matched.sh, drivers | | OK |
| A3 | torus batch 16 x T 1024, 98 x 900; ring 128 x 128, 98 x 300 | drivers | | OK |
| A3 | fast-attn: GENERALITY, TORUS_START, FACTORIAL; SIGN_MATCHED standard | run_generality.sh, train_torus_start.py, run_factorial.sh, run_sign_matched.sh | | OK (ROBUST torus also --fast-attn, not listed: error 6) |
| A3 | continuations: ring lr 1e-3, torus 1e-4 / 1e-3, offset 10^6 | GENERALITY_PREREG, WRAP_EXISTENCE_PREREG | | OK |
| A4 | torus eval: map 10000, 100 walks, seed 1234 + s | sign_floors.py docstring; eval_noise_refine | | OK |
| A4 | ring eval: env seed 10000, 400 walks, numpy 10^6 | FIGS_NOTES ring strata | | OK |
| A4 | dropout re-score "declared in the batches after 2026-10-04 ... changed no label in any batch cited here (each results file states it)" | grep of preregs / results | stated in RING_DIAL, RING_BUDGET, WRAP_EXISTENCE, GENERALITY, MECHANISM, WRAP_COPE; not declared or run in ACCOUNTS, TORUS_START, GAUGE; ROBUST declares none (invalid at 2 layers); FACTORIAL declared, not reported | WRONG (error 5) |
| A4 | run classes SOLVED / STALLED / DESCENDING rule | stats_core.classify_run | | OK |
| A4 | cancelling share definition (0.05, 0.1 rad) | fig_wrap_share.py | | OK |
| A5 SIGN_MATCHED | seeds 0-7; no independent check; gate; audit 2026-09-30; pilot none | SIGN_MATCHED_PREREG (no amendment), RESULTS | | OK |
| A5 RING_DIAL | 30-37; Amendment 1; pilot 130 | RING_DIAL_PREREG M3, l.38-44 | | OK |
| A5 RING_BUDGET | 40-47; yes; pilot none | RING_BUDGET_PREREG | | OK |
| A5 WRAP_EXISTENCE | 600-607; yes; pilot 690 | prereg l.71-77 | | OK |
| A5 GENERALITY | 700-707; yes; pilot 690 | prereg l.70-79 | | OK |
| A5 MECHANISM | 800-807; yes; CPU toy only | prereg l.63-67 | | OK |
| A5 WRAP_COPE | 900-907; yes; pilot 990 | prereg l.36-41 | | OK |
| A5 ACCOUNTS | T1 600-607, T2 1000-1007, T4 1100-1115; yes; none | prereg | | OK |
| A5 TORUS_START | 1200-1207; yes; CPU checks | prereg l.38-42 | | OK |
| A5 GAUGE | 1300-1307; yes; pilot 1390 | prereg l.32-45 | | OK |
| A5 FACTORIAL | 500-503 (n 4); yes; seed 400 probe | prereg l.41-45 | | OK |
| A5 TEXTWORLD | 0-7; yes, seeds 0-1 = pilot | TEXTWORLD CORRECTED block | | OK |
| A5 DYCK_MDEPTH | 0-7; no; four seed-0 smoke runs | DYCK_MDEPTH_PREREG l.64-69 | | OK |
| A5 CLOCK_* | 200-207; 210-225; 230-245; yes; calibration seed 100 | CLOCK_SIDE/SPEED/REP preregs | | OK |
| A5 | ROBUST row | ROBUST_RESULTS / PREREG | 1400-1407; yes (robust_audit.md); pilot 1490 | MISSING (error 6) |
| A5 | CANCEL row (its index values are used in B and Sec. 4.1) | CANCEL_RESULTS | seeds 0-7, s0-1 = pilot, no independent check | MISSING (error 4) |
| A5 drivers / trainers | all named scripts | ls | every file exists | OK |

## 3. Appendix B (floors and gates)

| location | claim as written | source | source value | verdict |
|---|---|---|---|---|
| B row 1 | blank 0.507 (0.497-0.511), seeds 10-25 | score_rank_floors_out.txt T=1024 | 0.5066 (0.4973-0.5108) | OK |
| B row 2 | n-gram 0.576 (0.568-0.582) | same | 0.5763 (0.5681-0.5815) | OK |
| B row 3 | retrace 0.843 (0.833-0.848) | same | 0.8427 (0.8332-0.8479) | OK |
| B row 4 | SIGN_MATCHED 0.506/0.575/0.843; GENERALITY 0.508/0.578/0.843 | sign_floors_out.txt, generality_floors_out.txt | 0.5059/0.5749/0.8429; 0.5077/0.5777/0.8431 | OK |
| B row 5 | ring retrace 0.860/0.758/0.612; 2000 walks, seed 4242 | ring_floors_out.txt, ring_floors.py | 0.8596/0.7582/0.6118 | OK |
| B row 6 | best constant per batch: SIGN 0.501-0.512 (0.506); GEN 0.504-0.518 (0.508); FACT 0.505-0.515 (0.509) | *_STRATA.json 'all' floor | 0.5007-0.5123 (0.5059); 0.5035-0.5184 (0.5077); 0.5052-0.5149 (0.5094) | OK |
| B row 7-8 | ring 0.554/0.544/0.520; fixed-lag 0.284/0.375/0.626; 1000 walks seed 1, n-gram 3000 walks map 0 | CANCEL_PREREG l.18-19; gate_cancel.py l.10-13 | | OK |
| B row 9 | ring strata constant 0.49-0.50 | FIGS_NOTES | | OK |
| B row 10 | line blank 0.503 | ACCOUNTS T4 | | OK |
| B row 11 | +1/+2 blank ~0.50 | GAUGE | | OK |
| B row 12 | text world 0.505 / 0.597 | TEXTWORLD_RESULTS | | OK |
| B row 13 | Dyck A2f floor 0.594 | DYCK_MDEPTH_RESULTS | 0.5935 | OK |
| B own-stream para | per-seed retrace 0.837-0.849; |.| below on 3/8 (0.637, 0.690, 0.717); softplus 4/8; RoPE 8/8; signed 0/8 | sign_floors_out.txt; SIGN_MATCHED.json | 0.8372-0.8490; as written | OK |
| B own-stream para | GENERALITY AO4 0.880 above mean retrace | GENERALITY E3; generality_floors_out | 0.8803 > 0.8431 | OK |
| B ring retrace para | blank 0.497-0.499; share inside a retrace 0.72/0.52/0.23 | ring_floors_out.txt | 0.4967-0.4987; 0.720/0.520/0.228 | OK |
| B ring retrace para | ring |.| 0.95-0.97 above retrace at p_plus 0.5 | RING_DIAL Abs 0.973 | | OK |
| B Gates | torus index 0.722-0.731 | GENERALITY 0.7219, SIGN_MATCHED 0.7311 | | OK |
| B Gates | ring index 0.717 / 0.676 / 0.825 (CANCEL) | CANCEL_RESULTS CORRECTED block | pools pilot seeds 0-1; fresh 0.710 / 0.677 / 0.824 | POOLED (error 4) |
| B Gates | ring 0.730 (GENERALITY), line 0.772, +1/+2 0.649 | GENERALITY C, ACCOUNTS T4, GAUGE | | OK |
| B Gates | below retrace on torus and ring 0.5 / 0.75 | as above | 0.72-0.73 < 0.843; 0.717 < 0.860; 0.676 < 0.758 | OK |

## 4. Appendix C (per-seed tables and re-analyses)

| location | claim as written | source | source value | verdict |
|---|---|---|---|---|
| C1 table (32 accuracy cells, 32 class + loss cells, 4 means) | per seed | SIGN_MATCHED.json; SIGN_MATCHED_ANALYSIS | every cell matches (s5 signed 0.98955 -> 0.990) | OK |
| C1 caption | -0.177, MDE ~0.14, p 0.0002; Fisher 0.0002; softplus -0.218 p 0.0003, 1/8 p 0.0014; r -0.992 / 32 | SIGN_MATCHED_ANALYSIS | MDE 0.138 | OK |
| C1 caption | seed 5 signed 0.9896 is the one < 0.99 | SIGN_MATCHED.json; solved_thresholds acc 7/8 | 0.98955 | OK |
| C2 table (48 cells) | per seed, S marks | RING_DIAL_VERDICTS.json | every cell matches | OK |
| C2 caption | d -0.0275 [-0.046, -0.009], p 0.0070; 2/8 Fisher 0.0070; -0.0320 p 0.0002; 1/8 0.0014; 0.9 -0.0459 p 0.0256, 4/8 p 0.0769 | RING_DIAL_ANALYSIS | | OK |
| C3 table (16 cells + losses) | per seed | RING_BUDGET_ANALYSIS | every cell matches | OK |
| C3 caption | PARTIAL, 3/8 v 3/8, p 1.00; signed 8/8 cross-batch descriptive | RING_BUDGET_RESULTS | | OK |
| C4 table (32 cells + 16 max losses) | per seed | WRAP_EXISTENCE_ANALYSIS | every cell matches | OK |
| C4 caption | E3 0.921 v 1.000, -0.079, p 0.0002, Fisher 0.0002; E2 HELD 8/8, 0/8 LOST | WRAP_EXISTENCE_ANALYSIS | | OK |
| C5 table (64 cells + 8 means) | per seed | MECHANISM_ANALYSIS | every cell matches | OK |
| C5 caption | M5, M2, M2b, M6, M1 (-0.017), M3, M4 (-0.081) with CIs and p | MECHANISM_ANALYSIS | | OK |
| C5 caption | end cancelling shares SO 0.86 ... SO_high 0.34 | MECHANISM secondaries | | OK |
| C6 table (4 arms x acc, SOLVED, 8 shares) | | ACCOUNTS_ANALYSIS T2 | 1.000/0.991/0.985/0.910; 8,7,7,1; per-seed shares match | OK |
| C6 caption | F1 1/8 v 7/8 p 0.010; -0.075 [-0.130, -0.020], perm 0.009; F2; SUPPORTED | ACCOUNTS_ANALYSIS | 0.0101, 0.0089 | OK |
| C7 table (48 cells, peak loss / epochs) | per seed | GENERALITY_ANALYSIS | every cell matches | OK |
| C7 caption | SO_bal 0.004-0.017; E3 0.880 v 1.000, -0.120; E2 6/8 LOST, Fisher 0.0070; AO_bal_hi 6/8, 0.988 | GENERALITY_ANALYSIS | | OK |
| C8 table (40 cells + 5 means) | per seed | TORUS_START_ANALYSIS | every cell matches | OK |
| C8 caption | TS1 +0.005 [-0.060, +0.070] p 0.89; TS2; TS3 2/8 < 4; TS4 1/8 v 8/8 p 0.0014, -0.104 [-0.150, -0.060]; strata | TORUS_START_ANALYSIS | | OK |
| C9 table (40 cells) | per seed | GAUGE_ANALYSIS | every cell matches | OK |
| C9 caption | "on +1/-1, 87.5% of revisits are backtracks" | GAUGE_RESULTS (copied from gauge_audit.md sample); GAUGE_ANALYSIS stratum counts | 138,800 / 155,456 = 89.3% on the registered eval stream | WRONG (source-inherited; error 8) |
| C9 caption | +1/+2 every revisit a lap with gap >= 16 | GAUGE_ANALYSIS | lap n = gap>=16 n = 153,872 | OK |
| C9b table (45 cells) | | GAUGE_ANALYSIS | every cell matches | OK |
| C9b note | laps 0.893 v 0.884 post hoc reading; 0.92 (0.57-1.00) v 0.01 | GAUGE | | OK |
| C10 intro | all 47 arms reproduce at 0.05 | solved_thresholds_out.md | 47 rows, "ALL REPRODUCE" | OK |
| C10 table (19 rows x 4 criteria = 76 cells) | | solved_thresholds_out.md | every cell matches | OK |
| C10 caption | only RING_DIAL p 0.5 changes its reading | same | true within C10 | OK |
| C11 table (7 rows x 7) | | FIGS_NOTES / ring_strata_out.txt | every cell matches | OK |
| C11 caption | -0.035 (p 0.031); -0.080, -0.105 (8/8, p 0.0078); wrap-only -0.063 (0.031), -0.007 (0.88), -0.032 (0.31); max diff 5e-5 | FIGS_NOTES | | OK |
| C12 bullet 1 | initial sums 0.01-6.58; 5% past pi; 20 channels pi..3pi/2 none cancel | fig_mechanism_out.txt; mech_channels_out.txt; mech_bins_out.txt | 0.01..6.58; 0.05; AO [3.14, 4.71) n 20 canc 0.00 | OK |
| C12 bullet 2 | 23% (signed 34%; 7% at start); capture 0.34 within 0.25 rad, 0.08 at 2-3 rad | MECHANISM secondaries; fig_mechanism_out; mech_capture_out | 0.23, 0.34, 0.072; 0.34, 0.08 | OK |
| C12 bullet 3 | flow: 12% cross zero; ~0.5 rad; 28% of U(1.7, 4.5) | potential_out.txt; THEORY_REVIEW l.178-180 | 0.12; 0.28 | OK |
| C12 bullet 3 | 505 channels; 0.329 v 0.230; kappa +0.086 (per run -0.26..+0.51); 95% / 28%; 59% not captured; SO_high +0.198 | capture_by_channel_out.txt | 0.953 / 0.279; 1 - 0.405 = 0.595 (69/116) | OK |
| C12 bullet 4 | 44% v 17%, 0.2%, 1.4% | fig_mechanism_out.txt | 0.441, 0.166, 0.002, 0.014 | OK |
| C12 bullet 4 | collapse ring 0.235-0.424, torus 0.601-0.695, softplus 0.624; signed ring 0.104-0.157, torus 0.343-0.402 | collapse_wall_out.md (a) | | OK |
| C12 bullet 4 | 77-81% stay (signed 29-35%); usage ratio 0.39-1.10; gradient 51-57%, MIXED | collapse_wall_out.md (b) | | OK |
| C12 table (26 values) | | collapse_wall_out.md (a) | every cell matches | OK |
| C12 last bullet | 13-22x faster; AO_mid ~5x | MECHANISM_RESULTS M2b | | OK |
| C13 header | seeds 1400-1407, registered, PARTLY UNMEASURED; 1.59M params | ROBUST_RESULTS | | OK |
| C13 decisions table (5 rows x 7) | | ROBUST_ANALYSIS | every cell matches | OK |
| C13 gate / lrx3 note | SO >= 4/8; Fisher 0.077, perm 0.026 < materiality | ROBUST_RESULTS | | OK |
| C13 secondaries table (5 x 5 pairs) | | ROBUST_ANALYSIS secondaries | every cell matches (declared secondaries, correctly labelled here) | OK |
| C13 per-seed block (64 ring + 16 torus cells) | | ROBUST_ANALYSIS | byte-identical | OK |

## 5. Appendix D (constructions)

| location | claim as written | source | source value | verdict |
|---|---|---|---|---|
| D1 | wrap in step space; omega crosses zero (pilot -0.012; RING_DIAL 31-33) | WRAP_EXISTENCE_PREREG A2 | | OK |
| D2 | construction steps; SO_bal same function, unwrapped | WRAP_EXISTENCE_PREREG A1; GENERALITY_PREREG | | OK |
| D2 E1 | ring 1.2e-6 / 6.1e-7 rad; logits 5.1e-4 (T 128), 1.1e-2 (T 512); torus 1.4e-6 / 5.8e-7 | WRAP_EXISTENCE_ANALYSIS; GENERALITY_ANALYSIS | | OK |
| D2 scale | ring W_out 0.31-0.38 -> 1.07-1.70; W_in 0.15-0.21 -> 0.19-0.33; step 6.28; torus 0.48-0.58 -> 1.23-1.55; omega 0.68-0.96 -> 1.00 | same | | OK |
| D3 | AO_wrap exact, 1.2e-5 rad; cos -0.987..-0.998; W_out 653-16,320; steps to 9,333; LOST 8/8, max 0.6-2.0, Fisher 0.0002; 7/8 re-solved; AO_wrapg 4.62, 8/8 LOST, 8/8 | WRAP_EXISTENCE_ANALYSIS | 0.644-2.035 | OK |
| D4 | min 5e-11; obs |Delta| <= 1.7e-6; 6.3e-4; 4.6e-4 on 2,000 pairs | docs/audits/2026-10-12/ring_dial_audit.md l.49-52 | | OK |
| D4 | no softplus construction in the files | DRAFT5_changes decision 4 grep | | OK |
| D5 | LOST / HELD / NOT HELD rule | WRAP_EXISTENCE_PREREG A4 l.96-99 | | OK |
| D5 ring | max loss 0.024-0.154 v 0.015-0.112 | WRAP_EXISTENCE_RESULTS | | OK |
| D5 torus | pilot seed 690 lifted signed to ~0.1 at 1e-3; AO_bal_hi 6/8 (0.988), 0.46-0.63; SO_bal_hi 0.15-0.21 | GENERALITY_PREREG l.37-39; ANALYSIS | 0.455-0.631; 0.147-0.205 | OK |
| D6 | 0.99 -> 0.98 weighted; 0.87 -> 0.18 unweighted; D1 -0.003 NO EFFECT; torus 1.00 -> 0.69; frozen probe 0.802 v 0.998 | ACCOUNTS T1; GENERALITY secondaries; WRAP_EXISTENCE sensitivity | | OK |

## 6. Appendices F-K

| location | claim as written | source | source value | verdict |
|---|---|---|---|---|
| F task / metric | L32 D12; A2f; CE-optimal 1.000; chance 0.500; floor 0.594 | DYCK_MDEPTH_RESULTS | | OK |
| F recipe | rank 2, base 32 / 10000; AdamW 1e-4, wd 0.01, batch 128; 560k / 3x; n 8, seeds 0-7 | DYCK_MDEPTH_PREREG l.23-26; train_dyck.py | | OK |
| F S1 | +0.353 / +0.130 / +0.045 / +0.024, 8/8, p 0.0078; MDE 0.003 / 0.025 / 0.009 / 0.007 | DYCK_MDEPTH_RESULTS S1 (exact-t) | | OK |
| F primary | ceiling 0.997-0.998 / 1.000; +0.002 CLOSES; +0.168 depth extrapolation | DYCK_MDEPTH_RESULTS | | OK |
| F mixture / base / budget | +0.110 (8/8); +0.001 / -0.001; +0.021 | S2, S3, S4 | | OK |
| F protocol | 2026-09-26; predates checks; four smoke runs; repro weight-exact | prereg l.64-69; RESULTS repro line; git log | | OK |
| G task | 58 words, 3 synonyms, T 1024 words (~132 steps); 200 eval walks | TEXTWORLD_RESULTS; PREREG l.31 | | OK |
| G registered | 0.969 +/- 0.054 (7/8) v 0.505 (0/8), 0.772 (0/8); +0.464, p 0.0002, Fisher 0.0014 | TEXTWORLD_ANALYSIS | | OK (pilot seeds pooled, disclosed in the next sentence) |
| G fresh | 0.979 +/- 0.048 (6/6) v 0.505 (0/6), +0.474, p 0.0022; RoPE 2L 0.793 | CORRECTED block | | OK |
| G re-score | 0.969 -> 0.993; 0.505 -> 0.506; 0.772 -> 0.772 | AUDIT block | | OK |
| G secondaries | 0.002-0.012 on 8/8; 4/8 clock, 31-38 of 64; B did not fire (4/8) | RESULTS; CORRECTED block | | OK |
| H CARoPE | P6: "any return interval advances each channel by a value in (0, 4]" | APPENDIX_theory_v2 P6(a) | an interval of k tokens is in (0, k]; only the shortest return is in (0, 4] (obs-obs) / (0, 3] (move-obs) | WRONG (error 9) |
| H CARoPE | torus 0.667 (0/8) WORSE THAN INDEX 0.722, -0.055, p 0.040; ring 0.741 v 0.730 UNMEASURED +0.011 p 0.39 | GENERALITY_ANALYSIS | 0.3852 | OK |
| H CoPE | gates 17/16, 15/16; period-2 table 1.000; P7 lattice P = 2 or 3 | WRAP_COPE_RESULTS; theory P7 | | OK |
| H table GENERALITY | 0.887, 0/8; -0.113 p 0.0002; +0.157 p 0.0002 | GENERALITY_ANALYSIS | | OK |
| H table WRAP_COPE | 0.840 / 0.913 / 0.867 / 0.943; 1, 2, 1, 0 of 8; all UNMEASURED | WRAP_COPE_ANALYSIS | 0.8398 / 0.9128 / 0.8671 / 0.9428 | OK |
| H table T4 | 0.914 / 0.873; 1/16 / 0/16; signed 1.000 8/8; index 0.772; L1 NO EFFECT; L2 SIGNED LEARNS BETTER | ACCOUNTS_ANALYSIS | | OK |
| H per seed (32 values) | line and ring lists | ACCOUNTS_ANALYSIS T4 | identical | OK |
| H | at loss < 0.1: 8/16 v 2/16, Fisher 0.054 (post hoc) | solved_thresholds_out.md | 0.0538 | OK |
| H | 1 of 32 pooled post hoc | as Sec. 4.6 | | OK |
| H diagnosis | 40 runs; NONE 20/40; OFF 11, wrap-only 0.55; LATTICE 9 with all 4 SOLVED; snap 0.63 -> 0.90; 11/11; predictions before | COPE_DIAGNOSIS.md l.3-13 | | OK |
| H diagnosis | line margin P / (D_max + 1) ~ 0.016 | ACCOUNTS T4; theory P7 | | OK |
| I table | 6/12 (0.946), 0/12 (0.516); 0/8 (0.406), 0/8 (0.385); 0/8, 0/8; 8/8, 11/11 | runs/toy_wrap/summary.txt; MECHANISM_PREREG exploratory | | OK |
| I T3 | seeds 10-17; 0/8 MAP SOLVED; 0.27 v 0.23 | runs/accounts/t3/summary.txt; ACCOUNTS_PREREG l.42 | | OK |
| J | AUC -0.197, p 0.053, 12/16; -0.106, p 0.19, 10/16; pooled post hoc -0.151, p 0.018 | CLOCK_SPEED(_REP)_ANALYSIS | 0.0533; 0.1871; 0.0177 | OK |
| J | CLOCK_SIDE +0.085 / +0.169 / +0.084, p 0.16-0.53; 0/40 SOLVED; RoPE -0.450 p 0.0011 | CLOCK_SIDE_ANALYSIS | 0.1587-0.5282 | OK |
| K | n 4, seeds 500-503; 1.000 (4/4) v 0.719, 0.860, 0.864; 0/4 STALLED; p 0.029 | FACTORIAL_ANALYSIS | 0.7193, 0.8595, 0.8641; 0.0286 | OK |
| K | floors 0.507 / 0.576 on another stream | score_rank_floors_out.txt | labelled as another stream | OK |
| K | within-128 0.74-0.79 -> 0.90-0.96; long-gap and wrap-only 0.37-0.53 | FACTORIAL strata | 0.735-0.787; 0.900-0.958; 0.367-0.531 | OK |
| K | words: 0.979 v 0.505, +0.474, 6/6 v 0/6, p 0.0022; 8-seed 0.969 v 0.505, p 0.0002; re-score 0.993 | TEXTWORLD | | OK |
| K | Liu et al.: median of 20 runs < 90% for C_4 at 1-15 layers; C_5, C_7, C_8 >= 98.5% from 3 layers | papers/new_2026-10-14/2210.10749/liu2023.txt (Fig. 9 table) | C4 56.8-85.8 (99.3 at 16); C5 >= 99.0, C7 >= 99.4, C8 >= 98.5 from 3 | OK |
| K | Dyck exchange rate ~3 layers | appendix F | | OK |

## 7. Cross-cutting checks

| check | finding | verdict |
|---|---|---|
| Seeds named for every batch | A5 names all but ROBUST (1400-1407) and CANCEL (0-7, pilot 0-1) | error 6, error 4 |
| n per cell | captions and C-tables state n; Fig. K1 n = 4; CoPE T4 n = 16 | OK |
| Independent pre-launch check | SIGN_MATCHED none (prereg has no amendment), as the draft says; DYCK none (disclosed); CANCEL none (not disclosed; its numbers enter Sec. 4.1 and B) | error 4 |
| Pilots pooled | TEXTWORLD 8-seed numbers quoted beside fresh-seed numbers, disclosed (OK); CANCEL index 1L values pool pilot seeds 0-1, undisclosed | error 4 |
| Post hoc pooling labelled | CoPE 1/32 and clock-speed pooled -0.151 labelled post hoc | OK |
| Post hoc presented as registered | none found | OK |
| Declared secondary presented as post hoc | ROBUST loss-0.1 sensitivity (Sec. 4.1 l.130) | error 3 |
| Rounding across main text / appendix / claim map | every repeated number rounds the same way (0.821, -0.177, 0.007 / 0.0070, 0.010 / 0.0101, 0.054 / 0.0538, +0.09 / +0.086, 0.880, 0.843, 0.860, 0.92, 0.887, +0.157, 0.667 / 0.722 / 0.040) | OK |
| Abstract vs main text vs appendix | abstract "8 of 8 for signed" vs lr /3 6/8 (error 11); otherwise consistent | error 11 |
| Sec. 4.1 vs C13 | "every other sign contrast holds at 0.1" vs ROBUST torus 5/8 v 8/8 at 0.1 | error 2 |
| CLAIM_MAP "Do not claim" (general list) | none of the 12 items is claimed; withdrawn readings (drift, fragility, near-wrap design rule, CoPE wrap obstacle, start-not-constraint) are marked withdrawn or unmeasured where mentioned | OK |
| CLAIM_MAP GAUGE "Do not claim" | "cost follows the relative sign", "no cost without an inverse pair": Sec. 4.5 asks it as a question and says not confirmed | OK |
| CLAIM_MAP ROBUST "Do not claim" / say-wording | main text 4.1 and Sec. 6 use the prescribed wording; abstract drops "and in accuracy at scale" | error 10 |
| CLAIM_MAP C2 "replication" | withdrawn (SIGN_MATCHED CORRECTED block); the draft says learnability, not replication | OK |

## Errors, with exact fixes (old -> new)

1. **WRONG -- Fig. K1 caption (DRAFT5 l.327-328).** The dashed line is the mean of the per-stratum best-constant floors
   over the 16 cell-runs (`fig_factorial_strata.py` l.19-21): 0.513. FACTORIAL's own-stream overall floor is 0.509
   (appendix B); 0.507 is the RANK_MI stream (seeds 10-25).
   Old: "the best-constant (predict-blank) floor, averaged over strata and runs (overall 0.507); the n-gram floor 0.576 is not drawn."
   New: "the best-constant (predict-blank) floor, averaged over strata and runs (0.513; overall 0.509 on this stream); the n-gram floor (0.576, computed on another stream of the same generator) is not drawn."

2. **WRONG -- Sec. 4.1 l.118-119.** ROBUST's torus SOLVED contrast fails at loss 0.1 (5/8 vs 8/8, Fisher p 0.20), and the
   sentence precedes the ROBUST paragraph.
   Old: "every other sign contrast holds at thresholds 0.02 and 0.1 and under accuracy >= 0.99 (appendix C)."
   New: "every other sign contrast in Table C10 holds at thresholds 0.02 and 0.1 and under accuracy >= 0.99 (appendix C)."

3. **STATUS-MISLABELLED -- Sec. 4.1 l.130.** SOLVED at loss 0.1 is a declared secondary of ROBUST (`ROBUST_PREREG.md`
   l.79-81), not post hoc.
   Old: "the SOLVED contrast does not survive a 0.1 loss threshold (5/8 vs 8/8; post hoc):"
   New: "the SOLVED contrast does not survive a 0.1 loss threshold (5/8 vs 8/8, Fisher p 0.20; a declared secondary):"

4. **POOLED / unchecked batch -- Sec. 4.1 l.116 and appendix B Gates l.154 (and A5).** CANCEL's index 1-layer values
   0.717 / 0.676 / 0.825 include seeds 0-1, byte-identical to its pilot (`CANCEL_RESULTS.md` CORRECTED block); fresh seeds
   give 0.710 / 0.677 / 0.824. CANCEL had no independent pre-launch check, so using it in the main text also breaks Sec. 2's
   protocol sentence.
   Main text old: "and the one-layer index model's 0.72-0.73 at p_plus 0.5)"
   Main text new: "and the one-layer index model's 0.730 at p_plus 0.5)" (GENERALITY only, a checked batch).
   Appendix B old: "ring p_plus 0.5 / 0.75 / 0.9 0.717 / 0.676 / 0.825 (`CANCEL_RESULTS.md`)"
   Appendix B new: "ring p_plus 0.5 / 0.75 / 0.9 0.710 / 0.677 / 0.824 on CANCEL's six fresh seeds (0.717 / 0.676 / 0.825 with its two pilot seeds; CANCEL had no independent pre-launch check; `CANCEL_RESULTS.md` CORRECTED block)".
   A5: add a row "CANCEL | B (index gate) | ring p_plus 0.5 / 0.75 / 0.9 / 1.0 | path 1 layer; index 1-3 layers | 0-7 | `train_cancel` / -- | no | seeds 0-1 (reused; CORRECTED block)".

5. **WRONG -- A4 "Dropout re-score" (APPENDIX l.83-84).** Not declared in ACCOUNTS, TORUS_START, GAUGE; ROBUST declares
   none (invalid at 2 layers, `ROBUST_PREREG.md` l.92-93); FACTORIAL declared it for 1-layer cells but its results file does
   not report it; SIGN_MATCHED's is in `docs/audits/2026-10-04/DROPOUT_RESCORE.md`, not its results file.
   Old: "(declared secondary in the batches after 2026-10-04): attention scaled by 1/(1 - p) at eval. It changed no label in any batch cited here (each results file states it)."
   New: "attention scaled by 1/(1 - p) at eval: a declared secondary in RING_DIAL, RING_BUDGET, WRAP_EXISTENCE, GENERALITY, MECHANISM and WRAP_COPE (no label changed; each results file states it), and applied to SIGN_MATCHED and TEXTWORLD by the 2026-10-04 audit (`DROPOUT_RESCORE.md`; no verdict changed). ACCOUNTS, TORUS_START, GAUGE and ROBUST were not re-scored (ROBUST: the correction is invalid at 2 layers); FACTORIAL declared it for the 1-layer cells and does not report it."

6. **MISSING -- A5 (and A1, A3): ROBUST.** Add to A5: "ROBUST | 4.1, C13 | ring 0.5 (+1/-1; four recipes); torus 2 layers d 256 | SignedAO / AbsAO (ring), SO4 / AO4 (torus) | 1400-1407 | `train_gauge`, `train_variant` / `run_robust.sh` | yes (Amendment 1, `docs/audits/2026-10-16/robust_audit.md`) | seed 1490 (ring; torus timing 1490-1492)". A1 layers row old: "1 (appendix K also 4 real layers and one block looped x4; appendix F 1-4 layers)" -> new: "1 (appendix K also 4 real layers and one block looped x4; appendix F 1-4 layers; ROBUST torus 2 layers, d_model 256, C13)". A3: add "ROBUST: ring lr 3e-3 / 3.3e-4 and wd 0 variants; torus 2 layers d 256 with `--fast-attn` (C13)".

7. **UNSOURCED -- Sec. 2 Protocol l.63.** No file records that the 2026-09-30 audit of the sign batch was independent; A5
   itself says only "audit after the run (2026-09-30)".
   Old: "that batch had a pre-launch gate by the authors and an independent audit after the run."
   New: "that batch had a pre-launch gate by the authors and an audit after the run (2026-09-30), which reproduced its probe."

8. **WRONG (inherited from the source) -- C9 caption l.317.** 87.5% is the pre-launch audit's sample
   (`gauge_audit.md` l.17, 36); on the registered eval stream GAUGE_ANALYSIS counts 138,800 backtracks of 155,456 revisits
   (89.3%). (`GAUGE_RESULTS.md` l.42 carries the same 87.5%.)
   Old: "on +1/-1, 87.5% of revisits are backtracks;"
   New: "on +1/-1, 89% of revisits are backtracks (138,800 of 155,456 on the eval stream; 87.5% in the pre-launch audit's sample);"

9. **WRONG -- appendix H, CARoPE paragraph l.625.** P6(a) bounds an interval of k tokens by (0, k]; only the shortest
   return is in (0, 4] (observation-observation; (0, 3] move-query to observation-key).
   Old: "any return interval advances each channel by a value in (0, 4] < 2 pi, so no channel reads every return at phase 0;"
   New: "the shortest return advances each channel by a value in (0, 4] < 2 pi, so no channel reads every return at phase 0;"

10. **CLAIM-MAP -- abstract l.16.** The claim map's ROBUST wording is "holds under wd 0 and lr /3 and at 2 layers d 256;
    shrinks at lr x3 and in accuracy at scale"; the abstract drops the accuracy-at-scale half (4.1 and Sec. 6 keep it).
    Old: "The cost holds without weight decay, at a third of the learning rate and in a 2-layer d 256 torus model; it shrinks at three times the learning rate."
    New: "The cost holds without weight decay, at a third of the learning rate and in a 2-layer d 256 torus model; it shrinks at three times the learning rate and, in accuracy, in the larger model."

11. **WRONG (scope) -- abstract l.11-12.** Signed steps solve 6/8 at lr /3 (ROBUST lrd3), a cell the abstract cites two
    sentences later.
    Old: "barrier steps solve the map on 0-4 of 8 seeds per batch against 8 of 8 for signed steps."
    New: "barrier steps solve the map on 0-4 of 8 seeds per batch against 8 of 8 for signed steps at the default recipe (6 of 8 at a third of the learning rate)."

12. **WRONG (cosmetic) -- figure list l.312-313.**
    Old: "Appendix tables C1-C12 (and C9b) are in `APPENDIX_draft5.md`."
    New: "Appendix tables C1-C13 (and C9b) are in `APPENDIX_draft5.md`."

Not errors, noted: Fig. 1's "64-98%" (mean of per-run shares) and RING_DIAL's "63-82%" (pooled shares) are different
aggregations and are not mixed in the draft. `TEXTWORLD_ANALYSIS.txt` prints "floor 0.512" (the gate's sample) while the
draft uses the results file's 0.505 on the eval set, which is the right one (`TEXTWORLD_PREREG.md` l.76).
