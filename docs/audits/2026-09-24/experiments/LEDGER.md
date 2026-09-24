# Experimental-record audit: claim ledger (auditor: EXPERIMENTS / EVIDENCE GAPS), 2026-09-24

Scope: claims the project currently presents as citable -- `RESULTS_INDEX.md` (last regenerated
2026-09-06/11), the top blocks of `CLAUDE.md` (2026-09-21..23), `.claude-memory/project_state.md`,
the live shared report `report/language_summary.html` (2026-09-23), and `report/report.tex` /
`report_short.tex` (built 2026-09-20; a row-by-row ledger for those two is in
`REPORT_TEX_LEDGER.md`, produced by a helper and spot-checked).

Tags: **V** = recomputed here from a JSON / log / per-seed table; **J** = judged from the results
file's own text. "t-MDE" = the MDE with the small-sample t multiplier
(t_{.975,n-1} + t_{.80,n-1}) instead of the project's z-based 2.8: it is 1.91x the project's
MDE at n=3, 1.49x at n=4, 1.33x at n=5, 1.16x at n=8, 1.10x at n=12. Helper: `tstat.py`.

Status key: SOLID / NEEDS CONTROL / UNDERPOWERED / CONTRADICTED BY LATER RESULT /
RETRACTED BUT STILL CITED.

## A. Navigation and torus line

| # | claim | file | n | floor? | converged? (criterion) | matched or OOD | rule 9 / loss overlap | effect vs MDE | one batch | status | V/J |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Position +0.461 on the paper task; index arms "on the 0.506 floor" (RESULTS_INDEX headline table, BASELINE_TABLE, CLAUDE.md top "navigation +0.461 at T=128->128", html "It agrees with MapFormer's own navigation result ... +0.461") | `INDEX_BASELINE_PAPER_TASK_n8.json` | 8 | yes 0.506 | **NO**: paper 16-epoch LinearLR; index arms 0.530/0.509 never left the floor (rule 10) | matched (T=128) | no overlap | +0.4615, MDE 0.027 (V) | yes | **CONTRADICTED BY LATER RESULT** -- `PAPER2X2_RESULTS.md` under a converged recipe gives **+0.243**, index RoPE 0.805. report.tex uses +0.243; the html, RESULTS_INDEX (as an n=3 table labelled "matched recipe"), BASELINE_TABLE, axes_measured.tex, mapformer_math.tex, project_state and CLAUDE.md still headline +0.461 (file:line in `PROPAGATION.md`). The companion claims "encoding ~0.003" / "PoPE is inert without path integration" are superseded too: converged, the encoding effect is detectable at every length (-0.049 / +0.114 / +0.189) | V |
| A2 | Position +0.243 (r=2) / +0.258 (r=4) at T=128 | `PAPER2X2_RESULTS.md`, `PAPER2X2_RAW_CONTRASTS.json` | 8 | yes | fixed 300-ep cosine recipe; index RoPE final loss 0.68-0.83 and still falling under the decayed LR (s0: 0.859 -> 0.834 over the last half) -- Amendment-2 classes recomputed from logs: RoPE 8/8 STALLED, PoPE-Flat 8/8 STALLED (loss 0.73-0.96), Vanilla r=2 4 SOLVED / 4 STALLED, r=4 arms 8/8 SOLVED | matched | groups do not overlap; loss-matched uninformative (stated) | +0.243, MDE 0.038, 8/8 (V) | yes (RoPE/Vanilla_r4 bitwise shared with `runs/sign`) | **SOLID at this budget**; the index ceiling is unmeasured (index arm slowly descending) | V |
| A3 | Position +0.390/+0.359 at 4x/8x; encoding +0.114/+0.189 beyond training length | same | 8 | yes | as A2 | **OOD** | -- | detectable | yes | SOLID as robustness, not capability | J |
| A4 | "Index models exceed the floor only at recurrence interval 1-2" (RESULTS_INDEX headline residual) | `REVISIT_DISTANCE.md` | -- | -- | old 16-ep recipe | matched | -- | -- | -- | **CONTRADICTED BY LATER RESULT**: under the converged recipe index RoPE leads index PoPE at 5-16-step revisits (`REVISIT_2X2`, project_state) | J |
| A5 | Match-Query: path 0.730 +/- 0.247 (n=5) vs index 0.154; 128^2 0.823 vs 0.192; context destruction 0.918 -> 0.074 | `MATCH_QUERY_SCALE.md`, `LOOP_HEADROOM.md` | 5 / 3 / 8 | chance 0.0625 | LinearLR default; re-confirmed under cosine in LOOP_HEADROOM (index 0.108 vs 0.456, n=8) | matched | no seed overlap | huge | yes | **SOLID**. Gaps: index never run at lr 1e-3; the architecture-matched RoPE index arm never run on MQ (RESULTS_INDEX open #5); MQ final losses irreproducible across batches (1.91 vs 3.48) | J |
| A6 | Loop x path integration super-additive on MQ: paired +0.414, interaction +0.315; `r=4+loop x4` 0.986 with 8/8 seeds >= 0.941, interaction +0.149 (RESULTS_INDEX "What else is citable"; axes_measured "the one super-additive pair") | `LOOP_HEADROOM.md`, `MQ_RANK_2X2.md`, `REFINE_RESULTS.md` | 8 | chance 0.0625 | 300 ep, lr 3e-4 | matched | -- | interaction +0.315 vs MDE 0.281 (t-MDE 0.33 -> fails) | yes | **CONTRADICTED BY LATER RESULT** for the interaction and the floor: `REFINE_RESULTS.md` (08-31) found same-seed retrains on MQ drift 0.185 per seed (bimodal basins), retracted "8/8 >= 0.77" (pooled 0.803 +/- 0.200, 1/16 failures) and ruled that every MQ comparison be read UNPAIRED -- the paired interaction and the one-batch "8/8 >= 0.941" for r=4+loop are the same kind of statistic. At lr 1e-3 the loop's p=0 advantage also fell +0.373 -> +0.146 (`MQ_NOISE_2X2_C2.md`). **What survives: loop main effect unpaired +0.346, t 3.75** | J (arithmetic V) |
| A7 | Rotation kills the effect, allocentric recoding restores it: +0.438 -> +0.050 -> +0.488 (report.tex abstract "Integrable input") | `KNOB_SWEEP_n8.md`, `ALLOCENTRIC_RECODING.md` | 8 | yes | **16-epoch LinearLR**: the baseline row IS A1 (RoPE 0.529 at floor); rotate Vanilla 0.558 near floor; allocentric RoPE 0.508 at floor | matched | no | as quoted | yes | **NEEDS CONTROL** -- all three numbers are at the recipe report.tex itself says is superseded; rerun rotate/allocentric under the PAPER2X2 recipe | J |
| A8 | H=12 allocentric +0.26 to +0.38 | `H12_BUDGET_CURVE.md` | 3 | 0.508 | bimodal basins; r(loss,acc) = -0.996 | matched | acc = loss | range | per budget | UNDERPOWERED / budget-sensitive (stated) | J |
| A9 | MiniWorld: aliasing story "FALSIFIED, sign INVERTED" (+0.178 at n_obs=16 vs +0.305 at 256); map-extent threshold (-0.010 / +0.015 / +0.305); "REAL, CONVERGED ... capability, not training speed" | `ALIASING_CONTROLLED.md`, `VISITS_TEST.md`, `POSITION_EFFECT_CONVERGED.md`, `CROSSOVER_CONVERGED.md`, `runs/alias_sweep`, `runs/alias_follow` | 3-5 | noise floor 0.150 | **"N/N flat"** = abs slope < 5e-4/epoch over final 10% (`experiment_audit.py:39`). **Reclassified here from the logged loss curves with Amendment 2's classes: the +0.305 endpoint's index arm (g32 n256, 800 ep) is DESCENDING on 3/3 seeds (loss 0.34-0.40, down from 0.66-0.69 at 400 ep) yet 'flat' on 3/3; the +0.015 threshold point's index arm (g16 n64) is DESCENDING 3/3 (0.13-0.15); the +0.178 comparator's index arm (n16) is STALLED 3/3 (0.42-0.50).** | matched (T=512) | r = -0.996; ranges don't overlap | +0.305 sd 0.048 (t-MDE 0.147) | cross-batch, repro-controlled | **NEEDS CONTROL**: both ends of the aliasing contrast and the threshold rest on index arms that were still learning; the inversion (and the threshold's size) can move with budget, as it already did once (+0.374 -> +0.305) | V (logs) |
| A10 | MiniGrid 2x2x2: two best arms are index models, MapWM last | `MINIGRID_FULL_2X2X2.md` | 8 | yes | 50 epochs, cached buffer; no convergence check reported | T=512/1024 (trained length not re-checked) | not reported | yes | yes | NEEDS CONTROL (budget/convergence unreported, rule 5/10) | J |
| A11 | **Sign**: Abs - Signed -0.363 / -0.280 at T=512/1024 (12/12); "monotone beats index nowhere at matched loss" | `SIGN_ABLATION.md` | 12 | RoPE row | 300 ep cosine. Amendment-2 classes from logs: Signed_r4 12/12 SOLVED; Abs_r4 4 SOLVED / 8 STALLED; Pos_r4 8/3/1; CARoPE_r4 6/6 -- so at training length the monotone cost is largely a learnability (stall) effect | **OOD** (train 128). Matched-length evidence exists only in LOSS (12/12 worse) and raw T=128 acc (Abs 0.946 +/- 0.070 vs 1.000, about one MDE) | loss-matched verdicts pool-dependent (report.tex admits) | large, 12/12 | yes | **NEEDS CONTROL** (a trained-at-T=1024 Abs_r4 arm; Vanilla_r4 at 900 ep already exists and solves 8/8) | J |
| A12 | **Rank**: r=4 +0.085 at T=1024 (8/8, t 3.57); "use r=4" | `RANK_SWEEP.md` | 8 | -- | T=128 train, old losses did not overlap | **OOD** (94% from short-gap revisits late in the sequence) | no overlap | +0.085 | yes | Old number NEEDS CONTROL; matched-length control **in progress**: `RANK_MATCHED_e900.md` r4 0.997 vs r2 0.894, 8/8 solved vs 0/8, perm p 0.0003 -- but **UNREADABLE as registered** (4 r=2 runs descending); continuation `runs/rank_matched_e900c` running; projected-r=2 frozen 0.995 (untracked `RANK_PROJ_FROZEN.md`) shows representability | V (e900 table) |
| A13 | r=2 learns a skewed basis (opposition 0.495 vs 0.092; C3 fixed by r=4) | `ACTION_GEOMETRY.md`, `PAPER_FIG4_REPRO.md` | 8 | -- | T=128 | descriptive | -- | -- | yes | SOLID as description; at T=1024 confounded with r=2 stalling (RANK_MATCHED geometry) | J |
| A14 | **Level 1.5 / InEKF**: helps only beyond training length, loss-matched +0.062 / +0.124 at T=512/1024; "stabilisation, not inference" | `L15_ABLATION.md`, `L15_LOOP_2X2.md`, `MQ_NOISE_2X2*.md` | 5 / 12 | -- | 300 ep; NoCorr "2/5 flat" (flat criterion) | **OOD** | loss-matched t 3.08/3.83 (n=5) | L15_LOOP +0.083 t 2.79, just under | per batch yes | **NEEDS CONTROL** (never had a matched-length arm) | J |
| A15 | Forget gate +0.086 at r=2, T=1024, 8/8 loss-matched; mechanism unidentified | `FORGET_GATE.md`, `FORGET_CONTROL.md`, `LAMBDA_TRACE.md` | 8 | -- | 300 ep | **OOD** | r = -0.311 at T=1024 | detectable | yes | **NEEDS CONTROL**; the registered forget-clock test (`FORGET_CLOCK_PREREG.md`) was deleted and, as registered, reads both tasks at OOD lengths, so re-running it as-is would still not give a matched-length answer | J |
| A16 | PoPE wrapping: length half holds 3/3 (octave refuted) | `POPE_WRAPPING.md` | 8 | -- | "grid 16 bimodal" | **OOD** | loss-matched +0.077..0.095 | -- | yes | **NEEDS CONTROL** | J |
| A17 | Loop on the torus "is convergence, not representation" (raw +0.052 at T=128, loss-matched +0.006) -- in both report abstracts, conclusions and the contributions table; LoopedSampled flattens the count curve 0.178 -> 0.001, 0.998 at one pass | `L15_LOOP_2X2.md`, `LOOP_SAMPLED.md` | 12 / 5 | -- | loss 0.008 | matched (T=128) | **the 'convergence' reading is a loss-matched residual at matched length -- the inference the project withdrew on 2026-09-23 (R12): it cannot separate 'trains faster' from 'more capable'** | -- | yes | **NEEDS CONTROL** for the convergence reading (unsupported, not contradicted); the count-curve flattening itself is SOLID | J |
| A18 | Gated content gate: ratio 4.16x (8/8), accuracy gain unmeasured | `GATED_RESULTS.md` | 8 | 1.35x floor | "8/8 flat" (flat criterion) except Gated_r2 5/8 | OOD for the accuracy half | -- | null stated as unmeasured | yes | SOLID as stated (a null, correctly worded) | J |
| A19 | Recency crossover: forcing monotone costs -0.280 on torus, -0.004 on recency; interaction ~+0.28 | `RECENCY_RESULTS.md`, `SIGN_ABLATION.md` | 8 / 12 | chance 0.0625 | recency "flat fraction 1/8-4/8", final-10% slope -0.002..-0.005/ep | **both halves OOD**: torus at 8x, recency moved to T=2048 (2x) because T=1024 is at ceiling | loss-matched stated | -- | two batches | NEEDS CONTROL (an interaction of two extrapolation readouts at different ratios) | J |
| A20 | Index code cannot count contextually: +0.750 at T=1024 (8/8) | `RECENCY_RESULTS.md` | 8 | yes | as A19 | matched (train T=1024) | -- | MDE 0.030 | yes | SOLID (a CoPE reproduction, correctly scoped) | J |
| A21 | EM - WM on recency -0.375 (0/8) | `RECENCY_EM_RESULTS.md` | 8 | 0.0625 / 0.0771 | EM final loss 0.92-1.89 (unconverged by design; registered readout = epochs to threshold) | matched | -- | MDE 0.154 | yes | SOLID as a fixed-budget learnability result (framed as search) | J |
| A22 | EM_P0 - WM on the paper task +0.035/+0.070/+0.085 at l=512/1024/2048 | `EM_WM_THEORY.md` | 8 | -- | -- | **OOD** | -- | MDE 0.031-0.034 | yes | NEEDS CONTROL (OOD only) | J |
| A23 | Phase freedom +0.146 (22/24), fresh seeds +0.113; PAIRSPLIT +0.091 / +0.124 (n=48); NOLEAK/WARM installed rewind 1.000 (8/8) | `MAGONLY_RESULTS.md`, `D5_RESULTS.md`, `PAIRSPLIT_RESULTS.md`, `NOLEAK_RESULTS.md`, `WARM_RESULTS.md` | 24 / 48 / 8 | yes | fixed budget | matched (recency T=1024) | -- | detectable, fresh-seed replication | yes | SOLID | J |
| A24 | COUNTER: identical counter -> MapWM 1.000, EM 0.740, TEM 0.337 | project_state | 4 | -- | -- | -- | -- | large gaps | yes | UNDERPOWERED in principle (n=4, t-MDE 1.49x), gaps large | J |
| A25 | Family tree: path over index +0.115 / +0.205 (3/3); non-commutativity +0.013 (MDE 0.008) | `FAMILY_TREE_RESULTS.md`, `N3_AUDIT.md` | 3 | hub floor 0.146-0.163 | -- | trained 64, eval 64/128 | -- | NC axis: t-MDE 0.015 > 0.013 -> **not detectable** | yes | UNDERPOWERED (NC axis) | V (arithmetic) |
| A26 | Stitching: paired +0.131 +/- 0.024 vs index -0.005 (floor exactly 0) | `STITCH_ATTENTION.md` | 3 | exact 0 | -- | matched | -- | large vs sd | yes | SOLID but n=3 | J |
| A27 | Hierarchy: +0.136 compositional (unmeasured), parity +0.012 ceiling | `HIER_RECHECK.md`, `HIER_PARITY.md` | 8 | -- | recipe-limited task | matched | -- | inside MDE | yes | UNDERPOWERED (stated correctly) | J |
| A28 | Timing benchmark 2.6-3.3x vs 14.5x vs 120x | `TIMING_BENCHMARK.md` | -- | -- | -- | -- | -- | -- | -- | SOLID | J |
| A29 | MoR oracle router +0.007 vs seed sd 0.152 | `LOOP_DEPTH_STRATA.md` | 8 | -- | eval-only | -- | -- | -- | -- | SOLID | J |
| A30 | lm200 Level15 0.990 vs Vanilla 0.742 | `LM200_CORRECTED_MULTISEED.md` | 3 | -- | -- | OOD T=512 | ExtraHead ties | -- | yes | NEEDS CONTROL (no context-destruction ablation ever; interpretation withdrawn) | J |
| A31 | Addition: signed MapFormer only code to learn 30 digits (3/3), holds to ~33-38 | `SAMEBLOCK_*`, project_state | 3 | control gate **FAILED** (role-format oracle 0.697) | -- | OOD in digits | -- | -- | -- | NEEDS CONTROL | J |

## B. Language line (Dyck, code, Bach, Indirect, enwik8)

| # | claim | file | n | floor? | converged? (criterion) | matched or OOD | rule 9 / loss | effect vs MDE | one batch | status | V/J |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | **Dyck ladder**: position +0.290/+0.209/+0.159/+0.168 at L32 D12, 1-4 layers, fixed width; "path integration buys something depth does not"; html: "every run converged", "at the training length" | `DYCK_LADDER_RESULTS.md/.json`, `runs/dyck_ladder` | 8 | chance 0.5, no-memory ~0.51 | **median final-window slope >= -0.005/1k** (flat-type) under a cosine schedule decayed to 10% at step 4375 (the paper budget, ~65 s/run): the slope is small because the LR is. Index arms end 0.03-0.18 above the CE floor 0.802 (RoPE 0.975/0.885/0.852/0.832 by depth, still falling with depth) | **matched LENGTH but 3x the training NESTING DEPTH** (train cell is L32 **D4**). At the true training cell L32 D4 the effect is +0.293/+0.081/+0.048/**+0.019** and index RoPE-4L reaches 0.979 | train loss: RoPE-4L 0.832 is LOWER than MapWM-1L 0.848 / MapPoPE-1L 0.850 while A2@D12 is 0.755 vs 0.915/0.988 -> the gap is generalisation in depth, not fit | all four clear t-MDE (V) | yes | **NEEDS CONTROL**: (i) train with nesting depth up to 12 (matched depth) -- by the project's own lesson this is an extrapolation claim in the depth variable; (ii) extend the index arms' budget, since "stop climbing" rests on a slope rule under a decayed LR and two depths | V |
| B2 | E2: RoPE base confound "dead" (base 32/128 close -3%..+3% of the gap) | `DYCK_DEPTH_RESULTS.md/.json` | 8 | -- | slope rule | registered read at L128 D12; **recomputed here at every cell from the JSON: base 10000/32/128 give RoPE-1L 0.627/0.615/0.629 at L32 D4 and 0.612/0.604/0.613 at L32 D12 (PoPE 0.627/0.632/0.632)** | -- | null at all three cells | own batch | **SOLID** for 1 layer at every cell; untested at 3-4 layers at fixed width (weak prior that it matters) | V |
| B3 | Dyck MapPoPE - MapWM +0.073 (1L, 7/8) and +0.050 (2L, 8/8); html "PoPE's encoding also improves MapFormer ... never worse" | `DYCK_LADDER_RESULTS.json` | 8 | -- | B1 | depth-OOD cell only; **at the training cell L32 D4 it is +0.002 / -0.002 / -0.003 (1/8, p 0.08) / -0.001** | -- | 1L: +0.073 vs MDE 0.070, **t-MDE 0.081 -> fails**, p 0.022; 2L: p 0.0002 | yes | 1L UNDERPOWERED; the "improves MapFormer on Dyck" claim holds only at 3x depth | V |
| B4 | F1 inverts the effect's shape: A2 +0.357 -> +0.136 while F1 +0.075 -> +0.306 | `DYCK_DEPTH_RESULTS.md` | 8 | yes | slope rule | L32D4 -> L128D12 (4x length AND 3x depth; html says "4x length") | -- | -- | yes | SOLID (metric comparison on the same checkpoints; numbers are from the narrow-width 1L arms) | J |
| B5 | MapFormer-1L - RoPE-2L +0.370 on F1 replicates the paper (+0.44) | `DYCK_RESULTS_bs128.md` | 8 | **F1 no-memory floor 0.884 > MapWM 0.868** | slope rule | L128 D12 (OOD) | -- | 8/8 | yes | SOLID as a replication of the paper's ordering; html presents it without the floor, against its own "every result beside a no-memory score" standard | J |
| B6 | Code C1: at matched length (2048) the encoding effect is -0.0030 (MDE 0.0046) -- the OOD -3.694 is retracted | `CODE_RESULTS.md`, `runs/code2048` | 3 | 0.858 no-memory (brackets) | **no**: constant LR, val slope negative at 36k | matched | train losses overlap for MapPoPE/PoPE | unmeasured | yes | **SOLID as a retraction** | V |
| B7 | Code C1 "REVERSAL": position +0.0055 (MDE 0.0033) and MapPoPE - PoPE +0.0033 (MDE 0.0031) "DETECTABLE against path integration" | same | 3 | -- | not converged | matched | -- | readout is **`best_val_bpc`** = min over 36 evals of **40 windows (~5.7% of val)**, the readout that produced the Table-5 error. t-test: position p 0.043 (t-MDE 0.0063 > 0.0055); MapPoPE-PoPE p **0.099** (t-MDE 0.0060); MapPoPE-MapWM -0.0052 p 0.056 | yes | **UNDERPOWERED** (the reversal); rescore the `.final.pt` checkpoints on the full val file | V |
| B8 | Code C2: with the envelope, PoPE-Decay - RoPE-Decay +0.0079, MapPoPE-Decay - PoPE-Decay +0.0063 (both "detectable against") | `CODE_DECAY_RESULTS.md` | 3 | -- | not converged | matched (512) | -- | p 0.052 / 0.038; t-MDE 0.0101 / 0.0067 -> **both fail** | yes | UNDERPOWERED | V |
| B9 | "RoPE + 48-param envelope is the best of all eight code models" (html, CODE_DECAY) | same | 3 | -- | not converged | matched | -- | vs MapWM-Decay +0.0034, **p 0.35**; four of the eight arms are in another batch | partly cross-batch | **UNDERPOWERED** | V |
| B10 | Envelope "improved all four code models on every seed, even at the training length" | same | 3 | batch floor 0.0021/0.0028 | not converged | matched | -- | 12/12 signs; per arm p 0.08 / 0.019 / 0.12 / 0.023 (2 of 4 pass a t-test) | **cross-batch** (`runs/code_decay` vs `runs/code`) | UNDERPOWERED / cross-batch (the file says "provisional") | V |
| B11 | Bach: PoPE - RoPE -0.032 (5/5) replicates (paper -0.019) | `JSB_RESULTS.md` | 5 | -- | best-val checkpoint (early stop at 1000-1750/3000) | matched (full context) | r = +0.064 (test NLL not loss) | detectable | yes | SOLID | J |
| B12 | Bach: MapPoPE - MapWM -0.0165 (5/5, MDE 0.0111) | same | 5 | -- | as B11 | matched | -- | t-MDE 0.0148, clears | yes | SOLID | J (arith V) |
| B13 | html "Detectable: adding path integration to PoPE is not [a good idea] on clock-like tasks" | `JSB_RESULTS.md`, `CODE_RESULTS.md` | 5 / 3 | -- | -- | matched | -- | Bach +0.0111 vs MDE 0.0116 (**unmeasured**, file says so); code +0.0033 p 0.099 (B7) | -- | **UNDERPOWERED** -- the "Detectable" chip is not supported on either task | V |
| B14 | 48-parameter envelope repairs Bach extrapolation: MapPoPE 4.616 -> 0.622, PoPE 1.597 -> 0.626 (5/5) | `DECAY_RESULTS.md` | 5 | -- | best-val | **OOD** (robustness, labelled) | -- | MDE 0.417 | decay arms have no same-batch baseline (file's audit note) | SOLID as a robustness repair | J |
| B15 | "...as well as a 786k-parameter alternative does" (0.622 vs 0.616) | same | 5 | -- | -- | OOD | -- | equivalence asserted from non-significance; the decay arm is **detectably worse** than the phase at 512-1024 (+0.0231, MDE 0.0140) and **costs in distribution** (+0.018 / +0.026, 0/5, detectable) -- neither caveat is in the html | **cross-batch** (T3 arm in `runs/jsb_forced`) | UNDERPOWERED / over-stated | J |
| B16 | Augmentation +0.107 NLL (5/5); "0.394 against their published 0.489" | `AUG_RESULTS.md` | 5 | -- | -- | matched | -- | detectable | yes | SOLID within batch; the headline compares an augmented model with a published non-augmented number from another pipeline (ours is 0.012 worse than theirs un-augmented) | J |
| B17 | Indirect Indexing replicates at 200k (7/8); 100k budget 1/8 | `INDIRECT_RESULTS_200k.md` | 8 | -- | late-transition; budget decides | matched | -- | -- | yes | SOLID (0.965 is mean among solvers, vs paper all-run 0.948 -- stated) | J |
| B18 | Indirect: path integration faster (6/7, directional); 3x more padding-robust (0.430 vs 0.149, 6/7, uncontrolled) | same | 7 | -- | -- | padding = OOD | -- | not established | yes | UNDERPOWERED / NEEDS CONTROL (both labelled so) | J |
| B19 | PoPE ablation: NoSigma does not blow up (-0.0108 vs RoPE +3.5885) -> bound account refuted; implementation matches the authors' to 1.7e-06; batch floor 0.0021/0.0028; Table 5 3 of 4 cells agree | `ABLATE_RESULTS.md` | 3 | -- | not converged (-0.002..-0.003 bpc/1k) | refutation is OOD; Table 5 cells matched | -- | NoDelta effect < batch floor | yes | SOLID as corrected (4th cell unmeasured) | J |
| B20 | JSB is overfitting-limited 3x | `AUG_RESULTS.md` | 5 | -- | -- | matched | -- | -- | yes | SOLID | J |
| B21 | JSB length: MapWM - RoPE -0.662 at 2-4x; MapPoPE collapse | `JSB_LENGTH_RESULTS.md` | 5 | -- | -- | **OOD** (labelled robustness) | -- | MDE 0.335 | yes | SOLID as robustness | J |
| B22 | CROSS: envelope damage splits steepness (+0.136 of +0.281) / metric (+0.145) | `CROSS_RESULTS.md` | 8 | chance 0.5 | residual confounded with convergence, r = -0.995 (stated) | Dyck distance >= 9 | -- | 8/8 | yes | NEEDS CONTROL (convergence confound, stated) | J |
| B23 | html "Averaged metrics favour the more local model. Six cases, always the same direction." | none found | -- | -- | -- | -- | -- | -- | -- | **UNSOURCED**: no results file, script or memory note names the six cases (violates "commit the script for every number") | J (grep of all *.md) |
| B24 | Code floor: a no-memory model gets 0.858 of brackets; 99.7% close within 8 levels | `CODE_GATES.md` | -- | is the floor | -- | -- | -- | -- | -- | SOLID | J |
| B25 | enwik8: MapPoPE - RoPE -0.0058, t 3.49 (n=3); singles n=1; bf16 not licensed | `enwik8_long/*.json`, `BF16_RESULTS.md` | 3 / 1 | -- | 36k iters | matched | -- | t-MDE ~0.0080 > 0.0058 | yes | UNDERPOWERED (enwik8); bf16 SOLID | J |
| B26 | "Path integration beats ordinary positions on Dyck-2, and depth doesn't close the gap" as THE language headline; "the two positive results that survive are matched-length" (CLAUDE.md, project_state, html) | B1 + A1 | -- | -- | -- | -- | -- | -- | -- | **Both legs weaker than stated**: navigation leg is A1 (superseded recipe), Dyck leg is depth-extrapolation (B1). The dividing-line framing "every surviving positive result is matched" is not true in the depth variable | V |
| B27 | html "On text, code and music, plain PoPE still wins" | `report/language_summary.html:234` | -- | -- | -- | matched | -- | code: PoPE - RoPE +0.0034 worse (ABLATE, MDE 0.0101), PoPE-Decay detectably worse than RoPE-Decay (p 0.052 by t); enwik8 no PoPE-vs-RoPE result on the page; the page's own table has code as PoPE ~ RoPE | -- | **CONTRADICTED** by the page's own table and by ABLATE / CODE_DECAY; only Bach supports it | J |

## C. Counts by status (58 rows, one status each, by the weakest live part)

| status | count | rows |
|---|---|---|
| SOLID | 24 | A2, A3, A5, A13, A18, A20, A21, A23, A26, A28, A29; B2, B4, B5, B6, B11, B12, B14, B16, B17, B19, B20, B21, B24 |
| NEEDS CONTROL | 17 | A7, A9, A10, A11, A12 (control in progress), A14, A15, A16, A17, A19, A22, A30, A31; B1, B18, B22, B26 |
| UNDERPOWERED | 12 | A8, A24, A25, A27; B3, B7, B8, B9, B10, B13, B15, B25 |
| CONTRADICTED BY LATER RESULT | 4 | A1, A4, A6, B27 |
| UNSOURCED | 1 | B23 |
| RETRACTED BUT STILL CITED | 44 live propagations across 11 documents (`PROPAGATION.md`), plus 2 in report.tex/report_short.tex (`REPORT_TEX_LEDGER.md` #64: the withdrawn Bach-collapse mechanism printed as "what survives", L2104-2106 and sL525-530) |

Several SOLID rows are SOLID only as robustness claims (A3, B14, B21) or as retractions (B6).
The independent report.tex/report_short.tex ledger (66 rows) came out SOLID 26 / NEEDS CONTROL 18 /
UNDERPOWERED 17 / CONTRADICTED 3 / RETRACTED BUT STILL CITED 2; its extra findings are in Section F.

## D. Cross-cutting evidence gaps

1. **The two "surviving positive results" are both weaker than the project says.** The
   navigation leg (+0.461) is measured at a recipe where the index arm never left the floor;
   the project's own converged 2x2 gives +0.243. The Dyck leg is matched in LENGTH but 3x out of
   distribution in NESTING DEPTH, the variable the claim is about; at the true training cell
   depth nearly closes the gap (+0.019 at 4L, index 0.979). By the project's own rule ("every
   extrapolation claim that got a matched control died"), the Dyck claim is the next one owed a
   matched control.
2. **MDE formula is z-based.** 2.8 sd/sqrt(n) understates the MDE by 1.91x at n=3 and 1.33x at
   n=5. Every n=3 "DETECTABLE" in the code line (C1 reversal, C2, envelope) fails a paired
   t-test at 0.05 or sits at p 0.04-0.10. The retraction survives; the reversals do not.
3. **Readout.** The code C1/C2 contrasts use `best_val_bpc` (min over 36 evaluations of 40
   windows) -- min-selection bias plus 5.7% coverage -- the readout that produced the withdrawn
   "Table 5 does not replicate". Full-val rescoring of the saved `.final.pt` files is eval-only.
4. **Flat-type convergence criteria** license every "capability, not training speed" reading in
   the MiniWorld and Dyck lines (Section E).
5. **Recipe-dependence.** Four headline effects were measured at the 16-epoch LinearLR recipe
   (A1, A4, A7 and the knob sweep) or at lr 3e-4 on Match-Query (A6), and the recipe has since
   been shown to change effect sizes by 2x.
6. **Cross-batch comparisons still in the shared report**: B10 (envelope at 512), B15 (786k
   comparison), B9 (best of eight), B16 (published number).

## E. Claims whose convergence rested on a "flat last-N" criterion (deliverable 4)

The criterion comes in three forms, all of which can pass a stalled run:
(a) `experiment_audit.py` FLAT_SLOPE: |final-10% loss slope| < 5e-4 per epoch;
(b) the RANK_MATCHED original ratio rule (final 10% within x of the 10% before), shown inverted
    (called 4 stuck r=2 runs flat, failed a solved r=4 run) and replaced by Amendment 2;
(c) the Dyck rule: median final-window slope >= -0.005 per 1k steps, read at the END of a
    cosine schedule decayed to 10%, where a small slope reflects the LR, not convergence.

| claim | file | form | what the criterion hid |
|---|---|---|---|
| MiniWorld "REAL, CONVERGED POSITION EFFECT ... capability, not training speed" (+0.173 grid 32; 6/6 flat) | `POSITION_EFFECT_CONVERGED.md` | (a) | index arms "flat" at loss 0.41-0.50 while descending 0.03 per 100 epochs; one path seed flat at 0.43 (V) |
| Aliasing inverted (+0.178 / +0.310 / +0.305, "10/10 flat", "6/6 flat at 800 ep") | `ALIASING_CONTROLLED.md` | (a) | **V: the n_obs=256 index arm at 800 ep is DESCENDING on 3/3 seeds under Amendment 2 while 'flat' on 3/3 by rule (a)**; 400 -> 800 ep had already moved the effect +0.374 -> +0.305 |
| Grid-8 crossover -0.010 ("6/6 flat") and the map-extent threshold | `CROSSOVER_CONVERGED.md`, `VISITS_TEST.md`, `MINIWORLD_ENDPOINTS` | (a) | V: grid 8 both arms essentially solved (ok); the grid-16 +0.015 point's index arm is DESCENDING 3/3 (loss 0.13-0.15) while 'flat' 3/3 |
| Dyck ladder "all arms converged"; html "every run converged" | `DYCK_LADDER_RESULTS.md` | (c) | index arms 0.03-0.18 above the CE floor at step 4375, still falling with depth; "index arms stop climbing" is read through it |
| Dyck bs128 replication and E1/E2 "not void" | `DYCK_RESULTS_bs128.md`, `DYCK_DEPTH_RESULTS.md` | (c) | same |
| Gated "8/8 flat" | `GATED_RESULTS.md` | (a) | the claim is a null, so low stakes |
| MapPoPE r=4 (Vanilla 2/8 flat) | `MAPPOPE_R4_RESULTS.md` | (a) | used to argue a convergence confound |
| Level 1.5 ablation (NoCorr 2/5 flat) -> the rule-9 flip | `L15_ABLATION.md` | (a) | the "NoCorr is the worst-converging arm" reading |
| RECIPE_POWER primary metric = converged fraction (already admitted wrong proxy) | `RECIPE_POWER.md` | (a) | ranked C2 > C1 > C0 while accuracy sd ranked C1 best |
| Recency "flat fraction 1/8-4/8" | `RECENCY_RESULTS.md` | (a) | reported, not used for a verdict |
| Rank matched 300-epoch first reading ("gap is training speed") | `RANK_MATCHED_RESULTS.md` | (b) | already withdrawn and replaced |

The only criterion in the repo that separates stalled from converged is Amendment 2's
SOLVED / STALLED / DESCENDING classes, and it exists only for the torus rank line. It needs a
task-appropriate SOLVED threshold (Dyck and code have irreducible loss floors; use loss minus the
measured CE floor) before it can be applied elsewhere.

## F. Additional findings from the report.tex / report_short.tex pass (`REPORT_TEX_LEDGER.md`)

- **Withdrawn mechanism printed as "what survives"**: report.tex L2104-2106 ("the collapse needs
  both an out-of-range accumulator and a kernel that cannot compensate") and report_short.tex
  L525-530 ("PoPE's kernel has non-negative amplitudes ... content cannot shift the peak") -- R6
  (withdrawn 2026-09-20) and R5 (refuted 2026-09-22).
- **R12 logic applied backwards to earlier claims**: "recursion's torus gain is convergence /
  vanishes at matched loss" (both abstracts, both conclusions, the contributions table) and "the
  encoding effect survives loss-matching at every length" (L770, sL210) both read a loss-matched
  residual at MATCHED length, which the project now says is uninformative.
- **Dyck "ordering replicates +0.370 / +0.390"** (L1929-1935, sL501-503) is on F1, at the F1
  no-memory floor, and 1L-vs-2L is width-confounded; the fixed-width ladder (09-23) is not in either
  report.
- **Indirect Indexing**: "solver mean 0.965 against their 0.948" (L1996-1999); the all-run mean is
  ~0.857 by the helper's recomputation.
- **"First clean null for path integration on a natural-sequence task"** (L2014): the contrasts are
  unmeasured (MDE 0.0116-0.0126), not a null (rule 11).
- **Understated**: the MiniGrid encoding (+0.076) and hierarchy main effects clear their MDEs
  (0.025 / 0.026, 8/8) though the report says "no sd recorded ... directional".
- Neither report contains the code line, the PoPE ablation, the Dyck ladder or rank-at-matched-length.
