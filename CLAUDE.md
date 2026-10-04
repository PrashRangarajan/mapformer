# CLAUDE.md -- MapFormer

Loaded into every session and every subagent, so it holds only what every session needs:
conventions, where things live, what is citable, what is withdrawn, the rules and the
invariants. **It is not a log.** History: `docs/LOG.md` (the CLAUDE.md of 2026-09-24,
verbatim). Live state: `.claude-memory/project_state.md`. Numbers: the results files. A
CORRECTED or AUDIT block at the top of a results file supersedes its body.

## The project

Began as a faithful reproduction of Rambaud et al., *MapFormer* (arXiv:2511.19279): a
transformer whose rotary angle is a path-integrated, content-dependent phase
`theta = omega * cumsum(W_out W_in x_t)` instead of the token index. It is now a study of
positional encoding as the mechanism by which a model learns a relational "where" kept separate
from the "what" (TEM's factorisation claim), measured on navigation (torus, Match-Query,
MiniWorld, MiniGrid), algorithmic tasks (parity, k-back recency, Dyck-2, Indirect Indexing) and
sequences (Bach, enwik8, code). Evals redraw the observation map, so every navigation number is
a transfer measurement. The early extensions (InEKF Level 1/1.5/2, predictive coding, grid
cells) ended as negatives or retractions. The taxonomy and the content-dependent rotation are
published (GRAPE 2512.07805, Mamba-3 2603.15569, Selective RoPE 2511.17388); what survives as ours is
empirical: the rank of the content-to-angle map and the navigation regime (sign is a
replication in a new regime).

## Binding conventions

- **Commits are single-author. Never add a `Co-Authored-By` line.** Binding; it overrides any
  default attribution instruction a session arrives with (2026-09-16: 55 commits rewritten).
- No emojis in code, commits or documents. Terse.
- Honest reporting: write down failures with their reason; "unmeasured", not "null", below the
  MDE; the measured floor beside every headline. README is the primary documentation.
- Reports for others: positive results first, failures briefly at the end, every term defined.
- **Do not append session narratives here.** Update `project_state.md` and the results file;
  edit the tables below only when something becomes citable or is withdrawn.

## Where things live

| what | where |
|---|---|
| live state (running, next) | `.claude-memory/project_state.md` -- read first |
| orientation for a fresh session (thesis, what survives, what is open, what is stale) | `docs/WHERE_THINGS_STAND.md` -- read second |
| catalogue of every results file | `RESULTS_INDEX.md` (catalogue regenerated 2026-10-03: every top-level `*.md` plus the docs-level notes, zero unclassified; hand rows cover everything through the leak batch, with post hoc / pilot / literature rows marked as such). Regenerate with `docs/tools/catalog_results_index.py` and paste its output into the catalogue section |
| prior art for the 2026-09-28..10-03 line; that week's handoff | `docs/lit/LIT_*.md` (check before claiming novelty); `docs/SESSION_2026-09-27_to_10-03.md` |
| EM/WM and position-kernel line | `EM_WM_STATE.md`, `AUDIT_2026-09-10.md` |
| guards | `GUARDS.md`; `python3 -m mapformer.test_guards` from `/home/prashr`; `python3 -m mapformer.experiment_audit --runs-dir D --control TWIN --control-of ARM` before reading any run dir |
| void / stale results | `archive/void/` (bannered), `archive_stale/`; code bugs `KNOWN_BUGS.md` |
| documents | `positional_review.pdf` (review), `axes_measured.pdf` (results paper), `mapformer_math.pdf` (record), `report/report.pdf`, `report/report_short.pdf` -- carry rank separation, Dyck matched depth and torus loop-rank (2026-09-27); 2026-09-30 corrected only where contradicted (sign now has its matched-length control; code full-val) in all but `mapformer_math` (nothing contradicted), rebuilt from source. **They do NOT carry rank 3, H1 part 1, H3, the text world, the context step, 3D rank / wrap, what/where or the leak** (`docs/WHERE_THINGS_STAND.md`, Known stale); corpus `papers/INDEX.md` (40 papers, read first-hand -- grep, don't re-search) |
| shared report | https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc, source `report/language_summary.html` (v10, 2026-09-30: lacks 3D rank / wrap, what/where, leak); republish WITH `url=` or the user's link breaks |
| run dirs of the Dyck / Bach / Indirect line | `docs/LOG.md`, block 2026-09-15..20 (`DYCK_T3_RESULTS.md` is an empty artefact) |
| model aliases | `train_variant.py::VARIANT_MAP`: MapWM-Flat=Vanilla, MapEM-Flat=VanillaEM, MapWM-Hier=Hourglass_k2, MapWM-FlatHG=HourglassFlat3; Plain-* use index RoPE |

## Citable results

**The dividing line is matched vs mismatched length.** Every "helps past the training length"
claim that got a matched-length control died -- except sign, which SURVIVED its control
(2026-09-28, `SIGN_MATCHED_RESULTS.md`). InEKF, forget gate, PoPE-wrapping and
rotate/allocentric have never had one: robustness, not capability, until they do. Depth counts too:
Dyck's +0.168 at 4 layers was depth extrapolation and CLOSED at matched depth (`DYCK_MDEPTH_RESULTS.md`);
what survives there is depth-substitution (1 layer of path integration ~ 3 layers of attention).

| result | numbers | file |
|---|---|---|
| Path integration helps on the paper's torus task, at training length | converged recipe: position **+0.243** (MDE 0.038, 8/8), index RoPE 0.805, path 0.971; grows to +0.359 at 8x. The often-quoted +0.461 is the 16-epoch recipe, where the index arm never left the 0.506 floor | `PAPER2X2_RESULTS.md` (not `BASELINE_TABLE.md`'s +0.461) |
| ...and is necessary for in-context maps (Match-Query) | 0.730 +/- 0.247 (n=5) vs index 0.154, chance 0.0625; context destruction 0.918 -> 0.074 | `MATCH_QUERY_SCALE.md`; the destruction pair is in `MATCH_QUERY_RESULTS.md` |
| Boundary: map extent, a threshold | -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells, matched aliasing | `ALIASING_CONTROLLED.md`, `VISITS_TEST.md` |
| Boundary: rotation actions; allocentric recoding fixes it | +0.050 -> +0.488 (8/8); 12 headings +0.26..+0.38 | `KNOB_SWEEP_n8.md` (the 8/8 pair), `H12_BUDGET_CURVE.md`; `ALLOCENTRIC_RECODING.md` is the n=3 mechanism run (+0.049 -> +0.485) |
| Dyck: path integration is worth ~3 layers of attention, at matched depth | trained AND tested at L32 D12 (`DYCK_MDEPTH_RESULTS.md`): position main +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers (8/8 each, A2f, floor 0.594). At 4 layers and 3x budget every arm is at ceiling (index 0.997-0.998, path 1.000; effect +0.002): **the ladder's +0.168 at 4L was depth extrapolation, trained at D4**. Mixture training (D in 4..12) keeps +0.110 at D12. Settled in the same batch: index base 32 vs 10000 within MDE at 4L; the 1x ladder budget limits the index arms (+0.021) | `DYCK_MDEPTH_RESULTS.md`, `DYCK_LADDER_RESULTS.md` |
| ...and the same exchange rate on a biased 1D walk (H3) | 1 path layer solves every cell (32/32); index attention needs 3 layers at p_plus 0.5 / 0.75 and 1 at p_plus 1.0; at 0.9 the 2-layer gap is 0.0115 vs the 0.01 threshold (knife-edge, 7/8 STALLED). Registered primary landed in NO branch (index 1L 0.717 / 0.676 / 0.825 / 1.000, non-monotone; holds on fresh seeds s2-s7). Seeds 0-1 were the pilot. Budget-scoped | `CANCEL_RESULTS.md` |
| Navigation told in words: PATH WINS IN WORDS | torus walk as English (58 words, synonyms, fillers), T=1024: path 1L **0.969** vs RoPE 1L 0.505 (floor 0.505, reversal-copy 0.597) and RoPE 2L 0.772; +0.464, p 0.0002, 7/8 vs 0/8 SOLVED. Fresh seeds only (s0-1 were the pilot): +0.474, 6/6 vs 0/6, p 0.0022. Step table: registered B did NOT fire (4/8); declared secondaries: opposites cancel on 8/8 after removing a common component; 4/8 seeds also run a real per-step clock (31-38 of 64 phase channels drift between visits). Scripted grammar, context-free steps | `TEXTWORLD_RESULTS.md` |
| What/where separation, causal (post hoc, eval-only, `docs/WHAT_WHERE_CHECKS.md`) | converged path models keep 0.974-0.989 with object identity removed from the score (cost 0.011-0.025, < MDE) and 0.988-1.000 with a shared kernel x content gain; RoPE/PoPE fall below floor; action/observation TYPE is needed. Separation also on a never-redrawn 32x32 map; a memorised 100-cell map has no 'where'. 'Separation is learned' itself is prior art (MapFormer Fig. 9) | `docs/WHAT_WHERE_CHECKS.md` |
| What-to-where leak and its remedy (`LEAK_RESULTS.md`, registered) | removing object tokens' steps (ActOnly) or normalising the step input (NormStep) gives +0.0107 unseen-object accuracy in distribution (p 0.0002, 8/8), exactly MapWM's leak; both converge (16/16 SOLVED) where MapWM is still descending (r(loss,acc) -0.94: partly training speed). x4 code-norm robustness is by construction | `LEAK_RESULTS.md` |
| Sign of the increment (replicates Sarrof / Grazzi / SRoPE in a new regime) | **at MATCHED length** (trained and tested T=1024, 900 ep): Abs - Signed **-0.177** (perm p 0.0002), solved 0/8 vs 8/8; Pos -0.218; opposition 0.06 signed vs 1.92-1.97 monotone; monotone arms stalled (budget-scoped). Trained at 128 the in-distribution cost was unmeasured (-0.054) and the OOD one -0.363 | `SIGN_MATCHED_RESULTS.md`, `SIGN_ABLATION.md` |
| Clock/map crossover | monotone costs -0.280 torus, -0.004 recency; magnitude-matched content increment +0.594 (8/8) | `RECENCY_RESULTS.md`, `RECENCY_GATE_ABLATION.md` |
| A shared block looped x4 helps path integration (Match-Query) | loop vs no loop **+0.346 unpaired** (t 3.75; loop pooled over two batches 0.803 +/- 0.200, 1/16 failures); matches 3 real layers at 1/3 the params. Runs do NOT reproduce across batches (per-seed drift 0.185), so the paired +0.315 interaction and "never fails" are single-batch and withdrawn; r=4 + loop x4 0.986 (8/8 >= 0.941) is one batch | `REFINE_RESULTS.md`, `LOOP_HEADROOM.md`, `MQ_RANK_2X2.md` |
| EM's recency deficit is search | EM - WM -0.375 (0/8); installed rewind frozen 1.000; per-pair origins +0.215 (n=48) | `EM_WM_STATE.md`, `SEARCH_RESULTS.md`, `PAIRSPLIT_RESULTS.md` |
| Paper replications | MapFormer v4 Dyck-2 ordering +0.370 on F1 (Hewitt closing accuracy +0.064; levels do not replicate); PoPE paper: Indirect at 200k iters 7/8 (1/8 at their 100k), Bach -0.032 NLL (5/5) | `DYCK_RESULTS_bs128.md`, `INDIRECT_RESULTS_200k.md`, `JSB_RESULTS.md` |
| Our PoPE is faithful; non-negativity is not what extrapolates | 1.7e-06 vs authors' code; NoSigma penalty -0.011 vs RoPE +3.59; 80.7% of `pope_delta` frozen in its clamp | `ABLATE_RESULTS.md` |
| PoPE's encoding helps the path row (Bach); "path integration hurts PoPE on clock tasks" is UNMEASURED | MapPoPE - MapWM: Bach -0.0165 (5/5); Dyck 2L +0.050 is depth-OOD (trained D4; at matched depth both 1.000); code -0.0054 full-val (t p 0.086). MapPoPE - PoPE: code +0.0034 full-val (t p 0.19), Bach +0.0111 (inside MDE): UNMEASURED. Code at matched length 2048, full val: position main +0.0056 (3/3, t p 0.024, n=3) -- path integration costs | `.claude-memory/project_mappope_asymmetry.md`, `CODE_FULLVAL_RESULTS.md` |
| Decay envelope vs long-range retrieval | steepness +0.136 and distance +0.145 of +0.281 (8/8); confounded with convergence | `CROSS_RESULTS.md` |
| Recipe beats architecture on compositional | warmup+cosine +0.160 (7/8); hierarchy +0.136 directional, unpowered | `COMP_HEADROOM.md`, `HIER_RECHECK.md` |
| Parallel scan | 2.6-3.3x over 16x length; MapEM-NC 14.5x, TEMFaithful 120x | `TIMING_BENCHMARK.md` |
| CSCG stitching control reproduces | +0.131 +/- 0.024 vs index -0.005 | `STITCH_ATTENTION.md` |
| Metric/data findings | Dyck F1 has a 0.88 no-stack floor (use Hewitt closing accuracy); Bach overfitting-limited 3x (transposition 0.107 NLL) | `DYCK_LITERATURE_METRICS.md`, `AUG_RESULTS.md` |
| Hierarchy on text is efficiency only | 1.4537 vs 1.4506 bpc at param parity; 1.23x throughput, -14% memory | `ENWIK8_HIERARCHY.md` |

**Rank: it is the PER-HEAD rank of the content-to-angle map** (torus, T=1024, budget-scoped;
`RANK_SEP_RESULTS.md`, `RANK_MI_RESULTS.md`, `RANK_PROJ_RESULTS.md`). Five arms, all built from our
r=2's initial weights at each seed, differing only in the bottleneck; SOLVED within 900 epochs:

| per-head rank | arms | solved |
|---|---|---|
| 2 | our shared r=2, per-head r=2, block-diagonal r=4 | 0/8, 2/8, 2/8 |
| 3 | per-head r=3 (`RANK3_RESULTS.md`; acc 0.987 vs r=2 0.885, perm p 0.027, Holm 0.054; vs r=4 UNMEASURED) | 6/8 |
| 3D torus (`RANK_ND_RESULTS.md`, `RANK_WRAP_RESULTS.md`) | rank 3 (= D) 1/8; rank 4 (= D+1) 4/8 on a small wrap-heavy grid (10/side) but **8/8 on grid 18** (few wrap-only revisits): the shortfall was the small-grid regime, not dimension. 2D control rank 2 1/8 vs rank 3 8/8 (p 0.0014). A 100-cell 2D grid is memorised (own map 0.986, unseen 0.273) | -- |
| 4 | shared r=4, per-head r=4 | 8/8, 8/8 |

Separated: per-head rank FIRES (D - C_bd, both block-diagonal, Fisher and permutation p 0.0070);
sharing one latent vs per-head latents UNMEASURED (D - C); `W_out` per-entry scale UNMEASURED
(C_bd - B). Initial angle scale does not explain it (A and B fail at normal scale). A rank-2 solution
EXISTS (0.9955) and is HELD under training, so this is SEARCH, not capacity. **Not rescued by 2x the budget**: at 1800 epochs (from scratch) rank 2 still solves 0/8 vs
rank 4 7/8 (Fisher p 0.0014; `LOOP_RANK_E1800_P1_RESULTS.md`; 5/8 rank-2 runs still descending). r=4's old +0.085 was
out-of-distribution only (`RANK_SWEEP.md`). The paper states a per-head `W_in` but not `W_out`'s
shape: at 2 heads its r=2 read literally is the per-head r=2 (2/8); with a full `W_out` it is our
r=4 (8/8). Scope: n_heads=2, one length, one recipe. MapPoPE r=4 +0.019 is an OOD, unmatched-init
number.

**Search aids partly recover rank 2, but do not reach rank 4** (`LOOP_RANK_RESULTS.md`, H1,
registered verdict UNMEASURED). At T=1024/900 ep: r=2 + loop x4 (bit-identical params and init to
r=2) solves 2/8 with accuracy 0.973; r=2 at 4 real layers 5/8 and 0.990; plain r=2 0/8 and 0.894;
r=4 8/8 and 0.998. Both aids fire on accuracy (perm p 0.0034 / 0.0012); only depth fires on solved
count (Fisher 0.026). Final losses form three regimes (r=2 0.05-0.68, the aids 0.011-0.13, r=4
0.002-0.010): extra search leaves r=2's regime but never enters r=4's. The registered 0.05 cutoff
falls inside the aids' spread -- at 0.08+ H1's condition would have been met, so the verdict is
uncertain rather than negative. Depth (4x params) beats the matched-parameter loop, so this is NOT
"search at constant capacity".

**Live negatives -- do not re-run:** Level 1.5 / InEKF is stabilisation, not inference (no
component load-bearing, capacity control ties, benefit does not grow with drift:
`L15_ABLATION.md`, `EXTRAHEAD_CONTROL.md`, `MQ_NOISE_2X2*.md`); refining theta across depth
(`NOISE_REFINE.md`); filter x loop (`L15_LOOP_2X2.md`); MoR routing (+0.007 oracle,
`LOOP_DEPTH_STRATA.md`); explicit content gate separates 4.16x and buys nothing
(`GATED_RESULTS.md`); forget gate +0.086, mechanism unidentified (`FORGET_CONTROL.md`);
non-commutativity +0.005-0.014 for 34x cost (`FAMILY_TREE_RESULTS.md`); Selective RoPE's
generator no better (`SELECTIVE_ROPE.md`); hex emergence; PC + Kalman are duals; oracle motif
pooling and frame reset (`COMPOSITIONAL_EXPERIMENT.md`); Flip-Flop and MQAR do not test our axis;
Habitat porting (`HABITAT_BUILD.md`); bf16 autocast (`BF16_RESULTS.md`, keep fp32).

## Withdrawn -- do not cite

- The lm200 era (Apr-May 2026 landmark tables: TEM leader, NoDrop +13pp, GSF, Cascade): baselines never converged. `CORRECTED_LM200_LEADERBOARD.md`
- hier-goal, the "MapFormer x hierarchy synergy", all four planner tasks and the +7.5pp frozen probe: solvable from the action stream. `HIERGOAL_ABLATION.md`, `PLANNER_TASK_AUDIT.md`
- Code OOD (encoding -3.694, MapPoPE - PoPE -0.102): the cost of extrapolating past 512; at matched length the sign reverses. `CODE_RESULTS.md`, `CODE_DECAY_RESULTS.md`
- "Position effect scales with aliasing" (sign inverted at fixed grid) and "distinct cells visited". `ALIASING_CONTROLLED.md`, `VISITS_TEST.md`
- The named Level 1.5 decomposition ("gate load-bearing", "not a clamp", "wins architecturally"). `L15_ABLATION.md`
- WM-vs-EM regime table; "MapWM is additive / an OR-gate"; Thm 3. `AUDIT_2026-09-10.md`
- "EM never finds the rewind (0/40)"; the early-window mechanism. `SEARCH_RESULTS.md`, `UNFREEZE_RESULTS.md`
- The accumulator + kernel account of the MapPoPE collapse; the PoPE-decoupling corollary. `T2_RESULTS.md`
- "The InEKF wrap bounds the accumulator"; the critical-dimension import. `ACCUMULATOR.md`, `LOCALISATION.md`
- Packing geometry as the account of v4 Table 6 (r=2 is best at D=5). `DXR_RANK_THRESHOLD.md`
- "The r=4 gap is training speed"; "r=2 loses because its basis is skewed" (within r=2 skew does not predict accuracy; a rank-2 solution exists). `RANK_MATCHED_RESULTS.md`, `RANK_PROJ_FROZEN.md`
- Navigation "+0.461, index on the floor" as the torus headline (16-epoch recipe; converged +0.243). `PAPER2X2_RESULTS.md`
- Dyck: the 4-layer +0.168 as a capability result, and "depth closes 40% of the gap then stops" (index arms keep climbing and hit ceiling at matched depth). `DYCK_MDEPTH_RESULTS.md`
- "Per-head rank, cross-head sharing and W_out scale are unseparated" (separated 2026-09-25: it is per-head rank). `RANK_SEP_RESULTS.md`
- "Nobody varies sign or rank deliberately" (Grazzi varies sign; MapFormer ablates r) and "Mamba-3 is the first to claim both slots" (Selective RoPE, earlier). `papers/txt/`
- "PoPE's Table 5 does not replicate" (stale checkpoint + 5% val; NoSigma cell unmeasured). `ABLATE_RESULTS.md`
- Dyck "depth substitutes, 4.4x" (width confound); Dyck F1 as a headline (inflates position 2.3x). `DYCK_LADDER_RESULTS.md`, `DYCK_LITERATURE_METRICS.md`
- "Scale hurts path integration at long range"; the horizon table is budget-limited. `LOOPED_PILOT.md`, `HORIZON_RESULTS.md`
- "The loop beats three real layers" (it matches them). `LOOP_HEADROOM.md`
- enwik8 hourglass "2.00 vs 2.07, better" (it is 1.4844 vs 1.4727, worse). `ENWIK8_HIER.md`
- The monotone H12 budget curve (bimodal). `H12_BUDGET_CURVE.md`
- The lap-counting mechanism (catastrophic forgetting). `LAP_TRANSFER_NOREWARD.md`
- Family tree "plain WM beats every published variant" (unmeasured at n=3). `N3_AUDIT.md`
- "Separate q0/k0 is refuted" -- true on the four map tasks only; on recency it is better. `D5_RESULTS.md`
- DoG hex result before the kernel fix (targets were all zero). `archive_stale/DOG_RESULTS_FIXED.md`
- The `Level15PC_v4` (predictive-coding) +3.4pp win and its RNG-drift explanation (control byte-identical). `archive_stale/V4_CONTROL_RESULTS.md`
- "No survey covers content-dependent phase" (Zhang et al. 2503.17407 does). `.claude-memory/reference_review_documents.md`
- Settled, not open: the Dyck `base=10000` frequency-ladder confound (base 32/128 move index arms <= 3%). `DYCK_DEPTH_RESULTS.md`

## Standing rules (each bought by a failure; details in `docs/LOG.md` and the memory files)

**Measurement**
1. Measure the floor, never assume it: a provably function-identical twin (MiniWorld |delta| 0.150) and the task's best trivial predictor (per cell, the better of n-gram and constant -- they cross). Report it beside every headline.
2. Is accuracy just loss? Compute r(final loss, acc) per eval length (-0.996 over 57 runs; -0.33 at T=1024 in loop arms). Loss-match only when losses overlap -- check before the batch. At matched length loss-matching is uninformative. Never infer held-out accuracy from loss (0.03 -> 0.674).
3. Convergence and schedule before comparison: final-10% slope; LinearLR from step one cannot escape a plateau (0.448 -> 0.990; compositional +0.160). Validate a convergence criterion on runs of known status; separate SOLVED from STALLED. A flat final window read after LR decay is NOT convergence (it passed stuck runs and runs still descending; `experiment_audit.py`'s slope rule is of this kind -- it now also prints the registered windowed classes and exits 2 on layouts it cannot read).
4. Budget cuts both ways: one weak point is not a negative (three false negatives in a day), two points are not a trend (H12 +0.383 -> +0.286). Prefer extending the budget to convergence-conditioning arguments.
5. Power: below the MDE say "unmeasured". **The house rule |d| > 2.8 sd/sqrt(n) is |t| > 2.8: a 10.7% false-positive test at n=3 (6.8% n=4, 2.7% n=8); the true 80%-power multiplier is 5.36 at n=3, 3.26 at n=8, and below n=6 no distribution-free test reaches p < .05.** Report the t-test p (and a permutation / Fisher test when seeds are bimodal) beside any n<8 "DETECTABLE" (e.g. code C1 MapPoPE-PoPE +0.0033 had t-test p ~0.1 on `best_val_bpc`; on the full val file it is +0.0034, p 0.19): `stats_core.py` has them all (exact-t MDE, paired/sign-flip p, vectorised permutation, Fisher, run classes); `stats_guard.Contrast.row_full()` prints them beside the unchanged house verdict. Import, do not re-derive. Check the verdict cell could have gone the other way (a baseline at 1.000 +/- 0.000). Conditioning on convergence can select into a ceiling.
6. Seeds: n <= 3 is not a point estimate (n=1 and n=3 readings have flipped sign at larger n). Put the seeds on the comparison you claim. A same-seed rerun is determinism, not replication (+0.237 bitwise -> +0.128 at n=24); report fresh seeds alone.
7. Pre-register, then report the registered primary readout even when the verdict looks obvious; set branch boundaries against the noise floor; commit the script behind every number.
8. Readouts must be invariant to the model's symmetries: check sign/scale gauges before registering a contrast (rho = +/-1 had expectation 0); theta wraps mod 2pi/omega (a linear rewind slope read 0/40 on found solutions).
9. Verify what a probe measures (a weight-norm vector reported "100% in top-2 SVs"). Read the code, not its comment. Post-hoc truncation is not trained-at-rank.
10. Robustness is not capability: an effect seen only past the training length needs a matched-length control.

**Design**
11. Gate the task before any GPU: action-stream-only n-gram (orders 1-5), measured chance, token ids in vocabulary (CUBLAS_STATUS_ALLOC_FAILED reads like OOM); context-destruction ablation on trained models. The gate must CALL the task code.
12. Retrain every arm in one batch; never compare to a stored checkpoint.
13. Audit the design before launch (Dyck width confound, OOM picker, rank loss non-overlap). Check the mechanism's premise applies; check whether the knob is a runtime argument (eval-only sweep: 90 s vs 12 runs); split compound hypotheses.
14. Existence before mechanism: construct a solution in the class, then warm-start it frozen AND trainable, at the weight scale training would use (frozen 1.000, trainable 0.642, 8x scale 0.941).
15. A parameterisation change is an optimiser change (a scale at 1.0 under Adam moves ~50x slower relatively).
16. Check whether a "failure to reproduce" is the paper's own reported result (Fig. 4 C3). A borrowed benchmark usually does not test your axis.
17. Known confounds: Hourglass variants ignore `--n-layers` (use n_layers=3 for flat vs hier); `MapFormerWM_PoPE` defaults to r=2; Selective-RoPE single-knob arms also delete omega; a control that trains better is not attributable.
18. Apply a retraction to the generators: grep the code for retracted numbers, not only the docs.
19. Verify a fast path row-exact / loss-exact before a batch uses it. Verified bit/byte-identical on the default path (2026-09-24, `docs/audits/2026-09-24/applied/`): the vectorised walk (`environment.FAST_WALK`), the power-of-two scale fold, the sync-free train loop, the vectorised strata/permutation code. `--fast-attn` agrees to ~1e-6 in logits but is NOT bit-identical in training (dropout RNG) and its backward is run-to-run deterministic only with `--deterministic` (strict, not warn_only) -- new series only (`FAST_ATTN_RANK.md`); TF32 broke it on the compositional trainer (6.0e-01); invalid for MapEM. bf16 moved a gap 0.0117; `--data-workers` changes the data stream (never beside a reproduction control).

**Operations**
20. Anything over ~2 minutes runs under `setsid`/`nohup` (foreground calls are SIGTERM'd at 2 min with their children; `run_in_background` jobs die with the session). Judge completion by the `.done` marker AND the artifact: `wait` returns even when every child died.
21. Drivers take a `flock -n` single-instance lock (`lib_driver.sh` has it, the comm-matched counter, the least-loaded picker at 2 jobs/GPU, the md5 guard and the artifacts-then-marker rule; new drivers source it). Killing a supervisor does not kill its children (16 launches for 12 runs). Guards against duplicate launch need both sides. A duplicate can overwrite `best.pt` with a stale one that loads cleanly.
22. Never edit a running bash script in place (bash reads by byte offset); write a new file and `mv` it over (the running copy keeps its inode). Do not edit a module a batch is spawning from; re-import any module after editing it.
23. Never `pkill -f` / `pgrep -f`: they match your own and the author's shells (a waiter sat 2 h on zero jobs). Use `ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/'`.
24. Place jobs by actual GPU occupancy: fill-first idled a 4090 for 3 h, blind `cuda:$((i%2))` OOM'd two 2048-context runs. Cap OMP threads; prebuild MiniWorld buffers (`prebuild_buffers.py`: one EGL context per worker).
25. `python3 -m mapformer.X` runs from `/home/prashr`: relative paths resolve to the parent and fail silently. Use an absolute `REPO` constant.
26. Verify before and after anything destructive. A failed `git add` stages nothing (check `git show HEAD:<file>`). `safe_clear.sh` refuses a run dir with any completion marker (a finished 48/48 batch was rm'd); fixed 2026-09-24 to fail CLOSED (it had failed open from /home/prashr) -- still verify, and never answer y on a real run dir you have not checked.
27. Small traps: `local a=$1 b=$2 c="$b"` under `set -u` expands before assigning; label aggregated rows by arm AND recipe, not checkpoint filename; case-sensitive verification greps (`rope` is in `property`); no mid-line `%` comments in .tex, build from source; `str.splitlines()` breaks on form feed.
28. A document grown by accretion needs an end-to-end read before it is shared.
29. Pilots run on seeds OUTSIDE the registered batch (textworld/cancel pilots were byte-identical to batch seeds
    0-1). Every batch gets an independent code-verification agent, blind to results; its findings go into an
    amendment before any result is read (it found missing branches, 1%-gap firings, unfailable tests, probe errors).

## Invariants (paper-faithful; do not break)

- `environment.py`: torus `(x+dx) % N`; interleaved `[a1, o1, a2, o2, ...]`; `revisit_mask` per trajectory. `train.py`: loss on revisited observation positions only.
- `PathIntegrator`: theta by cumsum of angles, not a prefix product of rotations; omega monotone decreasing, `omega_i = omega_max * (1/Delta_max)^(i/(n_b-1))` (the paper's eq. 17, eq. 18 in v4, has a sign typo).
- `ActionToLieAlgebra`: low-rank `W_out W_in`, r=2 default. **Deviation (found 2026-09-24): ours shares one r-dim latent across heads; the paper's is per head (nh x r latent dims).** Say so wherever r is compared with the paper.
- MapWM rotates content Q,K by the path angle, score `Q^T R(dtheta) K`: NOT additive. MapEM: `softmax(A_X (*) A_P)` with separate `q0_pos`, `k0_pos` (paper App. A.7); single-p0 is an ablation, both reportable.
- Defaults: 1 layer, 2 heads, h=64, d=128, AdamW lr 3e-4 wd 0.05, linear LR decay, batch 128, grid 64, T=128, K=16 obs, p_empty 0.5, 200K sequences. New work uses `--schedule cosine`; say which.
- RoPE baseline uses canonical `base^(-2c/d_head)` since 2026-09-04; `inv_freq` is a buffer, so older checkpoints keep theirs. Level15EM uses `log_R_init_bias=3.0`.
- Reproduction target is paper **v4** (10 May 2026), Table 2 2D: MapWM-r2 1.00/1.00/0.99, MapEM-os 1.00/1.00/1.00. Ours, 0.989 (WM) / 0.987 (EM), matched v1 (0.99/0.99/0.96, 1.0/0.99/0.97) and sits marginally below v4; name the version. v4 Table 6: MapWM collapses in 5D (0.75/0.50/0.35). Reproduce: `python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98`.
- Environments: torch 2.6.0+cu124 (pip picks cu130 and silently falls back to CPU) plus full `requirements.txt` (`HOURGLASS_README.md`); Habitat lives in a separate py3.9 conda env.
