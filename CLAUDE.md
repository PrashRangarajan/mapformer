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
| catalogue of every results file | `RESULTS_INDEX.md` (last regenerated 2026-09-11: missing Dyck/Bach/code/rank-matched; the tables below are current) |
| EM/WM and position-kernel line | `EM_WM_STATE.md`, `AUDIT_2026-09-10.md` |
| guards | `GUARDS.md`; `python3 -m mapformer.test_guards` from `/home/prashr`; `python3 -m mapformer.experiment_audit --runs-dir D --control TWIN --control-of ARM` before reading any run dir |
| void / stale results | `archive/void/` (bannered), `archive_stale/`; code bugs `KNOWN_BUGS.md` |
| documents | `positional_review.pdf` (review), `axes_measured.pdf` (results paper), `mapformer_math.pdf` (record), `report/report.pdf`, `report/report_short.pdf`; corpus `papers/INDEX.md` (40 papers, read first-hand -- grep, don't re-search) |
| shared report | https://claude.ai/artifact/LVfYeHhjs1KjwMpg3Pxggc, source `report/language_summary.html`; republish WITH `url=` or the user's link breaks |
| run dirs of the Dyck / Bach / Indirect line | `docs/LOG.md`, block 2026-09-15..20 (`DYCK_T3_RESULTS.md` is an empty artefact) |
| model aliases | `train_variant.py::VARIANT_MAP`: MapWM-Flat=Vanilla, MapEM-Flat=VanillaEM, MapWM-Hier=Hourglass_k2, MapWM-FlatHG=HourglassFlat3; Plain-* use index RoPE |

## Citable results

**The dividing line is matched vs mismatched length.** Every "helps past the training length"
claim that got a matched-length control died. InEKF, forget gate, PoPE-wrapping, sign and
rotate/allocentric have never had one: robustness, not capability, until they do. Depth counts
too: Dyck's surviving effect is at 3x the training nesting depth.

| result | numbers | file |
|---|---|---|
| Path integration helps on the paper's torus task, at training length | converged recipe: position **+0.243** (MDE 0.038, 8/8), index RoPE 0.805, path 0.971; grows to +0.359 at 8x. The often-quoted +0.461 is the 16-epoch recipe, where the index arm never left the 0.506 floor | `PAPER2X2_RESULTS.md` (not `BASELINE_TABLE.md`'s +0.461) |
| ...and is necessary for in-context maps (Match-Query) | 0.730 +/- 0.247 (n=5) vs index 0.154, chance 0.0625; context destruction 0.918 -> 0.074 | `MATCH_QUERY_SCALE.md` |
| Boundary: map extent, a threshold | -0.010 / +0.015 / +0.305 at 32 / 128 / 512 occupied cells, matched aliasing | `ALIASING_CONTROLLED.md`, `VISITS_TEST.md` |
| Boundary: rotation actions; allocentric recoding fixes it | +0.050 -> +0.488 (8/8); 12 headings +0.26..+0.38 | `ALLOCENTRIC_RECODING.md`, `H12_BUDGET_CURVE.md` |
| Dyck depth ladder (width fixed). **Training is L32 D4; D12 is 3x the training depth** | at the TRAINING cell L32 D4: +0.293 / +0.081 / +0.048 / +0.019 at 1-4 layers (8/8; index 0.979 at 4L). At L32 D12 (depth-OOD): +0.290 / +0.209 / +0.159 / +0.168, index plateaus 0.76-0.78. Matched length, NOT matched depth | `DYCK_LADDER_RESULTS.md` |
| Sign of the increment (replicates Sarrof / Grazzi / SRoPE in a new regime) | at T=512/1024 (trained 128): signed beats index +0.123 / +0.195 (12/12), monotone does not; opposition 0.11 vs 1.85-1.98. **At training length monotone scores 0.90-0.98 vs index 0.80: the accuracy cost is extrapolation-only; no matched-length control** | `SIGN_ABLATION.md` |
| Clock/map crossover | monotone costs -0.280 torus, -0.004 recency; magnitude-matched content increment +0.594 (8/8) | `RECENCY_RESULTS.md`, `RECENCY_GATE_ABLATION.md` |
| A shared block looped x4 helps path integration (Match-Query) | loop vs no loop **+0.346 unpaired** (t 3.75; loop pooled over two batches 0.803 +/- 0.200, 1/16 failures); matches 3 real layers at 1/3 the params. Runs do NOT reproduce across batches (per-seed drift 0.185), so the paired +0.315 interaction and "never fails" are single-batch and withdrawn; r=4 + loop x4 0.986 (8/8 >= 0.941) is one batch | `REFINE_RESULTS.md`, `LOOP_HEADROOM.md`, `MQ_RANK_2X2.md` |
| EM's recency deficit is search | EM - WM -0.375 (0/8); installed rewind frozen 1.000; per-pair origins +0.215 (n=48) | `EM_WM_STATE.md`, `SEARCH_RESULTS.md`, `PAIRSPLIT_RESULTS.md` |
| Paper replications | MapFormer v4 Dyck-2 ordering +0.370 on F1 (Hewitt closing accuracy +0.064; levels do not replicate); PoPE paper: Indirect at 200k iters 7/8 (1/8 at their 100k), Bach -0.032 NLL (5/5) | `DYCK_RESULTS_bs128.md`, `INDIRECT_RESULTS_200k.md`, `JSB_RESULTS.md` |
| Our PoPE is faithful; non-negativity is not what extrapolates | 1.7e-06 vs authors' code; NoSigma penalty -0.011 vs RoPE +3.59; 80.7% of `pope_delta` frozen in its clamp | `ABLATE_RESULTS.md` |
| PoPE's encoding helps the path row; path integration hurts PoPE on clock tasks | MapPoPE - MapWM: Bach -0.0165, Dyck 2L +0.050, code -0.0052; MapPoPE - PoPE code +0.0033 (0/3) | `.claude-memory/project_mappope_asymmetry.md` |
| Decay envelope vs long-range retrieval | steepness +0.136 and distance +0.145 of +0.281 (8/8); confounded with convergence | `CROSS_RESULTS.md` |
| Recipe beats architecture on compositional | warmup+cosine +0.160 (7/8); hierarchy +0.136 directional, unpowered | `COMP_HEADROOM.md`, `HIER_RECHECK.md` |
| Parallel scan | 2.6-3.3x over 16x length; MapEM-NC 14.5x, TEMFaithful 120x | `TIMING_BENCHMARK.md` |
| CSCG stitching control reproduces | +0.131 +/- 0.024 vs index -0.005 | `STITCH_ATTENTION.md` |
| Metric/data findings | Dyck F1 has a 0.88 no-stack floor (use Hewitt closing accuracy); Bach overfitting-limited 3x (transposition 0.107 NLL) | `DYCK_LITERATURE_METRICS.md`, `AUG_RESULTS.md` |
| Hierarchy on text is efficiency only | 1.4537 vs 1.4506 bpc at param parity; 1.23x throughput, -14% memory | `ENWIK8_HIERARCHY.md` |

**Open, not citable:** rank. r=4 +0.085 at T=1024 is out-of-distribution only (trained T=128,
`RANK_SWEEP.md`); MapWM-family only (MapPoPE r=4 +0.019, `MAPPOPE_R4_RESULTS.md`). At matched
length (T=1024) r=4 solves 8/8 vs r=2 0/8 at 900 ep and 7/8 vs 1/8 after a 900-ep warm-restart
continuation; both registered verdicts UNREADABLE (`RANK_MATCHED_RESULTS.md`). A rank-2
projection of each solved r=4 model scores 0.9955 mean (min 0.981) at T=1024, and trained from
there r=2 holds it 7/8 like the r=4 control (S1 STABLE, `RANK_PROJ_RESULTS.md`): exists, stable,
not found -- a SEARCH deficit. From-scratch r=2 stalls in non-cancelling codes (the skew is
the symptom); whether it could leave them is untested. **Our bottleneck is shared across heads; the paper's is
per head** (`W_in in R^{d x nh x r}`, mapformer.txt ~l.1512), so our r=2 has half the paper's
latent dims at 2 heads (ours r=2 < paper r=2 < ours r=4) and "use r=4" may only restore the
paper's latent-dimension count. `w_out` init scales 1/sqrt(r) -- unseparated from dimension. Follow-ups in
`project_state.md`.

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
- Dyck "in distribution / training length" (it is 3x the training depth). `DYCK_LADDER_RESULTS.md`
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
3. Convergence and schedule before comparison: final-10% slope; LinearLR from step one cannot escape a plateau (0.448 -> 0.990; compositional +0.160). Validate a convergence criterion on runs of known status; separate SOLVED from STALLED. A flat final window read after LR decay is NOT convergence (it passed stuck runs and runs still descending; `experiment_audit.py`'s rule is of this kind).
4. Budget cuts both ways: one weak point is not a negative (three false negatives in a day), two points are not a trend (H12 +0.383 -> +0.286). Prefer extending the budget to convergence-conditioning arguments.
5. Power: below the MDE say "unmeasured". **The house rule |d| > 2.8 sd/sqrt(n) is |t| > 2.8: a 10.7% false-positive test at n=3 (6.8% n=4, 2.7% n=8); the true 80%-power multiplier is 5.36 at n=3, 3.26 at n=8, and below n=6 no distribution-free test reaches p < .05.** Report the t-test p (and a permutation / Fisher test when seeds are bimodal) beside any n<8 "DETECTABLE" (e.g. code C1 MapPoPE-PoPE +0.0033 has t-test p ~0.1). Check the verdict cell could have gone the other way (a baseline at 1.000 +/- 0.000). Conditioning on convergence can select into a ceiling.
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
19. Verify a fast path row-exact / loss-exact before a batch uses it. `--fast-attn` is exact on the torus (1.4e-06) but TF32 breaks it on the compositional trainer (6.0e-01) and it is invalid for MapEM; bf16 moved a gap 0.0117; `--data-workers` changes the data stream (never beside a reproduction control).

**Operations**
20. Anything over ~2 minutes runs under `setsid`/`nohup` (foreground calls are SIGTERM'd at 2 min with their children; `run_in_background` jobs die with the session). Judge completion by the `.done` marker AND the artifact: `wait` returns even when every child died.
21. Drivers take a `flock -n` single-instance lock. Killing a supervisor does not kill its children (16 launches for 12 runs). Guards against duplicate launch need both sides. A duplicate can overwrite `best.pt` with a stale one that loads cleanly.
22. Never edit a running bash script in place (bash reads by byte offset); write a new file and `mv` it over (the running copy keeps its inode). Do not edit a module a batch is spawning from; re-import any module after editing it.
23. Never `pkill -f` / `pgrep -f`: they match your own and the author's shells (a waiter sat 2 h on zero jobs). Use `ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/'`.
24. Place jobs by actual GPU occupancy: fill-first idled a 4090 for 3 h, blind `cuda:$((i%2))` OOM'd two 2048-context runs. Cap OMP threads; prebuild MiniWorld buffers (`prebuild_buffers.py`: one EGL context per worker).
25. `python3 -m mapformer.X` runs from `/home/prashr`: relative paths resolve to the parent and fail silently. Use an absolute `REPO` constant.
26. Verify before and after anything destructive. A failed `git add` stages nothing (check `git show HEAD:<file>`). `safe_clear.sh` is meant to refuse a run dir with a completion marker (a finished 48/48 batch was rm'd) but FAILS OPEN from /home/prashr (relative marker path) until patched -- do not rely on it.
27. Small traps: `local a=$1 b=$2 c="$b"` under `set -u` expands before assigning; label aggregated rows by arm AND recipe, not checkpoint filename; case-sensitive verification greps (`rope` is in `property`); no mid-line `%` comments in .tex, build from source; `str.splitlines()` breaks on form feed.
28. A document grown by accretion needs an end-to-end read before it is shared.

## Invariants (paper-faithful; do not break)

- `environment.py`: torus `(x+dx) % N`; interleaved `[a1, o1, a2, o2, ...]`; `revisit_mask` per trajectory. `train.py`: loss on revisited observation positions only.
- `PathIntegrator`: theta by cumsum of angles, not a prefix product of rotations; omega monotone decreasing, `omega_i = omega_max * (1/Delta_max)^(i/(n_b-1))` (the paper's eq. 17, eq. 18 in v4, has a sign typo).
- `ActionToLieAlgebra`: low-rank `W_out W_in`, r=2 default. **Deviation (found 2026-09-24): ours shares one r-dim latent across heads; the paper's is per head (nh x r latent dims).** Say so wherever r is compared with the paper.
- MapWM rotates content Q,K by the path angle, score `Q^T R(dtheta) K`: NOT additive. MapEM: `softmax(A_X (*) A_P)` with separate `q0_pos`, `k0_pos` (paper App. A.7); single-p0 is an ablation, both reportable.
- Defaults: 1 layer, 2 heads, h=64, d=128, AdamW lr 3e-4 wd 0.05, linear LR decay, batch 128, grid 64, T=128, K=16 obs, p_empty 0.5, 200K sequences. New work uses `--schedule cosine`; say which.
- RoPE baseline uses canonical `base^(-2c/d_head)` since 2026-09-04; `inv_freq` is a buffer, so older checkpoints keep theirs. Level15EM uses `log_R_init_bias=3.0`.
- Reproduction target is paper **v4** (10 May 2026), Table 2 2D: MapWM-r2 1.00/1.00/0.99, MapEM-os 1.00/1.00/1.00. Ours, 0.989 (WM) / 0.987 (EM), matched v1 (0.99/0.99/0.96, 1.0/0.99/0.97) and sits marginally below v4; name the version. v4 Table 6: MapWM collapses in 5D (0.75/0.50/0.35). Reproduce: `python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98`.
- Environments: torch 2.6.0+cu124 (pip picks cu130 and silently falls back to CPU) plus full `requirements.txt` (`HOURGLASS_README.md`); Habitat lives in a separate py3.9 conda env.
