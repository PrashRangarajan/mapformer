# VERIFY: audit of `report/report.tex` (numbers and referee pass)

Audited 2026-09-13 against the primary results files at the repo root. The inventories were used only
to find the files. `report.tex` was not edited. Line numbers are `report.tex` lines.

**Summary**
- About **1,700 items checked**: numbers, table cells, seed counts, lengths, raw/loss-matched labels
  and verdict words. Every headline number was checked, plus all appendix tables.
- **26 must-fix errors** (Section 1). About 10 are wrong values or labels. The rest are verdict words
  that contradict their own MDE, provenance or batch errors, or retracted content still in use.
- **31 numeric imprecisions** (Section 1b).
- **27 overclaims or language problems** (Section 2).
- No number comes from `archive/void/`, and no April lm200 row is used.
- One retracted claim is still present: "every looped seed >= 0.77", retracted in `REFINE_RESULTS.md`.
- One excluded task may be present: the CSCG stitch task (flagged as uncertain in Section 2).

---

## 1. Errors (must fix)

| # | Location | Report says | Source says | Fix |
|---|---|---|---|---|
| E1 | 5.1, l.771: "Rotation accounts for $-0.388$ of the $-0.438$ swing between the baseline and the fully combined condition" | −0.438 is the baseline-to-combined swing | `BASELINE_TABLE.md` sec. I: "all five combined \| −0.076 \| −0.514". The −0.438 is the baseline effect falling to zero ("90% of the available swing"). The report's own Table `tab:knob` gives +0.438 → −0.076 = −0.514. | "Rotation alone removes 0.388 of the baseline's +0.438 (89%); all five knobs together move it by −0.514." |
| E2 | 8.1, l.1590–1591: "Its contribution is mostly to the floor: every looped seed scores at least $0.77$" | loop arm never fails | `REFINE_RESULTS.md`: "**RETRACTED: 'the loop arm never fails — 8/8 ≥ 0.77, sd 0.099.'** That was one lucky batch. Pooled over 16 draws the loop arm is **0.803 ± 0.200 with 1/16 catastrophic failures**." Also listed in inventory B `## Excluded`. `STORY.md` Claim 5 still carries the retracted wording. | Delete the floor sentence, or replace it with the pooled 0.803 ± 0.200, 1/16 failures. |
| E3 | 6.2, l.642–644: "under a warmup-plus-cosine recipe in one batch of eight seeds … $+0.348$ (\mde\ $0.215$, $8/8$)". Same issue at l.558, l.1551–1552, and Table `tab:loop` (caption "one batch"; l.1582–1583 paired MDEs and seed counts). | one batch, paired MDEs | `runs/loop_headroom/*.log` timestamps show three launches. Seeds 0–2 are from `run_loop_headroom.sh` (08-30 20:03–20:38). PI seeds 3–7 are from `run_loop_topup.sh` (08-30 20:51–21:15). IX and PI_L3 seeds 3–7 are from `run_loop_topup2.sh` (08-31 02:29–03:02). So the index and path-integrated top-ups are different batches. Later, `REFINE_RESULTS.md` retrained `Looped` with the same seeds and got mean per-seed drift 0.185 (seed 2: 0.772 → 0.123). It concludes: "Seed-pairing added nothing, so every Match-Query comparison should be read unpaired." Unpaired, the loop survives (**+0.346, se 0.092, t 3.75**). The interaction +0.315 has no unpaired re-analysis. | Say the seeds were pooled from three launches. Report Match-Query contrasts unpaired: cite +0.346 (t 3.75) for the loop. Call the interaction unmeasured unless recomputed unpaired. This applies to +0.348, +0.414, +0.099, +0.315 and, by the same logic, MQ_RANK's +0.154 / +0.149. |
| E4 | 6.4, l.846: "the effect exceeded 0.150 at all three levels, firing the pre-registered outcome B" | outcome B fired | `ALIASING_CONTROLLED.md` Verdict: "**NOT ALL ARMS CONVERGED. Nothing here is interpretable**". In `agg_alias.py` (l.208) the convergence check comes before the OUTCOME branches. | "…meets outcome B's numeric condition, but the scripted verdict was withheld for non-convergence." |
| E5 | 7.1, l.985: "the monotone arm beats it nowhere (every \arm{Abs} $-$ \arm{RoPE} contrast is unmeasured…)". Echoed at l.920–921 ("leaves no measurable gain over an index code") and in the abstract, l.51–52 ("removes every measurable gain over an index code"). | monotone ≤ index everywhere | `SIGN_ABLATION.md`, `Abs_r4 - RoPE` **raw**: "+0.146 (… MDE 0.055, 1/12 neg)", "+0.226 (… MDE 0.059, 0/12 neg)", "+0.213 (… MDE 0.097, 0/12 neg)". All three are detectable. Only the loss-matched column is unmeasured or negative. The report itself says (l.700–702) that loss-matching against RoPE "partials out the very deficit the index code has". | Qualify every instance with "at matched loss", and give the raw +0.213 at T=1024. |
| E6 | 9, l.1764–1765: "adds $+0.007$, $+0.037$ and $+0.101$ … (grid 64; only the last detectable, \mde\ $0.065$)" | only T=1024 detectable | `POPE_WRAPPING.md` grid 64, T=512: "+0.037 \| 0.032 \| 0.032 \| 7/8 \| DETECTABLE" | "the last two detectable (MDE 0.032, 0.065)" |
| E7 | 8.2, l.1633–1635: "the full generator is not better than MapFormer's: … $+0.031$ (\mde\ $0.030$, $7/8$) at $T=512$" | not better | `SELECTIVE_ROPE.md` torus: "SRoPEGen … +0.031 (sd 0.030, MDE 0.030, 7/8)". 0.031 > 0.030 is detectable under the report's own rule (l.383). The source's "does not beat" contradicts its own table. | "not better on parity (−0.009, unmeasured); marginally better on the torus at T=512 (+0.031, MDE 0.030), unmeasured at T=1024, with 16,385 more parameters". |
| E8 | App. H, l.2193–2194: "the registered interaction between tasks is unmeasured ($+0.113$, \mde\ $0.195$)" | +0.113 is the registered contrast | `N5_RESULTS.md` P2 (registered): "(plus - minus)_torus - (plus - minus)_recency = +0.058, MDE 0.182 UNMEASURED". +0.113 is the "restated" \|rho\| form, chosen after the data (`AUDIT_2026-09-10.md` #5). | Give +0.058 (MDE 0.182) as registered, and +0.113 (MDE 0.195) as post hoc. |
| E9 | App. H, l.2201: "minus a magnitude-locked control is $+0.165$" | AlignLock is magnitude-locked | `D5_RESULTS.md`: "`AlignFree - AlignLock` (phase) \| +0.165"; "The per-block scale ratio is free in `AlignLock`". AlignLock is phase-locked with magnitude free. | "minus a phase-locked control (magnitude free)" |
| E10 | App. H, l.2215: "Its exposure-matched cells were both at ceiling and uninformative." | both at ceiling | `SPREAD_RESULTS.md`: only "m16_e1200 - m4_e300 … ceiling, uninformative". The other pair, "m64_e1200 - m16_e300 \| -0.068 \| 0.133 \| 0.132 \| 1/8 \| unmeasured", is not at ceiling (0.928). (CLAUDE.md's START HERE block repeats the error; the file wins.) | "One exposure-matched pair was ceiling against ceiling; the other was unmeasured (−0.068, MDE 0.132)." |
| E11 | 8.4, l.1663–1665: "largest for the weakest arm ($+0.096$, $8/8$, RoPE with index position)" | RoPE+index is the weakest arm | `BASELINE_TABLE.md` sec. H: "RoPE + index \| 0.827 \| +0.096 (8/8)"; "RoPE + path-int \| 0.823 \| +0.070 (7/8)"; "the paper's own MapFormer-WM is last (0.823)". | "largest for a weak arm (RoPE + index, base 0.827)" |
| E12 | App. A Table `tab:gate-records`, l.1870–1871: "Match-Query $128^2$ … never-moved 0.0490"; "$n_{\mathrm{obs}}=4$ … never-moved 0.1515" | clean floors | Both come from the pre-fix scorer. `MATCH_GATES_128_16.md` reads 0.0490 and has no correction block. The same-day `MATCH_QUERY_GATES.md` says the old scorer "made ~44% of trials auto-fail" and read below chance. Both values are below their chance rates (0.0625, 0.25), which is that bug's signature. | Mark both as uncorrected-scorer values and re-run the validator, or drop them. Row 1 of the same table already flags the bug. |
| E13 | App. C, l.2027–2028: "path integration over index grows from $+0.115$ to $+0.180$" | reads as the Table `tab:family` contrast (MapWM-Flat − Plain-Flat +0.205) | `FAMILY_TREE_RESULTS.md`: "commutative − index … **+0.115**" (MapEM single-p0 minus Plain-Flat, earlier batch). `FAMILY_TREE_D7_RESULTS.md`: "path integration − index \| +0.115 \| +0.180". | "commutative MapEM − Plain-Flat grows from +0.115 to +0.180" |
| E14 | App. B, l.1973–1976: TEM-t "$0.759\pm0.011$ and $0.668\pm0.022$ under action noise", next to Table `tab:april` (Vanilla noise 0.954 / 0.739) | comparable protocols | `TEM_T_MULTISEED.md` uses eval-time noise (`run_tem_t_multiseed.sh`: `eval_noise = 0.10`). On that protocol Vanilla is "0.757 ± 0.013 \| 0.638 ± 0.035", so TEM-t ties or beats Vanilla. `RESULTS_PAPER.md` evaluates without noise. TEMFaithful numbers come from `eval_single_env.py` on the seed-0 training map (per the appendix-checking pass; not re-verified by me). | State the protocol and give Vanilla 0.757 / 0.638 beside TEM-t. Note TEMFaithful is scored on the training map. |
| E15 | 4, l.540: "paper task $0.898\pm0.108$ against single $p_0$'s $0.987\pm0.012$"; App. B l.1988–1989: "identically on the same and a fresh map" | 0.898 on both maps | `PAPER_TASK_ACCURACY.md`: "VanillaEM \| 0.898 ± 0.108 \| 0.901 ± 0.102" (same-map \| fresh-map). l.93 says every evaluation uses a fresh map. | Use 0.901 ± 0.102. Say "identical for MapWM and single-origin EM". |
| E16 | 7.3 Table `tab:gate-ablation`, l.1155: "equalize & all token magnitudes equalised" | all tokens | `RECENCY_RESULTS.md`: "`equalize` condition (filler increments set equal to content)". `ablate_recency_gate.py`: "Delta on filler := the sequence's mean Delta on CONTENT tokens". | "filler increments set to the mean content increment" |
| E17 | 7.4, l.1187 (and l.1777): "ablating the 32 lowest-frequency channels" | reads as all 32 channels (l.1073: "MapWM's 32") | `LOCALISATION.md` counts channels "/ 64". `probe_localisation.py` flattens `omega` of shape (2, 32) and takes the lowest k=32 across heads. | "the 32 lowest-frequency of its 64 channels (2 heads × 32)". Also make l.1073 "32 per head". |
| E18 | 7.1 Table `tab:sign-contrasts`, l.974: "\arm{CARoPE} $-$ \arm{Signed} … $-0.134$ (\mde\ $0.118$)" | no verdict, while neighbours say "detectable" | `SIGN_ABLATION.md`: "-0.134 (sd 0.146, t -3.18, MDE 0.118, 10/12 neg)", "**DETECTABLE NEGATIVE**" | Add "detectable". |
| E19 | 6.3.1 / 7.1, l.1228–1229: "with each arm's worst seed dropped is $+0.0000$ (\mde\ $0.169$)" | the MDE belongs to the trimmed tie | `VOCAB_EM.md`: MDE 0.169 is for the full paired contrast (+0.0597), "inflated by exactly that collapse". A worst-seed-dropped difference is not paired and has no MDE. | "full contrast +0.060 (MDE 0.169), all from one collapsed MapWM seed; with worst seeds dropped, +0.000" |
| E20 | 9 (correction), l.1684–1685: "The shortcut gates hold at their clean levels (… never-moved $0.1042$)" | never-moved at clean level | Clean 64² never-moved is 0.0893 (`MATCH_QUERY_GATES.md` correction), so 0.1042 is 1.17× higher. The n-grams do hold. | "the n-gram gates hold at chance; never-moved rises to 0.104 (clean 0.089)" |
| E21 | 5.1 Table `tab:knob` caption, l.745–747: "the eight-seed file does not restate its recipe" | budget unknown | `run_seeds_n8.sh` phase 2: `baseline:98`, `allcombined:98`, `rotate:392:…--score-moves-only`, `allocentric:392`, all `--epochs 16`. | "At n=8, rotate and allocentric were trained at 392 batches (matched supervised events), baseline and combined at 98." |
| E22 | App. J, l.2273: "no BLiMP gain ($0.78$ against $0.79$)", after "MapWM … $18.79\pm0.15$ … against RoPE's $19.14$" | MapWM 0.78 | `papers/txt/mapformer.txt` l.1994: "RoPE and MapWM perform on-par, with a 0.78 vs 0.79". Fig. 7 table: RoPE 0.78±0.03, MapWM 0.79±0.02. | "(0.79 against 0.78)" |
| E23 | App. J, l.2285–2286: "all arms are at $0.00\%$ error in distribution" | all 0.00% | `FLIPFLOP_RESULTS.md` table: `CARoPE_r4` in-distribution 0.03 (its prose says all 0.00%, contradicting the table) | "at or below 0.03%" |
| E24 | App. E, l.2109: "(best $0.825$ against $0.953$)" | commanded best 0.953 | `MINIGRID_FULL_2X2X2.md` T=1024: PoPE-Hier 0.955 ± 0.003. 0.953 is PoPE-Flat, the best before the 8th cell existed. | 0.955 |
| E25 | 7.2, l.1087: "rests on one independent detectable cell ($D=5$)" | the only D=5 number in the paragraph is +0.055, which is unmeasured | `DXR_RANK_THRESHOLD.md` D=5, r=2 deficit "+0.055 \| … \| unmeasured". The detectable cell is r=D+2 minus r=D at D=5, +0.073 (t 3.35, 8/8), stated in `mapformer_math.tex`, not in the cited files. | Name the cell and its numbers, and add `mapformer_math.tex` (and the paper, App. D.1, for "r=D") to `% src`. |
| E26 | App. G, l.2250–2253: per-count list "0.384, 0.835, 0.866, …" then "oracle … $0.854$ against $0.847$ for the best single count" | best single count (0.866 listed) exceeds the oracle (0.854) | `LOOP_DEPTH_STRATA.md`: "overall" is token-pooled. `oracle` and `fixed` are unweighted means over the 5 strata; k=3's strata-mean is 0.847. The numbers are right, but side by side they contradict. | List the strata-mean values (0.370, 0.817, 0.847, 0.845, 0.845, 0.842, 0.837), or say "on an equal-weight strata mean". |

### 1b. Numeric imprecisions (should fix; the number is not wrong)

- **l.51.** "costs $0.28$ at matched loss" gives no length. It is T=1024 (8× training length); T=512 is −0.215, and T=128 is −0.006, unmeasured.
- **l.58.** "MapEM … trails MapWM by $0.375$" is the single-origin ablation. Paper-faithful separate origins trail by −0.137 (MDE 0.106).
- **l.390.** "On three occasions". `PAIRSPLIT_RESULTS.md` calls itself the "Fourth instance of the pattern". Also, +0.215 is pooled n=48, not fresh seeds.
- **l.418–419.** "the same change". On Match-Query the change also doubled epochs (`run_mq_noise_c2.sh`: EP=600). On the compositional task it also changed schedule and epochs.
- **l.456, Tables `tab:repro` and `tab:paper2x2`.** "Our EM is the paper's MapEM-os". The rows are `VanillaEM_P0`, the single-$p_0$ ablation (`PAPER_OOD_EXTENDED_n8.md`: "MapEM-os (VanillaEM_P0)"). l.216–218 itself says the paper uses separate $q_0,k_0$.
- **l.476/479.** OOD-s is labelled $l=512$. `PAPER_OOD_RERUN.md` notes that the v4 Table 2 caption gives l=256 ("the paper is internally inconsistent"). Ours at l=256: WM 0.978/0.984, EM 0.988/0.988.
- **l.152.** "40 sources". `papers/txt` holds 42 files; the number comes from a memory note. Low stakes.
- **l.571.** "No sd or \mde\ was recorded". It is recomputable from `INDEX_BASELINE_PAPER_TASK_n8.json`: +0.461, sd 0.027, MDE 0.027, 8/8 (the same pairing reproduces encoding +0.003, MDE 0.029, 5/8).
- **l.612.** "about 50 times the per-arm sd". The actual range is 14–68×; EM with actions resampled is 0.626/0.0449 = 14×.
- **l.639.** Context destruction 0.918 → 0.074 is n=3 (`MATCH_QUERY_RESULTS.md`), not the n=5 of the surrounding sentences.
- **l.655.** "The effect shrinks with length". It rises first (+0.316 → +0.326 at L=32).
- **l.703.** "On Match-Query at the better recipe". `run_loop_headroom.sh` passes no `--lr`, so it ran at 3e-4 cosine, not the 1e-3 torus recipe just named.
- **l.756, l.2096.** The all-knobs-combined row lacks `KNOB_SWEEP.md`'s gate caveat: "Order-3 reaches 0.634 against a 0.536 marginal … should be read as approximate".
- **l.813.** The parameter range omits RoPE (614,090). Across five arms the spread is 0.18%, not under 0.11%.
- **l.925.** "the difference between the two tasks is detectable ($-0.287$)". This is Q3, for `SRoPEGen` only, on the raw torus cost. l.1124 has it right.
- **l.937.** "no negative increment in any constrained checkpoint". `SIGN_PROBE` probed seeds 0–2 of 12 per arm.
- **l.973.** The Pos/CARoPE T=128 cells are "--" although recorded: −0.004 (MDE 0.010) and −0.017 (MDE 0.023), both unmeasured.
- **l.985.** The signed arm is also detectably negative against RoPE at T=128 loss-matched (−0.021, MDE 0.013). The prose implies only the monotone arm is.
- **l.1086.** "the paper's own 5D run used $r=D$" is correct (v4 App. D.1), but the `% src` does not name the paper.
- **l.1120–1123.** The T=2048 values (−0.015, −0.052) are in MONOTONE's "Exploratory (not registered)" table; the prose calls the batch pre-registered.
- **l.1203–1205.** "between $-0.936$ and $-0.986$". MONOTONE recency is −0.987, and PAPERTASK in the same section is −0.461, so "every contrast in this section is a contrast of fit" needs "every recency contrast".
- **l.1403.** "the full 64-offset task from 0.578 to 0.928". These are the primary readout (k∈{4,16,64}); the whole trained set reads 0.609 → 0.930.
- **l.1466.** "Rule 9: $r=-0.954$ over 96 runs" covers EMPair and EMPairConst only.
- **l.1390.** The curriculum "epochs" cell is "--"; `SEARCH_RESULTS.md` has "11 (8/8) *, not comparable".
- **l.1549–1550.** "only recursion measurably adds accuracy at training length". `HIER_PARITY.md` at its training length L=16: "+0.012 path-int (16/16, MDE 0.002)". It is tiny, but detectable.
- **l.1593–1594.** "the best Match-Query arm measured". This ranks across batches on a task the report says does not reproduce across batches (l.442–445). Say "best in its batch".
- **l.1722–1723.** At n=12 the raw +0.129 (MDE 0.103) is detectable; the loss-matched +0.083 sits at its MDE (t 2.79). So "detectable … only after loss-matching" holds at n=5 only.
- **l.1967.** LSTM/Vanilla at T=2048 come from `long_sequence_eval.py`, which evaluates on the training map, next to a "fresh map" table caption.
- **l.2014, l.2017.** "MapEM-os (commutative)". The control is `VanillaEM_P0` (`ABLATE_FAMILY_TREE.md`); the report elsewhere calls this single-origin.
- **l.2029–2030.** "costs 14.4× length scaling". 14.4× is MapEM-NC-**L**; NC-NL, the family-tree arm, was never timed. Commutative EM grows 3.9×.
- **l.2115 vs l.783.** +0.263 vs +0.264 for the same 980-batch runs (`CONTINUOUS_ALLOC.md` vs `H12_BUDGET_CURVE.md`, two evaluations). Footnote it or use one.
- **l.2125.** "raises every arm by about 0.15". It is 0.148–0.178 (RoPE +0.178).
- **l.2172, l.2143.** The +0.019 is raw at T=1024; +0.018 and +0.028 are at T=512 and T=1024. The lengths are unstated.

---

## 2. Overclaims and language (should fix)

| # | Location | Says | Problem (source) | Suggested fix |
|---|---|---|---|---|
| O1 | Abstract l.59–60: "per-pair position origins recover $0.215$ of it ($n=48$)" | per-pair freedom recovers 0.215 | Three problems. (a) `PAIRSPLIT_RESULTS.md`: +0.124 of the 0.215 comes from `EMPairConst`, which keeps the kernel shared; freedom is +0.091. (b) The 0.375 gap is an n=8 contrast from another batch; the n=48 single-origin mean is 0.683 (computed from the per-run outputs; it is not in the results file), not 0.600. (c) EMPair − MapWM is −0.095 (MDE 0.135), unmeasured (l.1484). | "per-pair origins add 0.215 (n=48), of which 0.091 is per-pair freedom and 0.124 pathway capacity" |
| O2 | Abstract l.53–54: "On a counting task the same constraint costs little." Also l.925–926: "small on a counting task", and l.1125–1126. | monotone is cheap on recency | Same batch, same task: MapEM monotone − signed is **−0.198 (MDE 0.155), detectable**, and −0.169 at T=2048 (`MONOTONE_RESULTS.md` P1), against a torus cost of 0.28 loss-matched. The report itself says this at l.1436. `STORY.md` §6.4 pre-agreed that a detectable Q2 (it is: −0.068) weakens "the task sets the value" to "for MapFormer's generator". (MapEM's matched-loss cost is −0.043, unmeasured, but the torus 0.28 is also quoted at matched loss while recency is quoted raw.) | Scope it to MapWM and Selective RoPE's generator, name MapEM's −0.198, and use one scale (raw or loss-matched) on both tasks. |
| O3 | Intro l.131: "each established by intervention". Conclusion l.1840–1841: "isolated by single-operation interventions (the action record, the absolute value, the rank)" | "well-conditioned" is established by intervention | l.1065–1066: "the causal route from skew to accuracy is not tested". The rank intervention shows that r matters; conditioning is a description. Condition 3 rests on construction plus descriptive anatomy. "pays only under three conditions" (l.131) asserts necessity, but l.144 disclaims sufficiency and exhaustiveness, and the conditions were shown on different tasks. | "each is supported by an intervention on its own task; that conditioning is the route for rank is descriptive". Drop "only". |
| O4 | Claim 1 title (l.550), intro l.130, contributions row l.159: "not the choice of rotary encoding" | the encoding does not matter | The +0.003 encoding effect is measured with index arms at floor (0.509–0.530) and path-integrated arms near ceiling (0.967–0.994), where an encoding effect is compressed. Elsewhere PoPE on top of path integration is detectable (+0.037 at T=512, +0.101 at T=1024, `POPE_WRAPPING.md`), and encoding is MiniGrid's largest main effect (+0.076). PAPER2X2 is pending, and the l.159 row is unhedged. | "at training length on the paper task under the paper recipe". Add the floor/ceiling caveat and PoPE's OOD effect to Section 6.1. |
| O5 | 6.3.1 l.1197: "On map tasks it ties MapWM once its initialisation is fixed"; l.1229 "These are directional ties"; l.1535–1536 "the vocabulary sweep … shows no capacity advantage" | tie / no advantage | The ties rest on a contrast with no MDE (MiniGrid) and on MDE 0.169 (vocab). In the same section, PAPERTASK shows EM − WM +0.186 and +0.287 at l=1024/2048, 8/8 (exploratory). | "no detectable difference at training length" |
| O6 | l.1300 "there MapEM learns faster"; l.1537 "MapEM learned faster (54 against 99 epochs)" | learns faster | These are medians of epochs-to-threshold, n=8, with no inferential contrast (`SEARCH_RESULTS.md` S3). | "reached loss < 0.5 in a median 54 epochs against 99 (descriptive)" |
| O7 | 6.3.2 l.1243: "Rule 9 gives $r=-0.461$, so this is not a loss gap". l.1765–1766 states "MapEM degrades more slowly than MapWM" with no qualifier. | not a loss gap; unqualified | A weak correlation does not show absence (the source says "not simply a loss gap"). The convergence gate failed, so the result is exploratory (trap). | "not simply a loss gap"; add "(exploratory; gate failed)" at l.1766. |
| O8 | 6.3.5 l.1428–1429: "at per-token grain, failures are search" | mechanism conclusion | A descriptive probe of 8 checkpoints (the report says so in the next sentence). | "consistent with search" |
| O9 | 9 heading l.1674: "a powered negative"; l.1691: "a benefit larger than about $0.07$ is excluded" | powered negative | The registered prediction was the slope with drift, and that is unmeasured (p=0 MDEs 0.264/0.326). Arms are unconverged at p=0.10 and scores sit near a ~0.15 floor (l.1713–1716). An MDE is a power threshold, not a confidence bound (take-1 one-sided upper bound ≈ 0.084). | "no measurable benefit". Keep "powered" only for "benefit at p=0.10 larger than ~0.07 would likely have been detected, in unconverged arms". |
| O10 | l.1728: "not inference from observations"; App. I l.2237: "The landmark gain is capacity or optimisation, not measurement." | excludes inference/measurement | Every component ablation is unmeasured (not null). The landmark control is a tie at t 0.79, n=3. `LM200_ABLATION.md` says only "it is not evidence for the Kalman mechanism". | "no evidence that it works by inference / measurement" |
| O11 | App. E l.2077–2078: "Hierarchy's gain is generic compression, not alignment with task structure." | mechanism | This rests on n=3 oracle-pooling nulls, and the hierarchy gain itself is unmeasured (+0.136, MDE 0.173). | "Room-aligned pooling did not help (n=3)." |
| O12 | 7.2 l.1030: "Selective RoPE's sigmoid gate reached a similar $+0.086$"; App. G l.2165–2166: gate "on the torus, where it helps, … on parity, where it hurts" | attributes effects to the gate | `SELECTIVE_ROPE.md` CONFOUND block: every single-knob arm also replaces `diag(omega) W_out` with `tau I`; "The per-knob rows cannot attribute their effects to the conv, the rank, or the gate." (trap) | "the GateAngle arm (gate plus readout swap)"; "where that arm gains / loses". |
| O13 | l.1631: "MapFormer's bottleneck already separates the two token types." | measured separation | This is not measured in the gated batch. `GATED_RESULTS.md` cites a separate probe ("about five times more on actions than observations") on a trained ungated model. | Cite that probe and its n, or write "suggests". |
| O14 | 5.1 l.773–774: "restore the effect to above its baseline" | above baseline | +0.488 vs +0.438 has no sd/MDE for the effects, and allocentric was trained at 392 batches vs baseline 98 (event-matched by design, E21). | "to at least its baseline level at matched supervised events" |
| O15 | Claim 2 opening l.724–725: "with the sign inverted, it is larger when aliasing is lower" | settled direction | The endpoints cross budgets (800 vs 400 ep) and batches. The less-aliased endpoint is n=3 (exploratory). The 400-ep table's scripted verdict was withheld (E4). | Add "(exploratory; endpoints at different budgets)", as l.878–882 already does for extent. |
| O16 | l.682–683: "The index arms are at a capability limit, not undertrained: doubling the budget moves … (3 seeds)" | settled at n=3 | n ≤ 3 must be labelled exploratory (l.386). | Add "exploratory". |
| O17 | Unlabelled n=3 headings in appendices: l.2100 "Frequency learning is not the position effect" (n=3); l.2256 "Recursion buys an index model horizon" (n=3); l.2123–2126 fixed-map MiniWorld (n=3); l.2227–2231 refine-theta (n=3) | stated as findings | Rule l.386 | Label each exploratory, or add a blanket statement per appendix. |
| O18 | 8.1 l.1589: "Both contrasts are detectable and the interaction is super-additive." Abstract l.65: "Weight-shared recursion does add accuracy" | paired verdicts | See E3: paired stats are invalid on this task. The loop main effect survives unpaired (t 3.75); the super-additive interaction is not re-established. | "The loop effect survives an unpaired analysis; the interaction is not re-established unpaired." |
| O19 | 7.1 l.1010–1011: "The sign cost on navigation is therefore not specific to MapFormer's generator." Abstract l.51–53 mixes "0.28 at matched loss" (MapWM) with "the same raw cost" (SRoPE). | established generally | This holds raw (−0.355 vs −0.363). The registered loss-matched Q1 was falsified as registered, and the loss-matched support is post hoc (8-arm pool). | Say "raw" in both places, and keep the scale consistent in the abstract. |
| O20 | App. J l.2274: "loses to CoPE and PaTH on length extrapolation" | head-to-head loss | v4 App. B.5 compares with "the numbers reported for CoPE and PathAtt"; the main text says MapWM has "better length extrapolation" than RoPE. | "degrades more than the numbers reported for CoPE and PaTH" |
| O21 | 6.2 l.636–638 (64² Match-Query, n=5) | one comparison | `MATCH_QUERY_SCALE.md` correction: seeds 3–4 were evaluated and trained separately. The unpaired no-overlap claim survives, but the pooling crosses batches on a non-reproducing task. | One clause: "seeds pooled across two runs; unpaired". |
| O22 | App. B l.1980–1985: "Stitching, ported from CSCG" | a valid task | **Uncertain.** Inventory C `## Excluded` lists "CSCG stitch and schema tasks" (killed by `CSCG_TASK_GATES.md`: "negative control is still defeatable without a map: 0.617"; "Only ~14% of 'shared' scored events actually require stitching"). `RESULTS_INDEX.md` l.398 implies stitch was redesigned ("needs the redesign stitch received") and lists `STITCH_ATTENTION.md` as current. The report does not say which environment version was used or that it passed re-run gates. | State the redesign and its gates, or drop the paragraph. |
| O23 | l.1353 scope and l.1344: "The residual at 8× was content leaking into $\Delta$" | causal, fully | `NOLEAK_RESULTS.md` supports this by seed count (8/8 vs 4/8). The paired contrast is at ceiling and unmeasured (the report says so). `UNFREEZE_RESULTS.md`'s top block withdrew "two channels are exhaustive". | "Closing the leak removes the residual (8/8 vs 4/8 seeds ≥ 0.95)" |
| O24 | l.1488: "Mechanism, by intervention. … Freedom, not capacity, changes the mechanism." | intervention-established | The arms are interventions, but the readout is descriptive at n=8 (as stated), with no inferential contrast on route fraction. | Acceptable; add "on n=8 descriptive readouts" to the heading sentence. |
| O25 | l.1062: "so the paper is right about what is expressible" | expressibility | The learned plane energy says what trained codes use, not what is expressible; r=1 was never run. | "the learned codes are two-dimensional, as the paper's argument expects" |
| O26 | l.1691–1692 with Table `tab:mqnoise`: "refuted twice, in two recipes" | refutation | The registered slope is unmeasured in both takes (p=0 MDE > 0.26). The prediction "grows with drift" was not detected rather than refuted. The report says the slope is unmeasured one sentence later. | "not supported in either take (slope unmeasured)" |
| O27 | Contributions l.166: "State correction does not scale with drift" | established | Same as O26 | "no detectable scaling with drift" |

---

## 3. Referee objections per claim

**Claim 1: the accumulated phase, not the encoding, carries transfer.**
1. *Index baselines are undertrained, and at matched loss on the converged torus the index code is detectably better at T=128 (−0.021, MDE 0.013).* **Partly answered.** Section 6.5 concedes this, and PAPER2X2 is pending. The text around `\PENDING` does not presuppose the outcome. The contributions row l.159 does presuppose it, and should say "under the paper recipe; converged replication pending".
2. *The encoding main effect is measured where cells sit at floor and ceiling, and PoPE helps detectably beyond training length.* **Unanswered.** Cheap fix: O4.
3. *Match-Query evidence uses paired statistics on a task where same-seed retrains drift by 0.185.* **Unanswered.** The report says only "within-batch contrasts are cited", and the contrasts cited pool three launches. Cheap fix: E3 (the unpaired t 3.75 already exists for the loop; recompute Q1 unpaired).
4. *"Transfer" is in-context binding on a redrawn map, not transfer.* **Answered** (l.93–97, Limitations).

**Claim 2: displacement must be a function of the token.**
1. *Allocentric recoding hands the model the egocentric-to-allocentric transform, which is the hard part of path integration.* **Partly answered** (l.774: "supplies an integrable input, not the target"). Add that heading-to-displacement is exactly the computation removed. The result shows MapFormer cannot learn that transform, not that it is unnecessary.
2. *Budgets differ between the baseline and rotate/allocentric rows.* **Partly answered.** The caption discusses the 3-seed budget but wrongly says the 8-seed budget is unknown (E21). Fix the caption.
3. *The MiniGrid effect is tiny (+0.034) and conditional on r=4. The commanded factorial's main effects have no sd.* **Answered** (l.805–806, l.797).
4. *The aliasing falsification rests on unconverged 400-epoch runs and a cross-budget n=3 endpoint; map extent is post-hoc pooled.* **Partly answered.** Extent is well caveated (l.878–882). The aliasing sentence claims a fired outcome (E4) and a settled direction (O15).

**Claim 3: the increment must cancel, and cancel cleanly.**
1. *"No gain over an index code" is a loss-matched artefact: raw, the monotone arm beats RoPE by +0.213 at T=1024.* **Unanswered, and the headline as written is wrong for raw accuracy** (E5). Fix: add "at matched loss" everywhere, and report the raw gain.
2. *The "task sets the value" crossover fails for the shared-kernel architecture (MapEM −0.198 on recency).* **Unanswered in the claim opening and abstract**, answered in Section 6.3.6. Fix: O2.
3. *The rank benefit is correlational with skew: r=1 and r=3 not run, the causal route untested, torus and MapWM only.* **Answered** in 7.2, but overstated in the Intro and Conclusion (O3).
4. *The monotone arms carry an init confound and a loss gap; loss-matching conditions on a mediator.* **Answered** (l.397–400, l.993–994; the Abs arm is the clean isolation).

**Claim 4: a shared kernel is representable but harder to find.**
1. *One task. MapWM not given the 4× budget, so −0.375 may be a learning-rate difference rather than a search obstacle specific to EM. Every contrast is a fit contrast (|r| ≈ 0.95).* **Answered as a limitation** (l.1300–1302, l.1539–1541). The abstract's "trace the gap to per-token search" is stronger than that scope.
2. *Holdability is shown only with $W_{\mathrm{in}}$ content columns pinned.* **Answered** (l.1350–1352, Limitations).
3. *Per-pair origins' gain is mostly capacity, and they do not close the gap to MapWM.* **Answered in 6.3.7**, overstated in the abstract (O1).
4. *EM beats WM at length on the paper task, contradicting "ties on map tasks".* **Partly answered** (exploratory label in 6.3.2). The claim opening and 9 are unqualified (O5, O7).

**Claim 5: what added machinery buys.**
1. *The Match-Query loop numbers use paired statistics on pooled, non-reproducible runs, and the "never fails" floor claim is retracted.* **Unanswered** (E2, E3). Cheap fix: cite `REFINE_RESULTS.md`'s unpaired +0.346 (t 3.75), drop the floor sentence, and mark the interaction unmeasured unless recomputed.
2. *If the torus loop gain is convergence, the Match-Query gain may be too.* **Partly answered** ("where the base model trains unreliably"). No loss-matched Match-Query analysis is given; state it as associated.
3. *"Powered negative" for the filter while the registered slope is unmeasured and arms are unconverged.* **Partly answered** (the caveats are in the text; the heading and "refuted twice" overstate). Fix: O9, O26.
4. *Selective RoPE's full generator is detectably better at T=512.* **Unanswered** (E7).

---

## 4. Numbers checked

About 1,700 items (numbers, table cells, seed counts, verdict words). **26 errors** (Section 1) and **31 imprecisions** (1b). Everything below matched its primary file.

**Abstract, introduction, background, methods (l.1–445)**
- l.49 +0.438/+0.050/+0.488, n=8 → `KNOB_SWEEP_n8.md`
- l.51 0.28 (12/12) → `SIGN_ABLATION.md`
- l.53 −0.355 vs −0.363 → `MONOTONE_RESULTS.md`
- l.55 384 params → `RANK_SWEEP.md`
- l.58 0.375, 0/8 → `RECENCY_EM_RESULTS.md`; "exactly" (1.000, 8/8) → `WARM_RESULTS.md`
- l.60 0.215, n=48 → `PAIRSPLIT_RESULTS.md`
- l.109 three days → `papers/INDEX.md`
- l.125 HGRN Table 11, HGRN2 Table 1 → `papers/INDEX.md`
- l.203–205 eq. 18 → `papers/txt/mapformer.txt`; l.217 App. A.7 quote → mapformer.txt l.1414/1530
- l.264–267 v1 and v4 Table 2; Table 6 5D 0.75/0.50/0.35; 19.14±0.14 / 18.79±0.15 → mapformer.txt, `LANGUAGE_LANDSCAPE.md`
- l.278 d=128 ~204k; l.291 204,757 → `runs/sign` logs
- l.305–307 n=16, −0.0023 (MDE 0.0086) → `ROPE_CANONICAL.md`
- Table `tab:tasks`, all floors and lengths → `PAPER_TASK_ABLATION.md`, `PAPER_TASK_FLOORS.md`, `KNOB_SWEEP_n8.md`, `MATCH_QUERY_GATES.md`, `ALGORITHMIC_GATES.md`, `RECENCY_GATES_K64.md`, `MINIGRID_EM.md`, `ALIASING_GATES.md`
- l.353 128.3±11.8 → `RECENCY_GATES_K64.md` (other project files quote 129.7±10.3 at a different gap; not an error)
- l.391 +0.237/+0.073 → `DOF_RESULTS.md`, `D5_RESULTS.md`; +0.213/+0.113 → `MAGONLY_RESULTS.md`; +0.280/+0.215 → `PAIRSPLIT_RESULTS.md`
- l.415 +0.160 (MDE 0.140, 7/8) → `COMP_HEADROOM.md`
- l.417–419 sd 0.096 → 0.028 → `RECIPE_POWER.md`; 0.263 → 0.261 → `MQ_NOISE_2X2*.md`
- Table `tab:recipes`, every row → run scripts named by the checker (`run_sign.sh`, `run_rank_sweep.sh`, `run_recency_em.sh`, `run_loop_headroom.sh`, `run_minigrid_em.sh`)
- l.443 0.456 / 0.416 → `LOOP_HEADROOM.md`, `MQ_RANK_2X2.md`

**Reproduction (l.452–547)**
- l.458 0.969 / 0.968 → `PAPER_OOD_EXTENDED_n8.md`, `PAPER_OOD_RERUN.md`
- Table `tab:repro`: all 12 "ours" cells, paper columns, floors 0.522/0.216/0.799 → same files, `PAPER_TASK_FLOORS.md`
- l.486–493 Fig. 9 / C.1 / C.3 → mapformer.txt l.2014–2264; one of eight readings > 1 → `PAPER_FIG4_EM.md`
- Table `tab:fig4`, all 16 cells → `PAPER_FIG4_REPRO.md`, `PAPER_FIG4_EM.md`
- l.515–516 and Table `tab:timing`, every cell → `TIMING_BENCHMARK.md`
- l.540–546: 0.987±0.012; +0.358 (3/3); −0.154 (MDE 0.130, 0/8); +0.013 (MDE 0.008, 3/3); +0.205 (MDE 0.118, 3/3) → `PAPER_TASK_ACCURACY.md`, `MATCH_QUERY_EM.md`, `DOF_RESULTS.md`, `FAMILY_TREE_WM_GAP.md`, `N3_AUDIT.md`

**Claim 1 (l.554–712)**
- l.556–559 +0.461, +0.003 (MDE 0.029), +0.750 (MDE 0.030, 8/8) → `BASELINE_TABLE.md`, `RECENCY_RESULTS.md`
- l.568 within 0.4% → `INDEX_BASELINE_PAPER_TASK.md`
- Table `tab:paper2x2`: 6 accuracies, 5/8 → `INDEX_BASELINE_PAPER_TASK_n8.md`
- l.612–618: 0.9889, 0.2314, 0.1783, 0.2915, 0.3092, 0.9873, 0.2680–0.3831, NLL 3.68–4.94 vs 0.006–0.100, ln 21, 0.362 vs 0.178 → `PAPER_TASK_ABLATION.md`
- l.626–627 0.557/0.575/0.504/0.881, and Table `tab:revisit` (35 cells) → `REVISIT_DISTANCE.md`
- l.636–640 0.730±0.247, 0.154±0.018, 0.398, 0.178, 0.0893, 0.918 → 0.074/0.076 → `MATCH_QUERY_SCALE.md`, `MATCH_QUERY_RESULTS.md`
- l.643–647 values → `LOOP_HEADROOM.md` (batch caveat E3), `MATCH_QUERY_LONGQ.md`
- Table `tab:parity`, 15 cells → `ALGORITHMIC_RESULTS.md`
- l.681–685 +0.750/+0.704, MDEs, 0.251 → 0.242, 0.247 → 0.248, per-offset ranges → `RECENCY_RESULTS.md`
- l.692–705 0.799±0.018, 0.7844 vs 0.0002, −0.021/+0.123/+0.195 with MDEs, r −0.978, +0.205 → `SIGN_ABLATION.md`, `N3_AUDIT.md`
- l.710 0.216, floor ~0.072 → `BASELINE_TABLE.md` D

**Claim 2 (l.719–911)**
- l.721–723, l.737 gates 0.501/0.472/0.440/0.462 vs 0.507 → `KNOB_SWEEP_n8.md`, `ALLOCENTRIC_RECODING.md`, `MINIGRID_EM.md`
- Table `tab:knob`, all 16 cells → `KNOB_SWEEP_n8.md`
- l.781–786 0.661, 0.555, floors, +0.264/+0.383/+0.286, r −0.996 over 18 → `H12_BUDGET_CURVE.md`
- l.795–807 0.955/0.953/0.823, floor 0.490, +0.076/+0.048/−0.021, +0.034 (0.014, 8/8), −0.010 (0.056, 4/8), ~0.02 → `MINIGRID_FULL_2X2X2.md`, `MINIGRID_EM.md`, `MINIGRID_ALLOCENTRIC_2X2X2.md`
- Table `tab:minigrid-allo`, all cells → `MINIGRID_EM.md`
- l.835–840 0/9, 0/3, +0.173, −0.010 (sd 0.007) → `MINIGRID_GRID_SWEEP.md`, `POSITION_EFFECT_CONVERGED.md`, `CROSSOVER_CONVERGED.md`
- l.843–850 50.4, 33, +0.305 (sd 0.048, MDE 0.077, 3/3), non-overlapping losses → `ALIASING_GATES.md`, `MINIWORLD_ENDPOINTS.md`, `ALIASING_CONTROLLED.md`
- Table `tab:alias`, all four rows → `ALIASING_CONTROLLED.md`, `MINIWORLD_ENDPOINTS.md`
- l.874–882 0.005/0.030/0.285, +0.275, −0.010, 0.971–0.994, 0.150 → `VISITS_TEST.md`, `MINIWORLD_ENDPOINTS.md`, `MINIWORLD_GATE_CONTROL.md`
- Table `tab:extent`, all cells → `VISITS_TEST.md`, `MINIWORLD_ENDPOINTS.md`
- l.908 69–91% → `HABITAT_BUILD.md`

**Claim 3 (l.918–1188)**
- l.920–924 → `SIGN_ABLATION.md`, `MONOTONE_RESULTS.md`, `RANK_SWEEP.md`
- l.936–939 +0.000/+0.006/+0.006, r values → `SIGN_ABLATION.md`
- Table `tab:sign` (24 cells) and `tab:sign-contrasts` (all but E18) → `SIGN_ABLATION.md`; losses +0.1706/+0.0674/+0.2927 12/12 → same
- l.994 init confound → `SIGN_ABLATION_PREREG.md`
- l.998–1000 opposition scores → `SIGN_PROBE.md`
- l.1005–1015 every SRoPEGen value, Q1 pool losses, −0.188 → `MONOTONE_RESULTS.md`
- l.1026–1031 and Table `tab:rank` (params, 15 cells, t 3.57, 8/8) → `RANK_SWEEP.md`
- l.1062–1067 1.0000/0.9996/0.4950/0.0922/0.7833/0.1754 → `ACTION_GEOMETRY.md`
- l.1072–1076 +0.019 (5/8, LM +0.015); +0.005 (0.091); +0.154 (0.084, 8/8) → `MAPPOPE_R4_RESULTS.md`, `MQ_RANK_2X2.md`
- l.1083–1086 0.896±0.076, +0.110/+0.153/+0.055 → `DXR_RANK_THRESHOLD.md`, `ND_GATES.md`
- l.1094–1097 and Table `tab:recency` → `RECENCY_RESULTS.md`
- l.1121–1124 → `MONOTONE_RESULTS.md`
- l.1130–1131 0.591±0.028, 0.967±0.009 (an sd), 1.010, 0.976 → `RECENCY_H2.md`
- l.1135–1140 and Table `tab:gate-ablation` values → `RECENCY_GATE_ABLATION.md`
- l.1167–1169 1423/1423, 1417/1417, 1415/1415 → `AUDIT_2026-09-10.md` #2
- l.1176–1188 0.518/0.524/0.619/0.943, r +0.9995, −0.001, −0.139±0.092 → `LOCALISATION.md`

**Claim 4 (l.1195–1541)**
- Opening paragraph, every number → `RECENCY_EM_RESULTS.md`, `WARM_RESULTS.md`, `NOLEAK_RESULTS.md`, `SEARCH_RESULTS.md`, `SPREAD2_RESULTS.md`, `PAIRSPLIT_RESULTS.md`, `PAIRCONST_RESULTS.md`
- l.1215–1218 29,056; 1.947 (1.83–2.08); 2.722; 0.000; 3.267; 25/64 → `EM_WM_THEORY.md` 1b
- l.1226–1227 +0.0012 (4/8), +0.0035 (6/8), 0.002 vs 0.137 → `MINIGRID_EM_FIX.md` (recomputed MDE 0.007/0.012, so "unmeasured" is right)
- l.1233–1236 → `DOF_RESULTS.md`, `D5_RESULTS.md`
- l.1241–1244 and Table `tab:papertask-em` (9 accuracies, 6 contrasts, floors) → `PAPERTASK_RESULTS.md`, `PAPER_OOD_RERUN.md`, `PAPER_TASK_FLOORS.md`
- l.1275–1280 and Table `tab:recency-em` → `RECENCY_EM_RESULTS.md`, `AUDIT_2026-09-10.md` #9
- Table `tab:ladder` (all cells, T=2048 column, seed counts) and l.1335–1348 → `WARM_RESULTS.md`, `UNFREEZE_RESULTS.md`, `NOLEAK_RESULTS.md`
- l.1361–1376 and Table `tab:fixedk` → `SEARCH_RESULTS.md` S1, S3
- l.1399–1404 and Table `tab:spread2` → `SPREAD2_RESULTS.md`, `SPREAD_RESULTS.md`
- l.1427–1431 0.992/0.995, 181/166, 0.359/0.544, 1.7–4.3e-4 → `THEORY_SEARCH_AND_LENGTH.md` T1, `SEARCH_RESULTS.md` S2
- l.1435–1447 → `MONOTONE_RESULTS.md`
- l.1457–1460 2,048 / 128 params, logit diff 0 → `PAIRORIGIN_RESULTS.md`, `PAIRCONST_RESULTS.md`
- Tables `tab:pairsplit` and `tab:pair-mech`, and l.1482–1485 → `PAIRSPLIT_RESULTS.md`, `PAIRORIGIN_RESULTS.md` (EMPair − MapWM is paired in one batch), `PAIRCONST_RESULTS.md`
- l.1521–1527 → `MAGONLY_RESULTS.md`, `D5_RESULTS.md`, `SEARCH_RESULTS.md` S1

**Claim 5, correction, length, limitations (l.1548–1829)**
- l.1551–1557 (batch caveat E3) → `LOOP_HEADROOM.md`, `MQ_RANK_2X2.md`, `L15_LOOP_2X2.md`, `GATED_RESULTS.md`
- Table `tab:loop` values → `LOOP_HEADROOM.md`
- l.1593–1595 0.986±0.020, 0.941, +0.154/+0.005/+0.149 → `MQ_RANK_2X2.md`
- l.1598–1601 → `FRONTIER_ALGORITHMIC.md`
- l.1604–1607 +0.052 (0.048, 12/12), +0.006 (0.017), r −0.956, 0.0076/0.1549, 3/8, 4/8, 6/8 → `L15_LOOP_2X2.md`, `RECIPE_POWER.md`
- l.1613 +0.007, sd 0.152 → `LOOP_DEPTH_STRATA.md`
- l.1621–1630 258 params, 0.987±0.003, 0.244±0.043, 4.16, 3.33–5.44, 1.35, 1.00, +0.004 (0.008, 5/8), +0.003 (0.024), −0.039 (0.079), −0.016 (0.055), +0.009 (0.008) → `GATED_RESULTS.md`, `GATED_SEPARATION.md`, `GATED_TORUS.md`
- l.1639–1644 +0.081 (0.080, 7/8, raw), −0.002, −0.016 (0.118), +0.097 (0.105), r −0.516, 5/8, r −0.531 → `FORGET_CONTROL.md`, `FORGET_GATE.md`, `LAMBDA_TRACE.md`
- l.1652–1671 +0.136/+0.151, +0.160, +0.060 (0.075, 12/12), +0.071 (0.081, 11/12), +0.002 (5/8), 22.9%/19.9%/22.8%/66.5%/12.2%, 0.0032, 1.23× → `HIER_RECHECK.md`, `COMP_HEADROOM.md`, `LOOP_HIER_PARITY.md`, `BASELINE_TABLE.md`, `LOOP_HIER_COMPUTE.md`, `ENWIK8_HIERARCHY.md`
- l.1683–1692 13.05; gate n-grams; +0.038/+0.005, +0.003/−0.141 → `MQ_NOISE_2X2*.md`, `MATCH_QUERY_GATES_P010.md`
- Table `tab:mqnoise` (recipes confirmed from logs) → `MQ_NOISE_2X2.md`, `MQ_NOISE_2X2_C2.md`
- l.1714–1717 45%, 0.15, +0.121 (0.065), +0.057 (0.025) → `MATCH_QUERY_NOISE_ABLATION.md`, `MQ_NOISE_2X2*.md`
- l.1720–1727 +0.062 (t 3.08), +0.124 (t 3.83), +0.057 (0.150), +0.120 (0.180), +0.073 (0.105), 0.830/0.878 → `L15_ABLATION.md`, `L15_LOOP_2X2.md`
- Table `tab:l15loop` (15 accuracies, 5 losses, 3 r) → `L15_LOOP_2X2.md`
- l.1752 +0.010, t 0.79 → `LM200_ABLATION.md` (a fresh current-code batch, not the April retraction)
- l.1770–1790 alpha contrasts, 285.6/283.9 at T=1024, −0.170±0.084, −0.254±0.164, +0.077/+0.095/+0.079, collision values, 53% → `ACCUMULATOR.md`, `LOCALISATION*.md`, `POPE_WRAPPING.md`, `THEORY_SEARCH_AND_LENGTH.md` T2
- l.1820, l.1825 → `HABITAT_BUILD.md`, `FLIPFLOP_RESULTS.md`

**Appendices (l.1857–2375)**
- Table `tab:gate-records`, all rows except E12 → the files in its `% src`
- l.1891–1895 → `HIERGOAL_ABLATION.md`, `HIERGOAL_CLOSEDLOOP.md`, `PLANNER_TASK_AUDIT.md`
- l.1919–1934 → `MATCH_QUERY_SCALE.md`, `MATCH_QUERY_LONGQ.md`, `MATCH_QUERY_NOISE_ABLATION.md`, `MINIWORLD_GATE_CONTROL.md`
- Table `tab:april` (24 cells) → `RESULTS_PAPER.md` (clean/noise OOD, declared valid; population sds); l.1942 → `NOISE_CLEAN_REVALIDATION.md`
- l.1967–1968 values → `LONG_SEQ_clean.md`
- l.1973–1978 values → `TEM_T_MULTISEED.md`, `TEM_BACKGROUND_BASELINES.md`, `TEM_NOISE_FFN_RESULTS.md` (protocol caveat E14)
- l.1981–1983 → `STITCH_ATTENTION.md`
- l.1988 per-seed 0.778/0.931/0.986 → `PAPER_VALIDATION.md`
- Table `tab:family` (10 cells, 4 contrasts), l.2026–2027 → `FAMILY_TREE_WM_GAP.md`, `N3_AUDIT.md`, `ABLATE_FAMILY_TREE.md`, `FAMILY_TREE_D7_RESULTS.md`
- App. E text and Table `tab:comp-headroom` → `COMPOSITIONAL_EXPERIMENT.md`, `COMPOSITIONAL_MULTISEED.md`, `COMP_HEADROOM.md`, `HIER_RECHECK.md`, `DISSOCIATION_SWEEP.md`, `HIER_PARITY.md`
- App. F: +0.478…−0.084, +0.004/−0.008, +0.013/+0.020, 0.817/0.809, H12 per-seed (18 values), Habitat percentages, +0.020±0.013 / +0.008±0.009 → `KNOB_SWEEP.md`, `FREQ_CONTROL.md`, `MINIGRID_ALLOCENTRIC_*.md`, `MINIGRID_REPRO_CONTROL.md`, `H12_BUDGET_CURVE.md`, `CONTINUOUS_ALLOC.md`, `HABITAT_BUILD.md`, `MINIWORLD_FIXED_RESULTS.md`
- App. G: 0.576/0.992; Table `tab:srope` (all 16 cells); 1.35×/1.54×; +0.139…+0.123 with MDEs; 11.38–11.51, 5.4, r −0.363; 6/8 at 1% → `RANK_TRUNCATION.md`, `SELECTIVE_ROPE.md`, `GATE_PROBE.md`, `POPE_WRAPPING.md`, `DRIFT_PROBE.md`, `LAMBDA_TRACE.md`
- App. H: +0.292 (0.063, 8/8), +0.016 (0.034), 0.003 vs 0.06–0.08, +0.191 (0.158); DoF torus four arms and contrasts; +0.165 (0.088, 21/24), +0.173 (0.127); search-start values; SPREAD 300-ep values → `N5_RESULTS.md`, `AUDIT_2026-09-10.md`, `DOF_RESULTS.md`, `D5_RESULTS.md`, `SEARCH_RESULTS.md`, `SPREAD_RESULTS.md`, `AP_KERNEL_DIAGNOSTIC.md`
- App. I → `NOISE_REFINE.md`, `LM200_ABLATION.md`, `LEVEL15_MEETS_GATED_*.md`, `CORRECTION_COMPOSITIONAL.md`
- App. G/loops → `LOOPED_PILOT.md` (0.3628/0.0177/0.0286 recomputed), `LOOP_SAMPLED.md`
- App. J: v4 language values (r=4 confirmed, mapformer.txt Fig. 7b). enwik8 1.3746/1.3758/1.3799/1.3740, −0.0058 t 3.49, MDE 0.0041, 0.0067 recomputed from `enwik8_long/*.json`; the `% src` cites a memory note and should cite these JSONs instead. Also `ENWIK8_HIERARCHY.md`, `FLIPFLOP_*.md`, `MQAR_RESULTS.md`.
- App. K, both tables: every delta, sd/se, MDE, seed count, verdict, and r −0.742/−0.853/−0.987 → `MONOTONE_RESULTS.md`

**Paper-version checks** (all OK against `papers/txt/mapformer.txt`, v4):
- separate $q_0/k_0$ in App. A.7;
- Fig. 9 in App. C.1;
- "two separate pools of neurons" in App. C.3;
- the 5D run used inner rank = world dimension (App. D.1);
- eq. 18;
- Table 6;
- language model r=4.

**Traps checked with no issue:** PAPERTASK is labelled exploratory in 6.3.2 (not at l.1766, O7); r=4 is scoped to MapWM (7.2 Scope); Level 1.5 is confined to OOD with no load-bearing component; "loop beats three layers" is not claimed on Match-Query; the aliasing account is stated as falsified, with map extent post hoc at n=3; reproduction is compared to v1.

---

## 5. Bibliography

Every entry in the local corpus matched its first page (arXiv ID, title, authors): mapformer, srope, grape, mamba3, vetcha (survey_infext), liu2025survey (2503.17407, Liu, Zhu, …), dape, sarrof, grazzi, carope, pope, liere, fox, hgrn, hgrn2, path, cope (2405.18719), rope, mamba, tale_two_algorithms. Problems:

1. **`whittington2022temt`: authors wrong.** The bib has "Whittington, Warrington, Joseph and Foley, Jonathan and Behrens". MapFormer v4 ref. [17] (mapformer.txt l.789) reads "James C.R. Whittington, Joseph Warren, and Timothy E.J. Behrens". The entry was copied from `paper/references.bib`. The paper is not in the local corpus, and the bib does not say so. The venue "ICLR 2022" is unverified; MapFormer cites it as arXiv 2022. Fix the authors, and add a not-in-corpus note or drop the venue.
2. **`whittington2020tem`** is not in the local corpus and has no note, unlike the other external entries. The volume and pages cannot be checked locally. MapFormer cites the 2019 bioRxiv version.
3. **Prior-art ordering, intro l.103–105.** "is due to \citet{puranik2026group} and GRAPE \citep{zhang2026grape}; \citet{vetcha2026infinite} states the same decomposition independently". GRAPE (arXiv Dec 2025) and Vetcha (3 Jan 2026) both predate Puranik's blog (22 Apr 2026). Per `papers/INDEX.md`, the Jordan-form polynomial terms are Puranik's addition. Suggest "due to GRAPE and, with the Jordan-form completion, Puranik; Vetcha states the decomposition independently".
4. **Missing identifiers** (the bib notes them as unverified, and they are not in the corpus). From memory, so verify before adding: Zoology 2312.04927; Minigrid & Miniworld 2306.13831; Habitat 1904.01201. `bae2025mor` is noted as unverified.
5. The key `gu2024mamba` has `year = {2023}`. This is cosmetic; the entry is correct.
6. No novelty claim contradicts the prior art. Checked:
   - taxonomy → GRAPE/Puranik/Vetcha;
   - content-dependent rotation → Mamba-3 Prop. 3, Selective RoPE;
   - sign → Sarrof/Grazzi/SRoPE §4.2 (verified that §4.2 contains the parity experiment);
   - contextual counting → CoPE;
   - content-aware category → 2503.17407 §3.1.1;
   - recursion → MoR.

   One contribution row is labelled "ours": "An unconstrained MapFormer counts through a content gate, by intervention" (l.164). It is framed as an observation within CoPE's mechanism, which is acceptable.
