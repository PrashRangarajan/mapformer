I couldn't write `REPORT.md` to the scratchpad because the harness refuses report files from subagents, so the full report is below. Everything was read-only; I ran no builds and made no changes to git.

**Top findings**
1. **The shared page cites a navigation number from an undertrained batch.** "+0.461, index model on a 0.506 floor" (`report/language_summary.html:281`, `axes_measured.tex` abstract and transfer table, `RESULTS_INDEX.md` headline) comes from a 16-epoch run. Under the converged recipe (`PAPER2X2_RESULTS.md`) the effect is +0.243 and RoPE reaches 0.805 on a fresh map. `report.tex:2127` already says this, so the documents disagree with each other.
2. **The one surviving Dyck result is out of distribution in depth.** Training is at L32 D4 (`DYCK_PREREG.md:34`); the quoted L32 D12 is three times the training depth. At the training cell the effect is +0.019, a ceiling. So the "train at the length you test at" rule rests on a depth-extrapolation result.
3. **The shared page contradicts the code results and itself.** It says "plain PoPE still wins on code" and "PoPE is a fairly general improvement" (`language_summary.html:234`, `:361`), but with the decay envelope on both, PoPE is detectably worse than RoPE on code (+0.0079 bpc), and it is worse on the torus index row. Line 387 ("PoPE helps only on its own dataset") contradicts lines 337-338 (detectable gains on Dyck). "Use MapPoPE ... never worse" leaves out the Bach collapse beyond the training context (4.616 against MapWM's 1.397).
4. **A novelty claim the repo itself retracts is in both abstracts.** Sign and rank are called axes "nobody / no paper varies deliberately" (`positional_review.tex:86-89`, `:152`, `:1112`; `axes_measured.tex:27-29`, `:946`). `mapformer_math.tex:1809-1831` already says "That is false". The corpus agrees: Grazzi deliberately varies the eigenvalue sign, Selective RoPE §4.2 carries sign into this exact cell, and MapFormer itself ablates r ∈ {1,2}. Separately, "Mamba-3 is the first design to claim both slots" is contradicted by Selective RoPE, which pairs its rotation with decay gates four months earlier.
5. **"Our r=2" is not the paper's r=2.** The paper's bottleneck is per head (`papers/txt/mapformer.txt:1512-1525`, W_in ∈ R^{d×nh×r}); the code shares one across heads (`model.py:61-62`). At 2 heads our r=2 has half the paper's latent dimensions, and this deviation is undocumented. Every "use r=4, not the paper's r=2" sentence may be recommending the paper's own capacity.
6. **Rank needs rewriting.** A rank-2 projection of trained r=4 models scores 0.995. From scratch, r=2 solves 0/8 against r=4's 8/8. Within r=2, skew does not predict accuracy. So "not capacity" now holds, "caused by a skewed basis" is unsupported, and the difference is whether training finds the solution. The math note's untested line "expressible but not reachable" (`:553`) turns out right.
7. **The sign result's accuracy evidence is extrapolation only.** At T=128 the monotone arms score 0.95-0.98 against index 0.80. So "costs everything on navigation" is wrong at training length. The sign ablation is also missing from the "never given a matched-length control" list. The clock/map crossover compares 8x extrapolation on the torus with 2x on recency, a confound fixable by an eval-only run at T=256.
8. **Withdrawn claims still asserted.** "Mutually exclusive" is left unscoped. "The ordering inverts" is stated, though recency shows no inversion and does not need a clock. The growth exponent is called a "design gap" and a cause, in passages the same files withdraw. EM's "no additive fallback" explanation rests on the withdrawn claim that MapWM is additive. `report.tex:2099` and `report_short.tex:525` state the withdrawn two-part Bach collapse mechanism as surviving. `README.md:279` says the InEKF wrap bounds θ̂, which is refuted.

---

# Full report

**VERIFIED** = checked against results files, the code or `papers/txt`. **JUDGED** = my reading.

## Which documents are current

| document | last edited | missing |
|---|---|---|
| `mapformer_math.tex`, `positional_review.tex` | 09-07 | the EM/WM audit (09-10), the PoPE/Dyck line, code, the Dyck ladder, rank at matched length |
| `axes_measured.tex`, `README.md`, `RESULTS_INDEX.md` | 09-11 | the PoPE/Dyck line, PAPER2X2's converged recipe, code, the ladder, rank at matched length. README's InEKF framing is older still |
| `report.tex`, `report_short.tex` | 09-20 | code, the PoPE ablation, the ladder, `INDIRECT_ARITHMETIC.md`, rank at matched length |
| `language_summary.html` (shared) | 09-23 | rank at matched length |

## Severity 1: misleading a reader

**S1.1 The navigation number (VERIFIED).**
- Cited at `language_summary.html:281`, `axes_measured.tex:30-32, 139, 145-150, 419, 445`, `RESULTS_INDEX.md:24-37`, the top of `CLAUDE.md`, and the robustness memory.
- Source is `BASELINE_TABLE.md:59`, a 16-epoch batch.
- `PAPER2X2_RESULTS.md` uses the recipe `axes_measured.tex:193` names as its default. It gives RoPE 0.805±0.012, position effect +0.243 (MDE 0.038, 8/8).
- "An index code cannot locate itself on a fresh map" is therefore false under that paper's own recipe.
- **Edit:** quote +0.243, RoPE 0.805, path-integrated 0.971.

**S1.2 Dyck is depth-OOD (VERIFIED).**
- Claimed as training-length or in-distribution at `language_summary.html:233, 278, 384`, `DYCK_LADDER_RESULTS.md:45`, and the top of `CLAUDE.md`.
- Training is at L32 D4. At that cell, 4 layers give +0.019 (ceiling).
- **Edit:** "matched length, 3x the training depth". Restate the frame as matched length, not matched distribution.

**S1.3 PoPE described as generally better (VERIFIED).**
- `language_summary.html:234`, `:361`.
- Against it:
  - `CODE_DECAY_RESULTS.md`: PoPE-Decay − RoPE-Decay = +0.0079 bpc, detectable against PoPE. RoPE-Decay is the best of eight code arms.
  - `ABLATE_RESULTS.md`: PoPE is +0.0034 worse on code.
  - PAPER2X2: PoPE-Flat 0.679 against RoPE 0.805 on the torus.
  - The page's own code row reads "PoPE ≈ RoPE".

**S1.4 The page contradicts itself on where PoPE helps (VERIFIED).**
- `:387` says PoPE helps only on its own dataset.
- `:337-338` highlight Dyck gains of +0.010 and +0.026, both 8/8, as detectable.

**S1.5 "Use MapPoPE ... never detectably worse" (VERIFIED).**
- `language_summary.html:359`, `:234`.
- Beyond the Bach training context, MapPoPE scores 4.616 against MapWM's 1.397, 0/5 seeds (`report.tex` Bach length table).
- **Edit:** scope it to the training length.

**S1.6 "Sign and rank dominate the choice of encoding" (VERIFIED).**
- `positional_review.tex:152-156`, `:1090`; `axes_measured.tex:946-948`; `RESULTS_INDEX.md:37`.
- Under the converged recipe the encoding effect is −0.049 at T=128, and +0.189 raw / +0.179 loss-matched at T=1024. That is larger than rank's +0.085.

## Severity 2: withdrawn or untested theory still asserted

**S2.1 "Mutually exclusive" is unscoped (VERIFIED).**
- `positional_review.tex:44-47, 86, 145, 511, 526`; `mapformer_math.tex:1933`.
- The 09-10 audit scoped it: a rank-r accumulator can cancel in one subspace and count in another.
- So "a system that needs both must spend both slots" also falls: `positional_review.tex:584, 1111`; `mapformer_math.tex:2047`; `axes_measured.tex:746`.

**S2.2 "The ordering inverts" / "a task that wants a clock" (VERIFIED).**
- `axes_measured.tex:37`, the crossover labels, `positional_review.tex:148, 577`.
- On recency the cost is −0.004, inside the MDE: nothing inverts. `mapformer_math.tex:2031` calls the prediction malformed.
- A signed rewind solves recency exactly (`AUDIT_2026-09-10.md` finding 2).

**S2.3 The crossover mixes extrapolation factors (VERIFIED).**
- Torus: −0.280 at 8x. Recency: −0.004 at 2x (`RECENCY_RESULTS.md:4, 48-49`).
- **Fix:** eval-only, run the torus sign arms at T=256.

**S2.4 The sign result is extrapolation-only in accuracy (VERIFIED, `SIGN_ABLATION.md:23-27, 49-58`).**
- At T=128, monotone arms score 0.946/0.977/0.900 against signed 1.000 and RoPE 0.799.
- Abs − RoPE is +0.146 raw. Signed − RoPE loss-matched is −0.021, detectable against path integration, so loss-matching at T=128 is uninformative.
- Needs scoping: "costs everything on navigation" (`positional_review.tex:1186`), "disqualifying" (`:147`), "beats it nowhere" (`axes_measured.tex:35, 208`; `RESULTS_INDEX`).
- **Missing experiment:** a sign test with both arms trained and tested at T=1024.

**S2.5 The growth exponent α (VERIFIED).**
- "That sets the rate of degradation" (`mapformer_math.tex:1686-1692`) contradicts `:1998-2004` in the same file.
- "Design gap" (`mapformer_math.tex:1950-1952`; `positional_review.tex:543-546`) contradicts its own no-op / breaks-additivity withdrawal.
- F3 "One quantity accounts for F1 and F2" (`axes_measured.tex:216`) contradicts that paper's own abstract.

**S2.6 The critical-dimension account is presented approvingly (VERIFIED).**
- `positional_review.tex:1007-1019` presents it as "worth importing"; the refutation appears only at `:1170`.
- `:1128` says "the published account is refuted here". What was refuted is only its transposition to 1-layer path-integrated models.

**S2.7 EM's "no additive fallback" mechanism (VERIFIED).**
- `positional_review.tex:832-838`; `mapformer_math.tex:783-792`.
- MapWM is not additive.
- The separate-q0/k0 effect changes sign by task: +0.128 for separate on recency at n=24, against Match-Query's +0.358 for single p0 at n=3.

**S2.8 The Bach collapse mechanism (VERIFIED).**
- `report.tex:2099-2101` ("What survives ... collapse needs both ...") and `report_short.tex:525-528` state the two-part account.
- CLAUDE.md lists that account as withdrawn.
- Non-negativity was never isolated on Bach; NoSigma on code found it irrelevant.

**S2.9 README wrap claim (VERIFIED).**
- `README.md:279-280` says the InEKF wrap keeps θ̂ bounded.
- Refuted in `ACCUMULATOR.md`: range 285.6 against 283.9.

**S2.10 Forget gate as a clock (VERIFIED).**
- "Every otherwise-puzzling detail follows" (`positional_review.tex:560`) and "explains" (`axes_measured.tex:926`).
- The account is untested, and its batch was deleted before analysis.

**S2.11 Rank "caused by conditioning / skewed basis" (VERIFIED).**
- Stated at `mapformer_math.tex:546-559, 1497-1502, 1861, 2061`; `positional_review.tex:1067`; `axes_measured.tex:211, 309-355, 914`; `RESULTS_INDEX.md:147-150`; `report.tex:59, 178, 998`; `report_short.tex:290`.
- Evidence:
  - Within r=2, skew does not predict accuracy (r = −0.35 / +0.30).
  - 94% of the +0.085 is short-gap revisits late in the sequence.
  - At matched length, r=4 solves 8/8 and r=2 0/8.
  - The rank-2 projection scores 0.995±0.008.
- Conclusion: the difference is findability. In `report.tex`, rank belongs under Claim 4 ("representable but harder to find"), not Claim 3 ("cancel cleanly").

## Severity 3: over-claims and contradictions

- **S3.1 Per-head vs shared bottleneck (VERIFIED, undocumented).** See top finding 5. **Fix:** add a per-head r=2 arm (640 params) to the matched-length batch.
- **S3.2 Wrong comparison number (VERIFIED).** `axes_measured.tex:930` says "0.416 for either alone". 0.416 is the arm with neither; the loop alone scores 0.833 and r=4 alone 0.421 (`MQ_RANK_2X2.md:7-10`).
- **S3.3 "Every run converged" (VERIFIED).** `language_summary.html:278`. The criterion was an arm-median slope (`DYCK_LADDER_RESULTS.md:3`).
- **S3.4 Dyck replication quoted on the discredited metric (VERIFIED).** `language_summary.html:289` gives +0.370 on F1. Hewitt closing accuracy gives +0.064 (0.638 vs 0.574); the direction replicates on both.
- **S3.5 Clock/map used predictively on the page that calls it refuted (JUDGED).** `:408` lists "clock-versus-map" as refuted, while `:352-361` recommend by it. Say which account died: the PoPE-decoupling corollary.
- **S3.6 "First clean null" (JUDGED).** `report.tex:2014`. MapFormer v4's BLiMP null and HGRN Table 11 come first. Drop "first".
- **S3.7 Stale report text (VERIFIED).** `report.tex` abstract (`:77-78`) and Limitations (`:2117`) say "no language result", yet Dyck, Bach, Indirect Indexing and enwik8 sections exist. The date line reads 13 Sep. The OOD-length open problem needs the code and rank updates.
- **S3.8 Placement misdescribed (VERIFIED).** `mapformer_math.tex:2406-2408` says Selective RoPE uses "that layer's own queries", contradicting the same file's `:305-309`. It also says θ is "shared across heads", but θ is per head (`model.py:62`); only the latent is shared.
- **S3.9 Hierarchy claim unscoped (VERIFIED).** `positional_review.tex:1072`, "hierarchy wins on compositional transfer". It is directional and underpowered (`HIER_RECHECK.md`).
- **S3.10 Survey details (VERIFIED).** `positional_review.tex:101` says four surveys, `:117` says five. `:107` credits 2503.17407 to "Zhu et al."; the first author is Jiaheng Liu, and `refs.bib` already uses `liu2025survey`. `mapformer_math.tex:1697-1704` cites only the two surveys that lack the category.
- **S3.11 `INDIRECT_ARITHMETIC.md` "fails by construction" (VERIFIED/JUDGED).** NoPE's Theorem 1 (`papers/txt/nope.txt:644`) shows position can reach the hidden state, and so the values, implicitly. MapPoPE solves the task in distribution (8/8). **Edit:** "cannot generalise across shifts through its positional operator".
- **S3.12 RESULTS_INDEX stale (VERIFIED).** `:159` says a MapPoPE r=4 batch "is running"; it landed. `:13-19` omits `axes_measured` and `report.pdf`.

## Priority / novelty

- "Nobody varies sign or rank deliberately": contradicted by Grazzi (`grazzi.txt:53-54, 104`), Selective RoPE §4.2 (`srope.txt:931-935`) and MapFormer's own r ablation (`mapformer.txt:391-393, 2391`). Use the math note's wording instead: not a design constraint in the language positional-encoding literature, and never tested on navigation.
- "What no one sweeps is the input-side bottleneck" (`positional_review.tex:803`): weak form of the same error, since MapFormer ablates two values. Say "above the world dimension".
- "Mamba-3 is the first design to claim both slots" (`mapformer_math.tex:1624`, `positional_review.tex:930`): Selective RoPE did it first (`srope.txt:19-22, 45, 113-114`).
- "Every paper in the companion review ... none runs grid navigation" (`axes_measured.tex:408`): MapFormer does. Say "every other paper".
- No "no survey covers this" regression was found.

## Open theory

**(a) Rank.**
- The corpus contains no optimisation-landscape literature (grepped for overparameterisation, landscape, Burer-Monteiro, lottery, balanced, saddle).
- Relevant external work, not read first-hand; add to `papers/` before citing:
  - Burer–Monteiro, and Boumal, Voroninski & Bandeira 2016: extra rank removes spurious minima. The cos/sin readout adds wrapped minima of its own.
  - Arora, Cohen & Hazan 2018: overparameterisation accelerates training. Du, Hu & Lee 2018: gradient flow preserves imbalance.
  - Saxe et al. 2014 and Jacot et al. 2021: small-init rank is acquired one direction at a time. Hypothesis: at r=2 one latent direction gets spent on the action-vs-observation distinction or a count, leaving no free direction for the second axis. That would explain the bimodal r=2 codes.
  - Lottery ticket and "rethinking pruning": a network that can hold the solution often cannot be trained to it from scratch.
  - Khodak et al. 2021 and Pufferfish: practical low-rank training.
- Tests, cheapest first:
  1. r=3.
  2. Per-head r=2.
  3. Snapshots of r=4's singular values during training: do the 3rd and 4th directions carry energy transiently?
  4. Balanced or orthogonal init at r=2.
  5. Eval-only: are the stalled r=2 codes wrapped?
  6. The warm-start trainable test already pre-registered in `RANK_PROJ_PREREG.md`.

**(b) Placement.**
- θ_t = ω·W_out·W_in·C_t, where C_t counts each token type in the prefix. So MapFormer's position is a linear function of the prefix's token counts, blind to anything attention retrieves.
- Indirect Indexing's target (source position + k) is outside that class. Angles recomputed per layer (Selective RoPE, Mamba-3) are what would supply it.
- Placement cannot show up at one layer, and nearly every experiment here is one layer. State that as a limitation.
- Run the even/odd-shift test at depth ≥2 with three arms: MapWM, a per-layer-angle arm, and NoPE.

**(c) Why OOD benefits die at matched length (JUDGED).**
- The frame's reason, "unseen accumulator values", cannot matter for any relative kernel here: all of them are invariant to shifting θ.
- What can differ past the training length:
  - intervals larger than trained;
  - more keys in the softmax (dilution);
  - distractor keys at never-trained intervals (collisions).
- Rank's 94% short-gap share points to dilution or collisions. The report attributes 53% of the MapEM length drop to collisions.
- Matched-length training shapes all three in every arm, which removes pure robustness effects: code encoding, Bach path integration (the full-context null supports this), probably the InEKF, forget gate, PoPE wrapping and sign.
- What survives is either capacity (PAPER2X2 index final loss 0.77; contextual counting) or search under a budget (rank). Dyck D12 fits, because the matching bracket sits at interval ~0 whatever the depth.
- Taxonomy: **capacity / search / robustness**.
- Tests:
  1. Eval-time log-length logit scaling on the stored T=128 checkpoints.
  2. Masking keys to a window of the trained span.
  3. A cross-arm collision rate.
  4. A sign test at matched length.

## Minimal edits per document

- **`language_summary.html`**: `:281` (S1.1); `:233, 278, 384` (S1.2); `:234, 361` (S1.3); `:387` (S1.4); `:359` (S1.5); `:278` (S3.3); `:289` (S3.4); `:408` vs `:352-361` (S3.5).
- **`axes_measured.tex`**: abstract `:27-47`; `:139, 145-150, 419, 445`; F2 at `:211, 309-355`; F3 at `:216`; crossover labels; `:408`, `:926`, `:930`, `:946-948`.
- **`positional_review.tex`**: `:44-47, 86-89, 145-156`; `:101-117`; `:511, 526, 543-546, 560, 577-584` (also the dangling "the companion paper." at `:580`); `:803, 832-838, 930`; `:1007-1019, 1128`; `:1067, 1072, 1090`; `:1111-1113`; `:1136-1140` (add existence 0.995 and 0/8 search); `:1186-1188`.
- **`mapformer_math.tex`**: `:546-559, 1497-1502` (add the new rank evidence and the per-head caveat); `:783-792`; `:1624`; `:1686-1692, 1950-1952`; `:1697-1704`; `:1933, 2047`; `:2406-2408`; plus one pointer paragraph to `AUDIT_2026-09-10.md` and the PoPE withdrawals.
- **`report.tex`**: `:59, 77-78, 178, 998`; §Rank `:1104-1171` (move rank to Claim 4, add the new evidence and per-head caveat); `:2014`; `:2099-2101`; Limitations and Conclusion (add sign to the OOD-only list; update the length problem).
- **`report_short.tex`**: `:290, 336-341, 525-528`; Limitations.
- **`README.md`**: add a banner pointing to `report.pdf` and `RESULTS_INDEX.md` and marking the InEKF framing superseded; strike `:279-280`.
- **`RESULTS_INDEX.md`**: `:24-37, 147-150, 159, 13-19`.
- **`INDIRECT_ARITHMETIC.md`**: S3.11.
- **`DYCK_LADDER_RESULTS.md:45`**: "in-distribution" becomes "matched length, 3x the training depth".

## Internal memory

- The top of `CLAUDE.md` and `project_robustness_vs_capability.md` carry:
  - the stale +0.461;
  - "Dyck in distribution";
  - "rank never had a matched-length control", stale since 09-24;
  - no sign ablation on the never-matched list;
  - the wrong "why" (absolute values).
- The 09-21 "UNCONTROLLED CONFOUND (rope base)" note was closed by E2 (`DYCK_DEPTH_RESULTS.md:23`) but not marked closed.
- `.claude-memory/project_clock_vs_map.md` repeats six paragraphs twice.