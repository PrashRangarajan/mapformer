# MapPoPE: what PoPE's polar score adds to a path-integrated phase (expansion note, 2026-10-20)

*CORRECTED 2026-10-21: the T = 1024 C / C0 strata are re-read on the per-stratum head (rank 4 wrap-only C0 0.97 under either score; the "not a sufficient statistic" caveat was an artefact), and action-key attention is a retrieval-head statement (`docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md` B2, M-MP1, M-MP2). The body below is unchanged.*

Labels: **ESTABLISHED** = registered result (results file named); **ANALYSIS** = new post hoc CPU analysis run for
this note, eval-only on stored checkpoints (script and output named); **THEORY** = derivation, with its assumptions and a
check against data; **SPECULATION** = untested.

## Summary

1. **THEORY.** Both rotary scores have the form `s_ij = sum_c A_c cos(dphi_c + beta_c)`. RoPE and MapWM let content set the
   amplitude A and the phase beta. PoPE lets content set only A (softplus) and fixes beta = delta_c, a content-free bias.
   In our runs delta_c is effectively 0. So for every content pair, every PoPE head's kernel peaks at zero phase
   difference: s_ij <= sum_c A_c, with equality iff dphi_c = 0 mod 2 pi on every weighted channel.
2. With a **path phase**, "zero phase difference" means the same place, which is exactly the revisit target. PoPE's
   score removes MapWM's content phase, and MapWM's content phase misaims its kernel at the target. ANALYSIS: the
   kernel's alignment on the target key (C, defined in the theory section) is 0.70 for MapWM and 0.995 for MapPoPE
   (p 0.0002, 8/8). Removing the content phase alone would lift MapWM by +0.20. MapPoPE also spends 4x less
   attention on action keys.
3. With an **index phase**, "zero phase difference" means the query's own token. The 1-layer torus solution needs kernel
   peaks at fixed past offsets (4j-1). RoPE builds those peaks with its content phase; PoPE cannot. ANALYSIS: delta sits
   in the clamp's dead zone (98-100% of entries), K(3) > K(1) holds in 0% of PoPE content pairs, and 88% of the
   PoPE - RoPE gap (-0.110 of -0.126) is on multi-step reversals. This sign flip gives the +0.154 interaction, and about
   80% of the interaction is the index side.
4. The score acts only on the readout term (C - C0). It never acts on the code term C0, which is how well the phase
   alone puts the same cell at zero. At T = 1024 and rank 2 (ANALYSIS), the score repairs short-gap revisits (C
   +0.34, p 0.0002; C0 unchanged). On wrap-only revisits C0 is 0.41-0.46 under both scores, and only rank 4 raises it.
   So no score rule can rescue rank 2 there.
5. Paper text: a short paragraph (Sec. 6 or App. L) and a 1-page appendix are proposed below. The single new GPU
   experiment is a delta-freedom 2x2 (index/path x delta pinned/free), about 2.5-3 GPU-hours. It predicts that the
   interaction reverses.

## Established facts (registered unless marked)

| fact | numbers | file |
|---|---|---|
| 2x2 at T = 128, r2 | RoPE 0.805, PoPE 0.679, MapWM 0.971, MapPoPE 0.999; interaction **+0.154** (MDE 0.076, 8/8). Encoding main effect -0.049. At r4 the interaction is +0.125, with both path arms at 1.000 | `PAPER2X2_RESULTS.md` |
| Beyond training length the encoding helps the index arm | PoPE 0.639 / 0.607 vs RoPE 0.470 / 0.374 at T = 512 / 1024 (+0.169 / +0.233); r2 encoding main effect +0.114 / +0.189. At r4 the interaction is -0.172 / -0.221 (OOD; robustness, rule 10) | `PAPER2X2_RESULTS.md` |
| T = 128 floors | blank 0.507, n-gram 0.598, **retrace 0.934** (wrap-only share 0). Both index arms sit below retrace | `docs/audits/2026-10-16/paper2x2_T128_floors_out.txt` |
| The gain is the score rule, not the frequency count | SCORE +0.0243 (p 0.012, 16/16 vs 10/16 SOLVED); COUNT +0.0002 (at ceiling, uninformative). CORRECTED: plausibly convergence speed; it shrank to +0.009 on fresh seeds | `MAPPOPE_PAIR_RESULTS.md`, `GAIN_GRAIN_RESULTS.md` |
| A peak-at-zero gain kernel is enough on a task with no headroom | GainScalar (one gain per token, delta 0) 20/20, the same as MapPoPE (descriptive: loss < 0.05 at epoch 23 vs 74 vs MapWM 132; confounded) | `GAIN_GRAIN_RESULTS.md` |
| No rank-2 rescue at T = 1024 | 3/12 vs 3/12 SOLVED. Secondaries: short gap +0.115 (p 0.036), gap >= 128 +0.043, wrap-only -0.015 | `SCORE_RANK_RESULTS.md` |
| The score does not touch the what-to-where leak | GainRaw keeps MapWM's leak. ONE defect shown (CORRECTED) | `GAIN_PHASE_RESULTS.md` |
| Prior probes (post hoc) | MapWM r2 observation-key fields peak at d = 0 for 7% of content pairs (mean shift 13.7 cells); MapPoPE for 100%. Same-cell coherence C vs C0 at T = 128 | `docs/audits/2026-10-05/remap_probe_out.txt`, `neuro_samecell_phase_out.txt` |
| Language / other | Bach MapPoPE - MapWM -0.0165 NLL (5/5); PoPE - RoPE -0.032. Code and Dyck MapPoPE rows are UNMEASURED or depth-OOD. PoPE faithful (1.7e-6); 80.7% of `pope_delta` in its clamp dead zone on the language runs | `JSB_RESULTS.md`, `.claude-memory/project_mappope_asymmetry.md`, `ABLATE_RESULTS.md` |

## Theory

**THEORY 1 (the polar decomposition of a rotary score).** Write channel c's query and key as complex numbers. RoPE /
MapWM score: `s_ij = sum_c Re(q_c conj(k_c) e^{i dphi_c})` with `dphi_c = phi_c(i) - phi_c(j)`, so A_c = |q_c||k_c| and
beta_c = arg q_c - arg k_c, both content-dependent. PoPE / MapPoPE (code: `model_pope.py`, `cos(dphi - delta)`):
`A_c = softplus(q_c) softplus(k_c)`, beta_c = -delta_c, content-free. The phase dphi is the WHERE half: path
integration changes where it comes from (`omega * cumsum(W_out W_in x)` instead of `t * theta`). The score map is the
HOW half: whether content may enter the phase. The two halves are orthogonal design choices. This was already noted in
`.claude-memory/reference_language_and_pope.md`; what follows makes it predictive.

**Proposition 1 (peak pinning).** If beta_c = 0 for all c, then for any content pair `s_ij <= sum_c A_c`, with equality
iff dphi_c = 0 mod 2 pi on every channel with A_c > 0. As a function of displacement, the kernel is a non-negative
combination of cosines, which is positive-definite (Bochner). Content can reweight the frequencies, which changes the
kernel's width and shape, but it cannot move or invert the peak. *Assumption:* delta = 0. **Check:** delta is in the
clamp's dead zone (>= 0, zero gradient) on 98-100% of entries in every PoPE-Flat run, with mean |delta| = 0.00 (ANALYSIS
A1). In MapPoPE C = C0 to 3 decimals (A1, A2), so delta does not shift the peak there either.

**Corollary 1a (path phase).** If the code is faithful (equal phase iff equal cell, within the head's resolved range),
the pinned peak is "attend to where I am", which is the revisit target. A RoPE-form score must instead learn beta_c = 0
for every (action query, observation key) pair. A residual beta_c displaces the field in space by beta_c / omega_c, and
the displacement is largest on the coarse channels. **Check (A1, T = 128, r2, n = 8):** target coherence is C = 0.703
for MapWM vs 0.995 for MapPoPE (p 0.0002, 8/8 seeds above). MapWM's content phase costs C0 - C = +0.20 (7/8 seeds).
Same-cell attention mass is 0.747 vs 0.925 (p 0.0008).

**Corollary 1b (key suppression must be by magnitude).** Under a pinned peak, an action token at the same cell has the
same phase as the observation there, so the score can only down-weight it through softplus(k). **Check:** mean
softplus(k) is 0.61-1.56 for action keys vs 0.91-1.80 for observation keys. Attention on action keys is 0.041 for
MapPoPE vs 0.179 for MapWM (p 0.0002). MapWM's amplitude-weighted cos(beta) is nearly the same for both key types
(+0.76 vs +0.75), so it barely separates them in phase.

**Corollary 1c (index phase).** With an index phase, dphi = 0 is the query's own token. PoPE's frequencies are
theta_c = 10000^(-c/64) <= 1 rad/token. Each cos(theta_c D) decreases on D in [0, pi], so with beta = 0 **every** content
pair's kernel strictly decreases over offsets 0..3. A pinned-peak index head cannot prefer the observation 3 tokens back
to the one 1 token back. Every fixed nonzero target offset must come from beta, which RoPE supplies through content and
PoPE through delta (stuck at 0). The 1-layer torus task needs exactly such offsets: during a run that reverses the
previous run, step j's target sits at token offset 4j-1 (the retrace rule, 0.934 at T = 128). **Check (A1):**
- K(3) > K(1) holds in 0% of PoPE-Flat (head, action, observation) pairs on every seed, and in 53-100% for RoPE.
- The argmax over D in 1..16 of the content-pair kernel is at D = 1 for 1068/1088 PoPE pairs. For RoPE it is at
  D = 3 (414), 7 (277), 10-11 (147) and 2 (168), a comb over the retrace offsets.
- PoPE puts 0.307 of its attention on the newest observation (never the target; every step moves), vs 0.002 for RoPE
  (p 0.0002).
- The accuracy gap decomposes over retrace strata as R1 -0.006, R2 -0.005, R3+ **-0.110**, non-retrace -0.004, out of
  -0.125. R3+ accuracy is 0.566 vs 0.779 (p 0.0002, 0/8 seeds overlap). The cost is entirely where the target is far
  back at a fixed offset; R1/R2, where the target is among the few most recent keys, are nearly intact.

**Reading of the +0.154 interaction (THEORY + ANALYSIS).** (MapPoPE - MapWM) - (PoPE - RoPE) = (+0.028) - (-0.126).
The same constraint (peak pinned at zero phase difference) is aligned with the target under a path phase and misaligned
under an index phase. About 80% of the interaction is the index-side loss; at r4 the path side is at ceiling and the
interaction is +0.125, all index side. A pinned recency kernel is also length-agnostic. That fits PoPE's better
extrapolation as an index code (0.607 vs RoPE 0.374 at T = 1024) and the PoPE paper's own remark that zero-init delta
is "important for length generalization". This is a consistency check, not a test.

**THEORY 2 (what a score rule can and cannot fix).** Any score of this family sees position only through dphi. Write
the target coherence as `C = C0 + (C - C0)`:
- C0 is the code term: the A-weighted cos(dphi) on the true target, i.e. whether the phase alone puts the same cell at 0.
- C - C0 is the readout term, which is beta's contribution.

A pinned-peak score sets the readout term to 0 by construction. It can also reweight channels (A >= 0), so it can
**select** channels whose code is consistent but cannot **create** them. Wrap-only revisits need the code to be
periodic on the torus, so they live in C0. **Check (A2, T = 1024, SCORE_RANK checkpoints, r2 n = 12, r4 n = 8):**
- Short-gap revisits: C rises from 0.617 (MapWM) to 0.960 (MapPoPE), p 0.0002. C0 does not move (0.900 vs 0.960,
  p 0.16). Accuracy is +0.114 (p 0.036, replicating the registered +0.115). The repair is entirely the readout term.
- Gap >= 128: C rises from 0.524 to 0.840 (p 0.029), but accuracy (+0.042, p 0.63) and target mass (-0.035) do not.
  C0 stays at 0.81-0.84 vs 0.99 at r4. The code has drifted, so many wrong-cell keys score as well as the target.
  Aiming the kernel exactly is not enough when the code cannot discriminate.
- Wrap-only: C0 is 0.457 (MapWM) vs 0.411 (MapPoPE), p 0.79, and accuracy is 0.641 vs 0.631. Rank 4 raises MapWM's
  C0 to 0.840 (p 0.03) and accuracy to 0.998.

The score rule cannot rescue rank 2 because the rank-2 failure is in C0 (the code's periodicity and drift), which is
outside any score's reach. Note: the score does not touch the code at T = 1024, but at T = 128 MapPoPE's C0 is also
higher (0.996 vs 0.905, p 0.0002). A pinned peak may make a good code easier to *find* on short walks, consistent with
GAIN_GRAIN's faster convergence; the effect does not carry to T = 1024 (SPECULATION as to mechanism).

**Scope and assumptions.** All checks are 1-layer, 2-head models on the 64x64 torus. In a 1-layer model Q and K depend
only on the current token, so content cannot encode a computed offset. Deeper models could carry position into the
amplitudes; PoPE's Indirect Indexing result (`INDIRECT_RESULTS_200k.md`, 7/8) is multi-layer and outside this account.
"delta = 0" is a property of the faithful zero-init plus clamp. With delta ~ U(-2 pi, 0) the peak is still
content-independent but no longer at zero, which is the lever of the proposed experiment.

## Analyses run (all post hoc, eval-only, CPU)

All are in `/home/prashr/mapformer/docs/paper/expansion/scripts/mappope/`.
- `kernel_probe.py`
  - Rebuilds layer 0 by hand; every run's logits equal the model's own to 0.0e+00 (`verify`).
  - Uses the same evaluation stream as `eval_noise_refine` / `eval_rank_strata` (held-out map seed 10000, np seed
    1234 + s). Per-seed accuracies reproduce `runs/paper2x2/eval.log` exactly, e.g. RoPE s0 0.7892 and
    PoPE-Flat s0 0.6244.
  - A1 `paper2x2` (T = 128, 100 walks per seed, seeds 0-7): `kernel_probe_paper2x2_out.txt`, `.json`, `.log`.
  - A2 `score_rank` (T = 1024, **40** walks per seed, a subsample of the registered 100): `kernel_probe_score_rank_out.txt`,
    `.json`, `.log`.
  - Readouts: accuracy by retrace stratum (R1/R2/R3+/none) and by SCORE_RANK stratum (gap < 128 / >= 128 / wrap-only);
    attention mass on same-cell observation keys, the retrace target, the offset-1 observation, action keys and other
    observations; target coherence C and C0 on the most recent same-cell observation key; attention by token offset;
    content-pair kernel checks (K(3) > K(1), argmax offset, delta dead-zone share, zero-displacement score for action vs
    observation keys).
- `probe_stats.py` -> `probe_stats_out.txt`: unpaired permutation tests (`stats_core.perm2_p`) on per-seed readouts, and
  the stratum decomposition of PoPE - RoPE.
- Caveats:
  - The masses and C use the head with the larger mean same-cell mass. In MapPoPE r4, wrap-only C0 is 0.55 while
    accuracy is 0.995, so another head or the FFN may carry wrap revisits there, and C0 is not a sufficient statistic
    for success.
  - A2 uses 40 walks per seed.
  - All p-values are post hoc and unregistered. They describe these checkpoints; they are not verdicts.

## Proposed paper text

**Main text, one paragraph (end of Sec. 6 or App. L lead):**

> *Where the angle comes from and how content uses it are separate choices.* Any rotary score can be written
> `sum_c A_c cos(dphi_c + beta_c)`: the phase difference dphi says where, and content sets an amplitude A and, in
> RoPE, a phase beta. Path integration changes dphi. PoPE's polar score (Gopalakrishnan et al. 2025) removes beta.
> Without beta, every head's kernel peaks at zero phase difference whatever the content (App. L). Under a path phase,
> zero phase difference is the same place, which is the revisit target. Under an index phase it is the current token.
> The two predictions have opposite signs, and so do the results. On the short-walk torus PoPE's score raises the
> path-integrated model (MapPoPE 0.999 vs MapWM 0.971). There it aligns the attention kernel with the target (alignment
> 0.995 vs 0.70) and removes the cost of the content phase. It lowers the index model (PoPE 0.679 vs RoPE 0.805), whose
> solution needs kernel peaks at fixed past offsets. Nearly the whole loss (0.110 of 0.126) is on revisits where the
> target lies more than ten tokens back. A score rule acts only on how the kernel reads the phase, never on the phase
> itself. On long walks it repairs short-gap revisits (+0.115) and leaves wrap-around revisits, and the rank-2 deficit,
> untouched (3/12 vs 3/12 SOLVED): the rank-2 phase does not place a wrapped cell at zero.

**Appendix L (replacing the current two lines), about one page:**
1. The decomposition and Proposition 1, with the proof (one line) and its assumption (delta = 0; the dead-zone
   measurement).
2. Table L1, T = 128 2x2, one row per arm. Columns: accuracy; R3+ accuracy; target coherence C and C0; attention on
   the newest observation; attention on action keys; K(3) > K(1) share. Numbers are from
   `kernel_probe_paper2x2_out.txt` / `probe_stats_out.txt`.
3. The stratum decomposition of PoPE - RoPE (-0.006 / -0.005 / -0.110 / -0.004).
4. Table L2, T = 1024. C, C0 and accuracy by stratum for MapWM / PoPE score at r2 and r4. The readout term repairs
   short gaps; wrap-only lives in C0.
5. The registered facts (MAPPOPE_PAIR SCORE vs COUNT, SCORE_RANK NO RESCUE, GAIN_GRAIN no-headroom), each with its
   CORRECTED qualifier.
6. Label every probe number "post hoc, eval-only". Say plainly that the interaction is ~80% index side, and that the
   path-side gain is small (+0.024, shrinking to +0.009 on fresh seeds) and at T = 128 plausibly convergence speed.

What NOT to claim: that the polar score fixes rank or wrap; that MapPoPE beats MapWM at long walks (unmeasured at
T = 1024, 3/12 vs 3/12); "PoPE hurts index models" in general (one layer, one task; PoPE helps index models on Bach and
beyond training length here); "two defects, two fixes" (withdrawn).

## Proposed experiments (GPU), ranked

1. **DELTA-FREEDOM 2x2 (the causal test of peak pinning).**
   - Design: phase {index, path} x delta {pinned: zero-init + clamp (current PoPE-Flat / MapPoPE-Flat); free: init
     U(-2 pi, 0), clamp removed}. Paper torus, T = 128, r2, 300 epochs cosine lr 1e-3, n = 8 fresh seeds, one batch
     (rule 12). Optional 5th arm: MapPoPE with delta frozen at U(-2 pi, 0), which removes the pin with no learning
     route back.
   - Primary readout (pre-register): the delta x phase interaction on T = 128 accuracy.
   - Predictions:
     - Free delta helps the index arm, concentrated in R3+ (a comb at offsets 4j-1 becomes reachable).
     - Free or frozen-random delta hurts or slows MapPoPE, since target coherence C falls below C0.
     - Secondary: free delta costs the index arm at T = 1024 (loss of the recency kernel).
   - Gates: delta dead-zone share and K(3) > K(1) share, eval-only with `kernel_probe.py`.
   - Cost: 4 arms x 8 seeds = 32 runs at about 38 min each (7.5 s/epoch at 4 jobs/GPU) is about 2.5-3 h on two GPUs,
     plus eval. The 5-arm version is about 3.5 h. A pilot on seeds outside the batch and a code-verification agent come
     first (rule 29).
   - Why first: it is the only test where the account predicts a sign reversal. It also turns the interaction from a
     description into a mechanism.
2. **Pinned-peak MapWM at T = 1024, r2.** Score `sum_c |q_c||k_c| cos(dphi_c)` (content phase removed, signed
   magnitude kept), plus MapWM and MapPoPE-Pair as twins, n = 12. Separates "beta removed" from "softplus magnitudes".
   Predicted: it matches MapPoPE on short gaps and does nothing on wrap-only. Cost about 36 runs at T = 1024 / 900 ep,
   which is substantial (SCORE_RANK-scale, about 1 day). Lower priority: GAIN_GRAIN already shows a pinned scalar gain
   suffices at T = 128.
3. **Eval-only (no GPU needed, can run now): polarise trained MapWM r4.** Replace beta by 0 at eval on the SCORE_RANK
   r4 checkpoints and read the accuracy change by stratum. `WHAT_WHERE_CHECKS` did the related "shared kernel x gain"
   rebuild at T = 128 (MapWM r2 0.971 -> 0.791). Expect small losses at r4 (C0 - C = 0.07) and large ones at r2.
   Diagnostic only, because the network was trained with beta.
