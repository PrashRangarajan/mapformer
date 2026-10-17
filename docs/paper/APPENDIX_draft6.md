# Appendices A-D and F-K (for draft 6)

*Draft 6, 2026-10-16. Main text: `docs/paper/DRAFT6_workshop.md`. Appendix E (theory) is the separate file
`docs/paper/APPENDIX_theory_v3.md`; its statements are cited here by their labels there (A0, B0, T1 = Lemma 1, P1-P7,
Remark R). Every number below is copied from the file named beside it; a CORRECTED or AUDIT block at the top of a results
file supersedes its body, and was read first. Status words: **registered** (a pre-registered decision or declared
secondary), **post hoc** (computed after the registered analysis; never changes a verdict), **descriptive** (a registered
arm with no decision attached), **exploratory** (not registered). Nothing here is new computation: the tables are copies
of committed analysis outputs, reformatted.*

Contents. A: models and recipes. B: floors and gates. C: per-seed tables and post hoc re-analyses. D: constructions.
(E: theory, separate file.) F: Dyck-2 at matched depth. G: navigation in words. H: faithful CARoPE and CoPE. I: the toy.
J: the clock side. K: depth.

---

## A. Models and recipes

### A1. Architecture (every navigation batch)

Source: `model.py` (`MapFormerWM`, `WMTransformerLayer`, `ActionToLieAlgebra`, `PathIntegrator`), `model_sign.py`,
`model_wrap_exist.py`, `model_mono_baselines.py`, `model_baseline_rope.py`, and the trainers named in A5.

| component | setting |
|---|---|
| layers | 1 (appendix K also 4 real layers and one block looped x4; appendix F 1-4 layers; ROBUST torus 2 layers, d_model 256, C13) |
| heads, width | 2 heads, d_model 128, d_head 64; FFN 4 x d_model with GELU; LayerNorm before attention and before the FFN; dropout 0.1 on attention weights, attention output and FFN output |
| rotary channels | 32 two-dimensional blocks per head, 64 channels; every query and key coordinate is rotated |
| score | channel c contributes q_c^T R(theta_{t',c} - theta_{t,c}) k_c (MapWM: content Q, K rotated by the path angle; not additive) |
| step map | Delta = W_out W_in x, W_in: 128 -> r, W_out: r -> 64, no bias, r = 4 (`_r4`), one latent shared across the heads (MapFormer's is per head) |
| frequencies | omega_c = 2 pi (1 / Delta_max)^(i / 31) for block i = 0..31, the same ladder in each head; Delta_max = the grid size (torus 64, ring 32, line 256); omega is a free parameter (its sign is not constrained) |
| vocabulary | torus: 4 moves + 17 observation tokens (16 types + blank); ring and line: 2 moves + 17 |
| parameters | ~204k for the one-layer torus MapWM (`FACTORIAL_RESULTS.md`) |

### A2. Step parameterisations and arm names

| name in the files | main-text name | step Delta (per channel, after W_out) | source |
|---|---|---|---|
| `Signed_r4` (= `Vanilla_r4`) | signed | W_out W_in x | `model.py` |
| `Abs_r4` | \|.\| | \|W_out W_in x\| | `model_sign.py` mode "abs" |
| `Pos_r4` | softplus | softplus(W_out W_in x); softplus(0) = 0.693, so every token starts at about the same step | `model_sign.py` mode "pos" |
| `SignedAO_r4`, `AbsAO_r4` (ring); `SignedAO4_r4`, `AbsAO4_r4` (torus) | observation-step-free (AO) signed / \|.\| | as above for the move tokens; every other token's step is 0 by construction | `model_wrap_exist.py` |
| `RoPE` | index RoPE | canonical ladder base^(-2c / d_head), base 10000 | `model_baseline_rope.py` |
| `CARoPEF` | CARoPE | appendix H | `model_mono_baselines.py` |
| `CoPEF` (and `_Hi`, `_Lo`, `_LoS` table starts) | CoPE | appendix H | `model_mono_baselines.py` |

The |.| constraint is applied after W_out, on each of the 64 channel steps, so every channel is non-negative; applying it
on the r-dimensional latent would leave W_out free to mix signs (`model_sign.py` docstring). It adds no parameters.

**Initial move angles (Table 1 arms; `train_wrap_mech.py`).** default: the seed's initialisation. *Same-direction start*
(`_absmag`): |default angles| in the balanced gauge (appendix D2) -- the start |.| imposes. *Large, sums below pi*
(`AO_mid`): a(+1), a(-1) iid U(0.9, 1.5) rad per channel. *Large, sums above pi* (`AO_high`, `SO_high`): iid U(1.7, 4.5)
rad. Every other weight is the seed's default initialisation, identical across arms at a seed. Torus large starts
(`train_torus_start.py`): the same distributions, iid per channel and move, balanced gauge (`TORUS_START_PREREG.md`). *Frozen move
latents* (`_fz`, ACCOUNTS T2): W_in fixed and the move embeddings detached in the step path (they still train through the
content path), so only W_out and omega move the angles.

### A3. Optimisation (all from `train.train`, the trainers' argparse defaults and the drivers)

| | torus (Secs. 4.1, 4.2, 4.4, 4.6; K) | ring and line (Secs. 4.1-4.6) |
|---|---|---|
| optimiser | AdamW, weight decay 0.05 (`train.train` default; ACCOUNTS T1 also 0), default betas | same |
| peak learning rate | 1e-3 | 1e-3 |
| schedule | per step: 5% linear warm-up, then cosine to 10% of the peak | same |
| batch | 16 walks of T = 1024 moves (2,048 tokens, interleaved move / observation) | 128 walks of T = 128 moves |
| batches per epoch, epochs | 98, 900 | 98, 300 (RING_BUDGET also 900) |
| loss | cross-entropy on revisited observation positions only | same |
| data | 3 parallel data workers (`--data-workers 3`) | same |
| attention kernel | SIGN_MATCHED: standard; GENERALITY, TORUS_START, FACTORIAL, ROBUST: `--fast-attn` (SDPA, TF32 off) | standard |
| recipe variants | ROBUST torus: 2 layers, d_model 256 (C13) | ROBUST ring: lr 3e-3 and 3.3e-4 (wd 0.05), and wd 0 (lr 1e-3) (C13) |
| training map | one per run (GridWorld seed = run seed); walks are runs of 1-10 repeated moves | redrawn for every walk (`environment_cancel`) |

**Cross-batch torus comparisons.** Sec. 4.1 sets SIGN_MATCHED (full `Signed_r4` / `Abs_r4`, standard attention) beside
GENERALITY and TORUS_START (observation-step-free `SignedAO4_r4` / `AbsAO4_r4`, `--fast-attn`). These differ in attention
kernel and model class: `--fast-attn` agrees with the standard path to ~1e-6 in logits but is not bit-identical in training
(dropout RNG). Only within-batch contrasts carry verdicts. ROBUST's torus cell shares GENERALITY's kernel and classes and
differs in depth, width and seeds, so "a quarter of the one-layer gap" (Sec. 4.1) is out of batch but not across kernels.

Continuations (appendix D5): ring 300 epochs at lr 1e-3, schedule re-warmed, fresh optimiser state, walk stream offset
10^6 (`train_wrap_exist.py`); torus 300 epochs at lr 1e-4 (primary) or 1e-3 (descriptive), `--data-seed-offset 1000000`
(`run_generality.sh`).

### A4. Evaluation

- **Torus** (`eval_noise_refine.evaluate`): one held-out map (GridWorld seed 10000), 100 walks per run with numpy seed
  1234 + run seed, revisit accuracy at T = 1024 (registered) and 2048 (out of distribution, no verdict), eval mode (no
  dropout). Revisit strata from `eval_rank_strata` (lag < 128, lag >= 128, wrap-only).
- **Ring and line** (`train_cancel.evaluate`): maps redrawn per walk (env seed 10000), 400 walks, numpy seed 10^6 for every
  run, accuracy at T = 128 (registered) and 4T (out of distribution).
- **Dropout re-score**: attention scaled by 1/(1 - p) at eval. A declared secondary in RING_DIAL, RING_BUDGET, WRAP_EXISTENCE,
  GENERALITY, MECHANISM and WRAP_COPE (no label changed; each results file states it), and applied to SIGN_MATCHED and
  TEXTWORLD by the 2026-10-04 audit (`docs/audits/2026-10-04/DROPOUT_RESCORE.md`; no verdict changed). ACCOUNTS,
  TORUS_START, GAUGE and ROBUST were not re-scored (ROBUST: the correction is invalid at 2 layers); FACTORIAL declared it
  for the 1-layer cells and does not report it.
- **Run classes** (`stats_core.classify_run`, per-epoch training loss): *SOLVED* if the mean of the final 5% of epochs is
  < 0.05; otherwise *STALLED* if the final-10% mean is within 5% of the preceding 10% mean, else *DESCENDING*.
- **Cancelling share** (Fig. 2, "wrap share"; draft 5's Fig. 1): among channels with |wrap(a(+1))| > 0.05 rad, the share with
  |wrap(a(+1) + a(-1))| < 0.1 rad (wrap-invariant). *Usage-weighted* (ACCOUNTS T1): channels weighted by query x key norm.
  *Opposite-sign share*: channels whose raw move angles have opposite signs. *Collapsed*: |a| <= 0.05 rad. The unweighted
  share excludes collapsed channels from its denominator, and unused channels dominate it (ACCOUNTS T1: 0.87 -> 0.18
  unweighted vs 0.99 -> 0.98 weighted on the same runs); Fig. 2 is redrawn usage-weighted for draft 6.

### A5. Batches

| batch (results file) | main text | task | arms | seeds | trainer / driver | independent check before results | pilot |
|---|---|---|---|---|---|---|---|
| SIGN_MATCHED | 4.1 | torus | Signed_r4, Abs_r4, Pos_r4, RoPE | 0-7 | `train_variant` / `run_sign_matched.sh` | no: authors' pre-launch gate (`docs/audits/2026-09-27/gate_sign_matched.py`); audit after the run (2026-09-30), which reproduced its probe (no file records it as independent) | none recorded |
| RING_DIAL | 4.1 | ring p_plus 0.5 / 0.75 / 0.9 | Signed_r4, Abs_r4 | 30-37 | `train_ring_dial` / `run_ring_dial.sh` | yes (Amendment 1) | seed 130 |
| RING_BUDGET | 4.1 | ring 0.5 | Abs_r4 at 300 and 900 epochs | 40-47 | `train_ring_dial` / `run_ring_budget.sh` | yes | none recorded |
| WRAP_EXISTENCE | 4.1, 4.2 | ring 0.5 | SO, AO, constructed arms | 600-607 | `train_wrap_exist` / `run_wrap_existence.sh` | yes | seed 690 |
| GENERALITY | 4.1, 4.2, 4.6 | torus; ring 0.5 | SO4, AO4, constructed; Signed_r4, CARoPEF, CoPEF, RoPE | 700-707 | `train_variant`, `train_ring_variant` / `run_generality.sh` | yes | seed 690 |
| MECHANISM | 4.1, 4.3 | ring 0.5 | Table 1 arms | 800-807 | `train_wrap_mech` / `run_wrap_mech.sh` | yes | none (CPU toy only) |
| WRAP_COPE | 4.6, H | ring 0.5 | CoPEF table starts | 900-907 | `train_ring_variant` / `run_wrap_cope.sh` | yes | seed 990 |
| ACCOUNTS | 4.2, 4.3, 4.6 | ring 0.5; line | T1 continuations (600-607), T2 frozen latents (1000-1007), T4 line (1100-1115) | see arms | `train_wrap_exist`, `train_wrap_mech`, `train_ring_variant` / `run_accounts.sh` | yes | none recorded. T1's wd 0.05 arms are bit-identical retrains of WRAP_EXISTENCE's AO_bal / SO_bal (state dicts and loss curves, 16/16; reproduction audit 2026-10-16): determinism, not replication |
| TORUS_START | 4.4 | torus | SO4, AO4, AO4_high, SO4_high, AO4_mid | 1200-1207 | `train_torus_start` / `run_torus_start.sh` | yes | none (CPU construction checks) |
| GAUGE | 4.5 | ring 0.5, moves +1/-1 or +1/+2 | SO, AO, RoPE | 1300-1307 | `train_gauge` / `run_gauge.sh` | yes | seed 1390 |
| ROBUST | 4.1, C13 | ring 0.5 (+1/-1; four recipes); torus 2 layers d 256 | SignedAO / AbsAO (ring), SignedAO4 / AbsAO4 (torus) | 1400-1407 | `train_gauge`, `train_variant` / `run_robust.sh` | yes (Amendment 1, `docs/audits/2026-10-16/robust_audit.md`) | seed 1490 (ring; torus timing 1490-1492) |
| CANCEL | B (index gate only) | ring p_plus 0.5 / 0.75 / 0.9 / 1.0 | path 1 layer; index 1-3 layers | 0-7 | `train_cancel` / -- | no | seeds 0-1 (reused; CORRECTED block) |
| FACTORIAL | K | torus | path 1 layer; index 1, 4 layers, looped x4 | 500-503 (n = 4) | `train_variant` / `run_factorial.sh` | yes | timing probe only (seed 400) |
| TEXTWORLD | G, K | torus in words | path 1 layer; index 1, 2 layers | 0-7 | `train_textworld` / `run_textworld.sh` | yes (Amendment 1), but seeds 0-1 are byte-identical to the pilot (CORRECTED block) | seeds 0-1 (reused) |
| DYCK_MDEPTH | F, K | Dyck-2 | see F | 0-7 | `train_dyck` / `run_dyck_mdepth.sh` | no: predates the independent checks | four seed-0 smoke runs (F) |
| CLOCK_SIDE, CLOCK_SPEED, CLOCK_SPEED_REP | J | k-back recency | Signed_r4, Abs_r4, CARoPE_r4 (CLOCK_SIDE also Pos_r4, RoPE) | 200-207; 210-225; 230-245 | `run_clock_side.sh`, `run_clock_speed*.sh` | yes | CLOCK_SIDE calibration (Amendment 1) |

The independent check is an automated code-verification agent: a separately launched, read-only AI code-review session
given the pre-registration and the code but no results (none exist at that point), which may run smoke tests on seeds
outside the batch and whose findings are written into the pre-registration as an amendment before launch (example:
`docs/audits/2026-10-16/robust_audit.md`). It is independent of the session that wrote the code, not of the project. The
main text uses CANCEL only through GENERALITY's checked index value (0.730); CANCEL's own numbers appear in B, labelled.

---

## B. Floors and gates

A floor is the accuracy of the best map-free predictor on the same kind of evaluation walk. Every value is from the file
named; rows marked post hoc were computed for this draft (2026-10-16); the rest are copied.

| task | predictor | value | computed on | source |
|---|---|---|---|---|
| torus T = 1024 | always predict blank | 0.507 (per stream seed 0.497-0.511) | same generator and protocol as A4 (held-out map 10000, numpy seed 1234 + s, 100 walks) for s = 10-25 | `docs/audits/2026-10-05/score_rank_floors_out.txt` |
| torus T = 1024 | best token n-gram, orders 0-5, fit on the training maps | 0.576 (0.568-0.582) | same | same |
| torus T = 1024 | retrace (copy the observation 2j steps back while a run reverses the previous run) | 0.843 (0.833-0.848) | same | same; `SCORE_RANK_PREREG.md` |
| torus T = 1024 | blank / n-gram / retrace on the SIGN_MATCHED and GENERALITY walks (post hoc) | 0.506 / 0.575 / 0.843 and 0.508 / 0.578 / 0.843 | each batch's own eval stream | `docs/audits/2026-10-16/sign_floors_out.txt`, `generality_floors_out.txt` |
| torus T = 1024, 2 layers d 256 (ROBUST) | blank / n-gram / retrace (post hoc) | 0.509 / 0.578 / 0.844; every run of both arms is above retrace (lowest \|.\| 0.910) | ROBUST's own eval stream, seeds 1400-1407 | `docs/audits/2026-10-16/robust_floors_out.txt` |
| torus T = 128 (MapFormer's length; PAPER2X2 eval walks) | blank / n-gram / retrace (post hoc) | 0.507 / 0.598 / 0.934 (retrace per seed 0.9277-0.9455; no wrap-only revisits at this length); index RoPE (0.805) is below retrace, path r = 2 (0.971) just above | PAPER2X2's eval stream, seeds 0-7 | `docs/audits/2026-10-16/paper2x2_T128_floors_out.txt` (`sign_floors.py` with `FLOOR_T=128`; n-gram still fit on T = 1024 walks of maps 10-25) |
| ring p_plus 0.5 / 0.75 / 0.9 | retrace (post hoc) | 0.860 / 0.758 / 0.612 | 2000 walks, numpy seed 4242 | `docs/audits/2026-10-16/ring_floors_out.txt` |
| torus T = 1024 | best constant, on each batch's own evaluation walks (identical across arms at a seed) | SIGN_MATCHED 0.501-0.512 per seed (mean 0.506); GENERALITY 0.504-0.518 (0.508); FACTORIAL 0.505-0.515 (0.509) | the batch's registered eval stream | `SIGN_MATCHED_STRATA.json`, `GENERALITY_TORUS_SO4_STRATA.json`, `FACTORIAL_L1_STRATA.json` ('all' stratum, field `floor`) |
| ring p_plus 0.5 / 0.75 / 0.9 | better of best constant and n-gram (orders 1-5) over preceding tokens | 0.554 / 0.544 / 0.520 | 1000 held-out walks (env seed 10000, numpy seed 1); n-gram fit on 3000 walks of map seed 0 | `CANCEL_PREREG.md`; `docs/audits/2026-09-27/gate_cancel.py` |
| ring p_plus 0.5 / 0.75 / 0.9 | fixed-lag copy (exactly one lap back) | 0.284 / 0.375 / 0.626 | same | same |
| ring p_plus 0.5 | constant predictor, per revisit stratum | 0.49-0.50 | the registered ring eval stream (400 walks, numpy seed 10^6) | `docs/paper/figs/FIGS_NOTES.md` |
| line (256 cells) | predict blank | 0.503 | ACCOUNTS T4 eval stream | `ACCOUNTS_RESULTS.md` |
| ring, moves +1/+2 | predict blank | ~0.50 | GAUGE eval stream | `GAUGE_RESULTS.md` |
| torus in words | best constant / reversal-copy rule | 0.505 / 0.597 | the batch's eval set | `TEXTWORLD_RESULTS.md` |
| Dyck-2 L32 D12 | best A2f floor (n-grams of order 1-6 fit on the D12 training distribution; stack-free) | 0.594 | gate G4 | `DYCK_MDEPTH_PREREG.md`, `DYCK_MDEPTH_RESULTS.md` |

**Own-stream torus floors (post hoc, 2026-10-16).** The n-gram and retrace floors were first computed on stream seeds
10-25 only. Recomputed on each batch's own evaluation walks with the same script (`docs/audits/2026-10-16/sign_floors.py`, a
copy of `score_rank_floors.py` with the stream seeds as a parameter; n-gram still fit on the 16 training maps 10-25):
SIGN_MATCHED (seeds 0-7) blank 0.506, n-gram 0.575, retrace 0.843 (per seed 0.837-0.849); GENERALITY (700-707) blank 0.508,
n-gram 0.578, retrace 0.843 (`sign_floors_out.txt`, `generality_floors_out.txt` there). Against its own seed's retrace
floor, SIGN_MATCHED |.| falls below on 3/8 runs (0.637, 0.690, 0.717), softplus on 4/8, index RoPE on 8/8, signed on 0/8;
GENERALITY's observation-step-free |.| (AO4, mean 0.880) is above the mean retrace floor. So the unsolved barrier models'
torus accuracy is, on average, at or below what a map-free local rule reaches; the gap to the signed model (0.998) and
SOLVED are the evidence that the map was not found.

**Ring retrace floor (post hoc, 2026-10-16).** The same retrace rule on the ring (`docs/audits/2026-10-16/ring_floors.py`,
2000 walks per p_plus, scored on revisits as `train_cancel.evaluate`): 0.860 / 0.758 / 0.612 at p_plus 0.5 / 0.75 / 0.9
(blank 0.497-0.499; share of revisits inside a retrace 0.72 / 0.52 / 0.23). It is a stronger map-free floor than the
registered 0.554 / 0.544 / 0.520 at p_plus 0.5 and 0.75. The ring |.| models (0.95-0.97 at p_plus 0.5) are above it.
For moves +1/+2 (GAUGE) no move reverses another and retrace reduces to predicting blank (~0.50).

**Torus at T = 128 (post hoc, 2026-10-16).** The draft cites no T = 128 torus number; the row is for readers who know
MapFormer's T = 128 results. At that length almost every revisit is local, and the map-free retrace rule scores 0.934, so a
T = 128 torus accuracy below ~0.93 is not evidence of a map (PAPER2X2's converged index RoPE, 0.805, is below it; the
path model's 0.971 is above it by 0.04).

**Gates.** Each task was gated before any GPU run (rule 11): the floors above call the task code. Whether a task needs a map
at all is read from the in-batch one-layer index RoPE model, which cannot integrate content-dependent steps: torus 0.722-0.731
(GENERALITY, SIGN_MATCHED), ring p_plus 0.5 / 0.75 / 0.9 0.710 / 0.677 / 0.824 on CANCEL's six fresh seeds (0.717 / 0.676 /
0.825 with its two pilot seeds; CANCEL had no independent pre-launch check; `CANCEL_RESULTS.md` CORRECTED block) and 0.730
(GENERALITY, the value the main text uses),
line 0.772 (ACCOUNTS T4), moves +1/+2 0.649 (GAUGE). Each is above its blank and n-gram floors and below the signed path
model (1.000 or 0.998); on the torus and on the ring at p_plus 0.5 / 0.75 it is below the retrace floor.

---

## C. Per-seed tables and post hoc re-analyses

Accuracy is at the training length (torus T = 1024 on the held-out map; ring T = 128), eval mode. S / SOLV = SOLVED,
STAL = STALLED, DESC = DESCENDING; a number in brackets is the final-5% training loss.

### C1. Torus sign batch (Sec. 4.1; `SIGN_MATCHED.json`, `SIGN_MATCHED_ANALYSIS.txt`)

| seed | signed | \|.\| | softplus | index RoPE |
|---|---|---|---|---|
| 0 | 1.000 SOLV (0.003) | 0.931 DESC (0.201) | 0.590 STAL (1.250) | 0.730 STAL (0.983) |
| 1 | 0.997 SOLV (0.003) | 0.917 DESC (0.314) | 0.527 STAL (1.510) | 0.729 STAL (0.939) |
| 2 | 1.000 SOLV (0.005) | 0.868 STAL (0.492) | 0.630 STAL (1.084) | 0.720 STAL (0.931) |
| 3 | 1.000 SOLV (0.008) | 0.637 STAL (1.043) | 0.972 DESC (0.158) | 0.739 STAL (0.910) |
| 4 | 1.000 SOLV (0.004) | 0.690 DESC (0.962) | 0.674 STAL (1.009) | 0.737 STAL (0.928) |
| 5 | 0.990 SOLV (0.003) | 0.888 STAL (0.446) | 0.935 DESC (0.295) | 0.744 STAL (0.885) |
| 6 | 1.000 SOLV (0.004) | 0.922 STAL (0.349) | 0.922 DESC (0.345) | 0.722 STAL (0.956) |
| 7 | 1.000 SOLV (0.004) | 0.717 STAL (0.834) | 0.995 SOLV (0.049) | 0.728 STAL (0.976) |
| mean | 0.998 | 0.821 | 0.781 | 0.731 |

*Table C1.* Registered: |.| - signed -0.177 (MDE ~0.14, permutation p 0.0002; SOLVED 0/8 vs 8/8, Fisher p 0.0002);
softplus - signed -0.218 (p 0.0003; 1/8 vs 8/8, p 0.0014). r(final loss, accuracy) -0.992 over 32 runs. Seed 5's signed
run (0.9896) is the one below 0.99 under the accuracy criterion of C10.

### C2. Ring dial (Sec. 4.1; `RING_DIAL_VERDICTS.json`)

| seed | 0.5 signed | 0.5 \|.\| | 0.75 signed | 0.75 \|.\| | 0.9 signed | 0.9 \|.\| |
|---|---|---|---|---|---|---|
| 30 | 1.000 S | 1.000 S | 1.000 S | 0.994 S | 1.000 S | 1.000 S |
| 31 | 1.000 S | 0.977 | 1.000 S | 0.958 | 1.000 S | 0.999 S |
| 32 | 1.000 S | 0.987 | 1.000 S | 0.957 | 1.000 S | 0.885 |
| 33 | 1.000 S | 0.964 | 1.000 S | 0.954 | 1.000 S | 1.000 S |
| 34 | 1.000 S | 1.000 S | 1.000 S | 0.964 | 1.000 S | 1.000 S |
| 35 | 1.000 S | 0.944 | 1.000 S | 0.967 | 1.000 S | 0.825 |
| 36 | 1.000 S | 0.977 | 1.000 S | 0.993 | 1.000 S | 0.960 |
| 37 | 1.000 S | 0.931 | 1.000 S | 0.958 | 1.000 S | 0.964 |

*Table C2.* Columns are p_plus and arm. Registered (`RING_DIAL_ANALYSIS.txt`): p_plus 0.5 d -0.0275 (perm p 0.0070, 95% CI
[-0.046, -0.009]), SOLVED 2/8 vs 8/8 (Fisher p 0.0070); 0.75 d -0.0320 (p 0.0002), 1/8 vs 8/8 (p 0.0014); 0.9 UNMEASURED,
d -0.0459 (p 0.0256, below the 0.05 materiality floor), 4/8 vs 8/8 (p 0.0769).

### C3. Three times the budget (Sec. 4.1; `RING_BUDGET_ANALYSIS.txt`; |.|, ring p_plus 0.5)

| seed | 900 epochs | 300 epochs |
|---|---|---|
| 40 | 1.000 SOLV (0.002) | 0.957 STAL (0.150) |
| 41 | 0.986 STAL (0.070) | 0.982 STAL (0.098) |
| 42 | 0.969 DESC (0.129) | 0.946 STAL (0.252) |
| 43 | 1.000 SOLV (0.002) | 1.000 SOLV (0.003) |
| 44 | 0.951 DESC (0.215) | 0.895 DESC (0.301) |
| 45 | 0.944 STAL (0.195) | 0.998 SOLV (0.012) |
| 46 | 1.000 SOLV (0.000) | 1.000 SOLV (0.008) |
| 47 | 0.989 DESC (0.056) | 0.960 DESC (0.197) |

*Table C3.* Registered PARTIAL: 3/8 vs 3/8 (Fisher p 1.00); the solved sets differ. No in-batch signed arm (the signed
comparison, 8/8 at 300 epochs, is across batches and descriptive).

### C4. Observation-step-free ring, from scratch and constructed (Secs. 4.1, 4.2; `WRAP_EXISTENCE_ANALYSIS.txt`)

| seed | SO (signed) | AO (\|.\|) | AO_bal: frozen -> trained (max continuation loss) | SO_bal: frozen -> trained (max loss) |
|---|---|---|---|---|
| 600 | 1.000 S | 0.905 STAL | 1.000 -> 1.000 S (0.041) | 1.000 -> 1.000 S (0.019) |
| 601 | 1.000 S | 0.949 STAL | 1.000 -> 1.000 S (0.030) | 1.000 -> 1.000 S (0.042) |
| 602 | 1.000 S | 0.853 STAL | 1.000 -> 1.000 S (0.154) | 1.000 -> 1.000 S (0.072) |
| 603 | 1.000 S | 0.951 STAL | 1.000 -> 1.000 S (0.046) | 1.000 -> 1.000 S (0.017) |
| 604 | 1.000 S | 0.907 STAL | 1.000 -> 1.000 S (0.024) | 1.000 -> 1.000 S (0.039) |
| 605 | 1.000 S | 0.930 STAL | 1.000 -> 1.000 S (0.032) | 1.000 -> 1.000 S (0.028) |
| 606 | 1.000 S | 0.950 STAL | 1.000 -> 1.000 S (0.031) | 1.000 -> 1.000 S (0.112) |
| 607 | 1.000 S | 0.922 STAL | 1.000 -> 1.000 S (0.036) | 1.000 -> 1.000 S (0.015) |

*Table C4.* Registered: E3 SIGNED LEARNS BETTER (0.921 vs 1.000, d -0.079, perm p 0.0002; 0/8 vs 8/8, Fisher p 0.0002);
E2 HELD (AO_bal 8/8 SOLVED, 0/8 LOST). The ill-conditioned constructions are in D3.

### C5. The start manipulation (Sec. 4.3, Table 1; `MECHANISM_ANALYSIS.txt`)

| seed | SO | AO | SO_same | SO_absmag | AO_absmag | AO_high | SO_high | AO_mid |
|---|---|---|---|---|---|---|---|---|
| 800 | 1.000 S | 0.907 STAL | 1.000 S | 1.000 S | 0.865 STAL | 1.000 S | 1.000 S | 0.968 S |
| 801 | 1.000 S | 0.894 STAL | 1.000 S | 1.000 S | 0.922 STAL | 1.000 S | 1.000 S | 1.000 S |
| 802 | 1.000 S | 0.910 STAL | 1.000 S | 1.000 S | 0.861 STAL | 1.000 S | 1.000 S | 0.917 STAL |
| 803 | 1.000 S | 0.966 STAL | 1.000 S | 1.000 S | 0.860 STAL | 1.000 S | 1.000 S | 0.921 STAL |
| 804 | 1.000 S | 0.987 STAL | 1.000 S | 0.866 STAL | 0.933 STAL | 1.000 S | 1.000 S | 0.976 DESC |
| 805 | 1.000 S | 0.888 STAL | 1.000 S | 1.000 S | 0.857 STAL | 1.000 S | 1.000 S | 0.890 STAL |
| 806 | 1.000 S | 0.880 STAL | 1.000 S | 1.000 S | 0.889 STAL | 1.000 S | 1.000 S | 0.928 STAL |
| 807 | 1.000 S | 0.925 STAL | 1.000 S | 1.000 S | 0.945 STAL | 1.000 S | 1.000 S | 0.951 STAL |
| mean | 1.000 | 0.919 | 1.000 | 0.983 | 0.892 | 1.000 | 1.000 | 0.944 |

*Table C5.* SO_same = the default angles re-set in the balanced gauge (gauge control); SO_absmag / AO_absmag = the
same-direction start; AO_mid = large start with sums below pi; AO_high / SO_high = large start with sums above pi.
Registered (all n = 8): M5 AO_absmag vs SO_absmag SIGN COST AT THE |.| START (d -0.092 [-0.135, -0.045], perm p 0.0030;
Fisher p 0.0014); M2 AO_high vs AO_absmag RESCUES (+0.109 [+0.085, +0.135], p 0.0002); M2b AO_high vs AO_mid (+0.056
[+0.030, +0.085], p 0.0014; Fisher p 0.0070); M6 AO_high vs SO_high NO EFFECT DETECTED; M1 SO_absmag vs SO_same
UNMEASURED (-0.017); M3 gauge controls NO DIFFERENCE DETECTED; M4 AO vs SO SIGNED LEARNS BETTER (d -0.081, p 0.0002).
Declared secondaries, end-of-training cancelling share: SO 0.86, SO_same 0.96, SO_absmag 0.89, AO 0.07, AO_absmag 0.04,
AO_mid 0.05, AO_high 0.23, SO_high 0.34.

### C6. Frozen move latents (Sec. 4.3; ACCOUNTS T2, `ACCOUNTS_ANALYSIS.txt`, seeds 1000-1007)

| arm | start | move latents | mean accuracy | SOLVED | opposite-sign channels at the end, per seed |
|---|---|---|---|---|---|
| SO_same | default (balanced gauge) | free | 1.000 | 8/8 | 1.00 1.00 0.98 0.98 1.00 1.00 1.00 1.00 |
| SO_same_fz | same | frozen | 0.991 | 7/8 | 0.51 0.73 0.22 0.38 0.45 0.77 0.68 0.54 |
| SO_absmag | same-direction | free | 0.985 | 7/8 | 1.00 1.00 1.00 0.41 1.00 1.00 0.19 1.00 |
| SO_absmag_fz | same-direction | frozen | 0.910 | 1/8 | 0.21 0.15 0.09 0.18 0.14 0.16 0.04 0.21 |

*Table C6.* Registered: F1 WORSE (1/8 vs 7/8, Fisher p 0.010; -0.075 [-0.130, -0.020], perm p 0.009); F2 NO EFFECT
DETECTED; reading SUPPORTED. Per-seed accuracies are not printed in the analysis output; the run-level SOLVED counts are
in C10.

### C7. Torus, observation-step-free, from scratch and constructed (Secs. 4.1, 4.2; `GENERALITY_ANALYSIS.txt`)

| seed | SO4 | AO4 | SO_bal (lr 1e-4) | AO_bal (lr 1e-4): peak loss, epoch of peak / of re-entry | AO_bal_hi (lr 1e-3) |
|---|---|---|---|---|---|
| 700 | 1.000 S | 0.874 STAL | 1.000 S | 1.000 S; 0.168, 29 / 153 | 1.000 S |
| 701 | 1.000 S | 0.707 STAL | 1.000 S | 1.000 S; 0.110, 40 / 121 | 1.000 S |
| 702 | 1.000 S | 0.946 DESC | 1.000 S | 1.000 S; 0.173, 19 / 204 | 0.999 S |
| 703 | 1.000 S | 0.879 STAL | 1.000 S | 1.000 S; 0.143, 52 / 149 | 0.950 STAL |
| 704 | 1.000 S | 0.892 STAL | 1.000 S | 1.000 S; 0.131, 13 / 119 | 0.960 DESC |
| 705 | 1.000 S | 0.907 STAL | 1.000 S | 1.000 S; 0.155, 13 / 148 | 0.992 S |
| 706 | 1.000 S | 0.877 STAL | 1.000 S | 1.000 S; 0.208, 36 / 164 | 1.000 S |
| 707 | 1.000 S | 0.960 DESC | 1.000 S | 1.000 S; 0.117, 23 / 139 | 1.000 S |

*Table C7.* SO_bal's peak continuation losses 0.004-0.017. Registered: E3 SIGNED LEARNS BETTER (0.880 vs 1.000, d -0.120,
perm p 0.0002; 0/8 vs 8/8); E2 NOT HELD by the LOST rule (6/8 LOST vs 0/8, Fisher on not-LOST p 0.0070), read by the rule
fixed before launch as "perturbed and re-found" (8/8 SOLVED at the end). AO_bal_hi (descriptive): 6/8 SOLVED, 0.988.

### C8. Torus start manipulation (Sec. 4.4; `TORUS_START_ANALYSIS.txt`)

| seed | SO4 | AO4 | AO4_high | SO4_high | AO4_mid |
|---|---|---|---|---|---|
| 1200 | 1.000 S | 0.801 STAL | 0.934 DESC | 0.922 DESC | 0.937 DESC |
| 1201 | 1.000 S | 0.838 STAL | 0.794 STAL | 1.000 S | 0.908 STAL |
| 1202 | 1.000 S | 0.998 S | 0.975 DESC | 0.926 DESC | 0.824 STAL |
| 1203 | 1.000 S | 0.899 STAL | 0.931 DESC | 0.905 DESC | 0.906 DESC |
| 1204 | 1.000 S | 0.948 STAL | 0.951 DESC | 0.745 STAL | 0.840 STAL |
| 1205 | 1.000 S | 0.875 STAL | 0.876 STAL | 0.927 DESC | 0.949 DESC |
| 1206 | 1.000 S | 0.882 STAL | 0.915 DESC | 1.000 S | 0.939 DESC |
| 1207 | 1.000 S | 0.927 STAL | 0.828 STAL | 0.928 DESC | 0.990 DESC |
| mean | 1.000 | 0.896 | 0.900 | 0.919 | 0.912 |

*Table C8.* Registered: TS1 NO EFFECT DETECTED (+0.005 [-0.060, +0.070], p 0.89); TS2 UNMEASURED; TS3 UNREADABLE (SO4_high
2/8 < 4); TS4 SIGNED LEARNS BETTER (1/8 vs 8/8, Fisher p 0.0014; -0.104 [-0.150, -0.060]). Strata (lag < 128 / >= 128 /
wrap-only): SO4 1.000 / 1.000 / 1.000; AO4 0.939 / 0.730 / 0.632; AO4_high 0.949 / 0.777 / 0.534; SO4_high 0.970 / 0.716
/ 0.627; AO4_mid 0.950 / 0.809 / 0.621. An untested account of why the past-pi start does not transfer (moved from the main
text of draft 5): with 64 cells per axis and two inverse pairs per channel, a random large-angle code is far from any map.
The start lever is a ring result, not a design rule.

### C9. Moves +1/-1 vs +1/+2 (Sec. 4.5; `GAUGE_RESULTS.md`)

| seed | SO_m | AO_m | SO_p | AO_p | RoPE_p |
|---|---|---|---|---|---|
| 1300 | 1.000 S | 0.904 STAL | 0.987 STAL | 0.987 DESC | 0.642 STAL |
| 1301 | 1.000 S | 0.953 DESC | 1.000 S | 0.910 STAL | 0.663 STAL |
| 1302 | 1.000 S | 0.957 STAL | 1.000 S | 0.904 STAL | 0.655 STAL |
| 1303 | 1.000 S | 1.000 S | 1.000 S | 0.590 STAL | 0.661 STAL |
| 1304 | 1.000 S | 0.922 STAL | 0.935 STAL | 0.768 STAL | 0.654 STAL |
| 1305 | 1.000 S | 0.928 STAL | 1.000 S | 1.000 S | 0.602 STAL |
| 1306 | 1.000 S | 0.880 STAL | 0.654 DESC | 0.908 STAL | 0.660 STAL |
| 1307 | 1.000 S | 0.903 STAL | 1.000 S | 1.000 S | 0.660 STAL |

*Table C9.* _m = moves +1/-1, _p = +1/+2. Declared secondaries (revisit strata): on +1/-1, 89% of revisits are
backtracks (138,800 of 155,456 on the eval stream; 87.5% in the pre-launch audit's sample; `GAUGE_RESULTS.md` carries a
CORRECTED banner); on +1/+2 every revisit is a lap with gap >= 16.

| cell | mean accuracy | SOLVED | accuracy 4T | loss AUC | backtrack | lap | gap < 16 | gap >= 16 |
|---|---|---|---|---|---|---|---|---|
| SO_m | 1.0000 | 8/8 | 0.997 | 0.090 | 1.000 | 1.000 | 1.000 | 1.000 |
| AO_m | 0.9309 | 1/8 | 0.871 | 0.537 | 0.936 | 0.893 | 0.952 | 0.902 |
| SO_p | 0.9470 | 5/8 | 0.913 | 0.515 | -- | 0.947 | -- | 0.947 |
| AO_p | 0.8834 | 2/8 | 0.863 | 0.661 | -- | 0.884 | -- | 0.884 |
| RoPE_p | 0.6494 | 0/8 | 0.468 | 1.341 | -- | 0.651 | -- | 0.651 |

*Table C9b.* The laps comparison quoted in Sec. 4.5 (|.| 0.893 vs 0.884) is a post hoc reading of these declared
secondaries. Signed models on +1/+2 end with 0.92 of channels having both (wrapped) move angles of one sign (per seed
0.57-1.00), against 0.01 on +1/-1.

### C10. SOLVED under four criteria (post hoc; `docs/audits/2026-10-15/reanalyses/solved_thresholds_out.md`)

Predictions and decision rules were written before the run (`PREDICTIONS.md` there); all 47 arms reproduce their
registered SOLVED counts at the registered 0.05.

| batch | contrast | loss < 0.02 | loss < 0.05 (registered) | loss < 0.1 | accuracy >= 0.99 |
|---|---|---|---|---|---|
| SIGN_MATCHED torus | \|.\| vs signed | 0/8 vs 8/8 (p 0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 7/8 (0.0014) |
| SIGN_MATCHED torus | softplus vs signed | 0/8 vs 8/8 (0.0002) | 1/8 vs 8/8 (0.0014) | 1/8 vs 8/8 (0.0014) | 1/8 vs 7/8 (0.0101) |
| RING_DIAL p 0.5 | \|.\| vs signed | 2/8 vs 8/8 (0.0070) | 2/8 vs 8/8 (0.0070) | **4/8 vs 8/8 (0.0769)** | 2/8 vs 8/8 (0.0070) |
| RING_DIAL p 0.75 | \|.\| vs signed | 1/8 vs 8/8 (0.0014) | 1/8 vs 8/8 (0.0014) | 2/8 vs 8/8 (0.0070) | 2/8 vs 8/8 (0.0070) |
| RING_DIAL p 0.9 | \|.\| vs signed | 4/8 vs 8/8 (0.0769) | 4/8 vs 8/8 (0.0769) | 4/8 vs 8/8 (0.0769) | 4/8 vs 8/8 (0.0769) |
| WRAP_EXISTENCE ring | AO vs SO | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) |
| MECHANISM ring | AO vs SO | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 1/8 vs 8/8 (0.0014) | 0/8 vs 8/8 (0.0002) |
| MECHANISM ring | AO_absmag vs SO_absmag | 0/8 vs 7/8 (0.0014) | 0/8 vs 7/8 (0.0014) | 0/8 vs 7/8 (0.0014) | 0/8 vs 7/8 (0.0014) |
| MECHANISM ring | AO_high vs AO_absmag | 8/8 vs 0/8 (0.0002) | 8/8 vs 0/8 (0.0002) | 8/8 vs 0/8 (0.0002) | 8/8 vs 0/8 (0.0002) |
| MECHANISM ring | AO_high vs AO_mid | 8/8 vs 1/8 (0.0014) | 8/8 vs 2/8 (0.0070) | 8/8 vs 3/8 (0.0256) | 8/8 vs 1/8 (0.0014) |
| MECHANISM ring | AO_high vs SO_high | 8/8 vs 8/8 (1.00) | 8/8 vs 8/8 (1.00) | 8/8 vs 8/8 (1.00) | 8/8 vs 8/8 (1.00) |
| ACCOUNTS T2 ring | SO_absmag_fz vs SO_absmag | 1/8 vs 7/8 (0.0101) | 1/8 vs 7/8 (0.0101) | 1/8 vs 7/8 (0.0101) | 1/8 vs 7/8 (0.0101) |
| ACCOUNTS T2 ring | SO_same_fz vs SO_same | 7/8 vs 8/8 (1.00) | 7/8 vs 8/8 (1.00) | 7/8 vs 8/8 (1.00) | 7/8 vs 8/8 (1.00) |
| GENERALITY torus | AO4 vs SO4 | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) |
| GENERALITY ring | CoPE vs signed | 0/8 vs 8/8 (0.0002) | 0/8 vs 8/8 (0.0002) | 1/8 vs 8/8 (0.0014) | 0/8 vs 8/8 (0.0002) |
| TORUS_START torus | AO4 vs SO4 | 0/8 vs 8/8 (0.0002) | 1/8 vs 8/8 (0.0014) | 1/8 vs 8/8 (0.0014) | 1/8 vs 8/8 (0.0014) |
| TORUS_START torus | AO4_high vs AO4 | 0/8 vs 0/8 (1.00) | 0/8 vs 1/8 (1.00) | 0/8 vs 1/8 (1.00) | 0/8 vs 1/8 (1.00) |
| ACCOUNTS T4 | CoPE line vs signed line | 0/16 vs 8/8 (< 1e-4) | 1/16 vs 8/8 (< 1e-4) | 8/16 vs 8/8 (0.0222) | 7/16 vs 8/8 (0.0095) |
| ACCOUNTS T4 | CoPE line vs CoPE ring | 0/16 vs 0/16 (1.00) | 1/16 vs 0/16 (1.00) | 8/16 vs 2/16 (0.0538) | 7/16 vs 2/16 (0.1134) |

*Table C10.* Two-sided Fisher p in brackets. Within this table only the RING_DIAL p_plus 0.5 contrast changes its reading
(at loss < 0.1). Outside it, ROBUST's torus SOLVED contrast does not survive loss < 0.1 (5/8 vs 8/8, Fisher p 0.20; a
declared secondary of that batch, C13).

**Where loss-based SOLVED and held-out accuracy disagree (reproduction audit, 2026-10-16).** SOLVED is the train-mode
(dropout on) final-5% loss. In ACCOUNTS T4, 5/16 CoPE line runs score 0.9997-1.000 held out (NLL <= 8e-4) with loss tails
0.062-0.089, so they count as unsolved, while the one SOLVED line run (s1104) scores 0.988; the CoPE ring arm has 2/16 at
>= 0.9999 and 0/16 SOLVED. At a held-out accuracy cut of 0.999 the line reads 5/16 vs 2/16 instead of 1/16 vs 0/16 (post
hoc; no test run). Single runs show the same pattern: WRAP_COPE CoPEF_Hi s904 (0.9997, tail 0.055), CoPEF_LoS s902 (0.9992,
0.059), ROBUST ring lr x3 AO s1400 (1.000, 0.138), WRAP_EXISTENCE AO_wrap s604 (0.9998, 0.162); conversely MECHANISM AO_mid
s800 is SOLVED at 0.968. SIGN_MATCHED Pos_r4 s7 is the softplus arm's only SOLVED run at tail 0.049 (knife-edge); that
contrast fires on accuracy regardless.

### C11. Ring revisit strata and Fig. C1 (post hoc; `docs/paper/figs/ring_strata_out.txt`, `FIGS_NOTES.md`)

A revisit's *gap* is the number of moves since the last visit to the cell; a *wrap-only* revisit has every earlier visit
at unwrapped displacement +-32k with k != 0. Stream: the registered ring eval stream (400 walks); overall accuracy
reproduces every run's registered value (max |diff| 5e-5).

| batch | arm | SOLVED | plain gap 2-7 | plain 8-31 | plain 32-127 | wrap-only | plain, wrap-gap-matched | all |
|---|---|---|---|---|---|---|---|---|
| (stream) | n targets | | 12183 | 17420 | 5110 | 4134 | | 38847 |
| RING_DIAL | Signed_r4 | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| RING_DIAL | Abs_r4 | 2/8 | 0.992 (0.97-1.00) | 0.975 (0.93-1.00) | 0.957 (0.91-1.00) | 0.923 (0.80-1.00) | 0.956 | 0.973 |
| WRAP_EXISTENCE | SO | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| WRAP_EXISTENCE | AO | 0/8 | 0.961 (0.89-1.00) | 0.916 (0.85-0.97) | 0.881 (0.81-0.93) | 0.872 (0.80-0.93) | 0.878 | 0.921 |
| MECHANISM | SO | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| MECHANISM | AO | 0/8 | 0.978 (0.92-1.00) | 0.918 (0.85-1.00) | 0.873 (0.77-1.00) | 0.811 (0.75-0.88) | 0.874 | 0.919 |
| MECHANISM | AO_high | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

*Table C11 (Fig. C1, `docs/paper/figs/fig_ring_strata.png`).* Means over runs, range in brackets. |.| models lose accuracy
with the gap in every batch (gap 2-7 to 32-127: -0.035 for Abs_r4, sign-flip p 0.031; -0.080 and -0.105 for the two AO
batches, 8/8 runs lower, p 0.0078 each). The wrap-only deficit beyond gap-matched plain revisits varies between batches:
-0.063 (MECHANISM AO, p 0.031), -0.007 (WRAP_EXISTENCE AO, p 0.88), -0.032 (Abs_r4, p 0.31). Signed models are at 1.000
in every stratum. Uncorrected for 3 batches x 2 contrasts; accuracy tracks loss, so the strata locate errors, not a
mechanism. The constant predictor scores 0.49-0.50 in every stratum.

### C12. Loose ends of the ring mechanism (post hoc; moved from the main text of draft 4)

Rules for the wall check and the channel-by-channel flow were fixed before the run
(`docs/audits/2026-10-15/reanalyses/PREDICTIONS.md`).

- **The |.| start is small in most channels, not all.** Per-channel initial sums span 0.01-6.58 rad; 5% of channels start
  past pi (`docs/paper/figs/fig_mechanism_out.txt`; `docs/audits/2026-10-14/theory_review/mech_channels_out.txt`). Of the
  channels whose step does not collapse to zero, the 20 (over 8 runs) that start with sums between pi and 3 pi / 2 end with
  none cancelling (`mech_bins_out.txt`): being past pi is not sufficient per channel.
- **Rescued models use the wrap sparingly.** From past pi, |.| models solve with 23% of channels cancelling (signed 34%; 7%
  at the start by chance; `MECHANISM_RESULTS.md` secondaries). Capture falls gently with distance from 2 pi: 0.34 within
  0.25 rad, 0.08 at 2-3 rad (`mech_capture_out.txt`).
- **The one-channel flow does not predict which channels cancel.** Gradient flow on the exact one-channel retrieval
  potential at a fresh model's attention scale (appendix E, Remark R) reproduces neither the signed escape (12% of channels
  cross zero) nor a sharp past-pi rescue (only channels within ~0.5 rad of 2 pi are captured: 28% of a U(1.7, 4.5) start).
  Run from each AO_high channel's own start (505 channels over 8 runs; `capture_by_channel_out.txt`) it predicts 0.329
  cancelling against 0.230 observed; agreement with the trained models is barely above chance (Cohen's kappa +0.086; per run
  -0.26 to +0.51); it captures 95% of the channels starting within 0.5 rad of 2 pi, where the models cancel 28%; 59% of the
  models' cancelling channels are ones it does not capture. For SO_high, kappa +0.198. Whether the rescue is collective is
  untested; what the other 77% of channels do is unknown.
- **The wall.** 44% of channels in |.| models from the same-direction start end with the forward-move angle within 0.05 rad
  of zero, against 17% for signed models from that start and 0.2% and 1.4% for |.| from the large starts
  (`fig_mechanism_out.txt`). Per move step (`collapse_wall_out.md`): collapse is common to every |.| arm from a small start
  (ring 0.235-0.424, torus 0.601-0.695, softplus 0.624 included) and smaller in signed controls from the default start (ring
  0.104-0.157, torus 0.343-0.402). It is partly inherited: of the |.| steps that start within 0.05 rad of zero, 77-81% end
  there (signed 29-35%). Attention reads the collapsed |.| channels (usage ratio collapsed / not 0.39-1.10). At the end of
  training the loss gradient points into the wall for 51-57% of the collapsed steps in the AO arms (eval mode, ring eval
  stream), which the pre-stated rule reads as MIXED, not as P4. Torus gradients were not computed.

| batch | arm | collapsed move steps, mean (run range) |
|---|---|---|
| SIGN_MATCHED torus | Signed_r4 / Abs_r4 / Pos_r4 | 0.343 (0.28-0.41) / 0.695 (0.62-0.88) / 0.624 (0.54-0.71) |
| RING_DIAL p 0.5 / 0.75 / 0.9 | Signed_r4 | 0.104 / 0.107 / 0.156 |
| RING_DIAL p 0.5 / 0.75 / 0.9 | Abs_r4 | 0.302 / 0.270 / 0.235 |
| RING_BUDGET (900 epochs) | Abs_r4 | 0.366 |
| WRAP_EXISTENCE | SO / AO | 0.157 / 0.337 |
| MECHANISM | SO / SO_absmag / SO_high | 0.145 / 0.171 / 0.001 |
| MECHANISM | AO / AO_absmag / AO_mid / AO_high | 0.368 / 0.424 / 0.010 / 0.001 |
| GENERALITY torus | SO4 / AO4 | 0.357 / 0.660 |
| TORUS_START torus | SO4 / AO4 / SO4_high / AO4_high / AO4_mid | 0.402 / 0.601 / 0.062 / 0.061 / 0.127 |

*Table C12 (`collapse_wall_out.md` (a)).* Collapsed = |a| <= 0.05 rad, over (move, channel) entries.

- **Start size.** A start nearer the wrap also moves the angles 13-22x faster per Adam step; the sub-pi control (AO_mid,
  ~5x) separates sum from size only partly (`MECHANISM_RESULTS.md`, M2b).

---

### C13. Recipe and scale (Sec. 4.1; `ROBUST_RESULTS.md`, `ROBUST_ANALYSIS.txt`; registered, seeds 1400-1407)

Observation-step-free SignedAO / AbsAO. Ring: `train_gauge.py --deltas=1,-1`, 32 cells, T = 128, 300 epochs, four recipes.
Torus: GENERALITY's recipe at 2 layers, d_model 256 (1.59M parameters), T = 1024, 900 epochs. Reading: PARTLY UNMEASURED.

Registered decisions (AO vs SO; Fisher on SOLVED, permutation on accuracy)
| cell | recipe | AO acc, SOLVED | SO acc, SOLVED | d [95% CI] | perm p | Fisher p | label |
|---|---|---|---|---|---|---|---|
| ring def | lr 1e-3, wd 0.05 | 0.911, 1/8 | 1.000, 8/8 | -0.089 [-0.125, -0.055] | 0.0002 | 0.0014 | **SIGNED LEARNS BETTER** |
| ring wd0 | lr 1e-3, wd 0 | 0.873, 0/8 | 1.000, 8/8 | -0.127 [-0.200, -0.075] | 0.0002 | 0.0002 | **SIGNED LEARNS BETTER** |
| ring lrx3 | lr 3e-3, wd 0.05 | 0.987, 4/8 | 1.000, 8/8 | -0.013 [-0.025, -0.005] | 0.026 | 0.077 | **UNMEASURED** |
| ring lrd3 | lr 3.3e-4, wd 0.05 | 0.841, 0/8 | 0.950, 6/8 | -0.109 [-0.215, +0.010] | 0.089 | 0.0070 | **SIGNED LEARNS BETTER** |
| torus d256L2 | 2 layers, d 256 | 0.967, 2/8 | 0.999, 8/8 | -0.032 [-0.055, -0.015] | 0.0005 | 0.0070 | **SIGNED LEARNS BETTER** |

Gates passed in every cell (SO >= 4/8). lrx3 is UNMEASURED, not NO SIGN COST: SOLVED differs by 4 (Fisher p 0.077) and the
accuracy gap (perm p 0.026) is smaller than the registered 0.05 materiality. lrd3's accuracy contrast is not detected (perm
p 0.089, CI [-0.215, +0.010]); its label rests on SOLVED. Across recipes at matched seeds (descriptive) |.| solves 0/8 (lr
/3), 1/8 (default), 4/8 (lr x3): the comparison is at the signed arm's default rate, and the lr cells also change the
effective decay (lr x wd) and the cosine floor (ROBUST Amendment 1). Floors (post hoc): torus on this batch's walks blank
0.509, n-gram 0.578, retrace 0.844, every run of both arms above retrace (lowest |.| 0.910); torus strata (lag < 128 /
>= 128 / wrap-only) SO 1.000 / 0.999 / 0.992, AO 0.989 / 0.911 / 0.799. The SOLVED-at-loss-0.1 column below is a declared
secondary (not post hoc).

Declared secondaries:

| cell | arm | acc 4T (ring 512) / 2T (torus 2048) | loss AUC | SOLVED at loss 0.1 | wrap-cancel | opposite-sign |
|---|---|---|---|---|---|---|
| ring def | SO / AO | 0.996 / 0.773 | 0.117 / 0.607 | 8/8 / 1/8 | 0.85 / 0.07 | 0.99 / 0.04 |
| ring wd0 | SO / AO | 0.978 / 0.758 | 0.150 / 0.796 | 8/8 / 1/8 | 0.43 / 0.11 | 0.86 / 0.05 |
| ring lrx3 | SO / AO | 1.000 / 0.934 | 0.077 / 0.453 | 8/8 / 4/8 | 1.00 / 0.13 | 1.00 / 0.07 |
| ring lrd3 | SO / AO | 0.911 / 0.747 | 0.294 / 0.825 | 7/8 / 0/8 | 0.69 / 0.04 | 0.79 / 0.03 |
| torus | SO / AO | 0.983 / 0.888 | 0.070 / 0.319 | 8/8 / 5/8 | 1.00 / 0.11 | 1.00 / 0.06 |

Per seed (accuracy at the training length, class):
```
ring  seed | def:SO | def:AO | wd0:SO | wd0:AO | lrx3:SO | lrx3:AO | lrd3:SO | lrd3:AO
1400 | 1.0000 SOLV | 0.8230 STAL | 1.0000 SOLV | 0.9078 DESC | 1.0000 SOLV | 1.0000 DESC | 1.0000 SOLV | 0.8893 STAL
1401 | 1.0000 SOLV | 0.9155 STAL | 1.0000 SOLV | 0.6239 STAL | 1.0000 SOLV | 0.9596 STAL | 0.6194 STAL | 0.6221 STAL
1402 | 1.0000 SOLV | 0.9237 STAL | 1.0000 SOLV | 0.8878 STAL | 1.0000 SOLV | 0.9702 STAL | 1.0000 SOLV | 0.8319 STAL
1403 | 1.0000 SOLV | 0.9136 STAL | 1.0000 SOLV | 0.8800 STAL | 1.0000 SOLV | 0.9995 SOLV | 1.0000 SOLV | 0.9167 STAL
1404 | 1.0000 SOLV | 0.9118 STAL | 1.0000 SOLV | 0.9119 STAL | 1.0000 SOLV | 1.0000 SOLV | 1.0000 SOLV | 0.8313 STAL
1405 | 1.0000 SOLV | 0.8791 STAL | 1.0000 SOLV | 0.8858 STAL | 1.0000 SOLV | 0.9650 STAL | 1.0000 SOLV | 0.8526 STAL
1406 | 1.0000 SOLV | 0.9211 DESC | 1.0000 SOLV | 0.9110 STAL | 1.0000 SOLV | 1.0000 SOLV | 0.9794 STAL | 0.9062 STAL
1407 | 1.0000 SOLV | 0.9997 SOLV | 1.0000 SOLV | 0.9735 DESC | 1.0000 SOLV | 0.9998 SOLV | 1.0000 SOLV | 0.8801 STAL
torus 1400 SO 1.0000 SOLV | AO 0.9941 DESC     1404 SO 1.0000 SOLV | AO 0.9935 SOLV
      1401 SO 1.0000 SOLV | AO 0.9981 SOLV     1405 SO 1.0000 SOLV | AO 0.9706 DESC
      1402 SO 0.9941 SOLV | AO 0.9705 DESC     1406 SO 1.0000 SOLV | AO 0.9098 DESC
      1403 SO 1.0000 SOLV | AO 0.9454 DESC     1407 SO 1.0000 SOLV | AO 0.9544 DESC
```


## D. Constructions

### D1. Per channel (B0, P1.0)

For a signed step Delta and omega != 0, Delta' = Delta mod (2 pi / |omega|) lies in [0, 2 pi / |omega|) and
omega Delta' = omega Delta (mod 2 pi) for either sign of omega. The wrap is taken in step space: the earlier form
remainder(omega Delta, 2 pi) / omega is wrong for omega < 0, and trained omega does cross zero (pilot -0.012; RING_DIAL
seeds 31-33) (`WRAP_EXISTENCE_PREREG.md` A2).

### D2. The balanced construction (P1(a); the registered primary)

Source: `model_wrap_exist.wrap_construct(mode="balanced")`, `WRAP_EXISTENCE_PREREG.md` Amendment 1 (A1), on the torus
`build_torus_exist.py`. Given a solved signed observation-step-free model with m stepping move tokens (m = 2 ring, 4 torus;
r = 4):
1. Re-solve W_in so that the move latents are c e_1, ..., c e_m (c = the source's mean move-latent norm). W_in acts only on
   the move embeddings (every other step is 0), so this is a free gauge.
2. Per channel set omega' = max(|omega|, 1) and the steps to remainder(omega Delta, 2 pi) / omega' (angles in [0, 2 pi) for
   either sign of omega; the code's form).
3. Set W_out[:, j] = step_j / c for the m move columns and 0 elsewhere. Every other weight is copied.

The signed control SO_bal is built by the same function at the same W_in and omega' with the *unwrapped* signed angle (steps
omega Delta / omega'), so the two differ only in the wrap (same RNG consumption, same data stream). On the torus m = 4 = r,
exactly at P1(a)'s limit.

**Exactness (E1, registered).** Ring: frozen accuracy = source on 8/8 seeds; max angle error 1.2e-6 rad (AO_bal), 6.1e-7
(SO_bal); max logit difference 5.1e-4 at T = 128, 1.1e-2 at T = 512 (`WRAP_EXISTENCE_ANALYSIS.txt`). Torus: frozen = source on
8/8; max angle error 1.4e-6 rad (AO_bal), 5.8e-7 (SO_bal) (`GENERALITY_ANALYSIS.txt`).

**Weight scale.** Ring: max |W_out| 0.31-0.38 in the signed sources, 1.07-1.70 built (AO_bal); |W_in| 0.15-0.21 -> 0.19-0.33;
max step 6.28 (one turn) (`WRAP_EXISTENCE_ANALYSIS.txt`): within ~5x. Torus: max |W_out| 0.48-0.58 source -> 1.23-1.55 built
(AO_bal), max |omega| 0.68-0.96 -> 1.00 (`GENERALITY_ANALYSIS.txt`).

### D3. Why the naive construction fails as a test

Keeping the source's W_in and solving W_out (min-norm, `AO_wrap`) is exact (frozen accuracy 1.000 on 8/8, angle error
1.2e-5 rad) but ill-conditioned: in a solved signed ring model the two move latents are nearly antiparallel (cosine -0.987
to -0.998 over seeds 600-607), so W_out must be huge -- max |W_out| 653-16,320 against 0.31-0.38 signed, steps up to 9,333
(`WRAP_EXISTENCE_ANALYSIS.txt`). Under Adam's per-entry steps (rule 15: a parameterisation change is an optimiser change) the
continuation loses the solution in the warm-up: LOST on 8/8 (max loss 0.6-2.0; Fisher on not-LOST p 0.0002 against SO_cont)
and then re-finds it (7/8 SOLVED). A per-channel gauge version (`AO_wrapg`, max |W_out| 4.62 at seed 600) also loses it (8/8
LOST, 8/8 re-solved). By the reading fixed before launch, NOT HELD under an ill-conditioned construction says nothing about
the class. The pilot reading of this as an intrinsic "path-length fragility" was confounded and is withdrawn
(`WRAP_EXISTENCE_RESULTS.md`).

### D4. The full |.| arms (observation steps present; ring audit M1)

To show that the *task* (not a given signed table) is representable by `Abs_r4`, the ring-dial audit chose W_in to project
onto two directions orthogonal to all 17 observation embeddings and solved W_out per channel
(`docs/audits/2026-10-12/ring_dial_audit.md` M1; `wrap_construction_t2.py`). In the real modules: |.| steps >= 0 (min
5e-11), observation-token |Delta| <= 1.7e-6, cos / sin equal to the signed code's to 6.3e-4 over a 512-step walk, circular
mismatch <= 4.6e-4 on 2,000 same-cell revisit pairs. **Softplus (P1(f), appendix E; new in draft 6).** Softplus
cannot reach 0, but observation steps need only be 0 mod 2 pi: with W_in annihilating every observation embedding, each
observation steps softplus(0) = ln 2 in every channel, and omega'_c = 2 pi / ln 2 makes that exactly one turn; the move
columns are set to softplus^{-1}((remainder(Theta, 2 pi) + 2 pi) / omega'). Built in the real `MapFormerWM_Pos_r4` from the
solved MECHANISM ring model SO_s800 (`docs/paper/review_draft5/theory_check_extra.py` (2), CPU): min step 0.693 > 0, angle
error mod 2 pi 8.4e-15 rad over all 19 tokens, logits equal to the source's to 2.8e-11 (scale 21.6). On the torus m = 4 = r,
so the same construction applies (not built). Existence under training (frozen and trainable continuation) was not tested.

### D5. Continuations and the LOST rule

A constructed model is trained further on a fresh walk stream and compared with its signed control at the same seed.
**LOST** (registered; `WRAP_EXISTENCE_PREREG.md` A4, `GENERALITY_PREREG.md`): the run's max per-epoch training loss during
the continuation exceeds 0.3, or exceeds 10x its control's max at that seed and 0.05. **HELD** = SOLVED on >= n - 1 seeds AND
mean accuracy >= control - 0.01 AND LOST on <= 1 seed. **NOT HELD** = Fisher (SOLVED) p < .05 with the control ahead, OR
sign-flip p < .05 with d <= -0.05, OR Fisher (not LOST) p < .05 with more LOST in the constructed arm. Otherwise PARTIAL.

- **Ring** (lr 1e-3, 300 epochs, re-warmed schedule, fresh optimiser state): AO_bal HELD (8/8 SOLVED, 0/8 LOST; max per-epoch
  loss 0.024-0.154 vs the control's 0.015-0.112; Table C4).
- **Torus** (lr 1e-4 primary): at lr 1e-4 the control's losses are small, so LOST reduces to "left the solved region at some
  epoch"; the reading fixed before launch was that NOT HELD by LOST with SOLVED >= n - 1 at the end means "perturbed and
  re-found, not held in place" (`GENERALITY_PREREG.md` Amendment 1 A3). Why lr 1e-4: in the pilot (seed 690), re-warming to
  lr 1e-3 lifted the *signed* control itself from loss ~0 to ~0.1, so at 1e-3 the continuation cannot separate the arms; 1e-4
  is about the learning rate at which the source solution was reached (`GENERALITY_PREREG.md`). At lr 1e-3 (descriptive):
  AO_bal_hi 6/8 SOLVED (0.988), max loss 0.46-0.63, SO_bal_hi 8/8 (max loss 0.15-0.21) (Table C7).
- Ring and torus E2 labels are not directly comparable (continuation lr 1e-3 vs 1e-4).

### D6. What the held model keeps (ACCOUNTS T1, registered)

Weighted by usage (query x key norm per channel), the held ring AO_bal keeps 0.99 -> 0.98 of its cancelling code; the
unweighted share falls 0.87 -> 0.18 because unused channels leave the code (`ACCOUNTS_RESULTS.md`). T1's wd 0.05 runs are
bit-identical retrains of WRAP_EXISTENCE's AO_bal / SO_bal (reproduction audit 2026-10-16, integrity note 2), so T1 is a
second readout of the same runs, not a replication of E2 HELD; its wd 0 arms are new runs. Weight decay 0 vs 0.05
changes nothing detected (D1 -0.003, NO EFFECT DETECTED). WRAP_EXISTENCE's secondary "87% -> 18% drift" and the reading "not
an attractor" are withdrawn. On the torus the unweighted share goes 1.00 at construction -> 0.69 after training
(GENERALITY secondary; not usage-weighted). The frozen perturbation probe (1% multiplicative noise on omega drops the
wrapped ring solution to 0.802, the signed one stays 0.998) is a property of frozen models at a matched gauge, not of
training (appendix E, P2).

---

## F. Dyck-2 at matched depth

Source: `DYCK_MDEPTH_RESULTS.md`, `DYCK_MDEPTH_PREREG.md`, driver `run_dyck_mdepth.sh`, trainer `train_dyck.py`.

**Task and metric.** Dyck-2 (MapFormer v4 App. B.4), trained and scored at length L = 32 and nesting depth D = 12. Primary
metric A2f: Hewitt's distance-averaged closing accuracy over prefixes where the sampler can emit a closer (the CE-optimal
predictor scores 1.000; chance 0.500); best map-free floor 0.594 (appendix B).

**Models and recipe.** 2 heads, d_model 128; 1-4 layers; path-integrated arms MapWM and MapPoPE at rank 2 (`--rank 2`), index
arms RoPE and PoPE at base 10000 (and base 32 in the 4-layer 3x cell); omega base 32 (the training context). AdamW lr 1e-4,
weight decay 0.01, batch 128, 5% linear warm-up then cosine to 10% of the peak, dropout 0.1, next-token cross-entropy on every
position; 560,000 training sequences (1x) or 1,680,000 (3x). n = 8 seeds (0-7), paired.

**Results (registered).** Position main effect = ((MapWM - RoPE) + (MapPoPE - PoPE)) / 2, A2f at L32 D12:
- Depth ladder at 1x budget (declared secondary S1): +0.353 / +0.130 / +0.045 / +0.024 at 1 / 2 / 3 / 4 layers (8/8 seeds
  positive each, sign-flip p 0.0078; MDE 0.003 / 0.025 / 0.009 / 0.007).
- Primary, 4 layers at 3x budget: every arm at ceiling (index 0.997-0.998, path 1.000); effect +0.002: CLOSES. The earlier
  ladder's +0.168 at 4 layers was depth extrapolation (trained at D = 4).
- Mixture training (D drawn from 4..12, 4 layers): +0.110 at L32 D12 (8/8).
- Index base 32 vs 10000 at 4 layers: within MDE (+0.001 / -0.001). The 1x budget limits the index arms (3x - 1x +0.021).

So one layer of path integration is worth about three layers of index attention on Dyck-2 at matched depth (at 3 layers the
gap is +0.045; at 4, +0.024).

**Protocol.** The batch (2026-09-26) predates the independent code checks. Disclosed before registration: four n = 1 smoke
runs with non-default flags at seed 0, whose training losses were seen (no A2 / A2f computed); the batch retrained seed 0
from scratch and nothing from the smoke runs enters it. The in-batch reproduction of two stored ladder runs was weight-exact.

---

## G. Navigation in words

Source: `TEXTWORLD_RESULTS.md` (its AUDIT and CORRECTED blocks first), driver `run_textworld.sh`, trainer
`train_textworld.py`.

**Task.** The 64 x 64 torus walk rendered as English: 58 words, three synonyms per direction, movement-free verbs, fillers
and asides; T = 1024 words (~132 steps); held-out map; predict the observation word at revisits. Floors: best constant
0.505, reversal-copy rule 0.597 (appendix B).

**Recipe.** As the torus (A3): AdamW lr 1e-3, batch 16, 98 batches per epoch, 900 epochs, cosine; path MapWM r = 4 at one
layer vs index RoPE at one and two layers; 200 evaluation walks.

**Results.** Registered (8 seeds per arm): path 0.969 +/- 0.054 (7/8 SOLVED) vs index one layer 0.505 (0/8) and two layers
0.772 (0/8); path - index one layer +0.464 (exact permutation p 0.0002; Fisher p 0.0014). **Seeds 0 and 1 are the pilot**
(byte-identical to `runs/textworld_pilot`), so the numbers to quote are the six fresh seeds: path 0.979 +/- 0.048 (6/6
SOLVED) vs index 0.505 (0/6), +0.474, permutation and Fisher p 0.0022; index two layers 0.793. Dropout-scale re-score
(AUDIT 2026-10-04, all 24 runs): path 0.969 -> 0.993, index 0.505 -> 0.506 and 0.772 -> 0.772; the verdict only
strengthens. Declared secondaries: with the common component of the four direction steps removed, north + south and west +
east cancel to 0.002-0.012 on all 8 seeds; 4/8 seeds also run a per-step clock (31-38 of 64 phase channels drift between
visits to the same cell; CORRECTED block). The registered step-table branch (B) did not fire (4/8).

**Scope.** Scripted grammar; a direction word never appears outside a movement clause (context-free steps).

---

## H. Faithful CARoPE and CoPE

Sources: `model_mono_baselines.py` (docstring), `GENERALITY_RESULTS.md`, `WRAP_COPE_RESULTS.md`, `ACCOUNTS_RESULTS.md` T4,
`docs/audits/2026-10-14/cope_diagnosis/COPE_DIAGNOSIS.md` (post hoc), appendix E P6 and P7.

**CARoPE (`CARoPEF`).** Per head h and token t a scalar rate f_h(x_t) = 1 / (softplus(x_t W)_h + 1) in (0, 1) replaces RoPE's
base frequency, so channel i advances by f_h(x_t)^i per token. Initialised as index RoPE (W = 0, a per-head bias with
1 / (softplus(b) + 1) = 10000^(-2 / d_head)); the bias is ours (without it the start is RoPE with base ~2e7). Every token,
observations included, advances the phase; the layer is MapWM's, so only the phase differs. *P6 (proved, appendix E):* the
shortest return advances each channel by a value in (0, 4] < 2 pi, so no channel reads every return at phase 0; a channel
that reads the short returns is blind over the excursion's range. Registered (GENERALITY): torus 0.667 (0/8), WORSE THAN
INDEX (0.722, -0.055, p 0.040); ring 0.741 vs index 0.730, UNMEASURED (+0.011, p 0.39).

**CoPE (`CoPEF`).** Golovneva et al.'s Listing 1, verbatim in torch: per layer one module shared by the heads; gates
g_ij = sigmoid(q_i . k_j / sqrt(d_h)); count p_ij = sum_{k=j}^{i} g_ik, clamped to npos_max - 1 = 2047; position logit by
linear interpolation of q_i . e[floor p] and q_i . e[ceil p], table e zeros at initialisation; added to the attention logits;
no rotary position. *Representability (constructed, GENERALITY audit):* gates 17/16 and 15/16 per (move, observation) pair
make p_ij mod 2 a function of net displacement, read by a period-2 table (1.000 revisit accuracy). *P7 (appendix E, proved
under its setup):* the exact solutions form a lattice (period P = 2 or 3; alpha an odd multiple of P/32, i.e. move-gate
difference an odd multiple of P/16).

**Prior measurement.** MapFormer v4 already reports CoPE failing its 1D and 2D navigation tasks in distribution (Table 2:
1 layer 0.63 +/- 0.21 in 1D and 0.65 in 2D, 4 layers 0.88 and 0.76, MapWM-r2 1.00) and attributes it to the gate's sign
("cannot represent an action and its inverse", their App.). So "CoPE fails these maps" is a replication; what is new here is
that CoPE can represent ring cancellation and still fails, the line control, and the CARoPE results.

**Results.**
| batch | task | CoPE | SOLVED | comparison |
|---|---|---|---|---|
| GENERALITY (700-707) | ring | 0.887 | 0/8 | signed 1.000, 8/8 (-0.113, p 0.0002); index 0.730 (+0.157, p 0.0002) |
| WRAP_COPE (900-907) | ring, table at zeros (the paper's) / sinusoid past pi / below pi / slope-matched | 0.840 / 0.913 / 0.867 / 0.943 | 1/8, 2/8, 1/8, 0/8 | all contrasts UNMEASURED (bimodal seeds) |
| ACCOUNTS T4 (1100-1115) | line (256 cells) / ring | 0.914 / 0.873 | 1/16 / 0/16 | line: signed 1.000 8/8, index 0.772 0/8; L1 line vs ring NO EFFECT DETECTED; L2 SIGNED LEARNS BETTER |

Per seed (ACCOUNTS T4): CoPE line 0.836 0.868 0.867 0.824 0.988 1.000 0.848 0.620 1.000 0.886 0.999 1.000 0.894 1.000 0.998
1.000; CoPE ring 0.850 0.872 0.799 0.874 0.834 1.000 0.797 0.666 0.915 0.861 0.957 0.896 0.823 0.974 0.848 1.000. Several
CoPE seeds reach 0.99-1.00 accuracy without meeting the loss criterion; at loss < 0.1 half the line runs pass (8/16 vs 2/16
on the ring, Fisher p 0.054; post hoc, Table C10), and at a held-out accuracy cut of 0.999 the line reads 5/16 vs 2/16 (post
hoc; C10 note). The loss-based SOLVED count is fragile for CoPE, so the line-vs-ring comparison is unmeasured, not null. The "1 of 32 runs at the paper's initialisation" of Sec. 4.6 pools
GENERALITY (0/8), WRAP_COPE zeros (1/8) and ACCOUNTS T4 ring (0/16), post hoc.

**Diagnosis (post hoc, eval-only, 40 ring runs; predictions written before any checkpoint was opened).** Runs fall into
three regimes: NONE (20/40 runs have a query type with no locked count; identity-dependent observation gates), OFF (11 runs:
the count locks, local revisits are solved, but the move-gate difference is off the lattice, so wrap-only revisits sit near
the floor, mean 0.55), LATTICE (9 runs, all 4 SOLVED runs among them). In the OFF runs an eval-only snap of the move gates to
the nearest lattice point raises wrap-only accuracy 0.63 -> 0.90 and the straight path to the lattice rises in loss before it
falls (11/11). The registered line test then showed the wrap is not CoPE's main obstacle (T4). The precision a count needs
(on the line, a margin of at most P / (D_max + 1) ~ 0.016; appendix E, P7) is a plausible account, not a tested one; the
near-wrap table start does not measurably help (WRAP_COPE, unmeasured).

---

## I. The toy (exploratory, not registered)

Source: `MECHANISM_PREREG.md` (exploratory section and Amendment 1, A1), `ACCOUNTS_RESULTS.md` T3, `runs/toy_wrap/summary.txt`,
`runs/accounts/t3/summary.txt`. A 64-channel retrieval model with no transformer and no observation tokens: two move angles
per channel (rank-4 latents, signed or |.|, MapWM's frequency ladder), softmax retrieval of earlier moves at the same ring
cell by sum_c w_c cos(theta_c(t) - theta_c(t')). SOLVED = retrieval accuracy >= 0.99.

| start (move angles) | signed | \|.\| |
|---|---|---|
| default | 6/12 (median 0.946) | 0/12 (0.516) |
| \|default\| (the \|.\| start) | 0/8 (0.406) | 0/8 (0.385) |
| large, sums below pi, U(0.9, 1.5) | 0/8 | 0/8 |
| large, sums above pi, U(1.7, 4.5) | 8/8 | 11/11 |

It reproduces the |.| failure without a transformer, but its signed model fails often from the default start and never
escapes the |.| start (0/8), unlike the transformer (7/8). An observation-retrieval target (ACCOUNTS T3, seeds 10-17) does not
open the escape either (signed from the |.| start 0/8 MAP SOLVED; opposite-sign channels 0.27 vs 0.23 with the cell target).
The toy and the transformer differ for an unidentified reason; the toy's stronger "the cost is the start, not the constraint"
did not transfer and is not claimed.

---

## J. The clock side is not established

A barrier step might learn a content *clock* faster than a signed one. On k-back recency (k_max 128, T = 1024, 300 epochs, one
layer) it trends that way and is unmeasured twice (registered): loss AUC (mean training loss) |.| - signed -0.197 (exact
sign-flip p 0.053, faster on 12/16 seeds; `CLOCK_SPEED_RESULTS.md`), then -0.106 on 16 fresh seeds (p 0.19, 10/16;
`CLOCK_SPEED_REP_RESULTS.md`). Pooled post hoc -0.151 (p 0.018), pre-committed not to upgrade the verdict. On accuracy
(`CLOCK_SIDE_RESULTS.md`) every barrier-vs-signed contrast is UNMEASURED (|.| +0.085, CARoPE +0.169, softplus +0.084; perm p
0.16-0.53) and no run converged (0/40 SOLVED); index RoPE fails the task (-0.450, p 0.0011). We do not claim a double
dissociation. A consistent but untested account: signed steps contain every non-negative path, so a clock costs them a delay
(moving minority signs through zero), while a map costs |.| a block.

---

## K. Where a map is needed, one path layer beats index attention at the depths we tried

On the torus at T = 1024 (FACTORIAL, n = 4 per cell, seeds 500-503), one signed path layer scores 1.000 (4/4 SOLVED) against
index RoPE at one layer 0.719, four layers 0.860 and a block looped four times 0.864 (0/4 each, all stalled within 900
epochs; complete separations, p 0.029; floors: blank 0.507, n-gram 0.576, computed on another stream of the same generator,
appendix B). Depth and looping lift index accuracy only on revisits within 128 moves (0.74-0.79 -> 0.90-0.96); long-gap
(>= 128 moves) and wrap-only revisits stay at 0.37-0.53, near or below the blank floor (Fig. K1,
`docs/paper/figs/fig_factorial_strata.png`; its dashed line is the mean of the per-stratum best-constant floors on
FACTORIAL's own stream, 0.513, overall 0.509; the 0.507 / 0.576 above are another stream of the same generator). Told in English words (appendix G; pilot seeds excluded): path 0.979 vs index
0.505, +0.474 (6/6 vs 0/6 SOLVED, p 0.0022); the eight-seed registered numbers are 0.969 vs 0.505 (p 0.0002), and with the
dropout-scale re-score the path arm reads 0.993.

This is a learnability measurement, not a capacity bound. The torus walk is the group Z_64 x Z_64; Liu et al. (2023) show
that one attention layer with an MLP can simulate such a group (their Lemmas 6 and 8), using absolute positions and width and
weights growing with T, so their construction does not settle whether our 128-wide RoPE models can represent the task. Their
experiments are mixed for our purpose: at T = 100 with every state labelled, the median of 20 runs stays below 90% for C_4
at 1-15 layers (and C_2, C_6 are unreliable), while C_5, C_7 and C_8 reach >= 98.5% from 3 layers. Our labels are sparse and
our groups larger. On Dyck-2 at matched depth one path layer is worth about three index layers (appendix F).

**Related work on depth (moved from the main text of draft 5).** Shallow transformers learn automaton shortcuts unreliably
(Liu et al. 2023; Hahn and Rofin 2024), while Selective RoPE (Movahedi et al.) and PaTH (Yang et al. 2025) solve parity or
Flip-Flop in one layer.
