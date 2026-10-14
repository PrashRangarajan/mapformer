# Workshop paper outline (4-6 pages + appendix), 2026-10-13

Source of every claim: `docs/paper/CLAIM_MAP_clock_map.md` (status per claim) and the results files named below. Numbers
here are copied from those files; re-check against them before drafting text. Status tags: REG = pre-registered batch,
POST HOC, PENDING (batch not yet run).

**Working title:** *Maps and clocks in rotary attention: a learned phase rate is Howard's integrator*
(alt.: *When does a learned rotary phase beat the token index? Sign, wrapping and learnability*)

**One-sentence thesis.** Rotary position codes integrate a rate; whether the rate is constant, non-negative or signed is
Howard et al.'s (2014) time / path-length / position distinction, and on tasks that need a map a signed, content-dependent
rate wins at the training length -- not because the alternatives cannot represent a map (in the rotary slot they can, by
wrapping), but because training does not find it.

**Target venue.** A workshop on geometry / representations in neural networks (e.g. NeurReps-type) or on positional
encoding / sequence models. Check deadlines and page limits before drafting; nothing here assumes one.

## 1. Introduction (~0.6 page)
- Rotary position as an integrator: theta_t = omega * sum of a per-token step. Index RoPE, content-dependent monotone
  phases (CoPE, CARoPE, GRAPE-AP) and signed content-dependent phases (MapFormer, Selective RoPE, Mamba-3) differ only in
  the step.
- Neuroscience already has the taxonomy: Howard et al. 2014's integrator with a constant rate (time), a speed (path length)
  or a signed velocity (position).
- Questions: (Q1) where does a learned signed phase beat the index, at matched length and depth? (Q2) what does the sign of
  the step buy in the rotary slot? (Q3) what do models learn on natural text?
- Contributions (bullets, each one sentence with its number):
  1. The identification (a remark, stated as such): Howard's Cases I / II / III = index / monotone content / signed content
     rotary phase; in the rotary slot Cases II and III coincide representationally (wrap mod 2 pi).
  2. On navigation, 1 layer of signed path integration beats index attention at 1 layer, 4 layers and looped x4 (REG).
  3. Sign is learnability: a non-negative rotary step that can represent the map learns it on 0-3/8 seeds vs 8/8, never
     finds the wrapped code, and is not rescued by 3x training (REG); an exact non-negative solution is held under training
     (8/8) though never found from scratch (0/8) (REG).
  4. The effect is asymmetric: no established advantage for the non-negative step on a content-clock task (REG, twice
     unmeasured).
  5. On TinyStories / code the learned phase is a clock (growth exponent ~1.0) with no gain over the index at our scale
     (POST HOC / PILOT; scoped against CoPE).

## 2. Framework (~0.75 page)
- Eq.: dF/dt = alpha(t)(-s F + f) (Howard). Real s = decay memory; imaginary s = rotation = rotary attention. alpha(t) =
  the per-token step = selective-SSM Delta_t = MapFormer's W_out W_in x_t.
- Table 1: the three cases | rate | what the interval sum measures | rotary family | examples.
- Remark (construction, `docs/audits/2026-10-12/ring_dial_audit.md` M1): a non-negative step 2 pi / omega - Delta
  reproduces -Delta mod 2 pi, so a non-negative rotary code can represent any signed one. Prediction: if sign matters in
  practice, it matters for LEARNING, and the cost of wrapping is that each opposite move needs a step of size ~2 pi / omega
  (large on slow channels).
- Predictions stated before the results: map tasks favour signed; clock tasks no worse (or better) with non-negative;
  natural text learns Case I-like behaviour.

## 3. Setup (~0.5 page)
- Model: MapFormer-WM (1 layer, 2 heads, d 128), step = W_out W_in x with shared rank 4; arms Signed, Abs (|.|, the sign
  removed and nothing else), index RoPE; deeper / looped index transformers for the depth factorial.
- Tasks: 2D torus walk with observations (T = 1024, revisits scored, maps redrawn at eval: a transfer measurement); 1D ring
  with a dial p_plus (0.5: steps cancel; 1.0: pure clock); k-back recency (content clock); navigation told in English
  (58 words); Dyck-2 (depth substitution, appendix).
- Protocol: pre-registered decisions, n = 8 seeds (factorial n = 4), trained and tested at the same length, SOLVED = final
  5% training loss < 0.05, measured floors (best n-gram, blank), independent code check per batch.

## 4. Results (~2.2 pages)
### 4.1 Where a map is needed, one path layer beats several index layers (Q1)
- Table 2 (`FACTORIAL_RESULTS.md`, REG n = 4): path 1L 1.000 (4/4) vs RoPE 1L 0.719, 4L 0.860, looped x4 0.864 (0/4 each,
  all STALLED; complete separations p 0.029). Floors 0.507 / 0.576.
- Fig. 1: accuracy by revisit stratum: depth lifts index on revisits within 128 moves (0.74-0.79 -> 0.90-0.96) but not long
  gaps (0.37-0.53) or wrap-only revisits; path 1.000 everywhere.
- Converged torus recipe +0.243 at 8/8 (`PAPER2X2_RESULTS.md`); in words 0.969 vs 0.505 (+0.464, p 0.0002, 7/8 vs 0/8;
  `TEXTWORLD_RESULTS.md`). Dyck at matched depth: 1 path layer ~ 3 index layers (appendix, `DYCK_MDEPTH_RESULTS.md`).
### 4.2 In the rotary slot, sign is learnability, not representation (Q2)
- Torus at matched length (`SIGN_MATCHED_RESULTS.md`, REG): Abs - Signed -0.177 (p 0.0002), SOLVED 0/8 vs 8/8.
- Ring dial (`RING_DIAL_RESULTS.md`, REG): Abs SOLVED 2/8 and 1/8 vs 8/8 at p_plus 0.5 / 0.75 (Fisher p 0.007 / 0.0014);
  0.9 unmeasured. 3x budget: 3/8 vs 3/8 (`RING_BUDGET_RESULTS.md`, REG PARTIAL).
- Fig. 2 (`docs/paper/figs/fig_wrap_share.py`, wrap-invariant selection; post hoc re-description of the stored
  checkpoints): share of channels where opposite moves cancel mod 2 pi, mean per group: signed 0.79 / 0.78 / 0.64 (ring
  p_plus 0.5 / 0.75 / 0.9) and 0.98 (torus) vs non-negative 0.03-0.11 on every run, SOLVED or not -- the non-negative runs
  that solve do so without the wrapped (cancelling) code.
- Learning speed on the ring: signed faster on 8/8 paired seeds at every p (loss AUC, p 0.0078; POST HOC).
- Existence (rule 14; `WRAP_EXISTENCE_RESULTS.md`, REG): in observation-step-free ring models, a signed solution converted
  exactly to non-negative steps (balanced construction, weights within ~5x of the signed scale) is held under 300 epochs
  of training (8/8 SOLVED, never lost), while the same non-negative class from scratch solves 0/8 vs signed 8/8 (p 0.0002).
  Text: "the solution exists and training keeps it; training does not find it". Add the nuance: the held model drifts off the
  cancelling code (87% -> 18% of channels) and the wrapped code is more fragile to 1% parameter noise (frozen 0.80 vs 0.998)
  -- the exact map code is not an attractor under the non-negative constraint. Methods note (appendix D): the naive
  construction is exact but ill-conditioned (antiparallel action latents) and is lost under Adam; say why the balanced one
  is the right test.
### 4.3 The clock side is asymmetric (Q2, other direction)
- k-back recency, loss AUC Abs - Signed: -0.197 (p 0.053) and on fresh seeds -0.106 (p 0.19): UNMEASURED twice
  (`CLOCK_SPEED_RESULTS.md`, `CLOCK_SPEED_REP_RESULTS.md`, REG). Index RoPE fails the task (counts tokens, not content;
  `CLOCK_SIDE_RESULTS.md`). State plainly: no double dissociation.
### 4.4 Natural text learns a clock (Q3)
- TinyStories: growth exponent 1.00, R^2(phase ~ token index) 0.999 (3/3); MapWM = RoPE (1.5053 vs 1.5050 nats).
  Code: exponent 0.99-1.02; path integration costs +0.0056 bpc at matched length (n = 3). (POST HOC / PILOT;
  `docs/audits/2026-10-12/alpha_language_out.txt`, `TINYSTORIES_PILOT_RESULTS.md`, `CODE_FULLVAL_RESULTS.md`.)
- Scope sentence: CoPE gains over RoPE at larger scale; ours is a small-model observation.

## 5. Related work (~0.5 page)
Howard et al. 2014 and the Howard / Tiganj lab's real-s ML work (they set aside the rotary side); GridPE (VCO -> RoPE);
Sarrof 2024, Grazzi 2025, Khavari 2025 (sign / negative eigenvalues for state tracking: our sign result replicates theirs in
a navigation regime); Selective RoPE, Mamba-3, GRAPE (signed content rotation, the taxonomy); CoPE, CARoPE (monotone);
MapFormer and TEM (path-integrated position, what/where); PaTH, Liu et al. 2023 (depth and prefix sums). Use
`docs/lit/LIT_CLOCK_MAP.md` and `docs/lit/LIT_HOWARD_TRANSFORMER.md`; corpus in `papers/INDEX.md`.

## 6. Limitations (~0.3 page)
Small models (1 layer for map tasks, 6-9 layers on text: a cross-task confound); synthetic map tasks; fixed budgets
(learnability is budget-scoped, though 3x did not help); n = 4 in the factorial; no recurrent (LSTM / Mamba) or CoPE /
Selective RoPE baselines on the torus; the clock side unestablished.

## Appendix plan
A: models and recipes; B: floors and n-gram gates; C: per-seed tables; D: the wrap construction and code; E: Dyck matched
depth; F: navigation in words; G: per-head rank (rank 2 vs 4 on the torus) as a side result if space allows.

## Do not claim
Double dissociation; "a monotone rotary phase cannot represent a map"; the OOD crossover (-0.280 / -0.004) as a matched
result; "beats the index exactly when the task needs a map" as a general law; anything in CLAUDE.md's withdrawn list.

## Before drafting
1. Done: WRAP_EXISTENCE verdicts are in 4.2.
2. Figures 1-2 drafted (`docs/paper/figs/`); a framework figure (Table 1 as a diagram) is optional.
3. Optional for a stronger workshop version: factorial at n = 8 (~8 h GPU). Required only for a journal version: torus
   baselines (LSTM, Mamba-like, CoPE, Selective RoPE) and one scale check (~2 GPU days).
