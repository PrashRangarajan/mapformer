# What went wrong with CoPE on the ring, and does it match theory? (2026-10-14, post hoc diagnosis)

This is a post hoc analysis of stored checkpoints. It is eval-only and needed no training. It covers 40 runs: WRAP_COPE CoPEF, _Hi, _Lo and _LoS at seeds 900-907, plus GENERALITY CoPEF at seeds 700-707. Every number here comes from a script in this directory, and each script's output sits beside it. The predictions in Sec. 2 were written to `PREDICTIONS.txt` (18:05) before any checkpoint was opened. Nothing here is registered.

## 0. Answer

- **The failure is in learning, not in what CoPE can represent, and it is not one failure.** A seed lands in one of three discrete regimes. They account for the bimodal seeds, and the theory predicts two of them.
  - **NONE** (20/40 runs have at least one query type in it; run accuracy 0.58-0.96). No head ever locks a count that a periodic table can read. The observation gates have become identity-dependent: tau median 0.17 against about 0.0002 in heads that locked. So even D = 0 returns do not land on one table phase, and accuracy decays with the gap.
  - **OFF** (11 runs have it without NONE; accuracy 0.938-0.973). The count locks exactly (sigma = 1.0000 with period 2), and local revisits are solved (0.98-1.00). But the move-gate difference is off the lattice the theory requires: 8 (g+ - g-) is not an odd integer. So **wrap-only revisits sit near the floor**: mean 0.55 (0.43-0.94; floor 0.554).
  - **LATTICE** (both query types locked at a lattice point, odd k or aliased even k, in 9 runs; accuracy 0.969-1.000, all 4 SOLVED runs are among them). The heads sit on odd k to within 0.01, as derived.
- **Representational?** No.
  - In the OFF runs, an eval-only snap of the move gates to the nearest lattice point, leaving every other weight unchanged, raises wrap-only accuracy from 0.63 to 0.90 and lowers NLL from 0.153 to 0.068. Accuracy rises in 9/11 runs.
  - The straight path from the learned gates to the lattice rises in NLL before it falls in 11/11 OFF runs.
- **Is it the rotary obstacle?** Same root, different symptom.
  - CoPE's count is monotone, so opposite moves cannot cancel through zero. They cancel through the table's period instead (alpha + beta = P). That holds already at init, so the rotary "no path through zero" barrier does not arise.
  - The price is that the period is pinned at P = 2 or 3 counts, because gates are below 1. A 32-cell ring must therefore be coded as 1/16-count offsets. That turns the ring wrap into a fine lattice on the move-gate difference, and there is a barrier between lattice points.
  - The tie between gate and content logit makes this worse: it pushes the move gates toward 0, which shrinks |g+ - g-| below the first lattice point.

## 1. Theory

**Setup (exact for this model).**
- **One layer, no other position signal.** q_i and k_j are functions of the token identity alone, through LN(token_emb) and then qkv. So every gate is a table: G[h, a, kappa] = sigmoid(q_{a,h} . k_{kappa,h} / 8) over head h, query type a and key type kappa (19 types). Its variance across queries of one type is exactly 0 (verified: `v0_verify_out.txt`, max difference 0.0).
- **The queries are action tokens.** The revisit target o_t is predicted at the position of a_t, the move just made. So there are two query types, +1 and -1. Observation-token queries never receive a loss.
- **The count.** Take an observation key o_s (s < t). N = t - s moves lie between it and the query: n+ of them +1 and n- of them -1, with D = n+ - n-. The count p_ij sums gates over o_s, a_{s+1}, o_{s+1}, ..., a_t. That is one move gate and one observation gate per step:

  p = n+ alpha_a + n- beta_a + sum_u eps_u,  alpha_a = G[a,+] + cbar_a,  beta_a = G[a,-] + cbar_a

  eps_u is each observation gate's deviation from its frequency-weighted mean cbar_a. Equivalently p = N sigma + D kappa / 16, with sigma = (alpha + beta) / 2 and kappa = 8 (alpha - beta) = 8 (g+ - g-).
- **Effective table.** For each (h, a), z_{a,h}(p) = q_{a,h} . e(p), linear between integer knots. The score of key j is the content logit plus z(p_ij).
- **The tie.** The gate is sigmoid(content logit). The same number decides how much key j is attended to and how much it advances the count.

**Exact solution family.** A table with integer period P >= 2 picks out the returns (D = 0 mod 32), and only them, iff three conditions hold. (A period-1 piecewise-linear table is constant.)
- (i) For D = 0, p = n (alpha + beta) must be 0 mod P for every n, so alpha + beta is in PZ. Gates lie in (0, 1), so alpha + beta < 4, which leaves **P = 2 (sigma = 1) or P = 3 (sigma = 1.5)**.
- (ii) For wrap returns, D = 32 gives 32 alpha = 0 mod P, so alpha = k P / 32.
- (iii) No false returns: D alpha must be nonzero mod P whenever D is not 0 mod 32, so **k is odd**.

With k = 16 + kappa / sigma, the lattice is:
- P = 2: kappa an odd integer, i.e. g+ - g- = odd/8. An even k is "aliased": D = 16 scores as a return.
- P = 3: kappa = 1.5 (k - 16) with k odd.

The four gates averaging 1/2, which is the init (q.k ~ 0), already gives sigma = 1.

**Precision.** The 32 cells sit P/32 apart on the period circle, so the count error must stay below P/64:
- (N/2) |Delta(alpha+beta)|, i.e. |Delta sigma| < P/(64 N), about 2e-4 at N = 128.
- |Delta k| < 1/64 = 0.016 at |D| = 32, and half that at |D| = 64.
- Observation-gate spread tau < P / (64 sqrt N), about 0.003.

**Parity is free.** With sigma = 1, N = D mod 2, so p mod 2 = D + D kappa/16 (mod 2).
- At kappa = 0 the count carries only the parity of D.
- At small non-integer kappa it separates nearby displacements (a local map). The wrap returns, though, land at 2 kappa mod 2, off the peak unless kappa is an integer.
- So between lattice points, local discrimination improves while wrap-only revisits are lost. That is a barrier.

**Against the rotary obstacle.** A rotary channel has its own increment per move and needs theta(+) + theta(-) = 0 mod 2 pi. A non-negative step can meet that only by wrapping, so it lacks the path through zero (MECHANISM).
- In CoPE, every Fourier component f of the table shares one count. The return condition (alpha + beta) f in Z with alpha + beta = 2 leaves only f = 1/2 (Nyquist).
- So CoPE is **one rotary channel per head, with cancellation by wrap given at init but the frequency pinned at Nyquist**. Monotonicity (gates >= 0) is the root of that pinning: signed gates would allow alpha + beta = 0 and a long-period, smooth table.
- What the monotone count costs is the ring-wrap lattice on g+ - g-, not the cancellation itself.

## 2. Predictions (`PREDICTIONS.txt`, written before measurement)

- **P1.** Gates do not vary across queries of one type.
- **P2.** Count heads have sigma = 1, in all runs.
- **P3.** SOLVED runs have a lattice head (predicted k = 15/17, precise to about 1e-3) for each query type. Unsolved runs are off the lattice.
- **P4.** Observation-gate spread is small in solved runs and larger in unsolved ones.
- **P5.** A period-2 table appears in all runs.
- **P6.** Off-lattice runs fail on wrap-only revisits and at long gaps. Solved runs are flat across strata.
- **P7.** An eval-only gate snap to alpha + beta = 2, alpha - beta = +/-1/8 raises accuracy toward 1.
- **P8.** A constructed landscape along delta has a barrier between a small-delta optimum and the lattice.

## 3. Probe checks (rule 9) -- `v0_verify*`, `v1_headclass_check*`

- **Construction (gates 0.9625 / 0.8375 / 0.1, period-2 table).** `increments()` reads alpha 1.0625, beta 0.9375 and kappa 1.000. `func_eval` gives 1.000 on every stratum (wrap-only n 1154).
- **Learned run (CoPEF s900).**
  - The table gates equal the gates inside `CoPE.forward` on a real sequence (max |diff| 0.0).
  - q.e equals the Z table (0.0).
  - `score()` reproduces eval.json (0.944294 = 0.944294).
  - The snap hook with the run's own gates is a no-op (bitwise accuracy and NLL).
- **Head classifier.** Constructed P2 k17 reads LATTICE, kappa 2 reads ALIASED, kappa 0.3 reads OFF, P3 k11 reads LATTICE, and flat or anti-phase tables read noTable.

## 4. Measurements

**M1/M2: regime per (run, query type)** (`m1_*`, `m2_*`).
- A head counts as a count head when |2 sigma - P| < 0.01 for P in {2, 3} and its table has a period-P share > 0.15, peaking at p = 0 mod P.
- It is LATTICE when k is within 0.05 of an odd integer, ALIASED when k is near an even integer, and OFF otherwise.
- A query type is NONE when it has no count head.
- Revisits come from the registered eval stream: 400 trajectories, about 38.8k revisits, of which 10.6% are wrap-only (every earlier visit of the cell is at unwrapped |D| >= 32).

| regime (per run x query type) | n | wrap-only acc mean (min-max) | local acc mean (min-max) |
|---|---|---|---|
| LATTICE | 20 | 0.962 (0.457-1.000) | 0.999 (0.997-1.000) |
| ALIASED | 6 | 0.797 (0.501-0.968) | 0.991 |
| OFF | 20 | 0.550 (0.431-0.937) | 0.992 (0.978-1.000) |
| NONE | 34 | 0.544 (0.441-0.848) | 0.810 (0.576-0.969) |

- **Regime pair vs run accuracy (`m4_out.txt`).**
  - LATTICE/LATTICE: n 6, 0.9945 (0.969-1.000).
  - ALIASED/LATTICE: n 2, 0.989.
  - LATTICE/OFF: n 4, 0.970.
  - OFF/OFF: n 5, 0.948.
  - NONE/OFF: n 4, 0.899.
  - NONE/NONE: n 14, 0.762 (0.58-0.87).
  - The 4 registered SOLVED runs are all LATTICE or ALIASED on both query types.
  - OFF/OFF accuracy follows from the theory's arithmetic: 0.894 x 0.995 + 0.106 x 0.55 = 0.948.
- **Lattice heads are as precise as derived.**
  - k = 17.00, 13.00, 15.00, 19.00, 21.00, 11.01, ... (P = 2), and k = 21.00, 11.00, 13.00 (P = 3, three heads in Hi s901 / s904). The only allowed period besides 2 occurs, at exact sigma = 1.5000.
  - |k - odd| <= 0.01, against a tolerance of 0.016. sigma = 1.0000 +/- 2e-4, and tau <= 0.001.
  - k = 15/17 accounts for 11 of 18 P = 2 lattice heads; any odd k occurs.
- **One LATTICE exception (Hi s902, query -1, wrap-only 0.457) is a precision case.** Its lattice head has |Delta k| = 0.007, and cleaning sigma and tau alone (S1) raises the run's wrap-only accuracy from 0.718 to 0.912.
- **Gap (`m1_gap_table_out.txt`).** OFF runs are flat up to N < 64 and dip at N >= 64 (0.72-0.92), where more |D| values lie in range for a generic kappa to alias. LATTICE runs are flat (>= 0.99). NONE runs decay with N: for example 0.905 / 0.699 / 0.485 / 0.508.

**M3: gate snap, P7 (`m3_*`).** Snaps were applied only in count heads; content logits, table and every other weight were unchanged.

| group | S0 own | S1 sigma, tau clean | S2 kappa -> +/-1 (as registered) | S3 nearest odd k |
|---|---|---|---|---|
| OFF, no NONE (n 11) | acc 0.957, wrap 0.629, NLL 0.153 | 0.958 / 0.628 / 0.148 | 0.982 / 0.897 / 0.063 (3/11 >= 0.99) | 0.980 / 0.896 / 0.068 (4/11) |
| has NONE (n 20) | 0.811 / 0.566 / 0.640 | unchanged | 0.811 / 0.595 | 0.814 / 0.598 |
| LATTICE/ALIASED only (n 9) | 0.991 / 0.938 / 0.022 | 0.994 / 0.960 | 0.977 (hurts 2 runs) | 0.981 |

- The straight path from the learned kappa to the S3 target was evaluated at 11 points.
- In 11/11 OFF runs, NLL rises inside the path and falls at the end, for example 0.166 -> 1.140 -> 0.029 for CoPEF s900 (`m3_summary_out.txt`).
- Snapping hurts where a run already combines an aliased head with a local head (Lo s905, GEN s701, Hi s905 under S2): the snap breaks a working two-head code.

**P8, constructed (`p8_landscape_out.txt`; table amplitude M 30).**
- Moving delta from 0 to 1/16: local accuracy 0.48 -> 0.999 by delta = 0.016. Wrap-only accuracy falls to 0.23-0.30 in between and is 1.000 only at 1/16.
- The obs-share loss is 1.53 at 0 and 1.26 at 0.004, rises to 1.4-1.6 in between, and is 0.03 at 1/16.
- `p8_landscape_out_v1_rawmass.txt` is a first version whose loss counted attention on action keys. It is superseded and kept for the record.

**M5-M8, post hoc (not predicted).**
- **Move gates collapse in OFF heads.** In 18/32 OFF heads both move gates are < 0.1, against 0/21 lattice heads (Fisher p 1e-5; heads are not independent).
  - Medians: OFF 0.084 / 0.008 against lattice 0.215 / 0.077, with cbar 0.95 (`m5`, `m7`).
  - Mechanism: the model counts observations and suppresses move keys through the tied content logit. |kappa| = 8 |g+ - g-| then stays below 1, and the smaller gate's sigmoid slope is about 0.008 (median), so it is nearly frozen.
- **The init does not predict the regime.** Rebuilt inits have sigma 0.996 +/- 0.068 and kappa sd 0.78 (`m4`). Median |kappa_init| is 0.48 for LATTICE heads and 0.70 for OFF heads. So the run does not end OFF because it started at kappa 0.
- **NONE heads** have |2 sigma - P| median 0.19, with only 1/68 below 0.01, and tau median 0.17 (43/68 > 0.1) (`m6`).
- **sigma locks** (`m8`):
  - At sigma = 1: 61/160 head x query entries, against 5 at init.
  - Unexplained by the theory: 13 entries lock at sigma = 0.5, 0.75 or 1.25, against 0 at init. They have tiny tau and mostly no period structure. Their role is unknown.
- **Table start.** Runs with a NONE query type: sinusoidal starts 9/24 against zero starts 11/16 (Fisher p 0.11). That is unmeasured; see Sec. 7.

## 5. Verdict per prediction

| | verdict |
|---|---|
| P1 | HOLDS (exact, by construction) |
| P2 | PARTLY. Every count head has sigma = 1.0000 (or 1.5000). But 20/40 runs have a query type with no count head (NONE), so "all runs" is wrong |
| P3 | HOLDS. Wrap-only accuracy tracks the regime (0.96 lattice vs 0.55 off-lattice), and the 4 SOLVED runs are lattice on both query types. "k = 15/17" holds for 11/18 heads; other odd k occur |
| P4 | PARTLY. tau <= 0.001 in lattice heads, but it is just as small in OFF heads. A large tau marks NONE, not off-lattice |
| P5 | FAILS for NONE runs (14/40 runs have no period head at all). Where a count locked, the table is period-2 (or period-3). That explains WRAP_COPE's "every arm ends period-2": it is the parity count, not a map |
| P6 | HOLDS for OFF (local 0.99, wrap-only 0.55, a dip at N >= 64) and for solved runs (flat). The "delta ~ 0" sub-case did not occur |
| P7 | HOLDS in direction, partly in size. As registered (S2), wrap-only accuracy goes 0.63 -> 0.90 and NLL 0.15 -> 0.06, but only 3/11 runs reach 0.99 |
| P8 | HOLDS (constructed). The learned straight paths show a barrier in 11/11 OFF runs |

## 6. Account

- **Representation.** One layer and one head suffice. The construction scores 1.000, 9/40 trained runs reach a lattice code (odd k, or aliased even k) on both query types, and snapping the gates fixes most of an OFF model without touching anything else.
- **Learning: two separate search failures.**
  - (1) **The count never locks** (NONE). The obs gates become identity-dependent, presumably through the tie, and sigma drifts off 1. This failure is not derived here.
  - (2) **The count locks at parity, but the move-gate difference stays off the odd/8 lattice** (OFF). This one is derived. Wrap-only revisits are the only data that need the lattice, and between lattice points they are scored at chance. The tie pushes the move gates toward 0, and a straight path to the lattice crosses higher loss.
- **"Bimodal seeds"** is the regime a seed's heads fall into, not noise around one solution. WRAP_COPE's table start acts on the table, not on the gates where the obstacle lies, which is consistent with its UNMEASURED verdicts.

## 7. Decisive cheap experiments (not run; each would need a pre-registration)

- **E1, the wrap lattice part.** Train CoPEF on a 1D line (no wrap) or a ring of 128 cells, at T = 128 and with the same seeds, beside ring 32.
  - Prediction: the OFF regime counts as solved, so SOLVED rises to about the non-NONE share (about 4-5/8). The NONE runs still fail, so not 8/8.
  - If it reaches 8/8, the wrap also causes NONE, and part (1) of the account is wrong.
  - Cost: 8 x 12 min per arm, about 1.6 GPU-h for 2 arms.
- **E2, the tie part.** Train the paper's own sep-keys CoPE (gates from a separate key projection; `papers/txt/cope.txt` Sec. 3) on ring 32.
  - Prediction: no move-gate collapse and no identity-dependent obs gates, so fewer NONE and fewer OFF runs.
  - Same cost.
- **E3 (optional).** Train with signed gates (tanh). This would test the monotone root: alpha + beta = 0 with a long-period table should solve like signed MapWM.

## 8. Open / caveats

- Only final checkpoints exist. The order "lock sigma, then kappa" is inferred, not observed.
- A barrier on the straight path does not exclude a co-adapting path.
- The snaps break the gate-content tie by hand.
- NONE's cause is not derived, and neither are the sigma = 0.5 / 0.75 / 1.25 locks.
- How signed MapWM meets the same ring wrap with 32 channels per head is inferred, not measured.
- The P3 / P7 thresholds (0.05 on k, 0.15 period share) were set here, not registered.
- One task, ring 32, n = 40 runs.

## Files

All files are in `docs/audits/2026-10-14/cope_diagnosis/`.

- Libraries: `lib_cope.py` (probes, snap hook) and `func_cope.py` (functional construction).
- Checks: `v0_verify.py` and `v1_headclass_check.py`.
- Measurements and analyses: `m1_measure.py`, `m1_gap_table.py`, `m2_per_query.py`, `m3_snap.py`, `m3_summary.py`, `m4_regimes.py`, `m5_gate_values.py`, `m6_none_heads.py`, `m7_init_vs_final.py`, `m8_sigma_locks.py`, `m9_fisher.py` and `p8_landscape.py`.
- Each script has a matching `*_out.txt` / `*_out.json`.
