# Claim map: "a learned path-integrated phase is a map or a clock" (2026-10-12)

Working thesis for a paper: **a learned additive (path-integrated) phase is a MAP when its increments cancel and a CLOCK
when they do not; it beats the token index exactly when the task needs a map, and on clock-like data the model learns a
clock that does no better than counting tokens.** Each claim below is mapped to its evidence, its status and its gap.
Status words as in `docs/WHERE_THINGS_STAND.md`: REG (pre-registered batch), citable (CLAUDE.md row), POST HOC, PILOT,
WITHDRAWN. Prior art: `docs/lit/LIT_CLOCK_MAP.md` (2026-10-12; summary in the section below).

## Claims

| # | claim | evidence (numbers) | status | file | gap |
|---|---|---|---|---|---|
| C1 | Signed increments make the interval sum a net displacement (same place at different times collides: a map); monotone ones make it a path length (a clock) | by construction | definition | `.claude-memory/project_clock_vs_map.md` | prior art (C1 is likely known; see lit) |
| C2 | Forcing a clock destroys a map task, at the training length | torus trained and tested at T = 1024: Abs - Signed -0.177 (p 0.0002), 0/8 vs 8/8 SOLVED; opposite moves cancel only when signed (0.06 vs 1.92-1.97). (The often-quoted crossover -0.280 is OUT OF DISTRIBUTION: trained at T = 128, read at 1024). Not softmax-specific: in a one-layer diagonal complex SSM on the ring |.| 0/8 vs signed 8/8 (G-3, 2026-10-23) | REG | `SIGN_MATCHED_RESULTS.md` (`SIGN_ABLATION.md` for the OOD figure); `G3_SSM_RESULTS.md` | replication of Grazzi / Sarrof / Selective RoPE in a new regime: say so |
| C3 | A clock costs nothing where a clock is enough, and a signed phase can act as one | k-back recency, trained at T = 1024: at the training length monotone arms 0.961-1.000 vs signed 1.000 (near ceiling; no registered contrast at this length); the registered -0.004 is read at T = 2048 (OUT OF DISTRIBUTION). The unconstrained signed model runs a clock there (C4). The index (RoPE) fails the task (0.234): it counts tokens, not content (C5) | REG, but the clock-side contrast is OOD or at ceiling | `RECENCY_RESULTS.md` | a matched-length clock-side contrast below ceiling; "recency NEEDS a clock" is withdrawn (a rewind solves it) |
| C4 | The free phase chooses map or clock per task | growth exponent alpha of the unconstrained arm 0.591 (torus) vs 0.967 (recency), +0.376, se 0.009; alpha is a re-description of opposition (r +0.9995) | REG (H2) | `RECENCY_RESULTS.md`, `LOCALISATION.md` | alpha is a statistic, not a lever |
| C5 | What a clock counts matters | recency at the training length (T = 1024), eval-only intervention on the trained signed model: a constant step on content tokens vs on all tokens, magnitude-matched, 0.783 vs 0.189 (+0.594, 8/8); the index arms (which count tokens) sit at 0.234 | citable; the matched pair was designed after a failed control (labelled post hoc in the file) | `RECENCY_GATE_ABLATION.md` | one task; an intervention, not trained arms |
| C6 | Where a map is needed, path integration beats the index at matched length | torus +0.243 (8/8, converged recipe); Match-Query 0.730 vs 0.154 (n = 5); Dyck at matched depth +0.353 / +0.130 / +0.045 / +0.024 at 1-4 layers; navigation in words 0.969 vs 0.505; on Dyck the accumulated distance tracks stack depth (r 0.755) not token distance (0.278) | REG / citable (Match-Query not pre-reg., n = 5; the depth probe POST HOC) | `PAPER2X2_RESULTS.md`, `MATCH_QUERY_SCALE.md`, `DYCK_MDEPTH_RESULTS.md`, `TEXTWORLD_RESULTS.md`, `CROSS_RESULTS.md` | baselines (LSTM / Mamba-like / deeper RoPE) on the torus |
| C7 | A within-task dial: as steps stop cancelling, the index stops needing depth | biased 1D walk: index needs 3 layers at p_plus 0.5 / 0.75 / 0.9 (0.9 knife-edge, gap 0.0115 vs 0.01), 1 layer at p_plus 1.0 (a clock); path 1 layer solves every cell | REG secondary; the registered primary fell in NO branch | `CANCEL_RESULTS.md` | the only within-task dial; a clean registered readout would strengthen it |
| C8 | On clock-like natural data the learned phase is a clock and buys nothing over the index | TinyStories: alpha 1.00, R^2(phase ~ token index) 0.999 (3/3); MapWM = RoPE (1.5053 vs 1.5050). Code (9 layers, 2048): alpha 0.99-1.02, R^2 0.994-0.998, 75-92% of each byte's step shared; path integration costs +0.0056 bpc at matched length (p 0.024, n = 3). Bach: alpha ~1.00. With the clock removed, TinyStories' content part shows no cancelling structure (real order = shuffled); code's shows a weak one (real order slower on 5/6) | POST HOC / PILOT; code n = 3 | `docs/audits/2026-10-12/alpha_language_out.txt`, `TINYSTORIES_PILOT_RESULTS.md`, `docs/audits/2026-10-05/code_step_table_out.txt`, `CODE_FULLVAL_RESULTS.md` | **confound: clock tasks use 6-9-layer models, map tasks 1 layer**; n = 3 |
| C9 | Map and clock can coexist in separate channels given rank | text world: 4/8 seeds run a per-step clock in 31-38 of 64 channels alongside the map, solving as well | REG secondary | `TEXTWORLD_RESULTS.md`; theory `THEORY_KERNEL.md` Thm 1 (scoped) | -- |
| C10 | When a map fails, it fails by a clock leaking into it | rank-D heads that fail end with a drifting (CLOCK) head (39/40 on SCORE_RANK; thresholds set post hoc); 1D ring: the two slow rank-1 seeds are the drifting ones; drift forbidden by construction solves ~2x faster | descriptive; REG secondaries | `SCORE_RANK_RESULTS.md`, `RANK_TESTS_RESULTS.md` | **rank = D is NOT hard in 1D (REG)**: no causal or general claim |

## Correction 2026-10-12 (RING_DIAL audit M1; applies to C1, C2, C8)
For a ROTARY phase the clock / map split is NOT representational: a non-negative step wraps mod 2 pi, so the second
token's step 2 pi / omega - Delta(+1) reproduces the signed code exactly. Signed and non-negative rotary arms can represent
the same codes; C2's matched-length cost is learnability (search within the budget), and C1's "monotone = clock" holds for
Howard's real, non-wrapping rates (decays, gates), not for rotations. This is itself a candidate contribution: in the
imaginary (rotary) slot, Howard's Cases II and III coincide representationally and differ only in how easily training
finds a map.

## Update 2026-10-12: the clock side (`CLOCK_SIDE_RESULTS.md`, registered)
On k-back recency (k_max 128, T = 1024, 300 epochs) every Case II contrast reads UNMEASURED: Abs / CARoPE / Pos - Signed
+0.08 / +0.17 / +0.08 (perm p 0.16-0.53), no arm converged (0/40 SOLVED; r(loss, acc) -0.99), monotone arms directionally
ahead; the index fails (RoPE - Signed -0.45, p 0.001). C3's gap stays open: a converged, below-ceiling clock-side contrast
needs a longer budget (or a registered speed readout).

## Update 2026-10-12: the ring dial (`RING_DIAL_RESULTS.md`, registered)
Within one task family and one model, a non-negative rotary step (which CAN represent the map by wrapping) learns it worse
where steps cancel: SOLVED 2/8 and 1/8 vs 8/8 at p_plus 0.5 / 0.75 (Fisher p 0.007 / 0.0014); UNMEASURED at 0.9. Abs does
not find the wrap solution (4-7% of channels cancel mod 2 pi vs 63-82% for Signed). No graded dial. This supports C2 as a
LEARNABILITY claim within a task, and the rotary-slot statement (Cases II and III coincide representationally).

## Update 2026-10-12: budget (`RING_BUDGET_RESULTS.md`, registered PARTIAL)
Three times the training does not help the non-negative rotary step on the ring (3/8 at 900 vs 3/8 at 300 epochs, same
seeds); no run, solved or not, finds the cancelling (wrapped) map (2-11% of channels vs 82% signed). The sign handicap is
not speed: training settles on a non-cancelling code that extrapolates worse.

## Update 2026-10-12: learning speed on the clock (`CLOCK_SPEED_RESULTS.md`, registered)
On k-back recency, the non-negative rotary step (Abs) learns faster than the signed one on 12/16 paired seeds (loss AUC
-0.197, exact sign-flip p 0.053: UNMEASURED by the registered rule); CARoPE no difference on fresh seeds. With the ring
companion (signed faster on 8/8 at every p, AUC +0.23 to +0.35, p 0.0078 each): an asymmetric learnability result -- clear
where a map is needed, a trend the other way where a content clock is needed. Do NOT claim a double dissociation.

## Update 2026-10-13: replication (`CLOCK_SPEED_REP_RESULTS.md`, registered)
On 16 fresh seeds S1 (Abs - Signed, loss AUC on recency) reads UNMEASURED again: -0.106 (p 0.19, 10/16), about half the first
batch's effect. Pooled post hoc -0.151 (p 0.018; pre-committed not to upgrade; no third batch). The clock side stays
unestablished: the learnability effect is ASYMMETRIC (clear for maps, small and unestablished for content clocks).

## Update 2026-10-13: C6 on the torus vs depth (`FACTORIAL_RESULTS.md`, registered, n = 4)
Index RoPE at 4 layers (0.860) or looped x4 (0.864) does not catch up with 1-layer path integration (1.000) on the torus at
T = 1024 (0/4 vs 4/4 SOLVED each); depth helps only short-gap revisits. Unlike Dyck (3-4 layers suffice), the torus needs
more than 4 layers of index attention.

## Update 2026-10-13: existence (`WRAP_EXISTENCE_RESULTS.md`, registered)
Rule 14 met for C2-as-learnability on the ring: an exact non-negative rotary solution (balanced construction, weights within
~5x the signed scale) is HELD under 300 epochs of training (8/8 SOLVED, 0/8 LOST, 1.000) while the same observation-free
class from scratch solves 0/8 vs signed 8/8 (p 0.0002). Secondary: the held model drifts off the cancelling code (87% -> 18%
of channels; T = 512 0.978 vs 0.998), and the wrapped code is more fragile to 1% parameter noise (0.80 vs 0.998, frozen).
Read: the solution exists and is stable in task terms; training does not find it; the exact map code is not an attractor
under the non-negative constraint.

## Update 2026-10-14: mechanism and generality (`MECHANISM_RESULTS.md`, `GENERALITY_RESULTS.md`, registered)
- MECHANISM (ring): the |.| constraint removes the path through zero. Signed steps cancel through zero even from |.|'s
  start (7/8; 98% opposite-sign channels, none wrap); a non-negative step started with move-angle sums past pi learns the map
  8/8 by wrapping, with no sign cost from that start (M6), vs 0/8 from a small start and 2/8 from a large sub-pi start.
  The exploratory toy's stronger "start, not constraint" did not transfer (signed escapes the |.| start in the transformer).
- TORUS existence: from scratch 0/8 vs 8/8; the exact non-negative solution is perturbed and re-found (8/8 SOLVED at the end),
  not held in place (6/8 LOST at lr 1e-4).
- Published methods: faithful CARoPE fails both tasks and is no better than index RoPE (torus worse, -0.055); faithful CoPE
  fails the ring (0/8) although it can represent it, and beats the index by +0.157 there.

## Update 2026-10-14: WRAP_COPE (`WRAP_COPE_RESULTS.md`, registered): UNMEASURED
A near-wrap (or any sinusoidal) start of CoPE's position table does not measurably rescue CoPE on the ring (Hi 0.913 2/8, Lo
0.867 1/8, slope-matched Lo 0.943 0/8, zeros 0.840 1/8; all contrasts unmeasured, bimodal seeds). Every arm's table ends
period-2 (power near pi 0.40-0.68): in CoPE the wrap is in a free table; the start lever is for monotone rotary steps.

## Update 2026-10-15: ACCOUNTS (`ACCOUNTS_RESULTS.md`, registered)
- T2 SUPPORTED: the signed escape from the |.| start is collective (frozen move latents 1/8 vs 7/8; free 82% vs frozen 15%
  opposite-sign channels); no cost from the default start.
- T1: the held non-negative model's used channels stay 98% cancelling (usage-weighted); "87% -> 18% drift" counted unused
  channels -- withdrawn; weight decay is not involved.
- T4: CoPE fails on a non-wrapping line too (1/16 vs ring 0/16; signed 8/8): the wrap is not CoPE's obstacle (precision is).

## Update 2026-10-15: TORUS_START (`TORUS_START_RESULTS.md`, registered)
The past-pi start does NOT rescue the non-negative step on the torus (0/8 vs 1/8; TS1 NO EFFECT DETECTED) and the same large
start drags the signed model to 2/8 (vs 8/8 default); most large-start runs still descending at 900 epochs. The start lever is
a ring result; the sign cost replicates (1/8 vs 8/8).

## Update 2026-10-15: GAUGE (`GAUGE_RESULTS.md`, registered): AS FALLS
- G2: the sign cost replicates on +1/-1 (|.| 1/8 vs signed 8/8, Fisher p 0.0014; fresh seeds 1300-1307).
- G1 (primary, +1/+2): UNMEASURED (|.| 2/8, 0.883 vs signed 5/8, 0.947; d -0.064 [-0.200, +0.075]); G3 |.| +1/+2 vs +1/-1
  2/8 vs 1/8, no difference. +1/+2 needs integration (RoPE 0.649, 0/8) and is harder for signed too (long-gap laps only).
- Signed learns a sign-coherent code on +1/+2 (0.92 same-sign channels vs 0.01 on +1/-1).
- Do not claim: "the cost follows the relative sign" or "no cost without an inverse pair" (Lemma 1's learnability corollary
  is not confirmed); the barrier-at-zero account does not explain |.|'s difficulty on +1/+2.

## Update 2026-10-16: ROBUST (`ROBUST_RESULTS.md`, registered): PARTLY UNMEASURED
- Cost holds: ring default 1/8 vs 8/8, wd 0 0/8 vs 8/8, lr /3 0/8 vs 6/8; torus 2 layers d 256 2/8 vs 8/8 (acc -0.032, p 0.0005).
- lr x3: UNMEASURED (4/8 vs 8/8, acc 0.987 vs 1.000). Torus at scale: accuracy gap a quarter of 1-layer (out-of-batch), 6/8
  |.| descending, SOLVED contrast fails at loss 0.1 (5/8 vs 8/8): part of the cost may be speed at scale.
- Do not claim: "robust to the recipe" or "the cost does not depend on scale"; say "holds under wd 0 and lr /3 and at 2 layers
  d 256; shrinks at lr x3 and in accuracy at scale".

## Update 2026-10-17: LRSWEEP (`LRSWEEP_RESULTS.md`, registered): COST AT BEST RATES
- Each arm at its own best of four rates: |.| 3e-3 2/8 vs signed 1e-3 8/8 (decision p 0.0070 SOLVED, 0.0057 accuracy;
  selection redone in the null); |.| optimum bracketed; 3/8 |.| runs descending (budget-scoped); ring only.
- Claim allowed: "the cost holds with each step at its own best of four learning rates (ring)". Not: "at any learning rate".

## Update 2026-10-20: DELTA_FREE (`DELTA_FREE_RESULTS.md`, registered): PEAK-PINNING SUPPORTED
- Freeing PoPE's phase offset raises the index arm +0.122 at T = 128 (8/8, p 0.0078), mostly on multi-step reversals
  (+0.180); the path arm stays at ceiling (-0.018, n.s.); interaction -0.140. Kernel now prefers offsets 3-8 (0% -> 50-100%).
- Allowed: "PoPE's pinned offset fixes the kernel peak at zero phase difference; that helps a path phase and hurts an index
  phase". Not: "freeing delta reverses the interaction" (MAJOR 5); path-arm OOD cost is UNMEASURED (P1 Holm 0.094).

## Update 2026-10-21: SROPE_TORUS (`SROPE_TORUS_RESULTS.md`, registered): FAILS THE MAP BUT USES ITS PHASE
- Faithful Selective RoPE (best LM config) 0.961, 0/8 (all descending) vs MapWM 8/8; beats index RoPE +0.236; zero-angle 0.507.
- State-tracking config 0.769, 0/8: its +1 rad bias keeps every learned step non-negative (analysis) -- fails like |.|.
- Do not claim "Selective RoPE cannot learn maps" (budget-scoped, every run descending); claim "does not learn the torus map
  within 900 epochs in its published configurations; observations step its phase almost as much as moves".

## Update 2026-10-23: RANKCLOCK / G-1 (`RANKCLOCK_RESULTS.md`, registered)
- E2 THE CLOCK IS THE OBSTACLE: drift-forbidden rank 2 (inverse pairs tied to exact negatives, observation steps 0) solves
  the torus 8/8 at 1.000 vs actions-only rank 2 1/8 (Fisher p 0.0014). Primary P PARTIAL: per-axis clock slot 3/8 vs 1/8.
- Allowed (supersedes the do-not-claim "rank = D is hard because of the clock" FOR THE TORUS ONLY): "on the torus the rank-2
  obstacle is the clock, not capacity". Not: "extra rank helps because it gives the clock an exit" (not supported at the
  registered bar); not "rank = D is hard because of the clock" in general (rank 1 learns the ring); the weight-decay-0 rank-4
  result (1/8) is post hoc.

## Update 2026-10-23: G-3 SSM (`G3_SSM_RESULTS.md`, registered): SIGN COST IN THE SSM
- One-layer diagonal complex SSM with an input-dependent rotation, no decay (= unnormalised rotary linear attention), the
  attention ring task and recipe, lr 3e-3, 300 epochs: |.| 0/8 (0.821, below retrace 0.864) vs signed 8/8 (1.000), Fisher and
  permutation p 0.0002; softplus 1/8 (0.935; Holm-surviving; 5/7 unsolved still descending); exp 0/8 but 8/8 RUNAWAY.
- Rule 14: exact |.| / softplus / exp constructions (frozen = source 8/8); |.| perturbed and re-found at 3e-3 (8/8 SOLVED,
  peaks 0.32-0.68), at 1e-3 PERTURBED AND RE-FOUND by the registered 10x clause and HELD by the secondary E2' (peaks
  0.033-0.127). The registered SEARCH COST joint did not form.
- Allowed: "the sign cost is not specific to softmax attention: it appears in a one-layer diagonal complex SSM
  (unnormalised rotary linear attention)". Not: "exp fails in the SSM" (instability at 3e-3, A2 not run); "the SSM holds
  the non-negative solution" (re-found at 3e-3; held only on E2' at 1e-3); anything about Mamba-faithful SSMs (no decay,
  no gate, Delta in the phase only), m = 8, or other rates.

## Update 2026-10-24: SIGNSEP (`SIGNSEP_RESULTS.md`, registered): UNMEASURED
- A 2x2 from one common |.| start (ring, observation-step-free rank 4, 300 epochs) to separate the two accounts SIGNROUTE
  left together. lr 3e-3 (gates pass: signed 8/8, |.| 3/8): W = |.| plus a learnable per-move wrap offset (a collective
  route to a point where every channel cancels by wrapping; barrier kept) 0/8, A-LIKE (vs signed Fisher p 0.0002); F =
  signed with frozen move latents (no barrier, no collective route) 4/8, UNMEASURED; control P (W's class from past pi) 7/8,
  every run ending on the collective wrap sheet (tau_sum 2.00). lr 1e-3 UNREADABLE (signed anchor 5/8 < 6; from this start
  signed solved 7/8 in MECHANISM and in ACCOUNTS T2).
- Allowed: "from a small start a collective route to the wrap does not get the non-negative step past pi (registered
  half-reading BARRIER BLOCKS, ring, lr 3e-3)". Not: "the barrier is THE obstacle" or "the route is irrelevant" (F
  unmeasured; the 2x2's registered reading is UNMEASURED); nothing at 1e-3, on the torus, or from other starts. The
  mechanism of the sign cost stays an account.

## Update 2026-10-24: G-4 TORUS_LRSWEEP (`TORUS_LRSWEEP_RESULTS.md`, registered): COST AT BEST RATES
- Torus, T = 1024, observation-step-free rank 4, 1 layer, 900 epochs, lr 1e-3 / 3e-3 / 1e-2 with AdamW wd 0.05 coupled:
  each step at its own best rate, |.| 3e-3 0/8 (0.890) vs signed 1e-3 8/8 (1.000), decision p 0.0002 (SOLVED) / 0.0095
  (accuracy; selection redone in the null). |.| solves 0/8 at every rate; 5/8 at its best rate still descending
  (budget-scoped). Signed's best is the grid's lowest rate, at ceiling; at 3e-3 and 1e-2 signed is 0/8, all descending.
- Allowed: "the cost holds with each step at its own best learning rate on the ring (four rates) and on the T = 1024 torus
  (three rates, 1e-3 to 1e-2)". The LRSWEEP "ring only" qualifier is closed. Not: "at any learning rate", other optimisers
  or schedules, or a torus decay control (at 3e-3 the coupled decay is 3x the default's; stage 2 not triggered).

## Update 2026-10-24: RANKRECIPE (`RANKRECIPE_RESULTS.md`, registered): RANK D+1 NEEDED SURVIVES A THREE-RECIPE SEARCH
- Shared rank 2 0/8 at lr 1e-3 / wd 0.05, lr 3e-3, wd 0.2 (0/24) vs shared rank 4 8/8 at the default; best rank-2 recipe
  (lr 3e-3) vs rank 4: 0/8 vs 8/8, 0.910 vs 1.000, decision p 0.0002 both routes. 12/24 rank-2 runs still descending.
- Allowed: "rank D + 1 (with decay) is needed on the T = 1024 torus, and rank D fails at the best of three recipes". Not:
  "no recipe rescues rank 2" (no lower-decay or combined arm; both alternatives raise the total decay).

## Do not claim (withdrawn or failed)
- Clock vs map predicts when PoPE's encoding helps: both within-task tests failed (`T2_RESULTS.md`; CLAUDE.md withdrawn).
- Recency needs a clock (a signed rewind solves it).
- alpha causes out-of-distribution degradation (route not established; low-frequency account refuted).
- The forget gate is a clock (untested); the InEKF wrap bounds the accumulator (false).
- Rank = D is hard because of the clock (fails in 1D).
- A non-negative rotary phase cannot represent a map (it can, by wrapping; the sign result is learnability).
- "The wrapped solution is unstable / fragile under training" (an ill-conditioned construction artefact; the balanced one is held on the ring, re-found on the torus).
- "The held non-negative model drifts off the cancelling code" (unused channels; ACCOUNTS T1).
- "Weight decay drives the drift" and "the wrap is CoPE's obstacle" (ACCOUNTS T1, T4).
- "Initialise a monotone rotary step near the wrap" as a general design rule (ring only; fails on the torus).
- "A near-wrap start rescues CoPE" (unmeasured; WRAP_COPE).
- "The sign cost is only the start, not the constraint" (toy only; in the transformer signed escapes the |.| start through zero).
- "SIGNSEP shows the sign cost is the wrap barrier" (only the half-reading BARRIER BLOCKS; the registered reading is UNMEASURED).

## Gaps, cheapest first
1. **Prior art** (`docs/lit/LIT_CLOCK_MAP.md`, agent running): which of C1-C5 are published.
2. **The cross-task confound in C8.** Map tasks are 1-layer, clock tasks 6-9 layers. A same-size comparison: MapWM vs RoPE
   at the TinyStories size on a map task (or 1-layer on a clock task), or a registered within-task dial (C7 with a clean
   readout). GPU, ~1-3 h.
3. **Baselines for C6** (the trimmed baseline table, ~8-10 h).
4. Statistical status of C8 (n = 3, pilot): either run as a registered small batch or present as descriptive.

## Prior art (`docs/lit/LIT_CLOCK_MAP.md`, 2026-10-12; two claims spot-checked in `papers/txt/`)
- **C1 is published**: Howard et al. 2014 (J Neurosci 34:4692) -- one integrator with a constant rate (time), a speed rate
  (path length: different routes to one place differ) and a signed velocity rate ("like a bead on a wire": the same code
  for a position). The clock / map table is Howard's three cases applied to a rotary phase. Cite as the frame.
- **C2 (sign) is published as theory and in architectures**: Sarrof 2024, Grazzi 2025, Khavari 2025 (proofs); Selective
  RoPE and Mamba-3 (signed content-dependent rotation); MapFormer itself (CoPE "cannot represent ... its inverse ...
  even in 1D", `papers/txt/mapformer.txt:586`; its Fig. 9). Ours: the sign ablated on its own at matched length on
  navigation -- a replication in a new regime.
- **C3 "a signed phase can emulate a clock" is trivial** (the signed range contains the monotone one); keep only as a remark.
- **C6 depth substitution is known qualitatively** (Liu 2023 prefix sums, Yao 2021, PaTH, MapFormer). Ours: the matched-depth
  exchange rate (1 path layer ~ 3 attention layers) and navigation told in words.
- **C8 must be scoped**: CoPE (monotone, content-counting) beats RoPE in distribution on language and code perplexity
  (`papers/txt/cope.txt` Table 5); Grazzi's signed DeltaNet helps code and math. So "on clock-like data a learned phase
  buys nothing" is at OUR scale only, and our TinyStories "sentence clock" is close to what CoPE was built to learn.
- **Drop** "beats the index EXACTLY when the task needs a map" and "no better than counting tokens" as general claims.
- **"A monotone phase cannot be a map"** holds per channel in a rotary score, not for a population (two monotone channels
  read as a ratio give net displacement: Howard's own construction).
- **Plausibly new (empirical)**: the matched-length sign ablation on navigation; the free phase's per-task growth exponent
  (0.59 vs 0.97) and its use on language checkpoints; the step-table decompositions on code / Bach / TinyStories; the
  matched-depth exchange rate; navigation in words. **Plausibly new as a statement, may be called obvious**: Howard's
  rate alpha(t) = the selective-SSM step = MapFormer's content-dependent step, with sign separating his speed and
  velocity cases.
