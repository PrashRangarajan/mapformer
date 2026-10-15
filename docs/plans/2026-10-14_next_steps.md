# Next steps for the map-vs-clock paper (2026-10-14, advisor review)

Read before writing: project_state, DRAFT_workshop (draft 2), CLAIM_MAP, OUTLINE, THEORY_REVIEW, COPE_DIAGNOSIS, the eleven
results files, ACCOUNTS_PREREG + Amendment 1, `runs/accounts/t3/summary.txt`. ACCOUNTS T1/T2/T4 were still running at 20:30
(no `.accounts_done`); everything about them below is conditional. No GPU job was launched for this review.

## A. Where the paper stands

**Solid (registered, n = 8, matched length, theory-backed).** (1) Sign cost: torus 0/8 vs 8/8 (SIGN_MATCHED), ring 2/8 and 1/8 vs
8/8 (RING_DIAL), 3x budget 3/8 vs 3/8 (RING_BUDGET), observation-step-free ring and torus 0/8 vs 8/8 (WRAP_EXISTENCE E3,
GENERALITY E3). (2) Existence: an exact construction (Prop. 1), frozen = source, held on the ring and re-found on the torus.
(3) The start x constraint 2x2 on the ring (MECHANISM): |.| small 0/8, signed small same-direction 7/8, |.| past pi 8/8, signed
past pi 8/8, plus sum-not-size 8/8 vs 2/8. (4) CoPE representable but 0/8; CARoPE provably cannot cancel (Prop. 6). (5) Depth
factorial, n = 4, complete separations. The exact theory (Props. 1, 3-6, rescaling Prop. 2) is real but short.

**Accounts, not derivations.** Why a past-pi start flows to the wrap and why signed escapes the |.| start (per-channel gradient
flow reproduces neither; "collective" is T2's question); the drift 87% -> 18% (T1); the CoPE lattice (post hoc, T4); depth as a
precision problem (Liu et al. read at abstract level only); the clock-side block-vs-delay asymmetry.

**What a reviewer will hit first.**
1. *The representational half is known.* Mamba-3 (Sec. 3.2, `papers/txt/mamba3.txt:576-578`) writes parity as R(pi x_t), a
   non-negative rotation that inverts itself; Grazzi et al. implement Z_m with rotations in one layer. "A non-negative rotary
   step can represent the inverse by wrapping" is the cyclic-group fact -1 = m - 1. Not in our lit notes, not in the draft.
   Cite it and claim only learnability, the mechanism and the regime. This *helps*: Grazzi's theorem shows real non-negative
   eigenvalues cannot track state; in the rotary slot that theorem does not apply, yet the same advice (allow sign) still holds,
   for an optimisation reason. That is the paper's one-sentence contribution.
2. *"Of course |s| cannot cross zero."* Sign-constrained reparameterisations (Hadamard / w = u^2 - v^2 in the implicit-bias
   literature) are known to be unable to change sign; "start near a solution and it works" is unsurprising. The defence is
   the 2x2 (the cost is the constraint, not the start: signed escapes the same start 7/8) and the exact statement below (B, T-1)
   that the constraint removes *only* relative signs. Lead with the 2x2, not with the narrative.
3. *The title over-claims and the paper's own M2 contradicts it*: a non-negative step does learn the map from a past-pi start
   (8/8). Retitle, e.g. "Sign in rotary position is a matter of learnability: non-negative phase steps can represent a map but
   learn it only from past the wrap". "Within 300-900 epochs of AdamW" beside every "does not learn".
4. *The mechanism is ring-only*, the headline task is the torus. The single experiment worth running before submission (B1).
5. *4.1 reads as capacity* and cites Liu et al. unread. Either read Liu first-hand and frame as learnability, or shrink 4.1 to a
   motivating paragraph. The draft's "to verify" cannot go out.
6. *SOLVED is a training-loss threshold* and the ring accuracy gap is small (-0.03). Add a readout that shows "not a map":
   wrap-only / long-gap strata for the ring |.| runs (exist for the torus, not the ring) and the 4T drop.

**Cut.** 4.6 natural text (pilot, n = 3, depth-confounded) -> one sentence in Limitations. The CoPE diagnosis -> appendix;
main text keeps the registered CoPE facts and two sentences. The toy (appendix E) -> three lines or nothing: it fails even
the signed default (2/8 with the observation target, 6/12 before) and did not escape the |.| start under either target (T3
0/8), so it models nothing the transformer shows except the |.| failure. Prop. 2-generic (rank failure with > r stepping
tokens): state only the scoped Prop. 1 and say the observation-step-free arms carry the claim; no unproven sketch.

## B. Next work, ranked by value / cost

**Theory (no GPU; needed for the workshop).**
- **T-1 Relative-sign theorem (new, exact, half a page).** Per channel, flipping the sign of every step and reflecting that
  channel's q and k plane (J = diag(1, -1); J R(phi) J = R(-phi)) leaves the model unchanged; omega's own sign is already free in
  the code (`model.py:111`). So |.| does not restrict the sign of a channel, only the *relative* signs of the tokens that step in
  it: the |.| class is "all stepping tokens rotate the same way in each channel". Corollaries: (a) with one stepping token
  (parity, mod-m counting) |.| costs nothing -- consistent with Mamba-3/Grazzi and a prediction; (b) for inverse pairs of
  distinct tokens the excluded set is exactly the opposite-sign orthant, which contains every through-zero cancelling code
  (Prop. 3 restated as a theorem, not a picture); (c) the sign question is about inverse pairs, not about wrapping.
- **T-2 Appendix proofs**: Props. 1 (with the cyclic-group citation), 2 (rescaling, 2 pi eps n), 3-5, 6, and the propagator
  identity (cite Mamba-3 Prop. 3). Label the collective-route and CoPE-lattice material as accounts.
- **T-3 First-order certificate on the real runs (CPU, eval-only, predictions written first).** At the AO / AO_absmag endpoints
  (`runs/wrap_mech/p0`), map the |.| model to the signed class and compute dL/da per (channel, move): the account predicts
  channels pinned at a ~ 0 with dL/da > 0, i.e. constrained-stationary points of the |.| problem with a descent direction that
  crosses zero in the signed one; none at the AO_high endpoints. This turns "the constraint removes the path through zero" from
  a picture into a measured statement on the trained models. ~half a day of agent time.
- **T-4 (stronger venue, conditional on T2 SUPPORTED)**: the reduced dynamics along the latent route z(-) -> -z(+), averaged over
  the omega ladder (the per-channel potential is rugged; the frequency average may not be). Plateau-escape, not a barrier.

**CPU analyses (needed for the workshop, minutes each).** Ring strata (wrap-only, lag >= 64) for RING_DIAL / RING_BUDGET /
MECHANISM |.| runs; Fig. 3 (per channel, initial -> final (a(+1), a(-1)) for the four starts, over the cut-circle schematic).
Skip the CARoPE mean-rate test and the factorial gap curves unless a reviewer asks.

**Literature (agent, ~1 h).** First-hand: Liu et al. 2023 (does the 2-layer gridworld construction cover retrieval of an
observation at a revisited cell?); Mamba-3 Sec. 3.2 and Grazzi's Z_m rotations; sign-constrained reparameterisation dynamics
(Hadamard parameterisation); phase/frequency initialisation. Write into `docs/lit/`.

**GPU.**
- **B1 TORUS_START (workshop, ~3-4 h wall, 16 torus runs x 17-40 min).** AbsAO4_r4 from a past-pi start (each opposite pair's
  angles summing past pi per channel, MECHANISM's U(1.7, 4.5)) vs AbsAO4_r4 default, in-batch, n = 8, GENERALITY's torus recipe
  and trainer. Optional third arm SignedAO4 from the same start (+8 runs, +1.5 h): M6 on the torus. Prediction RESCUES. Either
  outcome is publishable: RESCUES puts the mechanism and the design rule on the headline task; NO RESCUE scopes it to 1D and is
  what Prop. 2 half-predicts (the torus wrapped code is ~34x more curvature-sensitive and was knocked out in E2). Prereg + rule-29
  check + a pilot on an outside seed.
- **Stronger venue only (~12-16 h wall total).** (i) Recipe robustness on the ring, Abs vs Signed under wd 0 and lr x3 (32 ring
  runs, ~2 h): "not an AdamW artefact" is the first question for a learnability claim. (ii) Scale: sign on the torus at 2 layers,
  d 256 (16 runs, ~6-8 h). (iii) One recurrent baseline (LSTM) and factorial seeds to n = 8 only if 4.1 stays in the main text.
  (iv) Optional: corollary T-1(a) on a self-inverse task (no sign cost predicted; ~1.5 h) -- the theorem may suffice.

## C. ACCOUNTS branches (fold in; no follow-up GPU in any branch)

**T1 (weight decay vs drift, usage-weighted).** First read AO_bal_wd's usage-weighted share at the end. If it is high (>~0.7),
the "87% -> 18%" drift was in unused channels: delete the drift sentence and "not an attractor" from the draft whatever D1 says.
- WEIGHT DECAY DRIVES THE DRIFT, D2 no effect: Prop. 2 becomes a tested account; the design rule becomes "start past the wrap, no
  weight decay on the step map"; add a wd-0 AO4_high arm to B1 as a declared secondary (+8 runs) only if B1 is not yet registered.
- NO EFFECT DETECTED: the toy's wd result did not transfer; keep Prop. 2 as the fragility derivation only; drift goes to an
  appendix secondary, uninterpreted. Stop.
- UNMEASURED, or D2 also fires: report; "rescaling-invariance is the difference" is not supported. Stop.

**T2 (frozen move latents).** SUPPORTED: one sentence plus T-4 for the stronger venue. NOT SUPPORTED: the per-channel route
suffices in the transformer; drop "collective" and the latent-route paragraph -- the paper gets simpler (T-1 + Props. 3-5 carry
it). AMBIGUOUS / UNREADABLE: drop the route paragraph, say "the escape route is not identified". Stop in every branch.

**T3 (done).** The observation target is not the toy/transformer difference. Toy out of the main text; stop toy work.

**T4 (CoPE on a line).** Check RoPE on the line first: if index RoPE also solves the line, the line does not need a map and
"LINE EASIER" says little. LINE EASIER with RoPE failing: one sentence in the main text (CoPE's monotone count cancels for free;
its obstacle is the wrap lattice, a different search failure), details in appendix H. NO EFFECT / RING EASIER: the lattice is
not the main obstacle (the NONE regime dominates); appendix H reports the diagnosis as inconclusive. Stop CoPE in every branch
(no sep-keys, no tanh gates, no torus CoPE).

## D. Writing path (workshop submittable in ~7-10 working days)

1. Fold ACCOUNTS (day 1). 2. T-1, T-2 into a half-page Sec. 2 "Theory" box + appendix (days 1-3). 3. Lit check, rewrite Related
work with Mamba-3 parity / Grazzi Z_m / reparameterisation (day 2). 4. B1 prereg -> check -> run (days 2-4, GPU ~4 h). 5. T-3, ring
strata, Fig. 3 (days 3-4). 6. Restructure: Sec. 4 = sign cost -> existence -> 2x2 mechanism (+ torus B1) -> published methods; 4.1
depth shortened to a motivating paragraph; 4.6 cut. 7. Rule-28 end-to-end read plus an independent mock-review agent (day 6).
Venue: confirm a workshop deadline before day 1 (unverified here). For the stronger venue, TMLR fits best: rolling, and it
judges claims by their support, not by novelty -- the honest, scoped style of this project is what it rewards. Add B (stronger
venue) items and 2-3 weeks of writing.

## E. Stop

Clock-side batches (twice unmeasured; no third batch was pre-committed). Natural-text / TinyStories / code alpha analyses for
this paper. All CoPE follow-ups after T4. The toy. Further continuation variants (lr, gauges) of the existence test. Chasing
the mobility confound (structural: any start nearer the wrap is a larger-angle start; state it in Limitations). More post hoc
per-channel readouts of MECHANISM beyond T-3 and Fig. 3. The rank line, RANK_NOWRAP, TW_AMBIG, TW_LANDMARK: a different
paper; parked. Rule: any new GPU job must name the reviewer objection (A1-A6) it answers.

## Action list

| # | what | why | cost | prereg |
|---|---|---|---|---|
| 1 | Fold T1/T2/T4 per Sec. C; delete drift claims if usage-weighted share is high | registered results; removes a likely artefact | CPU, half a day | no |
| 2 | Retitle; fix abstract ("within budgets", past-pi rescue) | title contradicted by M2 | text | no |
| 3 | Write T-1 relative-sign theorem + corollaries | exact; answers "obviously |s| can't cross zero" and the parity objection | 1 day | no |
| 4 | Appendix proofs Props. 1-6, propagator identity; drop Prop. 2-generic | "theory for everything"; removes an unproven sketch | 1-2 days | no |
| 5 | Lit: Mamba-3 parity, Grazzi Z_m, Liu first-hand, sign reparameterisation | representational half is prior art; 4.1 cites an unread paper | ~1 h agent | no |
| 6 | T-3 first-order certificate at AO endpoints (predictions first) | makes the mechanism a measured statement | half a day CPU | no (post hoc, labelled) |
| 7 | Ring strata for |.| runs; Fig. 3 | "not a map" beyond a loss threshold; the missing figure | CPU, hours | no |
| 8 | B1 TORUS_START: AbsAO4 past-pi vs default, n = 8 (+ SignedAO4 past-pi optional) | mechanism on the headline task | 16-24 torus runs, ~3-5 h wall | yes |
| 9 | Restructure draft; shrink 4.1, cut 4.6, CoPE/toy to appendix | focus | 1-2 days | no |
| 10 | Rule-28 read + mock-review agent; confirm venue/deadline | accretion; external check | 1 day | no |
| 11 | (stronger venue) ring recipe robustness (wd 0, lr x3) | learnability not an AdamW artefact | 32 ring runs, ~2 h | yes |
| 12 | (stronger venue) torus sign at 2 layers, d 256 | scale | 16 runs, ~6-8 h | yes |
| 13 | (stronger venue) T-4 reduced dynamics, if T2 SUPPORTED; LSTM baseline / factorial n = 8 if 4.1 stays | derivation replacing an account; baselines | theory days; ~4-8 h GPU | GPU parts yes |
