# GENERALITY: independent code check (rule 29), 2026-10-14

Scope: `GENERALITY_PREREG.md`, `model_mono_baselines.py`, `model_wrap_exist.py` (4-action generalisation),
`build_torus_exist.py`, `train_variant.py` (new VARIANT_MAP entries), `train_ring_variant.py`, `analyze_generality.py`,
`run_generality.sh` at HEAD 47a766b. I read the pilot data (`runs/torus_exist_pilot`, `runs/generality_pilot`, seed 690)
without writing to it. I did not look at any batch result (none exist). No repo file was edited except this one. Check
scripts and the smoke are in the session scratchpad (`chk/c1-c4.py`, `smoke/`). They ran on CPU or with tiny jobs on cuda:1.

Verdict: the code is correct. The constructions are exact, the 2-action path is bit-identical, the plumbing works end to end,
and every label can be reached. Three points need an amendment before launch, and none of them needs code changes:
- B's premise is false for CoPE (finding 1).
- On the pilot seed, the E2 LOST rule already fires at lr 1e-4 even though the run ends SOLVED (finding 2).
- B for CARoPEF cannot fail (finding 3).

## Findings

**1. BLOCKER (prereg premise): CoPE CAN represent ring cancellation.** "A count only grows" ignores the learned position table.
- With per-token gates g(+1) = 0.9625, g(-1) = 0.8375 and g(obs) = 0.1, each (move, obs) pair adds alpha = 17/16 or
  beta = 15/16 to p_ij, so alpha + beta = 2 exactly.
- So p_ij mod 2 = (17/16) x net displacement mod 2. That value is distinct for every displacement mod 32 and 0 at a return.
- A period-2 table z(2k) = +M, z(2k+1) = -M survives the linear interpolation, because its knots are integers.
- Test: I put this construction through the repo's `CoPE` module (Listing 1) on 200 real `GridWorldCancel` trajectories
  (p_plus 0.5, T = 128, M = 200, attention = content logit + CoPE logit, as in `CoPEAttn`). Revisit accuracy was **1.0000**
  (n = 19,391), with 0.991 of the attention mass on the correct observation (`chk/c3.py`).
- Two simplifications: the gates depend on the key only, and the readout is argmax of attention mass. Both are reachable
  through q/k/v in one layer.
- This is the same mechanism as the rotary wrap (ring_dial_audit M1), with period 2 instead of 2 pi.
- Consequence: on the ring, the only task CoPEF runs on, B-CoPE is a learnability test, not "in practice, not learnability".
  The prediction's stated reason "(they cannot represent cancellation)" is wrong for CoPE.
- The torus is different but untested. A homomorphism from Z_64^2 to the circle cannot be injective, so one count per head
  cannot encode a 2D cell, and one layer cannot intersect heads. CoPE on the torus is probably unrepresentable, which is moot
  because it is not run there.

Fix (Amendment 1):
- Restate B: CARoPE cannot represent cancellation (finding 3). CoPE can, through a periodic position table, so B-CoPE is a
  Case-II-with-wrap learnability test, like Abs_r4.
- Restate the prediction's reason.
- Optionally run a rule-14 existence step for CoPE (warm-start the construction above, frozen and trainable). It is not needed
  for launch.

**2. MAJOR: at lr 1e-4 the LOST rule reduces to "the loss ever exceeded 0.05". The pilot seed fires it, and the resulting
label is not comparable with the ring's.**
- Pilot s690, printed every 5 epochs: AO_bal (lr 1e-4) reached a max of **0.1195** (0.059 / 0.109 / 0.120 at epochs 10 / 15
  / 35, and 0.0017-0.014 from epoch 150). SO_bal's max was 0.0028.
- 10 x 0.0028 = 0.028 < 0.05, so the relative branch's floor 0.05 decides. AO_bal is LOST on s690, then re-solves.
- Synthetic replay through `analyze_generality.e2` (`chk/c4.py`): 8 seeds of that shape (max 0.12, ending at 0.005, acc
  0.999 vs 0.9996) give **NOT HELD** (LOST 8/8 vs 0/8, Fisher not-LOST p 0.0002), with SOLVED 8/8 and d -0.0006.
- The same AO_bal curves against a control whose max is 0.06 give **HELD**.
- The ring E2 (HELD) had AO_bal max 0.024-0.154 against control maxes of 0.015-0.112 at lr 1e-3. A torus "NOT HELD" next to a
  ring "HELD" would therefore partly reflect the lower control lr (quieter control), not a different phenomenon.

Fix: decide before launch, and write the reading down. Either:
- (a) keep the rule and pre-state that on the torus LOST means "left the solved region (loss > 0.05) at some epoch", and that
  NOT HELD-by-LOST with SOLVED >= n_s - 1 at the end reads "perturbed and re-found, not held in place". Also report the
  per-seed peak loss and the epoch of re-entry. Or:
- (b) primary LOST = absolute 0.3 only, plus "excursion > 0.05" as a declared secondary.

Also state that the ring and torus E2 labels are not directly comparable, because their continuation lrs differ (1e-3 vs
1e-4).

**3. MAJOR: B for CARoPEF cannot fail. Its only reachable outcomes are MONOTONE FAILS or UNREADABLE.**
- NO DIFFERENCE needs CARoPEF SOLVED >= Signed_r4 - 1, roughly 7/8 at loss < 0.05.
- CARoPEF cannot represent a map: every per-token step is in (0, 1] rad, so a 2-move loop advances each channel by (0, 4),
  never 0 mod 2 pi. One-layer index RoPE stops at 0.719 (torus) and 0.717 (ring).
- At init, CARoPEF IS index RoPE (`inv_freq` vs f^i: 9e-8).
- So "MONOTONE FAILS" for CARoPE is a sanity check (compare TW_STATECHANGE's RoPE row), not evidence about published methods
  "in practice".
- The informative contrast is CARoPEF vs index RoPE: does content dependence buy anything when it is monotone? The batch
  only has it cross-batch (FACTORIAL 0.719, at another seed set).

Fix: label B-CARoPE a sanity check in the prereg. If the write-up will claim "no better than index", add an in-batch `RoPE`
arm. On the ring that is 8 x 300 epochs, about 12 min each (`train_ring_variant` already accepts `RoPE`). On the torus it is
8 x 900 epochs. Register the CARoPEF - RoPE contrast.

**4. MINOR: no dropout re-score secondary, and `rescore_hook` would skip CoPEF.**
- CLAUDE.md: eval mode under-reports one-layer runs below ceiling (up to +0.17). Here the below-ceiling arms are AO4,
  CARoPEF and CoPEF, so the effect pushes the accuracy branches of E3 and B (perm p, d <= -0.05) toward the predicted label.
  The SOLVED branches are unaffected.
- WRAP_EXISTENCE declared the re-score as a secondary. This prereg does not.
- `rescore_hook.KNOWN` lacks `mapformer.model_mono_baselines.CoPEAttn`. Its o_proj input is dropout(softmax) @ V, so it is
  eligible. It would be skipped silently, apart from the stderr list.

Fix: declare the re-score secondary, add CoPEAttn to KNOWN, and report whether any label changes.

**5. MINOR: why lr 1e-4 is chosen, and omega sensitivity.**
- The primary lr was chosen after seeing both pilot arms (AO_bal 0.3-0.5 at 1e-3 vs 0.03-0.12 at 1e-4). It is disclosed and
  defensible: the source's cosine floor is 1e-4. Keep the 1e-3 pair as descriptive, as registered.
- The non-negative phase grows linearly in T, so d(theta)/d(omega) = cumsum(step) is about sqrt(T) times the signed random
  walk's. At T = 1024 (versus the ring's 128), an Adam-sized omega step moves AO_bal's late phases about 10x more than
  SO_bal's.
- The constructions set every omega' = 1 on the pilot (trained |omega| max 0.80), so the frequency ladder is collapsed in
  both arms. The two arms are matched, but the torus E2 is the more omega-sensitive test.

Fix: optional declared secondary: relative drift of omega vs W_out over the continuation, for each arm.

**6. MINOR: the phase-2 wait exits as soon as the checkpoint file appears, not when SO4 exits.**
`while [ ! -f "$SRC" ]` can succeed while `torch.save` is still writing. The window is milliseconds against 60 s polls,
so a hit is unlikely but possible, and `build_torus_exist` would then load a truncated file.

Fix: `while running "$R/torus/SO4/p0/SignedAO4_r4_s$S"; do sleep 60; done; [ -f "$SRC" ] || drv_fail ...`.

**7. MINOR: the "clean" check only logs.** `git status ... && echo DIRTY` is written to the log, but launch is not refused.
The md5 guard covers the 21 files, and all of them are tracked and clean at HEAD.

Fix: `drv_fail` on DIRTY, or keep it as is knowingly.

**8. MINOR: a crash in the secondaries loses the verdicts JSON.**
- `analyze_generality` writes `GENERALITY_VERDICTS.json` last.
- `np.mean([r[k]['acc'] ...])` raises on `None` (a stratum with n = 0 at some seed). An empty `src` with a non-empty `C` does
  not crash.
- At T = 1024, with 100 trials on the 64-torus, n = 0 is improbable, so the risk is low. A crash after the verdicts are
  printed would still mean `drv_done` never fires.

Fix: dump V before the secondaries, or filter out None.

**9. MINOR (documentation): CoPEF npos_max.**
- The docstring says "npos_max = the context length". On the ring it is 2048 against 255 tokens, so 112k of the 131k
  position parameters can never be indexed. They get zero gradients and zero Adam updates, so they are harmless.
- The prereg's 334,483 params is correct but includes them.

Fix the docstring.

**10. MINOR: HELD's "LOST on <= 1 seed" is absolute.** If the control itself is LOST on 2 or more seeds (possible in the
lr 1e-3 descriptive pair, where the absolute 0.3 applies to SO_bal_hi), HELD is unreachable even for identical arms. The
smoke showed this: identical arms read PARTIAL when both are LOST. This matters only for the descriptive pair.

Fix: either accept this, or state it.

## Checked and correct

- **2-action path is behaviour-identical.** I loaded `model_wrap_exist.py` at c578b7e and at HEAD and ran both on a perturbed
  ring SignedAO_r4. For all four modes (min, gauge, balanced, signed_balanced), the built state dicts are bit-identical, the
  reports are identical (except the new `latent_cond_signed` key), the logits are bit-identical, and `wrap_share` is identical.
- **4-action constructions are exact.** On a perturbed SignedAO4_r4 with 7 negative omegas:
  - balanced: angle error 9e-7 rad, step_min +0.003, so all steps are non-negative.
  - signed_balanced: angle error 3e-6 rad.
  - Argmax agreement 1.000 at T = 256. Observation-token steps are exactly 0, and `act` = [0, 1, 2, 3].
  - Pilot s690: angle error 1.2e-6, frozen 0.99962 = source.
- **Pairs.** `GridWorld.ACTION_DELTAS` = {0: N, 1: S, 2: W, 3: E}, so the pairs (0,1), (2,3) are right. `wrap_share` selects
  on the wrapped angle, so the old audit's item 7 is fixed.
- **CARoPEF against the paper.** The paper writes theta_i = 10000^(-2i/d) = theta_1^i, which is canonical RoPE with
  i = 0..d/2-1. So channel i stepping f^i with expo 0..nb-1 is faithful. Channel 0 is f^0 = 1, a content-independent index
  clock in each head, as in RoPE.
  - d = d_head.
  - The input is the token embedding, as the paper states ("x_t ... the embedding of the token"); in one layer this equals the
    layer input.
  - The bias makes the init RoPE: f = 0.749894 = theta_1, and the phase equals (t+1) theta_i to 5e-6. The bias is disclosed.
  - The layer is MapWM's, and `--fast-attn` applies to it (plain WMTransformerLayer).
- **CoPE against Listing 1 and Sec. 4.**
  - Gates are sigmoid of the scaled logits including the mask (-inf -> 0).
  - p_ij = sum_{k=j}^{i} g_ik (flip-cumsum-flip).
  - Clamp is at npos_max - 1, interpolation is linear between integer knots, and the position logit uses the unscaled query.
  - pos_emb is shared between the heads of a layer, matching the paper's text: "shared between the heads only".
  - A direct loop over the formulas agrees to 1.2e-6.
- **Driver.**
  - `--lr 1e-4` after `--lr 1e-3`: the last one wins (the checkpoints record lr 0.0001 for SO_bal / AO_bal and 0.001 for _hi).
  - `--init-from` accepts the constructed blobs: variant, seed and config keys match, the load is strict, and losses_prior = [].
  - `--data-seed-offset 1000000` is recorded. The base-seed shift (10^6 x 1000003 mod 2^31 = 1.42e9) cannot collide with the
    source's batch indices (<= 88,200).
  - Same map (`--seed`).
  - Seed outer, arm inner; SO4 is launched first at each seed.
  - 128 artifacts = 64 torus + 48 ring + 16 construct.json.
  - GUARD covers the whole import chain of the used classes (model, model_sign, lie_groups, train, data_parallel,
    environment[_cancel/_nd], train_cancel, ckpt_guard, stats_core, the evals and the analysis). All guard files are tracked.
- **Eval layout.** `<R>/torus/<ARM>/p0/<V>_s<S>/<V>.pt`. Each arm has its own `--runs-dir` and its own
  `GENERALITY_TORUS_<ARM>.json`, so SO4 / SO_bal / SO_bal_hi (all SignedAO4_r4) do not collide. The keys match: noise refine
  writes `0.0|V|T` -> [[s, acc, nll]] and strata writes `V|s|1024` with `plain_lag<128` / `plain_lag>=128` / `wrap`.
  `--check-json` passed on every arm.
- **Smoke** (4 seeds, 3 epochs, n-steps 64, batch 4; ring 2 epochs). Every train, build, continuation, eval, strata and
  analysis step ran. 64/64 artifacts were present. Every gate printed UNREADABLE at the floor, as it should.
  - With `classify_run` forced to SOLVED, the E1 EXISTS path, E2, E3, B (NO DIFFERENCE) and every secondary executed.
  - The branch logic matches the prereg. HELD and NOT HELD are disjoint, and every label in E1 / E2 / E3 / B is reachable.
  - Fisher, sign-flip and permutation are all two-sided, from stats_core.
- **Design checks that pass.**
  - E1 at 0.005 with 50 trials is adequate: source and construction see the same trajectories, so the expected gap is 0
    (pilot 0.0000), and a real construction error moves far more than about 100 targets.
  - Signed_r4 is the right Case-III reference for B on both tasks: it is retrained in-batch, and was 8/8 in SIGN_MATCHED and
    RING_DIAL.
