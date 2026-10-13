# Can depth or a loop let index attention catch up with path integration on the torus? -- pre-registration (2026-10-12, before any run)

## Question
Position code and architecture are separate axes. They have been crossed on Match-Query (`LOOP_HEADROOM.md`: a loop helps
both codes), on short-walk tori (`LOOPED_PILOT.md`: for index RoPE a loop x4 matches 4 real layers) and on Dyck at matched
depth (`DYCK_MDEPTH_RESULTS.md`: one path layer ~ three attention layers). On the 2D torus at the training length T = 1024
-- the regime of the rank and sign results -- the index model has only ever been run with ONE layer (RoPE 0.731 vs signed
path 0.998, `SIGN_MATCHED_RESULTS.md`). **Does a 4-layer or a looped x4 index transformer catch up with a 1-layer
path-integrated model on the torus at T = 1024?** (User's choice 2026-10-12: index cells plus the path 1-layer reference; the
deeper path cells are dropped because path at 1 layer already solves 8/8 here.)

## Cells (`train_variant`, unchanged models; one batch; seeds 500-503, fresh; n = 4 per cell)
| cell | variant | layers | params (probe) |
|---|---|---|---|
| RoPE_L1 | `RoPE` | 1 | ~204k |
| RoPE_L4 | `RoPE` --n-layers 4 | 4 real layers | 798,741 |
| RoPE_Loop | `RoPELooped` (one shared block x4; ignores --n-layers, run with 1) | depth 4, 1 block | 203,925 |
| Path_L1 | `Vanilla_r4` (MapWM, rank 4 shared) | 1 | ~204k |
Recipe = RANK_MI's (T = 1024, batch 16, 98 batches, lr 1e-3, cosine with 5% warmup, 900 epochs, 2 heads, d 128,
`--n-landmarks 0`, `--data-workers 3`) **plus `--fast-attn` in every cell** (SDPA, TF32 off; valid for RoPE and MapWM; logits
agree with the explicit path to ~1e-6 but training is not bit-identical: a new series, never compared with stored runs).
Evaluation: `eval_noise_refine` (held-out map; T = 1024 primary, 512 / 2048 secondary) and `eval_rank_strata` (T = 1024), as
RANK_MI. Floors on this stream (RANK_MI): blank 0.507, best n-gram 0.576.

## Registered decisions (`analyze_factorial.py`; accuracy at T = 1024; n = 4 vs 4)
Exact permutation tests (C(8,4) = 70 relabellings: minimum two-sided p 0.029) and Fisher on SOLVED (final-5% training loss <
0.05; 4/4 vs 0/4 gives p 0.029). At n = 4 only large splits are readable (rule 5); every label says so.
- **C0 (sanity, matched one layer): Path_L1 - RoPE_L1.** PATH WINS AT ONE LAYER: d >= 0.05 and perm p < .05. Otherwise
  reported as it falls (expected from SIGN_MATCHED: path 0.998 vs index 0.731).
- **C1 (depth): RoPE_L4 vs Path_L1.** INDEX AT 4 LAYERS CATCHES UP: mean acc(RoPE_L4) >= mean acc(Path_L1) - 0.02 AND
  SOLVED(RoPE_L4) >= 3/4. DOES NOT CATCH UP: Path_L1 - RoPE_L4 >= 0.05 with perm p < .05, OR Fisher(SOLVED) p < .05 with
  Path ahead. UNMEASURED otherwise.
- **C2 (loop): RoPE_Loop vs Path_L1.** The same three labels.
- **C3 (does depth / the loop help index at all): RoPE_L4 - RoPE_L1 and RoPE_Loop - RoPE_L1.** HELPS: d >= 0.05 and perm
  p < .05; otherwise reported as it falls.
Secondary (no verdict): SOLVED / STALLED / DESCENDING per cell; wrap-only and plain revisit strata; accuracy at 512 and 2048
(out of distribution, rule 10); r(final loss, acc); the dropout-scale re-score for the ONE-layer cells only (it lowers deep
and looped models).

## Process and cost
No separate pilot: a 5-epoch timing / memory probe of the deep cells (seed 400, outside the batch) gave 5.5-6.4 s/epoch at 2
concurrent with --fast-attn, ~2 GB per job (15-16 s/epoch, 7.5 GB without it). Independent code check (rule 29) before
launch. GPU 0 only (another user holds GPU 1), 4 jobs at a time: ~8 h.

## Amendment 1 (2026-10-12, after the independent check `docs/audits/2026-10-12/factorial_audit.md`; before launch)
The check ran the full pipeline on CPU (16 tiny runs with the driver's flags, eval, strata, analysis): the cells build as
claimed (RoPE --n-layers 4: 4 layers, 798,741 params; RoPELooped: one block x4; Vanilla_r4: shared rank 4, 1 layer),
--fast-attn takes effect in every cell (fast vs explicit 8e-7 in eval mode), the driver and analysis paths match. Changes:
- **D1 (budget).** Every C1 / C2 label carries "[within 900 epochs; <cell> still DESCENDING k/4]". Prediction, stated now:
  `LOOPED_PILOT.md` found index RoPE's reach on short-walk tori limited to gaps of 17-32 moves at 4 layers or a loop x4, so
  DOES NOT CATCH UP is expected at T = 1024.
- **D2.** CATCHES UP is renamed **INDEX WITHIN 0.02 OF PATH (margin rule, no equivalence test at n = 4)**.
- **D3 (memorisation).** Secondary: every run also evaluated on its OWN training map (`eval_noise_refine --env-seed <seed>`,
  T = 1024), printed beside the held-out accuracy.
- **D4 (scope).** With 1 and 4 layers only, the design asks whether 4 layers (or a loop x4) suffice, not the exchange rate.
- **D5.** C0-C3 are separate questions; no multiplicity correction is possible or claimed at n = 4.
- **Minor.** The strata eval cross-checks against the eval JSON (`--check-json`); an empty stratum no longer crashes the
  analysis; floors are RANK_MI's measured values on the same stream (quoted, not re-measured).
Smoke re-run: ALL PASS.
