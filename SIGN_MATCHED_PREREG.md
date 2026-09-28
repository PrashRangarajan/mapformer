# The sign ablation at matched length -- pre-registration (2026-09-27, before any run)

## Why
`SIGN_ABLATION.md` (trained at T=128, 300 epochs, 12 seeds): removing the sign of the phase
increment (`Abs_r4`, `|W_out W_in x|`) costs -0.303 / -0.363 at T=512 / 1024 against `Signed_r4`,
12/12, but at the TRAINING length it is -0.054 (MDE 0.057, UNMEASURED). That batch's own
pre-registration named this pattern in advance: "< -MDE at OOD lengths only -> NOT the predicted
result ... another instance of that unexplained axis". CLAUDE.md lists sign as robustness, not
capability, until a matched-length control exists (rule 10). This is that control: train AND test
at T=1024. The net-displacement mechanism (a monotone code can hold (n_E, n_W) but not n_E - n_W)
predicts the deficit should appear in distribution once walks are long enough to revisit through
cancellation many times; the robustness reading predicts it closes.

## Arms (torus, trained AND tested at T=1024, one batch, 8 seeds, all retrained)
| arm | Delta | role |
|---|---|---|
| `Signed_r4` | `W_out W_in x` | baseline (bit-identical construction path to the constrained arms) |
| **`Abs_r4`** | `|W_out W_in x|` | **primary**: sign removed, nothing else |
| `Pos_r4` | `softplus(.)` (GRAPE-AP) | secondary monotone arm (init confound, as in the original) |
| `RoPE` | token index | index reference; position effect for scale |
`CARoPE_r4` and the `Vanilla_r4` construction control are dropped (the latter was 0.000 at every
length in the original, and the gate below re-checks it on weights).

Recipe: the rank line's T=1024 recipe, which solves the torus at r=4 (8/8, `RANK_SEP_RESULTS.md`):
batch 16, T=1024 (16,384 tokens/step, the same as the original's 128 x 128), 900 epochs x 98
batches, lr 1e-3, warmup + cosine, 1 layer, 2 heads, d 128, `--data-workers 3`, `--save-full-state`,
explicit attention path. Gates re-run before launch: parameter counts, Signed vs Vanilla_r4 on
shared weights, Delta >= 0 at init for Abs/Pos, causal leak.

## Readouts and branches
Primary: `Abs_r4` - `Signed_r4` at T=1024, by exact permutation test on per-seed revisit accuracy
and Fisher on SOLVED counts (final-5% loss < 0.05; `stats_core`). A contrast FIRES if either p <
0.05. Accuracy floor: the per-stratum best constant (0.506 / 0.507, `RANK_MATCHED_RESULTS.md`),
recomputed by `eval_rank_strata` on these runs, and the index arm.
- **SIGN IS CAPABILITY** -- the primary fires with Abs below Signed. The citable sign result
  becomes in-distribution, and the -0.363 OOD number stops being the headline.
- **SIGN IS ROBUSTNESS** -- the primary does not fire AND the Abs - Signed accuracy difference
  is within its exact-t MDE AND Abs solves >= 6/8. The in-distribution sign cost is then
  unmeasured at T=1024 as at T=128; sign stays a robustness result.
- **UNMEASURED** otherwise (e.g. Abs fails to converge without a significant contrast).
Rule 2: r(final loss, accuracy) reported; if |r| > 0.5 the loss-matched residual contrast is
printed beside the raw one, but the registered verdict is on the raw contrast (at matched length
loss-matching is uninformative -- the loss IS the in-distribution readout).
Secondary: Pos_r4 - Signed_r4 (same tests); Signed - RoPE (position effect at matched length);
T=2048 (2x OOD) for each; strata at T=1024; post-training Delta >= 0 check (`probe_sign.py`).
Void: any of the 32 runs missing; a constrained arm with Delta < 0 after training; md5 guard trips.
Scope: torus, T=1024, n_heads 2, d 128, r=4 shared, one recipe, 900-epoch budget.
Cost: ~2 s/epoch solo, ~45 min per run at 2 jobs/GPU; 32 runs over 4 slots ~ 6-8 h.
