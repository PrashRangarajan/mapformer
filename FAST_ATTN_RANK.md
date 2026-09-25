# `--fast-attn` / `--deterministic` on the rank config (measured 2026-09-24)

Opt-in flags on `train_variant.py` (audit efficiency #1). **Default off; the default explicit
path is unchanged and bit-identical to before** (`docs/audits/2026-09-24/applied/cmp_train_after_fast_attn.txt`).
Use them for NEW series only: SDPA is not bit-identical to the explicit path (its dropout draws
from a different RNG stream and it sums in a different order), so it cannot be pooled with, or
continue, `rank_matched_e900`, `_e900c`, `rank_proj_train` or any other explicit-path batch.
Every arm of a batch must share the setting. Both flags are saved in the checkpoint config.

One RTX 4090, torch 2.10.0+cu128, TF32 off, Vanilla (MapWM r=2), B=16, T=1024, lr 1e-3 cosine,
1 layer, 2 heads, d=128, seed 0, GPU otherwise idle.

| path | s / epoch (98 batches), CLI, `--data-workers 3` | s / epoch, serial data (in-process) | peak GiB | run-to-run bitwise |
|---|---|---|---|---|
| explicit (default) | 2.0 | 1.97 | 2.19 | yes |
| `--fast-attn` | **0.9 (2.2x)** | 1.28 (1.54x) | **0.47** | **no** |
| `--fast-attn --deterministic` | 1.6 (1.25x) | 1.58 (1.25x) | 0.47 | **yes** |
| explicit + `--deterministic` | 2.2 | -- | -- | yes; bitwise equal to the default path over 5 x 98 batches |

- **Numerical agreement:** max |logit| difference SDPA vs explicit on identical weights (eval
  mode, B16 T1024) is **1.07e-06** (max |logit| 1.78). TF32 stays off: it, not SDPA, broke
  equivalence on the compositional trainer (6.0e-01).
- **Determinism needs the STRICT mode.** `torch.use_deterministic_algorithms(True,
  warn_only=True)` (the audit's proposal) only warns "Memory Efficient attention defaults to a
  non-deterministic algorithm" and training stays irreproducible (measured: two same-seed runs
  differ, 2.071453 vs 2.071460 after 2 x 20 batches). `--deterministic` therefore calls it with
  `warn_only=False` (and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` if unset); then two same-seed
  runs are bitwise identical. Cost: SDPA's advantage drops from 2.2x to 1.25x.
- **Trajectories differ.** After 5 x 98 batches the explicit run is at loss 1.2215 and the SDPA
  run at 1.0053 (same seed, same data): early training sits on a plateau whose exit timing is
  seed-sensitive, so an SDPA series is a different sample, not a faster copy.
- `--fast-attn` is refused for variants with no plain `model.WMTransformerLayer` (e.g.
  VanillaEM: MapEM's Hadamard scores cannot be expressed as SDPA); subclasses that override the
  layer's forward keep their own attention and are listed.
- Not measured: jobs per GPU after this change (the audit suggests re-measuring 1/2/3 per GPU;
  `lib_driver.sh` defaults to 2), and variants other than Vanilla.

Scripts and raw outputs: `docs/audits/2026-09-24/applied/verify_gpu_bitexact_repo.py`
(`gpu_bitexact_part2*.txt`), `cmp_fastattn.sh` (`fast_attn_cli_checks.txt`), `time_fastattn.sh`
(`fast_attn_cli_timing.txt`).
