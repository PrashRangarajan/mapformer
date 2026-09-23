# MAESTRO: plan (NOT run, NOT pre-registered yet)

Written 2026-09-23 so the plan survives a session boundary. Nothing below has been launched.

## Why MAESTRO

Bach Chorales is the only natural-sequence dataset in this project with a detectable
PoPE-over-RoPE gain (-0.032 NLL, 5/5, `JSB_RESULTS.md`; paper -0.019). On code at matched
length it is -0.001, unmeasured (`CODE_DECAY_RESULTS.md`, `runs/code2048`); on enwik8 it is
+0.0034 with PoPE worse (`ABLATE_RESULTS.md`). And Bach is **overfitting-limited 3x**: pitch
transposition alone is worth 0.107 NLL against the 0.032 PoPE-RoPE gap (`AUG_RESULTS.md`),
on 229 training pieces. MAESTRO (~200 h of piano, v3.0.0 MIDI) is PoPE's other music dataset
and large enough not to overfit. **It tests whether PoPE's gain survives without the ceiling.**

## The paper's recipe (PoPE arXiv:2509.10534, App. B, Tables 7 and 8; `papers/txt/pope.txt`)

| setting | value |
|---|---|
| model | d=384, 8 heads, 6 layers, RMSNorm, base 10,000, delta init range 2pi, dropout 0.1 |
| batch x seq | 16 x 2048 |
| lr | 6e-4 -> 6e-5 cosine, 500 warmup, decay over all 60,000 iters |
| iters | **60,000** |
| optimiser | AdamW, wd 0.01, grad clip 1.0, beta2 0.99 |
| data | MAESTRO v3.0.0 MIDI, 90/5/5 split, sequences of max length 2048 |
| tokeniser | REMI (+EOS, BOS, MASK, PAD) -> vocab 328 |
| augmentation | pitch transposition uniform in {-3..+3} |
| published test NLL | **RoPE 1.501, PoPE 1.486 (effect 0.015)** |

## Measured here (2026-09-23)

- **Batch 16 OOMs on a 24 GB card.** Use gradient accumulation 4 micro-batches x 4.
- Throughput at batch 4, fp32, one run alone (from the `check_bf16.py` fp32 runs):
  **RoPE 61.4k tok/s** (7.51 it/s x 8192), **MapPoPE 50.4k tok/s** (6.17 it/s x 8192).
  Peak memory RoPE 8.8 GiB, MapPoPE 9.4 GiB (measured in-session, no committed log).
- The cards are COMPUTE-bound: total throughput is ~110k tok/s across both GPUs regardless
  of how many runs share them, so concurrency does not shorten the batch.
- **bf16 is NOT licensed** (`BF16_RESULTS.md`): the MapWM-RoPE gap moves 0.0117 nats under
  bf16 against a 0.005 threshold, for only a 1.25x speedup. Run fp32.
- Budget: 60,000 x 16 x 2048 = **~2.0B tokens per run, ~9-11 h per run.**

| design | runs | wall time |
|---|---|---|
| PoPE vs RoPE, n=5 | 10 | **~2 days** |
| full 2x2 (RoPE, PoPE, MapWM, MapPoPE), n=3 | 12 | **~2.5 days** |
| full 2x2, n=5 | 20 | **~4 days** |

## Power -- the reason to prefer n=5

The published effect is 0.015 NLL, about half the 0.032 measured on Bach here (0.8x their
published Bach 0.019). Borrowing Bach's paired sd as a guess (MDE 0.0111 at n=5 on
MapPoPE-MapWM -> paired sd ~0.0089): MDE ~0.0144 at n=3, ~0.0111 at n=5. **At n=3 the
published effect sits on the MDE; n=5 is the minimum that can detect it** if MAESTRO's seed sd
is Bach-like. A larger dataset may well have a smaller sd; measure it on the first seeds
before committing to n.

## Prerequisites before launch

- **miditok is not installed.** REMI tokenisation with vocab 328 must be reproduced and its
  vocab size checked against the paper's 328.
- Data reachable: the MIDI zip is 58,416,533 bytes (HEAD request only; not downloaded).
- Check the trainer supports RMSNorm, dropout 0.1, beta2 0.99, gradient accumulation and
  transposition augmentation; any change is a new code path and needs its own equivalence check.
- Gates first (project rule): measured floor (unigram / n-gram NLL), and a pre-registration
  with the readout (test NLL at the best-validation checkpoint) and MDE written down.
- A single-instance `flock` guard on the driver and an occupancy-aware GPU picker
  (`run_code2048_fill.sh` pattern) -- both bought by the code batches.
