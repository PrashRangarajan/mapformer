"""Wall-clock scaling: does the parallel scan actually buy anything?

MapFormer's headline efficiency claim is that path integration is a cumsum, so it
is O(log T) parallel rather than O(T) sequential. Nothing in this repo has ever
measured it. We now hold both sides in one codebase:

  PARALLEL   MapFormerWM / MapFormerEM   theta = omega * cumsum(Delta)
  SEQUENTIAL MapEM-NC                    R_t = R_step(t) @ R_{t-1}, a genuine
                                         matrix product. The paper concedes this
                                         makes it "analogous to TEM-t".
  SEQUENTIAL TEMFaithful                 per-action W_a applied in a Python loop
  BASELINE   PlainFlat                   index RoPE, no path integration at all

Method notes that decide whether the numbers mean anything:
  - CUDA is asynchronous; every timed region is wrapped in torch.cuda.synchronize().
  - Warmup iterations are discarded (cuDNN autotuning, lazy init, allocator warmup).
  - MEDIAN of repeats, not mean -- a single scheduling hiccup skews a mean badly.
  - FORWARD+BACKWARD is reported alongside forward-only, since training cost is
    what actually matters and backward through a sequential loop is where the
    asymptotic difference bites.
  - Parameter counts are reported: these models are NOT parameter-matched, so
    absolute times are not a fair architecture comparison. What IS comparable is
    each model's SCALING with sequence length, which is the claim under test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent


def time_model(model, L, vocab, device, batch=4, warmup=3, reps=7, backward=False):
    """Forward-only timings run under no_grad.

    Without it, a 'forward only' measurement also builds the autograd graph, and
    that cost scales with NODE COUNT rather than FLOPs -- which penalises the
    Python-loop models (MapEM-NC, TEMFaithful) far more than the cumsum ones,
    biasing precisely the comparison this benchmark exists to make, in the
    direction that flatters the parallel-scan claim.
    """
    import contextlib
    x = torch.randint(0, vocab, (batch, L), device=device)
    ctx = (lambda: contextlib.nullcontext()) if backward else torch.no_grad
    for _ in range(warmup):
        with ctx():
            out = model(x)
        if backward:
            out.float().pow(2).mean().backward(); model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with ctx():
            out = model(x)
        if backward:
            out.float().pow(2).mean().backward()
        torch.cuda.synchronize(device)
        ts.append(time.perf_counter() - t0)
        if backward:
            model.zero_grad(set_to_none=True)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["Vanilla", "VanillaEM_P0", "PlainFlat",
                             "MapEM_NC_L", "TEMFaithful"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "TIMING_BENCHMARK.md"))
    args = ap.parse_args()
    dev = torch.device(args.device)

    res, params, notes = {}, {}, {}
    for v in args.variants:
        if v not in VARIANT_MAP:
            notes[v] = "not in VARIANT_MAP"; continue
        try:
            m = VARIANT_MAP[v](vocab_size=args.vocab, d_model=args.d_model,
                               n_heads=2, n_layers=args.n_layers, grid_size=64).to(dev)
        except Exception as e:
            notes[v] = f"build failed: {type(e).__name__}"; continue
        params[v] = sum(p.numel() for p in m.parameters())
        res[v] = {}
        for L in args.lengths:
            try:
                fwd = time_model(m, L, args.vocab, dev, args.batch, backward=False)
                bwd = time_model(m, L, args.vocab, dev, args.batch, backward=True)
                res[v][L] = {"fwd_ms": fwd * 1e3, "fwdbwd_ms": bwd * 1e3}
                print(f"{v:14s} L={L:5d}  fwd {fwd*1e3:8.2f} ms   fwd+bwd {bwd*1e3:8.2f} ms",
                      flush=True)
            except RuntimeError as e:
                res[v][L] = None
                print(f"{v:14s} L={L:5d}  OOM/err: {type(e).__name__}", flush=True)
                break
        del m; torch.cuda.empty_cache()

    LS = args.lengths
    lines = ["# Wall-clock scaling with sequence length", "",
             f"batch={args.batch}, d_model={args.d_model}, n_layers={args.n_layers}, "
             f"median of 7 reps after 3 warmups, `torch.cuda.synchronize()` around "
             f"every timed region.", "",
             "**These models are NOT parameter-matched, so absolute times are not an "
             "architecture comparison.** What is comparable is each model's SCALING "
             "with L — the O(log T) parallel-scan claim.", "",
             "## Forward + backward (training cost)", "",
             "| variant | params | " + " | ".join(f"L={L}" for L in LS) + " | growth (span shown) |",
             "|---" * (len(LS) + 3) + "|"]
    for v in args.variants:
        if v not in res:
            lines.append(f"| {v} | — | " + " | ".join("—" for _ in LS) + f" | {notes.get(v,'')} |")
            continue
        cells, first, last = [], None, None
        for L in LS:
            r = res[v].get(L)
            if r is None: cells.append("OOM")
            else:
                cells.append(f"{r['fwdbwd_ms']:.1f}")
                if first is None: first = r["fwdbwd_ms"]
                last = r["fwdbwd_ms"]
        # label the ratio with the lengths it ACTUALLY spans -- on an OOM-truncated
        # row this is not L=2048/L=128, and mislabelling it understates the
        # sequential models' scaling penalty (i.e. flatters the parallel claim)
        got = [L for L in LS if res[v].get(L) is not None]
        ratio = (f"{last/first:.1f}x ({got[-1]}/{got[0]})" if (first and last and got)
                 else "—")
        lines.append(f"| {v} | {params[v]:,} | " + " | ".join(cells) + f" | **{ratio}** |")
    lines += ["", "## Forward only", "",
              "| variant | " + " | ".join(f"L={L}" for L in LS) + " |",
              "|---" * (len(LS) + 1) + "|"]
    for v in args.variants:
        if v not in res: continue
        lines.append(f"| {v} | " + " | ".join(
            "OOM" if res[v].get(L) is None else f"{res[v][L]['fwd_ms']:.1f}"
            for L in LS) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump({"results": res, "params": params, "notes": notes},
              open(str(args.out).replace(".md", ".json"), "w"), indent=2)
    print("\n".join(lines)); print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
