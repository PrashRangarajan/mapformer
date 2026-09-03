"""MEASURE the compute of the loop x hierarchy 2x2. Do not infer it.

The pairing claim is that hierarchy pays for the loop's compute. Block counts make
that look like a 17% saving, but at these widths attention is a small fraction of a
block, so the analytic number and the real one differ -- the enwik8 run already
found an 8.6% "wall time saving" that was contamination from GPU co-tenancy. Each
arm is timed ALONE on an otherwise idle device, after warmup, with explicit
synchronisation.
"""
import argparse, json, time
from pathlib import Path

import torch

from mapformer.train_variant import VARIANT_MAP

ARMS = ["HourglassFlat3", "Hourglass_k2", "LoopedHourglassFlat", "LoopedHourglass"]


def bench(v, L, bs, dev, iters=30):
    m = VARIANT_MAP[v](vocab_size=2, d_model=128, n_heads=2, n_layers=1,
                       grid_size=64).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    tok = torch.randint(0, 2, (bs, L), device=dev)
    for _ in range(8):
        opt.zero_grad(); m(tok).sum().backward(); opt.step()
    torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad(); m(tok).sum().backward(); opt.step()
    torch.cuda.synchronize(dev)
    dt = (time.time() - t0) / iters
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    n = sum(p.numel() for p in m.parameters())
    del m, opt; torch.cuda.empty_cache()
    return dt * 1000, peak, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", nargs="+", type=int, default=[16, 128, 512, 2048])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="LOOP_HIER_COMPUTE.md")
    a = ap.parse_args()
    dev = torch.device(a.device)

    res = {}
    for L in a.lengths:
        for v in ARMS:
            ms, mem, n = bench(v, L, a.batch_size, dev)
            res[f"{v}|{L}"] = dict(ms=ms, mem=mem, params=n)
            print(f"L={L:5d} {v:22s} {ms:7.2f} ms/step  {mem:8.1f} MiB  {n:,}",
                  flush=True)

    o = ["# Loop x hierarchy: what each saves, measured", "",
         f"Forward+backward, batch {a.batch_size}, timed alone on an idle device "
         f"after warmup with explicit synchronisation.", "",
         "The loop saves PARAMETERS and costs COMPUTE; hierarchy saves COMPUTE and",
         "costs PARAMETERS. The question is whether combining them gives both.", "",
         "| length | arm | params | ms/step | peak MiB | vs flat-unshared |",
         "|---|---|---|---|---|---|"]
    for L in a.lengths:
        base = res[f"HourglassFlat3|{L}"]
        for v in ARMS:
            r = res[f"{v}|{L}"]
            o.append(f"| {L} | {v} | {r['params']:,} | {r['ms']:.2f} | "
                     f"{r['mem']:.1f} | {100 * (r['ms'] / base['ms'] - 1):+.1f}% time, "
                     f"{100 * (r['mem'] / base['mem'] - 1):+.1f}% mem |")
    o += ["", "## The pairing", ""]
    for L in a.lengths:
        b = res[f"HourglassFlat3|{L}"]; lh = res[f"LoopedHourglass|{L}"]
        lf = res[f"LoopedHourglassFlat|{L}"]; h = res[f"Hourglass_k2|{L}"]
        o.append(f"- **L={L}**: hierarchy alone {100*(h['ms']/b['ms']-1):+.1f}% time; "
                 f"sharing alone {100*(lf['ms']/b['ms']-1):+.1f}%; "
                 f"**both {100*(lh['ms']/b['ms']-1):+.1f}% time and "
                 f"{100*(lh['params']/b['params']-1):+.1f}% parameters**.")
    o += ["", "Sharing is expected to cost roughly nothing in time -- it changes "
          "which weights are read, not how many block applications run -- so any "
          "time saving in the combined arm comes from the hierarchy and any "
          "parameter saving comes from the sharing. That is the point: the two "
          "resources are independent, so the savings should compose exactly."]
    Path(a.out).write_text("\n".join(o) + "\n")
    json.dump(res, open(a.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(o))


if __name__ == "__main__":
    main()
