"""The three THEORY_MAPPOPE numbers that had no committed script (audit, 2026-09-19).

(a) full-context control: bucket NLL for the 2048-trained checkpoints in runs/jsb
(b) omega-compression dissociation: bucket NLL with omega scaled at EVAL only
(c) learned per-token phase magnitude, mean |d^q|,|d^k| in radians on real data

The originals were run inline at n=1-4 seeds or on 2 pieces; these use every seed. Where they differ
the numbers here supersede. Run from /home/prashr: python3 -m mapformer.probe_theory_numbers
"""
import glob
import numpy as np, torch

from mapformer.environment_jsb import splits
from mapformer.train_jsb import build, nll_buckets

DEV = "cuda:0"
B = ["0-512", "512-1024", "1024-2048"]


def load(name, d, arch, **kw):
    out = []
    for pt in sorted(glob.glob(f"/home/prashr/mapformer/{d}/{name}_s*/{name}.pt")):
        m = build(arch, 256, 8, 6, 2, 2048, 0.2, "uniform", torch.Generator(), **kw).to(DEV).eval()
        m.load_state_dict(torch.load(pt, map_location=DEV, weights_only=False))
        out.append(m)
    return out


def phase_mag(m, x):
    o = []
    for lay in m.layers:
        if hasattr(lay, "dq"):
            h = lay.norm1(m.token_emb(x))
            o += [float(lay.dq(h).abs().mean()), float(lay.dk(h).abs().mean())]
    return float(np.mean(o)) if o else float("nan")


def main():
    S = splits(); Xte, Mte = S["test"]
    L = ["# Theory numbers, recomputed from committed code (2026-09-19)", "",
         "Supersedes the inline values in `THEORY_MAPPOPE.md` and `T3GEN_RESULTS.md`, which were taken "
         "at 1-4 seeds or on 2 pieces.", "", "## (a) Full-context control: 2048-trained checkpoints, bucket NLL", "",
         "| arm | " + " | ".join(B) + " | seeds |", "|---|---|---|---|---|"]
    for name, arch in [("RoPE", "RoPE"), ("PoPE", "PoPE"), ("MapWM_r2", "MapWM"), ("MapPoPE_r2", "MapPoPE")]:
        ms = load(name, "runs/jsb", arch)
        r = [nll_buckets(m, Xte, Mte, DEV) for m in ms]
        L.append(f"| {name} | " + " | ".join(f"{np.mean([x[k] for x in r]):.4f}" for k in B) + f" | {len(ms)} |")
        print(L[-1], flush=True)
    L += ["", "## (b) Omega compression at EVAL only (512-trained checkpoints)", "",
          "| arm | scale | " + " | ".join(B) + " |", "|---|---|---|---|---|"]
    for name, arch in [("MapWM_r2", "MapWM"), ("MapPoPE_r2", "MapPoPE")]:
        for sc in (1.0, 0.5, 0.25):
            ms = load(name, "runs/jsb_len512", arch)
            for m in ms:
                with torch.no_grad():
                    m.path_integrator.omega.mul_(sc)
            r = [nll_buckets(m, Xte, Mte, DEV) for m in ms]
            L.append(f"| {name} | {sc:.2f} | " + " | ".join(f"{np.mean([x[k] for x in r]):.4f}" for k in B) + " |")
            print(L[-1], flush=True)
    L += ["", "## (c) Learned per-token phase magnitude (mean |d^q|, |d^k|, radians)", "",
          "| task | arm | phase |", "|---|---|---|"]
    x = Xte[:8].to(DEV)
    for lab, name, d in [("Bach", "MapPoPE_T3_r2", "runs/jsb_t3"),
                         ("Bach", "MapPoPE_T3_r2_pi0.1", "runs/jsb_forced")]:
        ms = load(name, d, "MapPoPE_T3")
        v = [phase_mag(m, x) for m in ms]
        L.append(f"| {lab} | {name} | {np.mean(v):.3f} +/- {np.std(v, ddof=1):.3f} |")
        print(L[-1], flush=True)
    open("/home/prashr/mapformer/THEORY_NUMBERS.md", "w").write("\n".join(L) + "\n")
    print("wrote THEORY_NUMBERS.md")


if __name__ == "__main__":
    main()
