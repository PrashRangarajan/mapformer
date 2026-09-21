"""Is MapPoPE's code advantage a STACK effect or a CONTEXT-LENGTH effect?

MapPoPE beats PoPE by -0.102 bpc at 1024-2048 but is flat-to-negative on the
distance-stratified closer metric (-0.000 / -0.005 / -0.027 / -0.005). Those are
different axes: distance from the matching opener vs distance from the training
context. This separates them.

Partition every predicted byte in the far bucket by the NESTING DEPTH at that
byte, reconstructed from the genuine bracket pairs (a difference array over
open/close offsets, so brackets inside strings and comments never count).

  depth 0    no bracket is open. There is no stack to track at all.
  depth >=1  inside at least one bracket.
  depth >=3  inside a non-trivial stack.

REGISTERED BEFORE RUNNING:
  If the advantage is NOT a stack effect, MapPoPE - PoPE is about the same at
  depth 0 as at depth >=1, and survives at depth 0 alone.
  If it IS a stack effect, the advantage concentrates at depth >=1 and collapses
  toward zero at depth 0.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

from .train_hourglass_enwik8 import build

_REPO = os.path.dirname(os.path.abspath(__file__))
LN2 = 0.6931471805599453
ARMS = ["MapPoPE-Flat", "PoPE-Flat", "RoPE", "Vanilla"]
NICE = {"MapPoPE-Flat": "MapPoPE", "PoPE-Flat": "PoPE", "RoPE": "RoPE", "Vanilla": "MapWM"}


def depth_timeline(n, open_pos, close_pos):
    d = np.zeros(n + 2, dtype=np.int32)
    np.add.at(d, open_pos, 1)
    np.add.at(d, close_pos + 1, -1)
    return np.cumsum(d)[:n]


@torch.no_grad()
def per_token_nll(model, data, T, dev, batch=4):
    n = data.shape[0]
    starts = np.arange(0, n - T - 1, T)
    out_nll = np.zeros(n, dtype=np.float64)
    out_hit = np.zeros(n, dtype=bool)
    for b0 in range(0, len(starts), batch):
        chunk = starts[b0:b0 + batch]
        x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in chunk]).to(dev)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + T].astype(np.int64)) for i in chunk]).to(dev)
        ce = F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1),
                             reduction="none").view(y.shape).cpu().numpy()
        for r, i in enumerate(chunk):
            # ce[r, t] is the loss on the byte at absolute position i+t+1
            lo = i + 1
            out_nll[lo:lo + T] = ce[r]
            out_hit[lo:lo + T] = True
        # only the FAR bucket matters, but scoring all of it is the same cost
    return out_nll, out_hit


def main(T=2048, dev="cuda:0"):
    data = np.fromfile(os.path.join(_REPO, "data", "code_val.bin"), dtype=np.uint8)
    z = np.load(os.path.join(_REPO, "data", "code_val_brackets.npz"))
    depth = depth_timeline(len(data), z["open_pos"], z["pos"])

    # position WITHIN its crop, so the far bucket can be selected
    n = len(data)
    starts = np.arange(0, n - T - 1, T)
    within = np.full(n, -1, dtype=np.int32)
    for i in starts:
        lo = i + 1
        within[lo:lo + T] = np.arange(T)
    far = (within >= 1024) & (within < 2048)

    res = {}
    for arm in ARMS:
        vals = {k: [] for k in ("all", "d0", "d1", "d3")}
        for seed in (0, 1, 2):
            ck = os.path.join(_REPO, "runs", "code", f"{arm}_s{seed}.best.pt")
            if not os.path.exists(ck):
                continue
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            c = blob["cfg"]
            m = build(c["model"], shorten=c["shorten"], dim=c["dim"], heads=c["heads"],
                      n_layers=c["n_layers"], grid_size=T,
                      bottleneck_r=c["bottleneck_r"]).to(dev)
            m.load_state_dict(blob["state_dict"], strict=False); m.eval()
            nll, hit = per_token_nll(m, data, T, dev)
            sel = hit & far
            for key, mask in (("all", sel), ("d0", sel & (depth == 0)),
                              ("d1", sel & (depth >= 1)), ("d3", sel & (depth >= 3))):
                vals[key].append(nll[mask].mean() / LN2)
            del m; torch.cuda.empty_cache()
        res[arm] = {k: np.array(v) for k, v in vals.items()}
        print(f"{NICE[arm]:9s} n={len(vals['all'])}  " +
              "  ".join(f"{k}={np.mean(v):.4f}" for k, v in vals.items()))

    counts = {"d0": int((far & (depth == 0)).sum()), "d1": int((far & (depth >= 1)).sum()),
              "d3": int((far & (depth >= 3)).sum()), "all": int(far.sum())}
    print(f"\nbytes scored in the far bucket: {counts}")

    def contrast(a, b, key):
        d = res[a][key] - res[b][key]
        mde = 2.8 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        det = abs(d.mean()) > mde
        return (f"{d.mean():+.4f} (MDE {mde:.4f}, better {int((d<0).sum())}/{len(d)}) "
                + ("DETECTABLE" if det else "unmeasured"))

    print("\nMapPoPE - PoPE, by nesting depth at the predicted byte:")
    for key, lab in (("all", "all far positions"), ("d0", "depth 0  (NO stack)"),
                     ("d1", "depth >=1"), ("d3", "depth >=3")):
        print(f"  {lab:22s} {contrast('MapPoPE-Flat','PoPE-Flat',key)}")
    print("\nMapPoPE - MapWM (the encoding gap), same split:")
    for key, lab in (("d0", "depth 0  (NO stack)"), ("d1", "depth >=1")):
        print(f"  {lab:22s} {contrast('MapPoPE-Flat','Vanilla',key)}")


if __name__ == "__main__":
    main()
