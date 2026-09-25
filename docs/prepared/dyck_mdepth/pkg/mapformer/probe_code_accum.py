"""Does the ACCUMULATOR or the KERNEL explain MapWM's code collapse?

Both MapWM and MapPoPE path-integrate, so they share an accumulator; they differ
only in how the phase is USED. Two candidate explanations for MapWM collapsing
past the training context while MapPoPE does not:

  (A) accumulator -- MapWM's S leaves its trained range faster, so the phase is
      evaluated where nothing calibrated it.
  (B) kernel -- both leave the range equally, but MapWM's kernel has PER-PAIR
      phases (it rotates content Q,K) while PoPE's shape is shared across pairs
      with only a non-negative content weight, so the same excursion decoheres
      one and not the other.

Discriminator: measure range(S) at the trained length (512) and at 2048. If both
arms' excursions grow the same way, (A) is dead and the difference is the kernel.

alpha is the log-log slope of range(S) against T, as in probe_dyck_alpha: ~0.5
is a map (diffusive, cancels), ~1.0 a clock (monotone token counter).
"""
import os
import numpy as np
import torch

from .train_hourglass_enwik8 import build
from .ckpt_guard import check_not_stale

_REPO = os.path.dirname(os.path.abspath(__file__))
LENS = [128, 256, 512, 1024, 2048]


@torch.no_grad()
def accum(model, data, T, n, dev):
    idx = np.random.RandomState(0).randint(0, len(data) - T - 1, n)
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in idx]).to(dev)
    emb = model.token_emb(x)
    delta = model.action_to_lie(emb)              # (B, T, H, n_blocks)
    S = delta.mean(-1).cumsum(1)                  # (B, T, H)
    rng = (S.max(1).values - S.min(1).values)     # (B, H)
    return float(rng.mean())


def main(n=16, dev="cuda:0"):
    data = np.fromfile(os.path.join(_REPO, "data", "code_val.bin"), dtype=np.uint8)
    print(f"{'arm':<22}{'alpha':>8}" + "".join(f"{'T='+str(t):>10}" for t in LENS)
          + f"{'2048/512':>10}")
    for arm in ["Vanilla", "MapPoPE-Flat"]:
        for seed in (0, 1, 2):
            ck = os.path.join(_REPO, "runs", "code", f"{arm}_s{seed}.best.pt")
            if not os.path.exists(ck):
                print(f"MISSING {ck} -- this arm will have fewer seeds", flush=True)
                continue
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            check_not_stale(ck, blob)
            c = blob["cfg"]
            m = build(c["model"], shorten=c["shorten"], dim=c["dim"], heads=c["heads"],
                      n_layers=c["n_layers"], grid_size=2048,
                      bottleneck_r=c["bottleneck_r"]).to(dev)
            m.load_state_dict(blob["state_dict"]); m.eval()   # strict
            rs = [accum(m, data, T, n, dev) for T in LENS]
            a = np.polyfit(np.log(LENS), np.log(np.maximum(rs, 1e-12)), 1)[0]
            print(f"{arm+'_s'+str(seed):<22}{a:>8.3f}"
                  + "".join(f"{r:>10.3f}" for r in rs)
                  + f"{rs[-1]/max(rs[2],1e-12):>10.2f}x")
            del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
