"""Accumulator exponent on Dyck-2, signed vs monotone (T2's manipulation check).

Committed 2026-09-20 after an audit found the T2 manipulation numbers had no script behind them and
did not reproduce under the auditor's convention. This fixes the convention explicitly and reports
the PER-SEED SPREAD, which the original table omitted: the signed arms' exponent is very noisy across
seeds, so only the direction (signed well below 1, monotone at 1) is a stable claim.

Convention: S = cumsum over tokens of the per-token increment averaged over frequency blocks, per
head; range(S) = max_t S - min_t S averaged over sequences and heads; alpha from a log-log fit of
range against T over one sequence family. Run: python3 -m mapformer.probe_dyck_alpha
"""
import glob, re
import numpy as np, torch

from mapformer.environment_dyck import DyckWorld
from mapformer.train_dyck import build

ARMS = [("MapWM signed", "MapWM-1L_r2", "MapWM", "runs/dyck_bs128"),
        ("MapPoPE signed", "MapPoPE-1L_r2", "MapPoPE", "runs/dyck_bs128"),
        ("MapWM monotone", "MapWM_abs-1L_r2", "MapWM_abs", "runs/dyck_t2"),
        ("MapPoPE monotone", "MapPoPE_abs-1L_r2", "MapPoPE_abs", "runs/dyck_t2")]
LENS = [(32, 4), (64, 4), (96, 4), (128, 12)]


def main(n=64, dev="cuda:0"):
    w = DyckWorld()
    data = {c: w.batch(n, *c, np.random.default_rng(11))[0].to(dev) for c in LENS}
    print(f"{'arm':<18}{'alpha (mean +/- sd)':>24}{'range L32':>12}{'range L128':>12}{'growth':>9}")
    out = []
    for lab, name, arch, d in ARMS:
        al, r0, r1 = [], [], []
        for pt in sorted(glob.glob(f"/home/prashr/mapformer/{d}/{name}_s*/{name}.pt")):
            m = build(arch, 5, 1, 1, 2, 32).to(dev).eval()
            m.load_state_dict(torch.load(pt, map_location=dev, weights_only=False))
            rs = []
            for c in LENS:
                with torch.no_grad():
                    S = m.action_to_lie(m.token_emb(data[c])).mean(-1).cumsum(1)   # (B,T,H)
                rs.append(float((S.max(1).values - S.min(1).values).mean()))
            al.append(np.polyfit(np.log([c[0] for c in LENS]), np.log(np.maximum(rs, 1e-9)), 1)[0])
            r0.append(rs[0]); r1.append(rs[-1])
        al, r0, r1 = np.array(al), np.array(r0), np.array(r1)
        print(f"{lab:<18}{al.mean():>14.3f} +/- {al.std(ddof=1):<6.3f}{r0.mean():>12.2f}{r1.mean():>12.2f}"
              f"{r1.mean()/r0.mean():>8.2f}x   (n={len(al)}, per-seed alpha {al.min():.2f}-{al.max():.2f})")
        out.append((lab, al, r0, r1))
    return out


if __name__ == "__main__":
    main()
