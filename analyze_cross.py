"""Regenerates every number in CROSS_RESULTS.md (the metric crossing and its controls).

Committed 2026-09-20 after a handoff audit found the project's one surviving positive claim had no
analysis script -- the exact defect its own LATEST block warns about. Reads the trained checkpoints
and prints the cells, the paired contrasts with MDEs, the per-distance breakdown, the learned decay
rates and the metric correlations.

Run from /home/prashr: python3 -m mapformer.analyze_cross
"""
import glob, json, math, re

import numpy as np, torch
import torch.nn.functional as F

from mapformer.environment_dyck import DyckWorld, CLOSE_P, CLOSE_B, OPEN_P, OPEN_B
from mapformer.probe_dyck_stack import top_distance
from mapformer.train_dyck import build

R = "/home/prashr/mapformer/runs"
ARMS = [("index / token", f"{R}/dyck_decay/PoPE_decay-1L_s*/PoPE_decay-1L.pt", "PoPE_decay", None),
        ("index / token, lam .084", f"{R}/dyck_lam/PoPE_decay-1L_lam0.084_s*/PoPE_decay-1L_lam0.084.pt", "PoPE_decay", 0.084),
        ("index / token, lam .034", f"{R}/dyck_lam/PoPE_decay-1L_lam0.034_s*/PoPE_decay-1L_lam0.034.pt", "PoPE_decay", 0.034),
        ("index / frozen state", f"{R}/dyck_cross/PoPE_decay_frozenmetric-1L_r2_s*/PoPE_decay_frozenmetric-1L.pt", "PoPE_decay_frozenmetric", None),
        ("index / learned state", f"{R}/dyck_cross/PoPE_decay_statemetric-1L_r2_s*/PoPE_decay_statemetric-1L.pt", "PoPE_decay_statemetric", None),
        ("path / token", f"{R}/dyck_cross/MapPoPE_decay_idxmetric-1L_r2_s*/MapPoPE_decay_idxmetric-1L_r2.pt", "MapPoPE_decay_idxmetric", None),
        ("path / state", f"{R}/dyck_decay/MapPoPE_decay-1L_r2_s*/MapPoPE_decay-1L_r2.pt", "MapPoPE_decay", None)]
BUCK = [("0-2", 0, 2), ("3-8", 3, 8), ("9-32", 9, 32), ("33+", 33, 10 ** 6)]


def main(n=512, dev="cuda:0"):
    w = DyckWorld(); L, D = 128, 12
    inp, tgt, valid, _ = w.batch(n, L, D, np.random.default_rng(424242 + 1000 * L + D))
    dist = top_distance(inp, tgt, L)
    dp = valid[..., CLOSE_P] | valid[..., CLOSE_B]
    corr = torch.where(valid[..., CLOSE_P], CLOSE_P, CLOSE_B)
    wrg = torch.where(valid[..., CLOSE_P], CLOSE_B, CLOSE_P)
    far = dp & (dist > 8)
    dep = np.zeros(inp.shape, dtype=int)
    for i in range(inp.shape[0]):
        d = 0
        for t in range(L):
            dep[i, t] = d; d += 1 if tgt[i, t].item() in (OPEN_P, OPEN_B) else -1
    A = {}
    print(f"{'arm':<26}{'acc d>=9':>10}{'F1':>8}{'loss':>8}{'lambda':>8}   " +
          "".join(f"{'d '+b[0]:>9}" for b in BUCK))
    for lab, pat, arch, lam in ARMS:
        acc, f1, loss, lams, per = {}, {}, {}, [], {b[0]: {} for b in BUCK}
        for pt in sorted(glob.glob(pat)):
            s = int(re.search(r"_s(\d+)/", pt).group(1))
            m = build(arch, 5, 1, 1, 2, 32, lam).to(dev).eval()
            m.load_state_dict(torch.load(pt, map_location=dev, weights_only=False))
            with torch.no_grad():
                P = torch.cat([m(inp[i:i + 128].to(dev)).float().softmax(-1).cpu() for i in range(0, n, 128)])
            ok = P.gather(-1, corr.unsqueeze(-1)).squeeze(-1) > P.gather(-1, wrg.unsqueeze(-1)).squeeze(-1)
            acc[s] = float(ok[far].double().mean())
            for nm, lo, hi in BUCK:
                sel = dp & (dist >= lo) & (dist <= hi)
                per[nm][s] = float(ok[sel].double().mean())
            j = json.load(open(pt.replace(".pt", ".json")))
            f1[s] = j["grid"]["L128_D12"]["F1"]; loss[s] = j["final_loss"]
            lams.append(float(F.softplus(m.layers[0].lam_raw)[0]))
        A[lab] = (acc, f1, loss, per)
        print(f"{lab:<26}{np.mean(list(acc.values())):>10.3f}{np.mean(list(f1.values())):>8.3f}"
              f"{np.mean(list(loss.values())):>8.4f}{np.mean(lams):>8.3f}   " +
              "".join(f"{np.mean(list(per[b[0]].values())):>9.3f}" for b in BUCK))

    def con(x, y, key=0):
        ss = sorted(set(A[x][key]) & set(A[y][key]))
        d = np.array([A[x][key][s] - A[y][key][s] for s in ss])
        mde = 2.8 * d.std(ddof=1) / math.sqrt(len(d))
        return f"{d.mean():+.3f} (MDE {mde:.3f}, {int((d>0).sum())}/{len(d)})" + (" DET" if abs(d.mean()) > mde else "")
    print("\nregistered contrasts, closer accuracy at d>=9:")
    for x, y, lab in [("index / learned state", "index / token", "metric swap, index row"),
                      ("path / token", "path / state", "metric swap, path row (sign: token-state)"),
                      ("index / token, lam .034", "index / token", "weaken envelope only"),
                      ("index / learned state", "index / token, lam .034", "metric at matched strength"),
                      ("index / frozen state", "index / token, lam .084", "frozen metric vs matched strength")]:
        print(f"  {lab:<42}{con(x, y)}")
    fl = [A[k][2][s] for k in A for s in A[k][2]]; ac = [A[k][0][s] for k in A for s in A[k][0]]
    print(f"\nrule 9 over all {len(fl)} runs: r(final train loss, acc d>=9) = {np.corrcoef(fl, ac)[0,1]:+.3f}")
    for lab, pat, arch in [("index / learned state", f"{R}/dyck_cross/PoPE_decay_statemetric-1L_r2_s*/PoPE_decay_statemetric-1L.pt", "PoPE_decay_statemetric")]:
        ct, cd = [], []
        for pt in sorted(glob.glob(pat)):
            m = build(arch, 5, 1, 1, 2, 32).to(dev).eval()
            m.load_state_dict(torch.load(pt, map_location=dev, weights_only=False))
            with torch.no_grad():
                S = m.metric_map(m.token_emb(inp.to(dev))).mean(-1).cumsum(1).transpose(1, 2)[:, 0].cpu().numpy()
            a, b = [], []
            for i in range(min(16, n)):
                t = np.arange(L); k = np.tril_indices(L, -1)
                Dm = np.abs(S[i][:, None] - S[i][None, :])
                a.append(np.corrcoef(Dm[k], np.abs(t[:, None] - t[None, :])[k])[0, 1])
                b.append(np.corrcoef(Dm[k], np.abs(dep[i][:, None] - dep[i][None, :])[k])[0, 1])
            ct.append(np.mean(a)); cd.append(np.mean(b))
        print(f"{lab} metric map: r(token distance) = {np.mean(ct):.3f} +/- {np.std(ct, ddof=1):.3f}, "
              f"r(depth difference) = {np.mean(cd):.3f} +/- {np.std(cd, ddof=1):.3f}")


if __name__ == "__main__":
    main()
