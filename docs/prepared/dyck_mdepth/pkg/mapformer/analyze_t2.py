"""Regenerates T2_RESULTS.md (monotone Dyck: T2, T2b, T2c). Committed after an audit found these
tables had no script. Run from /home/prashr: python3 -m mapformer.analyze_t2"""
import glob, json, math, re
import numpy as np, torch

from mapformer.environment_dyck import DyckWorld, CLOSE_P, CLOSE_B
from mapformer.probe_dyck_stack import top_distance
from mapformer.train_dyck import build

R = "/home/prashr/mapformer/runs"
ARMS = [("MapWM signed", f"{R}/dyck_bs128/MapWM-1L_r2_s*/MapWM-1L_r2.pt", "MapWM"),
        ("MapPoPE signed", f"{R}/dyck_bs128/MapPoPE-1L_r2_s*/MapPoPE-1L_r2.pt", "MapPoPE"),
        ("MapWM monotone", f"{R}/dyck_t2/MapWM_abs-1L_r2_s*/MapWM_abs-1L_r2.pt", "MapWM_abs"),
        ("MapPoPE monotone", f"{R}/dyck_t2/MapPoPE_abs-1L_r2_s*/MapPoPE_abs-1L_r2.pt", "MapPoPE_abs"),
        ("+ phase, init 0.1", f"{R}/dyck_t2/MapPoPE_abs_T3-1L_r2_s*/MapPoPE_abs_T3-1L_r2.pt", "MapPoPE_abs_T3"),
        ("+ phase, zero init", f"{R}/dyck_t2/MapPoPE_abs_T3zero-1L_r2_s*/MapPoPE_abs_T3zero-1L_r2.pt", "MapPoPE_abs_T3zero"),
        ("inert twin", f"{R}/dyck_t2/MapPoPE_abs_T3inert-1L_r2_s*/MapPoPE_abs_T3inert-1L_r2.pt", "MapPoPE_abs_T3inert")]
CELLS = ["L32_D4", "L128_D4", "L32_D12", "L128_D12"]


def main(n=512, dev="cuda:0"):
    w = DyckWorld(); L, D = 128, 12
    inp, tgt, valid, _ = w.batch(n, L, D, np.random.default_rng(424242 + 1000 * L + D))
    dist = top_distance(inp, tgt, L); dp = valid[..., CLOSE_P] | valid[..., CLOSE_B]
    corr = torch.where(valid[..., CLOSE_P], CLOSE_P, CLOSE_B)
    wrg = torch.where(valid[..., CLOSE_P], CLOSE_B, CLOSE_P)
    far = dp & (dist > 8)
    A = {}
    print(f"{'arm':<20}" + "".join(f"{c:>11}" for c in CELLS) + f"{'closer':>9}{'d>=9':>8}")
    for lab, pat, arch in ARMS:
        f1 = {c: {} for c in CELLS}; ca, cf = {}, {}
        for pt in sorted(glob.glob(pat)):
            s = int(re.search(r"_s(\d+)/", pt).group(1))
            j = json.load(open(pt.replace(".pt", ".json")))
            for c in CELLS: f1[c][s] = j["grid"][c]["F1"]
            m = build(arch, 5, 1, 1, 2, 32).to(dev).eval()
            m.load_state_dict(torch.load(pt, map_location=dev, weights_only=False))
            with torch.no_grad():
                P = torch.cat([m(inp[i:i + 128].to(dev)).float().softmax(-1).cpu() for i in range(0, n, 128)])
            ok = P.gather(-1, corr.unsqueeze(-1)).squeeze(-1) > P.gather(-1, wrg.unsqueeze(-1)).squeeze(-1)
            ca[s] = float(ok[dp].double().mean()); cf[s] = float(ok[far].double().mean())
        A[lab] = (f1, ca, cf)
        print(f"{lab:<20}" + "".join(f"{np.mean(list(f1[c].values())):>11.3f}" for c in CELLS) +
              f"{np.mean(list(ca.values())):>9.3f}{np.mean(list(cf.values())):>8.3f}")

    def d(x, y, key, cell="L128_D12"):
        get = (lambda s: A[x][0][cell][s] - A[y][0][cell][s]) if key == "f1" else \
              (lambda s: A[x][2][s] - A[y][2][s])
        ss = sorted(set(A[x][1]) & set(A[y][1])); v = np.array([get(s) for s in ss])
        mde = 2.8 * v.std(ddof=1) / math.sqrt(len(v))
        return v, f"{v.mean():+.4f} (MDE {mde:.4f}, {int((v<0).sum())}/{len(v)} neg)" + (" DET" if abs(v.mean()) > mde else "")
    for key in ("f1", "acc"):
        s1, _ = d("MapPoPE monotone", "MapWM monotone", key); s2, _ = d("MapPoPE signed", "MapWM signed", key)
        dd = s1 - s2; mde = 2.8 * dd.std(ddof=1) / math.sqrt(len(dd))
        print(f"\n[{key}] T2 difference of differences {dd.mean():+.4f} (MDE {mde:.4f}, "
              f"{int((dd<0).sum())}/{len(dd)} neg)" + (" DET" if abs(dd.mean()) > mde else ""))
        for x, y, lab in [("+ phase, init 0.1", "inert twin", "T2b phase(0.1) - inert"),
                          ("+ phase, zero init", "inert twin", "T2c phase(zero) - inert"),
                          ("inert twin", "MapPoPE monotone", "inert - plain")]:
            print(f"  {lab:<26}{d(x, y, key)[1]}")


if __name__ == "__main__":
    main()
