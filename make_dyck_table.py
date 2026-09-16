"""Build the shareable Dyck-2 table on the two unfused metrics.

invalid mass   probability placed on brackets that are ungrammatical at that position
               (= 1 - P_Val of the paper's F1). Lower is better; a uniform guesser is ~0.25.
closer acc     P(legal closer) > P(illegal closer) at positions where something is open.
               Chance 0.500. This is the only part of the task that needs the stack.
Run from /home/prashr: python3 -m mapformer.make_dyck_table
"""
import glob, json
import numpy as np, torch

from mapformer.environment_dyck import DyckWorld, f1_valid, CLOSE_P, CLOSE_B
from mapformer.validate_dyck import ngram_fit, ngram_probs
from mapformer.probe_dyck_stack import top_distance
from mapformer.train_dyck import build

RUNS = "/home/prashr/mapformer/runs/dyck_bs128"
CELLS = [(32, 4), (128, 4), (128, 12)]
ARMS = [("MapPoPE", "MapPoPE-1L_r2", "MapPoPE", 1, 1), ("MapFormer (MapWM)", "MapWM-1L_r2", "MapWM", 1, 1),
        ("PoPE", "PoPE-1L", "PoPE", 1, 1), ("RoPE", "RoPE-1L", "RoPE", 1, 1)]
N = 512


def probs(name, arch, nl, nh, inp):
    out = []
    for pt in sorted(glob.glob(f"{RUNS}/{name}_s*/{name}.pt")):
        m = build(arch, 5, nl, nh, 2, 32).cuda().eval()
        m.load_state_dict(torch.load(pt, map_location="cuda"))
        with torch.no_grad():
            out.append(torch.cat([m(inp[i:i + 128].cuda()).float().softmax(-1).cpu()
                                  for i in range(0, N, 128)]))
    return out


def main():
    w = DyckWorld()
    R = {}
    for L, D in CELLS:
        inp, tgt, valid, _ = w.batch(N, L, D, np.random.default_rng(424242 + 1000 * L + D))
        dist = top_distance(inp, tgt, L)
        dp = valid[..., CLOSE_P] | valid[..., CLOSE_B]
        corr = torch.where(valid[..., CLOSE_P], CLOSE_P, CLOSE_B)
        wrg = torch.where(valid[..., CLOSE_P], CLOSE_B, CLOSE_P)
        rows = [("n-gram baseline (no stack)", [ngram_probs(ngram_fit(w, 1, np.random.default_rng(101)), 1, inp)])]
        rows += [(lab, probs(n, a, nl, nh, inp)) for lab, n, a, nl, nh in ARMS]
        for lab, Ps in rows:
            inv = np.array([float((P * ~valid).sum(-1).mean()) for P in Ps])
            ok = [(P.gather(-1, corr.unsqueeze(-1)).squeeze(-1) >
                   P.gather(-1, wrg.unsqueeze(-1)).squeeze(-1)) for P in Ps]
            acc = np.array([float(o[dp].double().mean()) for o in ok])
            far = dp & (dist > 8)
            accf = np.array([float(o[far].double().mean()) for o in ok])
            f1 = np.array([float(f1_valid(P, valid)[0].mean()) for P in Ps])
            R[(lab, L, D)] = dict(inv=inv.mean(), inv_sd=inv.std(ddof=1) if len(inv) > 1 else 0.0,
                                  acc=acc.mean(), acc_sd=acc.std(ddof=1) if len(acc) > 1 else 0.0,
                                  accfar=accf.mean(), f1=f1.mean(), n=len(Ps))
            print(lab, L, D, R[(lab, L, D)], flush=True)
    json.dump({f"{k[0]}|L{k[1]}D{k[2]}": v for k, v in R.items()},
              open("/home/prashr/mapformer/DYCK_TWO_METRICS.json", "w"), indent=1)
    return R


if __name__ == "__main__":
    main()
