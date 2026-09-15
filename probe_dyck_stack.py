"""Metrics for Dyck-2 that the always-valid opening brackets cannot inflate.

The paper's F1 gives most of its range to tokens that need no stack: '(' and '[' are valid at
every position, and 66% of the positions that DO admit a closer have the matching open bracket
within 2 tokens. A 1-token lookup table therefore scores 0.857 while being at chance (0.50) for
picking the legal closer beyond distance 2.

Reported here, eval-only, on trained checkpoints:
  closer accuracy   P(legal closer) > P(illegal closer) at positions with depth > 0. Chance 0.5.
                    Stratified by the distance back to the open bracket on top of the stack --
                    the axis that grows when sequences get longer.
  strict-F1         the paper's F1 with BT's mean over Val(s) replaced by a min: EVERY valid
                    token, the legal closer included, must outrank the total invalid mass.
  paper F1          unchanged, for reference, plus its split by position type.

Run from /home/prashr:  python3 -m mapformer.probe_dyck_stack [--runs-dir DIR] [--out MD]
"""
import argparse, glob
import numpy as np
import torch

from mapformer.environment_dyck import DyckWorld, f1_valid, CLOSE_P, CLOSE_B, OPEN_P, OPEN_B
from mapformer.validate_dyck import ngram_fit, ngram_probs
from mapformer.train_dyck import build

ARMS = [("MapPoPE-1L_r2", "MapPoPE", 1, 1), ("MapWM-1L_r2", "MapWM", 1, 1),
        ("MapEM-1L_r2", "MapEM", 1, 1), ("PoPE-1L", "PoPE", 1, 1),
        ("RoPE-1L", "RoPE", 1, 1), ("RoPE-2L", "RoPE", 2, 2)]
BUCKETS = ["1-2", "3-8", "9-32", "33+"]


def strict_f1(P, valid):
    P = P.double(); pv = (P * valid).sum(-1); inv = (P * ~valid).sum(-1)
    bt = (((P > inv.unsqueeze(-1)) | ~valid).all(-1)).double()
    den = pv + bt
    return torch.where(den > 0, 2 * pv * bt / den.clamp_min(1e-300), torch.zeros_like(den))


def top_distance(inp, tgt, L):
    """Distance from each prefix to the open bracket currently on top of the stack (0 if empty)."""
    dist = torch.zeros(inp.shape, dtype=torch.long)
    for i in range(inp.shape[0]):
        st = []
        for t in range(L):
            dist[i, t] = (t - st[-1]) if st else 0
            x = tgt[i, t].item()
            if x in (OPEN_P, OPEN_B): st.append(t + 1)
            else: st.pop()
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/home/prashr/mapformer/runs/dyck_bs128")
    ap.add_argument("--out", default="/home/prashr/mapformer/DYCK_STACK_PROBE.md")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--cells", nargs="+", default=["32:4", "128:12"])
    a = ap.parse_args()
    world = DyckWorld()
    out = ["# Dyck-2: metrics the always-valid opens cannot inflate",
           "", "Eval-only on the checkpoints of `" + a.runs_dir + "`; " + str(a.n) +
           " sequences per cell, mean over seeds. Closer accuracy: P(legal closer) > P(illegal "
           "closer) at positions with depth > 0, chance 0.500. strict-F1: the paper's F1 with a "
           "min over Val(s) in place of the mean. Distance = how far back the open bracket on top "
           "of the stack is.", ""]
    for cell in a.cells:
        L, D = (int(x) for x in cell.split(":"))
        inp, tgt, valid, _ = world.batch(a.n, L, D, np.random.default_rng(424242 + 1000 * L + D))
        dist = top_distance(inp, tgt, L)
        depth_pos = valid[..., CLOSE_P] | valid[..., CLOSE_B]
        prev_open = (inp == OPEN_P) | (inp == OPEN_B)
        bmask = {"1-2": dist <= 2, "3-8": (dist > 2) & (dist <= 8),
                 "9-32": (dist > 8) & (dist <= 32), "33+": dist > 32}
        corr = torch.where(valid[..., CLOSE_P], CLOSE_P, CLOSE_B)
        wrng = torch.where(valid[..., CLOSE_P], CLOSE_B, CLOSE_P)
        share = {k: float((m & depth_pos).sum() / depth_pos.sum()) for k, m in bmask.items()}
        out += [f"## L = {L}, depth = {D}", "",
                "Share of depth>0 positions by distance: " +
                ", ".join(f"{k} {share[k]:.2f}" for k in BUCKETS), "",
                "| predictor | paper F1 | strict-F1 | closer acc | " +
                " | ".join("d " + k for k in BUCKETS) + " | acc d>=9 | after a close |",
                "|---|" + "---|" * (7 + len(BUCKETS) - 4)]
        rows = [("n-gram k=1", None), ("n-gram k=3", None)] + [(n, s) for n, *s in
                [(n, arch, nl, nh) for n, arch, nl, nh in ARMS]]
        for name, spec in rows:
            if spec is None:
                k = int(name[-1])
                Ps = [ngram_probs(ngram_fit(world, k, np.random.default_rng(100 + k)), k, inp)]
            else:
                arch, nl, nh = spec; Ps = []
                for pt in sorted(glob.glob(f"{a.runs_dir}/{name}_s*/{name}.pt")):
                    m = build(arch, 5, nl, nh, 2, 32).cuda().eval()
                    m.load_state_dict(torch.load(pt, map_location="cuda"))
                    with torch.no_grad():
                        Ps.append(torch.cat([m(inp[i:i + 128].cuda()).float().softmax(-1).cpu()
                                             for i in range(0, a.n, 128)]))
                if not Ps:
                    continue
            acc = [P.gather(-1, corr.unsqueeze(-1)).squeeze(-1) >
                   P.gather(-1, wrng.unsqueeze(-1)).squeeze(-1) for P in Ps]
            def mean(m):
                if not bool(m.any()): return float('nan')
                return np.mean([float(x[m].double().mean()) for x in acc])
            deep = depth_pos & (dist > 8)
            cells = [f"{np.mean([float(f1_valid(P, valid)[0].mean()) for P in Ps]):.3f}",
                     f"{np.mean([float(strict_f1(P, valid).mean()) for P in Ps]):.3f}",
                     f"{mean(depth_pos):.3f}"] + \
                    [("--" if np.isnan(mean(m & depth_pos)) else f"{mean(m & depth_pos):.3f}") for m in bmask.values()] + \
                    [f"{mean(deep):.3f}", f"{mean(depth_pos & ~prev_open):.3f}"]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
            print(out[-1], flush=True)
        out.append("")
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
