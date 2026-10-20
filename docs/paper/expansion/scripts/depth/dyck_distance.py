"""POST HOC (CPU, eval-only): Dyck-2 closing accuracy by DISTANCE to the matching open bracket, by the number of
complete balanced blocks between that open bracket and the prefix ('blocks'), and by STACK DEPTH, for
the matched-depth batch (runs/dyck_mdepth/T12: trained at L32 D12, 1x budget), index RoPE / PoPE at 1-4 layers and the
path arms MapWM / MapPoPE at 1 layer, seeds 0-7. Scored on the registered L32 D12 cell (eval_dyck_literature.cell_data,
512 sequences) on A2f's prefixes (closer feasible under the sampler); per prefix the renormalised legal-closer
probability pc / (pc + pw), as in A2f. A2f is recomputed per run (a2_masked); its arm means reproduce DYCK_MDEPTH_RESULTS.md (checked by hand: RoPE
0.648 / 0.889 / 0.964 / 0.982, PoPE 0.640 / 0.850 / 0.946 / 0.971, MapWM 0.996 / 1.000).

  cd /home/prashr && python3 mapformer/docs/paper/expansion/scripts/depth/dyck_distance.py
writes out/dyck_distance.json and out/dyck_distance_out.txt"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import REPO, OUTDIR  # noqa: E402
from mapformer.environment_dyck import DyckWorld, OPEN_P, OPEN_B  # noqa: E402
from mapformer.dyck_mdepth_common import cell_with_mask, a2_masked  # noqa: E402
from mapformer.train_dyck import build  # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
DBINS = [(2, 4), (4, 8), (8, 16), (16, 32)]


def stack_depth(tgt, L):
    """Depth of the stack at each prefix (number of unmatched opens after token t), same convention as top_distance."""
    d = torch.zeros(tgt.shape, dtype=torch.long)
    for i in range(tgt.shape[0]):
        k = 0
        for t in range(L):
            d[i, t] = k
            k += 1 if tgt[i, t].item() in (OPEN_P, OPEN_B) else -1
    return d


def sibling_blocks(tgt, L):
    """Number of complete balanced blocks between the open bracket on top of the stack and the current prefix
    (0 = the top open is the last token)."""
    nb = torch.zeros(tgt.shape, dtype=torch.long)
    for i in range(tgt.shape[0]):
        st = []                                   # per stack level: count of closed child blocks
        for t in range(L):
            nb[i, t] = st[-1] if st else 0
            if tgt[i, t].item() in (OPEN_P, OPEN_B):
                st.append(0)
            else:
                st.pop()
                if st:
                    st[-1] += 1
    return nb


def main():
    w = DyckWorld(); d = cell_with_mask(w, 32, 12)
    sd = stack_depth(d["tgt"], 32); nb = sibling_blocks(d["tgt"], 32)
    msk = d["dp"] & d["close_ok"]
    out = {}; lines = []
    for arch in ("RoPE", "PoPE", "MapWM", "MapPoPE"):
        for L in ((1, 2, 3, 4) if arch in ("RoPE", "PoPE") else (1, 2)):
            nm = f"{arch}-{L}L" + ("_r2" if arch.startswith("Map") else "") + "_tL32D12"
            per = []
            for s in range(8):
                base = f"{REPO}/runs/dyck_mdepth/T12/{nm}_s{s}/{nm}"
                m = build(arch, 5, L, 2, 2, 32); m.load_state_dict(torch.load(base + ".pt", map_location="cpu")); m.eval()
                with torch.no_grad():
                    P = torch.cat([m(d["inp"][i:i + 128]).float() for i in range(0, 512, 128)]).softmax(-1)
                pc = P.gather(-1, d["corr"].unsqueeze(-1)).squeeze(-1); pw = P.gather(-1, d["wrng"].unsqueeze(-1)).squeeze(-1)
                ratio = pc / (pc + pw).clamp_min(1e-12)
                a2f = a2_masked(P, d, d["close_ok"])
                r = {"A2f": a2f}
                for lo, hi in DBINS:
                    sel = msk & (d["dist"] >= lo) & (d["dist"] < hi)
                    r[f"dist{lo}-{hi - 1}"] = float(ratio[sel].mean()); r[f"n_dist{lo}-{hi - 1}"] = int(sel.sum())
                for k in range(1, 13):
                    sel = msk & (sd == k)
                    if sel.any():
                        r[f"depth{k}"] = float(ratio[sel].mean())
                for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 16)]:
                    sel = msk & (nb >= lo) & (nb < hi)
                    r[f"blocks{lo}-{hi - 1}"] = float(ratio[sel].mean()); r[f"n_blocks{lo}-{hi - 1}"] = int(sel.sum())
                # distance-1 prefixes are 'the previous token is the open bracket': a local rule
                per.append(r)
            out[nm] = per
            keys = list(per[0])
            mean = {k: float(np.mean([p[k] for p in per])) for k in keys if not k.startswith("n_")}
            lines.append(f"{nm:22s} " + "  ".join(f"{k} {v:.3f}" for k, v in mean.items()))
            print(lines[-1], flush=True)
    ns = {k: v for k, v in out["RoPE-1L_tL32D12"][0].items() if k.startswith("n_")}
    lines.append("n prefixes per distance bin: " + str(ns))
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(f"{OUTDIR}/dyck_distance.json", "w"), indent=1)
    open(f"{OUTDIR}/dyck_distance_out.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
