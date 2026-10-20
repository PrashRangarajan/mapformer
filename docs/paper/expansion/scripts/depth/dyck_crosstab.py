"""POST HOC (CPU): Dyck-2 index-RoPE closing accuracy cross-tabulated by sibling blocks x distance (is the binding
variable the number of balanced blocks to skip, or the distance in tokens?). Same data and scoring as dyck_distance.py.
  cd /home/prashr && python3 mapformer/docs/paper/expansion/scripts/depth/dyck_crosstab.py -> out/dyck_crosstab_out.txt"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import REPO, OUTDIR  # noqa: E402
from dyck_distance import sibling_blocks  # noqa: E402
from mapformer.environment_dyck import DyckWorld  # noqa: E402
from mapformer.dyck_mdepth_common import cell_with_mask  # noqa: E402
from mapformer.train_dyck import build  # noqa: E402

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
BB = [(1, 2), (2, 3), (3, 5)]
DB = [(2, 8), (8, 16), (16, 32)]


def main():
    d = cell_with_mask(DyckWorld(), 32, 12); nb = sibling_blocks(d["tgt"], 32); msk = d["dp"] & d["close_ok"]
    L = ["rows: sibling blocks to skip; columns: distance in tokens; cells: mean over seeds 0-7 of pc/(pc+pw) [n prefixes]"]
    for arch, nl in (("RoPE", 1), ("RoPE", 2), ("RoPE", 3), ("RoPE", 4), ("MapWM", 1)):
        nm = f"{arch}-{nl}L" + ("_r2" if arch == "MapWM" else "") + "_tL32D12"
        tab = np.zeros((8, len(BB), len(DB)))
        for s in range(8):
            m = build(arch, 5, nl, 2, 2, 32)
            m.load_state_dict(torch.load(f"{REPO}/runs/dyck_mdepth/T12/{nm}_s{s}/{nm}.pt", map_location="cpu")); m.eval()
            with torch.no_grad():
                P = torch.cat([m(d["inp"][i:i + 128]).float() for i in range(0, 512, 128)]).softmax(-1)
            pc = P.gather(-1, d["corr"].unsqueeze(-1)).squeeze(-1); pw = P.gather(-1, d["wrng"].unsqueeze(-1)).squeeze(-1)
            ratio = pc / (pc + pw).clamp_min(1e-12)
            for i, (b0, b1) in enumerate(BB):
                for j, (d0, d1) in enumerate(DB):
                    sel = msk & (nb >= b0) & (nb < b1) & (d["dist"] >= d0) & (d["dist"] < d1)
                    tab[s, i, j] = float(ratio[sel].mean()) if sel.sum() >= 10 else np.nan
        mt = np.nanmean(tab, 0)
        L.append(f"{nm}:  dist " + "  ".join(f"{d0}-{d1 - 1:>2d}" for d0, d1 in DB))
        for i, (b0, b1) in enumerate(BB):
            ns = []
            for d0, d1 in DB:
                ns.append(int((msk & (nb >= b0) & (nb < b1) & (d["dist"] >= d0) & (d["dist"] < d1)).sum()))
            L.append(f"   blocks {b0}-{b1 - 1}: " + "  ".join(f"{x:.3f} [{n}]" for x, n in zip(mt[i], ns)))
        print("\n".join(L[-4:]), flush=True)
    open(f"{OUTDIR}/dyck_crosstab_out.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
