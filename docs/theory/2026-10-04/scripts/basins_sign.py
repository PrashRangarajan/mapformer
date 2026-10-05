"""Basin classification (basins.py rules) on the depth/loop aids at shared rank 2 (runs/loop_rank) and the 1800-epoch
runs (runs/loop_rank_e1800). For multi-layer models the channel weights use the first block's Q/K on token embeddings
(approximate beyond layer 1); the unweighted classification is printed beside it."""
import numpy as np, torch
exec(open("/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/basins.py").read().split("sets = [")[0])
for d in ("runs/sign_matched/p0",):
    import os
    for arm in [a for a in sorted(set(x.rsplit("_s",1)[0] for x in os.listdir(R/d))) if a!="RoPE"]:
        row=[]; 
        for s in range(8):
            cp = R/d/f"{arm}_s{s}"/f"{arm}.pt"
            if not cp.exists(): row.append("NA"); continue
            try:
                o, fl = heads(cp, True); o2,_ = heads(cp, False)
            except Exception as ex:
                row.append("ERR"); print(arm, ex); continue
            row.append(f"{basin(o)[:3]}/{basin(o2)[:3]}{'+' if fl<0.05 else '-'}")
        print(f"{d.split('/')[1]:18s} {arm:14s} " + " ".join(row))
