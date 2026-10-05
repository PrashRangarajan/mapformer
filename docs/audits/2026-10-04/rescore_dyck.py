"""Re-score DYCK_MDEPTH (T12 condition: trained and tested at L32 D12, layers 1-4) under rescore_hook. Registered
metric A2f (closing accuracy over distances, feasible) at cell L32 D12, via analyze_dyck_mdepth.load_eval on the
analysis's exact data. Scale none (must reproduce the registered position main effects +0.353/+0.130/+0.045/+0.024)
and auto (attention x 1/(1-p) in EVERY layer -- see the caveat: the per-layer correction is not valid for deep stacks)."""
import json
import numpy as np, torch
import torch.nn as nn
import mapformer.rescore_hook as RH
from mapformer.analyze_dyck_mdepth import load_eval, run_name, KEY, RUNS
from mapformer.environment_dyck import DyckWorld
from mapformer.dyck_mdepth_common import cell_with_mask
from mapformer.stats_core import signflip_p

ORIG = nn.Module.eval
w = DyckWorld(); C = (32, 12); data = {C: cell_with_mask(w, *C)}
import mapformer.analyze_dyck_mdepth as A
A.CELLS = [C]                                   # load_eval loops over CELLS; score the primary cell only
res = {}
for L in (1, 2, 3, 4):
    for arm in ("RoPE", "PoPE", "MapWM", "MapPoPE"):
        nm = run_name(arm, L, "_tL32D12")
        for sc in (None, "auto"):
            nn.Module.eval = ORIG
            if sc:
                RH.install("auto")
            res[(L, arm, sc)] = [load_eval(f"{RUNS}/T12/{nm}_s{s}/{nm}.pt", arm, L, data)[f"L{C[0]}D{C[1]}"][KEY] for s in range(8)]
            nn.Module.eval = ORIG
out = {}
for L in (1, 2, 3, 4):
    for sc in (None, "auto"):
        path = (np.array(res[(L, "MapWM", sc)]) + np.array(res[(L, "MapPoPE", sc)])) / 2
        idx = (np.array(res[(L, "RoPE", sc)]) + np.array(res[(L, "PoPE", sc)])) / 2
        d = path - idx; out[f"L{L}|{sc}"] = {a: res[(L, a, sc)] for a in ("RoPE", "PoPE", "MapWM", "MapPoPE")}
        print(f"{L} layers, scale {sc or 'none':4s}: RoPE {np.mean(res[(L, 'RoPE', sc)]):.3f} PoPE {np.mean(res[(L, 'PoPE', sc)]):.3f} "
              f"MapWM {np.mean(res[(L, 'MapWM', sc)]):.3f} MapPoPE {np.mean(res[(L, 'MapPoPE', sc)]):.3f}  position main "
              f"{d.mean():+.3f} ({(d > 0).sum()}/8 +, sign-flip p {signflip_p(d)['p']:.4f})", flush=True)
json.dump(out, open("/home/prashr/mapformer/runs_rescore/dyck_mdepth.json", "w"), indent=1)
