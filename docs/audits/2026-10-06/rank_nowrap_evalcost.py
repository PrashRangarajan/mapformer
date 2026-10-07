"""Amendment 4: cost of the 400-walk strata on the CPU parts (walk + per-step info generation, per-target Python loop),
measured; the GPU forward is estimated, not measured (no GPU use allowed). 40 walks per measurement, scaled to 400."""
import sys, time
import numpy as np, torch
sys.path.insert(0, "/home/prashr")
torch.set_num_threads(4)
from mapformer import analyze_rank_nowrap as A
from mapformer.train_variant import VARIANT_MAP
b = torch.load("/home/prashr/mapformer/runs/rank_nd/D2/Vanilla_r2ph_s0/Vanilla_r2ph.pt", map_location="cpu", weights_only=False); c = b["config"]
m = VARIANT_MAP["Vanilla_r2ph"](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"], grid_size=c["grid_size"])
m.load_state_dict(b["model_state_dict"]); m.eval()
for N in (32, 256):
    t = time.time(); env, st = A.stream(N, 60, n=40); tg = time.time() - t
    t = time.time()
    with torch.no_grad():
        for w in st:
            m(w[0][None, :-1])
    tf = time.time() - t
    t = time.time(); A.strat(m, st, "cpu"); ts = time.time() - t
    print(f"grid {N}: per 400 walks -- stream+info {tg * 10:.1f} s, CPU forward {tf * 10:.1f} s, strata loop excl. forward "
          f"{(ts - tf) * 10:.1f} s")
