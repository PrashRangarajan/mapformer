import numpy as np, torch, math
from mapformer.environment_dyck import DyckWorld
from mapformer.eval_dyck_literature import cell_data, metrics, N
from mapformer.validate_dyck import sampler_probs
from mapformer.train_dyck import build
w = DyckWorld()
def feas_mask(L, D):
    # regenerate the cell exactly as cell_data does and read the sampler's per-position entropy:
    # 1.5 ln2 = open or close allowed, 0 = forced close, ln2 = forced open (or depth 0)
    _, _, _, ent = w.batch(N, L, D, np.random.default_rng(424242 + 1000 * L + D))
    return (ent - 1.5 * math.log(2)).abs().lt(1e-9) | ent.abs().lt(1e-12)
def a2(P, d, mask=None):
    dp = d["dp"] if mask is None else d["dp"] & mask
    pc = P.gather(-1, d["corr"].unsqueeze(-1)).squeeze(-1); pw = P.gather(-1, d["wrng"].unsqueeze(-1)).squeeze(-1)
    ratio = pc / (pc + pw).clamp_min(1e-12)
    js = sorted({int(x) for x in d["dist"][dp].unique()})
    return float(np.mean([float(ratio[dp & (d["dist"] == j)].mean()) for j in js]))
for c in [(32, 4), (32, 12), (128, 12)]:
    d = cell_data(w, *c); m = feas_mask(*c)
    assert torch.equal(d["inp"], w.batch(N, *c, np.random.default_rng(424242 + 1000 * c[0] + c[1]))[0])
    P = sampler_probs(d["tgt"].numpy(), *c)
    print(c, "dp positions", int(d["dp"].sum()), "forced-open among dp", int((d["dp"] & ~m).sum()),
          "sampler A2", round(a2(P, d), 4), "sampler A2f", round(a2(P, d, m), 4), "metrics() A2", round(metrics(P, d)["A2_close_acc_dist"], 4))
    if c == (32, 12):
        R = "/home/prashr/mapformer/runs/dyck_ladder"
        for arch, nm in [("RoPE", "RoPE-4L"), ("PoPE", "PoPE-4L"), ("MapWM", "MapWM-4L_r2"), ("MapPoPE", "MapPoPE-4L_r2")]:
            v, vf = [], []
            for s in range(8):
                mdl = build(arch, 5, 4, 2, 2, 32).cuda().eval(); mdl.load_state_dict(torch.load(f"{R}/{nm}_s{s}/{nm}.pt", map_location="cuda"))
                with torch.no_grad():
                    lg = torch.cat([mdl(d["inp"][i:i+128].cuda()).float().cpu() for i in range(0, N, 128)])
                v.append(a2(lg.softmax(-1), d)); vf.append(a2(lg.softmax(-1), d, m))
            print("  ladder D4-trained", nm, "A2 %.3f  A2f %.3f" % (np.mean(v), np.mean(vf)), "per-seed A2f", np.round(vf, 3))
