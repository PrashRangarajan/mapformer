import torch, numpy as np
R="/home/prashr/mapformer/runs"
for s in range(8):
    t=torch.load(f"{R}/rank_proj_train/p0/Vanilla_s{s}/Vanilla.pt",map_location="cpu",weights_only=False)
    p=torch.load(f"{R}/rank_proj/p0/Vanilla_s{s}/Vanilla.pt",map_location="cpu",weights_only=False)
    c=torch.load(f"{R}/rank_matched_e900c/p0/Vanilla_r4_s{s}/Vanilla_r4.pt",map_location="cpu",weights_only=False)
    ct=t["config"]; cc=c["config"]
    diff={k:(ct.get(k),cc.get(k)) for k in set(ct)|set(cc) if ct.get(k)!=cc.get(k)}
    l=np.array(t["losses"]); lc=np.array(c["losses"])
    lp=t.get("losses_prior"); 
    print(f"s{s} init_from={ct.get('init_from')} seed={t['seed']} dso={ct.get('data_seed_offset')} cfgdiff={diff}")
    print(f"   train first5 {l[:5].round(4)}  control first5 {lc[:5].round(4)}  prior-len {None if lp is None else len(lp)} prior-last {None if lp is None else round(float(lp[-1]),4)}")
    print(f"   proj cfg keys: top2 {p['config'].get('proj_top2_energy'):.5f} relerr {p['config'].get('proj_delta_rel_err'):.4f} src {p['config'].get('projected_from')}")
