"""Dyck-2 closer accuracy at L=512 with fine distance buckets (16x the training length).

Committed 2026-09-19 after an audit found the L=512 claim in the report had no artefact behind
it: the run existed only as a scratch script. Output: DYCK_FAR_PROBE.md.
Run from /home/prashr: python3 -m mapformer.probe_dyck_far
"""
import glob, sys; sys.path.insert(0,'/home/prashr')
import numpy as np, torch
from mapformer.environment_dyck import DyckWorld, CLOSE_P, CLOSE_B
from mapformer.probe_dyck_stack import top_distance
from mapformer.train_dyck import build
w=DyckWorld(); L,D=512,12
inp,tgt,valid,_=w.batch(256,L,D,np.random.default_rng(7))
dist=top_distance(inp,tgt,L); dp=valid[...,CLOSE_P]|valid[...,CLOSE_B]
corr=torch.where(valid[...,CLOSE_P],CLOSE_P,CLOSE_B); wrg=torch.where(valid[...,CLOSE_P],CLOSE_B,CLOSE_P)
B={"0-2":dist<=2,"3-8":(dist>2)&(dist<=8),"9-32":(dist>8)&(dist<=32),"33-64":(dist>32)&(dist<=64),
   "65-128":(dist>64)&(dist<=128),"129-256":(dist>128)&(dist<=256),"257+":dist>256}
out=[]
print("share: "+", ".join(f"{k} {float((m&dp).sum()/dp.sum()):.3f}" for k,m in B.items()))
print(f"{'model':<16}"+"".join(f"{k:>10}" for k in B))
for name,arch,nl,nh in [("MapPoPE-1L_r2","MapPoPE",1,1),("MapWM-1L_r2","MapWM",1,1),("MapEM-1L_r2","MapEM",1,1),("PoPE-1L","PoPE",1,1),("RoPE-1L","RoPE",1,1)]:
    A=[]
    for pt in sorted(glob.glob(f"/home/prashr/mapformer/runs/dyck_bs128/{name}_s*/{name}.pt")):
        m=build(arch,5,nl,nh,2,32).cuda().eval(); m.load_state_dict(torch.load(pt,map_location="cuda"))
        with torch.no_grad():
            P=torch.cat([m(inp[i:i+32].cuda()).float().softmax(-1).cpu() for i in range(0,256,32)])
        A.append(P.gather(-1,corr.unsqueeze(-1)).squeeze(-1)>P.gather(-1,wrg.unsqueeze(-1)).squeeze(-1))
    print(f"{name:<16}"+"".join(f"{np.mean([float(a[m2&dp].double().mean()) for a in A]):>10.3f}" for m2 in B.values()))
