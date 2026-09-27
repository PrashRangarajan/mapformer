import torch, numpy as np
R="/home/prashr/mapformer/runs"
def load(d,v,s):
    b=torch.load(f"{R}/{d}/p0/{v}_s{s}/{v}.pt",map_location="cpu",weights_only=False)
    return np.array(b["losses"]),b["config"]
print("== rank_matched (T=1024, 300 ep): loss at epochs 10,25,50,100,150,200,250,300; first epoch below 1.0/0.5/0.1")
for v in ("Vanilla","Vanilla_r4"):
    for s in range(8):
        l,c=load("rank_matched",v,s)
        pts=[l[i-1] for i in (10,25,50,100,150,200,250,300)]
        fb=[int(np.argmax(l<th))+1 if (l<th).any() else None for th in (1.0,0.5,0.1)]
        # scaled-window flat: last 10% vs previous 10%
        n=len(l); w=n//10
        r10=l[-w:].mean()/l[-2*w:-w].mean()
        print(f"{v:11s} s{s} "+" ".join(f"{x:.3f}" for x in pts)+f"  below1/.5/.1 {fb}  last30/prev30 {l[-30:].mean()/l[-60:-30].mean():.3f}  last10%/prev10% {r10:.3f} min {l.min():.4f}@{l.argmin()+1}")
print("config keys:",sorted(c.keys()))
print("\n== rank_sweep (T=128, 300 ep)")
for v in ("Vanilla","Vanilla_r4"):
    for s in range(8):
        try: l,c=load("rank_sweep",v,s)
        except Exception as e: print(v,s,e); continue
        pts=[l[i-1] for i in (10,25,50,100,150,200,250,300)]
        print(f"{v:11s} s{s} "+" ".join(f"{x:.4f}" for x in pts)+f"  last30/prev30 {l[-30:].mean()/max(l[-60:-30].mean(),1e-12):.3f}")
