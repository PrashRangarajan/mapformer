import torch, numpy as np, json
R="/home/prashr/mapformer/runs"
out={}
for tag in ("rank_matched_e900","rank_matched_e900c"):
    for v in ("Vanilla","Vanilla_r4"):
        for s in range(8):
            b=torch.load(f"{R}/{tag}/p0/{v}_s{s}/{v}.pt",map_location="cpu",weights_only=False)
            out[f"{tag}|{v}|{s}"]=[float(x) for x in b["losses"]]
            if s==0 and v=="Vanilla": print(tag, {k:(v if not hasattr(v,'shape') else v.shape) for k,v in b["config"].items()}, list(b.keys()))
json.dump(out,open("/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/review_rank2/curves.json","w"))
def tail(l,f=0.05):
    k=max(1,round(f*len(l))); return np.mean(l[-k:])
for v in ("Vanilla","Vanilla_r4"):
    for s in range(8):
        a=np.array(out[f"rank_matched_e900|{v}|{s}"]); c=np.array(out[f"rank_matched_e900c|{v}|{s}"])
        # min over continuation, argmin; peak in first 100 epochs
        print(f"{v:11s} s{s} c1 tail {tail(a):.4f} last {a[-1]:.4f} min {a.min():.4f} | c2 first5 {c[:5].round(3)} peak {c[:200].max():.3f}@{c[:200].argmax()+1} min {c.min():.4f}@{c.argmin()+1} tail {tail(c):.4f} last {c[-1]:.4f}  c2 dec-means {[round(float(c[i*90:(i+1)*90].mean()),3) for i in range(10)]}")
