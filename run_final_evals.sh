#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== MiniGrid full 8-cell factorial, n=8 ==="
python3 -u -m mapformer.eval_minigrid_2x2 --runs-dir mapformer/runs/minigrid_n8 \
  --variants Vanilla Hourglass_k2 MapPoPE-Flat MapPoPE-Hier RoPE PlainHourglass PoPE-Flat PoPE-Hier \
  --seeds 0 1 2 3 4 5 6 7 --lengths 512 1024 --device cuda:1 \
  --out mapformer/MINIGRID_FULL_2X2X2.md > mapformer/runs/minigrid_n8/eval8.log 2>&1
[ -f mapformer/MINIGRID_FULL_2X2X2.md ] || { echo "FAILED minigrid eval"; exit 1; }
echo "=== continuous displacement ==="
python3 -u -m mapformer.eval_knob_sweep --help >/dev/null 2>&1 || true
python3 -u - > mapformer/CONTINUOUS_ALLOC.md 2>mapformer/runs/continuous_alloc/eval.log <<'PY'
import numpy as np, torch, json
from collections import Counter
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
C={"commanded":dict(n_headings=12),
   "allocentric":dict(n_headings=12,action_record="allocentric"),
   "allocnoise":dict(n_headings=12,action_record="allocentric",heading_noise=0.15)}
@torch.no_grad()
def sc(m,env,nb,bs,T,dev,seed):
    m.eval(); np.random.seed(seed); torch.manual_seed(seed)
    ok=tot=0; mg=Counter()
    for _ in range(nb):
        tk,_o,rv,*_=env.generate_batch(bs,T); tk=tk.to(dev); mk=rv[:,1:].to(dev)
        if not mk.any(): continue
        pr=m(tk[:,:-1]).argmax(-1); tg=tk[:,1:]
        ok+=int((pr[mk]==tg[mk]).sum()); tot+=int(mk.sum()); mg.update(tg[mk].tolist())
    return ok/max(tot,1),(max(mg.values())/tot if tot else float('nan'))
dev=torch.device("cuda:1"); res={}; fl={}
for lbl,kw in C.items():
    for v in ("Vanilla","RoPE"):
        for s in range(3):
            ck=Path(f"mapformer/runs/continuous_alloc/{lbl}/{v}_s{s}/{v}.pt")
            if not ck.exists(): continue
            b=torch.load(ck,map_location="cpu",weights_only=False); c=b["config"]
            env=GridWorld(size=64,n_obs_types=16,p_empty=0.5,n_landmarks=0,seed=10000,
                          action_mode="rotate",score_moves_only=True,**kw)
            m=VARIANT_MAP[v](vocab_size=c["vocab_size"],d_model=c["d_model"],n_heads=c["n_heads"],
                             n_layers=c["n_layers"],grid_size=c["grid_size"]).to(dev)
            m.load_state_dict(b["model_state_dict"])
            a,f=sc(m,env,40,64,128,dev,5000+s)
            res.setdefault((lbl,v),[]).append(a); fl.setdefault(lbl,[]).append(f)
            del m; torch.cuda.empty_cache()
print("# Continuous displacement: does allocentric recoding survive Habitat's conditions?\n")
print("H=12 headings (Habitat turns 30 degrees), position real-valued, n=3, 980 batches.")
print("`allocnoise` adds 0.15 rad of Gaussian noise to each executed turn, so the")
print("recorded direction drifts off the true displacement -- Habitat actuation noise.\n")
print("| condition | floor | Vanilla | RoPE (index) | position effect |")
print("|---|---|---|---|---|")
out={}
for lbl in C:
    V=np.array(res.get((lbl,"Vanilla"),[])); R=np.array(res.get((lbl,"RoPE"),[]))
    if not len(V): continue
    f=float(np.mean(fl[lbl])); e=V.mean()-R.mean()
    out[lbl]={"floor":f,"vanilla":V.tolist(),"rope":R.tolist(),"effect":float(e)}
    print(f"| {lbl} | {f:.3f} | {V.mean():.3f} ± {V.std(ddof=1):.3f} | {R.mean():.3f} ± {R.std(ddof=1):.3f} | **{e:+.3f}** |")
print("\nReference, H=4 discrete (n=8): commanded **+0.050**, allocentric **+0.488**,")
print("translate baseline **+0.438**.")
json.dump(out,open("mapformer/CONTINUOUS_ALLOC.json","w"),indent=2)
PY
grep -q "^|" mapformer/CONTINUOUS_ALLOC.md || { echo "FAILED continuous eval"; exit 1; }
echo DONE; touch mapformer/runs/.final_evals_done
