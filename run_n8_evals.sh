#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -u -m mapformer.eval_minigrid_2x2 --runs-dir mapformer/runs/minigrid_n8 \
  --variants Vanilla Hourglass_k2 MapPoPE-Flat MapPoPE-Hier RoPE PlainHourglass PoPE-Flat \
  --seeds 0 1 2 3 4 5 6 7 --lengths 128 512 1024 --device cuda:1 \
  --out mapformer/MINIGRID_2X2X2_n8.md > mapformer/runs/minigrid_n8/eval.log 2>&1
touch mapformer/runs/minigrid_n8/.eval_done
python3 -u - <<'PY' > mapformer/KNOB_SWEEP_n8.md 2>mapformer/runs/knob_n8/eval.log
import numpy as np, torch, json
from collections import Counter
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
CONDS={"baseline":dict(size=64,n_obs_types=16),
       "allcombined":dict(size=16,n_obs_types=64,action_mode="rotate",obs_mode="ego",boundary="wall"),
       "rotate":dict(size=64,n_obs_types=16,action_mode="rotate",score_moves_only=True),
       "allocentric":dict(size=64,n_obs_types=16,action_mode="rotate",score_moves_only=True,action_record="allocentric")}
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
for lbl,kw in CONDS.items():
    for v in ("Vanilla","RoPE"):
        for s in range(8):
            ck=Path(f"mapformer/runs/knob_n8/{lbl}/{v}_s{s}/{v}.pt")
            if not ck.exists(): continue
            b=torch.load(ck,map_location="cpu",weights_only=False); c=b["config"]
            env=GridWorld(p_empty=0.5,n_landmarks=0,seed=10000,**kw)
            m=VARIANT_MAP[v](vocab_size=c["vocab_size"],d_model=c["d_model"],n_heads=c["n_heads"],
                             n_layers=c["n_layers"],grid_size=c["grid_size"]).to(dev)
            m.load_state_dict(b["model_state_dict"])
            a,f=sc(m,env,16,64,128,dev,5000+s)
            res.setdefault((lbl,v),[]).append(a); fl.setdefault(lbl,[]).append(f)
            del m; torch.cuda.empty_cache()
print("# Knob sweep at n=8\n")
print("| condition | floor | Vanilla | RoPE | position effect | n |")
print("|---|---|---|---|---|---|")
out={}
for lbl in CONDS:
    V=np.array(res.get((lbl,"Vanilla"),[])); R=np.array(res.get((lbl,"RoPE"),[]))
    if not len(V): continue
    f=float(np.mean(fl[lbl])); e=V.mean()-R.mean()
    out[lbl]={"floor":f,"vanilla":V.tolist(),"rope":R.tolist(),"effect":float(e)}
    print(f"| {lbl} | {f:.3f} | {V.mean():.3f} ± {V.std(ddof=1):.3f} | {R.mean():.3f} ± {R.std(ddof=1):.3f} | **{e:+.3f}** | {len(V)} |")
json.dump(out,open("mapformer/KNOB_SWEEP_n8.json","w"),indent=2)
PY
touch mapformer/runs/knob_n8/.eval_done
echo ALL_EVALS_DONE
