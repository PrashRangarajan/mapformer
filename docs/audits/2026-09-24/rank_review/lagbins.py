import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from mapformer.train_variant import VARIANT_MAP
from mapformer.environment import GridWorld
from mapformer.eval_rank_strata import kinds
R="/home/prashr/mapformer/runs/rank_matched/p0"
BINS=[(1,2),(3,8),(9,32),(33,127),(128,10**9)]
env=GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
for v,s in (("Vanilla",6),("Vanilla",0),("Vanilla",4),("Vanilla_r4",4),("Vanilla_r4",3),("Vanilla_r4",0)):
    b=torch.load(f"{R}/{v}_s{s}/{v}.pt",map_location="cpu",weights_only=False); c=b["config"]
    m=VARIANT_MAP[v](vocab_size=c["vocab_size"],d_model=c["d_model"],n_heads=c["n_heads"],n_layers=c["n_layers"],grid_size=c["grid_size"]); m.load_state_dict(b["model_state_dict"]); m.eval()
    np.random.seed(1234+s); H={}
    with torch.no_grad():
        for _ in range(100):
            tok,_o,rev=env.generate_trajectory(1024)
            pred=m(tok[None,:-1]).argmax(-1)[0]; tgt=tok[1:]; msk=rev[1:]; ks=kinds(tok,env.ACTION_DELTAS,64)
            for i in torch.nonzero(msk).flatten().tolist():
                w,lag=ks[i//2]
                key="wrap" if w else next(f"{a}-{bb}" for a,bb in BINS if a<=lag<=bb)
                h=H.setdefault(key,[0,0]); h[0]+=int(pred[i]==tgt[i]); h[1]+=1
    print(f"{v:11s} s{s} final loss {b['losses'][-1]:.3f}: "+"  ".join(f"{k} {H[k][0]/H[k][1]:.3f} (n={H[k][1]})" for k in [f'{a}-{bb}' for a,bb in BINS]+['wrap'] if k in H), flush=True)
