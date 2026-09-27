import torch, json, sys
from mapformer.eval_noise_refine import evaluate
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
R="/home/prashr/mapformer"
J={"RANK_PROJ_TRAIN":json.load(open(f"{R}/RANK_PROJ_TRAIN.json")),"RANK_MATCHED_e900c":json.load(open(f"{R}/RANK_MATCHED_e900c.json")),"RANK_PROJ_FROZEN":json.load(open(f"{R}/RANK_PROJ_FROZEN.json"))}
cases=[("rank_proj_train","Vanilla",6,"RANK_PROJ_TRAIN"),("rank_matched_e900c","Vanilla_r4",3,"RANK_MATCHED_e900c"),("rank_matched_e900c","Vanilla",5,"RANK_MATCHED_e900c"),("rank_proj","Vanilla",4,"RANK_PROJ_FROZEN")]
dev=torch.device("cuda:0")
for d,v,s,j in cases:
    b=torch.load(f"{R}/runs/{d}/p0/{v}_s{s}/{v}.pt",map_location="cpu",weights_only=False); c=b["config"]
    m=VARIANT_MAP[v](vocab_size=c["vocab_size"],d_model=c["d_model"],n_heads=c["n_heads"],n_layers=c["n_layers"],grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); m=m.to(dev).eval()
    env=GridWorld(size=64,n_obs_types=16,p_empty=0.5,seed=10000)
    acc,nll=evaluate(m,env,1024,100,0.0,dev,seed=1234+s)
    ref=dict((x[0],x) for x in J[j][f"0.0|{v}|1024"])[s]
    print(d,v,s,"recomputed",round(acc,4),round(nll,4),"json",ref)
