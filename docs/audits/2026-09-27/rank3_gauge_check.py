import torch, numpy as np
from mapformer.train_variant import VARIANT_MAP
from mapformer.environment import GridWorld
env=GridWorld(size=64,seed=0)
R="/home/prashr/mapformer/runs/rank3/p0"
for s in range(8):
    b=torch.load(f"{R}/Vanilla_r3ph_s{s}/Vanilla_r3ph.pt",map_location="cpu",weights_only=False)
    V=b["model_state_dict"]["token_emb.weight"].shape[0]
    m=VARIANT_MAP["Vanilla_r3ph"](vocab_size=V,d_model=128,n_heads=2,n_layers=1,grid_size=64)
    m.load_state_dict(b["model_state_dict"]); m.eval()
    with torch.no_grad(): D=m.action_to_lie(m.token_emb.weight[None])[0]
    om=m.path_integrator.omega.detach()
    D=(D).reshape(V,-1).numpy()
    ao=env.action_offset if hasattr(env,'action_offset') else 0
    A=D[ao:ao+4]; oo=env.obs_offset; O=D[oo:oo+env.n_obs_types+1]
    c=A.mean(0); o=O.mean(0)
    n=np.linalg.norm
    oppraw=np.mean([n(A[0]+A[1])/((n(A[0])+n(A[1]))/2), n(A[2]+A[3])/((n(A[2])+n(A[3]))/2)])
    R_=A-c
    oppc=np.mean([n(R_[0]+R_[1])/((n(R_[0])+n(R_[1]))/2), n(R_[2]+R_[3])/((n(R_[2])+n(R_[3]))/2)])
    net=c+o
    print(f"s{s}: opp raw {oppraw:.3f} minus-common {oppc:.3f} |c|/|A0| {n(c)/n(A[0]):.3f} cos(c, mean obs) {c@o/(n(c)*n(o)):+.3f} |c+obs|/|c| {n(net)/n(c):.3f}")
